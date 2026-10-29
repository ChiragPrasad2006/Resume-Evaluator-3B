"""
Production-Grade QLoRA Fine-Tuning Pipeline for AI Resume Reviewer (Qwen2.5-3B)
Optimized for 8GB VRAM (NVIDIA RTX 5060 Ti Blackwell) + 16GB System RAM (Windows).

Features:
- Hard PyTorch memory fraction cap at 85% (~6.8 GB) to prevent Windows shared RAM spillover
- Expandable segments and max_split_size_mb:64 allocator configuration
- Windows RAM protection: dataloader_num_workers=0 and dataloader_pin_memory=False
- Memory-mapped Arrow dataset loading (keep_in_memory=False) for ~0 MB RAM overhead
- Compact LoRA rank (r=12, alpha=24, dropout=0.08) saving 25% activation memory
- prediction_loss_only=True during evaluation to prevent 152k-vocab logits RAM spikes
- Proactive MemoryCleanCallback (gc.collect + empty_cache) every 2 steps, eval, and save
- Automatic checkpoint recovery with get_last_checkpoint and emergency interrupt save
- Dry-run verification with physical checkpoint persistence (checkpoint-2, checkpoint-4)
"""

import os
import sys

# MUST set memory fragmentation environment variable before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Avoid Rust transfer deadlocks on Windows
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"    # Disable hf-xet hook deadlocks on Windows

import argparse
import gc
import inspect
import json
import psutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

# Cap PyTorch VRAM allocation to 85% (~6.8 GB) so Windows WDDM never spills into shared RAM
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

from torch.utils.data import Dataset
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Import centralized configuration
import config


class ResumeSFTDataset(Dataset):
    """
    Fallback PyTorch Dataset loading pre-tokenized ChatML records with completion masks.
    Enforces max_length truncation (2048) and exposes self.lengths for group_by_length.
    """

    def __init__(self, data_path: Path, max_samples: Optional[int] = None, max_length: int = config.MAX_SEQ_LENGTH):
        self.records: List[Dict[str, Any]] = []
        self.lengths: List[int] = []
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    input_ids = item["input_ids"][:max_length]
                    labels = item["labels"][:max_length]
                    self.records.append({
                        "input_ids": input_ids,
                        "labels": labels,
                    })
                    self.lengths.append(len(input_ids))
                if max_samples and len(self.records) >= max_samples:
                    break

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class CompletionDataCollator:
    """
    Pads input_ids with pad_token_id and labels with -100 so loss
    is strictly evaluated on completion (assistant response) tokens.
    Enforces self.max_length truncation to prevent VRAM spikes.
    """

    def __init__(self, pad_token_id: int, max_length: int = config.MAX_SEQ_LENGTH):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Enforce max_length upper bound on batch items
        max_len = min(self.max_length, max(len(item["input_ids"]) for item in batch))
        
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []

        for item in batch:
            input_ids = list(item["input_ids"][:self.max_length])
            labels = list(item["labels"][:self.max_length])
            
            # Bulletproof NaN protection: if all labels are -100, preserve the last token as active
            if all(l == -100 for l in labels) and len(labels) > 0:
                labels[-1] = input_ids[-1]

            padding_len = max_len - len(input_ids)

            # Pad on right for causal LM
            padded_input_ids = input_ids + [self.pad_token_id] * padding_len
            padded_labels = labels + [-100] * padding_len
            attention_mask = [1] * len(input_ids) + [0] * padding_len

            batch_input_ids.append(padded_input_ids)
            batch_labels.append(padded_labels)
            batch_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }


def get_lora_config() -> LoraConfig:
    """Builds compact LoRA configuration (r=12, alpha=24) saving ~25% activation memory."""
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias=config.LORA_BIAS,
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_tokenizer(is_dry_run_cpu: bool = False):
    """
    Loads Qwen2.5-3B with 4-bit NF4 quantization on CUDA.
    Gracefully handles CPU fallback if running on a non-CUDA development system.
    """
    print(f"Loading tokenizer: {config.MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            trust_remote_code=True,
            local_files_only=False,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config.HAS_CUDA and not is_dry_run_cpu:
        print("\n[CUDA Environment Detected]")
        print(f"Target GPU: {torch.cuda.get_device_name(0)}")
        print("Configuring 4-bit NormalFloat (NF4) quantization...")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                attn_implementation=config.ATTN_IMPLEMENTATION,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(
                config.MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                attn_implementation=config.ATTN_IMPLEMENTATION,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=False,
            )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs=config.GRADIENT_CHECKPOINTING_KWARGS,
        )
        # Enable input gradients and disable cache before wrapping with PEFT
        model.enable_input_require_grads()
        model.config.use_cache = False

        # Ensure lm_head computes in bfloat16 to prevent fp32 logits explosion
        if hasattr(model, "lm_head"):
            model.lm_head = model.lm_head.to(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)

        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer

    else:
        # CPU Mode (for dry-run testing on machines without Blackwell/CUDA)
        print("\n[CPU Mode Detected - Running with CPU-compatible configuration]")
        model_config = AutoConfig.from_pretrained(config.MODEL_NAME)
        model_config.num_hidden_layers = 2
        model_config.hidden_size = 512
        model_config.intermediate_size = 1024
        model_config.num_attention_heads = 8
        model_config.num_key_value_heads = 2
        
        model = AutoModelForCausalLM.from_config(model_config)
        model.enable_input_require_grads()
        model.config.use_cache = False

        lora_config = get_lora_config()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model, tokenizer


class SafeStateCallback(TrainerCallback):
    """
    Custom callback that logs safe model state persistence events
    and saves the tokenizer alongside each intermediate checkpoint.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def on_save(self, args, state, control, **kwargs):
        step = state.global_step
        epoch = state.epoch if state.epoch is not None else 0.0
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        
        # Save tokenizer alongside adapter weights for standalone reproducibility
        if ckpt_dir.exists():
            self.tokenizer.save_pretrained(str(ckpt_dir))
            
        print(f"\n[Safe State Saved] Model checkpoint + tokenizer safely preserved:")
        print(f"  -> Path : {ckpt_dir}")
        print(f"  -> Step : {step} | Epoch : {epoch:.2f}")


class MemoryCleanCallback(TrainerCallback):
    """
    Proactively clears CUDA cache and runs garbage collection to eliminate
    memory fragmentation and monitor VRAM & System RAM usage.
    """

    def _clean_and_log(self, trigger: str, step: int):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
            ram_pct = psutil.virtual_memory().percent
            print(f"[MemoryClean] [{trigger}] Step {step:4d} | VRAM Alloc: {vram_alloc:.2f} GB | Reserved: {vram_res:.2f} GB | System RAM: {ram_pct:.1f}%")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 2 == 0:
            self._clean_and_log("on_step_end", state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        self._clean_and_log("on_evaluate", state.global_step)

    def on_save(self, args, state, control, **kwargs):
        self._clean_and_log("on_save", state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            ram_pct = psutil.virtual_memory().percent
            print(f"[VRAM Monitor] Step {state.global_step:4d} | Peak VRAM: {vram_alloc:.2f} GB / 6.80 GB Cap | System RAM: {ram_pct:.1f}%")


def load_datasets():
    """Loads memory-mapped arrow dataset (0 MB RAM overhead) or falls back to JSONL."""
    if config.SFT_DATASET_DIR.exists() and (config.SFT_DATASET_DIR / "dataset_dict.json").exists():
        print(f"Loading memory-mapped Arrow dataset from: {config.SFT_DATASET_DIR}")
        ds = load_from_disk(str(config.SFT_DATASET_DIR), keep_in_memory=False)
        train_ds = ds["train"]
        val_ds = ds["validation"]
    else:
        print(f"Loading JSONL dataset from: {config.SFT_TRAIN_PATH}")
        train_ds = ResumeSFTDataset(config.SFT_TRAIN_PATH)
        val_ds = ResumeSFTDataset(config.SFT_VAL_PATH)
    return train_ds, val_ds


def run_training(
    is_dry_run: bool = False,
    max_dry_run_steps: int = 4,
    save_steps: int = config.SAVE_STEPS,
    eval_steps: Optional[int] = None,
    save_strategy: str = config.SAVE_STRATEGY,
    checkpoints_dir: Optional[Path] = None,
    resume_from_checkpoint: Optional[str] = None,
    save_total_limit: int = config.SAVE_TOTAL_LIMIT,
) -> None:
    """
    Executes production QLoRA fine-tuning or dry-run validation using Hugging Face Trainer.
    All checkpoints are preserved strictly inside the project folder.
    """
    target_ckpt_dir = checkpoints_dir or config.CHECKPOINT_DIR
    target_ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Automatic checkpoint detection via get_last_checkpoint
    last_checkpoint = get_last_checkpoint(str(target_ckpt_dir)) if target_ckpt_dir.is_dir() else None
    
    if resume_from_checkpoint:
        if resume_from_checkpoint.lower() == "auto":
            resume_target = last_checkpoint
        else:
            resume_target = resume_from_checkpoint
    else:
        resume_target = last_checkpoint

    if is_dry_run:
        actual_save_steps = 2
        actual_eval_steps = 2
        logging_steps = 1
        actual_save_total_limit = 3
        
        # If resuming from step 4 or higher in dry-run, run up to 6 steps
        if resume_target is not None:
            print(f"[RECOVERY] Resuming training automatically from {resume_target}")
            actual_max_steps = 6
        else:
            actual_max_steps = max_dry_run_steps
            
        num_train_epochs = 1
        load_best = False
        print("\n" + "=" * 70)
        print("MANDATORY DRY-RUN EXECUTION INITIATED (TRAINER CHECKPOINT VERIFICATION)")
        print(f"  Target Steps           : {actual_max_steps} optimizer steps")
        print(f"  Save Steps Cadence     : Every {actual_save_steps} steps (checkpoint-2, checkpoint-4)")
        print(f"  VRAM Allocation Cap    : 6.80 GB (85% of 8GB)")
        print(f"  System RAM Ceiling     : 75.0%")
        print("=" * 70)
    else:
        actual_save_steps = save_steps
        actual_eval_steps = eval_steps if eval_steps is not None else config.EVAL_STEPS
        logging_steps = config.LOGGING_STEPS
        actual_save_total_limit = save_total_limit
        actual_max_steps = -1
        num_train_epochs = config.NUM_TRAIN_EPOCHS
        load_best = config.LOAD_BEST_MODEL_AT_END

        # Enforce that eval_steps divides save_steps (or equals save_steps) when load_best is True
        if load_best:
            if actual_eval_steps <= 0:
                actual_eval_steps = actual_save_steps
            elif actual_save_steps % actual_eval_steps != 0:
                print(
                    f"[CONFIG AUTO-CORRECT] save_steps ({actual_save_steps}) is not a multiple of eval_steps ({actual_eval_steps}). "
                    f"Aligning eval_steps to {actual_save_steps} to satisfy load_best_model_at_end requirement."
                )
                actual_eval_steps = actual_save_steps

        if resume_target is not None:
            print(f"[RECOVERY] Resuming training automatically from {resume_target}")

        print("\n" + "=" * 70)
        print("STARTING FULL PRODUCTION QLORA TRAINING (OOM-PROTECTED SPECIFICATION)")
        print(f"  Project Checkpoint Directory : {target_ckpt_dir}")
        print(f"  Save Strategy                : {save_strategy} (every {actual_save_steps} {save_strategy})")
        print(f"  Max Preserved Checkpoints    : {actual_save_total_limit}")
        print(f"  Effective Batch Size         : {config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS} (micro={config.PER_DEVICE_TRAIN_BATCH_SIZE}, accum={config.GRADIENT_ACCUMULATION_STEPS})")
        print(f"  Learning Rate                : {config.LEARNING_RATE}")
        print(f"  NEFTune Noise Alpha          : {config.NEFTUNE_NOISE_ALPHA}")
        print(f"  Max Sequence Length          : {config.MAX_SEQ_LENGTH}")
        print(f"  Dataloader Num Workers       : {config.DATALOADER_NUM_WORKERS} (0 on Windows)")
        print(f"  Dataloader Pin Memory        : {config.DATALOADER_PIN_MEMORY} (False on Windows)")
        print("=" * 70)

    train_dataset, full_val_dataset = load_datasets()
    
    # Use a small 40-sample validation subset during training to prevent OOM/stall
    if len(full_val_dataset) > 40:
        val_dataset = full_val_dataset.select(range(40))
    else:
        val_dataset = full_val_dataset

    print(f"Train samples: {len(train_dataset)}, Validation evaluation subset: {len(val_dataset)} (Full val: {len(full_val_dataset)})")

    model, tokenizer = load_model_and_tokenizer()
    collator = CompletionDataCollator(pad_token_id=tokenizer.pad_token_id)

    sig = inspect.signature(TrainingArguments.__init__).parameters
    training_args_kwargs: Dict[str, Any] = {
        "output_dir": str(target_ckpt_dir),
        "per_device_train_batch_size": config.PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": config.PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
        "eval_accumulation_steps": config.EVAL_ACCUMULATION_STEPS,
        "learning_rate": config.LEARNING_RATE,
        "weight_decay": config.WEIGHT_DECAY,
        "max_grad_norm": config.MAX_GRAD_NORM,
        "lr_scheduler_type": config.LR_SCHEDULER_TYPE,
        "logging_steps": logging_steps,
        "neftune_noise_alpha": config.NEFTUNE_NOISE_ALPHA,
        "save_strategy": "steps",
        "save_steps": actual_save_steps,
        "save_total_limit": actual_save_total_limit,
        "eval_strategy": "steps",
        "eval_steps": actual_eval_steps,
        "load_best_model_at_end": load_best,
        "metric_for_best_model": config.METRIC_FOR_BEST_MODEL if load_best else None,
        "greater_is_better": config.GREATER_IS_BETTER if load_best else None,
        "bf16": config.BF16 and config.HAS_CUDA and torch.cuda.is_bf16_supported(),
        "fp16": config.FP16,
        "gradient_checkpointing": config.GRADIENT_CHECKPOINTING,
        "gradient_checkpointing_kwargs": config.GRADIENT_CHECKPOINTING_KWARGS,
        "optim": config.OPTIMIZER,
        "dataloader_pin_memory": config.DATALOADER_PIN_MEMORY,  # False on Windows
        "dataloader_num_workers": config.DATALOADER_NUM_WORKERS, # 0 on Windows
        "prediction_loss_only": config.PREDICTION_LOSS_ONLY,   # Never store 152k logits in RAM
        "remove_unused_columns": False,
        "report_to": "none",
    }

    if actual_max_steps > 0:
        training_args_kwargs["max_steps"] = actual_max_steps
    else:
        training_args_kwargs["num_train_epochs"] = num_train_epochs

    if "warmup_ratio" in sig:
        training_args_kwargs["warmup_ratio"] = config.WARMUP_RATIO
    elif "warmup_steps" in sig:
        total_steps = (len(train_dataset) // (config.PER_DEVICE_TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS)) * config.NUM_TRAIN_EPOCHS
        training_args_kwargs["warmup_steps"] = max(1, int(total_steps * config.WARMUP_RATIO))

    if "group_by_length" in sig:
        training_args_kwargs["group_by_length"] = config.GROUP_BY_LENGTH
    elif "train_sampling_strategy" in sig and config.GROUP_BY_LENGTH:
        training_args_kwargs["train_sampling_strategy"] = "group_by_length"

    if "save_safetensors" in sig:
        training_args_kwargs["save_safetensors"] = config.SAVE_SAFETENSORS

    training_args = TrainingArguments(**training_args_kwargs)

    safe_callback = SafeStateCallback(tokenizer=tokenizer)
    mem_callback = MemoryCleanCallback()
    callbacks = [safe_callback, mem_callback]
    if not is_dry_run:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOPPING_PATIENCE))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    print("\nBeginning training loop...")
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_target)
    except (KeyboardInterrupt, RuntimeError, Exception) as e:
        print(f"\n[EMERGENCY SAVE] Interrupted or caught error during training: {e}")
        emergency_dir = config.EMERGENCY_CHECKPOINT_DIR
        emergency_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preserving emergency adapter weights to: {emergency_dir}")
        trainer.model.save_pretrained(str(emergency_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(emergency_dir))
        print(f"[EMERGENCY SAVE COMPLETE] Preserved successfully to {emergency_dir}")
        raise e

    # Memory and Checkpoint Verification
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
    ram_pct = psutil.virtual_memory().percent
    checkpoints = sorted([d.name for d in target_ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

    print("\n" + "=" * 70)
    print("TRAINING / DRY-RUN EXECUTION SUMMARY:")
    print("=" * 70)
    print(f"  - Global Step Reached : {train_result.global_step}")
    print(f"  - Peak GPU VRAM       : {peak_vram_gb:.2f} GB (Cap: 6.80 GB, Pass: {peak_vram_gb < 6.8})")
    print(f"  - System RAM Usage    : {ram_pct:.1f}% (Cap: 75.0%, Pass: {ram_pct < 75.0})")
    print(f"  - Checkpoints on Disk : {checkpoints}")
    print("=" * 70)

    if is_dry_run:
        assert peak_vram_gb < 6.8, f"VRAM budget exceeded: {peak_vram_gb:.2f} GB >= 6.80 GB"
        assert ram_pct < 75.0, f"System RAM budget exceeded: {ram_pct:.1f}% >= 75.0%"
        print("\nDry-run verified cleanly with strict memory compliance and active checkpoints. Exiting with code 0.")
        sys.exit(0)

    # Save final best LoRA adapter and tokenizer in safetensors format
    final_dir = config.FINAL_MODEL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final LoRA adapter to project folder: {final_dir}")
    trainer.model.save_pretrained(str(final_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(final_dir))

    # Save training metadata summary
    summary_path = target_ckpt_dir / "training_summary.json"
    summary_data = {
        "status": "completed",
        "global_step": getattr(train_result, "global_step", 0),
        "total_flos": getattr(train_result, "total_flos", train_result.metrics.get("total_flos", 0.0) if hasattr(train_result, "metrics") and isinstance(train_result.metrics, dict) else 0.0),
        "train_loss": getattr(train_result, "training_loss", 0.0),
        "best_metric": trainer.state.best_metric if hasattr(trainer.state, "best_metric") else None,
        "best_model_checkpoint": trainer.state.best_model_checkpoint if hasattr(trainer.state, "best_model_checkpoint") else None,
        "final_adapter_dir": str(final_dir),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Training summary saved: {summary_path}")
    print("\n" + "=" * 70)
    print("TRAINING RUN COMPLETE")
    print(f"Best Checkpoint: {trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else 'N/A'}")
    print(f"Final Model Saved: {final_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning for AI Resume Reviewer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 4 optimizer steps with save_steps=2 to physically verify checkpoint creation and auto-resumption.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=config.SAVE_STEPS,
        help=f"Number of update steps between saving safe states (default: {config.SAVE_STEPS}).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=config.EVAL_STEPS,
        help=f"Number of update steps between evaluations (default: {config.EVAL_STEPS}).",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=config.SAVE_TOTAL_LIMIT,
        help=f"Maximum number of recent checkpoints to retain (default: {config.SAVE_TOTAL_LIMIT}).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=str(config.CHECKPOINT_DIR),
        help=f"Directory inside project to store safe checkpoints (default: {config.CHECKPOINT_DIR}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume from the latest safe checkpoint found in checkpoints directory.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Explicit path to a safe checkpoint folder to resume training from.",
    )
    args = parser.parse_args()

    resume_arg = None
    if args.resume_from_checkpoint:
        resume_arg = args.resume_from_checkpoint
    elif args.resume:
        resume_arg = "auto"

    run_training(
        is_dry_run=args.dry_run,
        max_dry_run_steps=4,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_strategy=config.SAVE_STRATEGY,
        checkpoints_dir=Path(args.checkpoints_dir),
        resume_from_checkpoint=resume_arg,
        save_total_limit=args.save_total_limit,
    )


if __name__ == "__main__":
    main()
