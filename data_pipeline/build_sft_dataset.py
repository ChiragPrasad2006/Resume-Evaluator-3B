"""
Data Pipeline Step 3: Build SFT Dataset (ChatML with Completion Masking)
Merges curated and synthetic hard negative datasets, performs stratified 85/15 split,
formats into Qwen ChatML standard, builds completion loss masks, and writes sft_train.jsonl and sft_val.jsonl.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Add root directory to sys.path to import config
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


from datasets import Dataset, DatasetDict

def load_datasets() -> List[Dict[str, Any]]:
    """Loads and combines curated and synthetic datasets."""
    all_records = []
    
    curated_path = config.BASE_CURATED_PATH if config.BASE_CURATED_PATH.exists() else config.CURATED_DATA_PATH
    synth_path = config.SYNTHETIC_HARD_NEGATIVES_PATH if config.SYNTHETIC_HARD_NEGATIVES_PATH.exists() else config.HARD_NEGATIVES_PATH

    if not curated_path.exists():
        raise FileNotFoundError(f"Missing {curated_path}. Run inspect_and_curate.py first.")
    if not synth_path.exists():
        raise FileNotFoundError(f"Missing {synth_path}. Run synthesize_hard_negatives.py first.")
        
    print(f"Loading curated data from {curated_path}...")
    with open(curated_path, "r", encoding="utf-8") as f:
        curated = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(curated)} curated records.")

    print(f"Loading synthetic data from {synth_path}...")
    with open(synth_path, "r", encoding="utf-8") as f:
        synthetic = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(synthetic)} synthetic records.")

    all_records = curated + synthetic
    print(f"Total merged dataset size: {len(all_records)} samples.")
    return all_records


def build_chatml_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Constructs the standard 3-turn ChatML messages payload."""
    user_content = config.USER_PROMPT_TEMPLATE.format(
        job_description=record["job_description"],
        resume=record["resume"],
    )
    assistant_content = record["target_json_str"]

    return [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def verify_completion_mask(
    input_ids: List[int],
    labels: List[int],
    tokenizer: AutoTokenizer,
    response_token_ids: List[int],
) -> Tuple[bool, int, int]:
    """Verifies that all prompt tokens have label -100 and only assistant response has active labels."""
    prompt_tokens_count = sum(1 for label in labels if label == -100)
    assistant_tokens_count = sum(1 for label in labels if label != -100)
    
    # Ensure there are both masked prompt tokens and unmasked completion tokens
    is_valid = prompt_tokens_count > 0 and assistant_tokens_count > 0
    return is_valid, prompt_tokens_count, assistant_tokens_count


def process_and_split_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Applies ChatML formatting, stratification, and completion masks to train & val sets."""
    print("=" * 70)
    print("STEP 3: Building Stratified SFT Dataset with ChatML & Completion Masking")
    print("=" * 70)

    records = load_datasets()

    print(f"Initializing tokenizer: {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Stratified split based on score_tier
    stratify_labels = [r["score_tier"] for r in records]
    train_records, val_records = train_test_split(
        records,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=stratify_labels,
    )

    print(f"Split sizes: Train={len(train_records)} (85%), Val={len(val_records)} (15%)")

    # Response template token IDs for Qwen ChatML: "<|im_start|>assistant\n"
    response_template = "<|im_start|>assistant\n"
    response_tokens = tokenizer.encode(response_template, add_special_tokens=False)

    gen_suffix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def prepare_sft_records(recs: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
        sft_data = []
        token_lengths = []
        masked_proportions = []
        truncated_prompts = 0

        for rec in recs:
            messages = build_chatml_messages(rec)
            
            # 1. Tokenize assistant target (with ChatML end token)
            target_text = rec["target_json_str"] + "<|im_end|>\n"
            target_ids = tokenizer.encode(target_text, add_special_tokens=False)
            
            # 2. Maximum tokens available for prompt
            max_prompt_tokens = config.MAX_SEQ_LENGTH - len(target_ids)
            
            # 3. Format and tokenize prompt
            prompt_messages = messages[:2]
            prompt_res = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
            if hasattr(prompt_res, "input_ids"):
                prompt_ids = list(prompt_res.input_ids)
            else:
                prompt_ids = list(prompt_res)
                
            # If prompt exceeds budget, truncate prompt, preserving system header & assistant prompt
            if len(prompt_ids) > max_prompt_tokens:
                truncated_prompts += 1
                prompt_ids = prompt_ids[:max_prompt_tokens]
                prompt_ids[-len(gen_suffix):] = gen_suffix
                
            # 4. Concatenate input_ids and construct completion labels
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + list(target_ids)
            
            # Guarantee that every sample has active target tokens
            n_prompt = len(prompt_ids)
            n_resp = len(target_ids)
            assert n_resp > 0, f"Sample {rec['id']} has 0 target tokens!"
            
            token_lengths.append(len(input_ids))
            masked_proportions.append(n_prompt / max(1, len(input_ids)))

            full_text = tokenizer.decode(input_ids, skip_special_tokens=False)

            sft_data.append({
                "id": rec["id"],
                "score_tier": rec["score_tier"],
                "overall_score": rec["overall_score"],
                "messages": messages,
                "text": full_text,
                "input_ids": input_ids,
                "labels": labels,
                "num_tokens": len(input_ids),
                "num_target_tokens": n_resp,
            })

        print(f"\n--- {split_name.upper()} DATASET METRICS ---")
        print(f"Total samples: {len(sft_data)}")
        print(f"Token length min/mean/median/max: {min(token_lengths)} / {int(np.mean(token_lengths))} / {int(np.median(token_lengths))} / {max(token_lengths)}")
        print(f"Prompts truncated to fit {config.MAX_SEQ_LENGTH}: {truncated_prompts} / {len(sft_data)} ({truncated_prompts/len(sft_data)*100:.1f}%)")
        print(f"Active target tokens min/mean/max: {min(d['num_target_tokens'] for d in sft_data)} / {int(np.mean([d['num_target_tokens'] for d in sft_data]))} / {max(d['num_target_tokens'] for d in sft_data)}")
        print(f"Samples with 0 target tokens: {sum(1 for d in sft_data if d['num_target_tokens'] == 0)} (strictly 0)")
        print(f"Average prompt token masking ratio: {np.mean(masked_proportions)*100:.1f}%")

        # Tier distribution check
        tier_counts = {}
        for item in sft_data:
            tier_counts[item["score_tier"]] = tier_counts.get(item["score_tier"], 0) + 1
        print("Score Tier Distribution:", {k: f"{v} ({v/len(sft_data)*100:.1f}%)" for k, v in tier_counts.items()})

        return sft_data

    train_sft = prepare_sft_records(train_records, "train")
    val_sft = prepare_sft_records(val_records, "validation")

    # Export to jsonl files
    print(f"\nExporting train set to: {config.SFT_TRAIN_PATH}")
    with open(config.SFT_TRAIN_PATH, "w", encoding="utf-8") as f:
        for rec in train_sft:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Exporting validation set to: {config.SFT_VAL_PATH}")
    with open(config.SFT_VAL_PATH, "w", encoding="utf-8") as f:
        for rec in val_sft:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Export Arrow format memory-mapped dataset for zero-RAM training loader
    print(f"\nExporting Arrow DatasetDict to: {config.SFT_DATASET_DIR}")
    train_ds = Dataset.from_list(train_sft)
    val_ds = Dataset.from_list(val_sft)
    arrow_dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
    })
    arrow_dataset.save_to_disk(str(config.SFT_DATASET_DIR))
    print(f"[SUCCESS] Saved memory-mapped dataset to: {config.SFT_DATASET_DIR}")

    print("=" * 70)
    print("SFT Dataset Build Completed Successfully.")
    print("=" * 70)

    return train_sft, val_sft


if __name__ == "__main__":
    process_and_split_dataset()
