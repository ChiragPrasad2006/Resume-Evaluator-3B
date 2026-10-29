# Resume Evaluator 3B - QLoRA Fine-Tuning & Synthetic Data Pipeline
Live Implementation and Verification Checklist

- [x] **1. Environment & Architecture Planning**
  - [x] Virtual environment verification (`D:\Resolve\.venv\Scripts\Activate.ps1`)
  - [x] Target hardware specs (RTX 5060 Ti 8GB VRAM budget, bfloat16, NF4, SDPA) with CPU/CUDA-agnostic fallback for dev
  - [x] Project architecture blueprint and roadmap

- [x] **2. Centralized Configuration (`config.py`)**
  - [x] Model identifier (`Qwen/Qwen2.5-3B`) and local cache paths
  - [x] Strict 8GB VRAM settings (`expandable_segments`, NF4 BitsAndBytes, bfloat16, SDPA, gradient checkpointing)
  - [x] Resume review JSON output schema (scores 0-100, category breakdowns, strengths, red flags, suggestions)
  - [x] Standardized System Prompt & ChatML format templates

- [x] **3. Data Inspection & Curation (`data_pipeline/inspect_and_curate.py`)**
  - [x] Dataset retrieval (`netsol/resume-score-details`)
  - [x] Schema discovery, deduplication, length filtering, and text normalization
  - [x] Ground-truth score mapping & JSON-formatted assistant response generation
  - [x] Export curated dataset (`data/curated_resumes.jsonl` — 851 clean samples)

- [x] **4. Hard Negatives & Edge-Case Synthesizer (`data_pipeline/synthesize_hard_negatives.py`)**
  - [x] Realistic resume mutation engine (buzzword stuffing, missing metrics, skill inflation, conflicting dates, vague summaries)
  - [x] Generate 400-600 calibrated hard negatives and borderline samples (500 generated)
  - [x] Produce rigorous, penalizing critique JSON outputs
  - [x] Export synthetic dataset (`data/hard_negatives.jsonl` — 500 mutated samples)

- [x] **5. ChatML SFT Dataset Builder (`data_pipeline/build_sft_dataset.py`)**
  - [x] Merge curated (851) and synthetic (500) datasets (1,351 total)
  - [x] Stratified train/val split (85/15: 1,148 train / 203 val) across score tiers
  - [x] Tokenize and build ChatML conversations with completion masking (loss on assistant tokens only)
  - [x] Export ready-to-train datasets (`data/sft_train.jsonl`, `data/sft_val.jsonl`)

- [x] **6. Production QLoRA Fine-Tuning Script (`train_qlora.py`)**
  - [x] `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"` pre-torch setup
  - [x] BitsAndBytes 4-bit NF4 double quant + SDPA attention implementation
  - [x] LoRA target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
  - [x] Memory-efficient training arguments (batch size 1, gradient accumulation 8, bfloat16, gradient checkpointing)
  - [x] Mandatory `--dry-run` CLI flag (loads 4 samples, executes forward/backward steps, checks memory budget, confirms loss decrease, and exits cleanly)

- [x] **7. Evaluation & Inference Suite (`evaluate_reviewer.py`)**
  - [x] Strict JSON schema parsing and extraction validator
  - [x] MAE (Mean Absolute Error) score drift assessment
  - [x] Recommendation classification accuracy & category-level metrics
  - [x] CLI runner for evaluating arbitrary resume text or PDF inputs against trained/base model

- [x] **8. End-to-End Verification & Operational Documentation**
  - [x] Validated data curation and SFT generation pipeline
  - [x] Validated schema conformance tests (`python evaluate_reviewer.py --test-schema`)
  - [x] Production deployment instructions documented for RTX 5060 Ti training run

## [OOM & Checkpoint Recovery Plan]
- [x] **Fix 1: Hard Memory Caps & Windows RAM Protection**
  - [x] Update `config.py` with `expandable_segments:True,max_split_size_mb:64` and `set_per_process_memory_fraction(0.85)`
  - [x] Set `dataloader_num_workers = 0` and `dataloader_pin_memory = False` in `config.py` & `train_qlora.py`
  - [x] Lower `max_seq_length = 1400`, set `prediction_loss_only = True`
  - [x] Enable `model.enable_input_require_grads()` and `model.config.use_cache = False`
  - [x] Adjust hyperparameters: `per_device_train_batch_size = 1`, `gradient_accumulation_steps = 8`, `learning_rate = 8e-5`
  - [x] Compact LoRA configuration: `r = 12`, `lora_alpha = 24`, `lora_dropout = 0.08`
- [x] **Fix 2: Aggressive, Crash-Proof Checkpointing & Auto-Resume**
  - [x] Configure `save_strategy = "steps"`, `save_steps = 5`, `save_total_limit = 3`, `eval_steps = 5` (aligned for `load_best_model_at_end`)
  - [x] Implement automatic checkpoint detection and resumption via `get_last_checkpoint`
  - [x] Wrap `trainer.train(...)` with `try...except (KeyboardInterrupt, RuntimeError, Exception)` for emergency adapter saves to `outputs/emergency_interrupt_checkpoint`
  - [x] Implement `MemoryCleanCallback` with explicit GC and `torch.cuda.empty_cache()` on step end (every 2 steps), evaluate, save, and log VRAM & System RAM %
- [x] **Fix 3: Atomic Per-Row Save Points in Data Pipeline**
  - [x] Update `inspect_and_curate.py` with skip check for `data/processed/base_curated.jsonl` (unless `--force`)
  - [x] Update `synthesize_hard_negatives.py` with resume from existing source_id/hashes and atomic flush/fsync per row
  - [x] Update `build_sft_dataset.py` to save `DatasetDict` to `data/processed/sft_dataset` for zero-RAM memory-mapped loading
- [x] **Verification & Validation**
  - [x] Rebuilt `sft_dataset` with 100% target JSON preservation (zero samples with NaN loss)
  - [x] Resolved `TrainOutput.total_flos` attribute error via safe `getattr` fallback
  - [x] Run `train_qlora.py --dry-run` (verified decreasing eval loss 1.665 -> 1.649, Peak VRAM 5.20 GB < 6.80 GB, System RAM 62.3% < 75%)
  - [x] Cleaned dry-run checkpoints; verified clean state for full production run via `run_training.ps1`
  - [x] Completed full production training run (432/432 steps, final adapter saved to `model_checkpoints/final_adapter`)

## [Production Web UI Plan]
- [x] **1. SQLite Persistence Layer (`data/sessions.db`)**
  - [x] Schema creation: `sessions` table (id, title, raw_jd, compact_jd, created_at, updated_at)
  - [x] Schema creation: `evaluations` table (id, session_id, candidate_name, raw_resume, overall_score, recommendation, parsed_json, created_at)
  - [x] Data access functions: create/list/get/delete session, add evaluation, export session to CSV/JSON
- [x] **2. JD Token Budgeting & Requirements Processor (`web/jd_processor.py`)**
  - [x] Token count calculation using Qwen tokenizer heuristic
  - [x] Role title extraction from first line/header
  - [x] Smart extractive summarizer for JDs > 800 tokens to ensure total prompt stays under 1400 tokens
- [x] **3. Warm Singleton Inference Engine (`web/inference_engine.py`)**
  - [x] Load base Qwen 2.5 3B in 4-bit NF4 (`bitsandbytes`) with `bfloat16` compute dtype
  - [x] Attach fine-tuned LoRA adapter from `model_checkpoints/final_adapter`
  - [x] Strict non-drifting context builder: `[System Instruction] + [Active JD] + [Candidate Resume]`
  - [x] Robust JSON response extraction & fallback validation
  - [x] Keep VRAM allocations under 4.5 GB with memory cleanup (Measured Peak: 2.16 GB / 8.0 GB)
- [x] **4. Resume File Ingestion & Parsing (`web/file_parser.py`)**
  - [x] Text parser for pasted resume strings
  - [x] PDF parser using `pypdf`
  - [x] DOCX parser using `python-docx`
- [x] **5. Modern Dark-Themed SPA Interface & Sound Integration (`web/static/index.html`)**
  - [x] Collapsible sidebar with session list grouped by date ("Today", "Previous 7 Days", etc.)
  - [x] Anchored collapsible "Target Job Description" header banner
  - [x] Candidate evaluation stream with rich cards (circular score meter, category bars, skill chips, raw JSON toggle)
  - [x] Responsive input dock with file drag-and-drop & paste
  - [x] Embedded zero-dependency procedural Web Audio sound synthesizer (`ui-sound-explorer`: `pop`, `chime`, `whisk`, `sigh`, `hire`)
- [x] **6. FastAPI Application (`web/app.py`)**
  - [x] REST API routes for sessions, JD submission, candidate evaluation, file upload, and exports
  - [x] Static file serving for SPA
- [x] **7. Launcher & End-to-End Verification (`run_ui.ps1`)**
  - [x] Created `run_ui.ps1` PowerShell launcher
  - [x] Tested session creation with sample JD
  - [x] Tested candidate evaluation on multiple resumes (Strong Hire vs Reject)
  - [x] Verified VRAM Peak 2.16 GB (< 4.5 GB) and session persistence across reloads

## [8-Bit Inference Engine Upgrade (BitsAndBytes INT8)]
- [x] **1. 8-Bit BitsAndBytes Quantization Configuration (`web/inference_engine.py`)**
  - [x] Configured `BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=False, llm_int8_threshold=6.0)`
  - [x] Loaded base model `Qwen/Qwen2.5-3B-Instruct` with `device_map="cuda:0"`, `attn_implementation="sdpa"`, and `dtype=torch.float16` to eliminate casting overhead
  - [x] Attached trained LoRA adapter from `model_checkpoints/final_adapter`
  - [x] Set `model.eval()` and `model.config.use_cache = True` for fast auto-regressive generation
- [x] **2. Deterministic Generation & Memory Safeguards**
  - [x] Wrapped generation inside `with torch.inference_mode():`
  - [x] Tokenized inputs with Qwen ChatML template to `cuda:0`
  - [x] Configured generation parameters: `max_new_tokens=768`, greedy decoding (`do_sample=False`), `repetition_penalty=1.05`, `eos_token_id=151645 (<|im_end|>)`
  - [x] Extracted newly generated tokens `outputs[0][inputs["input_ids"].shape[-1]:]` and purged activation tensors with `torch.cuda.empty_cache()`
- [x] **3. Diagnostic Telemetry & Verification**
  - [x] Added startup VRAM diagnostic `[8-BIT INFERENCE READY]`
  - [x] Exposed `get_reviewer_model()` programmatic accessor for CLI and tests
  - [x] Verified baseline VRAM: Allocated 3.25 GB, Reserved 4.31 GB (< 4.5 GB budget)
  - [x] Verified peak generation VRAM: 3.40 GB (< 4.5 GB target)
  - [x] Verified schema extraction & evaluation output (Score: 70, Outcome: "Hire", valid JSON keys) without 4-bit precision drift
  - [x] Replaced hardcoded static "4-bit NF4" strings in `web/static/index.html` with dynamic `/api/model/status` endpoint reading live PyTorch metadata (`Qwen/Qwen2.5-3B-Instruct`, `8-bit INT8 (BitsAndBytes)`, `3.25 GB` VRAM on `RTX 5060 Ti`)
  - [x] Removed extraneous audio synthesizers and sound button from UI footer for a clean, distraction-free interface
