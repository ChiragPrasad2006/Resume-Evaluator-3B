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
