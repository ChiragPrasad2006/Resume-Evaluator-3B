# Resume-Evaluator-3B (QLoRA Fine-Tuning & Synthetic Data Pipeline)

A production-grade QLoRA fine-tuning and synthetic data augmentation pipeline for an AI Resume Reviewer based on `Qwen/Qwen2.5-3B` and the `netsol/resume-score-details` dataset.

Engineered for strict **8GB VRAM budget** execution (targeting **NVIDIA GeForce RTX 5060 Ti Blackwell architecture** with native `bfloat16` and 4-bit NormalFloat quantization).

---

## Architecture Overview

```text
├── TODO.md                         # Live implementation checklist
├── config.py                       # Centralized paths, hyperparameters, prompts, and schema
├── data/                           # Data artifacts directory
│   ├── curated_resumes.jsonl       # 851 cleaned and standardized resume evaluations
│   ├── hard_negatives.jsonl        # 500 mutated hard negatives & borderlines
│   ├── sft_train.jsonl             # 1,148 stratified ChatML training samples (with completion masks)
│   └── sft_val.jsonl               # 203 stratified ChatML validation samples
├── data_pipeline/
│   ├── inspect_and_curate.py       # Downloads & cleans netsol/resume-score-details
│   ├── synthesize_hard_negatives.py# Mutates resumes into 500 Hard Negatives & Borderlines
│   └── build_sft_dataset.py        # Merges, stratifies, formats ChatML, and builds completion masks
├── train_qlora.py                  # QLoRA Trainer with --dry-run CLI flag and 8GB memory guards
└── evaluate_reviewer.py            # JSON schema validator, MAE score drift & inference tester
```

---

## 8GB VRAM Memory Budget Breakdown

| Component | Precision / Configuration | Estimated Memory Footprint |
| :--- | :--- | :--- |
| **Qwen2.5-3B Base Weights** | 4-bit NormalFloat (NF4 double-quant) | **~2.2 GB** |
| **LoRA Trainable Adapters** | FP32 (`r=16`, `alpha=32`, all 7 projections) | **~35 MB** |
| **KV Cache & Activations** | Native PyTorch SDPA + Gradient Checkpointing | **~1.8 - 2.5 GB** |
| **Optimizer States** | `paged_adamw_8bit` (Paged into system RAM on spikes) | **~75 MB** |
| **Total Peak VRAM Footprint**| Max Seq Length 2048, Batch Size 1, Grad Accum 8 | **~4.5 - 5.5 GB** |
| **Safety Headroom** | Inside 8GB Physical VRAM Limit | **~2.5 - 3.5 GB Available** |

### Critical Memory Flags Implemented
- `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"`: Eliminates CUDA memory fragmentation.
- `attn_implementation="sdpa"`: Uses PyTorch's native Scaled Dot-Product Attention without Triton compilation dependencies.
- `gradient_checkpointing=True`: Trades recomputation for massive activation memory savings.
- `dataloader_pin_memory=False`: Prevents unmanaged RAM lockups on Windows.

---

## Quickstart Guide

### 1. Virtual Environment Activation
Always activate the virtual environment:
```powershell
powershell -ExecutionPolicy Bypass -Command "& 'D:\Resolve\.venv\Scripts\Activate.ps1'"
```

### 2. Data Pipeline Execution (Already Generated)
The data artifacts have already been processed and cached into `data/`:
1. **Curate Clean Resumes:**
   ```powershell
   python data_pipeline/inspect_and_curate.py
   ```
   *Result:* 851 curated records with ground-truth score distributions.
2. **Synthesize Hard Negatives & Borderlines:**
   ```powershell
   python data_pipeline/synthesize_hard_negatives.py
   ```
   *Result:* 500 mutated resumes covering buzzword stuffing, metric stripping, title inflation, chronological conflicts, and missing hard requirements.
3. **Build SFT ChatML Dataset:**
   ```powershell
   python data_pipeline/build_sft_dataset.py
   ```
   *Result:* 1,148 train and 203 validation samples with completion masks (`labels = -100` for prompts, active loss on assistant JSON output only).

### 3. Dry-Run Verification (Mandatory Before Full Training)
To verify memory allocation, forward/backward passes, and loss decrease on 4 samples:
```powershell
python train_qlora.py --dry-run
```

### 4. Full Production Training (Run on RTX 5060 Ti)
To launch the full 3-epoch QLoRA fine-tuning on the RTX 5060 Ti:
```powershell
python train_qlora.py
```
Checkpoints will be saved to `outputs/checkpoints/` and the final merged adapter to `outputs/final_adapter/`.

### 5. Evaluation & Inference Testing
- **Test Schema Validation Engine:**
  ```powershell
  python evaluate_reviewer.py --test-schema
  ```
- **Evaluate Trained Model against Validation Set:**
  ```powershell
  python evaluate_reviewer.py --val-file data/sft_val.jsonl --adapter-path outputs/final_adapter
  ```

---

## Structured Output Schema

The model is trained to strictly output objective, parseable JSON:

```json
{
  "overall_score": 82,
  "summary": "Candidate profile demonstrates high technical competency in backend architecture...",
  "category_scores": {
    "technical_skills": 85,
    "experience_relevance": 80,
    "education_and_certifications": 90,
    "leadership_and_impact": 75
  },
  "strengths": [
    "Extensive hands-on mastery of scalable microservices and distributed databases."
  ],
  "weaknesses_and_red_flags": [
    "Absence of metrics detailing latency reduction or system throughput improvements."
  ],
  "recommendation": "Hire",
  "actionable_feedback": [
    "Quantify impact of project milestones using explicit scale metrics (QPS, throughput, latency)."
  ]
}
```
