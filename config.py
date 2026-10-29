"""
Centralized Configuration for AI Resume Reviewer (Qwen2.5-3B QLoRA)
Targeted for 8GB VRAM (NVIDIA RTX 5060 Ti Blackwell architecture)
with robust environment fallback and strict memory management.
"""

import os
from pathlib import Path
from typing import Dict, Any

# ==============================================================================
# 1. Environment & CUDA Memory Configuration (Applied before torch import)
# ==============================================================================
def configure_memory_environment() -> None:
    """Configures PyTorch memory allocator to eliminate fragmentation on 8GB VRAM."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # Avoid Rust transfer deadlocks on Windows
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"    # Disable hf-xet hook deadlocks on Windows

configure_memory_environment()

import torch

# Determine execution device & cap VRAM allocation to prevent Windows shared RAM spillover
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"
if HAS_CUDA:
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

# ==============================================================================
# 2. File & Directory Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Dedicated project folder for safe model states and checkpoints
CHECKPOINT_DIR = BASE_DIR / "model_checkpoints"
FINAL_MODEL_DIR = CHECKPOINT_DIR / "final_adapter"
EMERGENCY_CHECKPOINT_DIR = BASE_DIR / "outputs" / "emergency_interrupt_checkpoint"

# Hugging Face Hub Model & Adapter Repositories
HF_ADAPTER_REPO = "Kami0867/Qwen-2.5-3B-Resume-Evaluator"
DEFAULT_ADAPTER_MODEL = os.getenv("RESUME_EVAL_ADAPTER", HF_ADAPTER_REPO)

# Data Pipeline Artifacts
CURATED_DATA_PATH = DATA_DIR / "curated_resumes.jsonl"
HARD_NEGATIVES_PATH = DATA_DIR / "hard_negatives.jsonl"
SFT_TRAIN_PATH = DATA_DIR / "sft_train.jsonl"
SFT_VAL_PATH = DATA_DIR / "sft_val.jsonl"

# Processed Arrow / Memory-Mapped Artifacts
BASE_CURATED_PATH = PROCESSED_DATA_DIR / "base_curated.jsonl"
SYNTHETIC_HARD_NEGATIVES_PATH = PROCESSED_DATA_DIR / "synthetic_hard_negatives.jsonl"
SFT_DATASET_DIR = PROCESSED_DATA_DIR / "sft_dataset"

for directory in [DATA_DIR, PROCESSED_DATA_DIR, CHECKPOINT_DIR, FINAL_MODEL_DIR, EMERGENCY_CHECKPOINT_DIR.parent]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 3. Model & Quantization Configuration
# ==============================================================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1536  # Bound ensuring peak VRAM stays <= 5.2 GB (< 6.80 GB cap) while accommodating full target JSONs

# Blackwell RTX 5060 Ti natively supports bfloat16
COMPUTE_DTYPE = torch.bfloat16 if HAS_CUDA and torch.cuda.is_bf16_supported() else (torch.float16 if HAS_CUDA else torch.float32)

# BitsAndBytes 4-bit NormalFloat configuration for 8GB VRAM budget
BNB_CONFIG_PARAMS: Dict[str, Any] = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.bfloat16 if HAS_CUDA else torch.float32,
}

# Scaled Dot-Product Attention for Windows efficiency without triton dependency
ATTN_IMPLEMENTATION = "sdpa"

# ==============================================================================
# 4. LoRA Hyperparameters (Optimized for 8GB VRAM & 16GB System RAM)
# ==============================================================================
LORA_R = 12         # Saves ~25% adapter activation memory vs r=16
LORA_ALPHA = 24     # 2 * r scaling
LORA_DROPOUT = 0.08 # Regularization for ~1.4k samples
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
LORA_BIAS = "none"

# ==============================================================================
# 5. Training Hyperparameters (Tuned for 8GB VRAM + Generalization)
# ==============================================================================
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8   # Reduced to 8 for faster optimizer steps & lower peak memory
EVAL_ACCUMULATION_STEPS = 4       # Prevents VRAM OOM spikes during eval
LEARNING_RATE = 8e-5              # Scaled for effective batch size 8
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.05
MAX_GRAD_NORM = 1.0
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 10

# Accuracy Booster — NEFTune
NEFTUNE_NOISE_ALPHA = 5.0
PREDICTION_LOSS_ONLY = True       # Never store 151,936-dim logits in RAM/VRAM during eval

# Precision & Hardware Optimizations
BF16 = True
FP16 = False
GRADIENT_CHECKPOINTING = True
GRADIENT_CHECKPOINTING_KWARGS = {"use_reentrant": False}
OPTIMIZER = "paged_adamw_8bit" if HAS_CUDA else "adamw_torch"  # NVIDIA Unified Memory (UVM) Paging

# Windows Multiprocessing & Page-Lock RAM Protection (CRITICAL for 16GB RAM)
DATALOADER_PIN_MEMORY = False     # Prevents page-locked physical RAM starvation on Windows
DATALOADER_NUM_WORKERS = 0        # Eliminates spawn duplicating memory across processes
GROUP_BY_LENGTH = True            # Packs similar-length resumes to minimize padding waste

# Safe States & Checkpointing Configuration (Saved inside project folder)
SAVE_STRATEGY = "steps"
SAVE_STEPS = 5                    # Frequent checkpointing every 5 optimizer steps
SAVE_TOTAL_LIMIT = 3              # Retain top 3 checkpoints
EVAL_STRATEGY = "steps"
EVAL_STEPS = 5                    # Evaluate every 5 steps (must divide save_steps for load_best_model_at_end)
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "eval_loss"
GREATER_IS_BETTER = False
SAVE_SAFETENSORS = True           # Secure, zero-copy safetensors format
EARLY_STOPPING_PATIENCE = 10      # Halts if validation loss plateaus for 10 evaluations (50 steps)

# ==============================================================================
# 6. Structured Output Schema & Reviewer Persona
# ==============================================================================
SYSTEM_PROMPT = """You are an elite, objective Senior Technical Recruiter and Hiring Manager.
Your role is to critically evaluate a candidate's Resume against a specified Job Description (JD).

Evaluation Guidelines:
1. Grounded Analysis: Assess ONLY verifiable claims in the resume. Do not assume unstated competencies.
2. Penalize Red Flags: Strictly penalize keyword stuffing, generic claims devoid of metrics/outcomes, inflated titles, and chronological inconsistencies.
3. Structured Output: Always format your review strictly as valid JSON adhering to the specified schema, without markdown formatting outside the JSON block.

JSON Output Schema:
{
  "overall_score": <integer from 0 to 100>,
  "summary": "<2-3 sentence executive evaluation of candidate fit>",
  "category_scores": {
    "technical_skills": <integer 0-100>,
    "experience_relevance": <integer 0-100>,
    "education_and_certifications": <integer 0-100>,
    "leadership_and_impact": <integer 0-100>
  },
  "strengths": [
    "<specific strength with evidence from resume>"
  ],
  "weaknesses_and_red_flags": [
    "<critical gap, missing requirement, or red flag>"
  ],
  "recommendation": "<Strong Hire | Hire | Borderline | Reject>",
  "actionable_feedback": [
    "<practical advice for resume improvement or interview focus>"
  ]
}"""

USER_PROMPT_TEMPLATE = """Evaluate the candidate's resume for the provided job description.

### JOB DESCRIPTION:
{job_description}

### CANDIDATE RESUME:
{resume}

Provide your structured evaluation as valid JSON matching the schema."""

# Expected keys in JSON response for schema validation
REQUIRED_JSON_KEYS = {
    "overall_score",
    "summary",
    "category_scores",
    "strengths",
    "weaknesses_and_red_flags",
    "recommendation",
    "actionable_feedback",
}

REQUIRED_CATEGORY_KEYS = {
    "technical_skills",
    "experience_relevance",
    "education_and_certifications",
    "leadership_and_impact",
}

VALID_RECOMMENDATIONS = {"Strong Hire", "Hire", "Borderline", "Reject"}
