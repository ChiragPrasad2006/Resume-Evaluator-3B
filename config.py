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
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

configure_memory_environment()

import torch

# Determine execution device
HAS_CUDA = torch.cuda.is_available()
DEVICE = "cuda" if HAS_CUDA else "cpu"

# ==============================================================================
# 2. File & Directory Paths
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_adapter"

# Data Pipeline Artifacts
CURATED_DATA_PATH = DATA_DIR / "curated_resumes.jsonl"
HARD_NEGATIVES_PATH = DATA_DIR / "hard_negatives.jsonl"
SFT_TRAIN_PATH = DATA_DIR / "sft_train.jsonl"
SFT_VAL_PATH = DATA_DIR / "sft_val.jsonl"

for directory in [DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, FINAL_MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 3. Model & Quantization Configuration
# ==============================================================================
MODEL_NAME = "Qwen/Qwen2.5-3B"
MAX_SEQ_LENGTH = 2048  # Strict bound for 8GB VRAM limit

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
# 4. LoRA Hyperparameters
# ==============================================================================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
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
# 5. Training Hyperparameters (8GB VRAM Optimized)
# ==============================================================================
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # 1 sample per step keeps memory tightly under 6GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
LOGGING_STEPS = 10
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2
EVAL_STEPS = 100
GRADIENT_CHECKPOINTING = True
OPTIMIZER = "paged_adamw_8bit" if HAS_CUDA else "adamw_torch"

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
