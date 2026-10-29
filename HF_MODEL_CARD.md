---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:Qwen/Qwen2.5-3B-Instruct
- lora
- transformers
- resume-screener
- hr-tech
- recruitment
- json-generation
- qwen2.5
license: mit
language:
- en
---

# Qwen-2.5-3B-Resume-Evaluator

An objective, rubric-based AI Resume Screener and Evaluator adapter fine-tuned on [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) using **QLoRA**.

This model evaluates candidate resumes against detailed job descriptions, scoring qualifications across four specific categories and returning strict, parseable JSON with actionable feedback, identified strengths, and potential red flags.

---

## Model Details

### Model Description

- **Developed by:** Kami0867
- **Model type:** Parameter-Efficient Fine-Tuning (PEFT) / LoRA on Causal Language Model
- **Base Model:** [`Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Language(s):** English (`en`)
- **License:** MIT
- **Target Quantization:** 8-bit INT8 (inference: ~3.25 GB VRAM) or 4-bit NF4 (~2.0 GB VRAM)

### Key Capabilities

1. **Objective Scoring (0–100):** Comprehensive fit rating calibrated to avoid grade inflation.
2. **Category-Level Breakdown:**
   - *Technical Skills* (0–100)
   - *Experience Relevance* (0–100)
   - *Education & Certifications* (0–100)
   - *Leadership & Impact* (0–100)
3. **Adversarial & Failure Mode Detection:** Specifically augmented to penalize:
   - **Buzzword Stuffing:** Lists of technologies without corresponding practical accomplishments.
   - **Metric Stripping:** Vague claims lacking quantifiable scale, latency, throughput, or business results.
   - **Title Inflation:** Discrepancies between claimed job titles and actual task complexity.
   - **Chronological Inconsistencies:** Overlapping tenures or unaccounted career gaps.
4. **Structured JSON Output:** 100% strict JSON schema adherence with zero preamble or markdown drift.

---

## Intended Uses

### Direct Use
- Automated first-round resume screening and candidate ranking.
- Generation of detailed interview question topics based on candidate red flags or missing technical requirements.
- Personalized, constructive candidate feedback and resume coaching.

### Downstream Use
- Integration into Applicant Tracking Systems (ATS) and hiring pipelines.
- Batch candidate triage and ranking microservices.

### Out-of-Scope Use
- Fully automated hiring/rejection decisions without human recruiter review.
- Evaluation of resumes in non-English languages without prior domain adaptation.

---

## How to Get Started with the Model

### Installation

```bash
pip install torch transformers peft bitsandbytes accelerate
```

### Python Inference Example (8-Bit INT8)

```python
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_REPO = "Kami0867/Qwen-2.5-3B-Resume-Evaluator"

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# 2. Configure 8-bit Quantization (Low memory: ~3.25 GB VRAM)
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="cuda:0",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# 3. Attach Fine-Tuned LoRA Adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()

# 4. Define Prompt
system_prompt = (
    "You are an expert AI Resume Screener and Technical Recruiter. "
    "Evaluate candidate resumes against job descriptions with strict objectivity. "
    "Output MUST be a valid JSON object matching the requested schema."
)

job_description = """
Role: Senior Python / Backend Engineer
Requirements:
- 4+ years experience with Python, FastAPI, and PostgreSQL
- Demonstrated experience optimizing distributed databases and caching (Redis)
- Strong background in microservice architectures and containerization (Docker, Kubernetes)
"""

resume = """
Jane Doe
Senior Backend Developer
- 5 years architecting backend microservices in Python, FastAPI, and SQLAlchemy.
- Reduced database query latency by 42% by introducing Redis multi-tier caching and query indexing on PostgreSQL.
- Maintained Docker and Kubernetes CI/CD deployment pipelines on AWS.
"""

user_prompt = f"### JOB DESCRIPTION:\n{job_description}\n\n### RESUME:\n{resume}\n\nEvaluate and return structured JSON."

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")

# 5. Generate Inference
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
result_json = json.loads(response)
print(json.dumps(result_json, indent=2))
```

---

## Output Schema Example

```json
{
  "overall_score": 88,
  "summary": "Candidate profile is an excellent technical match for the Senior Backend Engineer position, showcasing clear impact metrics and deep familiarity with FastAPI, PostgreSQL, and distributed caching.",
  "category_scores": {
    "technical_skills": 90,
    "experience_relevance": 92,
    "education_and_certifications": 80,
    "leadership_and_impact": 82
  },
  "strengths": [
    "Direct experience with core required stack: Python, FastAPI, PostgreSQL, and Redis.",
    "Demonstrated measurable business impact (42% query latency reduction)."
  ],
  "weaknesses_and_red_flags": [
    "No explicit mention of cloud governance or distributed tracing frameworks."
  ],
  "recommendation": "Strong Hire",
  "actionable_feedback": [
    "Highlight specific throughput numbers (e.g. QPS scale) handled in the production microservices."
  ]
}
```

---

## Training Details

### Training Data

The model was trained on a multi-stage dataset:
1. **Base Dataset:** Curated and standardized samples from `netsol/resume-score-details`.
2. **Synthetic Hard Negatives & Borderlines:** 500 synthesized candidates generated to teach the model how to spot common flaws:
   - Keyword stuffing without supporting accomplishments
   - Stripped outcome metrics
   - Discrepancies between job title and junior-level responsibilities
   - Unexplained career gaps and overlapping concurrent full-time roles
3. **Loss Masking:** Formatted using ChatML with completion masks (`labels = -100` on prompt tokens; loss computed strictly on assistant JSON completions).

### Training Hyperparameters

- **Training Regime:** QLoRA with 4-bit NormalFloat (NF4) compute in `bfloat16`
- **Optimizer:** `paged_adamw_8bit`
- **Learning Rate:** 8e-5 (Cosine decay, 5% warmup)
- **Effective Batch Size:** 8 (Micro-batch size: 1, Gradient accumulation steps: 8)
- **Epochs:** 3 (432 optimization steps)
- **NEFTune Noise Alpha:** 5
- **LoRA Configuration:**
  - `r`: 16
  - `lora_alpha`: 32
  - `lora_dropout`: 0.05
  - `target_modules`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (All 7 linear layers)

### Hardware & Compute

- **Hardware:** Single NVIDIA GeForce RTX 5060 Ti 8GB (Blackwell Architecture)
- **VRAM Utilization:** Peak 2.08 GB during training
- **Training Duration:** ~1.5 hours

---

## Evaluation Results

- **JSON Schema Conformance Rate:** 100.0%
- **Overall Score Mean Absolute Error (MAE):** 3.8 points on holdout validation split
- **Recommendation Classification Fidelity:** > 92% agreement with human recruiter rubrics

---

## Framework Versions

- **PEFT:** 0.21.2
- **Transformers:** 4.48+
- **PyTorch:** 2.4+ / 2.14+
- **BitsAndBytes:** 0.45+ / 0.50+
- **Datasets:** 3.0+
