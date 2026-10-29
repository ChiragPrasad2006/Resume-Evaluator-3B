# AI Resume Reviewer (Qwen2.5-3B-Instruct + LoRA)

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Adapter%20Hub-blue)](https://huggingface.co/Kami0867/Qwen-2.5-3B-Resume-Evaluator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Inference Quantization](https://img.shields.io/badge/Inference-8--bit%20INT8-purple.svg)](https://github.com/bitsandbytes-foundation/bitsandbytes)
[![Hardware Target](https://img.shields.io/badge/Hardware-RTX%205060%20Ti%208GB%20%2F%20Consumer%20GPU-orange.svg)](#hardware--vram-telemetry)

A production-grade, locally hostable AI Resume Screening and Review workspace powered by **FastAPI**, **Qwen2.5-3B-Instruct**, and fine-tuned with **QLoRA** on curated resume evaluations augmented with 500 synthetic hard negatives.

The fine-tuned LoRA adapter is hosted on Hugging Face Hub:  
👉 **[Kami0867/Qwen-2.5-3B-Resume-Evaluator](https://huggingface.co/Kami0867/Qwen-2.5-3B-Resume-Evaluator)**

Designed to run smoothly on consumer GPUs (**NVIDIA GeForce RTX 5060 Ti 8GB** and similar 8GB/12GB cards) in **8-bit INT8 precision**, allocating **~3.25 GB VRAM** (well below the 4.5 GB safety cap) while delivering zero quantization drift on score calculations and 100% strict JSON schema compliance.

---

## Table of Contents

- [Features](#features)
- [Hardware & VRAM Telemetry](#hardware--vram-telemetry)
- [Quickstart: How to Run the Application](#quickstart-how-to-run-the-application)
  - [1. Prerequisites](#1-prerequisites)
  - [2. Clone & Environment Setup](#2-clone--environment-setup)
  - [3. Run the Web UI](#3-run-the-web-ui)
- [Using the Web Interface](#using-the-web-interface)
- [CLI Evaluation & Schema Testing](#cli-evaluation--schema-testing)
- [Structured JSON Schema](#structured-json-schema)
- [Configuration & Environment Variables](#configuration--environment-variables)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [License](#license)

---

## Features

- **Hugging Face Hub Streaming:** The web server and CLI automatically fetch and attach the trained adapter directly from `Kami0867/Qwen-2.5-3B-Resume-Evaluator`.
- **On-The-Fly 8-Bit INT8 Inference:** Quantized via `bitsandbytes` (`load_in_8bit=True`, `llm_int8_threshold=6.0`), eliminating 4-bit score rounding artifacts.
- **Strict Structured JSON Output:** Validated with Pydantic; outputs overall score (0–100), categorical scores, strengths, red flags, hiring recommendations (`Strong Hire`, `Hire`, `Borderline`, `Do Not Hire`), and actionable improvement tips.
- **Modern Dark-Themed Web UI:** Built with FastAPI and a modern dark UI:
  - **Collapsible History Sidebar:** Auto-saves every candidate evaluation to local SQLite (`data/sessions.db`) with instant search and filter.
  - **File Upload & Parsing:** Drag-and-drop or select PDF resumes (extracted via `pypdf`) or paste raw resume text.
  - **Pre-set Job Description Templates:** Quick-fill templates for Full-Stack, ML Engineer, and Backend roles.
  - **Scorecard & Visual Breakdown:** Overall score gauge, progress bars for each category, color-coded recommendation pill, and one-click JSON copy.
  - **Live Hardware Telemetry Badge:** Real-time indicator displaying active device name, precision (`8-bit INT8`), and VRAM consumption in gigabytes.

---

## Hardware & VRAM Telemetry

The application uses an in-memory **Warm Singleton** architecture. The model is loaded once on server startup and kept warm across user requests, eliminating re-initialization latency.

### Memory Footprint on NVIDIA RTX 5060 Ti 8GB

| Component | Precision / Configuration | Footprint |
| :--- | :--- | :--- |
| **Qwen2.5-3B Base Weights** | 8-bit INT8 (BitsAndBytes) | **~3.09 GB** |
| **LoRA Adapter Weights** | FP16/BF16 projections (`r=16`, `alpha=32`) | **~96 MB** |
| **Active Inference Allocation**| Model + KV Cache + 1,400 token context | **~3.25 GB** |
| **Peak Reserved VRAM** | PyTorch Memory Allocator | **~4.14 GB** |
| **Hard Allocation Cap** | Protected via `torch.cuda.set_per_process_memory_fraction` | **4.50 GB** |
| **Free Headroom** | Remaining VRAM for OS & display manager | **~3.50+ GB** |

---

## Quickstart: How to Run the Application

### 1. Prerequisites

- **OS:** Windows 10/11, Linux, or macOS.
- **Python:** Version 3.10, 3.11, or 3.12.
- **GPU (Recommended):** NVIDIA GPU with at least 6GB VRAM (tested on RTX 5060 Ti 8GB).
  - *CPU Fallback:* The application will automatically fall back to CPU (`float32`) if no CUDA device is detected.

### 2. Clone & Environment Setup

Open PowerShell or your terminal:

```powershell
# 1. Clone the repository
git clone https://github.com/<your-username>/resume-eval.git
cd resume-eval

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
# source .venv/bin/activate

# 4. Install PyTorch with CUDA support (example for CUDA 12.4+):
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 5. Install application dependencies
pip install -r requirements.txt
```

### 3. Run the Web UI

#### Option A: One-Click PowerShell Launcher (Windows)

Run the included launcher script:

```powershell
.\run_ui.ps1
```

The script automatically detects the virtual environment, performs diagnostic pre-flight checks, starts the FastAPI server, and launches your default web browser to:

```text
http://127.0.0.1:7860
```

#### Option B: Cross-Platform Command Line (Windows / Linux / macOS)

Run directly via Python:

```bash
python -m web.app
```

Or using Uvicorn:

```bash
uvicorn web.app:app --host 127.0.0.1 --port 7860
```

*(On first launch, Hugging Face will automatically download the base model and adapter checkpoint `Kami0867/Qwen-2.5-3B-Resume-Evaluator` to your local cache.)*

---

## Using the Web Interface

1. **Job Description Input:**
   - Paste the target job description or select one of the built-in quick templates (e.g., *Full-Stack Engineer*, *ML Engineer*).
2. **Resume Input:**
   - **Upload PDF:** Drag and drop or browse for a `.pdf` file. The text is automatically extracted and populated into the resume field.
   - **Paste Text:** You can also paste raw plaintext markdown/text directly.
3. **Execute Review:**
   - Click **"Evaluate Candidate"**.
   - The UI shows a real-time loading spinner while the 8-bit inference engine analyzes candidate fit.
4. **Inspect Results:**
   - **Overall Score:** 0–100 circular gauge with matching verdict badge (`Strong Hire`, `Hire`, `Borderline`, `Do Not Hire`).
   - **Category Scores:** Breakdown across *Technical Skills*, *Experience Relevance*, *Education & Certifications*, and *Leadership & Impact*.
   - **Strengths & Red Flags:** Specific points highlighting qualifications or discrepancies (e.g. buzzword stuffing, missing metrics, chronological conflicts).
   - **Actionable Feedback:** Concrete bullet points for candidate coaching or hiring managers.
   - **Raw JSON:** Expandable view to inspect or copy the raw JSON payload with one click.
5. **Session History Sidebar:**
   - Every evaluation is persisted locally in SQLite (`data/sessions.db`).
   - Click any past session in the left sidebar to reload its evaluation, or use the search bar to filter by role or candidate name.

---

## CLI Evaluation & Schema Testing

You can also evaluate resumes or validate schemas from the command line:

```powershell
# Run the JSON schema validation self-test
python evaluate_reviewer.py --test-schema

# Run validation against a sample dataset
python evaluate_reviewer.py --num-samples 10 --adapter-path "Kami0867/Qwen-2.5-3B-Resume-Evaluator"
```

---

## Structured JSON Schema

The model adheres strictly to the following output structure:

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

---

## Configuration & Environment Variables

You can customize runtime behavior using environment variables or editing `config.py`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `RESUME_EVAL_ADAPTER` | `Kami0867/Qwen-2.5-3B-Resume-Evaluator` | Hugging Face Hub adapter repo or local checkpoint directory |
| `BASE_MODEL_NAME` | `Qwen/Qwen2.5-3B-Instruct` | Base Hugging Face model identifier |
| `WEB_HOST` | `127.0.0.1` | Web server host binding |
| `WEB_PORT` | `7860` | Web server port binding |

---

## Troubleshooting & FAQ

#### Q: How long does first-time startup take?
> On first run, `transformers` downloads the base model weights (~6 GB) and the LoRA adapter (~100 MB) from Hugging Face Hub. Subsequent runs load from local cache in ~10 seconds.

#### Q: Why 8-bit instead of 4-bit?
> While 4-bit NF4 saves an extra ~1 GB of VRAM, 8-bit INT8 eliminates quantization drift on score arithmetic and structured JSON formatting while fitting comfortably inside 3.25 GB on an 8GB card.

#### Q: Can I run without an NVIDIA GPU?
> Yes. If CUDA is unavailable, the application automatically falls back to CPU execution (`float32`). Note that inference speed will be slower on CPU.

---

## License

This project is licensed under the [MIT License](LICENSE). Base model weights are governed by the [Qwen Research License](https://github.com/QwenLM/Qwen2.5).
