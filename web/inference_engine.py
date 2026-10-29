"""
Warm Singleton Inference Engine for AI Resume Reviewer.
Loads Qwen/Qwen2.5-3B-Instruct in 8-bit (BitsAndBytes INT8) with SDPA attention
and attaches the fine-tuned LoRA adapter.
Eliminates 4-bit quantization drift on score calculations and structured JSON
outputs while keeping inference allocations under 4.5 GB VRAM on 8GB GPUs.
"""

import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Pre-torch allocator configuration to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Ensure PyTorch does not exceed safe memory bounds
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.85, device=0)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def detect_candidate_name(resume_text: str, fallback_idx: int = 1) -> str:
    """Extracts candidate name from top lines of the resume, or generates a clean fallback."""
    if not resume_text or not resume_text.strip():
        return f"Candidate #{fallback_idx}"

    lines = [line.strip() for line in resume_text.strip().splitlines() if line.strip()]
    if not lines:
        return f"Candidate #{fallback_idx}"

    for line in lines[:4]:
        # Filter out contact info lines
        if any(c in line for c in ["@", "http", "www", "github", "linkedin", "phone", "+"]):
            continue
        cleaned = re.sub(r"^[#*\-•>\s]+", "", line).strip()
        words = cleaned.split()
        if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            return cleaned[:40]

    return f"Candidate #{fallback_idx}"


class InferenceEngine:
    _instance: Optional["InferenceEngine"] = None
    _lock = threading.Lock()

    def __init__(self, adapter_path: Optional[Any] = None, model_name: Optional[str] = None):
        self.adapter_path = adapter_path or config.DEFAULT_ADAPTER_MODEL
        self.model_name = model_name or config.MODEL_NAME
        self.model = None
        self.tokenizer = None
        self.is_ready = False
        self._load_lock = threading.Lock()

    @classmethod
    def get_instance(cls, adapter_path: Optional[Path] = None, model_name: Optional[str] = None) -> "InferenceEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(adapter_path=adapter_path, model_name=model_name)
            return cls._instance

    def load_model(self) -> None:
        """Loads model into GPU memory once in 8-bit precision (Warm Singleton)."""
        with self._load_lock:
            if self.is_ready:
                return

            print("\n" + "=" * 60)
            print("[InferenceEngine] Initializing 8-Bit Warm Singleton Model...")
            print(f"  Base Model  : {self.model_name}")
            print(f"  Adapter Path: {self.adapter_path}")
            print("=" * 60)

            # 1. Load Tokenizer
            target_model_name = self.model_name
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    target_model_name,
                    trust_remote_code=True,
                    local_files_only=True,
                )
            except Exception:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        config.MODEL_NAME,
                        trust_remote_code=True,
                        local_files_only=True,
                    )
                    target_model_name = config.MODEL_NAME
                except Exception:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        target_model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                    )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. Configure 8-bit Quantization
            if torch.cuda.is_available():
                print(f"  Target Device: {torch.cuda.get_device_name(0)} (device_map='cuda:0')")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False,
                    llm_int8_threshold=6.0,  # Standard outlier threshold for INT8 precision stability
                )
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        target_model_name,
                        quantization_config=quantization_config,
                        dtype=torch.float16,
                        device_map="cuda:0",
                        attn_implementation="sdpa",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=True,
                    )
                except Exception:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        target_model_name,
                        quantization_config=quantization_config,
                        dtype=torch.float16,
                        device_map="cuda:0",
                        attn_implementation="sdpa",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        local_files_only=False,
                    )
            else:
                print("  Target Device: CPU (Fallback)")
                self.model = AutoModelForCausalLM.from_pretrained(
                    target_model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )

            # 3. Attach Fine-Tuned LoRA Adapter (Hugging Face Hub or Local Path)
            adapter_source = str(self.adapter_path)
            is_local = os.path.exists(adapter_source) and os.path.isdir(adapter_source)
            source_desc = "Local Disk" if is_local else "Hugging Face Hub"

            print(f"  Attaching LoRA Adapter from: {adapter_source} ({source_desc})")
            try:
                self.model = PeftModel.from_pretrained(self.model, adapter_source)
                print("  LoRA Adapter attached successfully.")
            except Exception as e:
                if not is_local and config.FINAL_MODEL_DIR.exists() and any(config.FINAL_MODEL_DIR.iterdir()):
                    print(f"  [Fallback] Remote hub loading failed ({e}). Loading from local checkpoint: {config.FINAL_MODEL_DIR}")
                    self.model = PeftModel.from_pretrained(self.model, str(config.FINAL_MODEL_DIR))
                    self.adapter_path = config.FINAL_MODEL_DIR
                    print("  LoRA Adapter attached from local fallback.")
                else:
                    raise RuntimeError(f"Could not attach LoRA adapter from {adapter_source}: {e}")

            self.model.eval()
            self.model.config.use_cache = True
            self.is_ready = True

            # 4. Startup VRAM Diagnostic
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
                vram_res = torch.cuda.memory_reserved() / (1024 ** 3)
                print(f"[8-BIT INFERENCE READY] VRAM Allocated: {vram_alloc:.2f} GB | Reserved: {vram_res:.2f} GB")
                print("=" * 60 + "\n")

    def get_status(self) -> Dict[str, Any]:
        """Returns live system telemetry and active model metadata directly from PyTorch/CUDA."""
        vram_alloc = round(torch.cuda.memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
        vram_res = round(torch.cuda.memory_reserved() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

        return {
            "is_ready": self.is_ready,
            "base_model": self.model_name,
            "quantization": "8-bit INT8 (BitsAndBytes)",
            "adapter_path": str(self.adapter_path),
            "device": device_name,
            "vram_allocated_gb": vram_alloc,
            "vram_reserved_gb": vram_res,
            "status_label": f"Model Ready (8-bit INT8 | {vram_alloc} GB)" if self.is_ready else "Initializing Model...",
        }

    def evaluate_resume(
        self,
        job_description: str,
        resume_text: str,
    ) -> Dict[str, Any]:
        """
        Executes strict non-drifting evaluation using 8-bit precision.
        Prompt strictly contains: [System Instruction] + [Active JD] + [Candidate Resume]
        """
        if not self.is_ready:
            self.load_model()

        # Build clean non-drifting ChatML prompt
        user_content = config.USER_PROMPT_TEMPLATE.format(
            job_description=job_description.strip(),
            resume=resume_text.strip(),
        )

        messages = [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        target_device = "cuda:0" if torch.cuda.is_available() else self.model.device
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(target_device)

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id is None:
            im_end_id = self.tokenizer.eos_token_id

        # Wrap generation strictly inside torch.inference_mode()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.05,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_output_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Free intermediate activation tensors
        del inputs
        del outputs
        del gen_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        parsed_json, parse_status = self._extract_json(raw_output_text)
        if parsed_json is None:
            # Fallback structure preserving raw response
            parsed_json = self._build_fallback_json(raw_output_text)

        # Normalize score and recommendation
        overall_score = int(parsed_json.get("overall_score", 50))
        recommendation = parsed_json.get("recommendation")
        if not recommendation:
            if overall_score >= 80:
                recommendation = "Strong Hire" if overall_score >= 90 else "Hire"
            elif overall_score >= 60:
                recommendation = "Borderline"
            else:
                recommendation = "Reject"
            parsed_json["recommendation"] = recommendation

        return {
            "overall_score": overall_score,
            "recommendation": recommendation,
            "parsed_json": parsed_json,
            "raw_text": raw_output_text,
            "parse_status": parse_status,
        }

    def _extract_json(self, text: str) -> Tuple[Optional[Dict[str, Any]], str]:
        """Robust JSON extraction handling markdown tags, preamble, or raw braces."""
        if not text or not text.strip():
            return None, "Empty model output"

        cleaned = text.strip()

        # Pattern 1: ```json ... ```
        block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
        if block_match:
            try:
                return json.loads(block_match.group(1).strip()), "Success (code block)"
            except Exception as e:
                pass

        # Pattern 2: Outermost curly braces
        brace_match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1).strip()), "Success (braces)"
            except Exception as e:
                pass

        # Pattern 3: Direct parse
        try:
            return json.loads(cleaned), "Success (direct)"
        except Exception as e:
            return None, f"JSON parse error: {str(e)}"

    def _build_fallback_json(self, raw_text: str) -> Dict[str, Any]:
        """Graceful fallback if LLM output fails strict JSON syntax."""
        return {
            "overall_score": 50,
            "recommendation": "Borderline",
            "summary": "Model output could not be strictly parsed as JSON. View raw response for details.",
            "category_scores": {
                "technical_skills": 50,
                "experience_relevance": 50,
                "education_and_certifications": 50,
                "leadership_and_impact": 50,
            },
            "strengths": ["Analysis completed"],
            "red_flags": ["Output formatting anomaly"],
            "missing_skills": [],
            "raw_response": raw_text,
        }


def get_reviewer_model(adapter_path: Optional[Any] = None, model_name: Optional[str] = None):
    """
    Returns (model, tokenizer) ensuring singleton is loaded in 8-bit.
    Allows easy programmatic access and verification tests.
    """
    engine = InferenceEngine.get_instance(adapter_path=adapter_path, model_name=model_name)
    if not engine.is_ready:
        engine.load_model()
    return engine.model, engine.tokenizer
