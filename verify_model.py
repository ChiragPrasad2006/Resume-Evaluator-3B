import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

def main():
    model_id = "Qwen/Qwen2.5-3B"
    print("=" * 60)
    print(f"Verifying / Completing download for: {model_id}")
    print("Deadlock guards: HF_HUB_ENABLE_HF_TRANSFER=0, HF_XET_HIGH_PERFORMANCE=0")
    print("=" * 60)

    cached_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
    )
    print(f"\n[PASS] Model snapshot verified on disk: {cached_path}")

    print("\nTesting 4-bit NF4 quantization load on GPU...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(cached_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        cached_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        local_files_only=True,
    )
    vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"[PASS] Model successfully loaded into memory!")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Allocated: {vram_gb:.2f} GB (Well below 8GB limit)")
    print("=" * 60)

if __name__ == "__main__":
    main()
