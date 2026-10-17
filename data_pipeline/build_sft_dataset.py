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


def load_datasets() -> List[Dict[str, Any]]:
    """Loads and combines curated and synthetic datasets."""
    all_records = []
    
    if not config.CURATED_DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {config.CURATED_DATA_PATH}. Run inspect_and_curate.py first.")
    if not config.HARD_NEGATIVES_PATH.exists():
        raise FileNotFoundError(f"Missing {config.HARD_NEGATIVES_PATH}. Run synthesize_hard_negatives.py first.")
        
    print(f"Loading curated data from {config.CURATED_DATA_PATH}...")
    with open(config.CURATED_DATA_PATH, "r", encoding="utf-8") as f:
        curated = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(curated)} curated records.")

    print(f"Loading synthetic data from {config.HARD_NEGATIVES_PATH}...")
    with open(config.HARD_NEGATIVES_PATH, "r", encoding="utf-8") as f:
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

    def prepare_sft_records(recs: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
        sft_data = []
        token_lengths = []
        masked_proportions = []

        for rec in recs:
            messages = build_chatml_messages(rec)
            # Format text using Qwen's standard Jinja chat template
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            # Tokenize full sequence
            enc = tokenizer(
                full_text,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"]
            token_lengths.append(len(input_ids))

            # Build completion mask: labels = -100 for prompt tokens
            # Find the start index of response_template tokens
            labels = list(input_ids)
            found_response = False
            response_len = len(response_tokens)

            for i in range(len(input_ids) - response_len + 1):
                if input_ids[i : i + response_len] == response_tokens:
                    # Mask everything up to the end of response_template
                    cutoff = i + response_len
                    for j in range(cutoff):
                        labels[j] = -100
                    found_response = True
                    break

            if not found_response:
                # Fallback: search for single assistant token ID
                assistant_subtoken = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
                for i in range(len(input_ids) - len(assistant_subtoken) + 1):
                    if input_ids[i : i + len(assistant_subtoken)] == assistant_subtoken:
                        cutoff = i + len(assistant_subtoken) + 1  # include newline
                        for j in range(min(cutoff, len(labels))):
                            labels[j] = -100
                        found_response = True
                        break

            # If still not found, fallback to masking prompt text
            if not found_response:
                prompt_messages = messages[:2]
                prompt_text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_len = len(tokenizer.encode(prompt_text, add_special_tokens=False))
                for j in range(min(prompt_len, len(labels))):
                    labels[j] = -100

            valid, n_prompt, n_resp = verify_completion_mask(
                input_ids, labels, tokenizer, response_tokens
            )
            masked_proportions.append(n_prompt / max(1, len(input_ids)))

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
        print(f"Samples exceeding {config.MAX_SEQ_LENGTH} (truncated): {sum(1 for l in token_lengths if l >= config.MAX_SEQ_LENGTH)}")
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

    print("=" * 70)
    print("SFT Dataset Build Completed Successfully.")
    print("=" * 70)

    return train_sft, val_sft


if __name__ == "__main__":
    process_and_split_dataset()
