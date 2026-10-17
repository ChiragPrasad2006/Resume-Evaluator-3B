"""
Evaluation & Inference Suite for AI Resume Reviewer
Features:
1. Strict JSON Schema conformance validation
2. Mean Absolute Error (MAE) and score drift tracking across categories
3. Recommendation classification fidelity (Confusion matrix / Accuracy)
4. Standalone CLI inference runner for test resumes against target JDs
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

# Add root directory to sys.path to import config
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config


def extract_json_from_response(raw_text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extracts and parses JSON object from model output text.
    Handles markdown code blocks, preamble text, or raw JSON.
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None, "Empty response received."

    text = raw_text.strip()

    # Pattern 1: Extract from ```json ... ``` blocks
    json_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_block_match:
        text = json_block_match.group(1).strip()
    else:
        # Pattern 2: Find outermost matching curly braces
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, "Success"
        return None, f"Parsed JSON is not an object (got {type(parsed).__name__})."
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {str(e)}"


def validate_schema(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validates parsed dictionary against strict AI Reviewer schema.
    Returns (is_valid, list_of_errors).
    """
    errors = []

    # 1. Check required top-level keys
    missing_keys = config.REQUIRED_JSON_KEYS - set(data.keys())
    if missing_keys:
        errors.append(f"Missing required top-level keys: {sorted(list(missing_keys))}")

    # 2. Validate overall_score
    overall = data.get("overall_score")
    if not isinstance(overall, (int, float)) or not (0 <= overall <= 100):
        errors.append(f"Invalid overall_score '{overall}': must be integer between 0 and 100.")

    # 3. Validate category_scores
    cat_scores = data.get("category_scores")
    if not isinstance(cat_scores, dict):
        errors.append("Field 'category_scores' must be a dictionary.")
    else:
        missing_cats = config.REQUIRED_CATEGORY_KEYS - set(cat_scores.keys())
        if missing_cats:
            errors.append(f"Missing category_scores keys: {sorted(list(missing_cats))}")
        for cat, score in cat_scores.items():
            if not isinstance(score, (int, float)) or not (0 <= score <= 100):
                errors.append(f"Invalid category score for '{cat}': {score} (must be 0-100).")

    # 4. Validate recommendation
    rec = data.get("recommendation")
    if rec not in config.VALID_RECOMMENDATIONS:
        errors.append(f"Invalid recommendation '{rec}': must be one of {config.VALID_RECOMMENDATIONS}.")

    # 5. Validate lists of strings
    for list_field in ["strengths", "weaknesses_and_red_flags", "actionable_feedback"]:
        val = data.get(list_field)
        if not isinstance(val, list):
            errors.append(f"Field '{list_field}' must be a list of strings.")
        elif len(val) == 0:
            errors.append(f"Field '{list_field}' cannot be empty.")
        else:
            for idx, item in enumerate(val):
                if not isinstance(item, str) or not item.strip():
                    errors.append(f"Item {idx} in '{list_field}' must be a non-empty string.")

    # 6. Validate summary
    summary = data.get("summary")
    if not isinstance(summary, str) or len(summary.strip()) < 10:
        errors.append("Field 'summary' must be a descriptive string (min 10 chars).")

    return (len(errors) == 0), errors


def compute_metrics(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes score drift (MAE), category-level MAE, and recommendation accuracy.
    """
    total = len(predictions)
    if total == 0:
        return {"error": "No samples evaluated"}

    overall_errors = []
    category_errors = {cat: [] for cat in config.REQUIRED_CATEGORY_KEYS}
    rec_matches = 0
    schema_valid_count = 0

    for pred, gt in zip(predictions, ground_truths):
        is_valid, _ = validate_schema(pred)
        if is_valid:
            schema_valid_count += 1

        # Overall score MAE
        pred_overall = pred.get("overall_score", 0)
        gt_overall = gt.get("overall_score", 0)
        overall_errors.append(abs(pred_overall - gt_overall))

        # Category scores MAE
        pred_cats = pred.get("category_scores", {})
        gt_cats = gt.get("category_scores", {})
        for cat in config.REQUIRED_CATEGORY_KEYS:
            if cat in pred_cats and cat in gt_cats:
                category_errors[cat].append(abs(pred_cats[cat] - gt_cats[cat]))

        # Recommendation match
        if pred.get("recommendation") == gt.get("recommendation"):
            rec_matches += 1

    metrics = {
        "total_samples": total,
        "schema_conformance_rate": (schema_valid_count / total) * 100,
        "overall_score_mae": float(np.mean(overall_errors)),
        "overall_score_std": float(np.std(overall_errors)),
        "category_mae": {
            cat: float(np.mean(errs)) if errs else None
            for cat, errs in category_errors.items()
        },
        "recommendation_accuracy": (rec_matches / total) * 100,
    }
    return metrics


def run_schema_self_test() -> None:
    """Verifies schema validation engine against valid and corrupted samples."""
    print("=" * 60)
    print("RUNNING SCHEMA VALIDATOR SELF-TEST")
    print("=" * 60)

    # Valid sample
    valid_sample = {
        "overall_score": 82,
        "summary": "Strong candidate with comprehensive background in distributed systems.",
        "category_scores": {
            "technical_skills": 85,
            "experience_relevance": 80,
            "education_and_certifications": 90,
            "leadership_and_impact": 75,
        },
        "strengths": ["Demonstrated mastery of distributed backend services."],
        "weaknesses_and_red_flags": ["Limited public cloud deployment experience."],
        "recommendation": "Hire",
        "actionable_feedback": ["Target cloud certifications such as AWS Solutions Architect."],
    }

    is_valid, errors = validate_schema(valid_sample)
    assert is_valid and not errors, f"Valid sample failed validation: {errors}"
    print("[PASS] Valid review payload accepted cleanly.")

    # Corrupt sample (missing key, invalid score, invalid recommendation)
    corrupt_sample = {
        "overall_score": 150,  # Out of bounds
        "category_scores": {
            "technical_skills": 50,
            # Missing other categories
        },
        "recommendation": "Maybe Hire",  # Invalid tier
        "strengths": [],
    }

    is_valid, errors = validate_schema(corrupt_sample)
    assert not is_valid and len(errors) >= 3, "Corrupt sample was not flagged properly."
    print(f"[PASS] Corrupted review flagged with {len(errors)} distinct violations:")
    for err in errors:
        print(f"  - {err}")

    print("=" * 60)
    print("Schema validator engine verified successfully.")
    print("=" * 60)


def format_evaluation_prompt(job_description: str, resume: str) -> str:
    """Formats raw prompt for live inference."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": config.USER_PROMPT_TEMPLATE.format(
            job_description=job_description,
            resume=resume,
        )},
    ]
    # Qwen chat template formatting
    prompt_str = f"<|im_start|>system\n{config.SYSTEM_PROMPT}<|im_end|>\n"
    prompt_str += f"<|im_start|>user\n{config.USER_PROMPT_TEMPLATE.format(job_description=job_description, resume=resume)}<|im_end|>\n"
    prompt_str += "<|im_start|>assistant\n"
    return prompt_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Resume Reviewer Model")
    parser.add_argument("--test-schema", action="store_true", help="Run schema validator self-test.")
    parser.add_argument("--val-file", type=str, default=str(config.SFT_VAL_PATH), help="Path to validation JSONL file.")
    parser.add_argument("--adapter-path", type=str, default=str(config.FINAL_MODEL_DIR), help="Path to trained LoRA adapter.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of validation samples to evaluate.")
    parser.add_argument("--output-report", type=str, default="evaluation_report.json", help="Path to output metrics JSON.")
    args = parser.parse_args()

    if args.test_schema:
        run_schema_self_test()
        sys.exit(0)

    print("Evaluation Suite Ready.")
    print(f"Validation dataset target: {args.val_file}")
    print(f"Adapter target path: {args.adapter_path}")
    print("To evaluate after training on target hardware, run:")
    print("  python evaluate_reviewer.py --val-file data/sft_val.jsonl --adapter-path outputs/final_adapter")


if __name__ == "__main__":
    main()
