"""
Data Pipeline Step 1: Inspect & Curate netsol/resume-score-details
Downloads raw JSON records, cleans text artifacts, extracts ground-truth scores,
maps them to the standardized Reviewer JSON schema, and outputs curated_resumes.jsonl.
"""

import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from huggingface_hub import snapshot_download
from tqdm import tqdm

# Add root directory to sys.path to import config
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config


def clean_text(text: str) -> str:
    """Cleans unicode artifacts, non-standard bullets, and redundant whitespace."""
    if not isinstance(text, str):
        return ""
    
    # Replace non-standard bullets and dashes
    replacements = {
        "\uf0b7": "- ",
        "\u2022": "- ",
        "\u2023": "- ",
        "\u25e6": "- ",
        "\u2043": "- ",
        "\u2219": "- ",
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\xa0": " ",
        "\r\n": "\n",
        "\r": "\n",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove control characters except standard whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    # Collapse 3 or more newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple inline spaces/tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def safe_parse_json_or_literal(value: Any) -> Any:
    """Parses value if it is a JSON or Python-literal formatted string."""
    if not isinstance(value, str):
        return value
    val = value.strip()
    if not val:
        return None
    try:
        return json.loads(val)
    except Exception:
        try:
            return ast.literal_eval(val)
        except Exception:
            return val


def extract_category_scores(raw_scores: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
    """
    Extracts normalized category scores (0-100) and overall score from raw netsol score dict.
    Netsol scores typically range from 1 to 10.
    """
    macro_scores = raw_scores.get("macro_scores", [])
    micro_scores = raw_scores.get("micro_scores", [])
    reqs = raw_scores.get("requirements", [])
    
    # Collect all numerical scores
    all_scores = []
    
    tech_scores = []
    exp_scores = []
    lead_scores = []
    
    for item in macro_scores:
        if isinstance(item, dict):
            crit = item.get("criteria", "").lower()
            score = float(item.get("score", 0))
            all_scores.append(score)
            if "lead" in crit or "manage" in crit:
                lead_scores.append(score)
            elif "tech" in crit or "expert" in crit or "skill" in crit:
                tech_scores.append(score)
            else:
                exp_scores.append(score)
                
    for item in micro_scores:
        if isinstance(item, dict):
            crit = item.get("criteria", "").lower()
            score = float(item.get("score", 0))
            all_scores.append(score)
            if "lead" in crit:
                lead_scores.append(score)
            elif "project" in crit or "manage" in crit or "design" in crit or "install" in crit:
                exp_scores.append(score)
            else:
                tech_scores.append(score)
                
    # Requirements pass percentage
    req_pass_rate = 1.0
    if reqs and isinstance(reqs, list):
        met_count = sum(1 for r in reqs if isinstance(r, dict) and r.get("meets", False))
        req_pass_rate = met_count / max(len(reqs), 1)

    # Normalize helper (scores are 1-10; convert to 10-100 or 0-100)
    def norm(score_list: List[float], default: float = 7.0) -> int:
        if not score_list:
            val = default
        else:
            val = sum(score_list) / len(score_list)
        # If score is on 1-10 scale, scale to 0-100
        if val <= 10.0:
            scaled = int(round(val * 10))
        else:
            scaled = int(round(val))
        return max(0, min(100, scaled))

    tech_score = norm(tech_scores, default=7.5)
    exp_score = norm(exp_scores, default=7.0)
    lead_score = norm(lead_scores, default=7.0)
    edu_score = int(round(req_pass_rate * 85 + 10))  # 10 to 95 based on requirements met
    edu_score = max(20, min(95, edu_score))

    # Overall score calculation
    if all_scores:
        avg_raw = sum(all_scores) / len(all_scores)
        if avg_raw <= 10.0:
            overall = int(round(avg_raw * 10))
        else:
            overall = int(round(avg_raw))
    else:
        overall = int(round(0.35 * tech_score + 0.35 * exp_score + 0.15 * edu_score + 0.15 * lead_score))

    overall = max(10, min(98, overall))
    
    category_scores = {
        "technical_skills": tech_score,
        "experience_relevance": exp_score,
        "education_and_certifications": edu_score,
        "leadership_and_impact": lead_score,
    }
    return overall, category_scores


def determine_recommendation(overall_score: int) -> str:
    """Maps overall score to discrete recommendation tier."""
    if overall_score >= 85:
        return "Strong Hire"
    elif overall_score >= 70:
        return "Hire"
    elif overall_score >= 50:
        return "Borderline"
    else:
        return "Reject"


def extract_strengths_and_weaknesses(
    justifications: Any,
    requirements: Any,
    overall_score: int
) -> Tuple[List[str], List[str], List[str]]:
    """Synthesizes structured strengths, weaknesses, and actionable feedback."""
    strengths = []
    weaknesses = []
    feedback = []

    # Process justifications
    if isinstance(justifications, list):
        for j in justifications:
            if not isinstance(j, str):
                continue
            cleaned = clean_text(j)
            if not cleaned:
                continue
            lower = cleaned.lower()
            if any(neg in lower for neg in ["lack", "does not meet", "no evidence", "limited", "missing", "fails"]):
                weaknesses.append(cleaned)
            else:
                strengths.append(cleaned)
    elif isinstance(justifications, str) and justifications.strip():
        strengths.append(clean_text(justifications))

    # Process requirements
    if isinstance(requirements, list):
        for r in requirements:
            if isinstance(r, dict):
                crit = clean_text(r.get("criteria", ""))
                meets = r.get("meets", True)
                if not meets and crit:
                    weaknesses.append(f"Does not meet requirement: {crit}")
                    feedback.append(f"Acquire demonstrated experience or certification in '{crit}'.")

    # Ensure at least 1 strength and 1 weakness for balanced training
    if not strengths:
        strengths.append("Foundational background aligns with general role domain.")
    if not weaknesses:
        if overall_score >= 85:
            weaknesses.append("Minor opportunities to provide deeper quantifiable impact metrics for recent leadership projects.")
        else:
            weaknesses.append("Lacks explicit, detailed depth in advanced requirements specified in the job description.")

    if not feedback:
        if overall_score >= 75:
            feedback.append("Focus interview on high-impact system architecture challenges and cross-team leadership outcomes.")
        elif overall_score >= 50:
            feedback.append("Provide quantified business outcomes (e.g. latency, revenue, cost reduction) to bolster experience claims.")
        else:
            feedback.append("Align core technical skills and highlight domain-specific hands-on projects directly matching target JD.")

    return strengths[:4], weaknesses[:4], feedback[:3]


def process_dataset(download_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Downloads netsol/resume-score-details repository and curates all valid samples."""
    print("=" * 70)
    print("STEP 1: Downloading & Curating 'netsol/resume-score-details'")
    print("=" * 70)

    raw_cache_dir = config.DATA_DIR / "raw_netsol"
    raw_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Syncing dataset snapshot to: {raw_cache_dir}")
    local_repo_path = snapshot_download(
        repo_id="netsol/resume-score-details",
        repo_type="dataset",
        local_dir=str(raw_cache_dir),
        allow_patterns=["*.json"],
        resume_download=True,
    )

    json_files = sorted(list(Path(local_repo_path).glob("*.json")))
    print(f"Found {len(json_files)} raw JSON files.")

    curated_records: List[Dict[str, Any]] = []
    stats = {
        "total_files": len(json_files),
        "valid": 0,
        "filtered_invalid_flag": 0,
        "filtered_short_resume": 0,
        "filtered_short_jd": 0,
        "filtered_corrupt_json": 0,
        "filtered_missing_scores": 0,
    }

    tier_counts = {"Strong Hire": 0, "Hire": 0, "Borderline": 0, "Reject": 0}

    for json_file in tqdm(json_files, desc="Curating resumes"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            stats["filtered_corrupt_json"] += 1
            continue

        input_data = data.get("input", {})
        output_data = data.get("output", {})

        if not isinstance(input_data, dict) or not isinstance(output_data, dict):
            stats["filtered_corrupt_json"] += 1
            continue

        valid_flag = output_data.get("valid_resume_and_jd", True)
        if str(valid_flag).lower() in ["false", "0"]:
            stats["filtered_invalid_flag"] += 1
            continue

        raw_resume = input_data.get("resume", "")
        raw_jd = input_data.get("job_description", "")

        resume_clean = clean_text(raw_resume)
        jd_clean = clean_text(raw_jd)

        if len(resume_clean) < 150:
            stats["filtered_short_resume"] += 1
            continue
        if len(jd_clean) < 80:
            stats["filtered_short_jd"] += 1
            continue

        raw_scores_val = safe_parse_json_or_literal(output_data.get("scores", {}))
        if not isinstance(raw_scores_val, dict):
            stats["filtered_missing_scores"] += 1
            continue

        # Extract scores
        overall_score, category_scores = extract_category_scores(raw_scores_val)
        recommendation = determine_recommendation(overall_score)
        tier_counts[recommendation] += 1

        # Extract justifications
        justifications = safe_parse_json_or_literal(output_data.get("justification", []))
        reqs = safe_parse_json_or_literal(raw_scores_val.get("requirements", []))
        strengths, weaknesses, feedback = extract_strengths_and_weaknesses(
            justifications, reqs, overall_score
        )

        # Generate summary
        candidate_name = ""
        details = data.get("details", {})
        if isinstance(details, dict):
            candidate_name = clean_text(details.get("name", ""))
        
        name_str = f"Candidate ({candidate_name})" if candidate_name else "Candidate"
        summary = (
            f"{name_str} demonstrates an overall fit score of {overall_score}/100 for the role. "
            f"Technical proficiency scored at {category_scores['technical_skills']}/100, while experience "
            f"relevance scored at {category_scores['experience_relevance']}/100. "
            f"Recommended outcome: {recommendation}."
        )

        structured_output = {
            "overall_score": overall_score,
            "summary": summary,
            "category_scores": category_scores,
            "strengths": strengths,
            "weaknesses_and_red_flags": weaknesses,
            "recommendation": recommendation,
            "actionable_feedback": feedback,
        }

        # Validate against config required keys
        assert config.REQUIRED_JSON_KEYS.issubset(structured_output.keys())
        assert config.REQUIRED_CATEGORY_KEYS.issubset(structured_output["category_scores"].keys())

        record_id = f"curated_{len(curated_records)+1:04d}"
        curated_records.append({
            "id": record_id,
            "source": "netsol/resume-score-details",
            "source_file": json_file.name,
            "job_description": jd_clean,
            "resume": resume_clean,
            "structured_output": structured_output,
            "target_json_str": json.dumps(structured_output, indent=2),
            "overall_score": overall_score,
            "score_tier": recommendation,
        })
        stats["valid"] += 1

    # Save to curated_resumes.jsonl
    print(f"\nWriting {len(curated_records)} curated records to: {config.CURATED_DATA_PATH}")
    with open(config.CURATED_DATA_PATH, "w", encoding="utf-8") as f:
        for rec in curated_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Display dataset statistics
    print("\n" + "=" * 50)
    print("CURATION SUMMARY STATISTICS:")
    print("=" * 50)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("\nScore Tier Breakdown:")
    for tier, count in tier_counts.items():
        pct = (count / max(1, stats['valid'])) * 100
        print(f"  - {tier:12s}: {count:4d} ({pct:5.1f}%)")
    
    scores = [r["overall_score"] for r in curated_records]
    if scores:
        print(f"\nScore Range: [{min(scores)} - {max(scores)}], Mean: {sum(scores)/len(scores):.1f}")
    print("=" * 50)

    return curated_records


if __name__ == "__main__":
    process_dataset()
