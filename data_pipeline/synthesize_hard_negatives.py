"""
Data Pipeline Step 2: Synthesize Hard Negatives & Borderlines
Generates 400-600 realistic mutated resumes with subtle pathologies
(buzzword stuffing, metric stripping, title inflation, chronological conflicts,
and missing hard requirements) paired with strict, calibrated recruiter reviews.
"""

import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add root directory to sys.path to import config
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

# Set random seed for deterministic dataset generation
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Pathology 1: Buzzword Dumps to inject
BUZZWORD_DUMPS = [
    "Expertise in: AI, ML, Deep Learning, Quantum Computing, Blockchain, Web3, Kubernetes, Microservices, Cloud Native, Big Data, LLMs, GenAI, MLOps, DevOps, CI/CD, Agile, Scrum, Kanban, ITIL, Zero Trust, Cyber Resilience, Metaverse.",
    "Core Competencies: Machine Learning, Neural Networks, Docker, Kubernetes, AWS, GCP, Azure, Snowflake, Databricks, PyTorch, TensorFlow, React, Angular, Vue, GraphQL, Kafka, Spark, Hadoop, Terraform, Rust, Go, Python, C++, Java, Solidity.",
    "Comprehensive Skills Matrix: Distributed Systems, High Availability, Disaster Recovery, Serverless, Event-Driven Architecture, Microfrontends, Data Mesh, Vector Databases, Retrieval-Augmented Generation, Fine-Tuning, Quantization, Prompt Engineering.",
]

# Pathology 2: Vague replacement verbs and phrases
VAGUE_BULLETS = [
    "- Involved in various development tasks and attended daily meetings with team members.",
    "- Handled day-to-day duties and worked on tickets assigned in the queue.",
    "- Assisted with general system maintenance and helped colleagues when needed.",
    "- Responsible for collaborating with cross-functional stakeholders on broad initiatives.",
    "- Participated in team discussions and contributed to project documentation.",
    "- Worked extensively with computers, databases, and software tools across departments.",
]

def mutate_buzzword_stuffer(resume: str) -> Tuple[str, List[str], int]:
    """Appends an unnatural buzzword dump section to the resume."""
    dump = random.choice(BUZZWORD_DUMPS)
    mutated = (
        f"KEY TECHNICAL SKILLS OVERVIEW (ALL PROFICIENT):\n{dump}\n\n"
        + resume
    )
    red_flags = [
        "Keyword stuffing detected: Superficial listing of extensive high-level technologies without supporting project context or verifiable execution."
    ]
    penalty = random.randint(18, 28)
    return mutated, red_flags, penalty


def mutate_metric_stripper(resume: str) -> Tuple[str, List[str], int]:
    """Strips quantitative metrics (percentages, dollar amounts, scale) and injects passive filler."""
    # Replace numbers, percentages, and metrics with generic words
    mutated = re.sub(r"\b\d+%\b", "some", resume)
    mutated = re.sub(r"\$\d+([kKmMbB]|\s*million|\s*thousand)?", "budgeted amount", mutated)
    mutated = re.sub(r"\b\d+\+?\s*(years|months|engineers|users|clients|projects)\b", "several items", mutated)
    
    # Inject vague filler bullets into employment section
    lines = mutated.split("\n")
    insert_idx = min(len(lines) // 2, 20)
    lines.insert(insert_idx, random.choice(VAGUE_BULLETS))
    lines.insert(insert_idx + 1, random.choice(VAGUE_BULLETS))
    mutated = "\n".join(lines)
    
    red_flags = [
        "Vague, unquantified impact: Bullet points describe generic duties rather than measurable outcomes, metrics, or technical scale."
    ]
    penalty = random.randint(15, 25)
    return mutated, red_flags, penalty


def mutate_seniority_inflation(resume: str) -> Tuple[str, List[str], int]:
    """Inflates titles to Principal/Chief/Executive while keeping dates short/junior."""
    inflated_titles = [
        "Chief Enterprise Systems Architect & Global Technology Director",
        "Principal Distinguished AI Architect & Head of Engineering",
        "Executive Vice President of Technology & Infrastructure",
    ]
    title = random.choice(inflated_titles)
    lines = resume.split("\n")
    # Replace the candidate's header or first job title
    mutated_lines = []
    replaced = False
    for line in lines:
        if not replaced and any(term in line.lower() for term in ["engineer", "developer", "lead", "specialist", "manager"]):
            mutated_lines.append(f"{title} (Experience: Jan 2023 - Present [1.5 Years Total])")
            replaced = True
        else:
            mutated_lines.append(line)
    
    if not replaced:
        mutated_lines.insert(0, f"Current Role: {title} (Jan 2023 - Present)")
    
    mutated = "\n".join(mutated_lines)
    red_flags = [
        "Severe seniority mismatch / inflated title: Claims executive/principal title with only 1-2 years of total verifiable industry tenure.",
        "Disproportionate scope of authority relative to demonstrated career progression."
    ]
    penalty = random.randint(25, 38)
    return mutated, red_flags, penalty


def mutate_chronological_conflict(resume: str) -> Tuple[str, List[str], int]:
    """Introduces overlapping dates and chronological impossibilities."""
    conflict_block = (
        "\nADDITIONAL WORK EXPERIENCE:\n"
        "- Senior Technical Lead | Apex Dynamics (Full-time: 2017 - 2024)\n"
        "- Full-Time Onsite Operations Director | Horizon Labs (Full-time: 2018 - 2023)\n"
        "- Concurrently enrolled in Full-Time Bachelor's Program (2019 - 2023)\n"
    )
    mutated = conflict_block + resume
    red_flags = [
        "Chronological contradictions: Unexplained concurrent full-time onsite roles overlapping for multiple years alongside full-time studies.",
        "Integrity concern regarding unverifiable timeline and concurrent commitments."
    ]
    penalty = random.randint(22, 35)
    return mutated, red_flags, penalty


def mutate_missing_hard_requirement(resume: str, jd: str) -> Tuple[str, List[str], int]:
    """Candidate is otherwise decent, but explicitly lacks mandatory qualification."""
    # Prepend a disclaimer or omit required credentials
    disclaimer = (
        "NOTE: Formal degree requirements not completed (High School Diploma only). "
        "No state engineering licenses or formal certifications held.\n\n"
    )
    mutated = disclaimer + resume
    red_flags = [
        "Fails mandatory educational threshold: Does not hold the required Bachelor's/Master's degree explicitly mandated in the Job Description.",
        "Lacks required statutory or professional licensing."
    ]
    penalty = random.randint(20, 30)
    return mutated, red_flags, penalty


MUTATION_FUNCTIONS = [
    ("buzzword_stuffer", mutate_buzzword_stuffer),
    ("metric_stripper", mutate_metric_stripper),
    ("seniority_inflation", mutate_seniority_inflation),
    ("chronological_conflict", mutate_chronological_conflict),
    ("missing_hard_requirement", mutate_missing_hard_requirement),
]


def generate_hard_negatives(
    target_count: int = 500,
    input_file: Path = config.CURATED_DATA_PATH,
    output_file: Path = config.HARD_NEGATIVES_PATH,
) -> List[Dict[str, Any]]:
    """Generates synthetic hard negative and borderline training pairs from curated seeds."""
    print("=" * 70)
    print(f"STEP 2: Synthesizing {target_count} Hard Negatives & Borderlines")
    print("=" * 70)

    if not input_file.exists():
        raise FileNotFoundError(f"Curated dataset not found at {input_file}. Run inspect_and_curate.py first.")

    with open(input_file, "r", encoding="utf-8") as f:
        seed_records = [json.loads(line) for line in f if line.strip()]

    # Sort seeds by overall_score descending to mutate higher-scoring resumes into deceptive hard negatives
    seed_records.sort(key=lambda x: x["overall_score"], reverse=True)
    seeds_pool = seed_records[: min(len(seed_records), 450)]

    synthetic_records: List[Dict[str, Any]] = []
    pathology_distribution: Dict[str, int] = {}
    tier_distribution = {"Borderline": 0, "Reject": 0}

    for i in range(target_count):
        seed = seeds_pool[i % len(seeds_pool)]
        mutation_name, mutation_func = MUTATION_FUNCTIONS[i % len(MUTATION_FUNCTIONS)]
        pathology_distribution[mutation_name] = pathology_distribution.get(mutation_name, 0) + 1

        raw_resume = seed["resume"]
        raw_jd = seed["job_description"]
        orig_output = seed["structured_output"]

        # Apply specific mutation
        if mutation_name == "missing_hard_requirement":
            mutated_resume, added_red_flags, penalty = mutation_func(raw_resume, raw_jd)
        else:
            mutated_resume, added_red_flags, penalty = mutation_func(raw_resume)

        # Calibrate penalized scores
        orig_score = orig_output["overall_score"]
        new_overall = max(18, min(68, orig_score - penalty))

        orig_cats = orig_output["category_scores"]
        new_tech = max(15, min(75, orig_cats["technical_skills"] - (penalty if "buzzword" in mutation_name or "metric" in mutation_name else penalty // 2)))
        new_exp = max(15, min(70, orig_cats["experience_relevance"] - (penalty if "seniority" in mutation_name or "conflict" in mutation_name else penalty // 2)))
        new_edu = max(15, min(70, orig_cats["education_and_certifications"] - (penalty * 2 if "missing" in mutation_name else 5)))
        new_lead = max(10, min(65, orig_cats["leadership_and_impact"] - (penalty if "metric" in mutation_name or "seniority" in mutation_name else penalty // 3)))

        new_category_scores = {
            "technical_skills": new_tech,
            "experience_relevance": new_exp,
            "education_and_certifications": new_edu,
            "leadership_and_impact": new_lead,
        }

        recommendation = "Borderline" if new_overall >= 50 else "Reject"
        tier_distribution[recommendation] += 1

        # Build updated strengths and weaknesses
        # Retain 1-2 residual strengths to teach nuanced discernment
        residual_strengths = orig_output.get("strengths", [])[:2]
        if not residual_strengths:
            residual_strengths = ["Familiarity with general industry nomenclature and tooling."]

        combined_weaknesses = added_red_flags + orig_output.get("weaknesses_and_red_flags", [])[:1]

        # Generate targeted feedback
        feedback = []
        if "buzzword" in mutation_name:
            feedback.append("Eliminate uncontextualized skill laundry lists; anchor each technology to a verifiable project.")
        elif "metric" in mutation_name:
            feedback.append("Rewrite bullet points using XYZ format: Accomplished [X] as measured by [Y] by doing [Z].")
        elif "seniority" in mutation_name:
            feedback.append("Align job title accurately with actual scope, responsibilities, and verifiable career tenure.")
        elif "conflict" in mutation_name:
            feedback.append("Clarify employment timeline, resolve concurrent commitments, and explain overlapping dates.")
        else:
            feedback.append("Ensure mandatory educational credentials and certifications required by the position are satisfied.")

        summary = (
            f"Candidate profile scored at {new_overall}/100 with critical reservations. "
            f"While basic domain background is present, the evaluation is heavily compromised by {added_red_flags[0].lower()} "
            f"Recommended outcome: {recommendation}."
        )

        structured_output = {
            "overall_score": new_overall,
            "summary": summary,
            "category_scores": new_category_scores,
            "strengths": residual_strengths,
            "weaknesses_and_red_flags": combined_weaknesses[:4],
            "recommendation": recommendation,
            "actionable_feedback": feedback[:3],
        }

        # Validate schema compliance
        assert config.REQUIRED_JSON_KEYS.issubset(structured_output.keys())
        assert config.REQUIRED_CATEGORY_KEYS.issubset(structured_output["category_scores"].keys())

        synth_id = f"synth_{mutation_name}_{i+1:04d}"
        synthetic_records.append({
            "id": synth_id,
            "source": f"synthetic_{mutation_name}",
            "seed_id": seed["id"],
            "pathology": mutation_name,
            "job_description": raw_jd,
            "resume": mutated_resume,
            "structured_output": structured_output,
            "target_json_str": json.dumps(structured_output, indent=2),
            "overall_score": new_overall,
            "score_tier": recommendation,
        })

    # Save to hard_negatives.jsonl
    print(f"\nWriting {len(synthetic_records)} synthetic hard negatives to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in synthetic_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "=" * 50)
    print("SYNTHETIC DATA SUMMARY STATISTICS:")
    print("=" * 50)
    print(f"Total Synthetic Records: {len(synthetic_records)}")
    print("\nPathology Breakdown:")
    for path_name, count in pathology_distribution.items():
        print(f"  - {path_name:26s}: {count:4d}")
    print("\nScore Tier Breakdown:")
    for tier, count in tier_distribution.items():
        pct = (count / len(synthetic_records)) * 100
        print(f"  - {tier:12s}: {count:4d} ({pct:5.1f}%)")
    scores = [r["overall_score"] for r in synthetic_records]
    print(f"\nScore Range: [{min(scores)} - {max(scores)}], Mean: {sum(scores)/len(scores):.1f}")
    print("=" * 50)

    return synthetic_records


if __name__ == "__main__":
    generate_hard_negatives(target_count=500)
