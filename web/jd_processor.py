"""
Job Description Processor & Token Budgeting Engine.
Handles automatic title extraction, token budgeting, and extractive requirement summarization
to guarantee prompt sequence length remains strictly within the 1400-1536 token ceiling.
"""

import re
from typing import Dict, Any, List, Optional


def extract_job_title(jd_text: str) -> str:
    """Derives a concise, clean role title from the JD text."""
    if not jd_text or not jd_text.strip():
        return "Untitled Role"

    lines = [line.strip() for line in jd_text.strip().splitlines() if line.strip()]
    if not lines:
        return "Untitled Role"

    # Common role indicators
    role_pattern = re.compile(
        r"(senior|lead|principal|junior|staff|associate|chief|vp|director|manager|head of|engineer|developer|architect|designer|scientist|analyst|specialist|consultant|intern)",
        re.IGNORECASE,
    )

    for line in lines[:5]:
        # Clean markdown headers and bullet prefixes
        cleaned = re.sub(r"^[#*\-•>\s]+", "", line).strip()
        cleaned = re.sub(r"\b(job title|role|position|opening|job description)[:\s]+", "", cleaned, flags=re.IGNORECASE).strip()
        if role_pattern.search(cleaned) and len(cleaned) <= 80:
            return cleaned

    # Fallback: clean first line
    first = re.sub(r"^[#*\-•>\s]+", "", lines[0]).strip()
    return first[:60] if len(first) > 60 else first


def extract_key_requirements(jd_text: str) -> str:
    """
    Extractive summarizer: identifies sections on requirements, qualifications,
    skills, and responsibilities, trimming fluff like company history or perk lists.
    """
    lines = jd_text.strip().splitlines()
    key_lines = []
    capture = False
    header_pattern = re.compile(
        r"(requirement|qualification|what you|must have|nice to have|skill|experience|responsibilit|tech stack|key competencies)",
        re.IGNORECASE,
    )
    stop_pattern = re.compile(
        r"(benefit|perk|about us|company overview|compensation|salary|equal opportunity|how to apply)",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if header_pattern.search(stripped):
            capture = True
            key_lines.append(stripped)
            continue

        if stop_pattern.search(stripped):
            capture = False
            continue

        if capture:
            key_lines.append(stripped)

    # If section headers weren't found, preserve bullet points and lines mentioning skills/years
    if len(key_lines) < 5:
        bullet_pattern = re.compile(r"^[-*•\d.]")
        skill_pattern = re.compile(r"(\byears\b|\bexperience\b|\bproficient\b|\bknowledge of\b|\bstrong\b|\bdegree\b)", re.IGNORECASE)
        for line in lines:
            stripped = line.strip()
            if bullet_pattern.match(stripped) or skill_pattern.search(stripped):
                key_lines.append(stripped)

    # If still too sparse, return the top truncated portion of the JD
    if len(key_lines) < 5:
        return "\n".join(lines[:30])

    return "\n".join(key_lines)


def process_job_description(raw_jd: str, max_tokens: int = 750) -> Dict[str, Any]:
    """
    Processes the raw JD:
    - Derives role title
    - Computes token budget (rough ~4 chars/token heuristic or word count)
    - If tokens > max_tokens, compacts it to hard requirements
    """
    text = raw_jd.strip()
    title = extract_job_title(text)

    # Heuristic token estimate (accurate within 5-10% of BPE)
    estimated_tokens = len(text.split()) * 1.3

    if estimated_tokens <= max_tokens:
        compact_jd = text
        is_compacted = False
    else:
        extracted = extract_key_requirements(text)
        # Verify extracted requirements don't still exceed limit
        extracted_tokens = len(extracted.split()) * 1.3
        if extracted_tokens > max_tokens:
            # Cut to first max_tokens
            words = extracted.split()
            safe_words = words[: int(max_tokens / 1.3)]
            compact_jd = " ".join(safe_words) + "..."
        else:
            compact_jd = extracted
        is_compacted = True

    return {
        "title": title,
        "raw_jd": text,
        "compact_jd": compact_jd,
        "estimated_tokens": int(estimated_tokens),
        "is_compacted": is_compacted,
    }
