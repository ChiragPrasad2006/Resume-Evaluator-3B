"""
Database Layer for Local AI Resume Reviewer.
Uses SQLite for zero-configuration, robust persistence of evaluation sessions and candidate records.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sessions.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initializes tables and indexes."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                raw_jd TEXT NOT NULL,
                compact_jd TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                candidate_name TEXT NOT NULL,
                raw_resume TEXT NOT NULL,
                overall_score INTEGER NOT NULL,
                recommendation TEXT NOT NULL,
                parsed_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_eval_session ON evaluations (session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session_created ON sessions (created_at DESC)")
    conn.close()


def create_session(
    title: str,
    raw_jd: str,
    compact_jd: str,
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO sessions (id, title, raw_jd, compact_jd, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, title, raw_jd, compact_jd, now, now),
        )
    conn.close()
    return {
        "id": session_id,
        "title": title,
        "raw_jd": raw_jd,
        "compact_jd": compact_jd,
        "created_at": now,
        "updated_at": now,
        "evaluations_count": 0,
    }


def list_sessions(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id, s.title, s.raw_jd, s.compact_jd, s.created_at, s.updated_at,
               COUNT(e.id) as evaluations_count
        FROM sessions s
        LEFT JOIN evaluations e ON s.id = e.session_id
        GROUP BY s.id
        ORDER BY s.updated_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_session(session_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    session_row = cursor.fetchone()
    if not session_row:
        conn.close()
        return None

    session_dict = dict(session_row)
    cursor.execute(
        "SELECT * FROM evaluations WHERE session_id = ? ORDER BY created_at ASC",
        (session_id,),
    )
    eval_rows = cursor.fetchall()
    conn.close()

    evaluations = []
    for r in eval_rows:
        e = dict(r)
        try:
            e["parsed_json"] = json.loads(e["parsed_json"])
        except Exception:
            pass
        evaluations.append(e)

    session_dict["evaluations"] = evaluations
    session_dict["evaluations_count"] = len(evaluations)
    return session_dict


def update_session(session_id: str, title: str, db_path: Optional[Path] = None) -> bool:
    conn = get_connection(db_path)
    now = datetime.utcnow().isoformat()
    with conn:
        cursor = conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (title, now, session_id),
        )
        updated = cursor.rowcount > 0
    conn.close()
    return updated


def delete_session(session_id: str, db_path: Optional[Path] = None) -> bool:
    conn = get_connection(db_path)
    with conn:
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def add_evaluation(
    session_id: str,
    candidate_name: str,
    raw_resume: str,
    overall_score: int,
    recommendation: str,
    parsed_json: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    eval_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    json_str = json.dumps(parsed_json, ensure_ascii=False)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO evaluations (id, session_id, candidate_name, raw_resume, overall_score, recommendation, parsed_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (eval_id, session_id, candidate_name, raw_resume, overall_score, recommendation, json_str, now),
        )
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.close()
    return {
        "id": eval_id,
        "session_id": session_id,
        "candidate_name": candidate_name,
        "raw_resume": raw_resume,
        "overall_score": overall_score,
        "recommendation": recommendation,
        "parsed_json": parsed_json,
        "created_at": now,
    }


def export_session_csv(session_id: str, db_path: Optional[Path] = None) -> str:
    session = get_session(session_id, db_path)
    if not session:
        return ""

    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Evaluation ID",
        "Candidate Name",
        "Timestamp",
        "Overall Score",
        "Recommendation",
        "Technical Skills",
        "Experience Relevance",
        "Education & Certifications",
        "Leadership & Impact",
        "Executive Summary",
        "Strengths",
        "Red Flags",
        "Missing Skills",
    ])

    for ev in session.get("evaluations", []):
        pj = ev.get("parsed_json") or {}
        cats = pj.get("category_scores") or {}
        writer.writerow([
            ev.get("id"),
            ev.get("candidate_name"),
            ev.get("created_at"),
            ev.get("overall_score"),
            ev.get("recommendation"),
            cats.get("technical_skills", "N/A"),
            cats.get("experience_relevance", "N/A"),
            cats.get("education_and_certifications", "N/A"),
            cats.get("leadership_and_impact", "N/A"),
            pj.get("summary", ""),
            "; ".join(pj.get("strengths") or []),
            "; ".join(pj.get("red_flags") or []),
            "; ".join(pj.get("missing_skills") or []),
        ])

    return output.getvalue()
