"""
FastAPI Server & REST API for Local AI Resume Reviewer.
Provides session lifecycle endpoints, document parsing, inference routing,
and export capabilities.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web import db
from web.file_parser import extract_text_from_file
from web.inference_engine import InferenceEngine, detect_candidate_name
from web.jd_processor import process_job_description

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database
    db.init_db()
    # Pre-warm singleton inference engine (load weights into GPU memory)
    print("\n[Server Startup] Pre-warming Inference Engine...")
    engine = InferenceEngine.get_instance()
    try:
        engine.load_model()
    except Exception as e:
        print(f"[Warning] Model pre-warm deferred or encountered error: {e}")
    try:
        yield
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        print("\n[Server Shutdown] Cleaning up resources...")


app = FastAPI(
    title="Local AI Resume Reviewer",
    description="Production Web GUI powered by Qwen 2.5 3B Instruct + QLoRA",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic Request Models
# ==============================================================================
class CreateSessionRequest(BaseModel):
    raw_jd: str
    title: Optional[str] = None


class UpdateSessionRequest(BaseModel):
    title: str


class EvaluateResumeRequest(BaseModel):
    resume_text: str
    candidate_name: Optional[str] = None


# ==============================================================================
# SPA HTML Serving
# ==============================================================================
@app.get("/", response_class=FileResponse)
async def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


# ==============================================================================
# Model Telemetry API (Direct from PyTorch / BitsAndBytes)
# ==============================================================================
@app.get("/api/model/status")
async def get_model_status():
    """Returns live model metadata, quantization precision, and GPU VRAM telemetry."""
    engine = InferenceEngine.get_instance()
    return engine.get_status()


# ==============================================================================
# Session Lifecycle API
# ==============================================================================
@app.get("/api/sessions")
async def get_sessions():
    """Lists all screening sessions ordered by most recent activity."""
    return db.list_sessions()


@app.post("/api/sessions")
async def create_new_session(payload: CreateSessionRequest):
    """Processes JD,Derives title and token budget, and initializes session."""
    raw_jd = payload.raw_jd.strip()
    if not raw_jd:
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    proc = process_job_description(raw_jd)
    title = payload.title or proc["title"]
    session = db.create_session(
        title=title,
        raw_jd=raw_jd,
        compact_jd=proc["compact_jd"],
    )
    session["estimated_tokens"] = proc["estimated_tokens"]
    session["is_compacted"] = proc["is_compacted"]
    return session


@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Retrieves session details and its full candidate evaluation thread."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.patch("/api/sessions/{session_id}")
async def update_session_title(session_id: str, payload: UpdateSessionRequest):
    """Updates the title of a screening session."""
    success = db.update_session(session_id, payload.title.strip())
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "updated", "id": session_id, "title": payload.title.strip()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Deletes a session and all its associated evaluation records."""
    success = db.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "deleted", "id": session_id}


# ==============================================================================
# Candidate Evaluation API (Strict Non-Drifting Inference)
# ==============================================================================
@app.post("/api/sessions/{session_id}/evaluate")
async def evaluate_candidate_text(session_id: str, payload: EvaluateResumeRequest):
    """Evaluates pasted resume text against the session's active Job Description."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    resume_text = payload.resume_text.strip()
    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text cannot be empty.")

    # Use compact JD if compacted, else raw JD
    active_jd = session["compact_jd"] or session["raw_jd"]

    # Auto-detect candidate name or fallback to index
    cand_count = session.get("evaluations_count", 0) + 1
    candidate_name = payload.candidate_name or detect_candidate_name(resume_text, fallback_idx=cand_count)

    engine = InferenceEngine.get_instance()
    try:
        eval_result = engine.evaluate_resume(active_jd, resume_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    record = db.add_evaluation(
        session_id=session_id,
        candidate_name=candidate_name,
        raw_resume=resume_text,
        overall_score=eval_result["overall_score"],
        recommendation=eval_result["recommendation"],
        parsed_json=eval_result["parsed_json"],
    )
    return record


@app.post("/api/sessions/{session_id}/evaluate/file")
async def evaluate_candidate_file(
    session_id: str,
    file: UploadFile = File(...),
    extra_text: Optional[str] = Form(None),
):
    """Evaluates an uploaded resume file (.pdf, .docx, or .txt)."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    file_bytes = await file.read()
    extracted_text, status_msg = extract_text_from_file(file_bytes, file.filename)
    if not extracted_text:
        raise HTTPException(status_code=400, detail=f"Could not extract text: {status_msg}")

    combined_resume = extracted_text
    if extra_text and extra_text.strip():
        combined_resume = extra_text.strip() + "\n\n" + extracted_text

    active_jd = session["compact_jd"] or session["raw_jd"]
    cand_count = session.get("evaluations_count", 0) + 1
    candidate_name = detect_candidate_name(combined_resume, fallback_idx=cand_count)

    engine = InferenceEngine.get_instance()
    try:
        eval_result = engine.evaluate_resume(active_jd, combined_resume)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    record = db.add_evaluation(
        session_id=session_id,
        candidate_name=candidate_name,
        raw_resume=combined_resume,
        overall_score=eval_result["overall_score"],
        recommendation=eval_result["recommendation"],
        parsed_json=eval_result["parsed_json"],
    )
    return record


# ==============================================================================
# Export Endpoints
# ==============================================================================
@app.get("/api/sessions/{session_id}/export/csv")
async def export_csv(session_id: str):
    csv_str = db.export_session_csv(session_id)
    if not csv_str:
        raise HTTPException(status_code=404, detail="Session has no evaluations or does not exist.")
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="evaluation_session_{session_id[:8]}.csv"'},
    )


@app.get("/api/sessions/{session_id}/export/json")
async def export_json(session_id: str):
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return Response(
        content=json.dumps(session, indent=2, ensure_ascii=False),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="evaluation_session_{session_id[:8]}.json"'},
    )


# ==============================================================================
# CLI Entrypoint with Auto-Browser Launch & Clean Shutdown
# ==============================================================================
if __name__ == "__main__":
    import threading
    import time
    import webbrowser
    import uvicorn

    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open("http://127.0.0.1:7860")
        except Exception:
            pass

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        uvicorn.run("web.app:app", host="127.0.0.1", port=7860, log_level="info")
    except KeyboardInterrupt:
        print("\n[Server Shutdown] Application stopped cleanly.")
