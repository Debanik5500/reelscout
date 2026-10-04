from __future__ import annotations

import os
import uuid
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import redis
from rq import Queue, Worker
from rq.job import Job

from reelscout.pipeline import analyze
from reelscout.config import Settings


# ─── Config ──────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))  # concurrent reels
UPLOAD_DIR = Path(tempfile.gettempdir()) / "reelscout_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

redis_conn = redis.from_url(REDIS_URL)
task_queue = Queue("reels", connection=redis_conn)


# ─── Models ──────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    vision: str = "auto"
    text_llm: str = "auto"
    resolve: str = "off"
    caption: str = ""


class JobResponse(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | started | finished | failed
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[str] = None


# ─── Background Worker ───────────────────────────────────────────────────
def process_reel_job(
    job_id: str,
    video_path: str,
    request: AnalyzeRequest,
) -> dict:
    """Run the pipeline on a single video file."""
    try:
        st = Settings.from_env()
        st.vision = request.vision
        st.text_policy = request.text_llm
        st.resolve = request.resolve

        result = analyze(
            source=video_path,
            settings=st,
            caption=request.caption,
        )
        return {"status": "finished", "result": result}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        # Cleanup video file immediately after processing
        Path(video_path).unlink(missing_ok=True)


def enqueue_job(video_path: str, request: AnalyzeRequest) -> str:
    job_id = str(uuid.uuid4())
    job = task_queue.enqueue(
        process_reel_job,
        job_id,
        video_path,
        request,
        job_id=job_id,
        job_timeout="10m",
    )
    return job.id


# ─── FastAPI App ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: warm up models (optional)
    yield
    # Shutdown
    pass


app = FastAPI(title="ReelScout API", lifespan=lifespan)


@app.post("/api/v1/analyze", response_model=JobResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    vision: str = Form("auto"),
    text_llm: str = Form("auto"),
    resolve: str = Form("off"),
    caption: str = Form(""),
):
    """Upload a video file, returns job_id for polling."""
    # Save upload to temp file
    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    request = AnalyzeRequest(
        vision=vision,
        text_llm=text_llm,
        resolve=resolve,
        caption=caption,
    )
    
    # Enqueue for background processing
    task_queue.enqueue(
        process_reel_job,
        job_id,
        str(video_path),
        request,
        job_id=job_id,
        job_timeout="10m",
    )
    
    return JobResponse(job_id=job_id)


@app.get("/api/v1/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Poll for job completion."""
    job = Job.fetch(job_id, connection=redis_conn)
    
    if job.is_queued:
        return JobStatus(job_id=job_id, status="queued", progress="waiting for worker")
    elif job.is_started:
        return JobStatus(job_id=job_id, status="started", progress="processing")
    elif job.is_finished:
        result = job.result or {}
        return JobStatus(
            job_id=job_id,
            status=result.get("status", "finished"),
            result=result.get("result"),
            error=result.get("error"),
        )
    elif job.is_failed:
        return JobStatus(job_id=job_id, status="failed", error=str(job.exc_info))
    else:
        return JobStatus(job_id=job_id, status="unknown")


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "workers": MAX_WORKERS, "queue_len": len(task_queue)}


# ─── Local Dev Runner (no Redis) ────────────────────────────────────────
def run_local_worker():
    """Run worker in same process for dev (no Redis needed)."""
    from concurrent.futures import ThreadPoolExecutor
    import queue
    import threading
    
    job_queue: queue.Queue = queue.Queue()
    results: dict = {}
    
    def worker():
        while True:
            job_id, video_path, request = job_queue.get()
            if job_id is None:
                break
            result = process_reel_job(job_id, video_path, request)
            results[job_id] = result
            job_queue.task_done()
    
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    for _ in range(MAX_WORKERS):
        executor.submit(worker)
    
    return job_queue, results, executor


# For local testing without Redis
_local_queue = None
_local_results = None
_local_executor = None


@app.post("/api/v1/analyze-local", response_model=JobResponse)
async def analyze_video_local(
    video: UploadFile = File(...),
    vision: str = Form("auto"),
    text_llm: str = Form("auto"),
    resolve: str = Form("off"),
    caption: str = Form(""),
):
    """Local dev endpoint: processes inline (blocking) or via local queue."""
    global _local_queue, _local_results, _local_executor
    
    if _local_queue is None:
        _local_queue, _local_results, _local_executor = run_local_worker()
    
    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    
    request = AnalyzeRequest(
        vision=vision,
        text_llm=text_llm,
        resolve=resolve,
        caption=caption,
    )
    
    # For local dev, process synchronously (simpler)
    result = process_reel_job(job_id, str(video_path), request)
    _local_results[job_id] = result
    
    return JobResponse(job_id=job_id)


@app.get("/api/v1/status-local/{job_id}", response_model=JobStatus)
async def get_status_local(job_id: str):
    global _local_results
    if _local_results is None or job_id not in _local_results:
        return JobStatus(job_id=job_id, status="queued")
    
    result = _local_results[job_id]
    return JobStatus(
        job_id=job_id,
        status=result.get("status", "finished"),
        result=result.get("result"),
        error=result.get("error"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)