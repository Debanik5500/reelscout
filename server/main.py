from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Depends, Request, Header
from fastapi.responses import JSONResponse, PlainTextResponse, FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import redis
from rq import Queue, Worker
from rq.job import Job
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from reelscout.pipeline import analyze, analyze_multi_post, analyze_reel_with_storage
from reelscout.config import Settings
from reelscout.storage import get_phone_storage, ReelResult
from reelscout.ingest import download_instagram
from server import quota
from server.jobs import manager as job_manager
from server.keymoments import is_supported_link, prepare_live, prepare_reel


# ─── Config ──────────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))
UPLOAD_DIR = Path(tempfile.gettempdir()) / "reelscout_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
API_KEY = os.getenv("API_KEY", "")  # Set to enable auth

# Optional Redis - only needed for job queues
try:
    redis_conn = redis.from_url(REDIS_URL)
    task_queue = Queue("reels", connection=redis_conn)
    multipost_queue = Queue("multipost", connection=redis_conn)
    _redis_available = True
except Exception as e:
    print(f"[warn] Redis unavailable: {e}. Job queue endpoints disabled.")
    redis_conn = None
    task_queue = None
    multipost_queue = None
    _redis_available = False


# ─── Prometheus Metrics ──────────────────────────────────────────────────
REQUEST_COUNT = Counter("reelscout_requests_total", "Total requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("reelscout_request_duration_seconds", "Request latency", ["method", "endpoint"])
QUEUE_DEPTH = Gauge("reelscout_queue_depth", "Queue depth", ["queue"])
ACTIVE_JOBS = Gauge("reelscout_active_jobs", "Active jobs")
STORAGE_SIZE = Gauge("reelscout_storage_bytes", "Storage size in bytes")
STORAGE_FILES = Gauge("reelscout_storage_files", "Number of stored files")


# ─── Auth ────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> bool:
    if not API_KEY:
        return True  # No auth configured
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True


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
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None
    progress: Optional[str] = None


class MultiPostRequest(BaseModel):
    """Request for multi-post carousel analysis.

    For Instagram carousel posts where user sends multiple images.
    index=0 means single post, index>=1 means carousel position (1-based).
    If index>=1, all images in the request are treated as carousel slides.
    """
    images: list[str]
    index: int = 0
    job_id: Optional[str] = None
    vision: str = "auto"
    text_llm: str = "auto"
    resolve: str = "off"
    caption: str = ""


class MultiPostResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class ReelStorageRequest(BaseModel):
    video: str
    job_id: Optional[str] = None
    vision: str = "auto"
    text_llm: str = "auto"
    resolve: str = "off"
    caption: str = ""
    max_frames: int = 20
    fixed_hz: Optional[float] = None
    max_side: int = 1280


class ReelStorageResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class InstagramURLRequest(BaseModel):
    """Request to analyze Instagram URL directly."""
    url: str
    job_id: Optional[str] = None
    vision: str = "auto"
    text_llm: str = "auto"
    resolve: str = "off"
    max_frames: int = 20


class InstagramURLResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ─── Background Workers ──────────────────────────────────────────────────
def process_reel_job(
    job_id: str,
    video_path: str,
    request: AnalyzeRequest,
) -> dict:
    ACTIVE_JOBS.inc()
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
        Path(video_path).unlink(missing_ok=True)
        ACTIVE_JOBS.dec()


def process_multi_post_job(
    job_id: str,
    images: list[bytes],
    request: MultiPostRequest,
) -> dict:
    ACTIVE_JOBS.inc()
    try:
        result = analyze_multi_post(
            images=images,
            job_id=job_id,
        )
        return {"status": "finished", "result": result.to_dict()}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        ACTIVE_JOBS.dec()


def process_instagram_url_job(
    job_id: str,
    url: str,
    request: InstagramURLRequest,
) -> dict:
    ACTIVE_JOBS.inc()
    try:
        # Download Instagram content
        downloaded = download_instagram(url)
        if not downloaded:
            return {"status": "failed", "error": "Failed to download Instagram content"}

        # Determine if reel or carousel
        if downloaded.get("is_reel"):
            video_path = downloaded["video_path"]
            st = Settings.from_env()
            st.vision = request.vision
            st.text_policy = request.text_llm
            st.resolve = request.resolve
            result = analyze_reel_with_storage(
                video_path=Path(video_path),
                job_id=job_id,
                max_frames=request.max_frames,
            )
        else:
            images = downloaded["images"]
            result = analyze_multi_post(
                images=images,
                job_id=job_id,
            )
        return {"status": "finished", "result": result.to_dict()}
    except Exception as e:
        return {"status": "failed", "error": str(e)}
    finally:
        ACTIVE_JOBS.dec()


def enqueue_job(video_path: str, request: AnalyzeRequest) -> str:
    job_id = str(uuid.uuid4())
    task_queue.enqueue(
        process_reel_job,
        job_id,
        video_path,
        request,
        job_id=job_id,
        job_timeout="10m",
    )
    QUEUE_DEPTH.labels(queue="reels").set(len(task_queue))
    return job_id


def enqueue_multi_post_job(images: list[bytes], request: MultiPostRequest) -> str:
    job_id = request.job_id or str(uuid.uuid4())
    multipost_queue.enqueue(
        process_multi_post_job,
        job_id,
        images,
        request,
        job_id=job_id,
        job_timeout="5m",
    )
    QUEUE_DEPTH.labels(queue="multipost").set(len(multipost_queue))
    return job_id


def enqueue_instagram_url_job(url: str, request: InstagramURLRequest) -> str:
    job_id = request.job_id or str(uuid.uuid4())
    task_queue.enqueue(
        process_instagram_url_job,
        job_id,
        url,
        request,
        job_id=job_id,
        job_timeout="15m",
    )
    QUEUE_DEPTH.labels(queue="reels").set(len(task_queue))
    return job_id


# ─── FastAPI App ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    if _redis_available and task_queue is not None:
        try:
            QUEUE_DEPTH.labels(queue="reels").set(len(task_queue))
            QUEUE_DEPTH.labels(queue="multipost").set(len(multipost_queue))
        except Exception:
            pass
    yield


app = FastAPI(title="ReelScout API", lifespan=lifespan)


# ─── Metrics Middleware ──────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    return response


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return generate_latest()


# ─── Health ──────────────────────────────────────────────────────────────
@app.get("/api/v1/health")
async def health():
    storage = get_phone_storage()
    stats = storage.get_storage_stats()
    STORAGE_SIZE.set(stats["total_size_bytes"])
    STORAGE_FILES.set(stats["total_frames"] + stats["total_images"])
    queue_len = 0
    multipost_queue_len = 0
    if _redis_available and task_queue is not None:
        try:
            queue_len = len(task_queue)
            multipost_queue_len = len(multipost_queue)
        except Exception:
            pass
    return {
        "status": "ok",
        "workers": MAX_WORKERS,
        "redis": _redis_available,
        "queue_len": queue_len,
        "multipost_queue_len": multipost_queue_len,
        "storage": stats,
    }


# ─── Reel Analysis (Video Upload) ────────────────────────────────────────
@app.post("/api/v1/analyze", response_model=JobResponse, dependencies=[Depends(verify_api_key)])
async def analyze_video(
    video: UploadFile = File(...),
    vision: str = Form("auto"),
    text_llm: str = Form("auto"),
    resolve: str = Form("off"),
    caption: str = Form(""),
):
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

    enqueue_job(str(video_path), request)
    return JobResponse(job_id=job_id)


@app.get("/api/v1/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
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


# ─── Multi-Post Analysis (Carousel) ──────────────────────────────────────
@app.post("/api/v1/analyze-multi-post", response_model=MultiPostResponse, dependencies=[Depends(verify_api_key)])
async def analyze_multi_post_endpoint(request: MultiPostRequest):
    """Analyze a multi-post carousel (series of images) directly.

    index=0: single post, process as single image
    index>=1: carousel, process all images as carousel slides
    """
    # Decode base64 images
    image_bytes_list = []
    for img_b64 in request.images:
        try:
            if img_b64.startswith("data:image"):
                img_b64 = img_b64.split(",")[1]
            image_bytes_list.append(base64.b64decode(img_b64))
        except Exception as e:
            return MultiPostResponse(
                job_id=request.job_id or "",
                status="error",
                error=f"Invalid image data: {e}"
            )

    if not image_bytes_list:
        return MultiPostResponse(
            job_id=request.job_id or "",
            status="error",
            error="No valid images provided"
        )

    # If index >= 1, treat as carousel (all images are slides)
    # If index == 0, treat as single post (only first image)
    if request.index == 0 and len(image_bytes_list) > 1:
        image_bytes_list = [image_bytes_list[0]]

    # Enqueue for background processing
    job_id = enqueue_multi_post_job(image_bytes_list, request)
    return MultiPostResponse(job_id=job_id, status="queued")


@app.get("/api/v1/multipost-status/{job_id}", response_model=MultiPostResponse)
async def get_multipost_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)

    if job.is_queued:
        return MultiPostResponse(job_id=job_id, status="queued")
    elif job.is_started:
        return MultiPostResponse(job_id=job_id, status="started")
    elif job.is_finished:
        result = job.result or {}
        return MultiPostResponse(
            job_id=job_id,
            status=result.get("status", "finished"),
            result=result.get("result"),
            error=result.get("error"),
        )
    elif job.is_failed:
        return MultiPostResponse(job_id=job_id, status="failed", error=str(job.exc_info))
    else:
        return MultiPostResponse(job_id=job_id, status="unknown")


# ─── Reel with Phone Storage ─────────────────────────────────────────────
@app.post("/api/v1/analyze-reel-storage", response_model=ReelStorageResponse, dependencies=[Depends(verify_api_key)])
async def analyze_reel_storage_endpoint(
    video: UploadFile = File(...),
    job_id: Optional[str] = Form(None),
    vision: str = Form("auto"),
    text_llm: str = Form("auto"),
    resolve: str = Form("off"),
    caption: str = Form(""),
    max_frames: int = Form(20),
    fixed_hz: Optional[float] = Form(None),
    max_side: int = Form(1280),
):
    if job_id is None:
        job_id = uuid.uuid4().hex[:12]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        shutil.copyfileobj(video.file, tmp)
        video_path = Path(tmp.name)

    try:
        result = analyze_reel_with_storage(
            video_path=video_path,
            job_id=job_id,
            max_frames=max_frames,
            fixed_hz=fixed_hz,
            max_side=max_side,
        )
        return ReelStorageResponse(
            job_id=result.job_id,
            status=result.status,
            result=result.to_dict(),
        )
    except Exception as e:
        return ReelStorageResponse(job_id=job_id, status="error", error=str(e))
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except Exception:
            pass


# ─── Instagram URL Analysis ──────────────────────────────────────────────
@app.post("/api/v1/analyze-instagram", response_model=InstagramURLResponse, dependencies=[Depends(verify_api_key)])
async def analyze_instagram_url(request: InstagramURLRequest):
    """Analyze Instagram URL directly (reel or carousel).
    
    Downloads content from Instagram, detects type (reel vs carousel),
    processes accordingly with frame/image storage on phone.
    """
    job_id = enqueue_instagram_url_job(request.url, request)
    return InstagramURLResponse(job_id=job_id, status="queued")


@app.get("/api/v1/instagram-status/{job_id}", response_model=InstagramURLResponse)
async def get_instagram_status(job_id: str):
    job = Job.fetch(job_id, connection=redis_conn)

    if job.is_queued:
        return InstagramURLResponse(job_id=job_id, status="queued")
    elif job.is_started:
        return InstagramURLResponse(job_id=job_id, status="started")
    elif job.is_finished:
        result = job.result or {}
        return InstagramURLResponse(
            job_id=job_id,
            status=result.get("status", "finished"),
            result=result.get("result"),
            error=result.get("error"),
        )
    elif job.is_failed:
        return InstagramURLResponse(job_id=job_id, status="failed", error=str(job.exc_info))
    else:
        return InstagramURLResponse(job_id=job_id, status="unknown")


# ─── Storage Endpoints ───────────────────────────────────────────────────
@app.get("/api/v1/result/{job_id}")
async def get_result_from_storage(job_id: str):
    storage = get_phone_storage()
    result = storage.load_result(job_id)

    if result is None:
        return {"error": "Result not found in local storage", "job_id": job_id}

    return result.to_dict()


@app.get("/api/v1/storage/stats")
async def get_storage_stats():
    storage = get_phone_storage()
    stats = storage.get_storage_stats()
    STORAGE_SIZE.set(stats["total_size_bytes"])
    STORAGE_FILES.set(stats["total_frames"] + stats["total_images"])
    return stats


class SubmitReelRequest(BaseModel):
    url: str


@app.get("/api/v1/plans")
async def get_plans():
    """Plans for the subscribe page, straight from plans.json."""
    return quota.load_plans()


@app.get("/api/v1/me")
async def get_me(x_device_id: Optional[str] = Header(default=None)):
    """This device's plan and how many reels it has used this month."""
    return quota.status(x_device_id)


@app.post("/api/v1/reels")
async def submit_reel(request: SubmitReelRequest, x_device_id: Optional[str] = Header(default=None)):
    """Start processing a reel in the background. Progress arrives on /api/v1/reels/stream."""
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="url must be http(s)")
    if not is_supported_link(url):
        raise HTTPException(status_code=422, detail="Not a reel or post link from a supported site")
    try:
        job = job_manager.submit(url, x_device_id)
    except quota.QuotaExceeded as exc:
        # 402: the app opens the plans page. Body carries the numbers it needs.
        raise HTTPException(status_code=402, detail={"error": "quota_exceeded", **exc.status})
    storage = get_phone_storage()
    if job.get("existing"):
        return {"id": job["id"], "existing": True, "reel": prepare_reel(job["id"], storage),
                "quota": quota.status(x_device_id)}
    return {"id": job["id"], "existing": False, "reel": prepare_live(job, storage), "quota": quota.status(x_device_id)}


@app.get("/api/v1/reels")
async def list_reels():
    """Prepared reel payloads for the iOS client: live/failed jobs first, then finished reels."""
    storage = get_phone_storage()
    active = job_manager.active()
    active_ids = {j["id"] for j in active}
    reels = [prepare_live(j, storage) for j in active]
    jobs = sorted(storage.metadata.items(), key=lambda kv: kv[1].get("created") or "", reverse=True)
    for job_id, _meta in jobs:
        if job_id in active_ids:
            continue
        try:
            reel = prepare_reel(job_id, storage)
        except Exception as exc:
            print(f"[reels] skipping {job_id}: {exc}")
            continue
        if reel:
            reels.append(reel)
    return {"reels": reels}


@app.delete("/api/v1/reels/{job_id}")
async def dismiss_reel(job_id: str):
    """Forget a failed job so its error card goes away. Finished and running reels are left alone."""
    if not job_manager.dismiss(job_id):
        raise HTTPException(status_code=409, detail="Only failed jobs can be dismissed")
    return {"status": "dismissed", "id": job_id}


@app.get("/api/v1/reels/stream")
async def stream_reels(request: Request):
    """Server-sent events: a snapshot of active jobs on connect, then one event per progress change."""
    storage = get_phone_storage()
    queue = job_manager.subscribe(asyncio.get_running_loop())

    def frame(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    async def events():
        try:
            yield frame("snapshot", {"reels": [prepare_live(j, storage) for j in job_manager.active()]})
            while not await request.is_disconnected():
                try:
                    job = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if job["status"] == "removed":
                    yield frame("removed", {"id": job["id"], "canonical": job["canonical"]})
                elif job["status"] == "ready":
                    reel = await asyncio.to_thread(prepare_reel, job["id"], storage)
                    yield frame("ready", {"id": job["id"], "reel": reel})
                else:
                    yield frame("progress", {"id": job["id"], "reel": prepare_live(job, storage)})
        finally:
            job_manager.unsubscribe(queue)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/v1/reels/{job_id}")
async def get_reel(job_id: str):
    """Prepared payload for a single reel."""
    storage = get_phone_storage()
    job = job_manager.get(job_id)
    if job and job["status"] != "ready":
        return {"reel": prepare_live(job, storage)}
    try:
        reel = prepare_reel(job_id, storage)
    except Exception as exc:
        print(f"[reels] failed {job_id}: {exc}")
        reel = None
    if reel is None:
        raise HTTPException(status_code=404, detail="Reel not found")
    return {"reel": reel}


@app.get("/api/v1/storage/{job_id}/frames/{frame_index}")
async def get_frame(job_id: str, frame_index: int):
    """Serve a frame image by index."""
    storage = get_phone_storage()
    if job_id not in storage.metadata:
        raise HTTPException(status_code=404, detail="Job not found")
    frames = storage.metadata[job_id].get("frames", [])
    if frame_index >= len(frames):
        raise HTTPException(status_code=404, detail="Frame not found")
    frame_path = Path(frames[frame_index]["localPath"])
    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame file not found")
    return FileResponse(frame_path, media_type="image/jpeg")


@app.get("/api/v1/storage/{job_id}/images/{filename}")
async def get_image(job_id: str, filename: str):
    """Serve a place/board image by filename."""
    storage = get_phone_storage()
    if job_id not in storage.metadata:
        raise HTTPException(status_code=404, detail="Job not found")
    image_dir = storage.images_dir / job_id
    image_path = image_dir / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(image_path, media_type="image/jpeg")


@app.delete("/api/v1/storage/{job_id}")
async def cleanup_storage(job_id: str):
    storage = get_phone_storage()
    storage.cleanup_job(job_id)
    stats = storage.get_storage_stats()
    STORAGE_SIZE.set(stats["total_size_bytes"])
    STORAGE_FILES.set(stats["total_frames"] + stats["total_images"])
    return {"status": "cleaned", "job_id": job_id}


# ─── Local Dev Runner (no Redis) ────────────────────────────────────────
def run_local_worker():
    import queue

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