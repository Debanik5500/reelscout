"""In-process job manager: runs reel processing in the background and pushes live progress.

No Redis needed. Jobs live in memory (finished reels live in phone storage, so a restart only loses
in-flight work, which the app re-submits). Subscribers are asyncio queues fed from worker threads.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from server import quota

logger = logging.getLogger("reelscout.jobs")

FAILED_TTL = 3 * 60  # seconds after the failure; the app removes failed cards after 2 minutes
LIVE = ("queued", "downloading", "watching", "keyMoments", "placesAndBoard")


class JobManager:
    def __init__(self, workers: int | None = None):
        self._pool = ThreadPoolExecutor(max_workers=workers or int(os.getenv("REELSCOUT_JOB_WORKERS", "2")))
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []

    # ── queries ──────────────────────────────────────────────────────────
    def get(self, job_id: str) -> dict | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def active(self) -> list[dict]:
        """Live and failed jobs, newest first. Ready jobs are served from storage instead.
        Failed jobs are forgotten after FAILED_TTL so old errors do not pile up in the app."""
        now = time.time()
        with self._lock:
            for job_id in [i for i, j in self._jobs.items()
                           if j["status"] == "failed" and now - j.get("failed_at", j["created"]) > FAILED_TTL]:
                del self._jobs[job_id]
            jobs = [dict(j) for j in self._jobs.values() if j["status"] != "ready"]
        return sorted(jobs, key=lambda j: j["created"], reverse=True)

    def dismiss(self, job_id: str) -> bool:
        """Forget a failed job (the user closed its error card). Live jobs cannot be dismissed."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job["status"] == "failed":
                del self._jobs[job_id]
                return True
        return False

    # ── submit ───────────────────────────────────────────────────────────
    def submit(self, url: str, device_id: str | None = None) -> dict:
        from reelscout import batch  # heavy imports (cv2, whisper) only when a job really starts

        from reelscout import context
        from reelscout.storage import get_phone_storage

        job_id = batch.job_id_for(url)
        # Already processed: nothing to run. The caller just returns the stored reel.
        if context.load_context(get_phone_storage().results_dir, job_id) is not None:
            return dict(id=job_id, url=url, status="ready", progress=1.0, title="", chips=[], details={},
                        error="", created=time.time(), existing=True)
        with self._lock:
            existing = self._jobs.get(job_id)
            if existing and existing["status"] in LIVE:
                return dict(existing)
            quota.reserve(device_id, job_id)  # raises QuotaExceeded at the monthly limit; already-paid jobs pass
            job = dict(id=job_id, url=url, status="queued", progress=0.0, title="", chips=[], details={},
                       error="", created=time.time(), device=device_id)  # fresh dict: no stale failed_at
            self._jobs[job_id] = job
        self._emit(job)
        self._pool.submit(self._run, job_id, url)
        return dict(job)

    def _run(self, job_id: str, url: str) -> None:
        from reelscout import batch
        from reelscout.config import Settings
        from server.keymoments import _resolve_username

        def on_progress(status: str, progress: float, detail: str = "", title: str = "", chips=None) -> None:
            self._update(job_id, status=status, progress=progress, detail=(status, detail), title=title, chips=chips)

        try:
            batch.process_reel(url, Settings.from_env(), resolve_username=_resolve_username, on_progress=on_progress)
            self._update(job_id, status="ready", progress=1.0)
        except batch.DuplicateReel as dup:
            # Same reel under another link form: drop this job and tell the app which reel it really is.
            with self._lock:
                device = (self._jobs.pop(job_id, None) or {}).get("device")
            quota.refund(device, job_id)  # a duplicate costs the user nothing
            self._emit(dict(id=job_id, status="removed", canonical=dup.job_id))
        except Exception as exc:
            logger.exception("job %s failed", job_id)
            quota.refund((self.get(job_id) or {}).get("device"), job_id)  # a failed reel is not charged
            self._update(job_id, status="failed", error=str(exc)[:200])

    def _update(self, job_id: str, status: str, progress: float | None = None, detail=None,
                title: str = "", chips=None, error: str = "") -> None:
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = status
            if progress is not None:
                job["progress"] = progress
            if detail and detail[1]:
                job["details"][detail[0]] = detail[1]
            if title:
                job["title"] = title
            if chips is not None:
                job["chips"] = chips
            if error:
                job["error"] = error
            if status == "failed" and "failed_at" not in job:
                job["failed_at"] = time.time()
            snapshot = dict(job, details=dict(job["details"]))
        self._emit(snapshot)

    # ── push ─────────────────────────────────────────────────────────────
    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.append((loop, queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        with self._lock:
            self._subscribers = [(l, q) for l, q in self._subscribers if q is not queue]

    def _emit(self, job: dict) -> None:
        with self._lock:
            targets = list(self._subscribers)
        for loop, queue in targets:
            loop.call_soon_threadsafe(queue.put_nowait, job)


manager = JobManager()
