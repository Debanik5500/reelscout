"""Process reels once, up front: frames + evidence + ReelContext land in phone storage (~/ReelScoutData).

The app then only reads stored results; no reel is analysed while the user waits.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable

from . import context, extract, ingest, llm, media, ocr, pipeline
from .config import Settings
from .storage import ReelResult, get_phone_storage

logger = logging.getLogger("reelscout.batch")

VISION_FRAMES = 6

# on_progress(status, progress, **info): status is the app's ReelStatus raw value, progress 0..1 inside that step.
ProgressFn = Callable[..., None]


class DuplicateReel(Exception):
    """The downloaded video is a reel we already have (reached via a share link, redirect or other URL form)."""

    def __init__(self, job_id: str):
        super().__init__(f"already have {job_id}")
        self.job_id = job_id


def cookies_file() -> str | None:
    """Instagram cookies (Netscape format) for yt-dlp: REELSCOUT_COOKIES_FILE, else ~/.reelscout/instagram_cookies.txt."""
    path = Path(os.getenv("REELSCOUT_COOKIES_FILE") or Path.home() / ".reelscout" / "instagram_cookies.txt")
    return str(path) if path.is_file() else None


_SHORTCODE = re.compile(r"/(?:reels?|p|tv)/([A-Za-z0-9_-]+)")


def job_id_for(source: str) -> str:
    """Stable id from the reel's shortcode, so tracking params (?stkn=...) or a trailing slash never change it."""
    match = _SHORTCODE.search(urlparse(source).path)
    if match:
        return "reel-" + match.group(1)
    if source.startswith(("http://", "https://")):
        # No shortcode (not a reel link): a hash keeps unrelated URLs from colliding on a shared path name.
        return "reel-" + hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    return "reel-" + Path(source).stem


def _fmt_duration(seconds) -> str:
    try:
        total = int(round(float(seconds)))
    except (TypeError, ValueError):
        return "?"
    return f"{total // 60}:{total % 60:02d}"


def _caption_hint(caption: str) -> str:
    """Provisional title from the caption's first line, shown until the real title exists."""
    for line in (caption or "").splitlines():
        words = [w for w in line.split() if not w.startswith(("#", "@")) and not w.startswith("http")]
        if words:
            return " ".join(words)[:60]
    return ""


def _vision_mode() -> str:
    """REELSCOUT_OBSERVE: on (always look at frames), off (text only), auto (only when the text is thin)."""
    mode = os.getenv("REELSCOUT_OBSERVE", "on").lower()
    return mode if mode in ("on", "off", "auto") else "on"


def _text_signals(analysis: dict, caption: str) -> dict:
    """How much readable text the reel gave us without looking at a single frame."""
    return dict(ocr_lines=len(context.evidence_ocr(analysis.get("ocr_lines") or [])),
                transcript_words=len((analysis.get("transcript_text") or "").split()),
                caption_words=len((caption or "").split()))


def _needs_vision(mode: str, signals: dict) -> bool:
    if mode != "auto":
        return mode == "on"
    # auto: a reel that already shows or says a lot of text is understood from the text alone
    return not (signals["ocr_lines"] >= 8 or signals["transcript_words"] >= 60 or signals["caption_words"] >= 40)


def _vision_frame_count(st: Settings) -> int:
    """Images sent to the vision model. Groq accepts at most 3 per request; override with REELSCOUT_VISION_FRAMES."""
    override = os.getenv("REELSCOUT_VISION_FRAMES")
    if override and override.isdigit() and int(override) > 0:
        return int(override)
    return 3 if st.for_stage("VISION").llm == "groq" else VISION_FRAMES


def _top_group(analysis: dict) -> str:
    """Kind of the best local candidate (github_repo, website, ...); picks the cover art and grouping."""
    cands = analysis.get("candidates") or []
    return str(cands[0].get("kind") or "") if cands else ""


def _evenly(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def _chips(ctx: dict) -> list[str]:
    """Short findings shown as chips while the app finishes the last step."""
    chips = [ctx["location"]["name"]] if ctx["location"]["name"] else []
    chips += [p["name"] for p in ctx["products"][:2]] + [l["label"] for l in ctx["links"][:2]]
    chips += ctx["useful_context"][:1]
    return [c[:40] for c in chips if c][:4]


def _finish(job_id: str, storage, st: Settings, analysis: dict, frame_paths: list[str], frame_label: str,
            meta: dict, caption: str, progress: ProgressFn, resolve_username, extra: dict) -> str:
    """Shared tail for reels and image posts: observe the visuals, research + synthesize, save the context."""
    progress("keyMoments", 0.0)
    observe_usage: dict = {}
    observations: dict = {}
    signals = _text_signals(analysis, caption)
    looked = _needs_vision(_vision_mode(), signals)
    if looked:
        try:
            observations, observe_usage = context.observe(frame_paths, st)
        except Exception as exc:
            logger.warning("observe failed for %s: %s", job_id, exc)

    progress("keyMoments", 1.0, detail=frame_label, title=_caption_hint(caption))
    progress("placesAndBoard", 0.0)
    ctx, debug = context.synthesize(analysis, observations, st)
    # Every LLM call this reel cost, by stage, so cost_report.py can price it exactly.
    debug["usage"]["observe"] = observe_usage
    debug["vision"] = dict(used=looked, mode=_vision_mode(), signals=signals)
    debug["usage"]["extract"] = (analysis.get("llm") or {}).get("stages", [])
    debug["usage"]["ledger"] = llm.ledger()  # authoritative: what each provider actually served
    handle = extra.pop("username", "")
    ctx.update(extra)
    if handle:
        ctx["username"] = handle
    if not ctx["username"] and resolve_username:
        ctx["username"] = resolve_username(job_id) or ""
    context.save_context(storage.results_dir, job_id, ctx, debug, observations)
    progress("placesAndBoard", 1.0, title=ctx["title"], chips=_chips(ctx),
             detail="1 place" if ctx["location"]["name"] else "no place")
    return job_id


def _process_post(source: str, images: list[bytes], meta: dict, job_id: str, st: Settings, storage,
                  progress: ProgressFn, resolve_username) -> str:
    """Image post / carousel: every slide becomes a frame (OCR + vision); there is no audio to transcribe."""
    import cv2
    import numpy as np

    caption = meta.get("caption", "")
    progress("downloading", 1.0, detail=f"{len(images)} slide{'s' if len(images) != 1 else ''}")
    progress("watching", 0.1, detail="reading slides")

    frames, ocr_lines = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, raw in enumerate(images):
            array = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if array is None:
                continue
            data = cv2.imencode(".jpg", array)[1].tobytes()
            path = Path(tmp) / f"{i}.jpg"
            path.write_bytes(data)
            seen: set[str] = set()
            texts = []
            for text, conf in ocr.ocr_frame(str(path)):
                key = "".join(ch for ch in text.lower() if ch.isalnum())
                if key and key not in seen:
                    seen.add(key)
                    texts.append(text)
                    ocr_lines.append(dict(text=text, conf=conf, t=float(i), n_frames=1))
            frames.append(storage.save_frame(job_id, i, float(i), data, ocr_text="\n".join(texts) or None))
            progress("watching", 0.1 + 0.8 * (i + 1) / len(images))
    if not frames:
        raise RuntimeError("Could not read any images from this post")

    # saved result so the app payload can find the slides
    storage.save_result(ReelResult(job_id=job_id, status="completed", summary="", candidates=[],
                                   timings={}, source_frames=frames))
    cands = extract.merge(extract.gather([], ocr_lines, caption) + extract.social_candidates(meta))
    analysis = dict(
        caption=caption, meta={k: v for k, v in meta.items() if k != "caption"}, ocr_lines=ocr_lines,
        candidates=[dict(kind=c.kind, name=c.name, url=c.url, confidence=c.confidence) for c in cands[:8]],
        transcript_text="", unresolved_clues=[], summary="")
    paths = [f.local_path for f in _evenly(frames, _vision_frame_count(st)) if f.local_path]
    uploader = str(meta.get("uploader") or "")
    extra = dict(kind="post", source_url=meta.get("webpage_url") or source, group=_top_group(analysis),
                 username=uploader if re.fullmatch(r"[A-Za-z0-9._]{2,30}", uploader) else "")
    return _finish(job_id, storage, st, analysis, paths, f"{len(frames)} slides", meta, caption, progress,
                   resolve_username, extra)


def process_reel(source: str, st: Settings | None = None, caption: str = "", force: bool = False,
                 resolve_username=None, on_progress: ProgressFn | None = None) -> str:
    """Analyse one reel or image post (URL or local video). Returns the job id.
    Skips items already processed unless force."""
    progress = on_progress or (lambda *a, **k: None)
    llm.reset_ledger()  # exact per-model usage for this reel, including any fallback calls
    st = st or Settings.from_env()
    if st.llm == "none":
        raise RuntimeError("Set REELSCOUT_LLM (and its API key) before processing reels")
    storage = get_phone_storage()
    job_id = job_id_for(source)
    if not force and context.load_context(storage.results_dir, job_id) is not None:
        logger.info("%s already processed", job_id)
        return job_id

    progress("downloading", 0.0)

    # /p/ links can be image carousels, which yt-dlp cannot read: try the image route first.
    if source.startswith(("http://", "https://")) and re.search(r"/p/", urlparse(source).path):
        try:
            post = ingest.fetch_post(source)
        except Exception as exc:
            raise RuntimeError(f"Could not read this post: {exc}") from exc
        if post:
            images, meta = post
            return _process_post(source, images, {**meta, "caption": caption or meta["caption"]}, job_id, st,
                                 storage, progress, resolve_username)

    video, meta = ingest.fetch(source, st.cache_dir / "downloads", cookies_file=cookies_file(), comments=False)
    caption = caption or meta.get("caption", "")
    real_id = job_id_for(meta["webpage_url"]) if meta.get("webpage_url") else job_id
    if real_id != job_id and not force and context.load_context(storage.results_dir, real_id) is not None:
        raise DuplicateReel(real_id)
    info = media.probe(video)
    progress("downloading", 1.0, detail=_fmt_duration(info.get("duration")) + " video")

    # frames + OCR + transcript saved to storage (what the app displays)
    progress("watching", 0.1, detail="captions + audio")
    stored = pipeline.analyze_reel_with_storage(Path(video), job_id=job_id, settings=st, frames_only=True)

    progress("watching", 0.5)
    # rich evidence; vision/resolve stay off because observe() and research() replace them
    analysis = pipeline.analyze(str(video), dataclasses.replace(st, vision="off", resolve="off"), caption=caption)
    analysis["caption"] = caption
    analysis["meta"] = {**analysis.get("meta", {}), **{k: v for k, v in meta.items() if k != "caption"}}

    paths = [f.local_path for f in _evenly(stored.source_frames, _vision_frame_count(st)) if f.local_path]
    page = meta.get("webpage_url") or (source if source.startswith(("http://", "https://")) else "")
    return _finish(job_id, storage, st, analysis, paths, f"{len(stored.source_frames)} frames", meta, caption,
                   progress, resolve_username, dict(kind="reel", source_url=page, group=_top_group(analysis)))
