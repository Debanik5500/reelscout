"""Key-moment selection and prepared reel payloads for the iOS client."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote, urlparse

import cv2
import numpy as np
import requests

from reelscout import context as reel_context
from server.schema import ReelPayload


logger = logging.getLogger("reelscout.keymoments")

TEST_REELS_METADATA = Path("/Users/debanikdebnath/work/reelscout/test_reels/metadata.json")

_FRAME_SIZE = (64, 64)
_MAX_MOMENTS = 8
_MIN_INDEX_GAP = 2
_MIN_SECONDS_GAP = 2.0
_BLUR_PERCENTILE = 15.0
_MAX_LABEL = 40
_MAX_SUMMARY = 200
_MAX_POINTS = 12
_MAX_SLIDES = 10

_IG_HANDLE_RE = re.compile(r"^[A-Za-z0-9._]{2,30}$")
_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 "
        "Mobile/15E148 Safari/604.1"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
_PLATFORM_BY_HOST = {
    "instagram.com": "instagram",
    "instagr.am": "instagram",
    "tiktok.com": "tiktok",
    "youtube.com": "youtube",
    "youtu.be": "youtube",
    "twitter.com": "twitter",
    "x.com": "twitter",
    "github.com": "github",
    "reddit.com": "reddit",
}

_source_meta: dict | None = None
_job_source_meta: dict[str, dict] = {}


def format_time(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


def _video_id(job_id: str) -> str:
    return job_id[5:] if job_id.startswith("reel-") else job_id


def _load_source_meta() -> dict:
    global _source_meta
    if _source_meta is None:
        try:
            _source_meta = json.loads(TEST_REELS_METADATA.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _source_meta = {}
    return _source_meta


def _find_source_meta(job_id: str) -> dict:
    cached = _job_source_meta.get(job_id)
    if cached is not None:
        return cached
    video_id = _video_id(job_id)
    entry: dict = {}
    for key, value in _load_source_meta().items():
        meta = (value or {}).get("meta") or {}
        if meta.get("id") and meta["id"] == video_id:
            entry = value or {}
            break
    if not entry and video_id:
        target = f"https://www.instagram.com/reels/{video_id}".rstrip("/")
        for key, value in _load_source_meta().items():
            if key.rstrip("/") == target:
                entry = value or {}
                break
    _job_source_meta[job_id] = entry
    return entry


def _unique_frames(frames: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for frame in frames:
        key = str(frame.get("localPath") or f"{frame.get('index')}:{frame.get('timestamp')}")
        if key in seen:
            continue
        seen.add(key)
        unique.append(frame)
    return unique


def _load_gray(path) -> np.ndarray | None:
    if not path:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.resize(image, _FRAME_SIZE, interpolation=cv2.INTER_AREA)


def _frame_metrics(frames: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    diffs = np.zeros(len(frames), dtype=np.float64)
    sharp = np.zeros(len(frames), dtype=np.float64)
    loaded: list[int] = []
    prev: np.ndarray | None = None
    for pos, frame in enumerate(frames):
        gray = _load_gray(frame.get("localPath") or frame.get("local_path"))
        if gray is None:
            continue
        loaded.append(pos)
        if prev is not None:
            delta = np.abs(gray.astype(np.int16) - prev.astype(np.int16))
            diffs[pos] = float(np.mean(delta)) / 255.0
        sharp[pos] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        prev = gray
    return diffs, sharp, loaded


# Two frames are "the same picture" when both the pixels and the colours barely differ. Measured: a locked-off
# 13-frame shot differs by at most 1.1% (colour correlation >= 0.996); real scene changes differ far more.
_SAME_PIXEL_DIFF = 0.045
_SAME_HIST_CORR = 0.985


def _signatures(frames: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Tiny grayscale thumbnail plus a colour histogram for every frame that can be read."""
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pos, frame in enumerate(frames):
        path = frame.get("localPath") or frame.get("local_path")
        image = cv2.imread(str(path)) if path else None
        if image is None:
            continue
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), _FRAME_SIZE, interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        out[pos] = (gray.astype(np.float32), cv2.normalize(hist, hist).flatten())
    return out


def _same_picture(sigs: dict[int, tuple[np.ndarray, np.ndarray]], a: int, b: int) -> bool:
    if a not in sigs or b not in sigs:
        return False
    pixel = float(np.mean(np.abs(sigs[a][0] - sigs[b][0])) / 255.0)
    colour = float(cv2.compareHist(sigs[a][1], sigs[b][1], cv2.HISTCMP_CORREL))
    return pixel < _SAME_PIXEL_DIFF and colour > _SAME_HIST_CORR


def _moment_candidates(diffs, sharp, loaded: list[int]) -> list[tuple[int, float, str]]:
    if not loaded:
        return []
    sharp_values = sharp[loaded]
    nonzero = diffs[diffs > 0]
    diff_threshold = float(np.percentile(nonzero, 35)) if nonzero.size else 0.0
    blur_threshold = float(np.percentile(sharp_values, _BLUR_PERCENTILE))
    sharp_median = float(np.median(sharp_values))
    out: list[tuple[int, float, str]] = []
    for pos in loaded:
        if sharp[pos] < blur_threshold:
            continue
        if diffs[pos] > 0 and diffs[pos] >= diff_threshold:
            out.append((pos, float(diffs[pos]), "Scene change"))
        elif pos == 0 and sharp[pos] >= sharp_median:
            out.append((pos, float(diffs[pos]), "Clear frame"))
    if not out:
        best = max(loaded, key=lambda p: sharp[p])
        out.append((best, float(diffs[best]), "Clear frame"))
    return out


def _pick_moments(
    frames: list[dict], candidates: list[tuple[int, float, str]], same_picture=None
) -> tuple[list[int], dict[int, str]]:
    def seconds(pos: int) -> float:
        return float(frames[pos].get("timestamp") or 0.0)

    def index(pos: int) -> int:
        try:
            return int(frames[pos].get("index", pos))
        except (TypeError, ValueError):
            return pos

    def fits(pos: int, chosen: list[int]) -> bool:
        for other in chosen:
            if abs(index(pos) - index(other)) < _MIN_INDEX_GAP:
                return False
            if abs(seconds(pos) - seconds(other)) < _MIN_SECONDS_GAP:
                return False
            if same_picture is not None and same_picture(pos, other):
                return False  # a static shot must not fill the stack with copies of one frame
        return True

    ordered = sorted(candidates, key=lambda c: (0 if c[0] == 0 else 1, -c[1], c[0]))
    chosen: list[int] = []
    for pos, _, _ in ordered:
        if len(chosen) >= _MAX_MOMENTS:
            break
        if fits(pos, chosen):
            chosen.append(pos)
    chosen.sort()
    notes = {pos: note for pos, _, note in candidates}
    return chosen, notes


def _ocr_map(job_id: str, storage) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        result = storage.load_result(job_id)
    except Exception:
        result = None
    for frame in getattr(result, "source_frames", None) or []:
        text = getattr(frame, "ocr_text", None)
        frame_index = getattr(frame, "index", None)
        if text and frame_index is not None:
            out[int(frame_index)] = str(text)
    for frame in ((storage.metadata.get(job_id) or {}).get("frames") or []):
        text = frame.get("ocrText") or frame.get("ocr_text")
        frame_index = frame.get("index")
        if text and frame_index is not None:
            out[int(frame_index)] = str(text)
    return out


def _label(ocr: str | None, seconds: float) -> str:
    if ocr:
        for line in ocr.splitlines():
            line = line.strip()
            if sum(ch.isalnum() for ch in line) >= 2:
                return line[:_MAX_LABEL]
    return f"Scene at {format_time(seconds)}"


def select_key_moments(job_id: str, storage) -> list[dict]:
    """Frames worth capturing, sorted by timestamp. Cached per job."""
    # Cap and spacing are part of the file name, so changing either never serves stale picks.
    cache_path = Path(storage.results_dir) / f"{job_id}.keymoments{_MAX_MOMENTS}-{int(_MIN_SECONDS_GAP * 10)}-{int(_BLUR_PERCENTILE)}-d.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                if len(cached) < 3:
                    cached = cached + _web_image_moments(job_id, len(cached))
                return cached
        except (OSError, ValueError):
            pass

    frames = _unique_frames((storage.metadata.get(job_id) or {}).get("frames") or [])
    moments: list[dict] = []
    if frames:
        ocr = _ocr_map(job_id, storage)
        diffs, sharp, loaded = _frame_metrics(frames)
        sigs = _signatures(frames)
        chosen, notes = _pick_moments(frames, _moment_candidates(diffs, sharp, loaded),
                                      same_picture=lambda a, b: _same_picture(sigs, a, b))
        for pos in chosen:
            frame = frames[pos]
            frame_index = int(frame.get("index", pos))
            seconds = float(frame.get("timestamp") or 0.0)
            moments.append({
                "id": hashlib.sha1(f"{job_id}{frame_index}".encode("utf-8")).hexdigest()[:16],
                "index": frame_index,
                "timestamp": format_time(seconds),
                "seconds": seconds,
                "label": _label(ocr.get(frame_index), seconds),
                "note": notes.get(pos, "Scene change"),
                "frame_url": f"/api/v1/storage/{job_id}/frames/{frame_index}",
            })
    if len(moments) < 3:
        moments = moments + _web_image_moments(job_id, len(moments))
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(moments, indent=2), encoding="utf-8")
    except OSError:
        logger.warning("could not cache key moments for %s", job_id)
    return moments


def _post_moments(job_id: str, storage) -> list[dict]:
    """Image post: every slide is a key moment, in order (no scene picking, no timestamps)."""
    frames = (storage.metadata.get(job_id) or {}).get("frames") or []
    ocr = _ocr_map(job_id, storage)
    total = len(frames)
    moments = []
    for pos, frame in enumerate(frames[:_MAX_SLIDES]):
        index = int(frame.get("index", pos))
        text = ocr.get(index) or ""
        label = next((line.strip()[:_MAX_LABEL] for line in text.splitlines()
                      if sum(ch.isalnum() for ch in line) >= 3), f"Slide {pos + 1}")
        moments.append({
            "id": hashlib.sha1(f"{job_id}{index}".encode("utf-8")).hexdigest()[:16],
            "index": index,
            "timestamp": f"{pos + 1}/{total}",
            "seconds": float(pos),
            "label": label,
            "note": "",
            "frame_url": f"/api/v1/storage/{job_id}/frames/{index}",
        })
    return moments


_IG_PATH = re.compile(r"^/(?:reels?|p|tv|share(?:/(?:reel|p))?)/[A-Za-z0-9_-]+")


def is_supported_link(url: str) -> bool:
    """True only for a link to one reel/post on a supported site. Search pages, home pages and GitHub are not."""
    parsed = urlparse(url or "")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    site = next((d for d in _PLATFORM_BY_HOST if d != "github.com" and (host == d or host.endswith("." + d))), None)
    if site is None:
        return False
    if site in ("instagram.com", "instagr.am"):
        return bool(_IG_PATH.match(parsed.path))  # the bare home page has no reel in it
    return len(parsed.path.strip("/")) > 0


def _platform(source_url: str) -> str:
    host = (urlparse(source_url or "").hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    for domain, name in _PLATFORM_BY_HOST.items():
        if host == domain or host.endswith("." + domain):
            return name
    return "unknown"


def _first_line(text: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _first_sentence(text: str) -> str:
    return (text or "").split(". ")[0].strip()


def _cand_value(candidate, field, default=None):
    if isinstance(candidate, dict):
        return candidate.get(field, default)
    return getattr(candidate, field, default)


def _strip_emoji(text: str) -> str:
    """Drop emoji/decorative symbols so titles stay plain words."""
    if not text:
        return ""
    keep: list[str] = []
    for char in text:
        code = ord(char)
        if code in (0x200D, 0x200E, 0x200F, 0xFE0E, 0xFE0F) or unicodedata.combining(char):
            continue
        if 0x2190 <= code <= 0x21FF:
            continue
        if unicodedata.category(char) in ("So", "Sk", "Cf", "Me"):
            continue
        keep.append(char)
    return "".join(keep)


def _clean_title(text: str) -> str:
    """Plain, hashtag/URL/emoji free title capped at 8 words."""
    text = _strip_emoji(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    for bad in ('"', "\u201c", "\u201d"):
        text = text.replace(bad, " ")
    words = [w for w in text.split() if w and not w.startswith("#")]
    return " ".join(words[:8]).strip(" \t\r\n,.;:\u00b7|-\u2013\u2014\u2026!?\u203a\u00bb")


def _clean_line(text: str, limit: int = 60) -> str:
    text = re.sub(r"\s+", " ", _strip_emoji(text or "")).strip()
    for bad in ('"', "\u201c", "\u201d"):
        text = text.replace(bad, " ")
    return text[:limit].strip()


def _candidates(result) -> list:
    if isinstance(result, dict):
        return list(result.get("candidates") or [])
    return list(getattr(result, "candidates", None) or [])


def _ig_handle(value) -> str | None:
    """Profile URL -> handle. Numeric ids and junk never pass."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith("http"):
        path = (urlparse(text).path or "").strip("/")
        text = path.split("/")[0] if path else ""
    else:
        return None
    if text.startswith("@"):
        text = text[1:]
    if not _IG_HANDLE_RE.match(text) or text.replace(".", "").isdigit():
        return None
    return text


def _resolve_username(job_id: str) -> str | None:
    """Best-effort IG handle for the uploader. Two timeboxed attempts."""
    video_id = _video_id(job_id)
    if not video_id:
        return None
    # 1) public oEmbed endpoint: no login, author_url is the profile URL
    oembed = (
        "https://www.instagram.com/api/v1/oembed/?url="
        + quote(f"https://www.instagram.com/p/{video_id}/", safe="")
        + "&maxwidth=640"
    )
    try:
        resp = requests.get(oembed, headers=_BROWSER_HEADERS, timeout=8, allow_redirects=True)
        if resp.status_code == 200:
            handle = _ig_handle((resp.json() or {}).get("author_url"))
            if handle:
                return handle
    except Exception as exc:
        logger.debug("oembed username lookup failed for %s: %s", video_id, exc)
    # 2) fetch the reel page itself and parse a handle out of the markup
    page = f"https://www.instagram.com/reels/{video_id}/"
    try:
        resp = requests.get(page, headers=_BROWSER_HEADERS, timeout=8, allow_redirects=True)
        text = resp.text or ""
        for pattern in (
            r'"owner"\s*:\s*\{[^}]*?"username"\s*:\s*"([A-Za-z0-9._]+)"',
            r'"username"\s*:\s*"([A-Za-z0-9._]{2,30})"',
            r'property="og:url"\s+content="https://www\.instagram\.com/([A-Za-z0-9._]+)/"',
        ):
            match = re.search(pattern, text)
            if match:
                handle = _ig_handle("https://www.instagram.com/" + match.group(1))
                if handle:
                    return handle
    except Exception as exc:
        logger.debug("reel page username lookup failed for %s: %s", video_id, exc)
    return None


def _top_kind(result) -> str:
    candidates = _candidates(result)
    if candidates:
        kind = str(_cand_value(candidates[0], "kind") or "").strip()
        if kind:
            return kind
    return "other"


def _author(meta: dict, username: str | None) -> str:
    if username:
        return "@" + username
    uploader = str(meta.get("uploader") or "").strip()
    if uploader and not uploader.replace(".", "").isdigit():
        return "@" + uploader
    return "Instagram User"


def _web_image_moments(job_id: str, have: int) -> list[dict]:
    """Stretch goal: SearXNG images when frame selection found too few moments."""
    if have >= _MAX_MOMENTS:
        return []
    base = os.getenv("SEARXNG_URL", "").strip().rstrip("/")
    meta = _find_source_meta(job_id).get("meta") or {}
    words = [
        w for w in _clean_line(_first_line(str(meta.get("caption") or "")), 120).split()
        if w and not w.startswith("#")
    ]
    query = " ".join(words[:6])
    if not base or not query:
        return []
    try:
        resp = requests.get(
            base + "/search",
            params={"q": query, "categories": "image", "format": "json"},
            timeout=2,
        )
        resp.raise_for_status()
        hits = (resp.json() or {}).get("results") or []
    except Exception:
        return []
    moments: list[dict] = []
    for hit in hits:
        image = str(hit.get("img") or hit.get("image") or "")
        if not image.startswith(("http://", "https://")):
            continue
        seconds = float(have + len(moments) + 1) * 5.0
        moments.append({
            "id": hashlib.sha1(f"{job_id}web{len(moments)}".encode("utf-8")).hexdigest()[:16],
            "timestamp": format_time(seconds),
            "seconds": seconds,
            "label": (_clean_line(str(hit.get("title") or ""), 80) or "Web image")[:_MAX_LABEL],
            "note": "From web search",
            "frame_url": image,
        })
        if have + len(moments) >= _MAX_MOMENTS:
            break
    return moments


def prepare_reel(job_id: str, storage) -> dict | None:
    """Full payload for one reel, or None when no result is saved."""
    job_meta = storage.metadata.get(job_id)
    if not job_meta:
        return None
    try:
        result = storage.load_result(job_id)
    except Exception:
        logger.exception("could not load result for %s", job_id)
        result = None
    if result is None:
        return None

    frames = job_meta.get("frames") or []
    meta = _find_source_meta(job_id).get("meta") or {}
    caption = (meta.get("caption") or "").strip()
    summary = (getattr(result, "summary", "") or "").strip()

    # LLM-decided part. Missing sidecar -> empty context, never a guess from raw candidates.
    ctx = reel_context.load_context(storage.results_dir, job_id) or {}
    reel_context.fill_item_lines(ctx)  # older saved reels gain a line per described item too
    title = _clean_title(ctx.get("title") or _first_line(caption) or _first_sentence(summary)) or "Reel"
    ctx_summary = (ctx.get("summary") or summary)[:_MAX_SUMMARY]
    useful = ctx.get("useful_context") or []

    video_id = _video_id(job_id)
    source_url = ctx.get("source_url") or meta.get("url") or f"https://www.instagram.com/reels/{video_id}/"
    is_post = ctx.get("kind") == "post"
    timestamps = [f.get("timestamp") for f in frames if isinstance(f.get("timestamp"), (int, float))]

    payload = ReelPayload(
        id=job_id,
        created=job_meta.get("created"),
        title=title,
        summary=ctx_summary,
        category=ctx.get("category") or "",
        location=ctx.get("location") or {},
        places=ctx.get("places") or ([ctx["location"]] if (ctx.get("location") or {}).get("name") else []),
        products=ctx.get("products") or [],
        people=ctx.get("people") or [],
        links=ctx.get("links") or [],
        things_mentioned=ctx.get("things_mentioned") or [],
        activities=ctx.get("activities") or [],
        useful_context=useful,
        sources=ctx.get("sources") or [],
        username=ctx.get("username") or "",
        author=_author(meta, ctx.get("username")),
        duration="0:00" if is_post or not timestamps else format_time(max(timestamps)),
        cover_url=f"/api/v1/storage/{job_id}/frames/0" if frames else "",
        group=ctx.get("group") or _top_kind(result),
        platform=_platform(source_url),
        source_url=source_url,
        frame_count=len(frames),
        key_moments=_post_moments(job_id, storage) if is_post else select_key_moments(job_id, storage),
        points=[{"text": t} for t in useful[:_MAX_POINTS]],
    )
    return payload.model_dump()


def prepare_live(job: dict, storage) -> dict:
    """Payload for a reel still being processed (or failed). Cover appears as soon as frames are saved."""
    job_id = job["id"]
    frames = (storage.metadata.get(job_id) or {}).get("frames") or []
    return ReelPayload(
        id=job_id,
        created=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(job["created"])),
        title=job.get("title") or "",
        status=job["status"],
        progress=float(job.get("progress") or 0.0),
        source_url=job["url"],
        platform=_platform(job["url"]),
        cover_url=f"/api/v1/storage/{job_id}/frames/0" if frames else "",
        frame_count=len(frames),
        details=job.get("details") or {},
        chips=job.get("chips") or [],
        error=job.get("error") or "",
    ).model_dump()
