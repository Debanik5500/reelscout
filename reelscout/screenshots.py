"""Screenshots -> notes, events and places.

A person uploads screenshots (posters, invitations, tickets, chats, schedules, articles...). We read the text
locally (OCR), take one look with the vision model, and ask the LLM what each screenshot is and whether it holds
an event worth a reminder. Dates are resolved against the user's own clock and timezone, then validated in code:
the model never decides what time zone or year something is in without a check.

Reuses the reel machinery: frames/OCR storage, the context sidecar, places geocoding, link checks, job events.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cv2
import numpy as np

from . import context, llm, ocr
from .events import EVENT_RULES, normalize_events
from .config import Settings
from .llm import complete_json
from .storage import ReelResult, get_phone_storage

logger = logging.getLogger("reelscout.screenshots")

MAX_IMAGES = 10
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SIDE = 2000
VISION_IMAGES = 3  # Groq accepts at most 3 images per request
SHOT_TYPES = ("event", "ticket", "booking", "schedule", "message", "article", "recipe", "product",
              "receipt", "social_post", "map", "other")

OBSERVE_SYSTEM = """You describe screenshots from a phone. You are an observer, not an identifier.
Say what kind of thing each screenshot is (poster, invitation, chat, ticket, schedule, app screen, article) and
describe visual details that text alone would miss: logos, venue photos, maps, QR codes, layout, highlighted dates.
Do NOT guess names, places or dates that are not written in the image.
Return JSON only, every list may be empty:
{"text_visible": [str], "people": [str], "objects": [str], "logos": [str], "screens_or_ui": [str], "food": [str], "landscape": [str], "distinctive_features": [str], "activities": [str], "other_clues": [str]}
Short phrases only."""

SYSTEM_TEMPLATE = """You read screenshots a person saved on their phone and pull out what is worth keeping, including events
that deserve a reminder. You get the text read from each screenshot (OCR), a neutral description of the visuals,
and the CURRENT date, time and timezone of the user.
Rules:
- Fill a field ONLY when the evidence supports it. Empty is correct and preferred over a guess. Never invent.
- shot_type: one of event, ticket, booking, schedule, message, article, recipe, product, receipt, social_post, map, other.
- title: max 8 words, plain words. summary: 1-2 sentences on what this is and why it was saved.
{event_rules}
- places: every real physical place the screenshot is about (a venue, an address on a ticket). Each has name,
  area, city, country, confidence 0-1, precision (exact|area|city) and note. Empty list when none.
- links: websites or apps named or shown (label, url only if written, kind, note). people: named people with a role.
- useful_context: up to 8 short facts worth remembering (prices, codes, steps, names). Do not repeat the events.
- Return JSON only:
{"title": str, "summary": str, "shot_type": str,
 "events": [{"title": str, "start": str, "end": str, "all_day": boolean, "tz": str, "year_inferred": boolean,
             "location": str, "notes": str, "confidence": number}],
 "places": [{"name": str, "area": str, "city": str, "country": str, "confidence": number, "precision": str, "note": str}],
 "links": [{"label": str, "url": str, "kind": str, "note": str}],
 "people": [{"name": str, "role": str}], "products": [{"name": str, "brand": str, "url": str}],
 "useful_context": [str], "things_mentioned": [str]}"""


# --- images ----------------------------------------------------------------------------------------------

def normalize_images(raw_images: list[bytes]) -> list[bytes]:
    """Validate that every upload is a real image and shrink it to at most MAX_SIDE pixels as a JPEG."""
    if not 1 <= len(raw_images) <= MAX_IMAGES:
        raise ValueError(f"send between 1 and {MAX_IMAGES} screenshots")
    out = []
    for raw in raw_images:
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("a screenshot is larger than 8 MB")
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("one of the files is not a readable image")
        height, width = image.shape[:2]
        scale = MAX_SIDE / max(height, width)
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out.append(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
    return out


def job_id_for(images: list[bytes]) -> str:
    """Same screenshots (in any order) give the same id, so re-sending them is free and not reprocessed."""
    digests = sorted(hashlib.sha1(i).hexdigest() for i in images)
    return "shot-" + hashlib.sha1("".join(digests).encode()).hexdigest()[:12]


def _zone(name: str | None) -> ZoneInfo | None:
    try:
        return ZoneInfo(name) if name else None
    except (ZoneInfoNotFoundError, ValueError):
        return None


SYSTEM = SYSTEM_TEMPLATE.replace("{event_rules}", EVENT_RULES.format(limit=8))


def build_evidence(ocr_by_image: list[list[str]], observations: dict, now: datetime, user_tz: str) -> str:
    lines = [f"CURRENT DATE AND TIME: {now.strftime('%A %Y-%m-%d %H:%M')} (user timezone: {user_tz})"]
    for index, texts in enumerate(ocr_by_image, 1):
        body = "\n".join(texts)[:2500] or "(no readable text)"
        lines.append(f"SCREENSHOT {index} TEXT (OCR):\n{body}")
    lines.append("VISUAL OBSERVATIONS: " + (context.json.dumps(observations, ensure_ascii=False) if observations else "(none)"))
    return "\n\n".join(lines)


# --- pipeline --------------------------------------------------------------------------------------------

def process(images: list[bytes], tz: str, st: Settings, progress, job_id: str | None = None, now: datetime | None = None) -> str:
    """Run the whole screenshot pipeline for already-normalised JPEGs. Returns the job id."""
    llm.reset_ledger()
    storage = get_phone_storage()
    job_id = job_id or job_id_for(images)
    zone = _zone(tz) or timezone.utc
    now = now or datetime.now(zone)
    tz_name = getattr(zone, "key", "UTC")

    progress("downloading", 1.0, detail=f"{len(images)} screenshot{'s' if len(images) != 1 else ''}")
    progress("watching", 0.1, detail="reading the text")

    frames, ocr_by_image, work = [], [], Path(st.cache_dir) / "screenshots"
    work.mkdir(parents=True, exist_ok=True)
    for index, data in enumerate(images):
        path = work / f"{job_id}-{index}.jpg"
        path.write_bytes(data)
        try:
            seen, texts = set(), []
            for text, _conf in ocr.ocr_frame(str(path)):
                key = "".join(ch for ch in text.lower() if ch.isalnum())
                if key and key not in seen:
                    seen.add(key)
                    texts.append(text)
        finally:
            path.unlink(missing_ok=True)
        ocr_by_image.append(texts)
        frames.append(storage.save_frame(job_id, index, float(index), data, ocr_text="\n".join(texts) or None))
        progress("watching", 0.1 + 0.8 * (index + 1) / len(images))

    storage.save_result(ReelResult(job_id=job_id, status="completed", summary="", candidates=[],
                                   timings={}, source_frames=frames))

    progress("keyMoments", 0.0)
    observations, observe_usage = {}, {}
    try:
        observations, observe_usage = context.observe([f.local_path for f in frames[:VISION_IMAGES]], st,
                                                      system=OBSERVE_SYSTEM)
    except Exception as exc:
        logger.warning("screenshot vision failed for %s: %s", job_id, exc)
    progress("keyMoments", 1.0, detail="understanding it")

    progress("placesAndBoard", 0.0)
    evidence = build_evidence(ocr_by_image, observations, now, tz_name)
    stg = st.for_stage("TEXT")
    data, usage = complete_json(SYSTEM, evidence, stg, max_tokens=2500)

    ctx = context._finalize(data, [], [])
    ctx["kind"] = "screenshot"
    ctx["category"] = ""
    shot_type = str(data.get("shot_type") or "").lower()
    ctx["shot_type"] = shot_type if shot_type in SHOT_TYPES else "other"
    ctx["events"] = normalize_events(data.get("events"), now, tz_name, limit=8)
    ctx["source_url"] = ""
    context._ground_location(ctx)
    context._verify_urls(ctx)
    context.fill_item_lines(ctx)

    debug = dict(usage=dict(final={**usage, "model": context.stage_label(stg)}, observe=observe_usage,
                            ledger=llm.ledger()),
                 research=dict(rounds=0, queries=[], findings=[], errors=[]),
                 vision=dict(used=bool(observations), mode="on"))
    context.save_context(storage.results_dir, job_id, ctx, debug, observations)
    first = ctx["events"][0] if ctx["events"] else None
    progress("placesAndBoard", 1.0, title=ctx["title"],
             chips=[first["title"][:40]] if first else [], detail=f"{len(ctx['events'])} event(s)" if first else "no event")
    return job_id
