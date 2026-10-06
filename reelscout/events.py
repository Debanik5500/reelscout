"""Dated things (events, appointments, tickets, deadlines) found in any item: reel, post, screenshot, web page.

The model reads "YYYY-MM-DDTHH:MM" wall-clock times; this module attaches the right timezone, resolves a missing
year to the next occurrence, drops implausible dates, and flags anything the user should double-check.
Shared by every pipeline so an event is treated the same wherever it came from.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Prompt rules shared by every pipeline that can find events (kept in one place so they never drift apart).
EVENT_RULES = """- events: dated things a person may want a reminder or calendar entry for: events, matches, shows, appointments,
  reservations, flights, deadlines, sales ending. One entry each (up to {limit}). Fields: title, start, end,
  all_day, tz, year_inferred, location (venue or address as written), notes (price, booking reference), confidence 0-1.
  - start/end are LOCAL wall-clock times as written: "YYYY-MM-DDTHH:MM" (24-hour). Only a date: "YYYY-MM-DD" and
    all_day true. Leave end empty when not stated.
  - Resolve relative words ("tomorrow", "this Saturday") from the CURRENT date given.
  - tz: an IANA timezone name ONLY if the source states a different timezone (e.g. "8pm PST" -> "America/Los_Angeles"),
    otherwise empty (the user's timezone is used).
  - year_inferred: true when the year is not shown and you assumed it.
  - Never invent a date, time or venue. No date means no event. A schedule or fixture list: include the upcoming ones."""


def zone(name: str | None) -> ZoneInfo | None:
    try:
        return ZoneInfo(name) if name else None
    except (ZoneInfoNotFoundError, ValueError):
        return None


def _parse(value: str) -> tuple[datetime, bool] | None:
    """(naive datetime, date_only) from 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM[:SS]'."""
    text = (value or "").strip().replace(" ", "T")
    if not text:
        return None
    try:
        if len(text) == 10:
            return datetime.fromisoformat(text), True
        # Structured data (JSON-LD) carries an offset or "Z"; keep it so the real timezone is not lost.
        return datetime.fromisoformat(text.replace("Z", "+00:00")), False
    except ValueError:
        return None


def normalize_event(raw: dict, now: datetime, user_tz: str) -> dict | None:
    """One validated event, or None when it has no usable date. `now` must be timezone-aware."""
    parsed = _parse(raw.get("start", ""))
    title = " ".join(str(raw.get("title") or "").split())[:120]
    if not parsed or not title:
        return None
    start_local, date_only = parsed
    explicit = start_local.tzinfo  # an offset written in the value wins over any guess
    tzinfo = explicit or zone(raw.get("tz")) or zone(user_tz) or timezone.utc
    all_day = bool(raw.get("all_day")) or date_only
    inferred = bool(raw.get("year_inferred"))

    start = start_local if explicit else start_local.replace(tzinfo=tzinfo)
    # A date read without a year that landed in the past means the next occurrence.
    while inferred and start < now - timedelta(days=1) and start.year < now.year + 2:
        start = start.replace(year=start.year + 1)
    if not (now - timedelta(days=400) <= start <= now + timedelta(days=1100)):
        return None  # an implausible date is more likely a misread than an event

    end_parsed = _parse(raw.get("end", ""))
    end = (end_parsed[0] if end_parsed[0].tzinfo else end_parsed[0].replace(tzinfo=tzinfo)) if end_parsed else None
    if end and inferred:
        end = end.replace(year=start.year)
    if end and end <= start:
        end = None

    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence") or 0)))
    except (TypeError, ValueError):
        confidence = 0.0
    if all_day:
        start = start.replace(hour=0, minute=0)
    return dict(
        id=hashlib.sha1(f"{title.lower()}|{start.isoformat()}".encode()).hexdigest()[:10],
        title=title,
        start=start.isoformat(timespec="minutes"),
        end=end.isoformat(timespec="minutes") if end else "",
        all_day=all_day,
        timezone=getattr(tzinfo, "key", None) or tzinfo.tzname(None) or "UTC",
        location=" ".join(str(raw.get("location") or "").split())[:160],
        notes=" ".join(str(raw.get("notes") or "").split())[:240],
        confidence=confidence,
        needs_check=inferred or all_day or confidence < 0.7,
    )


def normalize_events(raw_events, now: datetime, user_tz: str, limit: int = 8) -> list[dict]:
    """Validated, de-duplicated events sorted by start."""
    seen, out = set(), []
    for raw in (raw_events or [])[: limit * 2]:
        event = normalize_event(raw, now, user_tz) if isinstance(raw, dict) else None
        if event and event["id"] not in seen:
            seen.add(event["id"])
            out.append(event)
    out.sort(key=lambda e: e["start"])
    return out[:limit]
