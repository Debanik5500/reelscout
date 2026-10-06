"""Monthly reel quota per device, with plans read from plans.json.

A reel counts when it starts processing and is given back if it fails or turns out to be a duplicate, so the
user only spends quota on reels that actually landed. Reels that are already processed never count again.

Requests without an X-Device-Id header (curl, scripts, admin) are not metered. Real users must send one; before
a public launch this should become real authentication.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

PLANS_FILE = Path(__file__).resolve().parent.parent / "plans.json"
QUOTA_FILE = Path(os.path.expanduser("~/ReelScoutData/quota.json"))
_lock = threading.RLock()


class QuotaExceeded(Exception):
    def __init__(self, status: dict):
        super().__init__("monthly reel limit reached")
        self.status = status


def load_plans() -> dict:
    return json.loads(PLANS_FILE.read_text(encoding="utf-8"))


def _plan(plan_id: str, plans: dict) -> dict:
    by_id = {p["id"]: p for p in plans["plans"]}
    return by_id.get(plan_id) or by_id[plans["free_plan"]]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _month(now: datetime) -> str:
    return now.strftime("%Y-%m")


def _resets_at(now: datetime) -> str:
    year, month = (now.year + 1, 1) if now.month == 12 else (now.year, now.month + 1)
    return datetime(year, month, 1, tzinfo=timezone.utc).isoformat()


def _read() -> dict:
    try:
        return json.loads(QUOTA_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"devices": {}}


def _write(data: dict) -> None:
    QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUOTA_FILE.with_name(QUOTA_FILE.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(QUOTA_FILE)


def _unlimited(device_id: str) -> bool:
    return device_id in {d.strip() for d in os.getenv("REELSCOUT_UNLIMITED_DEVICES", "").split(",") if d.strip()}


def status(device_id: str | None) -> dict:
    """Plan, limit, used and reset date for a device (None limit = unmetered)."""
    plans = load_plans()
    now = _now()
    if not device_id or _unlimited(device_id) or os.getenv("REELSCOUT_QUOTA", "on").lower() == "off":
        return dict(plan="unmetered", plan_name="Unmetered", limit=None, used=0, remaining=None,
                    resets_at=_resets_at(now))
    with _lock:
        device = _read()["devices"].get(device_id, {})
    plan = _plan(device.get("plan") or plans["free_plan"], plans)
    used = sum(1 for e in device.get("usage", []) if e["month"] == _month(now))
    limit = int(plan["monthly_reels"])
    return dict(plan=plan["id"], plan_name=plan["name"], limit=limit, used=used,
                remaining=max(limit - used, 0), resets_at=_resets_at(now))


def reserve(device_id: str | None, job_id: str) -> dict:
    """Count a reel against the device's month. Idempotent per job; raises QuotaExceeded at the limit."""
    current = status(device_id)
    if current["limit"] is None:
        return current
    now = _now()
    with _lock:
        data = _read()
        device = data["devices"].setdefault(device_id, {"usage": []})
        if any(e["job"] == job_id and e["month"] == _month(now) for e in device["usage"]):
            return status(device_id)
        if current["used"] >= current["limit"]:
            raise QuotaExceeded(current)
        device["usage"].append({"job": job_id, "month": _month(now), "ts": now.isoformat()})
        device["usage"] = [e for e in device["usage"] if e["month"] >= _month(now)[:4] + "-01"][-500:]
        _write(data)
    return status(device_id)


def refund(device_id: str | None, job_id: str) -> None:
    """Give a reel back (it failed, or turned out to be one we already had)."""
    if not device_id:
        return
    now = _now()
    with _lock:
        data = _read()
        device = data["devices"].get(device_id)
        if not device:
            return
        kept = [e for e in device["usage"] if not (e["job"] == job_id and e["month"] == _month(now))]
        if len(kept) != len(device["usage"]):
            device["usage"] = kept
            _write(data)
