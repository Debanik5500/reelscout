"""Tiny LLM client. No SDKs: one OpenAI-compatible path + Anthropic's native path.

Switching from a free tier to a frontier model is an env-var change, not a code change.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time

import requests

from .config import Settings


logger = logging.getLogger("reelscout.llm")

# Waits longer than this are an exhausted quota (e.g. a daily token cap), not a per-minute blip.
LONG_WAIT_SECONDS = 75.0

_ledger = threading.local()  # per-thread record of every call made while processing one reel


class LLMError(RuntimeError):
    pass


class RateLimited(LLMError):
    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class _KeyPool:
    """Round-robin over several API keys for one provider. A key that hits a long rate limit (an exhausted
    quota) is skipped until it resets; when every key is blocked the caller falls back to another provider.

    Note: providers such as Groq apply limits per ORGANIZATION, so keys from the same account share one quota.
    Rotation only adds capacity when the keys belong to different organizations."""

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.blocked: dict[str, float] = {}
        self._next = 0
        self._lock = threading.Lock()

    def acquire(self) -> str | None:
        with self._lock:
            now = time.time()
            for _ in range(len(self.keys)):
                key = self.keys[self._next % len(self.keys)]
                self._next += 1
                if self.blocked.get(key, 0.0) <= now:
                    return key
            return None

    def block(self, key: str, seconds: float) -> None:
        with self._lock:
            self.blocked[key] = time.time() + min(seconds, 3600.0)

    def number(self, key: str) -> int:
        return self.keys.index(key) + 1


_pools: dict[tuple, _KeyPool] = {}
_pools_lock = threading.Lock()


def _api_keys(settings: Settings) -> list[str]:
    """All keys for the provider. Groq reads GROQ_API_KEYS (comma or space separated) plus GROQ_API_KEY."""
    preset = settings.preset or {}
    if not preset.get("key_env"):
        return [""]  # e.g. a local Ollama server
    keys = []
    if settings.llm == "groq":
        keys += [k for k in re.split(r"[,\s;]+", os.getenv("GROQ_API_KEYS", "")) if k]
    keys.append(settings.api_key)
    return list(dict.fromkeys(k for k in keys if k))


def _pool_for(settings: Settings) -> _KeyPool:
    keys = tuple(_api_keys(settings))
    with _pools_lock:
        if (settings.llm, keys) not in _pools:
            _pools[(settings.llm, keys)] = _KeyPool(list(keys))
        return _pools[(settings.llm, keys)]


def _auth(headers: dict, settings: Settings, key: str) -> None:
    if settings.llm == "anthropic":
        headers["x-api-key"] = key
    elif key:
        headers["Authorization"] = f"Bearer {key}"


def reset_ledger() -> None:
    _ledger.calls = {}


def ledger() -> dict[str, dict]:
    """provider:model -> {in, out, calls} for every LLM call since reset_ledger() on this thread."""
    return dict(getattr(_ledger, "calls", {}))


def _record(model_label: str, usage: dict) -> None:
    calls = getattr(_ledger, "calls", None)
    if calls is None:
        calls = _ledger.calls = {}
    row = calls.setdefault(model_label, {"in": 0, "out": 0, "calls": 0})
    row["in"] += usage.get("in", 0)
    row["out"] += usage.get("out", 0)
    row["calls"] += usage.get("calls", 1)


def _fallback(settings: Settings) -> Settings | None:
    """Provider to switch to when the main one is out of quota. Defaults to xAI when its key is set.
    REELSCOUT_FALLBACK_LLM=off disables it; REELSCOUT_FALLBACK_MODEL picks the model."""
    name = os.getenv("REELSCOUT_FALLBACK_LLM", "xai" if os.getenv("XAI_API_KEY") else "").lower()
    if not name or name == "off" or name == settings.llm:
        return None
    model = os.getenv("REELSCOUT_FALLBACK_MODEL", "grok-3" if name == "xai" else "")
    fb = Settings(llm=name, model=model, vision_model=model, timeout=settings.timeout)
    return fb if fb.preset and (fb.api_key or not fb.preset["key_env"]) else None


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a:b + 1])
        raise LLMError(f"model did not return JSON: {text[:200]!r}")


def _b64(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()


_ollama_checked: set = set()


def ollama_preflight(model: str, base_url: str) -> None:
    """Fail in 1 second with a useful message instead of timing out after minutes."""
    if (base_url, model) in _ollama_checked:
        return
    root = base_url.removesuffix("/v1")
    try:
        tags = requests.get(root + "/api/tags", timeout=3).json().get("models", [])
    except Exception:
        raise LLMError("Ollama is not reachable at " + root + " -- start it with `ollama serve` (or open the Ollama app)")
    names = [m.get("name", "") for m in tags]
    if not any(n == model or n.split(":")[0] == model or n == model + ":latest" for n in names):
        raise LLMError(f"Ollama model '{model}' is not pulled. Run: ollama pull {model}   (installed: {', '.join(names) or 'none'})")
    _ollama_checked.add((base_url, model))


def _complete(system: str, user: str, settings: Settings, image_paths: list[str] | None = None,
              max_tokens: int | None = None) -> tuple[dict, dict]:
    p = settings.preset
    if not p:
        raise LLMError(f"unknown llm '{settings.llm}'")
    model = (settings.vision_model if image_paths and settings.vision_model else settings.model) or p["model"]
    if p["key_env"] and not _api_keys(settings):
        raise LLMError(f"{p['key_env']} is not set")
    pool = _pool_for(settings)
    key = pool.acquire()
    if key is None:  # every key is inside a long rate-limit window
        raise RateLimited("all API keys are rate limited", min(pool.blocked.values()) - time.time())
    images = image_paths or []
    if settings.llm == "ollama":
        ollama_preflight(model, p["base_url"])
    timeout = settings.timeout or (600 if settings.llm == "ollama" else 120)

    if settings.llm == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(i)}} for i in images]
        content.append({"type": "text", "text": user})
        url = p["base_url"] + "/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": model, "max_tokens": max_tokens or 1500, "temperature": 0, "system": system,
                "messages": [{"role": "user", "content": content}]}
    else:
        content = [{"type": "text", "text": user}] + [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _b64(i)}} for i in images]
        url = p["base_url"] + "/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "content-type": "application/json"}
        body = {"model": model, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": content if images else user}]}
        if settings.llm == "openai" and not re.match(r"^gpt-4", model):
            body["max_completion_tokens"] = max(max_tokens or 0, 4000)   # gpt-5.x / o-series reason silently and reject custom temperature
        else:
            body["temperature"] = 0
            body["max_tokens"] = max_tokens or 1200

    last = None
    for attempt in range(5 + len(pool.keys)):  # free tiers rate-limit; back off, or move to the next key
        _auth(headers, settings, key)
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code in (429, 503):
            last = r.text[:200]
            # Honour the server's Retry-After (per-minute token limits reset within ~60s); else back off.
            try:
                wait = float(r.headers.get("retry-after", ""))
            except ValueError:
                wait = 2 ** attempt * 2
            if r.status_code == 429 and wait > LONG_WAIT_SECONDS:
                # This key's quota is used up. Try the next key straight away; with none left, let the caller
                # switch provider instead of waiting.
                pool.block(key, wait)
                next_key = pool.acquire()
                if next_key is None:
                    raise RateLimited(r.text[:300], wait)
                logger.warning("%s key #%d is out of quota (retry in %.0fs); trying key #%d", settings.llm,
                               pool.number(key), wait, pool.number(next_key))
                key = next_key
                continue
            time.sleep(min(max(wait, 1.0), LONG_WAIT_SECONDS))
            continue
        if r.status_code == 400 and attempt < 4:
            msg = r.text
            if "temperature" in msg and "temperature" in body:
                body.pop("temperature"); continue
            if "max_completion_tokens" in msg and "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens"); continue
            if "max_tokens" in msg and "max_completion_tokens" in body:
                body["max_tokens"] = body.pop("max_completion_tokens"); continue
        if r.status_code >= 400:
            raise LLMError(f"{r.status_code}: {r.text[:300]}")
        j = r.json()
        if settings.llm == "anthropic":
            text = "".join(b.get("text", "") for b in j["content"])
            usage = {"in": j["usage"]["input_tokens"], "out": j["usage"]["output_tokens"], "calls": 1}
        else:
            text = j["choices"][0]["message"]["content"]
            u = j.get("usage", {})
            usage = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0), "calls": 1}
        _record(f"{settings.llm}:{model}", usage)
        return _parse_json(text), usage
    raise LLMError(f"rate limited: {last}")


_blocked_until: dict[str, float] = {}  # provider -> time its quota resets, so we stop sending doomed requests


def complete_json(system: str, user: str, settings: Settings, image_paths: list[str] | None = None,
                  max_tokens: int | None = None) -> tuple[dict, dict]:
    """One JSON completion. If the provider's quota is exhausted, uses the fallback provider until it resets."""
    fallback = _fallback(settings)
    if fallback is not None and time.time() < _blocked_until.get(settings.llm, 0.0):
        return _complete(system, user, fallback, image_paths, max_tokens)
    try:
        return _complete(system, user, settings, image_paths, max_tokens)
    except RateLimited as exc:
        if fallback is None:
            raise
        _blocked_until[settings.llm] = time.time() + min(exc.retry_after, 3600.0)
        logger.warning("%s is rate limited (retry in %.0fs); using %s:%s until it resets", settings.llm,
                       exc.retry_after, fallback.llm, fallback.model)
        return _complete(system, user, fallback, image_paths, max_tokens)
