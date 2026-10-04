"""Tiny LLM client. No SDKs: one OpenAI-compatible path + Anthropic's native path.

Switching from a free tier to a frontier model is an env-var change, not a code change.
"""
from __future__ import annotations

import base64
import json
import re
import time

import requests

from .config import Settings


class LLMError(RuntimeError):
    pass


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


def complete_json(system: str, user: str, settings: Settings, image_paths: list[str] | None = None) -> tuple[dict, dict]:
    p = settings.preset
    if not p:
        raise LLMError(f"unknown llm '{settings.llm}'")
    model = (settings.vision_model if image_paths and settings.vision_model else settings.model) or p["model"]
    key = settings.api_key
    if p["key_env"] and not key:
        raise LLMError(f"{p['key_env']} is not set")
    images = image_paths or []
    if settings.llm == "ollama":
        ollama_preflight(model, p["base_url"])
    timeout = settings.timeout or (600 if settings.llm == "ollama" else 120)

    if settings.llm == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _b64(i)}} for i in images]
        content.append({"type": "text", "text": user})
        url = p["base_url"] + "/v1/messages"
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
        body = {"model": model, "max_tokens": 1500, "temperature": 0, "system": system,
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
            body["max_completion_tokens"] = 4000   # gpt-5.x / o-series reason silently and reject custom temperature
        else:
            body["temperature"] = 0
            body["max_tokens"] = 1200

    last = None
    for attempt in range(4):  # free tiers rate-limit; back off instead of failing
        r = requests.post(url, headers=headers, json=body, timeout=timeout)
        if r.status_code in (429, 503):
            last = r.text[:200]
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 400 and attempt < 3:
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
        return _parse_json(text), usage
    raise LLMError(f"rate limited: {last}")
