"""Fetch a web page as clean text. Jina Reader (r.jina.ai, keyless) first, plain HTTP as fallback.

Used to let the final adjudication read the top search hits instead of judging from titles alone.
(Technique borrowed from the Agent-Reach project's tool table; no dependency on it.)
"""
from __future__ import annotations

import re

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; reelscout/0.1)"}


def _jina(url: str) -> str:
    r = requests.get("https://r.jina.ai/" + url, headers=UA, timeout=5)  # Reduced from 15s
    r.raise_for_status()
    return r.text


def _direct(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=3)  # Reduced from 10s
    r.raise_for_status()
    t = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", r.text)
    return re.sub(r"<[^>]+>", " ", t)


def page_text(url: str, max_chars: int = 700) -> str:
    for fetch in (_jina, _direct):
        try:
            t = re.sub(r"\s+", " ", fetch(url)).strip()
            if len(t) > 80:
                return t[:max_chars]
        except Exception:
            continue
    return ""
