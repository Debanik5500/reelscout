"""Web search with free/open-source backends, tried in order until one returns results:

  1. SearXNG  (open source metasearch you host yourself; set SEARXNG_URL, JSON output must be enabled in its settings)
  2. ddgs     (pip install ddgs; open-source multi-engine scraper with automatic backend failover)
  3. DuckDuckGo HTML scrape (no dependency; most likely to get rate-limited)

All return [{title, url, snippet}]. Scraping-based backends are fine for testing; for a production app
use a proper search API or your own SearXNG instance.
"""
from __future__ import annotations

import os
import re
import urllib.parse

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; reelscout/0.1)"}


def _searxng(query: str, n: int) -> list[dict]:
    base = os.getenv("SEARXNG_URL", "").rstrip("/")
    if not base:
        return []
    try:
        r = requests.get(base + "/search", params=dict(q=query, format="json"), headers=UA, timeout=8)
        r.raise_for_status()
        return [dict(title=i.get("title", ""), url=i.get("url", ""), snippet=i.get("content", ""))
                for i in r.json().get("results", [])[:n]]
    except Exception:
        return []


def _ddgs(query: str, n: int) -> list[dict]:
    from ddgs import DDGS
    try:
        with DDGS(timeout=10) as ddgs:
            return [dict(title=r.get("title", ""), url=r.get("href", ""), snippet=r.get("body", ""))
                    for r in ddgs.text(query, max_results=n)]
    except Exception:
        return []


def _ddg_html(query: str, n: int) -> list[dict]:
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=UA, timeout=8)
        out = []
        for m in re.finditer(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', r.text, re.S):
            href = m.group(1)
            if "uddg=" in href:
                href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
            out.append(dict(title=re.sub(r"<.*?>", "", m.group(2)).strip(), url=href, snippet=""))
            if len(out) >= n:
                break
        return out
    except Exception:
        return []


BACKENDS = [_searxng, _ddgs, _ddg_html]


def web_search(query: str, n: int = 5) -> list[dict]:
    for backend in BACKENDS:
        try:
            res = backend(query, n)
            if res:
                return res
        except Exception:
            continue
    return []
