"""Bounded research loop: plan -> search -> verify, repeated at most `max_rounds` times.

The planner decides what (if anything) needs external lookup; many reels need none and the loop
exits with zero searches. The verifier only accepts a candidate the search results actually support,
so a vaguely similar hit never becomes "the answer".
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from . import reader, search, verify
from .config import Settings
from .llm import complete_json

MAX_ROUNDS = 3
MAX_TASKS_PER_ROUND = 3

# On-disk cache for search/geocode hits and page excerpts. Only non-empty results are stored, so a transient
# rate-limit or network failure is retried next time instead of being remembered.
CACHE_DIR = Path(os.getenv("REELSCOUT_SEARCH_CACHE", ".cache/research"))
CACHE_TTL_SECONDS = float(os.getenv("REELSCOUT_SEARCH_CACHE_TTL_DAYS", "7")) * 86400

PLAN_SYSTEM = """You plan external research for one short social-media video (reel).
You get observed evidence (caption, transcript, on-screen text, visual observations, candidates already found) and any findings from earlier rounds.
Decide which facts would materially benefit from a web lookup, e.g. which restaurant, product, landmark, repo, tool, film, person, event or artwork this is; whether a factual claim is true; what an unfamiliar name, acronym, symbol or cultural/historical reference means; or official information about something mentioned.
Do not assume the reel is about a location. Search only when external information would materially improve the result, not merely because something is unknown.
Rules:
- Combine related clues (visual + on-screen text + transcript + caption) into ONE specific query instead of several redundant ones. Add alternative queries only when the first is ambiguous or there are several plausible interpretations.
- If the evidence is too thin to form a useful query, return an empty task list.
- When the reel is about a specific venue (a stay, restaurant, shop, attraction) and the evidence only names a city or area, do NOT stop at the city: plan a lookup for the venue or brand itself, using its name or the uploader's account handle plus the area and any nearby landmarks mentioned (use target "place" only for a named place or address; use "web" to find the venue's official page or listing).
- Do NOT research things already supported with confidence >= 0.8, generic chatter, memes, or reels with nothing identifiable. Return an empty task list for those.
- Never ask a search to "find the location"; ask about a specific claim backed by evidence, and put the distinctive strings in the query (exact on-screen text, place names heard, brand + model).
- Use an earlier round's contradictions to change the query; do not repeat a query.
- target: "place" (geocode a named venue/landmark/area), "github" (clear owner/repo or well-known repo name only), "web" (everything else).
Return JSON only: {"tasks": [{"goal": str, "query": str, "target": "web|place|github", "priority": "high|medium|low"}]}  (max 3)"""

VERIFY_SYSTEM = """You verify candidate identifications for a short video against search results.
You get the original evidence, the research goals, and search results. For each goal decide:
- "supported": the results clearly match MULTIPLE independent clues in the evidence.
- "uncertain": plausible but only one clue matches or results conflict.
- "contradicted": the results clearly conflict with the evidence.
A result that matches one generic word is not support. Prefer "uncertain" over a guess.
Return JSON only:
{"findings": [{"goal": str, "candidate": str, "kind": "place|product|website|github_repo|tool|person|other", "url": str|null, "city": str, "country": str,
  "verdict": "supported|uncertain|contradicted", "confidence": number, "evidence": [str], "contradictions": [str], "source_urls": [str]}],
 "need_more": boolean}
need_more is true only when a high-priority goal is unresolved and a different query could plausibly resolve it."""


def _cache_file(kind: str, key: str) -> Path:
    digest = hashlib.sha256(f"{kind}|{key}".encode("utf-8")).hexdigest()[:40]
    return CACHE_DIR / f"{kind}-{digest}.json"


def _cache_get(kind: str, key: str):
    path = _cache_file(kind, key)
    try:
        if time.time() - path.stat().st_mtime > CACHE_TTL_SECONDS:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cache_put(kind: str, key: str, value) -> None:
    if not value:
        return
    path = _cache_file(kind, key)
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def _run_task(task: dict) -> list[dict]:
    q, target = task["query"].strip(), task.get("target", "web")
    key = f"{target}|{' '.join(q.lower().split())}"
    cached = _cache_get("hits", key)
    if cached is not None:
        return cached
    hits = _search(q, target)
    _cache_put("hits", key, hits)
    return hits


def _page_text(url: str) -> str:
    cached = _cache_get("page", url)
    if cached is not None:
        return cached
    text = reader.page_text(url)
    _cache_put("page", url, text)
    return text


def _search(q: str, target: str) -> list[dict]:
    if target == "github":
        return [dict(title=f"{h['full_name']} - {h['description'] or ''}", url=h["url"], snippet=f"{h['stars']} stars")
                for h in verify.github_search_text(q)]
    if target == "place":
        g = verify.geocode(q)
        if not g.get("ok"):
            return []
        return [dict(title=g["display_name"], url=g["map_url"], snippet=f"OpenStreetMap lat={g['lat']} lon={g['lon']}",
                     lat=g["lat"], lon=g["lon"])]
    return search.web_search(q, 4)


def _format_results(results: list[tuple[dict, list[dict]]]) -> str:
    lines: list[str] = []
    for task, hits in results:
        lines.append(f"GOAL: {task['goal']} | QUERY: {task['query']} ({task.get('target', 'web')})")
        if not hits:
            lines.append("  (no results)")
        for h in hits[:5]:
            lines.append(f"  - {h['title']} | {h['url']} | {h.get('snippet', '')[:160]}")
        for h in [h for h in hits if h["url"].startswith("http") and "github.com" not in h["url"]][:2]:
            if h.get("snippet") != "OpenStreetMap":
                txt = _page_text(h["url"])
                if txt:
                    lines.append(f"  PAGE EXCERPT {h['url']}: {txt}")
    return "\n".join(lines)


def research(evidence: str, st: Settings, max_rounds: int = MAX_ROUNDS) -> dict:
    """Returns {findings, sources, geo, rounds, queries, errors, usage}. Findings are supported/uncertain only."""
    st = st.for_stage("RESEARCH")
    out: dict = dict(findings=[], sources=[], geo={}, rounds=0, queries=[], errors=[], usage={})
    if st.llm == "none":
        return out

    def add_usage(u: dict) -> None:
        for k, v in (u or {}).items():
            if isinstance(v, (int, float)):
                out["usage"][k] = out["usage"].get(k, 0) + v

    seen: set[str] = set()
    for _ in range(max(1, min(max_rounds, MAX_ROUNDS))):
        prior = "\n".join(
            f"- {f['goal']} -> {f['candidate']} [{f['verdict']} {f['confidence']}] contradictions={f['contradictions']}"
            for f in out["findings"]) or "(none)"
        try:
            plan, u = complete_json(PLAN_SYSTEM, f"EVIDENCE:\n{evidence}\n\nEARLIER FINDINGS:\n{prior}\n\nQUERIES ALREADY RUN: {out['queries']}", st)
            add_usage(u)
        except Exception as e:
            out["errors"].append(f"plan failed: {e}")
            break
        tasks = [t for t in plan.get("tasks", []) if t.get("query") and t["query"].lower() not in seen][:MAX_TASKS_PER_ROUND]
        if not tasks:
            break
        results: list[tuple[dict, list[dict]]] = []
        for t in tasks:
            seen.add(t["query"].lower())
            out["queries"].append(t["query"])
            try:
                results.append((t, _run_task(t)))
            except Exception as e:
                out["errors"].append(f"search failed [{t['query']}]: {e}")
        out["rounds"] += 1
        if not any(hits for _, hits in results):
            continue
        try:
            ver, u = complete_json(VERIFY_SYSTEM, f"EVIDENCE:\n{evidence}\n\nSEARCH RESULTS:\n{_format_results(results)}", st)
            add_usage(u)
        except Exception as e:
            out["errors"].append(f"verify failed: {e}")
            break
        geo_by_goal = {t["goal"]: next((h for h in hits if "lat" in h), None) for t, hits in results}
        for f in ver.get("findings", []):
            f = dict(goal=f.get("goal", ""), candidate=(f.get("candidate") or "").strip(), kind=f.get("kind", "other"),
                     url=f.get("url"), city=f.get("city", ""), country=f.get("country", ""),
                     verdict=f.get("verdict", "uncertain"), confidence=float(f.get("confidence") or 0),
                     evidence=f.get("evidence") or [], contradictions=f.get("contradictions") or [],
                     source_urls=f.get("source_urls") or [])
            if not f["candidate"]:
                continue
            hit = geo_by_goal.get(f["goal"])
            if hit and f["verdict"] == "supported":
                f["lat"], f["lon"], f["map_url"] = float(hit["lat"]), float(hit["lon"]), hit["url"]
            out["findings"].append(f)
        for _, hits in results:
            out["sources"] += [dict(title=h["title"], url=h["url"]) for h in hits[:3] if h["url"].startswith("http")]
        if not ver.get("need_more"):
            break
    return out
