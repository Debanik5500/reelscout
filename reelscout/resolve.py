"""Stage 9: turn unresolved clues ("grey dashboard, sidebar: Projects / Deployments") into a name.

clues -> search queries -> open-source search (+ optional reverse image search / geolocation)
      -> one adjudication call that may only pick what the results actually support.

Works with llm=none too (queries = the clues themselves; you just get the raw hits back to read).
"""
from __future__ import annotations

from . import imagesearch, reader, search, verify
from .config import Settings
from .extract import Candidate
from .llm import complete_json
from .reason import mentions_to_candidates

QUERY_SYSTEM = """You turn clues from a short video into web search queries that would identify the specific product, software tool, repo, brand or place it shows.
Use distinctive nouns and exact on-screen strings; drop generic words.

TARGET RULES:
- "github": ONLY when clues contain a clear owner/repo pattern (e.g., "username/repo", "github.com/user/repo") OR a specific known tool name that is almost certainly a GitHub repo (e.g., "langchain", "autogpt", "ollama"). Generic terms like "vibe coded app", "AI tool", "debugging cheat" are NOT github targets.
- "place": when clues indicate a named venue, landmark, or address (e.g., "Eiffel Tower", "Central Park NYC", "restaurant name")
- "web": everything else — product names, brand names, website names, app names, generic descriptions

GITHUB SEARCH STRATEGY:
- For GitHub repos, use descriptive search terms: tool name + "github", "open source", "repo", main features
- If the video shows a GitHub page but URL isn't clear, search for: "github [tool name] [key features]"
- Example: "AI YouTube studio automation free tier Gemini Pexels" → search "github AI YouTube studio automation"

Return JSON only: {"queries": [{"q": str, "target": "web|github|place"}]}  (max 4)"""

ADJUDICATE_SYSTEM = """You decide what a short video is about, using ONLY the search results provided.
Return a mention only if a result's title/snippet clearly matches MULTIPLE clues. A result that matches one generic word is not a match.

GITHUB REPO PRIORITY:
- If a GitHub repo's name/description matches the video's topic (e.g., video describes "AI YouTube studio automation" and repo is named "youtube-agentic-ai-studio" with description about automating faceless channels), PROMOTE it with HIGH confidence (0.9+).
- Match on: tool name, key features, tech stack (Gemini, Pexels, Edge TTS, YouTube API), purpose (faceless channel, automation).
- Exact URL not needed — repo name + description match is sufficient.
- If multiple GitHub repos match, include all with appropriate confidence.

If nothing clearly matches, return an empty list -- an honest "unknown" is correct and preferred over a guess.
Return JSON only: {"mentions": [{"name": str, "kind": "website|github_repo|tool|package|product|brand|place|other", "url": str|null, "confidence": number, "evidence": "which result + which clues it matches"}], "note": str}"""


def fallback_queries(clues: list[str]) -> list[dict]:
    return [dict(q=" ".join(c.split()[:10]), target="web") for c in clues[:3]]


def make_queries(clues, summary, geo_hint, st: Settings, transcript: list = None, ocr_lines: list = None, caption: str = "") -> tuple[list[dict], dict]:
    if st.llm == "none":
        return fallback_queries(clues), {}
    
    # Build context from transcript, OCR, and caption
    context_parts = [f"SUMMARY: {summary}\nCLUES: {clues}\nREGION HINT: {geo_hint or 'none'}"]
    
    if transcript:
        tx = " ".join(s["text"] for s in transcript)[:2000]
        if tx:
            context_parts.append(f"TRANSCRIPT: {tx}")
    
    if ocr_lines:
        ocr_text = "\n".join(f"[{l['t']}s] {l['text'][:120]}" for l in ocr_lines[:20])
        context_parts.append(f"ON-SCREEN TEXT (OCR): {ocr_text}")
    
    if caption:
        context_parts.append(f"CAPTION: {caption[:500]}")
    
    prompt = "\n\n".join(context_parts)
    
    try:
        data, usage = complete_json(QUERY_SYSTEM, prompt, st)
        qs = [q for q in data.get("queries", []) if q.get("q")][:4]
        return (qs or fallback_queries(clues)), usage
    except Exception:
        return fallback_queries(clues), {}


def resolve(clues: list[str], summary: str, frame_paths: list[str], st: Settings, transcript: list = None, ocr_lines: list = None, caption: str = "") -> dict:
    """Returns {candidates, hits, image_hits, geo, errors, usage}."""
    st = st.for_stage("RESEARCH")
    out = dict(candidates=[], hits=[], image_hits=[], geo=None, errors=[], usage={},
               model=f"{st.llm}:{st.model or (st.preset or {}).get('model', '')}")

    geo_hint = None
    if st.geo and frame_paths:
        out["geo"] = imagesearch.geolocate(frame_paths)
        if out["geo"]:
            geo_hint = out["geo"]["guesses"][0]["place"]
        else:
            out["errors"].append("geoclip unavailable (pip install geoclip) or returned nothing")

    queries, usage = make_queries(clues, summary, geo_hint, st, transcript, ocr_lines, caption)
    for k, v in usage.items():
        out["usage"][k] = out["usage"].get(k, 0) + v
    for q in queries:
        if q.get("target") == "github":
            hits = [dict(title=f"{h['full_name']} - {h['description'] or ''}", url=h["url"], snippet=f"{h['stars']} stars")
                    for h in verify.github_search_text(q["q"])]
        elif q.get("target") == "place":
            g = verify.geocode(q["q"])
            hits = [dict(title=g["display_name"], url=g["map_url"], snippet="OpenStreetMap")] if g.get("ok") else []
        else:
            hits = search.web_search(q["q"], 3)  # Limit to 3 results for speed
        out["hits"] += [dict(h, query=q["q"]) for h in hits]

    if st.image_search and frame_paths:
        for p in frame_paths[:2]:
            hits, errs = imagesearch.reverse_search(p)
            out["image_hits"] += hits
            out["errors"] += errs
            if hits:
                break

    if st.llm != "none" and (out["hits"] or out["image_hits"]):
        lines = [f"- [{h['query']}] {h['title']} | {h['url']} | {h.get('snippet', '')[:160]}" for h in out["hits"][:16]]
        for h in [h for h in out["hits"] if h["url"].startswith("http") and "github.com" not in h["url"]][:3]:
            txt = reader.page_text(h["url"])                 # read the top pages, not just their titles
            if txt:
                lines.append(f"  PAGE EXCERPT of {h['url']}: {txt}")
        lines += [f"- [image:{h['engine']}] {h['title']} | {h['url']}" for h in out["image_hits"][:8]]
        prompt = (f"SUMMARY: {summary}\nCLUES: {clues}\nGEOLOCATION GUESS (coarse, region-level only): "
                  f"{out['geo'] and out['geo']['guesses']}\n\nSEARCH RESULTS:\n" + "\n".join(lines))
        try:
            data, usage = complete_json(ADJUDICATE_SYSTEM, prompt, st)
            for k, v in usage.items():
                out["usage"][k] = out["usage"].get(k, 0) + v
            src = "image_search" if out["image_hits"] and not out["hits"] else "search"
            out["candidates"] = mentions_to_candidates(data, src)
        except Exception as e:
            out["errors"].append(f"adjudication failed: {e}")

    if out["geo"]:   # coarse place guess is reported as a low-confidence candidate, never above what text says
        g = out["geo"]["guesses"][0]
        out["candidates"].append(Candidate("place", g["place"], None, confidence=min(0.45, 0.2 + g["prob"]),
                                           evidence=[dict(source="geoclip", text=f"region-level guess, p={g['prob']}", t=None)]))
    return out
