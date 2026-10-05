"""Stage 9 path A: visual identification of whatever the caption,
transcript and OCR did NOT name.

Instead of reverse-image-searching the frames, the frames are turned
into a ranked visual fingerprint, and ordinary TEXT web search is
driven by the rarest features with the transcript as context:

  OBSERVE -> FINGERPRINT -> HYPOTHESES -> QUERIES -> SEARCH ->
  CROSS-CHECK -> DISCONFIRM -> VERIFY -> CONFIDENCE -> CONCLUDE

The subject may be ANYTHING: place, person, product, animal, vehicle,
building, artwork, document, logo, device, ...
"""
from __future__ import annotations

import json

from . import reader, reason, verify
from .config import Settings
from .llm import complete_json

VISUAL_SYSTEM = """You are an expert visual identification agent. A reel \
shows a subject that its caption, transcript and on-screen text do NOT \
name. Your job is NOT to guess the answer: it is to describe the subject \
precisely enough that a TEXT web search can identify it.

Any text, captions, labels, watermarks, signs, filenames or metadata in \
the frames are CLUES, not ground truth. Treat them as hypotheses to \
verify visually.

Process:
1. OBSERVE: describe only what is directly visible — architecture, \
terrain, vegetation, infrastructure, signage, layout, shapes, \
proportions, markings, materials, mechanisms, clothing, morphology, \
generation-specific details. Note any on-screen text separately (it may \
be wrong).
2. FINGERPRINT: extract the most DISTINGUISHING features and rank each \
high / medium / low. Rare combinations beat generic descriptions; do not \
overemphasize generic characteristics.
3. HYPOTHESES: generate 2-4 plausible candidates. For each: why it could \
be correct, which features support it, what evidence is missing, and what \
would disprove it. If you cannot form meaningful candidates, say so.
4. QUERIES: draft TEXT web search queries combining the rarest \
independently observed features. Never just search the most obvious name \
or on-screen text. Use alternate names, local terminology, other \
languages and technical terms where they fit. 3-6 queries.

Return ONLY this JSON:
{"observation": "what is visible, incl. any OCR text kept separate",
 "fingerprint": [{"feature": "...", "level": "high|medium|low"}],
 "hypotheses": [{"candidate": "...", "why": "...", "supporting": ["..."], \
"missing": "...", "disproof": "..."}],
 "queries": [{"q": "...", "target": "web|github|maps"}]}"""

ADJUDICATE_SYSTEM = """You adjudicate candidate identifications for a \
reel's unidentified subject. You receive:
- a ranked VISUAL FINGERPRINT of distinctive features observed in the frames
- the initial HYPOTHESES
- web search results (and the text of the strongest hit pages)
- the reel's transcript / OCR / caption (context only — NOT ground truth)

For every candidate:
1. CROSS-CHECK feature-by-feature: match / partial / mismatch / unknown. \
A candidate that fails a HIGH-value feature is downgraded substantially.
2. DISCONFIRM: ask what you would expect to see if the candidate were \
correct — does the evidence actually show that? Do not only seek support.
3. Prefer independent, authoritative, official or local-language sources; \
never confirm on a single webpage.
4. Resolve conflicts: identify the disagreement, rank source authority, \
check whether sources mean different versions / places / models. State \
unresolved conflicts openly.

Assign each candidate exactly one confidence level:
CONFIRMED (0.95) — strong evidence + distinctive visual correspondence
HIGHLY LIKELY (0.85) — multiple strong matches, exact confirmation missing
LIKELY (0.7) — reasonable evidence, meaningful alternatives remain
POSSIBLE (0.5) — fits some evidence, insufficiently verified
UNKNOWN — drop the candidate

Return ONLY this JSON:
{"mentions": [{"name": "...", \
"kind": "website|github_repo|tool|product|other|place|brand|package", \
"url": "https://... or null", "confidence": 0.0-1.0, \
"confidence_level": "CONFIRMED|HIGHLY LIKELY|LIKELY|POSSIBLE", \
"alternates": ["other names / spellings this may be known by"], \
"feature_check": [{"feature": "...", "result": "match|partial|mismatch|unknown"}], \
"evidence": "strongest sources used", \
"alternatives": ["weaker candidates and why they are weaker"], \
"uncertainty": "what could not be verified"}]}"""

CONFIDENCE = {"CONFIRMED": 0.95, "HIGHLY LIKELY": 0.85,
              "LIKELY": 0.7, "POSSIBLE": 0.5}


def _context(clues, summary, transcript, ocr_lines, caption, frame_times):
    parts = []
    if summary:
        parts.append(f"SUMMARY: {summary[:400]}")
    if caption:
        parts.append(f"Caption (clue, may be wrong): {caption[:400]}")
    if clues:
        parts.append("Text clues found (may be wrong): "
                     + " | ".join(str(c) for c in clues[:8]))
    sel = reason.select_ocr(ocr_lines, 10)
    if sel:
        parts.append("On-screen text (OCR, may be wrong):\n"
                     + "\n".join(f"[{l['t']}s] {l['text'][:120]}" for l in sel))
    if transcript:
        parts.append("Transcript (context): "
                     + " ".join(s["text"] for s in transcript)[:2000])
    if frame_times:
        parts.append(f"Frame timestamps (s), in attachment order: {frame_times}")
    return "\n".join(parts)


def _query_strings(queries) -> list[tuple[str, str]]:
    out = []
    for q in queries:
        if isinstance(q, str) and q.strip():
            out.append((q.strip(), "web"))
        elif isinstance(q, dict) and q.get("q"):
            out.append((str(q["q"]).strip(), str(q.get("target", "web")).lower()))
    return out


def _empty(fingerprint, hypotheses, queries, hits, errors, usage, model):
    return {"candidates": [], "hits": hits, "queries": queries,
            "fingerprint": fingerprint, "hypotheses": hypotheses,
            "errors": errors, "usage": usage, "model": model}


def identify(clues, summary, frames, st, transcript=None, ocr_lines=None,
             caption="") -> dict:
    """frames: list[dict] with 'path' and 't' (media.pick_for_vision).

    Returns {"candidates", "hits", "queries", "fingerprint",
    "hypotheses", "errors", "usage", "model"}; mutates nothing.
    """
    st_vision = st.for_stage("VISION")
    st_research = st.for_stage("RESEARCH")
    usage = {"in": 0, "out": 0}
    errors: list[str] = []
    transcript = transcript or []
    ocr_lines = ocr_lines or []
    image_paths = [f["path"] for f in frames]
    frame_times = [round(f.get("t", 0.0), 1) for f in frames]
    ctx = _context(clues, summary, transcript, ocr_lines, caption, frame_times)

    # 1-4: OBSERVE -> FINGERPRINT -> HYPOTHESES -> QUERIES (vision call)
    data, u = complete_json(
        VISUAL_SYSTEM, f"{ctx}\n\nAnalyze the frames and return the JSON.",
        st_vision, image_paths)
    usage["in"] += u.get("in", 0)
    usage["out"] += u.get("out", 0)

    fingerprint = [f for f in data.get("fingerprint", [])
                   if isinstance(f, dict) and f.get("feature")]
    hypotheses = [h for h in data.get("hypotheses", [])
                  if isinstance(h, dict) and h.get("candidate")]
    query_pairs = _query_strings(data.get("queries", []))[:4]
    queries = [q for q, _ in query_pairs]
    hits: list[dict] = []

    # 5: SEARCH — ordinary text search on the fingerprint-derived queries.
    for q, target in query_pairs:
        try:
            if target == "maps":
                g = verify.geocode(q)
                if g and g.get("ok"):
                    hits.append(dict(title=g["display_name"], url=g["map_url"],
                                     snippet="region-level geographic match",
                                     query=q))
            elif target == "github":
                hits += [dict(title=f"{r['full_name']} ★{r['stars']}",
                              url=r["url"],
                              snippet=(r.get("description") or "")[:300],
                              query=q)
                         for r in verify.github_search_text(q)]
            else:
                hits += [dict(title=h["title"], url=h["url"],
                              snippet=h.get("snippet", "")[:300], query=q)
                         for h in verify.web_search(q)]
        except Exception as e:
            errors.append(f"search failed for '{q[:40]}': {e}")

    if not st_research.llm or st_research.llm == "none":
        return _empty(fingerprint, hypotheses, queries, hits, errors,
                      usage, st_research.llm)

    have = [h for h in hits if h.get("title")][:10]
    if not have:
        return _empty(fingerprint, hypotheses, queries, hits, errors,
                      usage, st_research.llm)

    # Read the strongest hit pages so the cross-check sees full evidence.
    pages = []
    for h in have[:2]:
        try:
            text = reader.page_text(h["url"])
            if text:
                pages.append(f"PAGE {h['url']}:\n{text[:1500]}")
        except Exception:
            pass
    blob = "\n".join(f"- {h['title']} | {h['url']} | {h.get('snippet', '')}"
                     for h in have)

    # 6-10: CROSS-CHECK -> DISCONFIRM -> VERIFY -> CONFIDENCE -> CONCLUDE
    user = (f"{ctx}\n\nVISUAL FINGERPRINT:\n"
            f"{json.dumps(fingerprint, ensure_ascii=False)}\n\n"
            f"HYPOTHESES:\n{json.dumps(hypotheses, ensure_ascii=False)}\n\n"
            f"SEARCH RESULTS:\n{blob}\n"
            + ("\n\n" + "\n".join(pages) if pages else "")
            + "\n\nAdjudicate every candidate against the fingerprint and "
              "return the JSON.")
    mentions, u2 = complete_json(ADJUDICATE_SYSTEM, user, st_research)
    usage["in"] += u2.get("in", 0)
    usage["out"] += u2.get("out", 0)

    raw = [m for m in mentions.get("mentions", [])
           if isinstance(m, dict) and (m.get("name") or "").strip()]
    cands = reason.mentions_to_candidates({"mentions": raw}, "search")
    kept = []
    for m, c in zip(raw, cands):
        level = str(m.get("confidence_level", "")).upper()
        try:
            conf = (float(m["confidence"])
                    if m.get("confidence") is not None
                    else CONFIDENCE.get(level, 0.0))
        except (TypeError, ValueError):
            conf = CONFIDENCE.get(level, 0.0)
        if level == "UNKNOWN" or conf < 0.5:
            continue
        c.confidence = min(conf, 0.99)
        extra = dict(c.extra or {})
        extra.update(dict(confidence_level=level or ("CONFIRMED" if conf >= 0.9 else "LIKELY"),
                          feature_check=m.get("feature_check", []),
                          alternatives=m.get("alternatives", []),
                          uncertainty=m.get("uncertainty", "")))
        c.extra = extra
        kept.append(c)

    return {"candidates": kept, "hits": hits, "queries": queries,
            "fingerprint": fingerprint, "hypotheses": hypotheses,
            "errors": errors, "usage": usage, "model": st_research.llm}
