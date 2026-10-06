"""Reel -> ReelContext: observe, research, then synthesize one fixed-shape result.

evidence (analyze) -> visual observations (vision, never names/places)
                   -> research loop (plan, search, verify; <=3 rounds)
                   -> final synthesis (schema-constrained, empty fields allowed)
                   -> deterministic post-processing (confidence gates, coordinates, sources)
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from . import research, verify
from .events import EVENT_RULES, normalize_events, zone
from .config import Settings
from .llm import complete_json
from .reason import select_ocr

logger = logging.getLogger("reelscout.context")

CONTEXT_SCHEMA = 2
MIN_LOCATION_CONFIDENCE = 0.6
CATEGORIES = ("Food", "Travel", "Fitness", "Home", "Tech", "Shopping", "Learning", "Entertainment")

OBSERVE_SYSTEM = """You describe frames from a short social-media video. You are an observer, not an identifier.
Describe everything visually useful for later identification or understanding. Do NOT name a specific place, person, brand or product unless it is literally written/printed in the frame (sign, caption overlay, logo, packaging, UI title). Never guess identities or locations; describe instead (e.g. "red cable car on a snowy ridge", not a resort name).
Return JSON only, every list may be empty:
{"text_visible": [str], "people": [str], "objects": [str], "architecture": [str], "signs": [str], "logos": [str], "landscape": [str], "screens_or_ui": [str], "food": [str], "distinctive_features": [str], "activities": [str], "other_clues": [str]}
Short phrases only. Merge duplicates across frames."""

_FINAL_TEMPLATE = """You create the useful context for one short social-media video (reel) that a user saved, for display in a mobile app.
You get: caption, transcript, on-screen text, neutral visual observations, entities found by local extraction, and VERIFIED research findings.
Rules:
- Fill a field ONLY when the evidence supports it. Empty is correct and preferred over a guess. Never invent.
- Visual observations are clues, not facts. Do not turn an observation into a named place/product unless a SUPPORTED research finding, the caption, transcript, or on-screen text names it.
- Findings marked uncertain/contradicted must not be stated as fact.
- places: every real physical place the reel is about, as a list. A reel about ONE place has one entry. A LIST reel ("5 places near X", "best cafes in Y") has one entry per place shown, in the order shown, up to 12; read every place name from the on-screen text. A GitHub repo, app, website, recipe, meme or tutorial has no places (empty list). Each entry has name, area, city, country, confidence 0-1, precision and note.
  - city = the nearest city or town. Use caption and hashtag context for places that only show a spot name (a caption saying "near Siliguri" means city "Siliguri" for each). area = the neighbourhood or locality if known, else empty.
  - name = the MOST SPECIFIC place the evidence supports (the venue if a SUPPORTED finding or the reel names it, else the area narrowed by landmarks, e.g. "Old Manali" when the caption says it is near Manu Temple). Never use a coarser name than the evidence allows.
  - precision: "exact" only when the exact venue/address is verified; "area" for a neighbourhood, landmark or natural spot; "city" for city only.
  - note: when precision is not "exact", one short sentence on what is missing or how it was narrowed. Empty when exact.
{event_rules}
- products: physical/commercial things. links: websites, GitHub repos, tools, packages, apps (kind: github_repo|website|tool|package|app|other). people: named people with a role.
- LISTS: when the reel presents a list of items (repos, tools, apps, places, products), include EVERY item (up to 12), not a sample. Give one `links` entry per item with label (the item's name), kind, note (what it does, max 90 chars, from the caption or slides) and url ONLY if the evidence shows it; leave url empty otherwise (it is looked up afterwards). Use a repo's own name as the label.
- Never list the uploader's own brand, watermark, handle or call-to-action ("Save for later", "Comment X") as a link or product.
- title: max 8 words, plain words, no emoji/hashtags/URLs/quotes/"comment ..." CTAs.
- summary: 1-2 sentences about what the reel is and why it is worth saving.
- category: one of Food, Travel, Fitness, Home, Tech, Shopping, Learning, Entertainment, or "".
- useful_context: for a list reel, ONE line per item formatted "Name: what it does" (max 100 chars each, up to 12); otherwise up to 4 short actionable facts (steps, prices, hours, how to get the link). things_mentioned/activities: short strings.
- username: an Instagram @handle only if the caption clearly shows it.
Return JSON only:
{"title": str, "summary": str, "category": str,
 "places": [{"name": str, "area": str, "city": str, "country": str, "confidence": number, "precision": "exact|area|city", "note": str}],
 "products": [{"name": str, "brand": str, "url": str}], "people": [{"name": str, "role": str}],
 "links": [{"label": str, "url": str, "kind": str, "note": str}], "things_mentioned": [str], "activities": [str],
 "events": [{"title": str, "start": str, "end": str, "all_day": boolean, "tz": str, "year_inferred": boolean,
             "location": str, "notes": str, "confidence": number}],
 "useful_context": [str], "username": str}"""

FINAL_SYSTEM = _FINAL_TEMPLATE.replace("{event_rules}", EVENT_RULES.format(limit=5))


def stage_label(st: Settings, vision: bool = False) -> str:
    """"provider:model" for a stage, as actually used (vision may have its own model)."""
    model = (st.vision_model if vision and st.vision_model else st.model) or (st.preset or {}).get("model", "")
    return f"{st.llm}:{model}"


def observe(frame_paths: list[str], st: Settings, system: str | None = None) -> tuple[dict, dict]:
    """One vision call over a handful of frames -> neutral observations."""
    if not frame_paths or st.llm == "none":
        return {}, {}
    st = st.for_stage("VISION")
    data, usage = complete_json(system or OBSERVE_SYSTEM, "Images are attached in order.", st, frame_paths)
    usage = {**usage, "model": stage_label(st, vision=True), "images": len(frame_paths)}
    return {k: [str(x) for x in v if str(x).strip()] for k, v in data.items() if isinstance(v, list) and v}, usage


def evidence_ocr(ocr_lines: list[dict], limit: int = 60) -> list[str]:
    """On-screen text for the LLM, in the order it appeared.

    `select_ocr` drops anything under 12 characters, which loses exactly the short overlay titles that list
    reels use ("Kothidara", "Sangam Waterfalls"). Keep confident, mostly-alphanumeric lines of any length,
    and skip watermarks (@handles)."""
    keep, seen = [], set()
    for line in sorted(ocr_lines, key=lambda l: l.get("t", 0)):
        text = (line.get("text") or "").strip()
        key = re.sub(r"[^a-z0-9]", "", text.lower())
        alnum = sum(ch.isalnum() for ch in text)
        if (len(key) < 3 or key in seen or text.startswith("@") or float(line.get("conf", 0.5)) < 0.8
                or alnum / max(len(text), 1) < 0.7):
            continue
        seen.add(key)
        keep.append(text[:100])
    merged = [t for t in dict.fromkeys(keep + [l["text"][:100] for l in select_ocr(ocr_lines, 25)])
              if not t.startswith("@")]
    return merged[:limit]


def build_evidence(analysis: dict, observations: dict, now: datetime | None = None, tz: str = "UTC") -> str:
    meta = analysis.get("meta") or {}
    cands = analysis.get("candidates") or []
    ocr = analysis.get("ocr_lines") or []
    stamp = (now or datetime.now(timezone.utc)).strftime("%A %Y-%m-%d %H:%M")
    lines = [
        f"CURRENT DATE AND TIME: {stamp} (user timezone: {tz})",
        f"CAPTION: {(analysis.get('caption') or '(none)')[:1500]}",
        f"UPLOADER: {meta.get('uploader')} | LOCATION TAG: {meta.get('location')} | HASHTAGS: {meta.get('hashtags')} | MENTIONS/TAGGED: {meta.get('mentions')} {meta.get('tagged')}",
        f"TRANSCRIPT: {(analysis.get('transcript_text') or '(no speech)')[:3000]}",
        "ON-SCREEN TEXT (in order seen): " + (" | ".join(evidence_ocr(ocr)) or "(none)"),
        "VISUAL OBSERVATIONS: " + (json.dumps(observations, ensure_ascii=False) if observations else "(none)"),
        "LOCAL ENTITIES: " + ("; ".join(f"{c['kind']}:{c['name']} (score {c.get('confidence')})" for c in cands[:8]) or "(none)"),
        f"UNRESOLVED CLUES: {analysis.get('unresolved_clues') or []}",
        f"ONE-LINE SUMMARY FROM EXTRACTION: {analysis.get('summary') or '(none)'}",
    ]
    return "\n".join(lines)


def _clean(text) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _supported(findings: list[dict]) -> list[dict]:
    return [f for f in findings if f["verdict"] == "supported" and f["confidence"] >= MIN_LOCATION_CONFIDENCE]


def _place_query(loc: dict) -> str:
    """"Manali, Manali, India" -> "Manali, India": skip parts already contained in an earlier part."""
    parts: list[str] = []
    for part in (loc["name"], loc.get("area", ""), loc["city"], loc["country"]):
        if part and not any(part.lower() in earlier.lower() for earlier in parts):
            parts.append(part)
    return ", ".join(parts)


def _geocode(query: str) -> dict | None:
    """Nominatim lookup with an on-disk cache. Only successes are cached."""
    hit = research._cache_get("geo", query)
    if hit is not None:
        return hit
    time.sleep(1.1)  # Nominatim allows one request per second
    try:
        geo = verify.geocode(query)
    except Exception as exc:
        logger.warning("geocode failed for %r: %s", query, exc)
        return None
    if not geo.get("ok"):
        return None
    hit = dict(lat=float(geo["lat"]), lon=float(geo["lon"]))
    research._cache_put("geo", query, hit)
    return hit


def _ground_place(loc: dict) -> None:
    """Give one place coordinates, falling back venue -> neighbourhood -> city and saying so."""
    if not loc["name"] or loc["lat"] is not None:
        return
    area = ", ".join(p for p in (loc["area"], loc["city"], loc["country"]) if p)
    city = ", ".join(p for p in (loc["city"], loc["country"]) if p)
    attempts = [(_place_query(loc), loc["precision"] or "area"), (area, "area"), (city, "city")]
    tried: set[str] = set()
    for query, precision in attempts:
        if not query or query in tried:
            continue
        tried.add(query)
        hit = _geocode(query)
        if not hit:
            continue
        loc["lat"], loc["lng"] = hit["lat"], hit["lon"]
        if precision != loc["precision"]:  # the place itself was not found: be honest about the pin
            loc["precision"] = precision
            where = area if precision == "area" else city
            loc["note"] = f"The exact spot isn't on the map; the pin shows {where}."
        return


def _ground_location(ctx: dict) -> None:
    """Every place gets coordinates when the map can give any, so it can appear on the Places page."""
    for loc in ctx["places"]:
        _ground_place(loc)


RESOLVE_SYSTEM = """You match list items from a social-media post to the right GitHub repository.
Each item has a name and a short description from the post, followed by candidate repositories (full name, stars, description).
Pick a candidate ONLY when its name AND description clearly match the item; otherwise null. A popular repo with a similar name but a different purpose is NOT a match.
Return JSON only: {"matches": [{"name": str, "repo": "owner/name" or null}]}"""


_GITHUB_REPO = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)")


def _github_candidates(link: dict) -> list[dict]:
    """Repos for one list item: GitHub name search first, then repos that a web search for it surfaces."""
    label = link["label"]
    hits = research._cache_get("ghsearch", label)
    if hits is None:
        hits = verify.github_search_text(label, 5)
        research._cache_put("ghsearch", label, hits)
        time.sleep(0.5)  # unauthenticated GitHub search allows about 10 requests a minute
    return hits


def _web_candidates(link: dict) -> list[dict]:
    """Second chance: repos mentioned in web results for "<name> github <what it does>"."""
    from . import search

    query = f"{link['label']} github {link['note']}".strip()
    hits = research._cache_get("ghweb", query)
    if hits is None:
        hits = []
        for result in search.web_search(query, 6):
            m = _GITHUB_REPO.search(result.get("url", ""))
            if m and m.group(2) not in ("issues", "pulls"):
                hits.append(dict(full_name=f"{m.group(1)}/{m.group(2).removesuffix('.git')}", stars="?",
                                 description=(result.get("title", "") + " " + result.get("snippet", ""))[:160]))
        research._cache_put("ghweb", query, hits)
    return hits


def _choose_repos(todo: list[dict], candidates: dict[str, list[dict]], rst: Settings) -> dict:
    blocks = [
        f"ITEM: {l['label']} | {l['note']}\nCANDIDATES:\n" + "\n".join(
            f"- {h['full_name']} ({h['stars']} stars): {(h['description'] or '')[:120]}" for h in candidates[l["label"]])
        for l in todo if candidates.get(l["label"])
    ]
    if not blocks:
        return {}
    try:
        data, usage = complete_json(RESOLVE_SYSTEM, "\n\n".join(blocks), rst)
    except Exception as exc:
        logger.warning("repo resolution failed: %s", exc)
        return {}
    chosen = {m.get("name"): m.get("repo") for m in data.get("matches", []) if isinstance(m, dict)}
    for link in todo:
        repo = chosen.get(link["label"])
        if repo and any(h["full_name"].lower() == str(repo).lower() for h in candidates.get(link["label"], [])):
            link["url"] = f"https://github.com/{repo}"
    return usage


def _resolve_list_links(ctx: dict, st: Settings) -> dict:
    """List reels name repos without linking them ("comment REPOS for the links"): find each one on GitHub."""
    todo = [l for l in ctx["links"] if not l["url"] and l["label"] and l["kind"] == "github_repo"][:12]
    if not todo:
        return {}
    rst = st.for_stage("RESEARCH")
    usage = _choose_repos(todo, {l["label"]: _github_candidates(l) for l in todo}, rst)
    left = [l for l in todo if not l["url"]]
    if left:
        extra = _choose_repos(left, {l["label"]: _web_candidates(l) for l in left}, rst)
        usage = {k: usage.get(k, 0) + extra.get(k, 0) for k in ("in", "out", "calls")} if usage and extra else (usage or extra)
    return {**usage, "model": stage_label(rst)} if usage else {}


ITEM_NOTES_SYSTEM = """You write one short line for each item in a list from a social-media post.
Use ONLY the evidence given (caption, transcript, on-screen text, observations). Max 90 characters per line, starting with what the item does or is.
If the evidence says nothing about an item, return an empty string for it. Never guess.
Return JSON only: {"notes": [{"name": str, "note": str}]}"""


def _fill_item_notes(ctx: dict, evidence: str, st: Settings) -> dict:
    """List reels need a gist per item. If the main answer described too few, one small focused call fills the rest."""
    links = ctx["links"]
    missing = [l for l in links if not l["note"]]
    if len(links) < 3 or len(missing) <= len(links) * 0.3:
        return {}
    rst = st.for_stage("TEXT")
    names = "\n".join(f"- {l['label']}" for l in missing)
    try:
        data, usage = complete_json(ITEM_NOTES_SYSTEM, f"{evidence}\n\nITEMS NEEDING A LINE:\n{names}", rst)
    except Exception as exc:
        logger.warning("item notes failed: %s", exc)
        return {}
    notes = {_clean(n.get("name")).lower(): _clean(n.get("note"))[:110]
             for n in data.get("notes", []) if isinstance(n, dict)}
    for link in missing:
        link["note"] = notes.get(link["label"].lower(), "")
    return {**usage, "model": stage_label(rst)}


def fill_item_lines(ctx: dict) -> None:
    """Every described link becomes a "Name: what it is" line in Worth remembering (idempotent, no LLM)."""
    lines = ctx.setdefault("useful_context", [])
    for link in ctx.get("links", []):
        label, note = link.get("label", ""), link.get("note", "")
        if label and note and not any(line.lower().startswith(label.lower()) for line in lines):
            lines.append(f"{label}: {note}")
    del lines[12:]


_URL_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; reelscout/0.1)"}


def _reachable(url: str) -> bool:
    """True when the host answers. Bot-blocking statuses (401/403/429) still mean the site exists."""
    import requests

    if research._cache_get("url", url):
        return True
    try:
        for method in (requests.head, requests.get):
            resp = method(url, headers=_URL_HEADERS, timeout=6, allow_redirects=True, stream=True)
            resp.close()
            if resp.status_code < 400 or resp.status_code in (401, 403, 429):
                research._cache_put("url", url, {"ok": True})
                return True
            if method is requests.head and resp.status_code in (404, 405, 501):
                continue  # some servers reject HEAD; confirm with GET
            return False
    except Exception:
        return False
    return False


def _verify_urls(ctx: dict) -> None:
    """The LLM may turn a handle or a half-remembered name into a URL. Keep only addresses that answer."""
    links, products = ctx["links"], ctx["products"]
    urls = {i["url"] for i in links + products if i.get("url", "").startswith(("http://", "https://"))}
    if not urls:
        ctx["links"] = [l for l in links if l.get("url", "").startswith(("http://", "https://"))]
        return
    with ThreadPoolExecutor(max_workers=6) as pool:
        ok = dict(zip(urls, pool.map(_reachable, list(urls))))
    ctx["links"] = [l for l in links if ok.get(l["url"])]
    for product in products:
        if product.get("url") and not ok.get(product["url"]):
            product["url"] = ""


def _empty_place() -> dict:
    return dict(name="", area="", city="", country="", lat=None, lng=None, map_url="", confidence=0.0,
                precision="", note="")


def _clean_place(entry: dict, findings: list[dict]) -> dict | None:
    """One place from the model, or None when it is unnamed or not confident enough."""
    name, conf = _clean(entry.get("name")), float(entry.get("confidence") or 0)
    if not name or conf < MIN_LOCATION_CONFIDENCE:
        return None
    loc = _empty_place()
    precision = _clean(entry.get("precision")).lower()
    loc.update(name=name, area=_clean(entry.get("area")), city=_clean(entry.get("city")),
               country=_clean(entry.get("country")), confidence=conf,
               precision=precision if precision in ("exact", "area", "city") else "area",
               note=_clean(entry.get("note"))[:160])
    key = name.lower()
    match = next((f for f in _supported(findings) if f["kind"] == "place" and "lat" in f
                  and (f["candidate"].lower() in key or key in f["candidate"].lower())), None)
    if match:
        loc.update(lat=match["lat"], lng=match["lon"], map_url=match["map_url"],
                   city=loc["city"] or match["city"], country=loc["country"] or match["country"])
    if not loc["map_url"]:
        loc["map_url"] = "https://maps.google.com/?q=" + quote(_place_query(loc))
    return loc


def _finalize(data: dict, findings: list[dict], sources: list[dict], category_hint: str = "") -> dict:
    """Deterministic gates: low-confidence guesses are dropped, coordinates come only from verified lookups."""
    data = data if isinstance(data, dict) else {}
    raw = data.get("places")
    if not isinstance(raw, list):  # older single-location shape
        raw = [data["location"]] if isinstance(data.get("location"), dict) else []
    places, seen_places = [], set()
    for entry in raw[:12]:
        place = _clean_place(entry, findings) if isinstance(entry, dict) else None
        if place and place["name"].lower() not in seen_places:
            seen_places.add(place["name"].lower())
            places.append(place)
    loc = places[0] if places else _empty_place()

    def items(key: str, fields: tuple[str, ...], need: str, limit: int = 12) -> list[dict]:
        out, seen = [], set()
        for it in data.get(key) or []:
            if not isinstance(it, dict) or not _clean(it.get(need)) or _clean(it.get(need)).lower() in seen:
                continue
            seen.add(_clean(it[need]).lower())
            out.append({f: _clean(it.get(f)) for f in fields})
        return out[:limit]

    def strs(key: str, limit: int) -> list[str]:
        return list(dict.fromkeys(_clean(s) for s in (data.get(key) or []) if _clean(s)))[:limit]

    category = _clean(data.get("category"))
    handle = _clean(data.get("username")).lstrip("@")
    src, seen_urls = [], set()
    for s in [*sources, *({"title": f["candidate"], "url": u} for f in _supported(findings) for u in f["source_urls"][:1])]:
        if s.get("url") and s["url"] not in seen_urls and s["url"].startswith("http"):
            seen_urls.add(s["url"])
            src.append({"title": _clean(s.get("title"))[:120], "url": s["url"]})
    return dict(
        schema=CONTEXT_SCHEMA,
        title=_clean(data.get("title")), summary=_clean(data.get("summary")),
        category=category if category in CATEGORIES else (category_hint if category_hint in CATEGORIES else ""),
        location=loc, places=places,
        products=items("products", ("name", "brand", "url"), "name"),
        people=items("people", ("name", "role"), "name"),
        links=items("links", ("label", "url", "kind", "note"), "label"),
        things_mentioned=strs("things_mentioned", 8), activities=strs("activities", 6),
        useful_context=strs("useful_context", 12), sources=src[:6],
        username=handle if re.fullmatch(r"[A-Za-z0-9._]{2,30}", handle) and not handle.replace(".", "").isdigit() else "",
    )


def synthesize(analysis: dict, observations: dict, st: Settings, max_rounds: int = 3,
               tz: str = "UTC") -> tuple[dict, dict]:
    """Research + final LLM. Returns (context dict, debug dict). Raises if the final LLM fails."""
    now = datetime.now(zone(tz) or timezone.utc)
    evidence = build_evidence(analysis, observations, now, tz)
    res = research.research(evidence, st, max_rounds)
    verified = "\n".join(
        f"- [{f['verdict']} {f['confidence']:.2f}] {f['goal']} -> {f['candidate']} ({f['kind']}) "
        f"city={f['city']} country={f['country']} url={f['url']} support={f['evidence']} contradictions={f['contradictions']}"
        for f in res["findings"]) or "(no research was needed or nothing verified)"
    stg = st.for_stage("TEXT")
    data, usage = complete_json(FINAL_SYSTEM, f"{evidence}\n\nRESEARCH FINDINGS:\n{verified}", stg, max_tokens=2200)
    ctx = _finalize(data, res["findings"], res["sources"], category_hint="")
    ctx["events"] = normalize_events(data.get("events"), now, tz, limit=5)
    _ground_location(ctx)
    resolve_usage = _resolve_list_links(ctx, st)
    # Describe EVERY listed item first; only then drop links that could not be verified. A repo whose URL is
    # unknown still belongs in Worth remembering.
    notes_usage = _fill_item_notes(ctx, evidence, st)
    fill_item_lines(ctx)
    _verify_urls(ctx)
    debug = dict(research=dict(rounds=res["rounds"], queries=res["queries"], findings=res["findings"], errors=res["errors"]),
                 usage=dict(final={**usage, "model": stage_label(stg)},
                            research={**res["usage"], "model": stage_label(st.for_stage("RESEARCH"))},
                            resolve=resolve_usage, notes=notes_usage))
    return ctx, debug


def context_path(results_dir: Path, job_id: str) -> Path:
    return Path(results_dir) / f"{job_id}.context.json"


def save_context(results_dir: Path, job_id: str, ctx: dict, debug: dict, observations: dict) -> None:
    p = context_path(results_dir, job_id)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(dict(context=ctx, observations=observations, debug=debug), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


def load_context(results_dir: Path, job_id: str) -> dict | None:
    try:
        data = json.loads(context_path(results_dir, job_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    ctx = data.get("context") if isinstance(data, dict) else None
    return ctx if isinstance(ctx, dict) and ctx.get("schema") == CONTEXT_SCHEMA else None
