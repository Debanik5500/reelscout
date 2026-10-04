"""The one LLM call: turn messy evidence into a clean list of things the viewer would want to open."""
from __future__ import annotations

import re

from .config import Settings
from .extract import Candidate, find_commands, find_in_text, merge
from .llm import complete_json

SYSTEM = """You analyse evidence extracted from a short social-media video (reel): speech transcript, on-screen text (OCR), the post caption, account mentions / location tag, the creator's own comments, and sometimes a few frames.
Goal: identify every specific real-world thing a viewer would want to open, buy, visit or look up: websites, GitHub repos, software tools, products (brand + model), brands, restaurants/hotels/attractions/places. Reels can be about anything (tech, travel, food, fashion, gadgets, home...).

Many creators never say the name and say "comment LINK" so a bot DMs the link. So:
- Work from every clue: on-screen text, @mentions, tagged accounts, hashtags, location tag, what is visibly shown (logos, packaging, UI, signage, landmarks), and what is said.
- Speech-to-text mishears brand names ("claw"/"clawed" = Claude, "trustmr.com" may be trustmrr.com). Put the corrected name in `name`, the heard form in `spoken_as`. Use on-screen text to fix spelling.
- Websites: bare domain in `name`. If unsure of the exact domain, give best guess plus `alternates`.
- GitHub: "owner/repo" only if owner AND repo are evidenced; otherwise kind "tool" with just the name.
- Products: brand + model if visible/said, else the most specific description you can justify.
- Places (restaurants, cafes, hotels, attractions, stays, venues): include the **full address or area** in `name` (e.g., "Dumont Experimental Ice Cream, Gachibowli, Hyderabad"), and if a specific address is mentioned, include it. The `url` should be the website if known, else the Google Maps / OSM link.
- **Content narration / educational reels**: If the creator is narrating ABOUT a topic (history, science, documentary, storytelling), the TOPIC is the primary entity, not the creator. The creator's handle is secondary. Example: "Scientists found X at Durupınar site" → primary entity is "Durupınar site", creator is secondary.
- "Click the link" / "watch my main video" / "link in bio" usually points to content ABOUT the topic, not the creator's profile. Follow the topic.
- **Tutorial/how-to videos**: if the video demonstrates steps (coding, setup, cooking, editing, etc.), add a `steps` array to each relevant mention: ordered list of what the viewer does/sees (e.g., ["Open youart.ai", "Connect MCP server", "Configure Claude", "Generate art"]). Include key commands, URLs, or UI labels seen in each step. **Always include `steps` field (empty array if not a tutorial).**
- NEVER invent. If the name is not determinable, do NOT guess a brand: add an entry to `unresolved_clues` describing what was seen/said (e.g. "matte black 40oz insulated tumbler with handle, no logo visible") so it can be searched later. Confidence 0-1 = how sure the item is real AND correctly named.
- Ignore generic UI chrome (browser menus, Instagram UI) and incidental background items the video does not recommend.
Return JSON only:
{"summary": "one sentence", "mentions": [{"name": str, "kind": "website|github_repo|tool|package|product|brand|place|other", "url": str|null, "alternates": [str], "spoken_as": str|null, "confidence": number, "evidence": "short paraphrase + which source", "steps": [str]}], "unresolved_clues": [str]}

EXAMPLE tutorial response:
{"summary": "Tutorial on connecting youart.ai to Claude via MCP", "mentions": [
  {"name": "youart.ai", "kind": "website", "url": "https://youart.ai", "confidence": 0.9, "evidence": "OCR shows URL repeatedly", "steps": ["Open youart.ai in browser", "Navigate to /mcp endpoint", "Copy server URL", "Add as custom connector in Claude"]},
  {"name": "Claude", "kind": "tool", "url": "https://claude.ai", "confidence": 0.9, "evidence": "UI shows 'Connect Claude to your data'", "steps": ["Open Claude settings", "Add custom connector", "Paste MCP server URL", "Test connection"]}
], "unresolved_clues": []}

EXAMPLE content narration response:
{"summary": "Reel explores scientific evidence for Noah's Ark at Durupınar site, Turkey", "mentions": [
  {"name": "Durupınar site", "kind": "place", "url": "https://www.openstreetmap.org/?mlat=39.44&mlon=44.23#map=16/39.44/44.23", "confidence": 0.95, "evidence": "Transcript describes soil tests at formation matching Noah's Ark dimensions", "steps": []},
  {"name": "noahsarkscans.com", "kind": "website", "url": "https://noahsarkscans.com", "confidence": 0.8, "evidence": "Creator says 'click the link to watch my main video' about the site", "steps": []},
  {"name": "Atatürk University", "kind": "place", "url": "https://www.ataturk.edu.tr", "confidence": 0.7, "evidence": "Lab report from Atatürk University Soil Science department", "steps": []}
], "unresolved_clues": ["Lab report document ID 2400365263"]}"""


VISION_ADDENDUM = """

FRAMES ARE ATTACHED (chronological; timestamps listed in the user message). The text evidence did not settle what this video is about, so look closely:
- Read ANY legible text: window/tab titles, terminal prompts and commands (pip/npm/brew/docker), package or file names, README headings, import lines, sidebar/menu labels, app names, logos, packaging labels, shop signs, street signs.
- **GitHub repositories: specifically look for github.com/owner/repo URLs in browser address bars, terminal output, code editors, or anywhere a repo URL might appear. Extract the full owner/repo path.**
- Dev/UI screens: name the product only if a logo, title or distinctive wordmark shows it; otherwise describe the UI and put it in unresolved_clues. Do not name a tool from UI style alone.
- Travel: name a place only from signage, a caption overlay, or a globally unmistakable landmark.
- Products: brand + model only if printed/visible; otherwise describe it (colour, material, shape, size, distinctive features).
- Prefer a short list of well-supported items over a long list of guesses.

IMPORTANT FRAME TAGGING:
For EACH frame, also output a `frame_tags` array with the same length as frames, where each entry is an object:
{"index": 0, "importance": "high|medium|low", "description": "one sentence what this frame shows", "tags": ["furniture", "product", "text", "ui", "landscape", "person", "code", "diagram"]}

Only mark "high" for frames showing: specific products, furniture designs, code/UI, text-heavy screens, distinctive landmarks, clear product shots. "medium" for supporting context. "low" for transitions, blurry, generic.
Return in JSON: {"summary": "...", "mentions": [...], "unresolved_clues": [...], "frame_tags": [...]}"""


def select_ocr(ocr_lines: list[dict], k: int = 30) -> list[dict]:
    """374 raw OCR lines is mostly junk (menu bars, buttons). Keep what could name something:
    URLs / install commands first, then phrases with some substance. Lines that sit on screen for many
    frames are usually UI chrome, so persistence is deliberately NOT a plus."""
    scored = []
    for l in ocr_lines:
        t = l["text"].strip()
        alnum = sum(ch.isalnum() for ch in t)
        strong = bool(find_in_text(t, "x") or find_commands(t, "x"))
        if alnum / max(len(t), 1) < 0.55 or (len(t) < 12 and not strong):
            continue
        sc = 10 if strong else 0
        sc += 1 if len(t.split()) >= 2 else 0
        sc += 1 if re.search(r"\b[A-Z][a-zA-Z]{3,}", t) else 0
        sc += l.get("conf", 0.5)
        scored.append((sc, l))
    top = sorted(scored, key=lambda x: -x[0])[:k]
    return [l for _, l in sorted(top, key=lambda x: x[1].get("t", 0))]


def build_prompt(transcript: list[dict], ocr_lines: list[dict], meta: dict, regex_found: list[Candidate], gated: dict | None = None) -> str:
    tx = " ".join(s["text"] for s in transcript)[:3500]
    ocr = "\n".join(f"[{l['t']}s x{l['n_frames']}] {l['text'][:120]}" for l in select_ocr(ocr_lines))[:2000]
    found = ", ".join(c.name for c in regex_found) or "(none)"
    caption = meta.get("caption", "")
    cc = [c["text"] for c in meta.get("comments", []) if c.get("by_creator")][:8]
    top = [c["text"] for c in sorted(meta.get("comments", []), key=lambda c: -c.get("likes", 0)) if not c.get("by_creator")][:8]
    extra = (f"UPLOADER: {meta.get('uploader')}\nLOCATION TAG: {meta.get('location')}\nHASHTAGS: {meta.get('hashtags')}\n"
             f"@MENTIONS / TAGGED: {meta.get('mentions')} {meta.get('tagged')}\nCREATOR'S OWN COMMENTS: {cc}\nTOP VIEWER COMMENTS: {top}\n"
             f"LINK GATED BEHIND COMMENT KEYWORD: {gated['keyword'] if gated else 'no'}\n\n")
    return (extra + f"CAPTION:\n{caption or '(none)'}\n\nTRANSCRIPT:\n{tx or '(no speech)'}\n\n"
            f"ON-SCREEN TEXT (OCR, time seen, frames seen in):\n{ocr or '(none)'}\n\n"
            f"URLS ALREADY FOUND BY REGEX: {found}")


def mentions_to_candidates(data: dict, source: str = "llm") -> list[Candidate]:
    out = []
    for m in data.get("mentions", []):
        name = (m.get("name") or "").strip()
        if not name:
            continue
        kind = m.get("kind") if m.get("kind") in ("website", "github_repo", "tool", "package", "product", "brand", "place", "other") else "other"
        url = m.get("url")
        if kind == "website":
            name = (url or name).lower().removeprefix("https://").removeprefix("http://").removeprefix("www.").split("/")[0]
        out.append(Candidate(kind, name, url, float(m.get("confidence") or 0.5),
                             [a.lower().removeprefix("https://").removeprefix("www.").split("/")[0] for a in m.get("alternates") or []],
                             m.get("spoken_as"), [dict(source=source, text=m.get("evidence") or "", t=None)],
                             extra=dict(steps=m.get("steps", [])) if m.get("steps") else {}))
    return out


def reason(transcript, ocr_lines, meta, regex_found, settings: Settings, image_paths=None, gated=None, frame_times=None):
    prompt = build_prompt(transcript, ocr_lines, meta, regex_found, gated)
    system = SYSTEM
    if image_paths:
        system += VISION_ADDENDUM
        prompt += f"\n\nFRAME TIMESTAMPS (s), in attachment order: {frame_times}"
    data, usage = complete_json(system, prompt, settings, image_paths)
    out = mentions_to_candidates(data, "llm")
    return data.get("summary", ""), out, usage, data.get("unresolved_clues", []), data.get("frame_tags", [])
