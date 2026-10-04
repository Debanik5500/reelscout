from __future__ import annotations

import argparse
import json

from .config import Settings
from .pipeline import analyze


def _human_summary(r: dict) -> str:
    cands = r.get("candidates", [])
    if not cands:
        return "Could not identify what this reel is about."
    
    # Start with LLM summary if available
    llm_summary = r.get("summary", "")
    if llm_summary:
        parts = [llm_summary]
    else:
        top = cands[0]
        if top["confidence"] < 0.4:
            return f"Unclear — best guess: {top['name']} ({top['kind']}), but confidence is low ({top['confidence']:.0%})."
        verified = " (verified)" if top.get("verification", {}).get("ok") else ""
        kind_label = _kind_label(top["kind"])
        url = f" — {top['url']}" if top.get("url") else ""
        meta = r.get("meta", {})
        uploader = meta.get("uploader_username") or meta.get("uploader")
        uploader_str = f" by @{uploader}" if uploader else ""
        parts = [f"This reel shows {kind_label}: {top['name']}{verified}{url}{uploader_str}"]
    
    # Show steps for tutorial videos
    top = cands[0]
    top_steps = top.get("extra", {}).get("steps", [])
    if top_steps:
        parts.append("Steps: " + " → ".join(top_steps[:5]) + ("..." if len(top_steps) > 5 else ""))
    
    others = [c for c in cands[1:4] if c["confidence"] >= 0.4]
    if others:
        parts.append("Also mentioned: " + ", ".join(f"{c['name']} ({_kind_label(c['kind'])})" + (f" — {c['url']}" if c.get("url") else "") for c in others))
    if r.get("gated_link"):
        g = r["gated_link"]
        parts.append(f"Creator gates the link via '{g['keyword']}' ({g['mechanism']}).")
    return " ".join(parts) + "."


def _kind_label(kind: str) -> str:
    return {
        "github_repo": "a GitHub repository",
        "website": "a website",
        "package": "a package/tool",
        "brand": "an Instagram account",
        "product": "a product",
        "tool": "a tool",
        "place": "a place",
        "other": "something",
    }.get(kind, kind)


def apply_llm_flags(st: Settings, llm: str | None, model: str | None) -> None:
    if llm and llm.lower() != st.llm and not model:
        st.model, st.vision_model = "", ""          # a model name from .env belongs to the OLD provider
    st.llm = (llm or st.llm).lower()
    st.model = model or st.model


def main() -> int:
    ap = argparse.ArgumentParser(prog="reelscout", description="Find the websites / repos / tools mentioned in a reel.")
    ap.add_argument("source", help="local video file, or an Instagram/YouTube/TikTok URL (needs yt-dlp)")
    ap.add_argument("--llm", help="gemini | groq | openrouter | ollama | anthropic | none (default: $REELSCOUT_LLM)")
    ap.add_argument("--model", help="override the provider's default model")
    ap.add_argument("--whisper", help="tiny|base|small|medium")
    ap.add_argument("--vision", choices=["off", "auto", "on"], help="auto (default): send frames to a vision LLM only when text signals are weak")
    ap.add_argument("--resolve", choices=["off", "auto"], help="auto (default): search the web for unresolved clues when results are still weak")
    ap.add_argument("--image-search", action="store_true", help="also reverse-image-search frames (PicImageSearch; scraping-based, may break)")
    ap.add_argument("--geo", action="store_true", help="also geolocate frames with GeoCLIP (pip install geoclip; needs torch)")
    ap.add_argument("--text-llm", choices=["auto", "always", "never"], help="auto (default): skip the first LLM call when local evidence is already decisive")
    ap.add_argument("--no-verify", action="store_true", help="skip network verification")
    ap.add_argument("--caption", default="", help="paste the post caption if you have it (big accuracy boost)")
    ap.add_argument("--cookies-from-browser", help="e.g. chrome, firefox (Instagram links usually need this)")
    ap.add_argument("--cookies", help="path to Netscape-format cookies.txt file (alternative to --cookies-from-browser)")
    ap.add_argument("--json", action="store_true", help="print raw JSON")
    ap.add_argument("--summary", action="store_true", help="print a one-sentence human-readable conclusion")
    a = ap.parse_args()

    st = Settings.from_env()
    apply_llm_flags(st, a.llm, a.model)
    for var, val in (("REELSCOUT_MODEL", st.model), ("REELSCOUT_VISION_MODEL", st.vision_model)):
        if st.llm in ("openai", "anthropic", "gemini", "groq") and ":" in val:
            print(f"warning: {var}={val!r} looks like an Ollama tag but the provider is {st.llm}; "
                  f"remove it from .env or the API call will fail", file=__import__("sys").stderr)
    st.whisper_model = a.whisper or st.whisper_model
    st.text_policy = a.text_llm or st.text_policy
    st.resolve, st.image_search, st.geo = a.resolve or st.resolve, a.image_search, a.geo
    st.vision, st.verify = a.vision or st.vision, not a.no_verify

    r = analyze(a.source, st, a.caption, a.cookies_from_browser, a.cookies)
    if a.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0
    if a.summary:
        print(_human_summary(r))
        return 0

    print(f"\n[{r['status'].upper()}] {r['summary'] or '(no LLM summary)'}")
    if r["used_vision"]:
        print("(text signals were weak -> frames were sent to the vision model)")
    if r["gated_link"]:
        g = r["gated_link"]
        print(f"\n⚠ creator gates the link ({g['mechanism']}, keyword={g['keyword']}) -> name must be inferred from content\n")
    for c in r["candidates"]:
        v = c["verification"]
        flag = "✓" if v.get("ok") else ("✗" if v else "·")
        print(f"{c['confidence']:.2f} {flag} [{c['kind']}] {c['name']}  {c['url'] or ''}")
        for e in c["evidence"][:3]:
            print(f"        - {e['source']}: {str(e['text'])[:90]}")
    if r["unresolved_clues"]:
        print("\nunresolved clues (search these next):")
        for c in r["unresolved_clues"]:
            print(f"   ? {c}")
    rs = r["resolution"]
    if rs.get("ran"):
        print(f"\nresolution: {len(rs['hits'])} web hits, {len(rs['image_hits'])} image hits, geo={'yes' if rs['geo'] else 'no'}")
        for h in (rs["image_hits"] + rs["hits"])[:6]:
            print(f"   > {h['title'][:70]} | {h['url'][:70]}")
        for e in rs["errors"]:
            print(f"   ! {e}")
    for sg in r["llm"]["stages"]:
        print(f"   llm[{sg['stage']}] {sg['model']}  in={sg.get('in', 0)} out={sg.get('out', 0)} calls={sg.get('calls', 1)}")
    print(f"\ntimings: {r['timings']}   llm: {r['llm']['provider']} {r['llm']['usage'] or ''} {r['llm']['error'] or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
