from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
import time
from pathlib import Path
from typing import Callable

from . import asr, extract, ingest, media, ocr, reason, resolve, verify
from .config import Settings


_GENERIC_TERMS = {"app", "tool", "ai", "vibe", "coded", "coding", "code", "debug", "cheat", "tutorial", "guide", "how to", "free", "open", "source", "github", "repo", "project", "build", "make", "create", "new", "best", "top", "latest", "update", "version", "release", "demo", "showcase", "example", "sample"}


def _fallback_clues(transcript: list[dict], ocr_lines: list[dict], caption: str) -> list[str]:
    """Generate search clues from raw text when LLM is unavailable."""
    import re
    clues = []
    
    def extract_phrases(text: str) -> list[str]:
        parts = re.split(r'[.,;:!?\n\-]+', text)
        out = []
        for p in parts:
            p = p.strip()
            if len(p.split()) < 2 or len(p) > 100:
                continue
            words = p.lower().split()
            meaningful = [w for w in words if w not in _GENERIC_TERMS and len(w) > 2]
            if len(meaningful) < 2:
                continue
            if sum(c.isalnum() for c in p) / max(len(p), 1) < 0.5:
                continue
            out.append(p)
        return out
    
    for s in transcript:
        clues.extend(extract_phrases(s["text"]))
    
    for l in ocr_lines:
        t = l["text"].strip()
        if len(t) >= 10:
            clues.extend(extract_phrases(t))
    
    if caption:
        clues.extend(extract_phrases(caption))
    
    seen = set()
    uniq = []
    for c in sorted(clues, key=len, reverse=True):
        key = c.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq[:5]


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _cached(path: Path, fn: Callable) -> tuple:
    if path.exists():
        return json.loads(path.read_text()), True
    val = fn()
    path.write_text(json.dumps(val, indent=1))
    return val, False


def _run_parallel(tasks: dict[str, Callable]) -> dict:
    """Run tasks in parallel, return {name: result}."""
    results = {}
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(fn): name for name, fn in tasks.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
            except Exception as e:
                _log(f"[parallel:{name}] failed: {e}")
                results[name] = None
    return results


def score(c: extract.Candidate) -> float:
    """Transparent heuristic: LLM belief + independent corroboration + verification."""
    s = c.confidence
    ocr_frames = max((e.get("n_frames", 1) for e in c.evidence if e["source"] == "ocr"), default=0)
    if "ocr" in c.sources:
        s += 0.15 if ocr_frames >= 2 else 0.10
    if "transcript" in c.sources:
        s += 0.10
    if c.sources & {"caption", "tagged", "creator_comment"}:
        s += 0.15
    if c.sources & {"search", "image_search", "geoclip"}:
        s += 0.05
    v = c.verification
    if v.get("ok"):
        s += 0.20  # verified domains get bigger boost
    elif v and c.kind in ("website", "github_repo"):
        s -= 0.25
    return round(max(0.0, min(1.0, s)), 2)


def decisive(cands: list, st: Settings) -> bool:
    """Local evidence is decisive when the best candidate is confident, verified, and backed by two
    independent signals (e.g. said out loud AND seen on screen). Then an LLM adds cost, not information.
    Also allow single-source if it appears in many OCR frames (strong visual evidence) or is verified."""
    if not cands:
        return False
    top = cands[0]
    indep = top.sources & {"ocr", "transcript", "caption", "tagged", "creator_comment"}
    ocr_frames = max((e.get("n_frames", 1) for e in top.evidence if e["source"] == "ocr"), default=0)
    if ocr_frames >= 3:
        indep = indep | {"ocr_persistent"}
    if top.verification.get("ok") and top.kind in ("website", "github_repo", "package"):
        indep = indep | {"verified"}
    return top.score >= st.weak_below and len(indep) >= 2 and top.verification.get("ok") is True


def analyze(source: str, settings: Settings | None = None, caption: str = "",
            cookies_from_browser: str | None = None, cookies_file: str | None = None) -> dict:
    st = settings or Settings.from_env()
    t0, timings = time.time(), {}

    def stage(name: str, t: float) -> None:
        timings[name] = round(time.time() - t, 2)

    # Stage 1: Ingest (download + metadata)
    t = time.time()
    video, meta = ingest.fetch(source, st.cache_dir / "downloads", cookies_from_browser, cookies_file)
    meta["caption"] = caption or meta["caption"]
    caption = meta["caption"]
    info = media.probe(video)
    work = st.cache_dir / media.video_key(video)
    work.mkdir(parents=True, exist_ok=True)
    stage("ingest", t)

    # Stage 2: Semantic segmentation (fast, needed for frame sampling strategy)
    t = time.time()
    segments = media.get_semantic_segments(video, min_segment=5.0, max_segment=120.0)
    stage("segmentation", t)
    _log(f"[segmentation] {len(segments)} segments")

    # Stage 3: Parallel execution of ASR + Frame Sampling
    t = time.time()
    frame_cache_key = f"frames_{st.max_frames}_{st.sample_hz}_{st.fixed_hz}.json"
    
    def get_frames():
        return media.sample_frames(
            video, work / "frames", st.max_frames,
            sample_hz=st.sample_hz, diff_thresh=6.0,
            fixed_hz=st.fixed_hz
        )
    
    def get_transcript():
        wav = work / "audio.wav"
        return asr.transcribe(str(wav), st.whisper_model) if media.extract_audio(video, wav) else []
    
    parallel_results = _run_parallel({
        "frames": lambda: _cached(work / frame_cache_key, get_frames)[0],
        "transcript": lambda: _cached(work / f"transcript_{st.whisper_model}.json", get_transcript)[0],
    })
    
    frames = parallel_results["frames"]
    transcript = parallel_results["transcript"]
    stage("asr+frames", t)
    _log(f"[asr] {len(transcript)} segments | [frames] {len(frames)} kept")

    # Stage 4: OCR on frames (parallelized via ocr_workers)
    t = time.time()
    ocr_cache_key = f"ocr_{st.ocr_workers}.json"
    ocr_lines, hit = _cached(work / ocr_cache_key, lambda: ocr.ocr_frames(frames, workers=st.ocr_workers))
    stage("ocr", t)
    _log(f"[ocr] {len(ocr_lines)} unique lines{' (cached)' if hit else ''} (workers={st.ocr_workers})")

    # Stage 5: Local extraction (fast, synchronous)
    regex_found = extract.merge(extract.gather(transcript, ocr_lines, caption) + extract.social_candidates(meta))
    gated = extract.detect_gating([caption, " ".join(s["text"] for s in transcript)] +
                                  [c["text"] for c in meta.get("comments", []) if c.get("by_creator")])
    _log(f"[regex] {[c.name for c in regex_found]}   gated={gated}")

    summary, usage, llm_error, cands, clues = "", {}, None, regex_found, []
    used_vision = False
    frame_tags = []
    stages_log: list[dict] = []

    def run_llm(images=None, times=None, label: str = "text"):
        nonlocal summary, usage, cands, clues, llm_error, frame_tags
        stg = st.for_stage("VISION" if images else "TEXT")
        model = (stg.vision_model if images and stg.vision_model else stg.model) or (stg.preset or {}).get("model")
        _log(f"[llm:{label}] {stg.llm}:{model}" + (f" with {len(images)} frames" if images else "") +
             (" (first load)" if stg.llm == "ollama" else ""))
        try:
            result = reason.reason(transcript, ocr_lines, meta, regex_found, stg, images, gated, times)
            if isinstance(result, tuple) and len(result) == 5:
                sm, llm_c, u, cl, frame_tags = result
            else:
                sm, llm_c, u, cl = result
                frame_tags = []
            summary = summary or sm
            cands = extract.merge(cands + llm_c)
            clues = list(dict.fromkeys(clues + cl))
            for k, v in u.items():
                usage[k] = usage.get(k, 0) + v
            stages_log.append(dict(stage=label, model=f"{stg.llm}:{model}", **u))
        except Exception as e:
            llm_error = str(e)
            _log(f"[llm:{label}] failed: {e}")

    def finish():
        if st.verify:
            todo = [c for c in cands if c.kind != "other" and not c.verification]

            def safe(c):
                try:
                    verify.verify(c)
                except Exception as e:
                    c.verification = dict(ok=False, error=str(e)[:100])
            with ThreadPoolExecutor(max_workers=6) as ex:
                list(ex.map(safe, [c for c in todo if c.kind != "place"]))
            for c in todo:
                if c.kind == "place":
                    safe(c)
        for c in cands:
            c.score = score(c)
        cands.sort(key=lambda c: -c.score)

    # Stage 6: Tier 0 - Local verification & scoring
    t = time.time()
    finish()
    stage("local_verify", t)

    # Stage 7: Tier 1 - Text LLM (if needed)
    is_decisive = decisive(cands, st)
    need_text_llm = (st.llm != "none" and st.text_policy != "never" and 
                     (st.text_policy == "always" or not is_decisive))
    
    # Stage 8: Tier 2 - Vision LLM (if needed) - check early to potentially parallelize
    weak_after_local = not any(c.score >= st.weak_below and c.kind != "other" for c in cands)
    tutorial_signals = _detect_tutorial_signals(ocr_lines, frames, transcript)
    # Only trigger vision for tutorials if local evidence is NOT already decisive
    need_vision = (st.llm != "none" and st.vision != "off" and 
                   (st.vision == "on" or weak_after_local or (tutorial_signals and not is_decisive)))
    
    # If vision will run, skip text LLM (vision model handles both text + frames)
    if need_vision:
        need_text_llm = False

    # Run Text LLM and Vision LLM in parallel if both needed
    if need_text_llm and need_vision:
        _log("[llm] Running TEXT + VISION in parallel")
        t = time.time()
        
        # Prepare vision frames upfront
        shots = media.pick_for_vision(frames, work / "vision", st.vision_frames)
        vision_reason = "forced" if st.vision == "on" else ("tutorial detected" if tutorial_signals else "text signals weak")
        _log(f"[vision] {vision_reason} -> sending {len(shots)} frames")
        
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_text = ex.submit(run_llm, None, None, "text")
            fut_vision = ex.submit(run_llm, [x["path"] for x in shots], [x["t"] for x in shots], "vision")
            fut_text.result()
            fut_vision.result()
        
        used_vision = True
        t = time.time()
        finish()
        stage("llm_parallel+verify", t)
    
    elif need_text_llm:
        t = time.time()
        run_llm()
        stage("llm_text", t)
        t = time.time()
        finish()
        stage("verify", t)
    
    elif need_vision:
        t = time.time()
        shots = media.pick_for_vision(frames, work / "vision", st.vision_frames)
        vision_reason = "forced" if st.vision == "on" else ("tutorial detected" if tutorial_signals else "text signals weak")
        _log(f"[vision] {vision_reason} -> sending {len(shots)} frames")
        run_llm([x["path"] for x in shots], [x["t"] for x in shots], "vision")
        used_vision = True
        t = time.time()
        finish()
        stage("llm_vision+verify", t)
    else:
        if st.llm != "none":
            _log("[llm] skipped: local evidence already decisive")
            stages_log.append(dict(stage="text", model="local (skipped: decisive)", calls=0))

    # Stage 9: Tier 3 - Resolve (only if explicitly enabled and still weak)
    # Default: resolve="off" - only run when user explicitly requests
    weak = not any(c.score >= st.weak_below and c.kind != "other" for c in cands)
    llm_entities_need_search = any(c.kind in ("place", "product", "brand", "tool", "package") for c in cands if c.sources & {"llm"})
    resolution = dict(ran=False)
    
    if st.resolve != "off" and (weak or llm_entities_need_search):
        effective_clues = clues or _fallback_clues(transcript, ocr_lines, caption)
        use_img = st.image_search or bool(gated)
        if effective_clues or use_img or st.geo:
            t = time.time()
            rst = dataclasses.replace(st, image_search=use_img)
            rframes = [x["path"] for x in media.pick_for_vision(frames, work / "resolve", 3)]
            _log(f"[resolve] searching on {len(effective_clues)} clue(s)" + (" + image search" if use_img else "") + (" + geo" if st.geo else ""))
            res = resolve.resolve(effective_clues, summary, rframes, rst, transcript, ocr_lines, caption)
            cands = extract.merge(cands + res["candidates"])
            for k, v in res["usage"].items():
                usage[k] = usage.get(k, 0) + v
            if res["usage"]:
                stages_log.append(dict(stage="research", model=res.get("model", ""), **res["usage"]))
            finish()
            resolution = dict(ran=True, hits=res["hits"], image_hits=res["image_hits"], geo=res["geo"], errors=res["errors"])
            stage("resolve", t)

    timings["total"] = round(time.time() - t0, 2)

    top = cands[0].score if cands else 0.0
    status = "identified" if top >= st.weak_below else "tentative" if top >= 0.4 else "unidentified"

    if not summary:
        summary = _fallback_summary(caption, transcript, ocr_lines, regex_found, gated, cands)

    return dict(
        status=status, source=source, video=info, caption=caption, summary=summary, gated_link=gated,
        unresolved_clues=clues, meta={k: v for k, v in meta.items() if k != "caption"},
        used_vision=used_vision, resolution=resolution,
        candidates=[dict(kind=c.kind, name=c.name, url=c.url, confidence=c.score, llm_confidence=c.confidence,
                         spoken_as=c.spoken_as, evidence=c.evidence, verification=c.verification, extra=c.extra)
                    for c in cands],
        transcript_text=" ".join(s["text"] for s in transcript),
        frame_tags=frame_tags if 'frame_tags' in locals() else [],
        llm=dict(provider=st.llm, usage=usage, error=llm_error, stages=stages_log), timings=timings,
    )


def _fallback_summary(caption: str, transcript: list, ocr_lines: list, regex_found: list, gated: dict | None, cands: list) -> str:
    """Generate a summary from local evidence when LLM is unavailable."""
    parts = []
    
    if gated:
        parts.append(f"Creator gates link via '{gated['keyword']}' ({gated['mechanism']})")
    
    if cands:
        top = cands[0]
        verified = " (verified)" if getattr(top, "verification", {}).get("ok") else ""
        parts.append(f"Likely about: {top.name} ({top.kind}){verified}")
        if len(cands) > 1:
            others = [f"{c.name} ({c.kind})" for c in cands[1:3] if getattr(c, "confidence", 0) >= 0.4]
            if others:
                parts.append("Also: " + ", ".join(others))
    elif regex_found:
        names = [c.name for c in regex_found[:3]]
        parts.append("Found: " + ", ".join(names))
    
    if caption and len(caption) > 50:
        cap = caption.strip().replace("\n", " ")[:200]
        parts.append(f"Caption: {cap}...")
    
    return " | ".join(parts) if parts else "No identifiable content found"


def _detect_tutorial_signals(ocr_lines: list, frames: list, transcript: list) -> bool:
    """Detect if frames/OCR suggest a tutorial/walkthrough even without clear text steps."""
    if not ocr_lines and not transcript:
        return False
    
    strong_keywords = {
        "step", "steps", "tutorial", "walkthrough", "how to", "guide", "demo",
        "click", "press", "type", "run", "install", "setup", "configure",
        "add", "create", "generate", "build", "deploy",
        "prompt", "command", "terminal", "console", "dashboard", "settings",
        "next", "then", "after", "finally"
    }
    
    weak_keywords = {"first", "second", "third", "1.", "2.", "3.", "①", "②", "③", "arrow", "→", "->"}
    
    for s in transcript:
        text = s.get("text", "").lower()
        if any(kw in text for kw in strong_keywords):
            return True
        words = text.split()
        for i, w in enumerate(words):
            if w in weak_keywords:
                context = " ".join(words[max(0,i-2):min(len(words),i+3)])
                if any(kw in context for kw in strong_keywords):
                    return True
    
    for l in ocr_lines:
        text = l.get("text", "").lower()
        if any(kw in text for kw in strong_keywords):
            return True
        words = text.split()
        for i, w in enumerate(words):
            if w in weak_keywords:
                context = " ".join(words[max(0,i-2):min(len(words),i+3)])
                if any(kw in context for kw in strong_keywords):
                    return True
        if re.match(r"^\s*[\d①②③④⑤][\.)]\s", text):
            if any(kw in text for kw in strong_keywords | {"click", "press", "run", "type", "install"}):
                return True
    
    if len(frames) >= 12:
        for l in ocr_lines:
            if any(kw in l.get("text", "").lower() for kw in strong_keywords):
                return True
    
    return False