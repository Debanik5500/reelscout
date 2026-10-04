from pathlib import Path

import cv2
import pytest

from reelscout import extract, ocr, reason
from reelscout.config import Settings
from reelscout.pipeline import score

DATA = Path(__file__).parent / "data"


@pytest.mark.skipif(not (__import__("shutil").which("tesseract")), reason="tesseract not installed")
def test_real_frame_ocr_reads_address_bar():
    lines = ocr.ocr_frames([dict(frame_id=0, t=0.0, path=str(DATA / "acquire.jpg"))])
    found = {c.name for l in lines for c in extract.find_in_text(l["text"], "ocr")}
    assert "acquire.com" in found


def test_reason_parses_llm_output_and_fixes_spelling(monkeypatch):
    fake = {"summary": "Finding SaaS ideas.", "mentions": [
        {"name": "trustmrr.com", "kind": "website", "url": "https://trustmrr.com", "alternates": [],
         "spoken_as": "trustmr.com", "confidence": 0.7, "evidence": "speaker says trustmr.com"},
        {"name": "Claude", "kind": "tool", "url": None, "spoken_as": "claw / clawed", "confidence": 0.9,
         "evidence": "share the URL with clawed"}], "unresolved_clues": ["green ceramic mug, no logo"]}
    monkeypatch.setattr(reason, "complete_json", lambda *a, **k: (fake, {"in": 1, "out": 1}))
    summary, cands, _, clues = reason.reason([], [], {"caption": ""}, [], Settings(llm="gemini"))
    assert clues == ["green ceramic mug, no logo"]
    assert {c.name for c in cands} == {"trustmrr.com", "Claude"}
    assert cands[0].spoken_as == "trustmr.com"


def test_score_penalises_dead_domains_and_rewards_corroboration():
    good = extract.Candidate("website", "a.com", confidence=0.5,
                             evidence=[dict(source="ocr", text="a.com", n_frames=3), dict(source="transcript", text="a.com")],
                             verification=dict(ok=True))
    dead = extract.Candidate("website", "b.com", confidence=0.5, evidence=[dict(source="transcript", text="b.com")],
                             verification=dict(ok=False))
    assert score(good) > 0.85 > 0.4 > score(dead)


def _run(monkeypatch, first, second):
    """Runs the real pipeline on a tiny video with ASR/OCR/LLM/verify stubbed; returns (result, list of image-lists the LLM saw)."""
    import tempfile
    from reelscout import asr, pipeline, verify
    seen = []

    def fake_reason(tr, ocr_l, meta, regex_found, st, images=None, gated=None, frame_times=None):
        seen.append(images)
        return "s", (second if images else first), {"in": 10, "out": 5}, (["grey laptop UI, no logo"] if images is None else [])

    monkeypatch.setattr(pipeline.reason, "reason", fake_reason)
    monkeypatch.setattr(pipeline.ocr, "ocr_frames", lambda frames: [])
    monkeypatch.setattr(pipeline.resolve, "resolve", lambda *a, **k: dict(candidates=[], hits=[], image_hits=[], geo=None, errors=[], usage={}))
    monkeypatch.setattr(verify, "verify", lambda c, web=True: setattr(c, "verification", dict(ok=True)))
    st = Settings(llm="gemini", cache_dir=Path(tempfile.mkdtemp()), max_frames=6)
    return pipeline.analyze(str(DATA / "text_video.mp4"), st), seen


def test_weak_text_signals_escalate_to_vision(monkeypatch):
    vis = [extract.Candidate("tool", "Coolify", confidence=0.7, evidence=[dict(source="llm", text="logo in frame 3")])]
    r, seen = _run(monkeypatch, [], vis)
    assert seen[0] is None and seen[1] and len(seen[1]) <= 8       # text pass first, then frames in ONE call
    assert r["used_vision"] and r["candidates"][0]["name"] == "Coolify"
    assert r["unresolved_clues"] == ["grey laptop UI, no logo"]
    assert r["llm"]["usage"] == {"in": 20, "out": 10}              # usage is summed across both calls


def test_strong_text_signals_skip_vision(monkeypatch):
    strong = [extract.Candidate("website", "acquire.com", confidence=0.8, evidence=[dict(source="llm", text="said it")])]
    r, seen = _run(monkeypatch, strong, [])
    assert seen == [None] and not r["used_vision"]


def test_select_ocr_keeps_urls_and_drops_junk():
    lines = [dict(text="acquire.com", t=1.0, n_frames=3, conf=0.9),
             dict(text="~~ // ..", t=1.1, n_frames=1, conf=0.9),
             dict(text="ok", t=1.2, n_frames=1, conf=0.9)] + \
            [dict(text=f"random word {i}", t=2 + i, n_frames=1, conf=0.5) for i in range(100)]
    kept = reason.select_ocr(lines, k=10)
    assert len(kept) == 10 and kept[0]["text"] == "acquire.com"
    assert all(len(l["text"]) >= 4 for l in kept)
