"""Tests for visual identification (Stage 9, path A)."""

import tempfile
from pathlib import Path

from reelscout import extract, pipeline, verify, visualid
from reelscout.config import Settings

DATA = Path(__file__).parent / "data"


def _settings(**kw):
    base = dict(llm="gemini", resolve="auto",
                cache_dir=Path(tempfile.mkdtemp()))
    base.update(kw)
    return Settings(**base)


def test_visualid_observes_fingerprints_then_adjudicates(monkeypatch):
    seen = {}

    def fake_json(system, user, st, images=None):
        if "OBSERVE" in system:  # vision pass
            seen["images"] = images
            return {"observation": "turquoise circular pool, wooden footbridge",
                    "fingerprint": [{"feature": "bright turquoise circular pool",
                                     "level": "high"}],
                    "hypotheses": [{"candidate": "Blue Lagoon",
                                    "why": "geothermal spring",
                                    "supporting": ["milky blue water"],
                                    "missing": "no signage read",
                                    "disproof": "no silica terraces"}],
                    "queries": [{"q": "turquoise circular pool wooden footbridge",
                                 "target": "web"}]}, {"in": 5, "out": 5}
        seen["user"] = user
        return {"mentions": [{
            "name": "Blue Lagoon", "kind": "place",
            "url": "https://example.com/blue",
            "confidence": 0.7, "confidence_level": "LIKELY",
            "alternates": ["Bláa Lónið"],
            "feature_check": [{"feature": "bright turquoise circular pool",
                               "result": "match"}],
            "evidence": "photo match",
            "alternatives": ["Pamukkale — no white terraces"],
            "uncertainty": "signage unreadable"}]}, {"in": 7, "out": 3}

    monkeypatch.setattr(visualid, "complete_json", fake_json)
    monkeypatch.setattr(verify, "web_search",
                        lambda q, n=5: [dict(title="Blue Lagoon photos",
                                             url="https://example.com/blue",
                                             snippet="turquoise lagoon")])
    out = visualid.identify([], "someone swims",
                            [{"path": "/tmp/f1.jpg", "t": 1.0}], _settings())

    assert seen["images"] == ["/tmp/f1.jpg"]        # frames go to the vision pass
    assert "bright turquoise circular pool" in seen["user"]  # fingerprint drives adjudication
    assert "Blue Lagoon photos" in seen["user"]      # search hits included
    assert "someone swims" in seen["user"]           # transcript kept as context

    c = out["candidates"][0]
    assert c.name == "Blue Lagoon" and c.kind == "place"
    assert c.sources == {"search"}
    assert c.extra["confidence_level"] == "LIKELY"
    assert c.extra["feature_check"][0]["result"] == "match"
    assert c.extra["alternatives"] == ["Pamukkale — no white terraces"]
    assert out["fingerprint"][0]["level"] == "high"
    assert out["usage"] == {"in": 12, "out": 8}


def test_visualid_drops_unknown_and_low_confidence(monkeypatch):
    def fake_json(system, user, st, images=None):
        if "OBSERVE" in system:
            return {"queries": [{"q": "rare feature combo", "target": "web"}]}, {"in": 1, "out": 1}
        return {"mentions": [
            {"name": "Kept", "kind": "product", "confidence": 0.5,
             "confidence_level": "POSSIBLE"},
            {"name": "Dropped", "kind": "product", "confidence": 0.4,
             "confidence_level": "POSSIBLE"},
            {"name": "Unknown", "kind": "product", "confidence": 0.9,
             "confidence_level": "UNKNOWN"},
        ]}, {"in": 2, "out": 2}

    monkeypatch.setattr(visualid, "complete_json", fake_json)
    monkeypatch.setattr(verify, "web_search",
                        lambda q, n=5: [dict(title="t", url="https://example.com",
                                             snippet="s")])
    out = visualid.identify([], "", [], _settings())
    assert [c.name for c in out["candidates"]] == ["Kept"]


def test_visualid_without_llm_returns_raw_hits(monkeypatch):
    def fake_json(system, user, st, images=None):
        return {"queries": [{"q": "rare feature combo", "target": "web"}]}, {"in": 1, "out": 1}

    monkeypatch.setattr(visualid, "complete_json", fake_json)
    monkeypatch.setattr(verify, "web_search",
                        lambda q, n=5: [dict(title="Hit", url="https://example.com",
                                             snippet="s")])
    out = visualid.identify([], "", [], _settings(llm="none"))
    assert out["candidates"] == []
    assert out["hits"][0]["title"] == "Hit"
    assert out["queries"] == ["rare feature combo"]


def test_visualid_search_failure_is_captured(monkeypatch):
    def fake_json(system, user, st, images=None):
        return {"queries": [{"q": "boom", "target": "web"}]}, {"in": 1, "out": 1}

    monkeypatch.setattr(visualid, "complete_json", fake_json)
    monkeypatch.setattr(verify, "web_search",
                        lambda q, n=5: (_ for _ in ()).throw(RuntimeError("search down")))
    out = visualid.identify([], "", [], _settings(llm="none"))
    assert out["candidates"] == []
    assert any("boom" in e for e in out["errors"])


def test_visualid_routes_maps_and_github_targets(monkeypatch):
    calls = {"maps": 0, "github": 0}

    def fake_json(system, user, st, images=None):
        return {"queries": [{"q": "turquoise pool mountain", "target": "maps"},
                            {"q": "rare widget", "target": "github"}]}, {"in": 1, "out": 1}

    def fake_geocode(q):
        calls["maps"] += 1
        return dict(ok=True, display_name="Nowhere", prob=0.5,
                    map_url="https://maps.example.com")

    def fake_gh(q, n=5):
        calls["github"] += 1
        return [dict(full_name="a/b", description="d",
                     url="https://github.com/a/b", stars=1)]

    monkeypatch.setattr(visualid, "complete_json", fake_json)
    monkeypatch.setattr(verify, "geocode", fake_geocode)
    monkeypatch.setattr(verify, "github_search_text", fake_gh)
    out = visualid.identify([], "", [], _settings(llm="none"))
    assert calls == {"maps": 1, "github": 1}
    assert any("maps.example.com" in h["url"] for h in out["hits"])
    assert any("github.com/a/b" in h["url"] for h in out["hits"])


def test_pipeline_prefers_visual_identification_over_resolve(monkeypatch):
    monkeypatch.setattr(pipeline.reason, "reason", lambda *a, **k: ("s", [], {}, ["clue"]))
    monkeypatch.setattr(pipeline.ocr, "ocr_frames", lambda frames, workers=1: [])
    monkeypatch.setattr(pipeline.verify, "verify",
                        lambda c: setattr(c, "verification", dict(ok=True)))

    def fake_visualid(clues, summary, frames, st, transcript, ocr_lines, caption):
        c = extract.Candidate("place", "Blue Lagoon", url="https://example.com",
                              confidence=0.7,
                              evidence=[dict(source="search", text="x")])
        return {"candidates": [c], "hits": [dict(title="t", url="u", snippet="s")],
                "queries": ["q"], "fingerprint": [{"feature": "f", "level": "high"}],
                "hypotheses": [], "errors": [], "usage": {"in": 1, "out": 1},
                "model": "fake"}

    monkeypatch.setattr(pipeline.visualid, "identify", fake_visualid)
    monkeypatch.setattr(pipeline.resolve, "resolve",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("resolve must not run")))

    r = pipeline.analyze(str(DATA / "text_video.mp4"), _settings(max_frames=4))
    assert r["resolution"]["ran"] is True
    assert r["resolution"]["fingerprint"] == [{"feature": "f", "level": "high"}]
    assert any(c["name"] == "Blue Lagoon" for c in r["candidates"])


def test_pipeline_falls_back_to_resolve_when_visualid_empty(monkeypatch):
    monkeypatch.setattr(pipeline.reason, "reason", lambda *a, **k: ("s", [], {}, ["clue"]))
    monkeypatch.setattr(pipeline.ocr, "ocr_frames", lambda frames, workers=1: [])
    monkeypatch.setattr(pipeline.visualid, "identify",
                        lambda *a, **k: {"candidates": [], "hits": [], "queries": [],
                                         "fingerprint": [], "hypotheses": [],
                                         "errors": [], "usage": {}, "model": "fake"})
    seen = {}

    def fake_resolve(clues, summary, frames, st, transcript, ocr_lines, caption):
        seen["clues"] = clues
        return {"candidates": [], "hits": [], "queries": [], "image_hits": [],
                "geo": None, "errors": [], "usage": {"in": 0, "out": 0},
                "model": "fake"}

    monkeypatch.setattr(pipeline.resolve, "resolve", fake_resolve)
    r = pipeline.analyze(str(DATA / "text_video.mp4"), _settings(max_frames=4))
    assert seen["clues"] == ["clue"]
    assert r["resolution"]["ran"] is True


def test_pipeline_skips_visual_identification_when_vision_off(monkeypatch):
    monkeypatch.setattr(pipeline.reason, "reason", lambda *a, **k: ("s", [], {}, ["clue"]))
    monkeypatch.setattr(pipeline.ocr, "ocr_frames", lambda frames, workers=1: [])
    monkeypatch.setattr(pipeline.visualid, "identify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("visualid must not run")))
    monkeypatch.setattr(pipeline.resolve, "resolve",
                        lambda *a, **k: {"candidates": [], "hits": [], "queries": [],
                                         "image_hits": [], "geo": None,
                                         "errors": [], "usage": {}, "model": "fake"})
    r = pipeline.analyze(str(DATA / "text_video.mp4"),
                         _settings(max_frames=4, vision="off"))
    assert r["resolution"]["ran"] is True
