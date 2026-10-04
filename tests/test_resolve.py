import tempfile
from pathlib import Path

from reelscout import extract, imagesearch, pipeline, resolve, search, verify
from reelscout.config import Settings

DATA = Path(__file__).parent / "data"


def test_search_backend_fallback_order(monkeypatch):
    calls = []
    def boom(q, n): calls.append("searxng"); raise RuntimeError("down")
    def empty(q, n): calls.append("ddgs"); return []
    def ok(q, n): calls.append("html"); return [dict(title="t", url="u", snippet="")]
    monkeypatch.setattr(search, "BACKENDS", [boom, empty, ok])
    assert search.web_search("x")[0]["title"] == "t" and calls == ["searxng", "ddgs", "html"]


def test_optional_libs_missing_degrade_gracefully():
    hits, errs = imagesearch.reverse_search("nope.jpg")          # PicImageSearch not installed in the sandbox
    assert hits == [] and errs and "PicImageSearch" in errs[0]
    assert imagesearch.geolocate(["nope.jpg"]) is None           # geoclip not installed


def test_resolve_turns_clues_into_candidate_via_search(monkeypatch):
    def fake_json(system, user, st, images=None):
        if "web search queries" in system:
            return {"queries": [{"q": "deployment dashboard projects sidebar self-hosted", "target": "github"}]}, {"in": 5, "out": 5}
        assert "coollabsio/coolify" in user                       # adjudicator sees the search hit
        return {"mentions": [{"name": "coollabsio/coolify", "kind": "github_repo", "url": "https://github.com/coollabsio/coolify",
                              "confidence": 0.7, "evidence": "repo description matches 'self-hosted' + 'deployments' clues"}]}, {"in": 7, "out": 3}
    monkeypatch.setattr(resolve, "complete_json", fake_json)
    monkeypatch.setattr(verify, "github_search_text", lambda q, n=5: [dict(full_name="coollabsio/coolify", url="https://github.com/coollabsio/coolify", stars=40000, description="self-hosted PaaS")])
    out = resolve.resolve(["dashboard with Projects and Deployments sidebar"], "a deploy tool demo", [], Settings(llm="gemini"))
    assert out["candidates"][0].name == "coollabsio/coolify" and out["candidates"][0].sources == {"search"}
    assert out["usage"] == {"in": 12, "out": 8}


def test_resolve_without_llm_returns_raw_hits_only(monkeypatch):
    monkeypatch.setattr(search, "web_search", lambda q, n=5: [dict(title="Some result", url="https://x.dev", snippet="")])
    out = resolve.resolve(["green ceramic mug with handle"], "", [], Settings(llm="none"))
    assert out["candidates"] == [] and out["hits"][0]["query"] == "green ceramic mug with handle"


def test_adjudicator_may_return_nothing(monkeypatch):
    monkeypatch.setattr(resolve, "complete_json", lambda s, u, st, images=None: ({"queries": [{"q": "q", "target": "web"}]} if "queries" in s else {"mentions": []}, {}))
    monkeypatch.setattr(search, "web_search", lambda q, n=5: [dict(title="unrelated", url="u", snippet="")])
    assert resolve.resolve(["c"], "", [], Settings(llm="gemini"))["candidates"] == []


def test_pipeline_runs_resolve_tier_when_everything_else_is_weak(monkeypatch):
    def fake_reason(tr, ocr_l, meta, rf, st, images=None, gated=None, frame_times=None):
        return "s", [], {}, ["clue"]
    found = extract.Candidate("github_repo", "a/b", confidence=0.7, evidence=[dict(source="search", text="match")])
    monkeypatch.setattr(pipeline.reason, "reason", fake_reason)
    monkeypatch.setattr(pipeline.ocr, "ocr_frames", lambda f: [])
    monkeypatch.setattr(pipeline.resolve, "resolve", lambda clues, s, fr, st: dict(candidates=[found], hits=[dict(title="t", url="u", query="q")], image_hits=[], geo=None, errors=[], usage={"in": 1}))
    monkeypatch.setattr(verify, "verify", lambda c, web=True: setattr(c, "verification", dict(ok=True)))
    r = pipeline.analyze(str(DATA / "text_video.mp4"), Settings(llm="gemini", cache_dir=Path(tempfile.mkdtemp()), max_frames=4))
    assert r["resolution"]["ran"] and r["candidates"][0]["name"] == "a/b" and r["candidates"][0]["confidence"] >= 0.65


def _analyze_stubbed(monkeypatch, caption, seen):
    from reelscout import ocr
    monkeypatch.setattr(pipeline.reason, "reason", lambda *a, **k: ("s", [], {}, []))
    monkeypatch.setattr(ocr, "ocr_frames", lambda f: [])
    def fake_resolve(clues, summary, frames, st):
        seen.append(st.image_search)
        return dict(candidates=[], hits=[], image_hits=[], geo=None, errors=[], usage={})
    monkeypatch.setattr(pipeline.resolve, "resolve", fake_resolve)
    st = Settings(llm="gemini", cache_dir=Path(tempfile.mkdtemp()), max_frames=4)
    return pipeline.analyze(str(DATA / "text_video.mp4"), st, caption=caption)


def test_gated_reel_turns_on_image_search_and_reports_unidentified(monkeypatch):
    seen = []
    r = _analyze_stubbed(monkeypatch, "Comment LINK and I'll DM you the link", seen)
    assert seen == [True] and r["gated_link"]["keyword"] == "LINK" and r["status"] == "unidentified"


def test_ungated_reel_does_not_force_image_search(monkeypatch):
    seen = []
    pipeline_result = _analyze_stubbed(monkeypatch, "nice day", seen)
    assert seen == []                  # no clues, not gated -> resolve tier not even entered


def test_reader_falls_back_to_direct_fetch_and_strips_html(monkeypatch):
    from reelscout import reader

    class R:
        def __init__(self, text, ok=True): self.text, self._ok = text, ok
        def raise_for_status(self):
            if not self._ok: raise RuntimeError("blocked")

    def fake_get(url, **kw):
        if url.startswith("https://r.jina.ai/"):
            return R("", ok=False)
        return R("<html><script>x()</script><p>" + "Coolify is a self-hosted PaaS. " * 5 + "</p></html>")
    monkeypatch.setattr(reader.requests, "get", fake_get)
    out = reader.page_text("https://example.org")
    assert out.startswith("Coolify is a self-hosted PaaS.") and "<" not in out and "x()" not in out
