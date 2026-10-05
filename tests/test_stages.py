from pathlib import Path
import tempfile

from reelscout import cli, extract, llm, pipeline, resolve, search
from reelscout.config import Settings

DATA = Path(__file__).parent / "data"


def test_stage_override_and_model_reset(monkeypatch):
    monkeypatch.setenv("REELSCOUT_RESEARCH_LLM", "openai")
    monkeypatch.setenv("REELSCOUT_RESEARCH_MODEL", "gpt-4.1")
    base = Settings(llm="ollama", model="qwen2.5:3b")
    r = base.for_stage("RESEARCH")
    assert (r.llm, r.model) == ("openai", "gpt-4.1")
    assert base.for_stage("TEXT") is base                          # no override -> main settings untouched
    monkeypatch.delenv("REELSCOUT_RESEARCH_MODEL")
    assert base.for_stage("RESEARCH").model == ""                  # new provider, no model named -> provider default, NOT qwen


def test_research_stage_calls_gpt_4_1_even_when_main_llm_is_ollama(monkeypatch):
    monkeypatch.setenv("REELSCOUT_RESEARCH_LLM", "openai")
    monkeypatch.setenv("REELSCOUT_RESEARCH_MODEL", "gpt-4.1")
    seen = []

    def fake_json(system, user, st, images=None):
        seen.append((st.llm, st.model))
        return ({"queries": [{"q": "q", "target": "web"}]} if "queries" in system else {"mentions": []}), {"calls": 1}
    monkeypatch.setattr(resolve, "complete_json", fake_json)
    monkeypatch.setattr(search, "web_search", lambda q, n=5: [dict(title="t", url="u", snippet="")])
    out = resolve.resolve(["clue"], "s", [], Settings(llm="ollama", model="qwen2.5:3b"))
    assert seen == [("openai", "gpt-4.1"), ("openai", "gpt-4.1")] and out["model"] == "openai:gpt-4.1"


def test_text_stage_stays_on_ollama_when_only_research_is_overridden(monkeypatch):
    monkeypatch.setenv("REELSCOUT_RESEARCH_LLM", "openai")
    used = []

    def fake_reason(tr, ocr_l, meta, rf, st, images=None, gated=None, frame_times=None):
        used.append((("vision" if images else "text"), st.llm))
        return "s", [], {"calls": 1}, []
    monkeypatch.setattr(pipeline.reason, "reason", fake_reason)
    monkeypatch.setattr(pipeline.ocr, "ocr_frames", lambda f, workers=1: [])
    monkeypatch.setattr(pipeline.resolve, "resolve", lambda *a, **k: dict(candidates=[], hits=[], image_hits=[], geo=None, errors=[], usage={"calls": 2}, model="openai:gpt-4.1"))
    r = pipeline.analyze(str(DATA / "text_video.mp4"), Settings(llm="ollama", resolve="auto", cache_dir=Path(tempfile.mkdtemp()), max_frames=4), caption="comment LINK please")
    assert used == [("vision", "ollama")]   # weak local evidence -> vision runs directly, text tier skipped as redundant
    assert [s["stage"] for s in r["llm"]["stages"]] == ["vision", "research"]
    assert r["llm"]["stages"][1]["model"] == "openai:gpt-4.1"


def test_cli_llm_flag_drops_model_from_the_old_provider():
    st = Settings(llm="openai", model="gpt-4.1")
    cli.apply_llm_flags(st, "ollama", None)
    assert st.llm == "ollama" and st.model == ""                   # would otherwise send 'gpt-4.1' to Ollama
    st = Settings(llm="openai", model="gpt-4.1")
    cli.apply_llm_flags(st, None, "gpt-4.1-mini")
    assert st.llm == "openai" and st.model == "gpt-4.1-mini"


def _local_evidence(monkeypatch, ocr_lines, transcript):
    from reelscout import media, verify
    monkeypatch.setattr(pipeline.ocr, "ocr_frames", lambda f, workers=1: ocr_lines)
    monkeypatch.setattr(pipeline.media, "extract_audio", lambda v, w: True)
    monkeypatch.setattr(pipeline.asr, "transcribe", lambda *a, **k: transcript)
    monkeypatch.setattr(verify, "verify", lambda c, web=True: setattr(c, "verification", dict(ok=True)))
    monkeypatch.setattr(pipeline.resolve, "resolve", lambda *a, **k: dict(candidates=[], hits=[], image_hits=[], geo=None, errors=[], usage={}, model=""))


BOTH = ([dict(text="acquire.com", t=1.0, n_frames=3, conf=0.9)], [dict(start=0, end=2, text="go to acquire dot com")])


def _run_policy(monkeypatch, ocr_lines, transcript, policy, **over):
    _local_evidence(monkeypatch, ocr_lines, transcript)
    calls = []
    monkeypatch.setattr(pipeline.reason, "reason", lambda *a, **k: (calls.append(1), ("s", [], {"calls": 1}, []))[1])
    st = Settings(llm="openai", text_policy=policy, cache_dir=Path(tempfile.mkdtemp()), max_frames=4, **over)
    return pipeline.analyze(str(DATA / "text_video.mp4"), st), calls


def test_decisive_local_evidence_skips_the_llm(monkeypatch):
    r, calls = _run_policy(monkeypatch, *BOTH, "auto")
    assert calls == [] and r["status"] == "identified" and r["candidates"][0]["name"] == "acquire.com"
    assert r["llm"]["stages"][0] == dict(stage="text", model="local (skipped: decisive)", calls=0)


def test_policy_always_forces_the_llm_even_when_decisive(monkeypatch):
    r, calls = _run_policy(monkeypatch, *BOTH, "always")
    assert calls == [1]


def test_persistent_verified_single_source_counts_as_decisive(monkeypatch):
    r, calls = _run_policy(monkeypatch, BOTH[0], [], "auto")        # OCR-only but seen on 3 frames + verified
    assert calls == [] and r["llm"]["stages"][0] == dict(stage="text", model="local (skipped: decisive)", calls=0)


def test_policy_never_skips_text_call_and_local_only_config_stays_fully_local(monkeypatch):
    r, calls = _run_policy(monkeypatch, [], [], "never", vision="off", resolve="off")
    assert calls == [] and r["status"] == "unidentified"


def test_policy_never_still_lets_weak_reels_escalate_to_frames(monkeypatch):
    r, calls = _run_policy(monkeypatch, [], [], "never")            # nothing local -> weak -> vision tier is allowed
    assert calls == [1] and r["used_vision"]
