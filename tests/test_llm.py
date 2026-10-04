import pytest

from reelscout import llm


class Resp:
    def __init__(self, j): self._j = j
    def json(self): return self._j


def test_preflight_reports_missing_model_instantly(monkeypatch):
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: Resp({"models": [{"name": "llama3.2:3b"}]}))
    llm._ollama_checked.clear()
    try:
        llm.ollama_preflight("qwen2.5:3b", "http://localhost:11434/v1")
        assert False, "should have raised"
    except llm.LLMError as e:
        assert "ollama pull qwen2.5:3b" in str(e) and "llama3.2:3b" in str(e)


def test_preflight_reports_server_down(monkeypatch):
    def boom(*a, **k): raise ConnectionError("refused")
    monkeypatch.setattr(llm.requests, "get", boom)
    llm._ollama_checked.clear()
    try:
        llm.ollama_preflight("qwen2.5:3b", "http://localhost:11434/v1")
        assert False
    except llm.LLMError as e:
        assert "not reachable" in str(e)


def test_preflight_accepts_installed_model_with_or_without_tag(monkeypatch):
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: Resp({"models": [{"name": "qwen2.5:3b"}]}))
    llm._ollama_checked.clear()
    llm.ollama_preflight("qwen2.5:3b", "http://localhost:11434/v1")
    llm._ollama_checked.clear()
    llm.ollama_preflight("qwen2.5", "http://localhost:11434/v1")


class PostResp:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._p, self.text = status, payload, text
    def json(self): return self._p


OK = {"choices": [{"message": {"content": '{"mentions": []}'}}], "usage": {"prompt_tokens": 11, "completion_tokens": 7}}


def _capture(monkeypatch):
    sent = []
    monkeypatch.setattr(llm.requests, "post", lambda url, headers=None, json=None, timeout=None: (sent.append((url, headers, json)), PostResp(200, OK))[1])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return sent


def test_openai_default_is_gpt_4_1_with_classic_params(monkeypatch):
    from reelscout.config import Settings
    sent = _capture(monkeypatch)
    data, usage = llm.complete_json("sys JSON", "user", Settings(llm="openai"))
    url, headers, body = sent[0]
    assert url == "https://api.openai.com/v1/chat/completions" and headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "gpt-4.1" and body["temperature"] == 0 and body["max_tokens"] == 1200
    assert usage == {"in": 11, "out": 7, "calls": 1}


def test_openai_reasoning_family_uses_max_completion_tokens(monkeypatch):
    from reelscout.config import Settings
    sent = _capture(monkeypatch)
    llm.complete_json("sys JSON", "user", Settings(llm="openai", model="gpt-5.4-mini"))
    body = sent[0][2]
    assert "max_completion_tokens" in body and "temperature" not in body and "max_tokens" not in body


def test_adapts_when_model_rejects_max_tokens(monkeypatch):
    from reelscout.config import Settings
    bodies = []
    def post(url, headers=None, json=None, timeout=None):
        bodies.append(dict(json))
        if "max_tokens" in json:
            return PostResp(400, text="Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.")
        return PostResp(200, OK)
    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setenv("GROQ_API_KEY", "k")
    llm.complete_json("sys JSON", "user", Settings(llm="groq"))
    assert "max_tokens" in bodies[0] and "max_completion_tokens" in bodies[1] and "max_tokens" not in bodies[1]


def test_missing_openai_key_is_a_clear_error(monkeypatch):
    from reelscout.config import Settings
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        llm.complete_json("s", "u", Settings(llm="openai"))
        assert False
    except llm.LLMError as e:
        assert "OPENAI_API_KEY" in str(e)
