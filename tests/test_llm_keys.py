import pytest

from reelscout import llm
from reelscout.config import Settings


class FakeResponse:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body or {}, headers or {}
        self.text = str(self._body)

    def json(self):
        return self._body


def ok():
    return FakeResponse(200, {"choices": [{"message": {"content": "{\"a\": 1}"}}],
                              "usage": {"prompt_tokens": 5, "completion_tokens": 2}})


def quota_hit(seconds=700):
    return FakeResponse(429, {"error": "TPD"}, {"retry-after": str(seconds)})


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    llm._pools.clear()
    llm._blocked_until.clear()
    monkeypatch.setenv("GROQ_API_KEYS", "k1,k2,k3")
    monkeypatch.setenv("GROQ_API_KEY", "k1")
    monkeypatch.setenv("REELSCOUT_FALLBACK_LLM", "off")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)


def settings():
    return Settings(llm="groq", model="m")


def used_keys(monkeypatch, responses):
    seen = []

    def fake_post(url, headers, json, timeout):
        seen.append(headers["Authorization"].replace("Bearer ", ""))
        return responses.pop(0)

    monkeypatch.setattr(llm.requests, "post", fake_post)
    return seen


def test_calls_rotate_across_keys(monkeypatch):
    seen = used_keys(monkeypatch, [ok(), ok(), ok(), ok()])
    for _ in range(4):
        llm.complete_json("s", "u", settings())
    assert seen == ["k1", "k2", "k3", "k1"]


def test_exhausted_key_is_skipped_within_the_same_call(monkeypatch):
    seen = used_keys(monkeypatch, [quota_hit(), ok()])
    data, _ = llm.complete_json("s", "u", settings())
    assert data == {"a": 1} and seen == ["k1", "k2"]


def test_blocked_key_stays_skipped_on_later_calls(monkeypatch):
    seen = used_keys(monkeypatch, [quota_hit(), ok(), ok(), ok()])
    for _ in range(3):
        llm.complete_json("s", "u", settings())
    assert "k1" not in seen[2:]


def test_all_keys_exhausted_raises_rate_limited(monkeypatch):
    used_keys(monkeypatch, [quota_hit(), quota_hit(), quota_hit()])
    with pytest.raises(llm.RateLimited):
        llm.complete_json("s", "u", settings())


def test_duplicate_and_blank_keys_are_ignored(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEYS", " k1 , , k2;k2 ")
    assert llm._api_keys(settings()) == ["k1", "k2"]
