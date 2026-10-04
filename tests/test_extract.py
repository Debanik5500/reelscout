import json
from pathlib import Path

from reelscout import extract

D = json.loads((Path(__file__).parent / "data" / "saas_reel.json").read_text())


def names(cands):
    return {c.name for c in cands}


def test_real_reel_regex_finds_spoken_and_onscreen_urls():
    found = names(extract.gather(D["transcript"], D["ocr"]))
    assert "acquire.com" in found          # spoken
    assert "trustmr.com" in found          # spoken (ASR spelling; LLM/verify fix it to trustmrr.com)
    assert "reddit.com" in found           # in the browser address bar (OCR)


def test_spoken_dot_com():
    assert names(extract.find_in_text("go to acquire dot com now", "transcript")) == {"acquire.com"}


def test_github_spoken_and_written():
    c = extract.find_in_text("check github dot com slash coollabsio slash coolify", "transcript")
    assert [(x.kind, x.name) for x in c] == [("github_repo", "coollabsio/coolify")]
    c = extract.find_in_text("https://github.com/foo/bar.git", "ocr")
    assert c[0].name == "foo/bar"


def test_ignores_noise():
    assert extract.find_in_text("follow us on instagram.com or mail me@gmail.com or open index.js", "ocr") == []


def test_merge_counts_sources():
    a = extract.find_in_text("acquire.com", "ocr", 1.0, 3)
    b = extract.find_in_text("acquire dot com", "transcript", 2.0)
    m = extract.merge(a + b)
    assert len(m) == 1 and m[0].sources == {"ocr", "transcript"}


def test_gating_detected():
    g = extract.detect_gating(["Want the link? Comment LINK below and I'll DM you"])
    assert g and g["keyword"] == "LINK"
    g = extract.detect_gating(["comment the word \"GUIDE\" to get it"])
    assert g["keyword"] == "GUIDE"
    assert extract.detect_gating(["leave a comment on this if you agree"]) is None


def test_social_candidates_tagged_mentions_and_creator_comment():
    meta = dict(uploader="travelguy", caption="Stayed at @aman_tokyo, shot with @travelguy", mentions=[], tagged=["aman_tokyo"],
                comments=[dict(author="travelguy", text="booking at amanresorts.com", by_creator=True),
                          dict(author="x", text="where??", by_creator=False)])
    cands = extract.merge(extract.social_candidates(meta))
    got = {(c.kind, c.name) for c in cands}
    assert ("brand", "@aman_tokyo") in got and ("website", "amanresorts.com") in got
    assert ("brand", "@travelguy") not in got          # the uploader is not a finding


def test_install_commands_from_screen_and_speech():
    got = {(c.extra["ecosystem"], c.name) for c in extract.find_commands("$ pip install -U browser-use", "ocr")}
    assert got == {("pypi", "browser-use")}
    got = {(c.extra["ecosystem"], c.name) for c in extract.find_commands("just run npx create-next-app and then npm i zustand", "transcript")}
    assert got == {("npm", "create-next-app"), ("npm", "zustand")}
    assert extract.find_commands("pip install -r requirements.txt", "ocr") == []
    assert extract.find_commands("brew install ffmpeg", "ocr")[0].extra["ecosystem"] == "brew"


def test_asr_misspelling_folds_into_the_spelling_seen_on_screen():
    spoken = extract.find_in_text("check out trustmr.com", "transcript", 3.0)
    seen = extract.find_in_text("trustmrr.com/startups", "ocr", 5.0, 4)
    m = extract.merge(spoken + seen)
    assert [c.name for c in m] == ["trustmrr.com"]
    assert m[0].spoken_as == "trustmr.com" and m[0].sources == {"transcript", "ocr"}


def test_two_different_sites_seen_on_screen_are_not_merged():
    a = extract.find_in_text("notion.so", "ocr")
    b = extract.find_in_text("motion.so", "ocr")
    assert len(extract.merge(a + b)) == 2
