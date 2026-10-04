from reelscout import evaluate as ev


def cand(name, conf, url=None):
    return dict(name=name, url=url, confidence=conf)


def test_matching_is_forgiving_about_form():
    assert ev.matches("coollabsio/coolify", cand("coolify", 0.7))
    assert ev.matches("acquire.com", cand("acquire.com", 0.9, "https://www.acquire.com/"))
    assert ev.matches("https://github.com/foo/bar.git", cand("foo/bar", 0.7))
    assert not ev.matches("acquire.com", cand("trustmrr.com", 0.9))


def test_wrong_confident_is_tracked_separately_from_abstaining():
    wrong = ev.score_reel(["acquire.com"], dict(candidates=[cand("other.com", 0.9)]))
    assert not wrong["correct"] and wrong["wrong_confident"] and not wrong["abstained"]
    unsure = ev.score_reel(["acquire.com"], dict(candidates=[cand("other.com", 0.3)]))
    assert not unsure["wrong_confident"] and unsure["abstained"]


def test_bucket_e_success_means_abstaining():
    assert ev.score_reel([], dict(candidates=[]))["correct"]
    assert not ev.score_reel([], dict(candidates=[cand("guess.com", 0.9)]))["correct"]


def test_summary_groups_by_bucket():
    rows = [dict(bucket="A", correct=True, top3=True, abstained=False, wrong_confident=False),
            dict(bucket="A", correct=False, top3=True, abstained=True, wrong_confident=False),
            dict(bucket="C", correct=False, top3=False, abstained=False, wrong_confident=True)]
    s = ev.summarize(rows)
    assert s["A"]["top1"] == 0.5 and s["C"]["wrong_confident"] == 1.0 and s["ALL"]["n"] == 3
