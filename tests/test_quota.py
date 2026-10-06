import pytest

from server import quota


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(quota, "QUOTA_FILE", tmp_path / "quota.json")
    monkeypatch.delenv("REELSCOUT_UNLIMITED_DEVICES", raising=False)
    monkeypatch.delenv("REELSCOUT_QUOTA", raising=False)


FREE_LIMIT = next(p["monthly_reels"] for p in quota.load_plans()["plans"] if p["id"] == "free")


def test_free_plan_allows_its_limit_then_blocks():
    for i in range(FREE_LIMIT):
        quota.reserve("dev-1", f"reel-{i}")
    assert quota.status("dev-1")["remaining"] == 0
    with pytest.raises(quota.QuotaExceeded) as err:
        quota.reserve("dev-1", "reel-over")
    assert err.value.status["limit"] == FREE_LIMIT and err.value.status["used"] == FREE_LIMIT


def test_same_reel_is_not_counted_twice():
    quota.reserve("dev-1", "reel-a")
    quota.reserve("dev-1", "reel-a")
    assert quota.status("dev-1")["used"] == 1


def test_failed_reel_is_refunded():
    quota.reserve("dev-1", "reel-a")
    quota.refund("dev-1", "reel-a")
    assert quota.status("dev-1")["used"] == 0


def test_devices_are_independent_and_missing_id_is_unmetered():
    for i in range(FREE_LIMIT):
        quota.reserve("dev-1", f"reel-{i}")
    assert quota.status("dev-2")["used"] == 0
    assert quota.status(None)["limit"] is None
    quota.reserve(None, "x")  # does not raise


def test_unlimited_device_env(monkeypatch):
    monkeypatch.setenv("REELSCOUT_UNLIMITED_DEVICES", "owner")
    assert quota.status("owner")["limit"] is None


def test_plans_file_has_a_free_plan():
    plans = quota.load_plans()
    free = next(p for p in plans["plans"] if p["id"] == plans["free_plan"])
    assert free["monthly_reels"] >= 1 and free["price_usd_month"] == 0  # the number itself is the owner's choice


def test_failed_reel_errors_are_plain_language():
    from server.jobs import friendly_error
    assert "private" in friendly_error(RuntimeError("Fetching Post metadata failed."))
    assert "private" in friendly_error(RuntimeError("Instagram sent an empty media response"))
    assert "limiting" in friendly_error(RuntimeError("HTTP Error 429: Too Many Requests"))
    assert "isn't a reel" in friendly_error(RuntimeError("Unsupported URL: https://x"))
    assert "went wrong" in friendly_error(RuntimeError("segfault in thingamajig"))
