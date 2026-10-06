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
    assert free["monthly_reels"] == 5 and free["price_usd_month"] == 0
