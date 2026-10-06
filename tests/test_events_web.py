from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest

from reelscout import events, screenshots, webpage

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 10, 7, 14, 0, tzinfo=IST)


# ---- events -------------------------------------------------------------------------------------------

def test_event_read_in_users_timezone():
    e = events.normalize_event({"title": "Tech Meetup", "start": "2026-11-14T18:30", "confidence": 0.9}, NOW, "Asia/Kolkata")
    assert e["start"] == "2026-11-14T18:30+05:30" and not e["needs_check"] and e["timezone"] == "Asia/Kolkata"


def test_stated_timezone_beats_users():
    e = events.normalize_event({"title": "Webinar", "start": "2026-11-14T20:00", "tz": "America/Los_Angeles",
                                "confidence": 0.9}, NOW, "Asia/Kolkata")
    assert e["start"] == "2026-11-14T20:00-08:00"


def test_missing_year_rolls_to_next_occurrence():
    e = events.normalize_event({"title": "Diwali party", "start": "2026-03-01T19:00", "year_inferred": True,
                                "confidence": 0.9}, NOW, "Asia/Kolkata")
    assert e["start"].startswith("2027-03-01") and e["needs_check"]


def test_date_only_is_all_day_and_flagged():
    e = events.normalize_event({"title": "Sale ends", "start": "2026-10-20", "confidence": 0.9}, NOW, "Asia/Kolkata")
    assert e["all_day"] and e["needs_check"] and e["start"].startswith("2026-10-20T00:00")


def test_garbage_and_implausible_dates_are_dropped():
    for start in ("", "tomorrow evening", "2019-01-01T10:00", "2099-01-01T10:00", "2026-13-45T10:00"):
        assert events.normalize_event({"title": "x", "start": start}, NOW, "Asia/Kolkata") is None
    assert events.normalize_event({"title": "", "start": "2026-11-14T18:00"}, NOW, "Asia/Kolkata") is None


def test_offset_in_structured_data_is_kept():
    e = events.normalize_event({"title": "Match", "start": "2026-11-14T14:00:00Z", "confidence": 0.95}, NOW, "Asia/Kolkata")
    assert e["start"] == "2026-11-14T14:00+00:00"


def test_end_before_start_is_ignored_and_events_sorted_deduped():
    raw = [{"title": "B", "start": "2026-11-15T10:00", "end": "2026-11-15T09:00", "confidence": 0.9},
           {"title": "A", "start": "2026-11-14T10:00", "confidence": 0.9},
           {"title": "A", "start": "2026-11-14T10:00", "confidence": 0.9}]
    out = events.normalize_events(raw, NOW, "Asia/Kolkata")
    assert [e["title"] for e in out] == ["A", "B"] and out[1]["end"] == ""


# ---- address safety -----------------------------------------------------------------------------------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://10.0.0.5/admin", "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data",
    "http://[::1]/", "file:///etc/passwd", "ftp://example.com/x", "http://user:pw@example.com/",
    "http://example.com:8080/", "http://localhost/", "http://printer.local/", "http://2130706433/", "", "javascript:alert(1)",
])
def test_unsafe_urls_rejected(url):
    with pytest.raises(webpage.UnsafeURL):
        webpage.check_url(url)


def test_public_ip_literal_allowed():
    assert webpage.check_url("https://8.8.8.8/x") == "https://8.8.8.8/x"


def test_redirect_to_private_address_is_blocked(monkeypatch):
    class Redirect:
        is_redirect = True
        headers = {"location": "http://127.0.0.1/secret"}

        def close(self):
            pass

    monkeypatch.setattr(webpage.requests.Session, "get", lambda self, *a, **k: Redirect())
    with pytest.raises(webpage.UnsafeURL):
        webpage.fetch("https://8.8.8.8/start")


def test_normalize_url_drops_tracking_and_fragment():
    a = webpage.normalize_url("https://Example.com/path/?utm_source=x&b=2&a=1#top")
    assert a == "https://example.com/path?a=1&b=2"
    assert webpage.job_id_for("https://example.com/p?utm_medium=z") == webpage.job_id_for("https://example.com/p")


def test_google_search_links_are_recognised():
    assert webpage.search_query("https://www.google.com/search?q=india+cricket+schedule&oq=x") == "india cricket schedule"
    assert webpage.search_query("https://www.google.co.in/search?q=ind+vs+aus") == "ind vs aus"
    assert webpage.search_query("https://example.com/search?q=hello") is None
    assert webpage.search_query("https://www.google.com/maps") is None


# ---- page reading -------------------------------------------------------------------------------------

HTML = """<html><head><title>Fixtures</title><meta property="og:description" content="All matches">
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[
 {"@type":"SportsEvent","name":"IND vs AUS, 1st T20","startDate":"2026-11-14T19:30:00+05:30",
  "location":{"@type":"Place","name":"Wankhede Stadium","address":{"streetAddress":"D Rd","addressLocality":"Mumbai"}}},
 {"@type":"Organization","name":"x"}]}</script><style>.a{}</style></head>
<body><nav>menu</nav><h1>Schedule</h1><script>var secret=1</script>
<table><tr><td>14 Nov</td><td>IND v AUS</td></tr></table></body></html>"""


def test_page_reader_gets_text_table_and_structured_events():
    page = webpage.extract_page(HTML)
    assert page["title"] == "Fixtures" and page["description"] == "All matches"
    assert "secret" not in page["text"] and "var " not in page["text"]
    assert "14 Nov | IND v AUS" in page["text"].replace("\n", "")
    assert page["ld_events"][0]["title"] == "IND vs AUS, 1st T20"
    assert "Wankhede Stadium" in page["ld_events"][0]["location"] and "Mumbai" in page["ld_events"][0]["location"]
    e = events.normalize_event(page["ld_events"][0], NOW, "Asia/Kolkata")
    assert e["start"] == "2026-11-14T19:30+05:30"


# ---- screenshots --------------------------------------------------------------------------------------

def png(width=300, height=200, value=200):
    return cv2.imencode(".png", np.full((height, width, 3), value, np.uint8))[1].tobytes()


def test_screenshot_ids_ignore_order_and_differ_for_different_images():
    a, b = png(value=10), png(value=90)
    assert screenshots.job_id_for([a, b]) == screenshots.job_id_for([b, a])
    assert screenshots.job_id_for([a]) != screenshots.job_id_for([b])


def test_images_are_validated_and_shrunk():
    out = screenshots.normalize_images([png(4000, 3000)])
    decoded = cv2.imdecode(np.frombuffer(out[0], np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) <= screenshots.MAX_SIDE
    with pytest.raises(ValueError):
        screenshots.normalize_images([b"not an image"])
    with pytest.raises(ValueError):
        screenshots.normalize_images([])
    with pytest.raises(ValueError):
        screenshots.normalize_images([png()] * (screenshots.MAX_IMAGES + 1))
