from reelscout import context


def _place_finding(**kw):
    base = dict(goal="g", candidate="Shymbulak Mountain Resort", kind="place", url=None, city="Almaty",
                country="Kazakhstan", verdict="supported", confidence=0.9, evidence=[], contradictions=[],
                source_urls=["https://example.com/a"], lat=43.1, lon=77.0, map_url="https://osm.example/x")
    return {**base, **kw}


def test_verified_place_gets_coordinates():
    ctx = context._finalize({"location": {"name": "Shymbulak", "confidence": 0.9}}, [_place_finding()], [])
    assert ctx["location"]["lat"] == 43.1 and ctx["location"]["city"] == "Almaty"


def test_low_confidence_location_is_empty():
    ctx = context._finalize({"location": {"name": "San Francisco", "confidence": 0.3}}, [], [])
    assert ctx["location"]["name"] == "" and ctx["location"]["lat"] is None


def test_unverified_place_has_no_coordinates():
    ctx = context._finalize({"location": {"name": "Shymbulak", "confidence": 0.9}},
                            [_place_finding(verdict="uncertain")], [])
    assert ctx["location"]["name"] == "Shymbulak" and ctx["location"]["lat"] is None


def test_github_reel_has_no_location_but_keeps_links():
    ctx = context._finalize({"title": "A repo", "links": [{"label": "repo", "url": "https://github.com/a/b", "kind": "github_repo"},
                                                          {"label": "", "url": ""}]}, [], [])
    assert ctx["location"]["name"] == "" and len(ctx["links"]) == 1


def test_bad_username_dropped():
    assert context._finalize({"username": "12345"}, [], [])["username"] == ""
    assert context._finalize({"username": "@real.user"}, [], [])["username"] == "real.user"


def test_short_overlay_titles_survive_evidence_filter():
    lines = [dict(text="Kothidara", conf=1.0, t=10.5), dict(text="@mr.gixx_x_plorerr", conf=1.0, t=4.5),
             dict(text="Sangam Waterfalls", conf=0.99, t=8.0), dict(text="x", conf=1.0, t=1.0),
             dict(text="blurry", conf=0.4, t=2.0)]
    kept = context.evidence_ocr(lines)
    assert "Kothidara" in kept and "Sangam Waterfalls" in kept
    assert not any(k.startswith("@") for k in kept) and "blurry" not in kept


def test_list_reel_keeps_every_place_and_drops_unsure_ones():
    data = {"places": [
        {"name": "Kothidara", "city": "Siliguri", "country": "India", "confidence": 0.9, "precision": "area"},
        {"name": "Sangam Waterfalls", "city": "Siliguri", "country": "India", "confidence": 0.9},
        {"name": "Maybe Place", "confidence": 0.3},
        {"name": "kothidara", "confidence": 0.9},  # duplicate name
    ]}
    ctx = context._finalize(data, [], [])
    assert [p["name"] for p in ctx["places"]] == ["Kothidara", "Sangam Waterfalls"]
    assert ctx["location"]["name"] == "Kothidara"  # first place doubles as the single location


def test_reel_with_no_places_has_empty_list():
    ctx = context._finalize({"places": []}, [], [])
    assert ctx["places"] == [] and ctx["location"]["name"] == ""


def test_described_links_become_worth_remembering_lines():
    ctx = {"links": [{"label": "Hindsight", "note": "Agent memory", "url": "u"},
                     {"label": "Magpie", "note": "", "url": "u"},
                     {"label": "Orca", "note": "Runs parallel agents", "url": "u"}],
           "useful_context": ["Hindsight: Agent memory"]}
    context.fill_item_lines(ctx)
    context.fill_item_lines(ctx)  # idempotent
    assert ctx["useful_context"] == ["Hindsight: Agent memory", "Orca: Runs parallel agents"]
