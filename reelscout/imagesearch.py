"""Open-source image-based identification. Both parts are OPTIONAL and lazily imported.

reverse_search : PicImageSearch (MIT) drives Google Lens / Yandex / Bing reverse image search.
                 These engines have no free official API, so the library scrapes them. That means it can
                 break when they change, hit bot-protection (you may need cookies/proxy), and sits in a
                 terms-of-service grey area. Fine for testing; not something to ship in a product.
geolocate      : GeoCLIP (open-source image -> GPS model). Gives a *region/city-level* guess, not a venue.
                 Needs torch (~GBs); the first run downloads weights.
"""
from __future__ import annotations

import time

import requests

UA = {"User-Agent": "reelscout/0.1 (testing)"}


def reverse_search(image_path: str, engines=("GoogleLens", "Yandex", "Bing"), n: int = 6) -> tuple[list[dict], list[str]]:
    """Returns (hits, errors). Tries engines in order and stops at the first that returns anything."""
    errors: list[str] = []
    try:
        import PicImageSearch.sync as pis
    except Exception as e:
        return [], [f"PicImageSearch not installed ({e}); pip install PicImageSearch"]
    for name in engines:
        try:
            resp = getattr(pis, name)().search(file=image_path)
            hits = []
            for r in (getattr(resp, "raw", None) or [])[:n]:
                title, url = getattr(r, "title", None), getattr(r, "url", None)
                if title or url:
                    hits.append(dict(engine=name, title=title or "", url=url or ""))
            if hits:
                return hits, errors
            errors.append(f"{name}: no results (possibly bot protection)")
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
    return [], errors


_geo = None


def _reverse_geocode(lat: float, lon: float) -> str | None:
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse", headers=UA, timeout=8,
                         params=dict(lat=lat, lon=lon, format="jsonv2", zoom=10))
        return r.json().get("display_name")
    except (requests.RequestException, ValueError):
        return None


def geolocate(image_paths: list[str], top_k: int = 3) -> dict | None:
    """Pools GeoCLIP predictions over a few frames, reverse-geocodes the most probable ones."""
    global _geo
    try:
        from geoclip import GeoCLIP
    except Exception:
        return None
    if _geo is None:
        _geo = GeoCLIP()
    preds = []
    for p in image_paths:
        gps, prob = _geo.predict(p, top_k=top_k)
        preds += [(float(prob[i]), float(gps[i][0]), float(gps[i][1])) for i in range(len(prob))]
    preds.sort(reverse=True)
    out, seen = [], set()
    for prob, lat, lon in preds:
        name = _reverse_geocode(lat, lon)
        time.sleep(1.0)                                   # Nominatim usage policy: max 1 request/second
        if name and name not in seen:
            seen.add(name)
            out.append(dict(place=name, lat=lat, lon=lon, prob=round(prob, 3)))
        if len(out) >= 3:
            break
    return dict(guesses=out) if out else None
