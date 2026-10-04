"""Free verification: does the domain exist, does the GitHub repo exist, what does search say."""
from __future__ import annotations

import os
import re
import urllib.parse

import requests

UA = {"User-Agent": "Mozilla/5.0 (compatible; reelscout/0.1)"}


def check_domain(host: str, timeout: float = 6.0) -> dict:
    for scheme in ("https", "http"):
        try:
            r = requests.get(f"{scheme}://{host}", headers=UA, timeout=timeout, stream=True, allow_redirects=True)
            head = r.raw.read(40000, decode_content=True).decode("utf-8", "ignore")
            m = re.search(r"<title[^>]*>(.*?)</title>", head, re.I | re.S)
            return dict(ok=r.status_code < 500, status=r.status_code, final_url=r.url,
                        title=re.sub(r"\s+", " ", m.group(1)).strip()[:120] if m else None)
        except requests.RequestException:
            continue
    return dict(ok=False)


def _gh_headers() -> dict:
    h = {"Accept": "application/vnd.github+json", **UA}
    if os.getenv("GITHUB_TOKEN"):
        h["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    return h


def github_repo(full_name: str) -> dict:
    try:
        r = requests.get(f"https://api.github.com/repos/{full_name}", headers=_gh_headers(), timeout=8)
        if r.status_code == 200:
            j = r.json()
            return dict(ok=True, url=j["html_url"], stars=j["stargazers_count"], description=j["description"])
    except requests.RequestException:
        pass
    return dict(ok=False)


def github_search(name: str, n: int = 3) -> list[dict]:
    try:
        r = requests.get("https://api.github.com/search/repositories", headers=_gh_headers(), timeout=8,
                         params=dict(q=f"{name} in:name", sort="stars", per_page=n))
        if r.status_code == 200:
            return [dict(full_name=i["full_name"], url=i["html_url"], stars=i["stargazers_count"],
                         description=i["description"]) for i in r.json().get("items", [])]
    except requests.RequestException:
        pass
    return []


def github_search_text(query: str, n: int = 5) -> list[dict]:
    try:
        r = requests.get("https://api.github.com/search/repositories", headers=_gh_headers(), timeout=8,
                         params=dict(q=f"{query} in:name,description,readme", sort="stars", per_page=n))
        if r.status_code == 200:
            return [dict(full_name=i["full_name"], url=i["html_url"], stars=i["stargazers_count"],
                         description=i["description"]) for i in r.json().get("items", [])]
    except requests.RequestException:
        pass
    return []


def web_search(query: str, n: int = 5) -> list[dict]:
    from .search import web_search as _ws
    return _ws(query, n)


def _gh_from(urls) -> str | None:
    for u in urls:
        m = GITHUB_URL.search(u or "")
        if m:
            return f"https://github.com/{m.group(1)}/{re.sub(r'[.]git$', '', m.group(2))}"
    return None


GITHUB_URL = re.compile(r"github\.com[/:]([\w-]+)/([\w.-]+)", re.I)


def check_package(name: str, eco: str) -> dict:
    """Registry lookups are free and keyless; a hit usually links straight to the source repo."""
    try:
        if eco == "pypi":
            r = requests.get(f"https://pypi.org/pypi/{name.split('[')[0]}/json", headers=UA, timeout=8)
            if r.status_code == 200:
                i = r.json()["info"]
                repo = _gh_from(list((i.get("project_urls") or {}).values()) + [i.get("home_page")])
                return dict(ok=True, registry="pypi", summary=i.get("summary"), repo=repo, home=i.get("home_page"))
        elif eco == "npm":
            r = requests.get(f"https://registry.npmjs.org/{name}", headers=UA, timeout=8)
            if r.status_code == 200:
                j = r.json()
                repo_field = j.get("repository")
                repo_url = repo_field.get("url") if isinstance(repo_field, dict) else repo_field
                return dict(ok=True, registry="npm", summary=j.get("description"), repo=_gh_from([repo_url]), home=j.get("homepage"))
        elif eco == "brew":
            r = requests.get(f"https://formulae.brew.sh/api/formula/{name}.json", headers=UA, timeout=8)
            if r.status_code == 200:
                j = r.json()
                return dict(ok=True, registry="brew", summary=j.get("desc"), repo=_gh_from([j.get("homepage")]), home=j.get("homepage"))
        elif eco == "cargo":
            r = requests.get(f"https://crates.io/api/v1/crates/{name}", headers=UA, timeout=8)
            if r.status_code == 200:
                c = r.json()["crate"]
                return dict(ok=True, registry="crates.io", summary=c.get("description"), repo=_gh_from([c.get("repository")]), home=c.get("homepage"))
        elif eco == "docker":
            ns, _, repo_name = (name if "/" in name else f"library/{name}").partition("/")
            r = requests.get(f"https://hub.docker.com/v2/repositories/{ns}/{repo_name.split(':')[0]}", headers=UA, timeout=8)
            if r.status_code == 200:
                return dict(ok=True, registry="dockerhub", summary=r.json().get("description"), repo=None, home=None)
    except (requests.RequestException, ValueError, KeyError):
        pass
    return dict(ok=False)


def geocode(query: str) -> dict:
    """OpenStreetMap Nominatim: free, no key (1 req/s policy, fine for testing)."""
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search", headers=UA, timeout=8,
                         params=dict(q=query, format="json", limit=1, addressdetails=0))
        j = r.json()
        if j:
            return dict(ok=True, display_name=j[0]["display_name"], lat=j[0]["lat"], lon=j[0]["lon"],
                        map_url=f"https://www.openstreetmap.org/?mlat={j[0]['lat']}&mlon={j[0]['lon']}#map=16/{j[0]['lat']}/{j[0]['lon']}")
    except (requests.RequestException, ValueError):
        pass
    return dict(ok=False)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def verify(cand, web: bool = True) -> None:
    """Mutates cand.verification / cand.url. Tries LLM-supplied alternates if the first guess is dead."""
    v: dict = {}
    if cand.kind == "website":
        for host in [cand.name] + [a for a in cand.alternates if a != cand.name]:
            res = check_domain(host)
            if res["ok"]:
                v = dict(res, resolved_host=host, corrected=host != cand.name)
                cand.url = res["final_url"]
                break
        else:
            v = dict(ok=False)
            if web:
                v["search"] = web_search(f"{cand.name.split('.')[0]} website")
    elif cand.kind == "github_repo":
        v = github_repo(cand.name)
        if v.get("ok"):
            cand.url = v["url"]
        elif cand.url or cand.name:
            hits = github_search(cand.name.split("/")[-1])
            v = dict(ok=False, search=hits)
    elif cand.kind in ("tool", "product"):
        hits = github_search(cand.name)
        exact = [h for h in hits if _norm(h["full_name"].split("/")[1]) == _norm(cand.name)]
        v = dict(ok=bool(exact), github=exact[0] if exact else None, search=hits[:3])
        if exact:
            cand.url = cand.url or exact[0]["url"]
        if web and not cand.url:
            v["web"] = web_search(f"{cand.name} official site")
    elif cand.kind == "package":
        v = check_package(cand.name, cand.extra.get("ecosystem", "pypi"))
        if v.get("ok"):
            cand.url = v.get("repo") or v.get("home")
        else:
            v = dict(ok=False, search=github_search(cand.name.split("/")[-1]))
    elif cand.kind == "place":
        v = geocode(cand.name)
        if v.get("ok"):
            # For places, prefer map URL as primary; keep website as alternate if LLM provided one
            website_url = cand.url
            cand.url = v["map_url"]
            if website_url and website_url != v["map_url"]:
                cand.alternates = list(dict.fromkeys([website_url] + [a for a in cand.alternates if a != website_url]))
    if cand.kind == "brand" and not cand.name.startswith("@") and web:
        hits = web_search(f"{cand.name} official site")
        v = dict(ok=bool(hits), web=hits)
        cand.url = cand.url or (hits[0]["url"] if hits else None)
    cand.verification = v
