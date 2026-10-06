"""Web pages, Google results and pasted text -> notes, events and places.

A person shares a page (a fixture list, a ticket page, an article) or a Google search such as "india cricket
schedule". We fetch it safely, read its visible text and any structured event data, and ask the LLM what it is and
which dated things deserve a calendar entry.

SAFETY: the server fetches addresses chosen by users, so every request (and every redirect hop) is checked: http(s)
only, no credentials in the URL, only ports 80/443, and the host must resolve ONLY to public addresses. Size and time
are capped. Residual risk: DNS can change between our check and the connection (rebinding); on the VPS also block
outbound traffic to private ranges and the cloud metadata address at the firewall.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse, urlunparse

import requests

from . import context, llm, search
from .config import Settings
from .events import EVENT_RULES, normalize_events, zone
from .llm import complete_json
from .storage import ReelResult, get_phone_storage

logger = logging.getLogger("reelscout.webpage")

MAX_URL = 2000
MAX_TEXT = 20000
MAX_BYTES = 2 * 1024 * 1024
FETCH_TIMEOUT = 10
MAX_REDIRECTS = 4
PAGE_CHARS = 7000
UA = "Mozilla/5.0 (compatible; KeptBot/1.0; +personal-notes)"
TRACKING = re.compile(r"^(utm_|fbclid|gclid|igshid|igsh|ref|mc_)", re.I)
PAGE_TYPES = ("event", "schedule", "article", "recipe", "product", "listing", "booking", "other")


class UnsafeURL(ValueError):
    pass


# --- URL safety ------------------------------------------------------------------------------------------

def check_url(url: str, resolve: bool = True) -> str:
    """Return the URL if it is safe to fetch, else raise UnsafeURL."""
    if not url or len(url) > MAX_URL:
        raise UnsafeURL("The link is missing or too long.")
    parts = urlparse(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise UnsafeURL("Only http and https links can be read.")
    if parts.username or parts.password:
        raise UnsafeURL("Links with a username or password are not allowed.")
    if parts.port not in (None, 80, 443):
        raise UnsafeURL("That port is not allowed.")
    host = parts.hostname.strip("[]").lower()
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        raise UnsafeURL("That address is not public.")
    try:
        addresses = [ipaddress.ip_address(host)]  # a literal IP
    except ValueError:
        if not resolve:
            return url.strip()
        try:
            addresses = [ipaddress.ip_address(info[4][0]) for info in
                         socket.getaddrinfo(host, parts.port or 443, proto=socket.IPPROTO_TCP)]
        except OSError:
            raise UnsafeURL("That address could not be found.")
    for address in addresses:
        if (address.is_private or address.is_loopback or address.is_link_local or address.is_multicast
                or address.is_reserved or address.is_unspecified):
            raise UnsafeURL("That address is not public.")
    return url.strip()


def normalize_url(url: str) -> str:
    """Stable form for dedupe: no fragment, no tracking parameters, lowercase host."""
    parts = urlparse(url.strip())
    query = "&".join(f"{k}={v[0]}" for k, v in sorted(parse_qs(parts.query).items()) if not TRACKING.match(k))
    return urlunparse((parts.scheme, parts.netloc.lower(), parts.path.rstrip("/") or "/", "", query, ""))


def job_id_for(url: str = "", text: str = "") -> str:
    basis = normalize_url(url) if url else "text:" + " ".join(text.split())
    return "web-" + hashlib.sha1(basis.encode()).hexdigest()[:12]


def search_query(url: str) -> str | None:
    """The search terms when the link is a search results page (Google, Bing, DuckDuckGo), else None."""
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    query = parse_qs(parts.query)
    is_search = (
        (re.search(r"(^|\.)google\.[a-z.]+$", host) and parts.path.startswith("/search"))
        or (host.endswith("bing.com") and parts.path.startswith("/search"))
        or (host.endswith("duckduckgo.com") and parts.path in ("", "/"))
    )
    terms = (query.get("q") or [""])[0].strip()
    return terms[:200] if is_search and terms else None


# --- fetching and reading --------------------------------------------------------------------------------

def fetch(url: str) -> tuple[str, str]:
    """(final_url, html). Safe, size-capped, redirects re-checked at every hop."""
    current = check_url(url)
    session = requests.Session()
    for _ in range(MAX_REDIRECTS + 1):
        response = session.get(current, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*;q=0.5"},
                               timeout=FETCH_TIMEOUT, allow_redirects=False, stream=True)
        if response.is_redirect and response.headers.get("location"):
            current = check_url(urljoin(current, response.headers["location"]))
            response.close()
            continue
        response.raise_for_status()
        kind = response.headers.get("content-type", "").lower()
        if kind and not any(k in kind for k in ("html", "xml", "text/plain", "json")):
            raise UnsafeURL("That link is not a web page.")
        body = b""
        for chunk in response.iter_content(65536):
            body += chunk
            if len(body) > MAX_BYTES:
                break
        response.close()
        return current, body[:MAX_BYTES].decode(response.encoding or "utf-8", errors="replace")
    raise UnsafeURL("Too many redirects.")


class _Reader(HTMLParser):
    """Visible text plus title, meta description and JSON-LD blocks, using only the standard library."""

    SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe"}
    BREAK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "ul", "ol"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.title, self.meta, self.jsonld, self.parts = "", {}, [], []
        self._skip, self._in_title, self._in_ld, self._ld = 0, False, False, []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "title":
            self._in_title = True
        elif tag == "meta":
            key = (attrs.get("property") or attrs.get("name") or "").lower()
            if key in ("og:title", "og:description", "description", "og:site_name"):
                self.meta[key] = (attrs.get("content") or "").strip()
        elif tag == "script" and (attrs.get("type") or "").lower() == "application/ld+json":
            self._in_ld, self._ld = True, []
        if tag in self.SKIP and tag != "head":
            self._skip += 1
        if tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in self.BREAK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag == "script" and self._in_ld:
            self.jsonld.append("".join(self._ld))
            self._in_ld = False
        if tag in self.SKIP and tag != "head" and self._skip:
            self._skip -= 1
        if tag in self.BREAK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_ld:
            self._ld.append(data)
        elif self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def extract_page(html: str) -> dict:
    """{title, description, text, ld_events} from raw HTML."""
    reader = _Reader()
    try:
        reader.feed(html)
    except Exception:  # broken markup is normal on the web; use what was read
        pass
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(reader.parts))
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    return dict(title=" ".join(reader.title.split()) or reader.meta.get("og:title", ""),
                description=reader.meta.get("og:description") or reader.meta.get("description", ""),
                site=reader.meta.get("og:site_name", ""), text=text[:PAGE_CHARS],
                ld_events=_ld_events(reader.jsonld))


def _ld_events(blocks: list[str]) -> list[dict]:
    """Machine-readable events (schema.org Event, SportsEvent, MusicEvent...) in JSON-LD, as raw event dicts."""
    found = []

    def walk(node):
        if isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            kinds = node.get("@type")
            kinds = [kinds] if isinstance(kinds, str) else (kinds or [])
            if any(str(k).endswith("Event") or str(k) == "Festival" for k in kinds) and node.get("startDate"):
                place = node.get("location")
                place = place[0] if isinstance(place, list) and place else place
                where = ""
                if isinstance(place, dict):
                    address = place.get("address")
                    address = address if isinstance(address, str) else ", ".join(
                        str(address.get(k)) for k in ("streetAddress", "addressLocality", "addressCountry")
                        if isinstance(address, dict) and address.get(k))
                    where = ", ".join(x for x in (str(place.get("name") or ""), address) if x)
                elif isinstance(place, str):
                    where = place
                found.append(dict(title=str(node.get("name") or ""), start=str(node["startDate"]),
                                  end=str(node.get("endDate") or ""), location=where,
                                  notes=" ".join(str(node.get("description") or "").split())[:200], confidence=0.95))
            for value in node.values():
                if isinstance(value, (list, dict)):
                    walk(value)

    for block in blocks:
        try:
            walk(json.loads(block))
        except ValueError:
            continue
    return found


# --- LLM -------------------------------------------------------------------------------------------------

_TEMPLATE = """You read a web page, a Google search result list, or a piece of pasted text that a person saved on
their phone, and pull out what is worth keeping, including dated things that deserve a calendar entry.
You get the page text, its title, and the CURRENT date, time and timezone of the user.
Rules:
- Fill a field ONLY when the evidence supports it. Empty is correct. Never invent.
- shot_type: one of event, schedule, article, recipe, product, listing, booking, other.
- title: max 8 words, plain words. summary: 1-2 sentences on what this is and why it is worth keeping.
{event_rules}
- For a fixture list or schedule, return the UPCOMING entries first and skip ones that already happened.
- places: every real venue or address the page is about (name, area, city, country, confidence 0-1, precision
  exact|area|city, note). Empty list when none.
- links: the most useful links or sites named (label, url only if written, kind, note). people: named people with a role.
- useful_context: up to 8 short facts worth remembering (prices, rules, steps). Do not repeat the events.
- Return JSON only:
{"title": str, "summary": str, "shot_type": str,
 "events": [{"title": str, "start": str, "end": str, "all_day": boolean, "tz": str, "year_inferred": boolean,
             "location": str, "notes": str, "confidence": number}],
 "places": [{"name": str, "area": str, "city": str, "country": str, "confidence": number, "precision": str, "note": str}],
 "links": [{"label": str, "url": str, "kind": str, "note": str}],
 "people": [{"name": str, "role": str}], "products": [{"name": str, "brand": str, "url": str}],
 "useful_context": [str], "things_mentioned": [str]}"""
SYSTEM = _TEMPLATE.replace("{event_rules}", EVENT_RULES.format(limit=15))


def gather(url: str, text: str) -> tuple[str, str, str, list[dict]]:
    """(display_url, title, evidence_text, structured_events) for a link, a search link, or pasted text."""
    if text and not url:
        return "", "", "PASTED TEXT:\n" + text[:MAX_TEXT], []
    query = search_query(url)
    if query:  # a Google results page cannot be read directly: search for the same words ourselves
        hits = search.web_search(query, 6)
        lines = [f"SEARCH QUERY: {query}"] + [f"- {h['title']} | {h['url']} | {h.get('snippet', '')[:240]}" for h in hits]
        ld: list[dict] = []
        read = 0
        for hit in hits:
            if read >= 2:
                break
            try:
                final, html = fetch(hit["url"])
            except Exception:
                continue
            page = extract_page(html)
            if len(page["text"]) > 200 or page["ld_events"]:
                read += 1
                ld += page["ld_events"]
                lines.append(f"\nPAGE {final}\nTITLE: {page['title']}\n{page['text'][:3500]}")
        return "", query[:100], "\n".join(lines), ld
    final, html = fetch(url)
    page = extract_page(html)
    body = [f"URL: {final}", f"TITLE: {page['title']}", f"SITE: {page['site']}", f"DESCRIPTION: {page['description']}",
            "PAGE TEXT:\n" + (page["text"] or "(the page has no readable text; it may need JavaScript)")]
    return final, page["title"], "\n".join(body), page["ld_events"]


def process(url: str, text: str, tz: str, st: Settings, progress, job_id: str | None = None,
            now: datetime | None = None) -> str:
    llm.reset_ledger()
    storage = get_phone_storage()
    job_id = job_id or job_id_for(url, text)
    tzinfo = zone(tz) or timezone.utc
    now = now or datetime.now(tzinfo)
    tz_name = getattr(tzinfo, "key", "UTC")

    progress("downloading", 0.0)
    final_url, title_hint, evidence, ld_events = gather(url, text)
    progress("downloading", 1.0, detail="page read")
    progress("watching", 0.5, detail="reading it")
    progress("keyMoments", 1.0, detail="understanding it", title=title_hint[:60])

    progress("placesAndBoard", 0.0)
    evidence = f"CURRENT DATE AND TIME: {now.strftime('%A %Y-%m-%d %H:%M')} (user timezone: {tz_name})\n\n{evidence}"
    stg = st.for_stage("TEXT")
    data, usage = complete_json(SYSTEM, evidence, stg, max_tokens=3500)

    ctx = context._finalize(data, [], [])
    ctx["kind"] = "webpage"
    ctx["category"] = ""
    page_type = str(data.get("shot_type") or "").lower()
    ctx["shot_type"] = page_type if page_type in PAGE_TYPES else "other"
    # Structured data on the page is more reliable than the model's reading of it.
    structured = normalize_events(ld_events, now, tz_name, limit=15)
    ctx["events"] = structured or normalize_events(data.get("events"), now, tz_name, limit=15)
    ctx["source_url"] = final_url
    ctx["domain"] = urlparse(final_url).hostname or ""
    context._ground_location(ctx)
    context._verify_urls(ctx)
    context.fill_item_lines(ctx)

    # No frames for a page, but the app payload needs a stored record to find.
    storage.metadata.setdefault(job_id, {"frames": [], "images": [], "created": datetime.now().isoformat()})
    storage._save_metadata()
    storage.save_result(ReelResult(job_id=job_id, status="completed", summary="", candidates=[],
                                   timings={}, source_frames=[]))
    debug = dict(usage=dict(final={**usage, "model": context.stage_label(stg)}, ledger=llm.ledger()),
                 research=dict(rounds=0, queries=[], findings=[], errors=[]), vision=dict(used=False, mode="off"))
    context.save_context(storage.results_dir, job_id, ctx, debug, {})
    first = ctx["events"][0] if ctx["events"] else None
    progress("placesAndBoard", 1.0, title=ctx["title"], chips=[first["title"][:40]] if first else [],
             detail=f"{len(ctx['events'])} event(s)" if first else "no event")
    return job_id
