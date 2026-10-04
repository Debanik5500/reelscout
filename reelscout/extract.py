"""Deterministic (free, instant) extraction of URLs / domains / GitHub repos from text.

This runs BEFORE any LLM: for the common reel ("go to acquire.com") it is often enough,
and it gives the LLM a verified starting list instead of asking it to hallucinate one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TLDS = "com|io|ai|dev|app|co|org|net|sh|so|xyz|tech|gg|me|to|in|tools|cloud|page|site|run|codes|fm|tv|sc|new"
DOMAIN_RE = re.compile(rf"(?<![\w@.-])((?:[a-z0-9][a-z0-9-]{{1,62}}\.)+(?:{TLDS}))(?![a-z0-9-])(/[^\s\"'<>)]*)?", re.I)
GITHUB_RE = re.compile(r"github\.com/([A-Za-z0-9-]{1,39})/([A-Za-z0-9._-]{1,100})", re.I)
GITHUB_PAGES_RE = re.compile(r"([A-Za-z0-9-]{1,39})\.github\.io/([A-Za-z0-9._-]{1,100})", re.I)
IGNORE = {"instagram.com", "cdninstagram.com", "threads.net", "gmail.com", "example.com", "facebook.com", "fb.com"}


def normalize_spoken(text: str) -> str:
    """'acquire dot com' -> 'acquire.com';  'github dot com slash foo slash bar' -> 'github.com/foo/bar'."""
    t = re.sub(rf"\b([a-z0-9][\w-]*)\s+dot\s+({TLDS})\b", r"\1.\2", text, flags=re.I)
    t = re.sub(r"(\.(?:%s))\s+slash\s+" % TLDS, r"\1/", t, flags=re.I)
    t = re.sub(r"(?<=[\w.-])\s+slash\s+(?=[\w.-])", "/", t, flags=re.I)
    return t


@dataclass
class Candidate:
    kind: str                      # website | github_repo | tool | product | other
    name: str                      # normalized: domain, "owner/repo", or a tool name
    url: str | None = None
    confidence: float = 0.4
    alternates: list[str] = field(default_factory=list)
    spoken_as: str | None = None
    evidence: list[dict] = field(default_factory=list)   # {source, text, t, n_frames?}
    verification: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)      # e.g. {"ecosystem": "pypi"}
    score: float = 0.0                             # final, evidence-based score (confidence stays the raw belief)

    @property
    def sources(self) -> set[str]:
        return {e["source"] for e in self.evidence}


def _clean_domain(d: str) -> str:
    d = d.lower().strip(".,;:)")
    return d[4:] if d.startswith("www.") else d


def find_in_text(text: str, source: str, t: float | None = None, n_frames: int = 1) -> list[Candidate]:
    text = normalize_spoken(text)
    out: list[Candidate] = []
    gh_spans = []
    for m in GITHUB_RE.finditer(text):
        owner, repo = m.group(1), re.sub(r"\.git$", "", m.group(2)).rstrip(".")
        gh_spans.append(m.span())
        out.append(Candidate("github_repo", f"{owner}/{repo}".lower(), f"https://github.com/{owner}/{repo}",
                             evidence=[dict(source=source, text=m.group(0), t=t, n_frames=n_frames)]))
    for m in GITHUB_PAGES_RE.finditer(text):
        owner, repo = m.group(1), re.sub(r"\.git$", "", m.group(2)).rstrip(".")
        gh_spans.append(m.span())
        out.append(Candidate("github_repo", f"{owner}/{repo}".lower(), f"https://github.com/{owner}/{repo}",
                             evidence=[dict(source=source, text=m.group(0), t=t, n_frames=n_frames)]))
    for m in DOMAIN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in gh_spans):
            continue
        host = _clean_domain(m.group(1))
        if host in IGNORE or host.startswith("github.com"):
            continue
        out.append(Candidate("website", host, f"https://{host}",
                             evidence=[dict(source=source, text=m.group(0).strip(".,;:)"), t=t, n_frames=n_frames)]))
    return out


def gather(transcript: list[dict], ocr_lines: list[dict], caption: str = "") -> list[Candidate]:
    raw: list[Candidate] = []
    for s in transcript:
        raw += find_in_text(s["text"], "transcript", s.get("start")) + find_commands(s["text"], "transcript", s.get("start"))
    for l in ocr_lines:
        raw += find_in_text(l["text"], "ocr", l.get("t"), l.get("n_frames", 1)) + find_commands(l["text"], "ocr", l.get("t"), l.get("n_frames", 1))
    if caption:
        raw += find_in_text(caption, "caption")
    return merge(raw)


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fold_asr_variants(cands: list[Candidate]) -> list[Candidate]:
    """'trustmr.com' (speech) + 'trustmrr.com' (seen on screen) are the same site: keep the seen spelling.
    Only ever folds a speech-only candidate into one backed by on-screen/caption text, so two genuinely
    different sites that both appear on screen are never merged."""
    webs = [c for c in cands if c.kind == "website"]
    drop = set()
    for a in webs:
        if a.sources != {"transcript"}:
            continue
        for b in webs:
            if b is a or not (b.sources & {"ocr", "caption", "tagged", "creator_comment"}):
                continue
            if _lev(a.name, b.name) <= (2 if len(a.name) >= 12 else 1):
                b.evidence += a.evidence
                b.spoken_as = b.spoken_as or a.name
                drop.add(id(a))
                break
    return [c for c in cands if id(c) not in drop]


def merge(cands: list[Candidate]) -> list[Candidate]:
    merged: dict[tuple, Candidate] = {}
    for c in cands:
        k = (c.kind, c.name.lower())
        if k in merged:
            m = merged[k]
            m.evidence += c.evidence
            m.alternates = sorted(set(m.alternates) | set(c.alternates))
            m.confidence = max(m.confidence, c.confidence)
            m.url = m.url or c.url
            m.spoken_as = m.spoken_as or c.spoken_as
            # Merge extra fields (e.g., steps from tutorial videos)
            if c.extra:
                m.extra = {**m.extra, **c.extra}
                # For steps, concatenate and deduplicate preserving order
                if "steps" in c.extra and "steps" in m.extra:
                    seen = set()
                    combined = []
                    for s in m.extra["steps"] + c.extra["steps"]:
                        if s not in seen:
                            seen.add(s)
                            combined.append(s)
                    m.extra["steps"] = combined
        else:
            merged[k] = c
    return _fold_asr_variants(list(merged.values()))


# ---------------------------------------------------------------------------------------
# Social-post signals: @mentions / tagged accounts are very often the brand, and "comment X"
# is the tell-tale sign the creator hides the link behind a DM automation (ManyChat etc.).
# ---------------------------------------------------------------------------------------
MENTION_RE = re.compile(r"(?<![\w.])@([A-Za-z0-9._]{2,30})")
_GATE_STOP = {"on", "below", "if", "and", "your", "a", "the", "this", "what", "down", "here", "something", "me",
              "anything", "there", "to", "for", "with", "i", "you", "it", "in", "or", "of", "we", "us", "at"}
GATE_RE = re.compile(
    r"\bcomment(?:\s+(?:the|this)?\s*(?:word|keyword|code)?)?\s*[:\-]?\s*[\"'“‘]?([A-Za-z0-9]{2,20})[\"'”’]?", re.I)
BIO_RE = re.compile(r"\b(link in (?:my )?bio|dm me|send me a dm|dm (?:the word|\"?\w+\"?) )", re.I)


def detect_gating(texts: list[str]) -> dict | None:
    """Returns {keyword, mechanism} if the creator gates the link, else None."""
    for t in texts:
        for m in GATE_RE.finditer(t or ""):
            kw = m.group(1)
            if kw.lower() not in _GATE_STOP:
                return dict(keyword=kw.upper(), mechanism="comment-to-DM", snippet=t[max(0, m.start() - 20):m.end() + 30].strip())
    for t in texts:
        if BIO_RE.search(t or ""):
            return dict(keyword=None, mechanism="link-in-bio/DM", snippet=BIO_RE.search(t).group(0))
    return None


def social_candidates(meta: dict) -> list[Candidate]:
    out = []
    uploader = (meta.get("uploader") or "").lower()
    seen = set()
    for handle, src in [(h, "tagged") for h in meta.get("tagged", [])] + \
                       [(h, "caption") for h in meta.get("mentions", [])] + \
                       [(m.group(1), "caption") for m in MENTION_RE.finditer(meta.get("caption") or "")]:
        h = handle.lower().strip(".")
        if h and h != uploader and h not in seen:
            seen.add(h)
            out.append(Candidate("brand", f"@{h}", f"https://instagram.com/{h}", confidence=0.5,
                                 evidence=[dict(source=src, text=f"@{h}", t=None)]))
    for c in meta.get("comments", []):       # the creator answering "what is this?" in their own comments
        if c.get("by_creator"):
            out += find_in_text(c["text"], "creator_comment")
    return out


# ---------------------------------------------------------------------------------------
# Install / run commands visible on screen or spoken. For dev reels that never name the tool,
# the terminal very often does: `pip install foo`, `npx create-foo`, `brew install foo`.
# ---------------------------------------------------------------------------------------
_PKG = r"(@?[A-Za-z0-9][\w.@/-]{1,80})"
_FLAGS = r"(?:-{1,2}[\w-]+(?:=\S+)?\s+)*"
COMMANDS = [
    ("pypi", re.compile(rf"\bpip3?\s+install\s+{_FLAGS}{_PKG}", re.I)),
    ("pypi", re.compile(rf"\b(?:uv\s+(?:pip\s+install|add)|uvx|pipx\s+install)\s+{_FLAGS}{_PKG}", re.I)),
    ("npm", re.compile(rf"\b(?:npm|pnpm|yarn|bun)\s+(?:i|install|add)\s+{_FLAGS}{_PKG}", re.I)),
    ("npm", re.compile(rf"\b(?:npx|bunx|pnpm\s+dlx)\s+{_FLAGS}{_PKG}", re.I)),
    ("brew", re.compile(rf"\bbrew\s+install\s+{_FLAGS}{_PKG}", re.I)),
    ("cargo", re.compile(rf"\bcargo\s+install\s+{_FLAGS}{_PKG}", re.I)),
    ("docker", re.compile(rf"\bdocker\s+(?:pull|run)\s+{_FLAGS}{_PKG}", re.I)),
]
_PKG_STOP = {"install", "add", "run", "dev", "start", "build", "test", "requirements.txt", "-r", "latest", "update", "upgrade"}


def find_commands(text: str, source: str, t: float | None = None, n_frames: int = 1) -> list[Candidate]:
    text = normalize_spoken(text)
    out = []
    for eco, rx in COMMANDS:
        for m in rx.finditer(text):
            pkg = m.group(1).rstrip(".,;:")
            if pkg.lower() in _PKG_STOP or pkg.startswith("-") or pkg.endswith(".txt"):
                continue
            out.append(Candidate("package", pkg.lower(), None, confidence=0.6, extra=dict(ecosystem=eco),
                                 evidence=[dict(source=source, text=m.group(0), t=t, n_frames=n_frames)]))
    return out
