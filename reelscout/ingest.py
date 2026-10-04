"""Turn a path or URL into (local video path, metadata).

Metadata is where a lot of the answer hides for reels that never say the product name:
caption, @mentions, tagged accounts, location tag, hashtags, and -- best of all -- the
creator's own replies in the comments ("it's the Stanley Quencher!").

Backends (all open source, no official Instagram API):
  1. yt-dlp      : downloads video + caption (+ comments when the extractor allows it)
  2. instaloader : fallback for richer post metadata (tagged users, location); comments need login
Both are best-effort: Instagram changes often and may demand a logged-in session, so pass
--cookies-from-browser chrome|firefox. If both fail, download the reel yourself and pass the file.
"""
from __future__ import annotations

import re
from pathlib import Path


def empty_meta() -> dict:
    return dict(caption="", uploader=None, hashtags=[], mentions=[], tagged=[], location=None, comments=[])


def _from_ytdlp(info: dict) -> dict:
    m = empty_meta()
    m["caption"] = "\n".join(filter(None, [info.get("description"), info.get("title")]))
    uploader_id = info.get("uploader_id") or info.get("uploader") or info.get("channel")
    m["uploader"] = uploader_id
    # Try to extract username from "Video by @username" or "Video by username" in caption
    import re
    vid_by = re.search(r"Video by\s+@?(\w[\w.]+)", m["caption"])
    if vid_by:
        m["uploader_username"] = vid_by.group(1)
    m["hashtags"] = info.get("tags") or re.findall(r"#(\w+)", m["caption"])
    m["location"] = info.get("location")
    m["comments"] = [dict(author=c.get("author"), text=c.get("text", ""), likes=c.get("like_count") or 0,
                          by_creator=bool(c.get("author_is_uploader")))
                     for c in (info.get("comments") or [])[:60]]
    return m


def _fetch_ytdlp(url: str, workdir: Path, cookies: str | None, cookies_file: str | None = None) -> tuple[Path, dict]:
    import yt_dlp
    base = {"quiet": True, "no_warnings": True, "format": "best[height<=720]/best",
            "outtmpl": str(workdir / "%(id)s.%(ext)s")}
    if cookies_file:
        base["cookiefile"] = cookies_file
    elif cookies:
        base["cookiesfrombrowser"] = (cookies,)
    try:                                   # ask for comments first, retry without if the extractor balks
        opts = dict(base, getcomments=True)
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Handle carousel posts (multiple entries)
            if "entries" in info:
                # Find first video entry
                for entry in info["entries"]:
                    if entry and entry.get("url") and entry.get("ext") in ("mp4", "mov", "mkv", "webm"):
                        info = entry
                        break
            return Path(ydl.prepare_filename(info)), _from_ytdlp(info)
    except Exception:
        with yt_dlp.YoutubeDL(base) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info:
                for entry in info["entries"]:
                    if entry and entry.get("url") and entry.get("ext") in ("mp4", "mov", "mkv", "webm"):
                        info = entry
                        break
            return Path(ydl.prepare_filename(info)), _from_ytdlp(info)


def _fetch_instaloader(url: str, workdir: Path) -> tuple[Path, dict]:
    import instaloader, requests
    code = re.search(r"/(?:reel|reels|p|tv)/([\w-]+)", url).group(1)
    L = instaloader.Instaloader(download_videos=False, save_metadata=False, quiet=True)
    post = instaloader.Post.from_shortcode(L.context, code)
    m = empty_meta()
    m.update(caption=post.caption or "", uploader=post.owner_username, hashtags=list(post.caption_hashtags),
             mentions=list(post.caption_mentions), tagged=list(post.tagged_users),
             location=getattr(post.location, "name", None) if post.location else None)
    try:                                   # comments need a logged-in session; skip quietly if not
        for c in list(post.get_comments())[:60]:
            m["comments"].append(dict(author=c.owner.username, text=c.text, likes=c.likes_count,
                                      by_creator=c.owner.username == post.owner_username))
    except Exception:
        pass
    path = workdir / f"{code}.mp4"
    path.write_bytes(requests.get(post.video_url, timeout=60).content)
    return path, m


def fetch(source: str, workdir: Path, cookies_from_browser: str | None = None, cookies_file: str | None = None) -> tuple[Path, dict]:
    if not source.startswith(("http://", "https://")):
        return Path(source), empty_meta()
    workdir.mkdir(parents=True, exist_ok=True)
    errors = []
    for name, fn in (("yt-dlp", lambda: _fetch_ytdlp(source, workdir, cookies_from_browser, cookies_file)),
                     ("instaloader", lambda: _fetch_instaloader(source, workdir))):
        try:
            path, meta = fn()
            # yt-dlp is great at video, instaloader at tags/location: top up missing fields if we can
            if name == "yt-dlp" and not (meta["tagged"] or meta["location"]):
                try:
                    _, extra = _fetch_instaloader(source, workdir)
                    for k in ("mentions", "tagged", "location"):
                        meta[k] = meta[k] or extra[k]
                    meta["comments"] = meta["comments"] or extra["comments"]
                except Exception:
                    pass
            return path, meta
        except Exception as e:
            errors.append(f"{name}: {e}")
    raise RuntimeError("Could not ingest the URL (" + "; ".join(errors) + "). "
                       "Try --cookies-from-browser chrome, or download the reel and pass the file path.")
