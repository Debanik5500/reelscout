"""Standalone check: can THIS machine download Instagram reels? Paste/upload this one file anywhere (Colab,
Codespaces, a VPS). It needs only:  pip install yt-dlp instaloader requests   (add curl_cffi to test browser mode)

  python download_check_standalone.py urls.txt                    # anonymous (no login, no risk to any account)
  python download_check_standalone.py urls.txt --cookies c.txt    # with a Netscape cookies file (use a throwaway account)
  python download_check_standalone.py URL1 URL2 --no-comments

Prints the machine's public IP, then OK/FAIL per link with the reason, and a success rate.
"""
from __future__ import annotations

import argparse
import re
import tempfile
import time
from collections import Counter
from pathlib import Path


def why(error: str) -> str:
    text = error.lower()
    if "login" in text or "log in" in text or "sign in" in text:
        return "login required (Instagram wants a session)"
    if "429" in text or "rate" in text or "too many" in text:
        return "rate limited"
    if any(w in text for w in ("400", "403", "challenge", "checkpoint")):
        return "blocked / checkpoint (typical for datacenter IPs)"
    if "unsupported url" in text:
        return "not a reel link"
    if "private" in text or "not available" in text or "unavailable" in text:
        return "private or removed"
    return "other: " + error.strip().splitlines()[0][:90]


def ytdlp(url: str, work: Path, cookies: str | None, comments: bool) -> str:
    import yt_dlp
    opts = {"quiet": True, "no_warnings": True, "format": "best[height<=720]/best", "outtmpl": str(work / "%(id)s.%(ext)s")}
    if cookies:
        opts["cookiefile"] = cookies
    if comments:
        opts["getcomments"] = True
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        entry = (info.get("entries") or [info])[0]
        return f"video {Path(ydl.prepare_filename(entry)).stat().st_size / 1e6:.1f} MB"


def instaloader_post(url: str, cookies: str | None) -> str:
    import instaloader
    import requests
    code = re.search(r"/(?:p|reel|reels|tv)/([\w-]+)", url).group(1)
    loader = instaloader.Instaloader(quiet=True, download_pictures=False, download_videos=False, save_metadata=False)
    if cookies:  # reuse the same Netscape cookie file for instaloader's session
        from http.cookiejar import MozillaCookieJar
        jar = MozillaCookieJar(cookies)
        jar.load(ignore_discard=True, ignore_expires=True)
        loader.context._session.cookies.update(jar)
    post = instaloader.Post.from_shortcode(loader.context, code)
    if post.typename == "GraphVideo":
        raise RuntimeError("is a video post (use the video path)")
    nodes = list(post.get_sidecar_nodes()) if post.typename == "GraphSidecar" else []
    urls = [n.display_url for n in nodes] or [post.url]
    requests.get(urls[0], timeout=60).raise_for_status()  # prove the image CDN is reachable too
    return f"image post, {len(urls)} slide(s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="a file of links (one per line) and/or links")
    ap.add_argument("--cookies", help="Netscape cookies.txt (optional)")
    ap.add_argument("--no-comments", action="store_true", help="skip the comments request (fewer calls)")
    ap.add_argument("--wait", type=float, default=4.0, help="seconds between links")
    args = ap.parse_args()

    urls = []
    for item in args.inputs:
        urls += Path(item).read_text().splitlines() if Path(item).is_file() else [item]
    urls = [u.strip() for u in urls if u.strip().startswith("http")]

    try:
        import requests
        print("public IP of this machine:", requests.get("https://api.ipify.org", timeout=10).text)
    except Exception:
        print("public IP: unknown")
    try:
        import yt_dlp.networking.impersonate  # noqa: F401
        from yt_dlp import YoutubeDL
        print("browser impersonation (curl_cffi):", "available" if YoutubeDL()._impersonate_target_available(
            __import__("yt_dlp.networking.impersonate", fromlist=["ImpersonateTarget"]).ImpersonateTarget()) else "NOT installed")
    except Exception:
        print("browser impersonation: unknown")
    print("cookies:", args.cookies or "none (anonymous)", "\n")

    work = Path(tempfile.mkdtemp(prefix="dlcheck-"))
    ok_count, reasons = 0, Counter()
    for url in urls:
        started, ok, note = time.time(), False, ""
        try:
            if "/p/" in url:
                try:
                    note = instaloader_post(url, args.cookies)
                except RuntimeError:
                    note = ytdlp(url, work, args.cookies, not args.no_comments)
            else:
                note = ytdlp(url, work, args.cookies, not args.no_comments)
            ok = True
        except Exception as exc:
            note = why(str(exc))
            reasons[note] += 1
        ok_count += ok
        print(f"{'OK  ' if ok else 'FAIL'} {time.time() - started:5.1f}s  {url[:58]:<60} {note}")
        time.sleep(args.wait)

    print(f"\n{ok_count}/{len(urls)} downloaded ({100 * ok_count // max(len(urls), 1)}%)")
    for reason, n in reasons.most_common():
        print(f"  {n} x {reason}")


if __name__ == "__main__":
    main()
