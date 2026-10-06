"""Can THIS machine download Instagram reels? Run it on the server you are considering.

  python scripts/download_check.py urls.txt            # one reel/post link per line
  python scripts/download_check.py URL1 URL2 URL3

Tries exactly what the pipeline does (yt-dlp, then instaloader for image posts), with your cookies file if one
exists (~/.reelscout/instagram_cookies.txt), and reports success rate, time, and WHY failures happen.
Nothing is saved to your library: downloads go to a temp folder.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from reelscout import batch, ingest


def reason(error: str) -> str:
    text = error.lower()
    if "login" in text or "log in" in text or "sign in" in text:
        return "login required (Instagram wants a session)"
    if "429" in text or "rate" in text or "too many" in text:
        return "rate limited"
    if "400" in text or "403" in text or "challenge" in text or "checkpoint" in text:
        return "blocked / checkpoint (typical for datacenter IPs)"
    if "unsupported url" in text:
        return "not a reel link"
    if "private" in text or "not available" in text or "unavailable" in text:
        return "private or removed"
    return "other: " + error[:80]


def main(args: list[str]) -> None:
    urls = [l.strip() for a in args for l in (Path(a).read_text().splitlines() if Path(a).is_file() else [a])]
    urls = [u for u in urls if u.startswith("http")]
    if not urls:
        sys.exit(__doc__)
    try:
        print("this machine's public IP:", requests.get("https://api.ipify.org", timeout=10).text)
    except Exception:
        print("this machine's public IP: unknown")
    cookies = batch.cookies_file()
    print("cookies file:", cookies or "none (anonymous requests)", "\n")

    work = Path(tempfile.mkdtemp(prefix="dlcheck-"))
    results, reasons = [], Counter()
    for url in urls:
        started, ok, note = time.time(), False, ""
        try:
            if "/p/" in url:
                post = ingest.fetch_post(url)
                if post:
                    ok, note = True, f"image post, {len(post[0])} slides"
                else:
                    video, _ = ingest.fetch(url, work, cookies_file=cookies)
                    ok, note = True, f"video {video.stat().st_size / 1e6:.1f} MB"
            else:
                video, _ = ingest.fetch(url, work, cookies_file=cookies)
                ok, note = True, f"video {video.stat().st_size / 1e6:.1f} MB"
        except Exception as exc:
            note = reason(str(exc))
            reasons[note] += 1
        results.append(ok)
        print(f"{'OK  ' if ok else 'FAIL'} {time.time() - started:5.1f}s  {url[:60]:<62} {note}")
        time.sleep(3)  # do not hammer Instagram while testing

    wins = sum(results)
    print(f"\n{wins}/{len(results)} downloaded ({100 * wins // max(len(results), 1)}%)")
    for why, n in reasons.most_common():
        print(f"  {n} x {why}")
    print("\nJudge it on 10+ links across a few hours, not one run.")
    for f in work.glob("*"):
        f.unlink(missing_ok=True)
    work.rmdir()


if __name__ == "__main__":
    main(sys.argv[1:])
