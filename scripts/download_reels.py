#!/usr/bin/env python3
"""Batch download Instagram reels for testing."""
from __future__ import annotations
import sys
import json
from pathlib import Path
import yt_dlp

# Load cookies from our cookies.txt
COOKIES_FILE = Path(__file__).parent.parent / "cookies.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "test_reels"
OUTPUT_DIR.mkdir(exist_ok=True)

# Reel URLs to download (add more here)
REEL_URLS = [
    "https://www.instagram.com/reels/Dd7x6noPjAM/",
    "https://www.instagram.com/reels/Dd6VsR7PI5w/",
    "https://www.instagram.com/reels/Dd8s6U4ztQK/",
    "https://www.instagram.com/reels/Dd7sPgmD32w/",
    "https://www.instagram.com/reel/Ddoe6dpyyka/",
    "https://www.instagram.com/reels/DdToB05gH14/",
    "https://www.instagram.com/reels/Dd7F1jzE0J7/",  # carousel - will fail
    # Add more URLs here
]

def download_reels(urls: list[str], output_dir: Path, cookies_file: Path) -> dict:
    """Download reels and return metadata."""
    results = {}
    
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "best[height<=720]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "cookiefile": str(cookies_file),
        "extract_flat": False,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                print(f"Downloading: {url}")
                info = ydl.extract_info(url, download=True)
                if "entries" in info:
                    # Carousel - find first video
                    video_info = next((e for e in info["entries"] if e and e.get("url")), None)
                    if not video_info:
                        results[url] = {"error": "No video in carousel"}
                        continue
                    info = video_info
                
                video_id = info.get("id", "unknown")
                filename = ydl.prepare_filename(info)
                meta = {
                    "id": video_id,
                    "caption": info.get("description", "") or info.get("title", ""),
                    "uploader": info.get("uploader_id") or info.get("uploader"),
                    "hashtags": info.get("tags") or [],
                    "url": url,
                    "local_path": filename,
                }
                results[url] = {"meta": meta, "local_path": filename}
                print(f"  ✓ Saved: {filename}")
                
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                results[url] = {"error": str(e)}
    
    return results

def main():
    print(f"Downloading {len(REEL_URLS)} reels to {OUTPUT_DIR}")
    results = download_reels(REEL_URLS, OUTPUT_DIR, COOKIES_FILE)
    
    # Save metadata
    meta_file = OUTPUT_DIR / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetadata saved to {meta_file}")
    
    # Print summary
    ok = sum(1 for r in results.values() if "error" not in r)
    print(f"\nDownloaded: {ok}/{len(REEL_URLS)} successful")

if __name__ == "__main__":
    main()