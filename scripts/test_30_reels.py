#!/usr/bin/env python3
"""Test 30 Instagram reels from URLs - download and evaluate."""
from __future__ import annotations
import sys
import json
import csv
from pathlib import Path
import yt_dlp
import subprocess
import time

COOKIES_FILE = Path(__file__).parent.parent / "cookies.txt"
OUTPUT_DIR = Path(__file__).parent.parent / "test_reels_30"
OUTPUT_DIR.mkdir(exist_ok=True)

# 30 Instagram reel URLs to test (add more real URLs here)
REEL_URLS = [
    "https://www.instagram.com/reels/Dd7x6noPjAM/",
    "https://www.instagram.com/reels/Dd6VsR7PI5w/",
    "https://www.instagram.com/reels/Dd8s6U4ztQK/",
    "https://www.instagram.com/reels/Dd7sPgmD32w/",
    "https://www.instagram.com/reel/Ddoe6dpyyka/",
    "https://www.instagram.com/reels/DdToB05gH14/",
    "https://www.instagram.com/reels/Dd7sPgmD32w/",
    "https://www.instagram.com/reels/Dd6VsR7PI5w/",
    "https://www.instagram.com/reels/Dd8s6U4ztQK/",
    "https://www.instagram.com/reels/Dd7x6noPjAM/",
    # Add 20 more real Instagram reel URLs here
    # You can find these by browsing Instagram and copying reel URLs
]

# Additional URLs - add 20 more real Instagram reel URLs here
# Format: "https://www.instagram.com/reels/XXXXXXXXX/",
# You can get these by browsing Instagram.com/reels/ and copying URLs

def download_reel(url: str, output_dir: Path, cookies_file: Path) -> dict:
    """Download a single reel."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "format": "best[height<=720]/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "cookiefile": str(output_dir.parent / "cookies.txt"),
    }
    
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "format": "best[height<=720]/best", 
                               "outtmpl": str(Path("test") / "%(id)s.%(ext)s"), 
                               "cookiefile": str(Path("cookies.txt"))}) as ydl:
            info = ydl.extract_info(url, download=True)
            if "entries" in info:
                video_info = next((e for e in info["entries"] if e and e.get("url")), None)
                if not video_info:
                    return {"error": "No video in carousel"}
                info = video_info
            
            video_id = info.get("id", "unknown")
            filename = f"test_reels/{video_id}.mp4"
            meta = {
                "id": info.get("id", "unknown"),
                "caption": info.get("description", "") or info.get("title", ""),
                "uploader": info.get("uploader_id") or info.get("uploader"),
                "hashtags": info.get("tags") or [],
                "url": url,
            }
            return {"meta": meta, "local_path": f"test_reels/{video_id}.mp4", "url": url}
    except Exception as e:
        return {"error": str(e), "url": url}


def run_reelscout(video_path: str) -> dict:
    """Run reelscout on a video and return summary."""
    try:
        result = subprocess.run(
            ["python3", "-m", "reelscout", video_path, "--text-llm", "always", "--vision", "auto", "--resolve", "auto", "--summary"],
            capture_output=True, text=True, timeout=120, cwd="/Users/debanikdebnath/work/reelscout"
        )
        # Extract summary from output
        output = result.stdout + result.stderr
        summary_line = ""
        for line in output.split("\n"):
            if line and not line.startswith("[") and not line.startswith("Deprecated") and not line.startswith("objc["):
                summary_line = line.strip()
                break
        return {"summary": summary_line, "success": True}
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "url": ""}
    except Exception as e:
        return {"error": str(e)}


def main():
    print(f"Testing with {len(REEL_URLS)} reels")
    results = []
    
    for i, url in enumerate(REEL_URLS):
        print(f"\n[{i+1}/{len(REEL_URLS)}] Processing: {url}")
        
        # Download
        print("  Downloading...")
        download_result = download_reel(url, Path("test_reels"), Path("cookies.txt"))
        
        if "error" in download_result:
            print(f"  Download failed: {download_result['error']}")
            results.append({"url": url, "error": download_result["error"]})
            continue
        
        local_path = download_result["local_path"]
        print(f"  Downloaded: {local_path}")
        
        # Run reelscout
        print("  Analyzing...")
        result = run_reelscout(local_path)
        
        if "error" in result:
            print(f"  Analysis failed: {result['error']}")
            results.append({"url": url, "error": result["error"]})
        else:
            print(f"  Summary: {result['summary'][:100]}...")
            results.append({"url": url, "summary": result["summary"], "local_path": local_path})
        
        # Rate limiting - be nice to Instagram
        time.sleep(5)
    
    # Save results
    with open("test_30_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"✗ {r['url']}: {r['error']}")
        else:
            print(f"✓ {r['url']}: {r['summary'][:80]}...")

if __name__ == "__main__":
    main()