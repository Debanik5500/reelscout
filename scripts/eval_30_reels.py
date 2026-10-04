#!/usr/bin/env python3
"""Evaluate 30 reels - mix of local files and Instagram URLs."""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

# Add your 30 Instagram reel URLs here (one per line)
REEL_URLS = [
    "https://www.instagram.com/reels/Dd7x6noPjAM/",
    "https://www.instagram.com/reels/Dd6VsR7PI5w/",
    "https://www.instagram.com/reels/Dd8s6U4ztQK/",
    "https://www.instagram.com/reels/Dd7sPgmD32w/",
    "https://www.instagram.com/reel/Ddoe6dpyyka/",
    "https://www.instagram.com/reels/DdToB05gH14/",
    # Add 24 more real Instagram reel URLs here:
    # "https://www.instagram.com/reels/XXXXXXXXX/",
    # "https://www.instagram.com/reels/XXXXXXXXX/",
    # ...
]

# Local test files we already have
LOCAL_FILES = [
    "test_reels/Dd7x6noPjAM.mp4",
    "test_reels/Dd6VsR7PI5w.mp4", 
    "test_reels/Dd8s6U4ztQK.mp4",
    "test_reels/Dd7sPgmD32w.mp4",
    "test_reels/Ddoe6dpyyka.mp4",
    "test_reels/DdToB05gH14.mp4",
]

COOKIES_FILE = Path("cookies.txt")

def run_reelscout(source: str) -> dict:
    """Run reelscout and return parsed result."""
    try:
        result = subprocess.run(
            ["python3", "-m", "reelscout", source, "--text-llm", "always", "--vision", "auto", "--resolve", "auto", "--summary"],
            capture_output=True, text=True, timeout=180, cwd="/Users/debanikdebnath/work/reelscout"
        )
        output = result.stdout + result.stderr
        # Extract the summary line (non-debug line)
        summary = ""
        for line in output.split("\n"):
            line = line.strip()
            if line and not line.startswith("[") and not line.startswith("Deprecated") and not line.startswith("objc["):
                summary = line
                break
        return {"success": True, "summary": summary, "raw_output": output}
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}

def test_source(source: str, name: str) -> dict:
    print(f"  Testing: {name}")
    start = time.time()
    result = run_reelscout(source)
    elapsed = time.time() - start
    
    if "error" in result:
        return {"name": name, "source": source, "error": result["error"], "time": elapsed}
    
    print(f"  ✓ Done in {elapsed:.1f}s: {result['summary'][:80]}...")
    return {"name": name, "source": source, "summary": result["summary"], "time": elapsed, "raw": result.get("raw", "")}

def main():
    all_sources = []
    
    # Add local files
    for f in LOCAL_FILES:
        path = Path(f)
        if path.exists():
            all_sources.append((str(path), f"local:{path.name}"))
        else:
            print(f"Missing: {f}")
    
    # Add URLs
    for url in REEL_URLS:
        all_sources.append((url, f"url:{url.split('/')[-2]}"))
    
    print(f"Testing {len(all_sources)} reels...")
    results = []
    
    for i, (source, name) in enumerate(all_sources):
        print(f"\n[{i+1}/{len(all_sources)}] {name}")
        result = test_source(source, name)
        results.append(result)
        time.sleep(3)  # Rate limit
    
    # Save results
    with open("eval_30_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== SUMMARY ===")
    for r in results:
        if "error" in r:
            print(f"✗ {r['name']}: {r['error']}")
        else:
            print(f"✓ {r['name']} ({r['time']:.1f}s): {r['summary'][:100]}...")

if __name__ == "__main__":
    import json
    main()