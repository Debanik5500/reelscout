#!/usr/bin/env python3
"""Quick labeling script for test reels."""
from __future__ import annotations
import csv
import json
from pathlib import Path

META_FILE = Path(__file__).parent.parent / "test_reels" / "metadata.json"
LABELS_FILE = Path(__file__).parent.parent / "eval" / "labels.csv"

# Bucket definitions:
# A = name spoken/on screen
# B = only in caption/comments/tags
# C = visual only
# D = link gated but inferable
# E = not inferable

# Pre-filled labels based on what we know
LABELS = {
    "Dd7x6noPjAM.mp4": {
        "bucket": "D",
        "expected": "raunakpatil/youtube-agentic-ai-studio;youtube-agentic-ai-studio;AI YouTube Studio;seb.ai"
    },
    "Dd6VsR7PI5w.mp4": {
        "bucket": "A",
        "expected": "JayWebtech/autoshorts;autoshorts;AutoShorts;ffmpeg"
    },
    "Dd8s6U4ztQK.mp4": {
        "bucket": "A",
        "expected": "youart.ai;youart.ai/mcp;Claude"
    },
    "Dd7sPgmD32w.mp4": {
        "bucket": "A",
        "expected": "Durupınar site;Noah's Ark;Atatürk University;noahsarkscans.com"
    },
    "Ddoe6dpyyka.mp4": {
        "bucket": "D",
        "expected": "Manali;stayo.india;Manu Temple"
    },
    "DdToB05gH14.mp4": {
        "bucket": "A",
        "expected": "api.anthropic.com;Claude Code;gittrend.io;github.com/anthropics/claude-code"
    },
}

def main():
    # Load metadata
    with open(META_FILE) as f:
        metadata = json.load(f)
    
    # Build labels.csv
    rows = []
    for url, data in metadata.items():
        if "error" in data:
            continue
        local_path = data.get("local_path")
        if not local_path:
            continue
        
        filename = Path(local_path).name
        label = LABELS.get(filename, {"bucket": "E", "expected": ""})
        
        rows.append({
            "source": local_path,
            "bucket": label["bucket"],
            "expected": label["expected"]
        })
    
    # Write labels.csv
    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "bucket", "expected"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Created {LABELS_FILE} with {len(rows)} labeled reels")
    for r in rows:
        print(f"  {Path(r['source']).name}: bucket={r['bucket']}, expected={r['expected'][:60]}...")

if __name__ == "__main__":
    main()