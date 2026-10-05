"""Process all local test reels through the pipeline with phone storage."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reelscout.config import Settings
from reelscout.pipeline import analyze_reel_with_storage
from reelscout.storage import get_phone_storage

TEST_REELS = Path(__file__).resolve().parent / "test_reels"


def main() -> None:
    meta_path = TEST_REELS / "metadata.json"
    metadata = json.loads(meta_path.read_text())
    storage = get_phone_storage()

    # Map reel id -> local mp4
    local_videos = {p.stem: p for p in TEST_REELS.glob("*.mp4")}
    print(f"Found {len(local_videos)} local videos: {list(local_videos)}")

    for url, entry in metadata.items():
        reel_id = entry.get("meta", {}).get("id") or url.rstrip("/").rsplit("/", 1)[-1]
        video = local_videos.get(reel_id)
        if video is None:
            print(f"[skip] {reel_id}: no local mp4")
            continue
        existing = storage.load_result(f"reel-{reel_id}")
        if existing is not None:
            print(f"[skip] {reel_id}: already in storage")
            continue
        print(f"[run] {reel_id} ({video.name})")
        result = analyze_reel_with_storage(
            Path(str(video)),
            settings=Settings.from_env(),
            job_id=f"reel-{reel_id}",
        )
        print(f"[done] {reel_id}: {len(result.source_frames)} frames, "
              f"{len(result.candidates)} candidates")

    print("\nStorage contents:")
    stats = storage.get_stats()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
