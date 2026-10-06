"""Pre-process reels into phone storage so the app can load everything locally.

  python process_reels.py --limit 10                     # every mp4 in test_reels/
  python process_reels.py --urls reels.txt --limit 10    # one reel URL per line
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reelscout import batch
from reelscout.config import Settings
from reelscout.storage import get_phone_storage
from server.keymoments import _resolve_username

TEST_REELS = Path(__file__).resolve().parent / "test_reels"


def sources(args: argparse.Namespace) -> list[tuple[str, str]]:
    """(source, caption) pairs."""
    if args.urls:
        return [(u.strip(), "") for u in Path(args.urls).read_text().splitlines() if u.strip() and not u.startswith("#")]
    meta_path = TEST_REELS / "metadata.json"
    captions = {}
    if meta_path.exists():
        for url, entry in json.loads(meta_path.read_text()).items():
            captions[(entry.get("meta") or {}).get("id") or url.rstrip("/").rsplit("/", 1)[-1]] = (entry.get("meta") or {}).get("caption", "")
    return [(str(p), captions.get(p.stem, "")) for p in sorted(TEST_REELS.glob("*.mp4"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--urls", help="text file with one reel URL per line")
    ap.add_argument("--force", action="store_true", help="re-process reels already in storage")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    st = Settings.from_env()
    todo = sources(args)[: args.limit]
    print(f"Processing {len(todo)} reel(s)")
    ok = 0
    for i, (source, caption) in enumerate(todo, 1):
        try:
            job_id = batch.process_reel(source, st, caption, args.force, _resolve_username)
            ok += 1
            print(f"[{i}/{len(todo)}] done {job_id}")
        except Exception as exc:
            print(f"[{i}/{len(todo)}] FAILED {source}: {exc}")
    print(f"\n{ok}/{len(todo)} processed")
    print(json.dumps(get_phone_storage().get_stats(), indent=2))


if __name__ == "__main__":
    main()
