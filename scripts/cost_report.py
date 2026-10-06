"""Per-reel LLM cost from the usage each reel recorded (results/<id>.context.json -> debug.usage).

  python scripts/cost_report.py            # all reels in ~/ReelScoutData
Prices come from pricing.json (USD per million tokens). A model with no price is listed but not costed.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRICES = {k: v for k, v in json.loads((ROOT / "pricing.json").read_text()).items() if not k.startswith("_")}
DATA = Path(os.path.expanduser("~/ReelScoutData/results"))


def calls(usage: dict) -> list[tuple[str, dict]]:
    """(stage, {in,out,calls,model}) rows. A reel's `ledger` (exact per-model totals) wins over the per-stage rows."""
    if usage.get("ledger"):
        return [("ledger", {**row, "model": model}) for model, row in usage["ledger"].items()]
    rows = []
    for stage, value in usage.items():
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, dict) and (item.get("in") or item.get("out")):
                rows.append((stage if not item.get("stage") else f"{stage}/{item['stage']}", item))
    return rows


def cost(row: dict) -> float | None:
    price = PRICES.get(row.get("model", ""))
    if not price:
        return None
    return (row.get("in", 0) * price["in"] + row.get("out", 0) * price["out"]) / 1_000_000


def main() -> None:
    reels = []
    for path in sorted(glob.glob(str(DATA / "*.context.json"))):
        job = os.path.basename(path).replace(".context.json", "")
        data = json.load(open(path))
        usage = data.get("debug", {}).get("usage", {})
        reels.append((job, data["context"].get("title", "")[:34], calls(usage), "observe" in usage))

    providers = sorted({r.get("model", "?").split(":")[0] for _, _, rows, _ in reels for _, r in rows})
    by_model: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    complete = []
    print(f"{'reel':<20}{'title':<36}" + "".join(f"{p:>9}" for p in providers) + f"{'total':>9}")
    for job, title, rows, has_vision in reels:
        per = defaultdict(float)
        for _, r in rows:
            per[r.get("model", "?").split(":")[0]] += cost(r) or 0
            m = by_model[r.get("model", "?")]
            m[0] += r.get("in", 0); m[1] += r.get("out", 0); m[2] += r.get("calls", 1)
        total = sum(per.values())
        note = "" if has_vision else "  (older run: vision not recorded)"
        print(f"{job:<20}{title:<36}" + "".join(f"{per.get(p, 0):>9.4f}" for p in providers) + f"{total:>9.4f}{note}")
        if has_vision:
            complete.append(total)

    print("\nTokens by model (all reels):")
    for model, (tin, tout, n) in by_model.items():
        price = PRICES.get(model)
        usd = f"${(tin * price['in'] + tout * price['out']) / 1e6:.4f}" if price else "no price"
        print(f"  {model:<28} calls={n:<3} in={tin:<8} out={tout:<7} {usd}")
    if complete:
        avg = sum(complete) / len(complete)
        print(f"\nFully measured reels: {len(complete)} | average ${avg:.4f}/reel | 100 reels ~ ${avg * 100:.2f} | 1000 reels ~ ${avg * 1000:.2f}")


if __name__ == "__main__":
    main()
