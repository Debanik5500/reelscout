"""Tiny eval harness: label real reels once, then measure accuracy per bucket and per pipeline config.

labels.csv columns:  source, bucket, expected
  source   file path or URL
  bucket   A name spoken/on screen | B only in caption/comments/tags | C visual only | D link gated but inferable | E not inferable
  expected ';'-separated names/domains/repos that count as correct (leave EMPTY for bucket E: success = abstaining)

  python -m reelscout.evaluate eval/labels.csv --llm ollama --ablate
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
from collections import defaultdict

from .config import Settings
from .pipeline import analyze

CONFIDENT = 0.65


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\.", "", s).rstrip("/")
    return re.sub(r"\.git$", "", s).lstrip("@")


def forms(name: str, url: str | None = None) -> set[str]:
    out = set()
    for raw in (name, url):
        n = _norm(raw or "")
        if not n:
            continue
        out |= {n, n.split("/")[-1], n.split("/")[0], n.split(".")[0], n.split("/")[0].split(".")[0]}
    return {f for f in out if len(f) >= 3}


def matches(expected: str, cand: dict) -> bool:
    return bool(forms(expected) & forms(cand["name"], cand.get("url")))


def score_reel(expected: list[str], result: dict) -> dict:
    cands = result["candidates"]
    top = cands[0] if cands else None
    confident = bool(top and top["confidence"] >= CONFIDENT)
    hit = lambda cs: any(matches(e, c) for e in expected for c in cs)
    if not expected:                                  # bucket E: the right behaviour is to abstain
        return dict(correct=not confident, top3=not confident, abstained=not confident, wrong_confident=confident)
    return dict(correct=hit(cands[:1]), top3=hit(cands[:3]), abstained=not confident,
                wrong_confident=confident and not hit(cands[:1]))


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    for r in rows:
        by[r["bucket"]].append(r)
        by["ALL"].append(r)
    out = {}
    for b, rs in sorted(by.items()):
        n = len(rs)
        out[b] = dict(n=n, top1=sum(r["correct"] for r in rs) / n, top3=sum(r["top3"] for r in rs) / n,
                      abstain=sum(r["abstained"] for r in rs) / n, wrong_confident=sum(r["wrong_confident"] for r in rs) / n)
    return out


CONFIGS = {
    "local only": dict(text_policy="never", vision="off", resolve="off"),
    "hybrid (default)": dict(),
    "LLM every time": dict(text_policy="always"),
}


def run(labels_path: str, base: Settings, ablate: bool) -> dict:
    labels = list(csv.DictReader(open(labels_path, newline="")))
    configs = CONFIGS if ablate else {"full": {}}
    report = {}
    for cname, over in configs.items():
        st = dataclasses.replace(base, **over)
        rows = []
        for l in labels:
            exp = [e.strip() for e in (l.get("expected") or "").split(";") if e.strip()]
            try:
                res = analyze(l["source"], st, cookies_from_browser=None)
                row = dict(score_reel(exp, res), used_vision=res["used_vision"], resolved=res["resolution"].get("ran", False),
                           seconds=res["timings"]["total"], top=(res["candidates"][0]["name"] if res["candidates"] else None))
            except Exception as e:
                row = dict(correct=False, top3=False, abstained=True, wrong_confident=False, error=str(e)[:200])
            rows.append(dict(row, source=l["source"], bucket=(l.get("bucket") or "?").strip().upper(), expected=exp))
        report[cname] = dict(summary=summarize(rows), rows=rows)
    return report


def print_report(report: dict) -> None:
    for cname, r in report.items():
        print(f"\n== {cname} ==")
        print(f"{'bucket':8}{'n':>4}{'top1':>8}{'top3':>8}{'abstain':>9}{'WRONG+confident':>17}")
        for b, s in r["summary"].items():
            print(f"{b:8}{s['n']:>4}{s['top1']:>8.0%}{s['top3']:>8.0%}{s['abstain']:>9.0%}{s['wrong_confident']:>17.0%}")
        errs = [x for x in r["rows"] if x.get("error")]
        if errs:
            print(f"  {len(errs)} reel(s) errored, e.g. {errs[0]['source']}: {errs[0]['error']}")
    if len(report) > 1:
        names = list(report)
        print("\nDoes the extra machinery earn its keep? (top1 vs the first config)")
        for n in names[1:]:
            print(f"  {n}:")
            for k, v in report[n]["summary"].items():
                base = report[names[0]]["summary"].get(k)
                if base:
                    print(f"    {k:6} {v['top1'] - base['top1']:+.0%}   wrong+confident {v['wrong_confident'] - base['wrong_confident']:+.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels")
    ap.add_argument("--llm")
    ap.add_argument("--ablate", action="store_true", help="run local-only / hybrid / LLM-every-time side by side")
    ap.add_argument("--out", default="eval/last_run.json")
    a = ap.parse_args()
    st = Settings.from_env()
    st.llm = (a.llm or st.llm).lower()
    report = run(a.labels, st, a.ablate)
    print_report(report)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nper-reel detail: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
