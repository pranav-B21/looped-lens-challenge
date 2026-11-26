"""Snapshot and summarize the public leaderboard: who scores what, reading which visits.

    .venv/bin/python research/leaderboard.py [--top 15] [--history 10] [--team "Pranav B"]

Fetches https://ndif-looped-lens-challenge.hf.space/api/leaderboard (public; the same feed
https://waka.mx/loops analyses), saves it to research/data/leaderboard/<UTC time>.json, and prints:
each team's best entry with its per-dataset scores and visits, how often each visit is read, and the
average score of entries that read a visit against those that do not. ``--history N`` also prints
every submission of the top N teams in time order (score, datasets, run time, flag, visits), which
shows whether a team gained by changing visits or by changing its lens; ``--team`` does the same
for one team. Entries only report the
visits they read, not their lens or positions, so the visit statistics are a hint, not a cause.
"""

import argparse
import json
import re
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

URL = "https://ndif-looped-lens-challenge.hf.space/api/leaderboard"
STARTER = [12, 24, 34, 35, 36, 43, 44, 45, 46, 47]


def visits_of(row) -> list[int]:
    match = re.search(r"Layers read: ([\d, ]+)", row.get("message") or "")
    return [int(v) for v in match.group(1).split(",")] if match else []


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--history", type=int, default=0, metavar="N", help="print every submission of the top N teams")
    parser.add_argument("--team", action="append", default=[], help="print every submission of this team (repeatable)")
    args = parser.parse_args()

    with urllib.request.urlopen(urllib.request.Request(URL, headers={"User-Agent": "lens-research"}), timeout=60) as response:
        data = json.load(response)
    rows = [r for r in data["rows"] if r.get("metrics", {}).get("pass_rate") is not None]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).parent / "data" / "leaderboard"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stamp}.json").write_text(json.dumps(data, indent=1))

    best = sorted((r for r in rows if r.get("best")), key=lambda r: -r["metrics"]["pass_rate"])
    print(f"snapshot {stamp}: {len(rows)} scored entries, {len(best)} teams\n")
    print(f"{'#':>3} {'team':<22} {'score':>6}  {'A':>5} {'B':>5} {'C':>5} {'secs':>5}  visits")
    for n, r in enumerate(best[: args.top], 1):
        parts = {b["name"][-1]: b["metrics"]["pass_rate"] for b in r.get("breakdown", [])}
        visits = visits_of(r)
        tag = "  (starter's visits)" if visits == STARTER else ""
        tag += "  FLAGGED" if r.get("flagged") else ""
        print(f"{n:>3} {r['team'][:22]:<22} {r['metrics']['pass_rate']:.3f}  {parts.get('A', 0):.2f}  {parts.get('B', 0):.2f}  {parts.get('C', 0):.2f} {r.get('seconds') or 0:>5.0f}  {visits}{tag}")

    starters = [r["metrics"]["pass_rate"] for r in rows if visits_of(r) == STARTER]
    if starters:
        print(f"\nentries reading exactly the starter's visits: {len(starters)}, scores {min(starters):.3f} to {max(starters):.3f}")

    counted = [(visits_of(r), r["metrics"]["pass_rate"]) for r in rows if visits_of(r)]
    frequency = Counter(v for visits, _ in counted for v in visits)
    print(f"\nvisits read, over all {len(counted)} entries (visit: entries, score with minus without):")
    cells = []
    for v in range(48):
        with_v = [s for visits, s in counted if v in visits]
        without = [s for visits, s in counted if v not in visits]
        if with_v and without:
            cells.append(f"{v}:{len(with_v)} {100 * (sum(with_v) / len(with_v) - sum(without) / len(without)):+.1f}")
    for i in range(0, len(cells), 8):
        print("  " + "   ".join(f"{c:<12}" for c in cells[i : i + 8]))

    teams = [r["team"] for r in best if not r.get("baseline")][: args.history] + args.team
    for team in teams:
        history = sorted((r for r in rows if r["team"] == team), key=lambda r: r["submitted_at"])
        print(f"\n{team}: {len(history)} scored submissions")
        for r in history:
            parts = {b["name"][-1]: b["metrics"]["pass_rate"] for b in r.get("breakdown", [])}
            flag = " FLAGGED" if r.get("flagged") else ""
            print(f"  {r['submitted_at'][:16]} {r['metrics']['pass_rate']:.3f}  A {parts.get('A', 0):.2f} B {parts.get('B', 0):.2f} C {parts.get('C', 0):.2f} {r.get('seconds') or 0:>5.0f}s{flag}  {visits_of(r)}")


if __name__ == "__main__":
    main()
