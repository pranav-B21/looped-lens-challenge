"""Score the lens variants try_lenses.py collected, alone and in combinations.

    .venv/bin/python research/score_variants.py [--variants research/data/variants] [--max-readouts 4]

A candidate readout is either one state's top ten, (visit, readout, position, filtered), or a
pooled list, (pool name, filtered). Each candidate votes for the option its best-ranked naming
token names (official normalization, research/score.py). A design is a set of candidates with
distinct labels; it is judged, like the leaderboard, by the mean of family pass rates, separately
on the two halves of the prompts, and ranked by the worse half so that a lucky fit does not win.
"""

import argparse
import itertools
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from score import norm  # noqa: E402
from try_lenses import POOLS, VISITS  # noqa: E402

READOUTS = ("active", "mlp")
POSITIONS = ("read", "?")


def candidates():
    for vi, visit in enumerate(VISITS):
        for r, readout in enumerate(READOUTS):
            for p, position in enumerate(POSITIONS):
                for f in (0, 1):
                    yield (f"v{visit}/{readout}/{position}" + ("/filt" if f else ""), visit, ("top", vi, r, f, p))
    for k, name in enumerate(POOLS):
        for f in (0, 1):
            yield (f"pool:{name}" + ("/filt" if f else ""), None, ("pool", k, f))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--variants", type=Path, default=here / "data" / "variants")
    parser.add_argument("--questions", type=Path, default=here / "data" / "dev" / "questions.jsonl")
    parser.add_argument("--max-readouts", type=int, default=4)
    parser.add_argument("--style", default="chat")
    args = parser.parse_args()

    questions = {q["id"]: q for q in map(json.loads, open(args.questions))}
    rows = [r for path in sorted(args.variants.glob("batch-*.pt")) for r in torch.load(path)]
    rows = [r for r in rows if questions[r["id"]]["style"] == args.style]
    tokenizer = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Mini")
    decoded = {}

    def text(i):
        if i not in decoded:
            decoded[i] = norm(tokenizer.decode([i]))
        return decoded[i]

    cands = list(candidates())
    votes = np.full((len(rows), len(cands)), -1, dtype=np.int8)
    for n, row in enumerate(rows):
        q = questions[row["id"]]
        options = {norm(o): k for k, o in enumerate(q["options"])}
        for c, (_, _, key) in enumerate(cands):
            ids = row["tops"][key[1], key[2], key[3], key[4]] if key[0] == "top" else row["pools"][key[1], key[2]]
            for i in ids.tolist():
                if (t := text(i)) in options:
                    votes[n, c] = options[t]
                    break
    gold = np.array([questions[r["id"]]["options"].index(questions[r["id"]]["intermediate"]) for r in rows])
    fam = np.array([questions[r["id"]]["family"] for r in rows])
    half = np.array([zlib.crc32(r["id"].removesuffix("-plain").encode()) % 2 == 0 for r in rows])

    def passed(design):
        counts = np.zeros((len(rows), 5), dtype=np.int16)
        for c in design:
            ok = votes[:, c] >= 0
            counts[np.nonzero(ok)[0], votes[ok, c]] += 1
        g = counts[np.arange(len(rows)), gold].copy()
        counts[np.arange(len(rows)), gold] = -1
        return g > counts.max(1)

    def rate(ok, mask):
        return float(np.mean([ok[mask & (fam == f)].mean() for f in np.unique(fam[mask])]))

    def report(label, design):
        ok = passed(design)
        fams = "  ".join(f"{f[:6]} {ok[fam == f].mean():.2f}" for f in np.unique(fam))
        print(f"  {rate(ok, half):.3f}/{rate(ok, ~half):.3f} all {rate(ok, np.ones_like(half)):.3f}  {label}\n      {fams}")

    index = {name: c for c, (name, _, _) in enumerate(cands)}
    print(f"{len(rows)} {args.style} questions, {len(cands)} candidate readouts\n")
    print("reference designs:")
    report("v1: v44,v45,v47 active read + v46 active ?", [index[n] for n in ("v44/active/read", "v45/active/read", "v46/active/?", "v47/active/read")])
    report("v1 + junk filter", [index[n] for n in ("v44/active/read/filt", "v45/active/read/filt", "v46/active/?/filt", "v47/active/read/filt")])

    print("\nsingle readouts (best 10 by worse half):")
    singles = sorted(((min(rate(passed([c]), half), rate(passed([c]), ~half)), c) for c in range(len(cands))), reverse=True)
    for _, c in singles[:10]:
        report(cands[c][0], [c])

    # Combinations of up to --max-readouts from the 40 best single candidates, with distinct labels.
    pool = [c for _, c in singles[:40]]
    best = []
    for k in range(2, args.max_readouts + 1):
        for design in itertools.combinations(pool, k):
            labels = [cands[c][1] for c in design]
            visits = [v for v in labels if v is not None]
            if len(set(visits)) != len(visits) or labels.count(None) > 1:
                continue
            ok = passed(design)
            best.append((min(rate(ok, half), rate(ok, ~half)), design))
    best.sort(key=lambda x: -x[0])
    print(f"\nbest designs of 2 to {args.max_readouts} readouts (by worse half):")
    for _, design in best[:10]:
        report(" + ".join(cands[c][0] for c in design), list(design))


if __name__ == "__main__":
    main()
