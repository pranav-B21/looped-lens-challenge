"""Score designs (sets of visits, all at one position) with a trained answer lens, offline.

    .venv/bin/python research/eval_designs.py [--position ?] [--weights research/data/answer_lens-qmark.safetensors]

Loads the lens weights research/answer_lens.py wrote and the collected states, computes every
visit's junk-filtered top ten for the held-out training questions and the dev questions (chat and
plain), and reports a list of designs, plus the best designs of 2 to 5 visits found by exhaustive
search on the held-out training questions. Dev is only reported, never searched on.
"""

import argparse
import itertools
import json
import zlib
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from answer_lens import HERE, MODEL_ID, VOCAB, lenses, load_states, pass_rate, votes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--position", choices=("read", "?"), default="?")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--search-from", type=int, nargs="*", default=list(range(34, 48)), help="visits the search may use")
    args = parser.parse_args()
    pos = 0 if args.position == "read" else 1
    weights_path = args.weights or HERE / "data" / f"answer_lens-{'read' if pos == 0 else 'qmark'}.safetensors"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    head = load_file(str(HERE / "data" / "unembed.safetensors"))
    unembed, gain = head["unembed"][:VOCAB].float().to(device), head["norm"].float().to(device)
    lens_weights = {k: v.float().to(device) for k, v in load_file(str(weights_path)).items()}
    junk = torch.tensor(lenses.junk_ids(tokenizer, VOCAB))

    train_q = {q["id"]: q for q in map(json.loads, open(HERE / "data" / "train" / "questions.jsonl"))}
    dev_q = {q["id"]: q for q in map(json.loads, open(HERE / "data" / "dev" / "questions.jsonl"))}
    train_rows, visits = load_states(HERE / "data" / "states" / "train", train_q)
    dev_rows, _ = load_states(HERE / "data" / "states" / "dev", dev_q)
    val_rows = [r for r in train_rows if zlib.crc32(train_q[r["id"]]["prompt"].encode()) % 10 == 0]
    splits = {
        "val": (val_rows, train_q),
        "dev-chat": ([r for r in dev_rows if dev_q[r["id"]]["style"] == "chat"], dev_q),
        "dev-plain": ([r for r in dev_rows if dev_q[r["id"]]["style"] == "plain"], dev_q),
    }

    cache, tables = {}, {}
    for split, (rows, qs) in splits.items():
        ids = [r["id"] for r in rows]
        X = torch.stack([r["x"][:, pos] for r in rows]).float().to(device)  # [N, visits, hidden]
        table = {}
        with torch.no_grad():
            for k, visit in enumerate(visits):
                h = X[:, k]
                s = h + (h @ lens_weights[f"v{visit}.U"]) @ lens_weights[f"v{visit}.V"].T + lens_weights[f"v{visit}.b"]
                s = s / s.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * gain
                logits = (s @ unembed.T).cpu().index_fill(1, junk, float("-inf"))
                table[visit] = votes(logits.topk(10).indices, qs, ids, tokenizer, cache)
        gold = torch.tensor([qs[i]["options"].index(qs[i]["intermediate"]) for i in ids])
        tables[split] = (table, gold, [qs[i]["family"] for i in ids])

    def report(design):
        cells = []
        for split, (table, gold, fams) in tables.items():
            score, rates = pass_rate(table, design, gold, fams)
            cells.append(f"{split} {score:.3f}")
        _, rates = pass_rate(tables["dev-chat"][0], design, tables["dev-chat"][1], tables["dev-chat"][2])
        worst = min(rates.items(), key=lambda kv: kv[1])
        print(f"  {str(design):<34} " + "  ".join(cells) + f"   dev-chat weakest: {worst[0]} {worst[1]:.2f}")

    print(f"position {args.position}, weights {weights_path.name}")
    for design in ([46], [44], [43, 44, 45, 46], [44, 45, 46], [43, 44, 45, 46, 47], [34, 36, 43, 44, 45, 46], [34, 36, 40, 43, 44, 45, 46, 47]):
        report(design)

    print("\nbest designs by held-out training questions (exhaustive, 2-5 visits):")
    table, gold, fams = tables["val"]
    found = []
    for size in range(2, 6):
        for design in itertools.combinations(args.search_from, size):
            found.append((pass_rate(table, list(design), gold, fams)[0], list(design)))
    found.sort(key=lambda x: -x[0])
    for _, design in found[:8]:
        report(design)


if __name__ == "__main__":
    main()
