"""Train and score an "answer lens": a low-rank affine map per visit before the model's own head.

    .venv/bin/python research/answer_lens.py [--position read] [--rank 64] [--epochs 40] [--out ...]

For each visit v the lens reads one state h (at one position, the same for every visit of a
prompt, as LENS_RULES.md requires) and scores the vocabulary as

    logits = unembed @ (rmsnorm(h + (h @ U_v) @ V_v^T + b_v) * norm_weight)

with the model's own unembedding and final-norm weight (research/fetch_unembed.py). U_v, V_v and
b_v start at zero, so before training this is exactly the starter's logit lens. Training raises
the log-probability of the question's intermediate (any single-token form of it) on our own
training questions (research/make_train_questions.py), never the dev ones.

It then scores each visit alone on held-out training questions and on the dev set (the voting
rule, exact match, junk filter as in submission/lenses.py), chooses up to --max-visits visits on
the held-out *training* questions, and reports that design on dev against the plain logit lens at
the same visits. Weights go to --out as safetensors in bf16: v{visit}.U [hidden, rank],
v{visit}.V [hidden, rank], v{visit}.b [hidden].
"""

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "submission"))
sys.path.insert(0, str(HERE))
import lenses  # noqa: E402
from probe_visits import variants  # noqa: E402
from score import norm as norm_text  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
VOCAB = 200058  # what lm_head scores


def load_states(directory, questions):
    """Rows from ``directory`` and its siblings named like it (``train`` and ``train-rest``, collected
    at different request sizes), one per question id."""
    rows, visits, seen = [], None, set()
    directory = Path(directory)
    for path in sorted(directory.parent.glob(directory.name + "*/batch-*.pt")):
        d = torch.load(path)
        visits = d["visits"]
        for r in d["rows"]:
            if r["id"] in questions and r["id"] not in seen:
                seen.add(r["id"])
                rows.append(r)
    return rows, visits


class Lens(torch.nn.Module):
    def __init__(self, hidden, rank, unembed, gain):
        super().__init__()
        self.U = torch.nn.Parameter(torch.zeros(hidden, rank))
        self.V = torch.nn.Parameter(torch.randn(hidden, rank) * 1e-3)
        self.b = torch.nn.Parameter(torch.zeros(hidden))
        self.unembed, self.gain = unembed, gain

    def state(self, h):
        return h + (h @ self.U) @ self.V.T + self.b

    def forward(self, h):
        s = self.state(h)
        s = s / s.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * self.gain
        return s @ self.unembed.T


def target_mask(tokenizer, questions, ids):
    """[N, vocab] bool: the single-token forms of each question's intermediate."""
    mask = torch.zeros(len(ids), VOCAB, dtype=torch.bool)
    for n, qid in enumerate(ids):
        q = questions[qid]
        forms, owners = variants(tokenizer, q["options"])
        gold = q["options"].index(q["intermediate"])
        for f, o in zip(forms, owners):
            if o == gold and f < VOCAB:
                mask[n, f] = True
    return mask


def votes(top_ids, questions, ids, tokenizer, cache):
    """The option index each readout votes for (-1 for none): [N]."""
    out = []
    for n, qid in enumerate(ids):
        opts = {norm_text(o): k for k, o in enumerate(questions[qid]["options"])}
        vote = -1
        for i in top_ids[n].tolist():
            if i not in cache:
                cache[i] = norm_text(tokenizer.decode([i]))
            if cache[i] in opts:
                vote = opts[cache[i]]
                break
        out.append(vote)
    return torch.tensor(out)


def pass_rate(vote_table, design, gold, families):
    """Mean over families of the share passing, for readouts at the visits in ``design``."""
    passed = []
    for n in range(len(gold)):
        count = Counter(int(vote_table[v][n]) for v in design if int(vote_table[v][n]) >= 0)
        g = count.pop(int(gold[n]), 0)
        passed.append(g > max(count.values(), default=0))
    passed = torch.tensor(passed)
    rates = {f: passed[[i for i, x in enumerate(families) if x == f]].float().mean().item() for f in sorted(set(families))}
    return sum(rates.values()) / len(rates), rates


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--position", choices=("read", "?"), default="read")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--max-visits", type=int, default=10)
    parser.add_argument("--style", default="chat", help="dev questions to report on: chat, plain or all")
    parser.add_argument("--out", type=Path, default=HERE / "data" / "answer_lens.safetensors")
    args = parser.parse_args()
    pos = 0 if args.position == "read" else 1
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    weights = load_file(str(HERE / "data" / "unembed.safetensors"))
    unembed = weights["unembed"][:VOCAB].float().to(device)
    gain = weights["norm"].float().to(device)
    junk = torch.tensor(lenses.junk_ids(tokenizer, VOCAB))

    train_q = {q["id"]: q for q in map(json.loads, open(HERE / "data" / "train" / "questions.jsonl"))}
    dev_q = {q["id"]: q for q in map(json.loads, open(HERE / "data" / "dev" / "questions.jsonl"))}
    if args.style != "all":
        dev_q = {k: q for k, q in dev_q.items() if q["style"] == args.style}
    train_rows, visits = load_states(HERE / "data" / "states" / "train", train_q)
    dev_rows, _ = load_states(HERE / "data" / "states" / "dev", dev_q)
    print(f"{len(train_rows)} train and {len(dev_rows)} dev questions with states; visits {visits[0]}-{visits[-1]}; position {args.position}; device {device}")

    # Hold out a tenth of the training *prompts* (by entity text) to choose visits on.
    def held(qid):
        return zlib.crc32(train_q[qid]["prompt"].encode()) % 10 == 0
    fit_rows = [r for r in train_rows if not held(r["id"])]
    val_rows = [r for r in train_rows if held(r["id"])]

    def stack(rows):
        return torch.stack([r["x"][:, pos] for r in rows]).float()  # [N, visits, hidden]

    X_fit, X_val, X_dev = stack(fit_rows), stack(val_rows), stack(dev_rows)
    ids_fit, ids_val, ids_dev = [r["id"] for r in fit_rows], [r["id"] for r in val_rows], [r["id"] for r in dev_rows]
    T_fit = target_mask(tokenizer, train_q, ids_fit)
    keep = T_fit.any(1)
    X_fit, T_fit, ids_fit = X_fit[keep], T_fit[keep], [i for i, k in zip(ids_fit, keep) if k]

    cache = {}
    tables = {"val": ({}, {}), "dev": ({}, {})}  # split -> (trained votes, identity votes) per visit
    tensors = {}
    for k, visit in enumerate(visits):
        lens = Lens(X_fit.shape[-1], args.rank, unembed, gain).to(device)
        optimizer = torch.optim.AdamW(lens.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        x, t = X_fit[:, k].to(device), T_fit.to(device)
        for epoch in range(args.epochs):
            order = torch.randperm(len(x))
            for start in range(0, len(x), 128):
                idx = order[start : start + 128]
                logits = lens(x[idx])
                loss = (torch.logsumexp(logits, -1) - torch.logsumexp(logits.masked_fill(~t[idx], float("-inf")), -1)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            for split, X, ids, qs in (("val", X_val, ids_val, train_q), ("dev", X_dev, ids_dev, dev_q)):
                h = X[:, k].to(device)
                for which, logits in ((0, lens(h)), (1, Lens(h.shape[-1], 1, unembed, gain).to(device).forward(h))):
                    logits = logits.cpu().index_fill(1, junk, float("-inf"))
                    tables[split][which][visit] = votes(logits.topk(10).indices, qs, ids, tokenizer, cache)
        gold_val = torch.tensor([train_q[i]["options"].index(train_q[i]["intermediate"]) for i in ids_val])
        gold_dev = torch.tensor([dev_q[i]["options"].index(dev_q[i]["intermediate"]) for i in ids_dev])
        acc = lambda split, which, gold: (tables[split][which][visit] == gold).float().mean().item()
        print(f"  visit {visit}: final loss {loss.item():.2f} | single-readout hit, trained vs logit lens: "
              f"val {acc('val', 0, gold_val):.2f} vs {acc('val', 1, gold_val):.2f}, dev {acc('dev', 0, gold_dev):.2f} vs {acc('dev', 1, gold_dev):.2f}", flush=True)
        tensors[f"v{visit}.U"] = lens.U.detach().cpu().to(torch.bfloat16).contiguous()
        tensors[f"v{visit}.V"] = lens.V.detach().cpu().to(torch.bfloat16).contiguous()
        tensors[f"v{visit}.b"] = lens.b.detach().cpu().to(torch.bfloat16).contiguous()

    fam_val = [train_q[i]["family"] for i in ids_val]
    fam_dev = [dev_q[i]["family"] for i in ids_dev]
    # Greedy choice of visits on the held-out training questions, trained lens.
    design, best = [], -1.0
    for _ in range(args.max_visits):
        score, visit = max((pass_rate(tables["val"][0], design + [v], gold_val, fam_val)[0], v) for v in visits if v not in design)
        if score <= best:
            break
        design, best = design + [visit], score
    design.sort()
    dev_trained, rates_t = pass_rate(tables["dev"][0], design, gold_dev, fam_dev)
    dev_plain, rates_p = pass_rate(tables["dev"][1], design, gold_dev, fam_dev)
    print(f"\nvisits chosen on held-out training questions: {design} (val {best:.3f})")
    print(f"dev ({args.style}), trained lens: {dev_trained:.3f}  " + " ".join(f"{f} {x:.2f}" for f, x in rates_t.items()))
    print(f"dev ({args.style}), logit lens:   {dev_plain:.3f}  " + " ".join(f"{f} {x:.2f}" for f, x in rates_p.items()))
    save_file(tensors, str(args.out), metadata={"visits": json.dumps(design), "position": args.position, "rank": str(args.rank)})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
