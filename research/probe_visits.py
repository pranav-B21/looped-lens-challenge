"""Probe every layer visit of Inkling 2 Mini on our labelled questions, on NDIF.

    .venv/bin/python research/probe_visits.py [--questions research/data/dev/questions.jsonl]
        [--out research/data/probe] [--per-request 8] [--limit N] [--positions 12]

For each question, one forward pass reads all 48 visits under the four readouts of
``probe_lens.READOUTS``, at the last ``--positions`` tokens of the input, and sends back only:

- ``ranks`` [48, readouts, positions, variants] int16: the rank (0 = best) of each single-token
  form of each option (" Saturday", "saturday", ...), which is enough to replay the voting rule for
  any choice of visits, readout and position;
- ``top`` [48, readouts, 10]: the ten best tokens at the read position, to see what the lens shows.

Each request's results are written to ``--out/batch-NNNN.pt`` as soon as they arrive, and batches
already there are skipped, so an interrupted probe resumes where it stopped.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import nnsight
import torch
from nnsight import TransformersModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))
import lenses  # noqa: E402  visit_address, layers
import probe_lens  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
VISITS = 48
TOP = 10


def variants(tokenizer, options):
    """([token id], [option index]) for every single-token form of every option."""
    ids, owners = [], []
    for o, option in enumerate(options):
        for word in dict.fromkeys([option, option.lower(), option.capitalize()]):
            for form in (word, " " + word):
                encoded = tokenizer(form, add_special_tokens=False)["input_ids"]
                if len(encoded) == 1 and encoded[0] not in ids:
                    ids.append(encoded[0])
                    owners.append(o)
    return ids, owners


def probe_batch(model, addresses, batch):
    """One NDIF request over ``batch``: [(input ids [1, n], positions [P], variant ids [V])]."""
    results = []
    with model.session(remote=True):
        for ids, positions, variant_ids in batch:
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                per_visit = []
                for layer_index, run in addresses:
                    for _ in tracer.iter[run]:
                        layer = lenses.layers(model)[layer_index]
                        hidden, blocks = layer.inputs[0][:2]
                        dtype = hidden.dtype
                        h = hidden[:, positions]  # [1, P, hidden]
                        b = blocks[:, positions].to(h.device)  # [1, P, finished blocks, hidden]; a sharded model keeps them on another GPU
                        states = [h, probe_lens.mixed(model, b, h)]
                        states.append(layer.post_attention_layernorm.input[:, positions].to(h.device))
                        after = torch.cat([b, h.unsqueeze(-2)], dim=-2) if layer.is_block_boundary else b
                        states.append(probe_lens.mixed(model, after, layer.output[:, positions].to(h.device)))
                        rank_rows, top_rows = [], []
                        for state in states:
                            logits = probe_lens.unembed(model, state, dtype)[0]  # [P, vocab]
                            rank_rows.append(probe_lens.ranks(logits, variant_ids.to(logits.device)))
                            top_rows.append(logits[-1].topk(TOP).indices.to(torch.int32).to(rank_rows[-1].device))
                        per_visit.append((torch.stack(rank_rows).cpu(), torch.stack(top_rows).cpu()))
            results.append(per_visit)
        results = nnsight.save(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--questions", type=Path, default=here / "data" / "dev" / "questions.jsonl")
    parser.add_argument("--out", type=Path, default=here / "data" / "probe")
    parser.add_argument("--per-request", type=int, default=8)
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="only the first N questions")
    parser.add_argument("--visits", type=int, default=VISITS, help="only the first N visits (for a quick test)")
    parser.add_argument("--shard", default="0/1", help="i/n: only batches whose number is i modulo n, to run n probes side by side")
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.questions) if line.strip()][: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    nnsight.register(lenses)
    nnsight.register(probe_lens)
    model = TransformersModel(MODEL_ID)
    addresses = [lenses.visit_address(model.config.text_config, v) for v in range(args.visits)]
    args.out.mkdir(parents=True, exist_ok=True)

    shard, shards = map(int, args.shard.split("/"))
    for start in range(0, len(questions), args.per_request):
        path = args.out / f"batch-{start // args.per_request:04d}.pt"
        if path.exists() or (start // args.per_request) % shards != shard:
            continue
        chunk = questions[start : start + args.per_request]
        batch, meta = [], []
        for q in chunk:
            ids = tokenizer(q["model_input"], add_special_tokens=False)["input_ids"]
            assert len(ids) - 1 == q["read_position"], q["id"]
            positions = list(range(max(0, len(ids) - args.positions), len(ids)))
            variant_ids, owners = variants(tokenizer, q["options"])
            batch.append((torch.tensor([ids]), torch.tensor(positions), torch.tensor(variant_ids)))
            meta.append({"id": q["id"], "positions": positions, "variant_ids": variant_ids, "owners": owners,
                         "tokens": [tokenizer.decode([i]) for i in ids[positions[0]:]]})
        for attempt in range(3):
            try:
                began = time.monotonic()
                results = probe_batch(model, addresses, batch)
                break
            except Exception as error:  # a passing NDIF error: wait and try again
                print(f"batch at {start}: attempt {attempt + 1} failed: {type(error).__name__}: {error}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            sys.exit(f"batch at {start} failed three times")
        rows = []
        for m, per_visit in zip(meta, results):
            m["ranks"] = torch.stack([r for r, _ in per_visit])  # [visits, readouts, P, V]
            m["top"] = torch.stack([t for _, t in per_visit])  # [visits, readouts, 10]
            rows.append(m)
        torch.save(rows, path)
        print(f"probed {min(start + args.per_request, len(questions))} of {len(questions)} questions ({time.monotonic() - began:.0f}s for this request)", flush=True)


if __name__ == "__main__":
    main()
