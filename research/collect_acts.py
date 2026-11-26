"""Collect activations on NDIF to fit a tuned lens: visit states and the model's final state.

    .venv/bin/python research/collect_acts.py [--prompts 300] [--per-request 10] [--shard 0/1]

The texts are generic: Wikitext paragraphs as they are, and Alpaca instructions wrapped in the
model's chat format (the format of the challenge's questions), never our dev questions or their
answers. For each text, at ``--positions`` random token positions, a trace saves in bf16:

- ``x`` [visits, positions, hidden]: the state entering each visit in CANDIDATES (what the
  starter's logit lens reads);
- ``y`` [positions, hidden]: the input of the final norm (``llm.norm.input``), the state the model
  itself unembeds.

A tuned lens fitted on these maps each visit's state to the final one (research/train_lens.py).
Batches land in research/data/acts/batch-NNNN.pt; batches already there are skipped.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import datasets
import nnsight
import torch
from nnsight import TransformersModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))
import lenses  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
CHAT = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{}<|end_message|><|message_model|>"
CANDIDATES = [24, 26, 28, 29, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]


def texts(n, seed):
    rng = random.Random(seed)
    wiki = [t.strip() for t in datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:40000]")["text"]
            if len(t.strip()) > 300 and not t.strip().startswith("=")]
    alpaca = [r["instruction"] + ((" " + r["input"]) if r["input"] else "") for r in datasets.load_dataset("tatsu-lab/alpaca", split="train")]
    rng.shuffle(wiki)
    rng.shuffle(alpaca)
    out = []
    for i in range(n):
        out.append(wiki[i] if i % 2 == 0 else CHAT.format(alpaca[i]))
    return out


def collect(model, addresses, batch):
    results = []
    with model.session(remote=True):
        for ids, positions in batch:
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                xs = []
                for layer_index, run in addresses:
                    for _ in tracer.iter[run]:
                        hidden = lenses.layers(model)[layer_index].inputs[0][0]
                        xs.append(hidden[0, positions].to(torch.bfloat16).cpu())
                y = model.model.language_model.norm.input[0, positions].to(torch.bfloat16).cpu()
            results.append((xs, y))
        results = nnsight.save(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts", type=int, default=300)
    parser.add_argument("--per-request", type=int, default=10)
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "acts")
    args = parser.parse_args()
    shard, shards = map(int, args.shard.split("/"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    nnsight.register(lenses)
    model = TransformersModel(MODEL_ID)
    addresses = [lenses.visit_address(model.config.text_config, v) for v in CANDIDATES]
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    items = []
    for text in texts(args.prompts, args.seed):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][: args.max_tokens]
        if len(ids) < 16:
            continue
        # Skip the first few tokens, whose states are dominated by the start of the text.
        positions = sorted(rng.sample(range(4, len(ids)), min(args.positions, len(ids) - 4)))
        items.append((torch.tensor([ids]), torch.tensor(positions)))

    for start in range(0, len(items), args.per_request):
        number = start // args.per_request
        path = args.out / f"batch-{number:04d}.pt"
        if path.exists() or number % shards != shard:
            continue
        batch = items[start : start + args.per_request]
        for attempt in range(3):
            try:
                began = time.monotonic()
                results = collect(model, addresses, batch)
                break
            except Exception as error:
                print(f"batch {number}: attempt {attempt + 1} failed: {type(error).__name__}: {str(error)[-300:]}", flush=True)
                time.sleep(15 * (attempt + 1))
        else:
            sys.exit(f"batch {number} failed three times")
        x = torch.cat([torch.stack(xs, 0) for xs, _ in results], dim=1)  # [visits, N, hidden]
        y = torch.cat([y for _, y in results], dim=0)  # [N, hidden]
        torch.save({"visits": CANDIDATES, "x": x, "y": y}, path)
        print(f"batch {number}: {y.shape[0]} positions in {time.monotonic() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
