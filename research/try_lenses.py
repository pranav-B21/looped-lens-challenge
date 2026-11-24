"""Try many lens variants in one forward pass per question, on NDIF, and keep their top tokens.

    .venv/bin/python research/try_lenses.py [--style chat] [--out research/data/variants] [--per-request 20]

For each question one trace reads visits 40-47 at two positions: the read position and the
question's last "?" (the read position if there is none). For each (visit, readout, position)
it keeps the top ten tokens, with and without the junk filter, and it pools log-probabilities over
several sets of states into single ranked lists. Everything comes back as token ids, so the voting
rule can be replayed for any combination offline (research/score_variants.py).

Readouts: ``active`` (the state entering the layer) and ``mlp`` (the MoE input); see probe_lens.
The junk filter removes tokens with no letter or digit and the chat markup tokens: a fixed
property of the tokenizer, the same for every question.
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
import lenses  # noqa: E402
import probe_lens  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
VISITS = list(range(40, 48))
TOP = 10
# Pools: name -> (readouts, visits, positions), summed log-probabilities. Position 0 is the read
# position, 1 the question's "?".
POOLS = {
    "act43-47both": (("active",), range(43, 48), (0, 1)),
    "act40-47both": (("active",), range(40, 48), (0, 1)),
    "act43-47read": (("active",), range(43, 48), (0,)),
    "act44-46q": (("active",), range(44, 47), (1,)),
    "mlp43-47both": (("mlp",), range(43, 48), (0, 1)),
    "both43-47both": (("active", "mlp"), range(43, 48), (0, 1)),
    "both40-47both": (("active", "mlp"), range(40, 48), (0, 1)),
}


def junk_mask(tokenizer, vocab_size: int) -> torch.Tensor:
    """True for tokens that cannot name an option: no letter or digit, chat markup, or padding."""
    mask = torch.ones(vocab_size, dtype=torch.bool)
    for token_id in range(min(len(tokenizer), vocab_size)):
        text = tokenizer.decode([token_id])
        if any(ch.isalnum() for ch in text) and not (text.startswith("<|") and text.endswith("|>")):
            mask[token_id] = False
    return mask


def question_end(tokenizer, ids, read_position):
    for i in range(read_position - 1, -1, -1):
        if "?" in tokenizer.decode([ids[i]]):
            return i
    return read_position


def run_batch(model, addresses, batch, mask):
    results = []
    with model.session(remote=True):
        for ids, positions in batch:
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                tops = []
                pooled = {}
                for visit, (layer_index, run) in zip(VISITS, addresses):
                    for _ in tracer.iter[run]:
                        layer = lenses.layers(model)[layer_index]
                        hidden = layer.inputs[0][0]
                        dtype = hidden.dtype
                        states = {"active": hidden[0, positions], "mlp": layer.post_attention_layernorm.input[0, positions].to(hidden.device)}
                        per_readout = []
                        for readout in ("active", "mlp"):
                            logits = probe_lens.unembed(model, states[readout], dtype)  # [2, vocab]
                            keep = mask.to(logits.device)
                            filtered = logits.masked_fill(keep, float("-inf"))
                            per_readout.append(torch.stack([logits.topk(TOP).indices, filtered.topk(TOP).indices]))  # [2 filters, 2 positions, 10]
                            logprobs = torch.log_softmax(logits, -1)
                            for name, (readouts, visits, pos) in POOLS.items():
                                if readout in readouts and visit in visits:
                                    term = logprobs[list(pos)].sum(0)
                                    pooled[name] = term if name not in pooled else pooled[name] + term.to(pooled[name].device)
                        tops.append(torch.stack(per_readout).cpu())  # [2 readouts, 2 filters, 2 positions, 10]
                pool_tops = []
                for name in POOLS:
                    scores = pooled[name]
                    filtered = scores.masked_fill(mask.to(scores.device), float("-inf"))
                    pool_tops.append(torch.stack([scores.topk(TOP).indices, filtered.topk(TOP).indices]).cpu())  # [2 filters, 10]
            results.append((tops, pool_tops))
        results = nnsight.save(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--questions", type=Path, default=here / "data" / "dev" / "questions.jsonl")
    parser.add_argument("--style", default="chat", help="chat, plain, or all")
    parser.add_argument("--out", type=Path, default=here / "data" / "variants")
    parser.add_argument("--per-request", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    questions = [json.loads(l) for l in open(args.questions) if l.strip()]
    if args.style != "all":
        questions = [q for q in questions if q["style"] == args.style]
    questions = questions[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    nnsight.register(lenses)
    nnsight.register(probe_lens)
    model = TransformersModel(MODEL_ID)
    vocab = model.config.text_config.unpadded_vocab_size  # what lm_head outputs
    mask_path = args.out / "junk_mask.pt"
    args.out.mkdir(parents=True, exist_ok=True)
    if mask_path.exists():
        mask = torch.load(mask_path)
    else:
        mask = junk_mask(tokenizer, vocab)
        torch.save(mask, mask_path)
    print(f"junk filter: {int(mask.sum())} of {vocab} tokens", flush=True)
    addresses = [lenses.visit_address(model.config.text_config, v) for v in VISITS]

    for start in range(0, len(questions), args.per_request):
        path = args.out / f"batch-{start // args.per_request:04d}.pt"
        if path.exists():
            continue
        chunk = questions[start : start + args.per_request]
        batch = []
        for q in chunk:
            ids = tokenizer(q["model_input"], add_special_tokens=False)["input_ids"]
            batch.append((torch.tensor([ids]), [q["read_position"], question_end(tokenizer, ids, q["read_position"])]))
        for attempt in range(3):
            try:
                began = time.monotonic()
                results = run_batch(model, addresses, batch, mask)
                break
            except Exception as error:
                print(f"batch at {start}: attempt {attempt + 1} failed: {type(error).__name__}: {str(error)[-300:]}", flush=True)
                time.sleep(10 * (attempt + 1))
        else:
            sys.exit(f"batch at {start} failed three times")
        rows = []
        for q, (tops, pool_tops) in zip(chunk, results):
            rows.append({"id": q["id"], "tops": torch.stack(tops).to(torch.int32), "pools": torch.stack(pool_tops).to(torch.int32)})
        torch.save(rows, path)
        print(f"ran {min(start + args.per_request, len(questions))} of {len(questions)} ({time.monotonic() - began:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
