"""Collect, on NDIF, the states a trained lens reads: visits 30-47 at two candidate positions.

    .venv/bin/python research/collect_states.py QUESTIONS.jsonl OUT_DIR [--per-request 20] [--shard 0/1]

For each question, one forward pass saves in bf16 ``x`` [visits, 2, hidden]: the state entering
each visit in VISITS at the read position (index 0) and at the question's last "?" (index 1; the
read position when there is none). A compliant lens later reads *one* of the two for all its
visits; collecting both lets that choice be made offline. ``check`` [10] is the logit lens's top
ten at visit 44 and the read position, computed on NDIF, so that research/answer_lens.py can
confirm its local head reproduces NDIF's. Batches already in OUT_DIR are skipped.
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import nnsight
import torch
from nnsight import TransformersModel
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))
import lenses  # noqa: E402

MODEL_ID = "thinkingmachines/Inkling-2-Mini"
VISITS = list(range(30, 48))
CHECK_VISIT = 44


def question_end(tokenizer, ids, read_position):
    for i in range(read_position - 1, -1, -1):
        if "?" in tokenizer.decode([ids[i]]):
            return i
    return read_position


def collect(model, addresses, batch):
    results = []
    with model.session(remote=True):
        for ids, positions in batch:
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                xs = []
                check = None
                for visit, (layer_index, run) in zip(VISITS, addresses):
                    for _ in tracer.iter[run]:
                        hidden = lenses.layers(model)[layer_index].input[0]
                        xs.append(hidden[positions].to(torch.bfloat16).cpu())
                        if visit == CHECK_VISIT:
                            check = lenses.logit_lens(model, hidden[positions[0]]).float().topk(10).indices.cpu()
            results.append((xs, check))
        results = nnsight.save(results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--per-request", type=int, default=20)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--timeout", type=int, default=900, help="seconds to wait for one request before sending it again")
    args = parser.parse_args()
    shard, shards = map(int, args.shard.split("/"))
    questions = [json.loads(l) for l in open(args.questions) if l.strip()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    nnsight.register(lenses)
    model = TransformersModel(MODEL_ID)
    addresses = [lenses.visit_address(model.config.text_config, v) for v in VISITS]
    args.out.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(questions), args.per_request):
        number = start // args.per_request
        path = args.out / f"batch-{number:04d}.pt"
        if path.exists() or number % shards != shard:
            continue
        chunk = questions[start : start + args.per_request]
        batch = []
        for q in chunk:
            ids = tokenizer(q["model_input"], add_special_tokens=False)["input_ids"]
            assert len(ids) - 1 == q["read_position"], q["id"]
            batch.append((torch.tensor([ids]), [q["read_position"], question_end(tokenizer, ids, q["read_position"])]))
        results = None
        for attempt in range(4):
            began = time.monotonic()
            box = {}

            def run():
                try:
                    box["results"] = collect(model, addresses, batch)
                except BaseException as error:  # reported below
                    box["error"] = error

            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            # A request NDIF never answers would otherwise block this collector for good.
            worker.join(timeout=args.timeout)
            if "results" in box:
                results = box["results"]
                break
            reason = f"no answer after {args.timeout}s" if worker.is_alive() else f"{type(box['error']).__name__}: {str(box['error'])[-300:]}"
            print(f"batch {number}: attempt {attempt + 1} failed: {reason}", flush=True)
            time.sleep(20 * (attempt + 1))
        if results is None:
            print(f"batch {number}: giving up for now; rerun to retry it", flush=True)
            continue
        rows = [{"id": q["id"], "x": torch.stack(xs), "check": check} for q, (xs, check) in zip(chunk, results)]
        torch.save({"visits": VISITS, "rows": rows}, path)
        print(f"batch {number}: {len(rows)} questions in {time.monotonic() - began:.0f}s", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # an abandoned request's thread must not keep the process alive
