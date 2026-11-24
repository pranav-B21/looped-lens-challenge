"""The starter submission: read one dataset of questions with a logit lens, on NDIF.

The leaderboard starts this file once per dataset, with these environment variables set:

    DATASET_NAME   the dataset's name
    DATASET_PATH   a directory holding questions.jsonl, the questions to read (see questions.py)
    OUTPUT_DIR     where readouts.jsonl goes
    MODEL_ID       the model being read
    NDIF_API_KEY, NDIF_HOST, HF_TOKEN   your credentials; nnsight and huggingface_hub read them

For each question the lens reads the model at seven layer visits and reports the ten tokens it
finds most likely at each, with the starter's logit lens (lenses.logit_lens). Six visits are read
at the question's read position. One is read at the end of the question itself, its last "?":
on questions such as "What is the day after Friday?" the model has worked the intermediate out
there, and the instruction that follows ("Answer in one word.") washes it out before the read
position. The choice was measured on our own labelled questions (research/), not on the practice
set: see agents/findings.md.
"""

import os
import sys
import threading
import time

import nnsight
import torch
from nnsight import TransformersModel
from transformers import AutoTokenizer

import lenses
import questions

MODEL_ID = os.environ["MODEL_ID"]

# The layer visits this lens reads, the same ones for every question. The model has 24 layers,
# but layers 6-17 run three times per forward pass, so a token passes through 48 visits, numbered
# 0-47 in the order they run (lenses.visit_address). At most ten, in ascending order: a trace reads
# the model in the order the model runs. Fewer readouts are better than more when the extra ones
# vote for wrong options, which raises the bar the intermediate has to clear.
# Layer 12's input in passes 2 and 3 (visits 24 and 36: a block boundary, holding the finished
# block of layers 6-11) and the coda carry most questions at the read position; the leaderboard's
# entries that read 24, 36 and 43-47 score far above those that read the coda alone.
VISITS = [24, 36, 43, 44, 45, 46, 47]
# Visits read at the end of the question (its last "?") rather than at the read position.
AT_QUESTION = {46}
# Tokens that cannot name an option (punctuation, whitespace, chat markup) are left out of every
# readout, so that the ten places go to tokens that can (lenses.junk_ids).
TOP = 10  # tokens reported per visit
# One NDIF request runs this many forward passes. LENS_QUESTIONS_PER_REQUEST overrides it, so that
# a local test can make the run send several requests.
QUESTIONS_PER_REQUEST = int(os.environ.get("LENS_QUESTIONS_PER_REQUEST", 25))

# Robustness. The leaderboard gives all three datasets 45 minutes together and fails the whole
# submission if one run errors, so a request that fails is retried, one that hangs is abandoned
# after REQUEST_TIMEOUT seconds and retried, and when this dataset's time budget runs out the
# questions read so far are written and the run still exits cleanly.
REQUEST_TIMEOUT = 300  # seconds for one attempt at one request, queue wait included
ATTEMPTS = 3  # tries per request
DATASET_BUDGET = 13 * 60  # seconds for this whole dataset; three of them must fit in 45 minutes

assert VISITS == sorted(set(VISITS)) and len(VISITS) <= questions.MAX_LAYERS, VISITS

STARTED = time.monotonic()
nnsight.register(lenses)  # the lens functions run on NDIF, so they are shipped with each request
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
items = questions.load(tokenizer)
model = TransformersModel(MODEL_ID)
# Each visit as (layer, which of that layer's runs): what the trace below asks NDIF for.
addresses = [lenses.visit_address(model.config.text_config, visit) for visit in VISITS]
# lm_head scores the unpadded vocabulary.
junk = torch.tensor(lenses.junk_ids(tokenizer, model.config.text_config.unpadded_vocab_size))


def question_end(ids: list[int], read_position: int) -> int:
    """The position of the question's last "?" before the read position: a fixed rule on the
    prompt's punctuation, never on what it asks. The read position when there is none."""
    for i in range(read_position - 1, -1, -1):
        if "?" in tokenizer.decode([ids[i]]):
            return i
    return read_position


def read_batch(batch):
    """One NDIF request: one forward pass per question, the ten best token ids per visit back."""
    inputs = []
    for item in batch:
        question = question_end(item.ids, item.position)
        # The token each visit is read at, in the order of VISITS.
        wheres = [question if visit in AT_QUESTION else item.position for visit in VISITS]
        inputs.append((torch.tensor([item.ids]), wheres))
    best = []
    with model.session(remote=True):
        for ids, wheres in inputs:
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                per_visit = []
                for (layer, run), where in zip(addresses, wheres):
                    # A looped layer runs three times in this one forward pass; tracer.iter
                    # picks which of its runs the next read is taken from.
                    for _ in tracer.iter[run]:
                        h = lenses.layers(model)[layer].input[0, where]  # the state entering the visit: [hidden]
                        scores = lenses.logit_lens(model, h)
                        per_visit.append(lenses.top_tokens(scores, junk, TOP).cpu())
            best.append(per_visit)
        best = nnsight.save(best)
    if len(best) != len(batch):
        raise RuntimeError(f"the request returned {len(best)} results for {len(batch)} questions")
    return best


abandoned = False  # whether a hung request was left running in a background thread


def read_with_retries(batch):
    """``read_batch`` with a timeout and retries. Returns None when every attempt failed."""
    global abandoned
    for attempt in range(1, ATTEMPTS + 1):
        remaining = DATASET_BUDGET - (time.monotonic() - STARTED)
        if remaining < 30:
            print(f"{questions.DATASET}: out of time, not sending more requests", flush=True)
            return None
        result = {}

        def run():
            try:
                result["best"] = read_batch(batch)
            except BaseException as error:  # reported below; a thread cannot raise into the caller
                result["error"] = error

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout=min(REQUEST_TIMEOUT, remaining))
        if "best" in result:
            return result["best"]
        if worker.is_alive():
            abandoned = True
            reason = f"no answer after {min(REQUEST_TIMEOUT, remaining):.0f}s"
        else:
            reason = f"{type(result['error']).__name__}: {result['error']}"
        print(f"{questions.DATASET}: request failed (attempt {attempt} of {ATTEMPTS}): {reason}", file=sys.stderr, flush=True)
        if attempt < ATTEMPTS:
            time.sleep(5 * attempt)
    return None


readouts = {}
for start in range(0, len(items), QUESTIONS_PER_REQUEST):
    batch = items[start : start + QUESTIONS_PER_REQUEST]
    best = read_with_retries(batch)
    if best is None:
        print(f"{questions.DATASET}: giving up on questions {start} to {start + len(batch) - 1}", file=sys.stderr, flush=True)
        continue
    for item, per_visit in zip(batch, best):
        readouts[item.id] = [{"layer": visit, "tokens": questions.token_strings(tokenizer, token_ids.tolist())} for visit, token_ids in zip(VISITS, per_visit)]
    print(f"{questions.DATASET}: read {min(start + QUESTIONS_PER_REQUEST, len(items))} of {len(items)} questions", flush=True)

questions.OUTPUT.parent.mkdir(parents=True, exist_ok=True)
questions.write(items, readouts)
print(f"{questions.DATASET}: wrote {len(readouts)} of {len(items)} questions at visits {VISITS} in {time.monotonic() - STARTED:.0f}s", flush=True)
sys.stdout.flush()
sys.stderr.flush()
# A request abandoned while it hung, or a connection nnsight leaves open, can keep the interpreter
# from exiting; the readouts are written, so leave now.
os._exit(0)
