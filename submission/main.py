"""The starter submission: read one dataset of questions with a logit lens, on NDIF.

The leaderboard starts this file once per dataset, with these environment variables set:

    DATASET_NAME   the dataset's name
    DATASET_PATH   a directory holding questions.jsonl, the questions to read (see questions.py)
    OUTPUT_DIR     where readouts.jsonl goes
    MODEL_ID       the model being read
    NDIF_API_KEY, NDIF_HOST, HF_TOKEN   your credentials; nnsight and huggingface_hub read them

For each question the lens reads one layer visit at one token position and reports the ten tokens
it finds most likely there, with the answer lens (lenses.answer_lens): a trained low-rank
correction of that state, then the model's own final norm and unembedding. Tokens that cannot name
anything are left out.

The position is chosen per prompt from its punctuation, as LENS_RULES.md allows: the question's
last "?", read at visit 46 with the lens trained there; a prompt with no "?" is read at its read
position, at visit 44, with the lens trained there. Either way every readout of a prompt is at one
position. The corrections were fitted on our own labelled questions (research/answer_lens.py) and
are published on the Hugging Face Hub; NDIF fetches them once per request (lenses.fetch).
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

# The trained lens: where its weights are published, and which visit each position is read at.
# Visits are numbered 0-47 in the order they run (lenses.visit_address); layers 6-17 run three times.
LENS_REPOSITORY = "pranavbelligundu/inkling-looped-lens"
LENS_FILE = "answer_lens_v1.safetensors"
AT_QUESTION = ("qmark.v46", 46)  # the question's last "?"
AT_READ = ("read.v44", 44)  # the read position, for prompts with no "?"
# Tokens that cannot name an option (punctuation, whitespace, chat markup) are left out of every
# readout, so that the ten places go to tokens that can (lenses.junk_ids).
TOP = 10  # tokens reported per visit
# One NDIF request runs this many forward passes. A scored dataset (90-100 questions) fits in one,
# so it waits in NDIF's queue once; only ten token ids per question come back. Overridden by
# LENS_QUESTIONS_PER_REQUEST, so that a local test can make the run send several requests.
QUESTIONS_PER_REQUEST = int(os.environ.get("LENS_QUESTIONS_PER_REQUEST", 100))

# Robustness. The leaderboard gives all three datasets 45 minutes together and fails the whole
# submission if one run errors. A request that fails with an error is retried. One that has no
# answer yet is waited for until this dataset's budget runs out, not resent: a resent request
# joins the back of NDIF's queue, which under load only makes the wait longer. When the budget
# runs out the questions read so far are written and the run still exits cleanly.
ATTEMPTS = 3  # tries per request, after errors
DATASET_BUDGET = 13 * 60  # seconds for this whole dataset; three of them must fit in 45 minutes


STARTED = time.monotonic()
nnsight.register(lenses)  # the lens functions run on NDIF, so they are shipped with each request
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
items = questions.load(tokenizer)
model = TransformersModel(MODEL_ID)
# lm_head scores the unpadded vocabulary.
junk = torch.tensor(lenses.junk_ids(tokenizer, model.config.text_config.unpadded_vocab_size))


def question_end(ids: list[int], read_position: int) -> int | None:
    """The position of the prompt's last "?" before the read position, or None: a fixed rule on the
    prompt's punctuation, which never looks at what the question asks."""
    for i in range(read_position - 1, -1, -1):
        if "?" in tokenizer.decode([ids[i]]):
            return i
    return None


def plan(item):
    """Where one question is read: (position, lens name, visit). One position for the whole prompt."""
    question = question_end(item.ids, item.position)
    if question is not None:
        return (question,) + AT_QUESTION
    return (item.position,) + AT_READ


def read_batch(batch):
    """One NDIF request: one forward pass per question, the ten best token ids of its readout back."""
    inputs = []
    for item in batch:
        position, name, visit = plan(item)
        layer, run = lenses.visit_address(model.config.text_config, visit)
        inputs.append((torch.tensor([item.ids]), position, name, layer, run))
    best = []
    with model.session(remote=True):
        # Fetched by NDIF itself, once per request and before the first trace.
        weights = lenses.fetch(LENS_REPOSITORY, LENS_FILE)
        for ids, position, name, layer, run in inputs:
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                # A looped layer runs three times in this one forward pass; tracer.iter picks
                # which of its runs the read is taken from.
                for _ in tracer.iter[run]:
                    h = lenses.layers(model)[layer].input[0, position]  # the state entering the visit: [hidden]
                    scores = lenses.answer_lens(model, h, weights, name)
                    top = lenses.top_tokens(scores, junk, TOP).cpu()
            best.append(top)
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
        worker.join(timeout=remaining)
        if "best" in result:
            return result["best"]
        if worker.is_alive():
            abandoned = True
            print(f"{questions.DATASET}: no answer within the budget ({remaining:.0f}s)", file=sys.stderr, flush=True)
            return None
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
    for item, token_ids in zip(batch, best):
        _, _, visit = plan(item)
        readouts[item.id] = [{"layer": visit, "tokens": questions.token_strings(tokenizer, token_ids.tolist())}]
    print(f"{questions.DATASET}: read {min(start + QUESTIONS_PER_REQUEST, len(items))} of {len(items)} questions", flush=True)

questions.OUTPUT.parent.mkdir(parents=True, exist_ok=True)
questions.write(items, readouts)
at_question = sum(question_end(item.ids, item.position) is not None for item in items)
print(f"{questions.DATASET}: wrote {len(readouts)} of {len(items)} questions in {time.monotonic() - STARTED:.0f}s "
      f"({at_question} read at their \"?\" with visit 46, {len(items) - at_question} at the read position with visit 44)", flush=True)
sys.stdout.flush()
sys.stderr.flush()
# A request abandoned while it hung, or a connection nnsight leaves open, can keep the interpreter
# from exiting; the readouts are written, so leave now.
os._exit(0)
