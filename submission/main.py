"""The starter submission: read one dataset of questions with a logit lens, on NDIF.

The leaderboard starts this file once per dataset, with these environment variables set:

    DATASET_NAME   the dataset's name
    DATASET_PATH   a directory holding questions.jsonl, the questions to read (see questions.py)
    OUTPUT_DIR     where readouts.jsonl goes
    MODEL_ID       the model being read
    NDIF_API_KEY, NDIF_HOST, HF_TOKEN   your credentials; nnsight and huggingface_hub read them

For each question the lens reads the model at one token position, at up to ten layer visits of
its choosing, and reports the ten tokens it finds most likely at each. To submit your own lens:
write a function that turns the state entering a layer visit into what your lens says about it
(lenses.py), call it in the trace where ``lenses.logit_lens`` is called, and choose the visits
it reads.
"""

import os

import nnsight
import torch
from nnsight import TransformersModel
from transformers import AutoTokenizer

import lenses
import questions

MODEL_ID = os.environ["MODEL_ID"]

# The layer visits this lens reads, the same ones for every question. The model has 24 layers,
# but layers 6-17 run three times per forward pass, so a token passes through 48 visits, numbered
# 0-47 in the order they run (lenses.visit_address). These are layer 12 in each of its three
# runs (12, 24, 36), the two visits before the last of those, and the last five. At most ten, in
# ascending order: a trace reads the model in the order the model runs.
# Drop early visit 12 (rarely votes) and final 47 (special-token noise); add denser late
# third-pass / early-coda visits where dual reads surface calendar intermediates.
VISITS = [24, 34, 35, 36, 40, 41, 43, 44, 45, 46]
TOP = 10  # tokens reported per visit
QUESTIONS_PER_REQUEST = 25  # one NDIF request runs this many forward passes

# Skip chat markup and function words when picking an earlier "anchor" token to read beside
# the official read_position. The last remaining token is usually the entity the question
# asks about (a day, month, landmark, or number).
_ANCHOR_SKIP = {
    "word", ".", "answer", "in", "one", "the", "what", "is", "after", "before", "month", "day",
    "?", ",", "a", "an", "of", "to", "and",
}

assert VISITS == sorted(set(VISITS)) and len(VISITS) <= questions.MAX_LAYERS, VISITS


def anchor_position(tokenizer, ids: list[int], read_position: int) -> int:
    """Index of the last contentful prompt token before ``read_position`` (else ``read_position``)."""
    chosen = None
    for i, token_id in enumerate(ids):
        if i >= read_position:
            break
        text = tokenizer.decode([token_id])
        if text.startswith("<|") and text.endswith("|>"):
            continue
        if not text.strip() or text.strip().lower() in _ANCHOR_SKIP:
            continue
        chosen = i
    return chosen if chosen is not None else read_position


nnsight.register(lenses)  # the lens functions run on NDIF, so they are shipped with each request
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
items = questions.load(tokenizer)
model = TransformersModel(MODEL_ID)
# Each visit as (layer, which of that layer's runs): what the trace below asks NDIF for.
addresses = [lenses.visit_address(model.config.text_config, visit) for visit in VISITS]

readouts = {}
for start in range(0, len(items), QUESTIONS_PER_REQUEST):
    batch = items[start : start + QUESTIONS_PER_REQUEST]
    inputs = [
        (torch.tensor([item.ids]), item.position, anchor_position(tokenizer, item.ids, item.position))
        for item in batch
    ]

    # One request: everything in this block runs on NDIF, one forward pass per question. Only
    # the ten best token ids per visit come back.
    best = []
    with model.session(remote=True):
        for ids, position, anchor in inputs:
            with model.trace({"input_ids": ids, "attention_mask": torch.ones_like(ids)}) as tracer:
                per_visit = []
                for layer, run in addresses:
                    # A looped layer runs three times in this one forward pass; tracer.iter
                    # picks which of its runs the next read is taken from.
                    for _ in tracer.iter[run]:
                        block = lenses.layers(model)[layer]
                        h_read = block.input[0, position]
                        h_anchor = block.input[0, anchor]
                        scores = lenses.dual_logit_lens(model, h_read, h_anchor)
                        per_visit.append(scores.topk(TOP).indices.cpu())
            best.append(per_visit)
        nnsight.save(best)

    for item, per_visit in zip(batch, best):
        readouts[item.id] = [{"layer": visit, "tokens": questions.token_strings(tokenizer, token_ids.tolist())} for visit, token_ids in zip(VISITS, per_visit)]
    print(f"{questions.DATASET}: read {min(start + QUESTIONS_PER_REQUEST, len(items))} of {len(items)} questions", flush=True)

questions.write(items, readouts)
print(f"{questions.DATASET}: wrote {len(readouts)} questions at visits {VISITS}")
