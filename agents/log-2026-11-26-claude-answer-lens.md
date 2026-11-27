# 2026-11-26/27, Claude (session "looped-lense-challenge-planning"): the answer lens

Branch `idea/answer-lens`, started from the compliant v4 (`idea/one-position`, da67f01). Times are UTC.

## Done

- **19:36 Leaderboard check.**
  - The top score is 0.990, the 10th team is at 0.906, and the median team at 0.769.
  - The official starter baseline is unchanged at 0.662, so the evaluation did not change.
  - Our v2 (0.712) shows `flagged: true`. v4 (0.666) is not flagged.
  - The conclusion, in agreement with the "looped-lense research" session: top teams improved their *lens*, not their visits.
- **Trained lenses are allowed.** LENS_RULES.md permits "trained weights fitted on our own prompts, with no answer tables".
- **The unembedding** (`research/fetch_unembed.py`): `model.llm.unembed.weight` [201024, 2304] bf16 and `model.llm.norm.weight`, fetched by HTTP range request from the public checkpoint (12 s, 0.93 GB). The local head reproduces NDIF's logit lens top 10 **exactly**, checked on 2 questions at visit 44.
- **Training questions** (`research/make_train_questions.py`): 1,518 questions in 8 families, chat format (about 25% also plain).
  - New entities and new phrasings; 0 dev prompts. 16 clashes with dev were dropped.
  - Calendar facts repeat (7 days, 12 months), so dev is only partly independent for shift1 and shift2.
- **State collection** (`research/collect_states.py`): visits 30–47 at the read position and at the `?`, in bf16, 40 questions per request. A compliant lens later uses one of the two positions for *all* its visits.
- **The trainer** (`research/answer_lens.py`, 43937a8):
  - the map: `h + (h U) V^T + b` per visit (rank 64, starting as the plain logit lens), then the model's own RMS norm and unembedding;
  - the loss: the log-probability of any single-token form of the intermediate;
  - visits are chosen on held-out *training* questions and reported on dev.
- **Small test** (240 training and 40 dev questions, 3 families, 5 epochs, 24 s on MPS): the trained lens beats the logit lens at almost every visit.
  - Single-readout hit rates on dev: v36 0.77 vs 0.62; v40 0.43 vs 0.00; v44 0.88 vs 0.80; v46 0.85 vs 0.70.
  - Small sample only; not a result.
- **Judge:** `research/judge_score.py --provider xai --model <grok model>` sends the official judge prompts to Grok. It needs `XAI_API_KEY` in `.env`, which is not there yet. The user has no OpenAI access.
- **HF token:** the user's `looped-lense-2` token can write (`repo.write`) under `pranavbelligundu`, enough to publish lens weights.

## Incidents

- **20:02 Data loss, recovered.** The checkout was switched to `main` in GitHub Desktop. `main`'s `.gitignore` doesn't ignore `research/data/`, so Desktop stashed all 118 data files as "!!GitHub_Desktop<main>", and the folder vanished mid-collection.
  - Restored with `git archive stash@{0} research/data | tar -x`. The stash is left in place.
  - `research/data/` is now in `.git/info/exclude`, so it is ignored on every branch.
- **20:15–01:28 The collection hung.** All 4 collectors sat on NDIF requests that never answered, for more than 5 hours. At 00:31 I misreported it as about a third done: those were old batch counts, not live progress.
  - Fixed: each request now times out (`--timeout`, 600 s here) and is sent again, and the script ends with `os._exit(0)`.
  - Restarted at 01:30. The 13 training and 5 dev batches already collected are kept.
  - The fix is committed as 06a0516. Committing does not change the file the running collectors read.
- **The other session's commits on this branch:** ed572e4 (agents/research.md, technical-notes.md) and 5c9c7ee (a `CLAUDE.md` addition pointing agents to the notes). The user has been told about the `CLAUDE.md` change, and it is the user's call.

## Now (01:30)

- **Collection** is running: 13/38 training and 5/11 dev batches.
- **The user's instruction:** don't train until every batch is in.

## Next

1. **Train on all 8 families**, at the read position and at the `?`. Choose visits on held-out training questions; compare with the logit lens on dev.
2. **Publish** the weights to a public HF repo.
3. **Wire** the lens into `submission/` (fetch with `lenses.fetch`, apply per visit, keep the junk filter).
4. **Run the full tests** and the LENS_RULES.md checklist, then submit.

## 01:50 The collection stalled again: large downloads stopped coming back

- **What happened:** after the restart, all 4 requests (40 questions each, about 6.6 MB of results) reached "RUNNING" within 10–40 s on NDIF and then returned nothing for 600 s. A 4-question request (about 0.7 MB) finished in 226 s. NDIF reports the model running.
  - The upstream repository recently added "Stream large downloads through the egress proxy", which fits large results failing to come back.
- **The change:**
  - the questions not yet collected (998 training, 234 dev) are in `research/data/states/{train,dev}-rest/questions.jsonl`;
  - they are collected **10 per request** (about 1.7 MB), with 6 training and 2 dev collectors in parallel;
  - `research/answer_lens.py` reads `states/train*` and `states/dev*`, one row per question id.
- **Expected time:** 124 requests over 8 collectors is about 16 rounds of about 4 min, so roughly 60–70 min if NDIF holds.

## ~02:10 The collection is complete

- **All 1,518 training and all 434 dev questions** have states, with none missing and no failed requests. Checked by question id over `states/{train,train-rest,dev,dev-rest}`.
- **The 10-question requests took 1.5–4 min each.** The 8 collectors finished the remaining 124 requests in about 20 min.

## 02:15 User questions answered and added to the notes

- **Are we allowed to collect states for our own questions?** Yes: AGENTS.md "Training a lens" describes exactly this. The answer is in research.md section 9.
- **What are we training?** The lens, not the model. Answered in research.md section 9.
- **Is publishing to HF and fetching on NDIF allowed?** Yes: the README and AGENTS.md describe this route, and LENS_RULES.md allows tuned lenses on one visit at one position.
  - The gray area: the unembedding was downloaded locally for training. The rule against reading weights applies to code on NDIF, and the entry only calls `model.lm_head`. The user may confirm on Discord.
- **Which stage is trained, where it is stored, why it should help:** added to research.md section 4 (plain words) and to technical-notes.md (formula, tensors, storage). Commit 4c2110a.

## 02:30 First full training run: every visit at the read position

`research/answer_lens.py --position read --epochs 40` trained on 1,518 questions and was scored on 217 dev chat questions. Weights are in `research/data/answer_lens-read.safetensors`.

- **Single-readout hit rate on dev, trained vs. plain logit lens:**

  | Visits | Trained | Logit lens |
  |---|---|---|
  | 30–35 | 0.38–0.77 | 0.00–0.16 |
  | 36 | 0.79 | 0.29 |
  | 37–42 | 0.55–0.66 | 0.00–0.05 |
  | 43 | **0.90** | 0.37 |
  | 44 | 0.86 | 0.49 |
  | 45 | 0.81 | 0.47 |
  | 46 | 0.85 | 0.43 |
  | 47 | 0.67 | 0.04 |

- **Visits chosen on held-out training questions:** `[44]` only (val 0.810). No second readout improved the held-out pass rate.
- **Dev (chat) for that design:**
  - **trained 0.868:** arith1 0.73, arith2 0.97, country 1.00, hop_country 1.00, instrument 0.84, shift1 0.50, shift2 0.90, sport 1.00;
  - **logit lens at the same visit, 0.521:** arith1 0.17, arith2 0.43, country 0.73, hop_country 0.80, instrument 0.63, shift1 0.00, shift2 0.40, sport 1.00.
  - For reference, v4's 7-visit design with the logit lens scores about 0.55.
- **Readings:**
  - this is the first lens that reads arithmetic (0.17 → 0.73) and two-step questions (arith2 0.97, shift2 0.90);
  - shift1 (0.50) is the weak spot;
  - the greedy choice stopped at one readout. Next, test designs with several strong visits (43–46), which may help the hidden sets even when the held-out set says one is enough.
- **Caveat:** calendar facts overlap between training and dev, with different wording, so shift1 and shift2 dev scores may be optimistic.
- **Running:** the same training with every visit at the `?`.

## The answer-lens pipeline (reference)

Where the trained part sits in the leaderboard's pipeline for each hidden question. Only the
correction box is trained; Inkling and its norm and unembedding stay frozen.

```
 hidden question (we never see it)
        │
        ▼
 Inkling 2 Mini runs on NDIF  ─────────────  FROZEN (nobody can change it)
        │   at each chosen visit, at ONE position:
        ▼
 hidden state h (2,304 numbers)
        │
        ▼
 ┌──────────────────────────────┐
 │  correction for this visit    │  ◀── THE ONLY PART WE TRAIN
 │  h + (h·U)·Vᵀ + b             │      (U, V: 2,304 × 64;  b: 2,304)
 └──────────────────────────────┘
        │
        ▼
 Inkling's own final norm + unembedding ─── FROZEN (called on NDIF as model.lm_head)
        │
        ▼
 scores for 200,058 tokens → junk filter → top 10 tokens = one readout
        │
        ▼
 up to 10 readouts → leaderboard judge → each readout votes → pass / fail
```
