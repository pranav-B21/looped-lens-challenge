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
  - The fix is not committed yet; it will be once the collection finishes, because nnsight reads the script's source while it runs.
- **The other session's commits on this branch:** ed572e4 (agents/research.md, technical-notes.md) and 5c9c7ee (a `CLAUDE.md` addition pointing agents to the notes). The user has been told about the `CLAUDE.md` change, and it is the user's call.

## Now (01:30)

- **Collection** is running: 13/38 training and 5/11 dev batches.
- **The user's instruction:** don't train until every batch is in.

## Next

1. **Train on all 8 families**, at the read position and at the `?`. Choose visits on held-out training questions; compare with the logit lens on dev.
2. **Publish** the weights to a public HF repo.
3. **Wire** the lens into `submission/` (fetch with `lenses.fetch`, apply per visit, keep the junk filter).
4. **Run the full tests** and the LENS_RULES.md checklist, then submit.
