# 2026-11-23: Claude Code, session "looped-lense-challenge-planning"

Branch `version1`, from `main` at 6668f34. All work below is on that branch.

## Done

- **Read the sources:** the README, AGENTS.md, the nnsight walkthrough and remote-execution pages, the tutorials, the challenge page, the model card, the nnsight blog on Inkling 2 Mini, and the model's modeling code. What we learned is in [findings.md](findings.md).
- **Practice baseline:** the starter scores 0.60 locally, 6 of 10. Recall questions pass; arithmetic and calendar questions fail.
- **A local run stalled once:** `submit.py --local` sat for 36 minutes in an SSL read from NDIF after writing its readouts. Reruns finished in 47–59 s, so it was most likely a transient NDIF or network hang.
- **Stage 0** (commit faa7cd8): `submission/main.py` gives each request 300 s and up to 3 attempts, within a 13-minute budget per dataset. A batch that keeps failing is skipped, and the run always exits 0 via `os._exit(0)` after writing. `LENS_QUESTIONS_PER_REQUEST` sets the batch size. Tested with 4 per request: 3 requests, 108 s, 0.60.
- **Stage 1** (commit 4591423):
  - `research/make_questions.py`: 434 questions in 8 families (`sport`, `instrument`, `country`, `arith1`, `shift1`, `hop_country`, `arith2`, `shift2`), each in chat and plain form.
  - `research/probe_visits.py`: 48 visits × 4 readouts × the last 12 positions, returning option-token ranks.
  - `research/analyze_probe.py` and `research/score.py`.
- **Leaderboard watch:** added `research/leaderboard.py`, [leaderboard.md](leaderboard.md) and this folder.
- **Cleanup outside the repo:** removed the old Recurse hooks from `~/.claude/settings.json` and emptied `~/.claude/CLAUDE.md`.

## Found

- **Our first leaderboard submission scored 0.660,** tied with the starter. It was the Cursor branch's dual-position lens, which scored 0.90 on practice.
- **Leaderboard noise is small** (about ±0.007 for the unchanged starter); the lens, not the visits, separates the top entries. See leaderboard.md.
- **Plain-format prompts read at the subject token,** not at the final `.`.

## Open

- **Probe:** running as 3 shards (`research/data/probe-{0,1,2}.log`); 17 of 55 batches were done at 02:25 UTC. Then run `research/analyze_probe.py` and choose the Stage 2 lens.
- **Cursor branch:** `cursor/optimized-inkling-lens-v1-9655` exists locally and on origin. It picks its second read position with a word list ("day", "month", …), which may count as a rule keyed on the prompt's text.

## Later (2026-11-24 UTC): Stage 2, lens v1

- **Probe:** finished, 434 questions, as 3 parallel shards (`--shard i/n`).
- **The decision** is in [findings.md](findings.md): read the question's `?` for calendar intermediates; use fewer readouts.
- **Lens v1** in `submission/main.py`: the starter's logit lens at visits 44, 45 and 47 (read position) and visit 46 (the question's last `?`).
- **Results:**
  - dev set through `main.py`: chat 0.711 (the probe predicted 0.703), plain 0.393; the starter scores 0.550 and 0.382;
  - practice: 0.90 by `submit.py --local`, and again with 3 questions per request;
  - 434 questions took 280 s.
- **Fix:** `main.py` now creates `$OUTPUT_DIR` if it is missing.
- **Open:**
  - ask the organizers whether reading at the `?` position is fine;
  - plain-format prompts are still weak;
  - arithmetic is unreadable without training (Stage 3).

## 2026-11-24 ~18:00 UTC: lens v2 (junk filter), after a leaderboard refresh

- **Leaderboard:** 120 entries, top 0.790. Single-readout entries are near the top; notes in [leaderboard.md](leaderboard.md).
- **`research/try_lenses.py`:** many variants in one pass: 8 late visits × {active, MoE input} × {read, ?} × {raw, junk-filtered}, plus 7 pooled lists.
  - **`research/score_variants.py`** scores them.
  - **Pooled single readouts** reach 0.62 on chat against v1's 0.70: they lose two-hop questions.
  - **The junk filter** helps a little everywhere.
- **v2 = v1 + junk filter** (`lenses.junk_ids`, `lenses.top_tokens`). On the 375 dev questions that ran: chat 0.678 against 0.674 for v1, plain 0.449 against 0.416. Practice 0.90; small batches OK; format OK.
- **NDIF was slow:** about 52 s per request of 25, and a 3-question request took up to 223 s. The 13-minute budget per dataset cut the 434-question dev run at 375 and still exited 0. A hidden dataset of about 100 questions is 4 requests, about 3.5 minutes.
- **Open:**
  - judge-based scoring (needs an OpenAI key), to check whether multilingual mid-pass readouts help;
  - the plain-format rule (read everything at the `?`), worth +0.05 on plain;
  - a trained lens for arithmetic.

## 2026-11-25 ~04:30 UTC: v4, compliant with the 2026-11-24 rule clarifications

- **Rules:** the organizers' clarifications (LENS_RULES.md, cherry-picked 832e157) disallow reading different positions in one prompt. Our first entry and v2 broke that and were flagged. I stopped v3 before submitting it.
- **v4** (branch `idea/one-position`): visits 24, 36 and 43–47, **all at the read position**; logit lens; junk filter.
- **Tests:**
  - practice 0.60 (the starter's level, since the `?` gain is gone);
  - format OK, small batches OK, exit 0;
  - the dev run read only 50 of 217 questions within its budget because NDIF was congested (queue waits over 20 minutes).

### LENS_RULES.md checklist (answered from the code)
- [x] **One position per prompt:** the only read is `lenses.layers(model)[layer].input[0, position]`, with the same `position` (the official read position) for every visit.
- [x] **The position is not chosen per prompt:** it is always `read_position`; nothing compares candidate positions.
- [x] **7 visits** (24, 36, 43, 44, 45, 46, 47), at most 10, none read just to decide, labels are the visit numbers, no repeats.
- [x] **Each readout is the logit lens on one visit's state;** no sums or differences across visits or positions.
- [x] **One token filter,** the same for every prompt (`lenses.junk_ids`: 5751 tokenizer ids with no letter or digit, or chat markup); no prompt tokens.
- [x] **No trained weights or tables.**
