# Leaderboard notes

**Source:** `https://ndif-looped-lens-challenge.hf.space/api/leaderboard`, the public JSON that both the leaderboard and <https://waka.mx/loops> (Mihai Cosma, team waka.mx) read. To refresh, run `.venv/bin/python research/leaderboard.py`, which saves a snapshot to `research/data/leaderboard/` and prints a summary. Each entry reports only its scores, its run time and the visits it read. Nothing shows its lens or its read positions.

## How to use these notes

- **When to read them:** before choosing visits, picking the next lens idea, or deciding whether to submit. Read the newest snapshot first; the older ones are history.
- **Refreshing the data:** `.venv/bin/python research/leaderboard.py --top 15 --history 10`.
  - `--history N` prints every submission of the top N teams in time order: score, datasets, run time, flag, visits.
  - `--team "<name>"` prints one team's history; `--team "Pranav B"` is ours.
- **What the feed shows and doesn't:** it holds scores, run time, the union of visits read, and a `flagged` field. It never shows a lens, its positions or its code, so read the patterns as hints. Every idea taken from here still has to pass `LENS_RULES.md`.

## Snapshot 2026-11-26 19:10 UTC (751 entries, 103 teams)

Scores roughly doubled their margin over the starter in two days. The median entry per day went 0.665 (11-23), 0.724 (11-24), 0.773 (11-25), 0.826 (11-26). The official starter baseline is still listed at 0.662, from 11-23; it was not rerun.

| # | Team | Score | A | B | C | Visits (all questions combined) | Subs |
|---|---|---|---|---|---|---|---|
| 1 | team-macromackie | **0.990** | 1.00 | 0.99 | 0.98 | 37–41, 43–47 | 15 |
| 2 | ekhadley | 0.983 | 1.00 | 0.98 | 0.97 | 12, 24, 36–40, 44–47 (11 combined) | 6 |
| 3 | mingly | 0.975 | 0.99 | 0.96 | 0.98 | 34, 36, 38, 40, 42–47 | 3 |
| 4 | Arnav Raj | 0.970 | 0.98 | 1.00 | 0.93 | 5, 17, 22, 29, 39, 42–46 | 17 |
| 5 | Aadi Palnitkar | 0.940 | 0.91 | 1.00 | 0.91 | 9, 26, 37, 38, 40, 43–47 | 17 |
| 6 | Daniil Ognev | 0.923 | 0.90 | 0.98 | 0.89 | 17 combined (17, 21–23, 26, 29, 30, 33–35, 40–46) | 15 |
| 7 | NRYOUNG | 0.920 | 0.86 | 0.99 | 0.91 | 24, 28, 35, 36, 39, 40, 42, 44–46 | 9 |
| 8 | hluo | 0.914 | 0.93 | 0.93 | 0.88 | 16, 24, 34, 36, 40, 43–47 | 17 |
| 9 | hosh | 0.906 | 0.89 | 0.99 | 0.84 | 35, 36, 42–47 | 9 |
| 10 | Looped Readers | 0.906 | 0.89 | 0.99 | 0.84 | 24, 34–36, 42–47 | 9 |
| 75 | **Pranav B (us)** | 0.712 (flagged) / 0.666 (v4, compliant) | 0.68 | 0.57 | 0.75 | v4: 24, 36, 43–47 | 3 |

### How the top 10 build their entries

1. **Most of the gain comes from the lens, not the visits.** Several teams kept one visit set and still climbed:
   - **Aadi Palnitkar** kept one set from 11-25 04:41 and went 0.809 → 0.872 → 0.885 → 0.920 → 0.940.
   - **team-macromackie** read the same visits for its last 7 submissions and went 0.949 → 0.963 → 0.990.
   - **hluo** kept its set for 13 submissions and went 0.730 → 0.914.

   The lens must be what changed: probably retrained tuned or translator lenses, or ensembles at the same visit. Entries that read exactly the starter's visits now reach 0.82, against 0.66 for the plain logit lens.
2. **The visits have converged:**
   - the coda, 42/43–47, which almost every entry reads;
   - late in the third core pass, **37–41**, plus 34–36;
   - at most one to three early visits as extras: Arnav 5, 17, 22, 29; Aadi 9, 26; ekhadley 12, 24.

   team-macromackie's jump from 0.886 to 0.956 came when it moved from 16/24/34 to 37–41. Over all 751 entries, reading 37–42 or 43–47 goes with +3 to +10 points, and reading 12 with −4.
3. **The coda alone collapses C.** ekhadley read only 42–47 twice, and C fell to 0.17. Adding back 36–40 brought C up to 0.96.
4. **Some teams vary visits by prompt.** Daniil Ognev and Arnav list up to 17 visits combined, while each prompt still reads at most 10. The clarified rules allow this.
5. **The top teams resubmit on a fixed 2-hour cycle, around the clock.** Aadi submits at :41, hluo near :51/:00, macromackie at :35. That looks like an automated train → submit loop.
   - It partly tunes against the hidden sets. Phase 2 brings new questions, and the final ranking mixes phases, so treat leaderboard-tuned gains as fragile.
   - Same-entry resubmissions move by 0.004–0.01 (0.963 → 0.959), so treat differences under about 0.01 as noise.
6. **B is solved and A/C decide the ranking.** B is 0.93–1.00 across the top 10. The leaders got there by raising **A** (macromackie 0.57 → 1.00, Arnav 0.38 → 0.98) and then C. They now hold A ≥ 0.98 and C ≥ 0.97.
7. **Their lenses are heavier.** Top runs take 1,000–2,400 s, against the 2,700 s limit for all three datasets (Arnav 2,379 s). That points to coda lenses, which run the model's own later layers on the state, or several lenses per visit. Leave headroom: a timeout fails the whole submission.
8. **The flags hit 21 entries, one per team, all from 11-24 and 11-25.** They include Srujun, CB26, team-macromackie's first, and our 0.712. Flagged teams resubmitted compliant entries and kept climbing.

### What it means for our lens

- **Train the lens.** Visit hunting is mostly done, and our gap of about 0.3 is the lens. Start with a tuned lens per visit, fitted locally on activations collected on NDIF from our own prompts (`research/`, branch `idea/tuned-lens`). Publish it to the HF Hub and fetch it with `lenses.fetch`.
- **Start from the converged visits:** 35/36, 37–41 and 43–47. Then test whether one or two early visits help.
- **Measure on our own dev set, not on practice.** Before each submission, also run the `LENS_RULES.md` checklist and keep the run well under 2,700 s.

## Snapshot 2026-11-24 17:32 UTC (120 entries, 38 teams)

| # | Team | Score | A | B | C | Visits ("Layers read") |
|---|---|---|---|---|---|---|
| 1 | Daria Kryvosheieva | 0.790 | 0.73 | 0.79 | 0.85 | [157]: one readout, labelled 157 |
| 2 | Tokenless | 0.786 | 0.73 | 0.79 | 0.84 | 24, 27, 36, 39, 40, 43–47 |
| 3 | CB26 | 0.777 | 0.70 | 0.80 | 0.83 | 12 visits (they vary by question) |
| 4 | Standard Logits | 0.774 | 0.67 | 0.81 | 0.84 | 16, 24, 34, 42–47 |
| 5 | Srujun | 0.763 | 0.64 | 0.79 | 0.86 | [12]: one readout |
| 11–12 | defiance, lukxst | 0.748, 0.744 | | | | [11], [14]: one readout each |
| | Starter's visits (15 entries) | 0.654–0.728 | | | | |

**New since the first snapshot:**
- **Single-readout entries now place high** (0.74–0.79). Presumably they pool many visits or positions into one ranked list. On our dev set, pooled lists score 0.62 against 0.70 for v1: they lose two-hop questions, where the pooled list puts the final answer (a distractor) first. Their datasets evidently reward pooling more than ours does.
  - Caution: a label like 157 is not a layer, and the judge is told each token comes from one activation. Pooling across visits may be against the spirit of the rules; ask the organizers before copying it.
- **"Layers read" is the union over questions,** so entries such as CB26's list 12 or 13 visits while giving each question at most 10 readouts.
- **The best scores per dataset are now** A 0.75, B 0.83, C 0.86. Dataset A is still the hardest.
- **Mid-pass visits are common among the top 10:** 24–29 and 36–41 come up again and again. On our exact-match dev set they don't help. One explanation: their tokens are often non-English (`星期`, `요일`), which the leaderboard's judge counts as translations and our exact-match scorer doesn't. Scoring with the judge (an OpenAI key) would settle it.

## Snapshot 2026-11-24 02:23 UTC (24 entries, 16 teams)

| # | Team | Score | A | B | C | Visits |
|---|---|---|---|---|---|---|
| 1 | janinja82 | 0.754 | 0.73 | 0.81 | 0.72 | 9, 20, 22, 26, 37, 38, 43, 44, 45, 46 |
| 2 | Neutrox | 0.727 | 0.65 | 0.70 | 0.83 | 11, 17, 29, 35, 41, 42, 43, 44, 45, 46 |
| 3 | Numericus Group | 0.724 | 0.67 | 0.72 | 0.78 | the starter's |
| 4 | Srujun | 0.723 | 0.67 | 0.69 | 0.81 | 11, 23, 29, 42, 44, 45, 46, 47 |
| 5 | Phoenix | 0.722 | 0.63 | 0.74 | 0.79 | 22, 26, 28, 35, 36, 39, 43, 44, 45, 46 |
| 13 | Starter logit lens (official) | 0.662 | 0.66 | 0.56 | 0.77 | the starter's: 12, 24, 34, 35, 36, 43–47 |
| 14 | **Pranav B (us)** | 0.660 | 0.58 | 0.61 | 0.79 | 24, 34, 35, 36, 40, 41, 43–46 |

## What we took from it on 2026-11-24 (superseded where the 11-26 snapshot disagrees)

- **Our first submission tied the starter.** It was the Cursor branch's dual-position lens (`cursor/optimized-inkling-lens-v1-*`), which scored 0.90 on the 10 practice questions and 0.660 on the hidden sets. Practice gains do not carry over, so measure on our own dev set (`research/`).
- **Run-to-run noise is small, so the lens is what matters.** Five entries that look like the unchanged starter (same visits) score 0.654–0.668, about ±0.007. Numericus Group reads the same visits and scores 0.724, so its *lens* adds about 6 points. Visit choice alone does not explain the top scores.
- **The starter's profile is A 0.66–0.68, B 0.53–0.56, C 0.75–0.78.**
- **Dataset B is where entries differ** (0.54–0.81). Dataset A ranges 0.58–0.73. Dataset C is about 0.78 for nearly everyone. Each dataset holds one family of questions (`evaluator/evaluate.py` requires it).
- **Dataset A breaks easily.** These entries dropped A to 0.39–0.58 while lifting B or C:
  - our dual-position entry (A 0.58);
  - waka.mx's third entry (A 0.46, C 0.84);
  - JB's first (0.46) and janinja's first (0.41);
  - waterloo, which reads a single visit, 7 (A 0.39).

  Our guess is that A's intermediate is read best at the official position, and moving or blending positions costs A.
- **Dataset B gains the most.** The best entries move B from 0.56 to 0.70–0.81 (janinja 0.81, Phoenix 0.74). It is probably the family whose intermediate the starter misses, such as two-hop or computed questions.
- **Fewer readouts can work.** Srujun scores 0.723 with 8, Neutrox 0.717 with 7, and interpretune went from 0.654 to 0.680 with 6. A readout that votes for a wrong option raises the bar, so dropping weak visits helps.
- **Visits 43–46 are close to universal**, and entries reading them score higher. The top entry adds early and mid-pass visits (9, 20, 22, 26, 37, 38). This is a hint only: it doesn't say which lens or position those entries use.
- **Run times are 205–1675 s** for the three datasets together, against a 2700 s limit.

## Check again

Refresh after each of our submissions and roughly daily. Watch for:
- new top entries and the visits they read;
- the starter's score moving, which would mean the organizers changed the questions or the judge.
