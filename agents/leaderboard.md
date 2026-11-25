# Leaderboard notes

**Source:** `https://ndif-looped-lens-challenge.hf.space/api/leaderboard`, the public JSON that both the leaderboard and <https://waka.mx/loops> (Mihai Cosma, team waka.mx) read. To refresh, run `.venv/bin/python research/leaderboard.py`, which saves a snapshot to `research/data/leaderboard/` and prints a summary. Each entry reports only its scores, its run time and the visits it read. Nothing shows its lens or its read positions.

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
| 14 | **Pranav B (us)** | 0.660 | 0.58 | 0.61 | 0.79 | 24, 34, 35, 36, 40, 41, 43–46 (flagged: two positions) |
| later | Pranav B v2 | 0.712 | 0.66 | 0.767 | 0.71 | 44, 45, 47 at the read position + 46 at `?` (flagged) |
| later | Pranav B v4 (compliant) | 0.666 | 0.68 | 0.567 | 0.75 | 24, 36, 43–47 at the read position |

## What we take from it

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
