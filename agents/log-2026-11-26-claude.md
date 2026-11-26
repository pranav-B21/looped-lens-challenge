# 2026-11-26, Claude (session "looped-lense research")

## Done

- **Snapshot of the leaderboard at 19:10 UTC** (751 entries, 103 teams), from the public API that waka.mx/loops charts. The patterns are in `agents/leaderboard.md`, under "Snapshot 2026-11-26".
- **`research/leaderboard.py` options:**
  - `--history N` prints the top N teams' submissions in time order;
  - `--team NAME` prints one team's;
  - flagged entries are marked.
- **`CLAUDE.md` now lists `agents/leaderboard.md`** as a resource to read before choosing what to build.

## Found

- **Top score is 0.990** (team-macromackie); the top 10 are at 0.906 and above. We are 75th of 102, with 0.712 (flagged) and 0.666 (v4, compliant).
- **Most top-team gains came with the visits unchanged,** so the lens is what improved (see the snapshot, point 1). Visits converged on 35–41 plus 43–47.
- **B is saturated across the top 10.** A and C now decide the ranking.

## Next

- **Tuned lens at the converged visits,** all at one position, measured on our dev set (branch `idea/tuned-lens`).
