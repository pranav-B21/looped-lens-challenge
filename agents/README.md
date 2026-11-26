# Agent work logs

Notes that agents (Claude Code, Cursor, …) keep while working on this entry, so that the next session, or another agent, can pick up where the last one stopped. These notes are not part of the entry: `submission/` is what gets uploaded.

| File | What it holds |
|---|---|
| [`leaderboard.md`](leaderboard.md) | What the public leaderboard (and waka.mx/loops) shows: scores, visits other teams read, how the top teams improved over time, and what we take from it. Newest snapshot first. Refresh with `.venv/bin/python research/leaderboard.py --top 15 --history 10`. |
| [`findings.md`](findings.md) | What we know about Inkling 2 Mini and about lenses on it, with the evidence for each claim. |
| `log-YYYY-MM-DD-<agent>.md` | One log per agent per day: what was done, what was found, what is open. |

## Conventions

- **Add a log entry before ending a session:** what changed (commits, branches), what was measured (with numbers), and what is next.
- **Write absolute dates and UTC times.**
- **Record a claim with its evidence:** a command, a file under `research/data/`, or a commit.
- **Never paste API keys, tokens or hidden-question text here.**
- **One agent at a time in this checkout.** Two agents switching branches in one working tree overwrite each other's files. A second agent works in its own `git worktree`.
