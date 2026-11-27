@AGENTS.md

# Competition rules for this entry

The organizers clarified the rules on 2026-11-24 and flagged one of our entries for reading two
positions in one prompt. Follow them strictly; never trade a rule for score.

@LENS_RULES.md

# Resources to read before deciding what to build

Our notes live in `agents/`. Read the relevant ones at the start of a session and keep them current.

- **`agents/findings.md`**: what we have measured about Inkling 2 Mini and our lenses, each claim with its evidence. Read it before proposing an idea, so as not to repeat a measured dead end. Add new measurements here.
- **`agents/leaderboard.md`**: what the public leaderboard, and waka.mx/loops, which charts it, show about other teams. It covers their scores per dataset, the visits they read, and how the top teams improved over time, with what that means for our lens. Read its newest snapshot before choosing visits, picking the next idea or deciding to submit. Refresh it with `.venv/bin/python research/leaderboard.py --top 15 --history 10` after each of our submissions and about daily, and add a dated snapshot.
- **`agents/research.md`**: the user's plain-language guide to the project. Keep it free of code references. When the user asks a conceptual question, answer it and add the question and a simple answer to its section 9. When the plan changes, update its sections 4 and 6 in plain words.
- **`agents/technical-notes.md`**: how a lens works at code level, the answer-lens pipeline, and how a lens is scored (the format check, the judge, the vote, local scoring against the leaderboard). Read it before changing the lens or what a readout contains, and update its status line when the plan moves.
- **`agents/log-*.md`**: what each session did and what is open. Add an entry before ending a session.

Ideas taken from other teams' results are hints, never a reason to bend `LENS_RULES.md`.
