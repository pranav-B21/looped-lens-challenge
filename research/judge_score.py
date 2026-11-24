"""Score readouts on our dev questions with the leaderboard's own judge (final-evals/lens_eval).

    .venv/bin/python research/judge_score.py READOUTS.jsonl [--name NAME] [--model string-match]

Needs ``uv sync --group evaluator`` and OPENAI_API_KEY, in the environment or in the repository's
``.env``. ``--model string-match`` runs the same code with no model, for a free check. Each of our
questions gets the two texts the judge is shown with it: the practice set's rules (the same for
every practice question) and a task line for its family, written like the practice set's. The
score card goes to research/data/judge/<name>/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "final-evals"))

RULES = """- The options are candidate answers to the model's task. An option named by a token counts
  whether or not the line also names other options; list them all."""
SUFFIX = ", to answer with a single word"
TASKS = {
    "sport": "the model was asked which sport a famous athlete plays",
    "instrument": "the model was asked which instrument a famous musician is known for playing",
    "country": "the model was asked which country a famous landmark is in",
    "arith1": "the model was asked to compute a single-step arithmetic problem with small whole numbers",
    "shift1": "the model was asked which day of the week or month comes right after or before a given one",
    "hop_country": "the model was asked for the capital or the language of the country a famous landmark is in; the options are candidate countries, the step it works out on the way",
    "arith2": "the model was asked to compute (a + b) × c with small whole numbers; the options are candidates for the sum a + b, the step it works out on the way",
    "shift2": "the model was asked which day of the week or month comes two after or before a given one; the options are candidates for the one in between, the step it works out on the way",
}


def load_env():
    env = ROOT / ".env"
    if env.exists() and not os.environ.get("OPENAI_API_KEY"):
        for line in env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("readouts", type=Path)
    parser.add_argument("--questions", type=Path, default=ROOT / "research" / "data" / "dev" / "questions.jsonl")
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", default=None, help="judge model (default: the leaderboard's), or string-match")
    args = parser.parse_args()
    load_env()

    import lens_eval
    from lens_eval import config

    submissions = [json.loads(l) for l in open(args.readouts) if l.strip()]
    wanted = {s["id"] for s in submissions}
    questions = {}
    for line in open(args.questions):
        q = json.loads(line)
        if q["id"] in wanted:
            q = dict(q, task_type=TASKS[q["family"]] + SUFFIX, rules=RULES, family=f"{q['family']}-{q['style']}")
            questions[q["id"]] = q
    name = args.name or args.readouts.parent.name
    out_dir = ROOT / "research" / "data" / "judge" / name
    card = lens_eval.evaluate(submissions, name=name, model=args.model or config.JUDGE_MODEL, questions=questions, out_dir=out_dir)
    for family, entry in sorted(card["families"].items()):
        print(f"  {family:<20} {entry['passed']:>4} / {entry['n']:<4} {entry['score']}")
    by_style = {}
    for family, entry in card["families"].items():
        by_style.setdefault(family.rsplit("-", 1)[1], []).append(entry["score"])
    for style, scores in by_style.items():
        print(f"  {style}: mean of families {sum(scores) / len(scores):.3f}")


if __name__ == "__main__":
    main()
