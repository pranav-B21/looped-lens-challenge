#!/usr/bin/env bash
# Check and submit the entry on a branch, the way we do it by hand.
#
#     research/submit_entry.sh [BRANCH]        # default: idea/hybrid
#     MIN_PRACTICE=0.8 research/submit_entry.sh # a stricter practice threshold (default 0.65)
#     DRY_RUN=1 research/submit_entry.sh       # run the checks (steps 1-4) only; submit nothing
#
# 1. Switches to BRANCH and refuses to go on if submission/ has uncommitted changes.
# 2. Rules sanity check (LENS_RULES.md): the entry reads the model in exactly one place.
# 3. Checks the published lens weights download from the Hugging Face Hub without a token.
# 4. Runs `submit.py --local` on the practice set and requires a pass rate of at least MIN_PRACTICE
#    (default 0.65: above the plain logit lens's 0.60, so a broken entry is caught).
# 5. Waits for the team's 2-hour submission slot to open (checks every 5 minutes).
# 6. Submits with `submit.py --wait` and prints the A/B/C scores.
# 7. Appends the result to agents/submissions.md and saves a leaderboard snapshot.
#
# Everything it prints also goes to research/data/submissions/<UTC time>.log. Stopping it while it
# waits for the slot is safe; stopping it after "submitting" only stops the watching (the
# submission carries on; `.venv/bin/python submit.py --status` shows it).
set -euo pipefail

BRANCH="${1:-idea/hybrid}"
MIN_PRACTICE="${MIN_PRACTICE:-0.65}"
DRY_RUN="${DRY_RUN:-0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p research/data/submissions
LOG="research/data/submissions/$STAMP.log"
exec > >(tee -a "$LOG") 2>&1

say() { printf '\n[%s UTC] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
fail() { say "STOPPED: $*"; exit 1; }

say "1/7 Branch $BRANCH"
if [ -n "$(git status --porcelain submission)" ]; then
    fail "submission/ has uncommitted changes on $(git branch --show-current); commit or stash them first"
fi
git checkout -q "$BRANCH" || fail "could not switch to $BRANCH"
[ -z "$(git status --porcelain submission)" ] || fail "submission/ has uncommitted changes on $BRANCH"
COMMIT="$(git rev-parse --short HEAD)"
echo "on $BRANCH at $COMMIT: $(git log -1 --format=%s)"

say "2/7 Rules sanity check (LENS_RULES.md): one read of the model"
READS="$(grep -c -E '\.(input|output|inputs)\[' submission/main.py submission/lenses.py | awk -F: '{s += $2} END {print s}')"
grep -n -E '\.(input|output|inputs)\[' submission/main.py submission/lenses.py
[ "$READS" = "1" ] || fail "expected exactly one read of the model in submission/, found $READS; check LENS_RULES.md by hand"
echo "ok: one read; still answer the full checklist in LENS_RULES.md for a new design"

say "3/7 Published weights download without a token"
env -u HF_TOKEN "$PY" - <<'EOF' || fail "the lens weights could not be fetched from the Hugging Face Hub"
import re, sys
sys.path.insert(0, "submission")
import lenses
src = open("submission/main.py").read()
repo = re.search(r'LENS_REPOSITORY = "([^"]+)"', src)
file = re.search(r'LENS_FILE = "([^"]+)"', src)
if not (repo and file):
    print("no trained weights in this entry; nothing to fetch")
else:
    weights = lenses.fetch(repo.group(1), file.group(1))
    print(f"ok: {repo.group(1)}/{file.group(1)}: {sorted(weights)}")
EOF

say "4/7 Practice run (submit.py --local)"
LOCAL="$("$PY" submit.py --local 2>&1 | tr '\r' '\n' | grep -v -E 'QUEUED|RUNNING|DISPATCHED|RECEIVED|Fetching|^\s*$' || true)"
echo "$LOCAL" | tail -8
RATE="$(echo "$LOCAL" | awk '$1 == "practice" {print $2}' | tail -1)"
[ -n "$RATE" ] || fail "the practice run did not produce a score"
"$PY" -c "import sys; sys.exit(0 if float('$RATE') >= float('$MIN_PRACTICE') else 1)" || fail "practice pass rate $RATE is below $MIN_PRACTICE"
echo "ok: practice pass rate $RATE"

if [ "$DRY_RUN" = "1" ]; then
    say "DRY RUN: every check passed; nothing was submitted"
    exit 0
fi

say "5/7 Waiting for the submission slot"
until "$PY" submit.py --status 2>&1 | grep -q "1 of 1 left"; do
    "$PY" submit.py --status 2>&1 | grep -E "left|queued|running" | head -2 || true
    sleep 300
done
echo "slot open"

say "6/7 Submitting $BRANCH at $COMMIT"
# Stream the output (never capture it): if watching fails, the message must be on screen, and the
# submission carries on regardless, so keep following it by its id.
OUT="research/data/submissions/$STAMP.submit.txt"
set +e
"$PY" submit.py --wait 2>&1 | tr '\r' '\n' | grep --line-buffered -v -E '^\s*$' | tee "$OUT"
set -e
ID="$(grep -o -E 'submitted as .*: [0-9]{8}T[0-9]{6}-[0-9a-f]+' "$OUT" | awk '{print $NF}' | tail -1)"
[ -n "$ID" ] || fail "no submission id in the output above; check with: $PY submit.py --status"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if grep -q -E '^\s*overall' "$OUT"; then
        break
    fi
    if grep -q -E 'failed' "$OUT" && ! grep -q -E '^\s*overall' "$OUT"; then
        if "$PY" submit.py --status 2>&1 | grep "$ID" | grep -q "failed"; then
            fail "submission $ID failed on the leaderboard; see: $PY submit.py --status"
        fi
    fi
    say "watching stopped before the score (attempt $attempt); following $ID again in 60s. The submission is still running."
    sleep 60
    set +e
    "$PY" submit.py --status "$ID" 2>&1 | tr '\r' '\n' | grep --line-buffered -v -E '^\s*$' | tee -a "$OUT"
    set -e
done
grep -q -E '^\s*overall' "$OUT" || fail "no score yet for $ID; follow it with: $PY submit.py --status $ID"
RESULT="$(tail -15 "$OUT")"

say "7/7 Recording the result"
SCORE="$(echo "$RESULT" | awk '$1 == "overall" {print $2}')"
A="$(echo "$RESULT" | awk '$1 == "Dataset" && $2 == "A" {print $3}')"
B="$(echo "$RESULT" | awk '$1 == "Dataset" && $2 == "B" {print $3}')"
C="$(echo "$RESULT" | awk '$1 == "Dataset" && $2 == "C" {print $3}')"
VISITS="$(echo "$RESULT" | grep -o 'Layers read: .*' || true)"
[ -f agents/submissions.md ] || printf '# Submissions made with research/submit_entry.sh\n\n| UTC | Branch | Commit | Overall | A | B | C | Visits |\n|---|---|---|---|---|---|---|---|\n' > agents/submissions.md
printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' "$STAMP" "$BRANCH" "$COMMIT" "${SCORE:-?}" "${A:-?}" "${B:-?}" "${C:-?}" "${VISITS#Layers read: }" >> agents/submissions.md
echo "appended to agents/submissions.md (not committed)"
"$PY" research/leaderboard.py --top 10 --team "Pranav B" 2>&1 | sed -n '1,13p;/^Pranav B/,$p' || true
say "done; full log in $LOG"
