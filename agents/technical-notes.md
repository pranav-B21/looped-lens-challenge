# Technical notes: how a lens works, the answer lens, how it is scored (for agents)

The code-level version, with file and function references, for agents working on the entry. The plain-language version for people is `agents/research.md`.

| Section | What it covers |
|---|---|
| [How a lens works](#how-a-lens-works) | The state a lens reads, the logit lens, and why it fails on reasoning questions |
| [The answer lens](#the-answer-lens-what-we-are-building) | What we are changing, why, the training pipeline, and the risks |
| [How a lens is scored](#how-a-lens-is-scored) | The format check, the judge, the vote, and local scoring against the leaderboard |

Measurements behind these claims are in `agents/findings.md`, what other teams score is in `agents/leaderboard.md`, and the rules any lens must follow are in `LENS_RULES.md`.

## How a lens works

### The state a lens reads

- **Inkling has 24 layers, run as 48 visits per token.** Layers 0–5 run once (prelude). Layers 6–17 run three times with the same weights (core passes 1–3, visits 6–41). Layers 18–23 run once (coda, visits 42–47). `lenses.visit_address` maps a visit to its layer and run.
- **The state is the vector entering a visit.** Each token of the prompt carries one vector of a few thousand numbers, and each visit reads it and adds to it. That vector is the model's working memory for that token, and it is not readable as it stands.
- **A lens translates one state into words:**

```
state at (visit v, position p)  ->  lens  ->  a score for every vocabulary token  ->  top 10: " hockey", " ice", ...
```

- **In the code,** the entry reads it as `lenses.layers(model)[layer].input[0, position]` inside `tracer.iter[run]` (`submission/main.py`). The read runs on NDIF, and only the top-10 token ids come back.
- **Inkling has no single residual stream.** Each layer reads its own mix of what earlier layers wrote (the challenge page says so). Lenses that assume one running sum may therefore read worse here than on an ordinary transformer.

### The logit lens (what the starter and our v4 use)

The model already turns its final state into next-token scores with a final norm and an output head (`lm_head`). The logit lens applies that same translator early, with no training:

```
logit lens(h) = lm_head( norm(h) )          # lenses.logit_lens
```

Our v4 adds a fixed filter (`lenses.junk_ids`) that keeps punctuation, whitespace and chat markup out of the top 10. The filter is the same for every prompt.

**Why it partly works:** visits 43–47 already speak roughly the output head's language. A recalled fact (`hockey`, `Australia`) reads cleanly there, which is why recall questions pass.

**Why it fails on reasoning questions:** the head was built for one state (the last) and one job (the next token to write).
1. **Earlier visits use a different encoding.** The head garbles them into `,`, ` and`, ` и`. The model may hold "Saturday" at visit 38 in a direction the head does not map to that word.
2. **The last visits have moved on to formatting.** At the chat read position (`<|message_model|>`) the model is about to think, so the head says `<|content_thinking|>`, plus topic words (`arithmetic`, `week`, `calendar`). It does not say `7` or `Saturday`.
3. **Measured on our questions** (`agents/findings.md`): no untrained readout made arithmetic intermediates readable at any visit or position. The best reached 0.13 on one-step arithmetic. Two-hop country questions, by contrast, read at 0.90–1.00.

**The rules shut off the cheap workarounds** (`LENS_RULES.md`):
- reading some visits at the question's `?` and others at the read position (our v1/v3, flagged);
- summing two positions (our first entry).

Every readout of a prompt must come from one position and one visit. Within those limits the remaining lever is the lens itself, and the leaderboard agrees: top teams gained 0.1–0.2 with their visits unchanged (`agents/leaderboard.md`, snapshot 2026-11-26).

## The answer lens (what we are building)

**Status on 2026-11-27, 01:30 UTC:**
- **Branch:** `idea/answer-lens`.
- **Collection:** running on NDIF, 13/38 training and 5/11 dev batches. It was restarted after a 5-hour hang; each request now has a timeout.
- **Training:** a 240-question smoke test only; the full run waits for all batches.
- **Entry:** not wired into `submission/` yet.
- **Submissions:** none with this lens.

The log is [log-2026-11-26-claude-answer-lens.md](log-2026-11-26-claude-answer-lens.md).

### What changes

A small learned correction goes in front of the model's own head, one per visit:

```
logit lens:   lm_head( norm( h ) )
answer lens:  lm_head( norm( h + (h @ U_v) @ V_v^T + b_v ) )
                             \---- learned per visit v ----/
```

(`research/answer_lens.py`, class `Lens`)

- **The correction is low-rank (rank 64).** It can nudge the state along 64 directions but not rewrite it, so the readout stays a reading of the model's state rather than a separate answer model. It also limits overfitting.
- **It starts at zero.** Untrained, it is exactly the logit lens, so training can only add to it.
- **It is trained toward the intermediate.** Training raises the log-probability of the question's intermediate, in any single-token form (`7`, ` Saturday`, ` France`), at each visit. That is what the judge rewards.
  - A standard tuned lens would instead learn the model's next written token, which at our read position is `<|content_thinking|>`.
  - That is why we call it an *answer* lens.
- **The head stays the model's own.** Readouts are in Inkling's vocabulary and come from one activation, as the judge is told.

### Why it fits the rules

- **One position:** `--position read` or `?` is fixed in advance and used for every visit of every prompt. Collecting both positions only lets us make that fixed choice offline. It is never chosen per prompt by comparing readouts.
- **One visit per readout:** each readout comes from one visit's state, with no sums or differences across visits.
- **No prompt text:** the lens's only input is the activation.
- **The rules list tuned lenses as allowed** ("Logit, tuned, translator and coda lenses on one visit at the chosen position").
- **The training data is our own questions,** never practice or hidden ones, so there is no answer table.

### The pipeline

| Step | File | Runs on | What it does |
|---|---|---|---|
| 1. Questions | `research/make_train_questions.py` | Mac | Writes 1,518 labelled training questions, kept separate from the 434 dev questions. Families: one- and two-step arithmetic, one- and two-step day/month shifts, two-hop country, country, sport, instrument; chat format |
| 2. States | `research/collect_states.py` | NDIF | One Inkling forward pass per question. Saves the state entering visits 30–47 at the read position and at the last `?` (bf16), with a 600 s timeout per request and retries. The first 520 training and 200 dev questions went 40 per request (`states/train`, `states/dev`). After results of about 6.6 MB stopped coming back, the rest went 10 per request (`states/train-rest`, `states/dev-rest`). All 1,518 and 434 were collected |
| 3. Head | `research/fetch_unembed.py` | Mac, from the HF Hub | Downloads Inkling's unembedding and final-norm weight (926 MB), so training can run locally. The local head reproduces NDIF's logit-lens top 10 exactly |
| 4. Train | `research/answer_lens.py` | Mac (MPS) | Fits `U, V, b` per visit. Picks up to 10 visits on held-out *training* prompts, then reports that design on dev against the logit lens at the same visits |
| 5. Publish | the HF Hub | | Weights as safetensors in bf16 (about 10 MB) |
| 6. Entry | `submission/main.py`, `lenses.py` | NDIF | Fetches the weights once per request with `lenses.fetch`, moves them to the activation's device and dtype, and applies `lm_head(norm(h + correction))` at each chosen visit, all at one position |

### Which stage is trained

```
hidden question → Inkling on NDIF (frozen)
  → h = layers[L].input[0, position]            one position per prompt, the same for every visit
  → s = h + (h @ U_v) @ V_v.T + b_v             ← trained: U_v, V_v [2304, 64], b_v [2304]
  → logits = lm_head(norm(s))                   frozen; on NDIF the entry calls model.lm_head
                                                and model.model.language_model.norm
  → index_fill(junk_ids, -inf) → topk(10)       lenses.top_tokens; junk_ids fixed per tokenizer
  → readout labelled v → judge → plurality vote
```

- **Parameters:** 2 × 2304 × 64 + 2304, about 297k per visit, about 5.3M for all 18 visits. Inkling has 26.4B.
- **Initialization:** `U = 0`, `V` near 0, `b = 0`, so before training it equals the starter's logit lens.
- **Loss:** `logsumexp(logits) − logsumexp(logits[target forms])`, where the target forms are every single-token casing and spacing of the intermediate (`probe_visits.variants`).
- **Training:** AdamW (lr 3e-3, weight decay 1e-2), 40 epochs, batch 128, on MPS. The local head is `unembed[:200058] @ (rms(s) * norm_weight)`, from `research/data/unembed.safetensors`.

### Storage

| Where | What |
|---|---|
| `research/data/answer_lens-{read,qmark}.safetensors` (local, git-ignored) | `v{visit}.U`, `v{visit}.V`, `v{visit}.b` for visits 30–47, in bf16, about 10 MB. Metadata: `visits` (the chosen design, JSON), `position`, `rank` |
| A public HF model repo (to do; the user's `looped-lense-2` token can write) | only the chosen visits, about 6 MB. No Inkling weights, no question files |
| NDIF, per request | `lenses.fetch(repo, file)` before the first trace; tensors arrive on the CPU, so move each to `h.device` and `h.dtype` |

### Why it should raise the score

- **What the plain head misses.** It assumes mid-network states are already in the final state's basis. Here each pass restarts from the prelude's state and the residual is a block mix, so at visits 37–42 the raw head hit the intermediate **0%** of the time in the smoke test.
- **Smoke test** (240 train and 40 dev chat questions, 3 families, 5 epochs), single-readout hit rate on dev, trained vs. raw head:
  - v36: 0.77 vs 0.62
  - v40: 0.43 vs 0.00
  - v44: 0.88 vs 0.80
  - v46: 0.85 vs 0.70
- **More readouts voting for the intermediate** raises the plurality margin, at exactly the visits top teams use (35–41 and 43–47).
- **The gate:** the dev design must beat the logit lens at the same visits, per family, before publishing.

Inkling is called in steps 2 and 6 only, both through nnsight with `model.session(remote=True)`. Downloading the open weights locally (step 3) is fine. The rule that code cannot read the weights applies to code running on NDIF.

### What we expect, and the risks

- **Expected gains:** reasoning questions where the raw head reads nothing, mainly arithmetic and day/month shifts, at visits 35–41, where top teams converged and the raw head struggles.
- **Overfitting to our templates.** Our families are guesses at the hidden ones. A lens can learn our wording and score well on dev without helping datasets A and C, which now decide the ranking.
  - Report dev results per family, and only submit if the lens clearly beats the logit lens across families.
  - Hold out by entity or prompt, not by row.
- **A wrong vote costs as much as a right one gains.** On two-hop questions, training toward the intermediate must not push late visits toward the final answer, which is usually a distractor. Check per-visit vote tables for wrong-option votes.
- **Run time.** Fetching weights and the extra matrix products add time, and all three datasets together must finish well under 2,700 s.
- **Device and dtype.** `lenses.fetch` returns weights on the CPU, so move each one to the activation's device and dtype on NDIF.

## How a lens is scored

How the leaderboard turns `readouts.jsonl` into `pass_rate`, read from the scoring code on
2026-11-26, and what that means for the lens we build. The code is the authority. The organizers
may change the judge, its prompts or the scoring while the competition runs, so recheck these
files when a score moves without our entry changing:

| Part | File |
|---|---|
| Frame: reads each dataset, checks it, calls the judge, builds the result | `evaluator/evaluate.py` |
| Checking each line of `readouts.jsonl` | `final-evals/lens_eval/data.py` (`clean_submission`) |
| The judge's prompts | `final-evals/judge_prompts/system_mc.txt`, `user_mc.txt` |
| Building the prompt (what the judge sees) | `final-evals/lens_eval/prompts.py` |
| Judge model and limits | `final-evals/lens_eval/config.py` (`JUDGE_MODEL = "gpt-6-luna"`, 10 readouts, 10 tokens, 64 characters) |
| Votes and pass/fail | `final-evals/lens_eval/scoring.py` |
| The exact-match judge used locally | `final-evals/lens_eval/judge.py` (`StringMatchJudge`) |

### The pipeline in short

1. **The format check** (`data.py`) runs on every line.
2. **The judge**, an LLM, is shown one question's readout and five options. It lists which tokens name which options.
3. **Code checks those citations**, gives each readout one vote, and passes the question if the intermediate has strictly the most votes.
4. **Scores:** a dataset's score is the share of its questions that pass. `pass_rate` is the mean of datasets A, B and C, and it decides the ranking.

### Stage 1: The format check

| Problem | Cost |
|---|---|
| No line for a question | That question fails (`missing`) |
| Empty readouts, or more than 10 | That question fails (`invalid`) |
| A layer label twice in one question | That question fails |
| A token over 64 characters | That question fails |
| More than 10 tokens in a readout | Tokens after the tenth are dropped |
| Two lines for one question, or a line that is not JSON | **The whole dataset is not scored** |
| No `readouts.jsonl`, or a non-zero exit | The whole submission fails |

The code accepts any integer or string as a layer label. **The rules require visit numbers 0–47**, labelled with the visit read (LENS_RULES.md).

### Stage 2: The judge

Each question is one call. The prompt is built in `prompts.py`:

```
TASK TYPE: the model was asked which sport a famous athlete plays ...   <- the kind of task only

READOUT (top 10 tokens per layer, best first):
[α] hockey | ice | sport | skating | ...
[β] NHL | hockey | Hockey | ...
[γ] , | and | the | ...

OPTIONS:
A. tennis
B. hockey
C. golf
D. sumo
E. karate
```

**What the judge never sees:**
- the question's text, or which option is right;
- our layer numbers. The lines are relabelled α, β, γ…, so a label carries no information;
- other questions. Each is judged alone.

**What it does:** for every line, it lists every option that some token on that line *names*, with the token. That is its only job; it does not judge the lens.

**A token names an option if it is:**
- the option itself, in any case, inflection or spelling;
- an established alias, or a faithful translation in any language or script (`星期六` names Saturday);
- a word formed from the option's own word (`Turkish` names Turkey, `golfer` names golf, `guitarist` names guitar);
- a fragment of at least 4 characters that can only start that option (`gymn` for gymnastics);
- for a number option, a single token that writes the whole number.

**These do not name an option:**
- clues: a related place, person or thing (`Istanbul` and `Ottoman` for Turkey, `NHL` for hockey);
- a category (`sport` for boxing), a symbol, a definition, or a calculation (`9-2` for 7);
- a word made by joining tokens. Each token is judged on its own, so `Sat` + `urday` names nothing.

**Each dataset brings its own extra rules** (the `{rules}` slot) and a task-type line. The practice questions show theirs; the hidden datasets' are not published.

### Stage 3: Votes (`scoring.py`)

1. **Citations are checked** (`verified_mapping`). A citation whose token is not actually on that line is dropped. Each remaining one records the token's rank on its line, 1 being first.
2. **One vote per readout** (`layer_votes`). Each readout votes for the option named by its highest-ranked naming token. The tokens below that one change nothing. A readout that names nothing does not vote. If one top token names two options equally, that readout casts no vote.
3. **Pass or fail** (`plurality`). The question passes only if the intermediate has **strictly more** votes than every other single option. A tie fails.

### A worked example

"What is the day after Friday?" The intermediate is Saturday.

| Line (visit) | Tokens, best first | Highest naming token | Vote |
|---|---|---|---|
| α (36) | ` week`, ` Saturday`, ` Sunday` | `Saturday` (rank 2) | Saturday |
| β (43) | ` Sunday`, ` Saturday` | `Sunday` (rank 1) | **Sunday** |
| γ (44) | ` 星期六`, ` days` | `星期六` (a translation) | Saturday |
| δ (47) | `,`, ` the` | none | none |

Saturday has 2 votes and Sunday 1, so the question **passes**. If δ had led with Sunday, it would be 2–2, a tie, and it would **fail**.

### Local scoring and the leaderboard

| | `submit.py --local` | Leaderboard |
|---|---|---|
| Questions | 10 practice questions, all one kind | 290 hidden questions in datasets A, B and C |
| Judge | Exact match, ignoring case, spaces and punctuation (`StringMatchJudge`), unless `OPENAI_API_KEY` is set | Always the LLM judge |
| Translations and inflections | Not counted | Counted |
| Noise | None | The judge changes its verdict on 1–2 questions in a few hundred between runs |

**What this means for measuring:**
- **A local score can be off in either direction.** Exact match undercounts translations (`星期六`) and inflections. To use the real judge locally, run `uv sync --group evaluator` and set `OPENAI_API_KEY`. `research/judge_score.py` also sends the judge prompts to other providers.
- **Practice moves in steps of 0.10,** one question each. We have seen a lens score 0.90 on practice and 0.660 on the leaderboard, so measure on our own dev set (`research/data/dev/`).
- **On the leaderboard, treat differences under about 0.01 as noise.** Identical resubmissions by other teams move by 0.004–0.01.

### What this means for the lens

- **The only thing that counts is which option a line names first.** Each line comes down to its highest-ranked token that names an option. Ranking the intermediate above every distractor on a line matters more than anything else in its top 10.
- **Junk tokens cost nothing but waste places.** `,` or `\n\n` never name an option. They only push naming tokens out of the top 10, which is why the junk filter (`lenses.junk_ids`) helps. The filter is the same for every prompt, so the rules allow it.
- **A readout that votes wrong cancels a right one.** More readouts help only when they agree. Dropping visits that often lead with a distractor can raise the score, which is why strong entries read 6–8 visits.
- **Two-step questions are the trap.** Where the intermediate is a step the model never writes, the model's final answer is usually one of the distractors. Late visits that have moved on to that answer vote against us.
- **Translations count on the leaderboard,** so a visit whose tokens are non-English can score there and not locally.
- **The judge never sees layer numbers or the question.** Nothing in a label or a readout can help except tokens that name the intermediate.

## Related

- `LENS_RULES.md`: what a lens may and may not do. Scoring never overrides it.
- `agents/leaderboard.md`: how other teams score per dataset.
- `agents/findings.md`: what our measurements say about where the intermediates are readable.
