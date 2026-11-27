# Research notes: what we are doing, in plain words

A guide to the Looped Lens Challenge and our entry, written to be understood rather than to point
at code. Questions get answered at the bottom as they come up. (The code-level version, for
agents, is `agents/technical-notes.md`.)

---

## 1. The challenge in one paragraph

Inkling 2 Mini is a language model. When it reads a question, it works things out *inside*
itself before it writes anything. Some of those things it never writes at all. For "what is the
capital of the country where the Eiffel Tower is?", it has to think "France" on the way to
"Paris", but it only writes "Paris". The challenge is to build a **lens**: a tool that looks
inside the model while it is thinking and says, in words, what it is thinking about. We pass a
question when our lens surfaces the thing the model worked out, called the **intermediate**.

---

## 2. How the model thinks: the forward pass

**Words become numbers.** The question is split into **tokens**, pieces of words. Each token
is turned into a long list of a few thousand numbers, called its **state**. Think of the state
as that token's notebook page: everything the model knows about this spot in the sentence is
written on it, in a code only the model reads.

**Layers edit the notebook.** The model is a stack of **layers**. Each layer reads every token's
page, lets the tokens look at each other ("Sidney Crosby" can inform the last token), adds what
it figured out, and passes the pages on. One trip through all the layers, from the question to
a guess at the next word, is a **forward pass**. Our lens needs only one forward pass per
question, and never has the model write anything.

**Inkling is looped.** Most models run each layer once. Inkling has 24 layers, but runs the
middle 12 **three times in a row**, re-using the same layers. A token therefore goes through 48
steps, which we call **visits**:

```
visits 0–5    prelude     (layers 0–5, once)
visits 6–17   core pass 1 (layers 6–17)
visits 18–29  core pass 2 (the same layers again)
visits 30–41  core pass 3 (and again)
visits 42–47  coda        (layers 18–23, once)
```

The idea is that looping gives the model more thinking time without more weights. It also means
that "layer 12" happens three times. We always say which visit we mean.

**At the very end,** the model has a built-in translator, the **output head**. It turns the
last notebook page into a score for every word in its vocabulary, and the top-scoring word is
what it writes next.

**We never run the model ourselves.** It runs on NDIF's GPUs. Our code describes what to read,
NDIF runs the model and our reading code next to it, and only the small result (top-10 word ids)
comes back.

---

## 3. What a lens is

A lens takes **one notebook page** (one token's state, at one visit) and translates it into
words: a ranked list of the 10 words the page seems to be about.

```
the state at visit 40, last token   ->   lens   ->   " Saturday", " weekend", " days", ...
```

We may read up to **10 visits per question**, so each question gets up to 10 of these ranked
lists, which are called **readouts**.

### The simplest lens: the logit lens

Borrow the model's own output head, the end-of-the-line translator, and use it on a page from
the *middle* of the thinking instead of the end. It is like asking someone to read a half-written
draft aloud.

- **What it is good at:** near the end of the network (visits 43–47) the pages are written in
  nearly the same code the translator expects. Simple recalled facts come out clean: "What sport
  does Sidney Crosby play?" gives ` hockey`.
- **Where it fails:** the translator was built for one page (the last) and one job (choosing the
  next word to write). That causes two problems:
  1. **Middle pages are in a different code.** The translator mangles them into junk like `,`,
     ` and`, ` и`. The model may be holding "Saturday" at visit 38, but not in a form the
     translator reads as "Saturday".
  2. **The last pages are about formatting, not the answer.** On chat prompts the model is about
     to start *thinking out loud*. Its next word is a special "start thinking" marker, so the
     translator reports that, plus topic words (` arithmetic`, ` week`), but not ` 7` or
     ` Saturday`.

That is why the starter, and our current entry, pass the fact-recall questions and fail the
reasoning ones (arithmetic, "the day after Friday").

---

## 4. What we are changing: the answer lens

**The idea:** keep the model's own translator, but put a small **adjustment** in front of it,
one per visit, that we **train** to make the translator read the page correctly.

```
logit lens:    translator( page )
answer lens:   translator( page + small learned adjustment )
```

An analogy: the translator is a reader who only knows the final dialect. The adjustment is a
short phrasebook for each earlier dialect. It does not change what the page says; it only helps
the same reader understand it.

**How we train the adjustment:**
1. **Write practice questions with known answers.** We wrote about 1,500 of our own (arithmetic,
   days and months, "which country…", sports, instruments). We know the intermediate for each.
2. **Run the model on them** (on NDIF) and save the pages at visits 30–47. We don't save
   whole forward passes, only those pages.
3. **Learn the adjustment** on a laptop. For each visit, tune it so that the correct
   intermediate climbs to the top of the readout.
4. **Check it on questions it never trained on** (our separate 434-question dev set), and
   compare it with the plain logit lens.
5. **If it is clearly better, put it in the entry.** The adjustment's numbers (about 10 MB) go on
   Hugging Face, and the entry downloads them on NDIF and applies them while reading.

**Why it is called an *answer* lens:** a standard trained lens, a "tuned lens", learns to predict
the word the model will *write next*. Here that word is the "start thinking" marker, which is
useless to us. Ours is trained to surface the *intermediate*, the thing the judge checks for.

**Why the adjustment is kept small:**
- **It starts at zero.** Before training it is exactly the logit lens, so training can only
  help.
- **It is low-rank.** It can only nudge the page in a limited number of directions, not rewrite
  it. That keeps it a *reading* of what the model holds, rather than a separate answer machine,
  and it makes memorising our practice questions harder.

**Why this route, not another:** the organizers clarified the rules (section 7). The shortcuts
we tried first (reading at two places in the question, mixing positions) are not allowed. A
trained lens on one visit at one position is explicitly allowed. The leaderboard also shows that
teams improved mostly by improving their lens, not by picking different visits (section 6).

**What could go wrong:**
- **It might learn our questions, not the model.** If our practice questions don't look like
  the hidden ones, the lens can score well on ours and not help on the leaderboard. So we check
  each kind of question separately before submitting.
- **A wrong vote costs as much as a right one gains** (section 5). On "two-step" questions the
  lens must not start showing the model's *final* answer, which is usually one of the wrong
  options.

---

## 5. How a lens is scored

**The setup:** each hidden question has 5 options. One is the intermediate; four are
distractors, often including the model's final answer.

**The judge** is an AI model. It sees only our readouts and the 5 options. It never sees the
question, which option is right, or which visit each readout came from. Its only job is to say,
for each readout, which options its words *name*.

**What counts as naming an option:**
- **Counts:** the word itself in any form or language. `hockey`, `Hockey`, `golfer` → golf,
  `Turkish` → Turkey, `星期六` → Saturday.
- **Doesn't count:** clues or related things. `Istanbul` doesn't name Turkey, `NHL` doesn't
  name hockey, `sport` doesn't name boxing.
- **Each word is judged on its own.** `Sat` + `urday` does not make Saturday.
- **A number** must be written whole in one token.

**The vote:**
1. **Each readout gets one vote,** for the option named by its *highest-ranked* naming word. The
   other words don't matter.
2. **A readout that names nothing doesn't vote.**
3. **The question passes only if the right option has strictly more votes than any other.** A
   tie fails.
4. **Scores:** the score per dataset is the share of questions passed. The leaderboard score is
   the average of datasets A, B and C.

**An example:** "What is the day after Friday?" The right option is Saturday.

| Readout | Words, best first | Votes for |
|---|---|---|
| 1 | week, **Saturday**, Sunday | Saturday |
| 2 | **Sunday**, Saturday | Sunday |
| 3 | **星期六**, days | Saturday |
| 4 | `,`, the | (no vote) |

Saturday has 2 votes and Sunday 1, so it **passes**. One more readout leading with Sunday would
make it a tie, and it would **fail**.

**What follows from this:**
- **Order matters more than anything.** The right word needs to come before any wrong option on
  a readout.
- **More readouts are not always better.** A readout that votes wrong cancels a right one, so
  strong entries often read only 6–8 visits.
- **Our local test differs from the leaderboard.** It uses exact word matching, so it misses
  translations like `星期六`, and it has only 10 questions. One question moves it by 0.10. We
  trust our own 434-question dev set more.

---

## 6. What the leaderboard tells us

As of 2026-11-26 (details in `agents/leaderboard.md`):

- **The starter scores 0.66. The best teams now score 0.90–0.99. We are at 0.666** (rank 75 of
  102).
- **Top teams improved mostly by improving the lens, not by choosing visits.** Several kept the
  exact same visits and still climbed 0.1–0.2, so the translator is what got better.
- **Everyone has converged on the same visits:** late in the third core pass (35–41) plus the
  coda (43–47).
- **Dataset B is basically solved by the top teams.** Datasets A and C now decide the ranking.
- **Many top teams submit every 2 hours, around the clock.** It looks automated, and it partly
  tunes them to the hidden questions, which may not carry over to the next phase.

---

## 7. The rules we must follow

On 2026-11-24 the organizers clarified the rules, and one of our entries was flagged. The full
version is in `LENS_RULES.md`. In plain words:

1. **One place per question.** All readouts for a question come from the same token. We may
   choose that token by a simple rule (the last token, or the last question mark), but every
   readout uses it.
   - *Our flagged entry read one visit at the "?" and the rest at the last token. That is not
     allowed.*
2. **One visit per readout.** Each readout comes from one visit, with no adding or mixing
   readouts from different visits or positions. There are at most 10 visits per question,
   counting any visit read only to decide.
3. **No reading the question's text.** The lens may not use the words of the question to build,
   filter or reorder its readouts. Removing junk like punctuation is fine, as long as it is done
   the same way for every question.
4. **It has to really read the model.** No computing the answer ourselves, no answer tables, no
   other AI models, no trying to see the hidden questions.

**Score is never a reason to bend these.** A rule-breaking entry gets removed, and winners are
audited.

---

## 8. Glossary

| Word | Meaning |
|---|---|
| **Token** | A piece of a word; the unit the model reads and writes. |
| **State** | The long list of numbers attached to one token at one point in the thinking (its "notebook page"). |
| **Layer** | One processing step of the model. Inkling has 24. |
| **Visit** | One *run* of a layer. Inkling makes 48 per token, because the middle layers run three times. |
| **Forward pass** | One trip of the question through all 48 visits, ending in a guess at the next word. |
| **Output head** | The model's built-in translator from the final state to word scores. |
| **Lens** | Our translator from any state, at any visit, to a ranked list of words. |
| **Logit lens** | Using the model's own output head early, without training. |
| **Tuned lens / answer lens** | A logit lens with a small trained adjustment in front. Ours is trained toward the intermediate. |
| **Readout** | One lens's top-10 words from one visit. Up to 10 per question. |
| **Intermediate** | The thing the model works out on the way, which our lens must surface. |
| **Distractor** | A wrong option; often the model's final answer on two-step questions. |
| **NDIF** | The service that hosts the model on GPUs and runs our reading code next to it. |
| **nnsight** | The Python library we use to describe what to read inside the model. |
| **Dev set** | Our own labelled questions, kept aside to measure a lens fairly. |

---

## 9. Questions and answers

New questions go here as they come up, newest at the bottom.

**What does `python submit.py --local` do?**
It runs our entry on our own computer, the same way the leaderboard would, but on the 10 public
practice questions:
1. It downloads the practice questions.
2. It starts our entry. The entry sends the questions to NDIF, where the model runs and our lens
   reads it.
3. It writes the readouts to a file.
4. It scores them by exact matching.

Nothing is submitted, and it costs nothing.

**What happens when we submit to the leaderboard?**
1. Our entry is zipped and uploaded.
2. The organizers run it on 290 hidden questions in three datasets. It still uses NDIF under our
   key.
3. Their AI judge scores it, and our best score appears on the leaderboard after about 12
   minutes.

We may submit once every 2 hours. A crash on any dataset fails the whole submission, and we are
not shown why, so we test locally first.

**Why do we read the "last token"?**
The question names the token to read at: the last one. By that point the model has read the
whole question. Reading somewhere else is allowed, but then every readout of that question must
be read there.

**Why did the reasoning questions fail?**
At the last token the model is about to *start thinking* in text. Its translator shows that
intention, plus the topic, but not the computed answer. Either the answer isn't formed yet at
that token, or it is there in a form the untrained translator can't read. The answer lens is our
attempt at the second case.

**Where do we actually use Inkling?**
Only on NDIF, in two places:
1. When we collect the notebook pages to train on.
2. When the entry runs, locally or on the leaderboard.

Training the adjustment happens on our own computer, using saved pages and a downloaded copy of
the model's translator.

**Why did our entry get flagged?**
It read one visit at the question mark and the others at the last token: two positions in one
question. The clarified rules say all readouts of a question come from one position.
