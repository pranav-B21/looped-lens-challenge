# The lens rules: read before every change to `submission/`

The organizers clarified the rules on 2026-11-24 and flagged entries that break them. One of
ours was flagged: lens v1 and v3 read visit 46 at the question's last `?` and every other visit
at the read position. Source: the "Rule clarifications" section of
<https://ndif.us/looped-lens-challenge/>. Reread it there when in doubt, since it may change.

**Score never justifies breaking these.** A lens that scores higher by bending a rule is a
worse entry: it gets removed from the leaderboard and disqualified at the audit. When an idea
is near a line, do not build it. Tell the person, and have them ask on the challenge Discord
first.

## The rules

1. **One position per prompt.** Every readout of a prompt is taken at the same single token
   position.
   - It may be chosen from the activations, counting every visit read to decide.
   - It may also be chosen from the prompt's punctuation or chat template, for example the last
     `?`.
   - It may differ between prompts.
   - It may **not** be chosen by comparing what readouts at several candidate positions say.
2. **One visit per readout.**
   - A prompt reads at most 10 of the 48 visits, **counting visits used only to make
     decisions**.
   - Each readout is the top 10 of one lens applied to the state at one visit, labelled with
     that visit's number (0 to 47).
   - No visit appears twice.
   - A lens may run the model's own later layers on that state (a coda lens).
3. **No prompt text.** Readouts may not be built, filtered or reranked using the prompt's text
   or token ids.

## Allowed

- Logit, tuned, translator and coda lenses on one visit at the chosen position.
- An ensemble of lenses on the **same visit and position**, combined into one readout, for
  example the logit lens and a tuned lens at visit 40 with their probabilities averaged.
- Different visits, or a different position, for different prompts.
- Cleaning a readout:
  - removing control tokens such as `<|end_message|>`;
  - removing repeats of a token in another casing or spacing (`" Paris"` after `"Paris"`).

  Apply the same rule to every prompt, never one that depends on the prompt.

## Not allowed

- **More than one readout per visit**, or labels that are not visit numbers (`"24b"`).
- **Combining positions:**
  - a best probability over the last N tokens;
  - alternating two positions' top tokens;
  - subtracting another position's state;
  - a weighted sum of two positions' logit lenses (our old `dual_logit_lens`);
  - reading different visits at different positions within one prompt (our v1 and v3).
- **Combining visits:**
  - a weighted sum of logit lenses over several visits labelled as one;
  - subtracting one visit's logits from another's.
- **Reading many and keeping the ones that agree:**
  - reading extra visits or positions and submitting the subset whose top tokens agree;
  - picking between candidate position sets by agreement.

  Visits read only to decide count toward the 10.
- **Using the prompt's tokens:** dropping or demoting tokens because they appear in the prompt
  (removing "9" and "2" for "What is 9 − 2?").
- The README's standing rules:
  - no readouts that do not come from activations;
  - no other models;
  - nothing computed from the prompt (arithmetic, fact lookup, rules keyed on question text
    or id);
  - no answer tables;
  - no attempts to get the hidden questions out.

## Check before every submission

Answer each question from the code, not from memory, and write the answers in the day's log
under `agents/`:

- [ ] Is every readout of a prompt taken at one position? Grep `main.py` for every index used
      on `.input` / `.output` and confirm it is the same variable for every visit.
- [ ] If the position is chosen per prompt, does the choice use only punctuation or chat-template
      tokens, or activations? Does it avoid comparing readouts at candidate positions?
- [ ] Are there 10 visits or fewer per prompt, including any read only to decide? Is each one
      labelled with its own number, with no repeats?
- [ ] Does each readout come from the state at one visit only, with no sums or differences
      across visits or positions?
- [ ] Is every token filter the same for every prompt, with no prompt tokens and no
      question-dependent lists?
- [ ] Are any trained weights fitted on our own prompts, with no answer tables?

If any box is unclear, do not submit. Ask the person first.
