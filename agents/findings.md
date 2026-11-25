# Findings

What we know, with the evidence for each claim. Newest at the top of each section.

## The model (Inkling 2 Mini)

- **No single residual stream.** Layers sum into 6-layer *blocks*, and at layers 0, 6, 12 and 18 the finished block is frozen into `block_states`. Each sublayer's input is a learned-query softmax mix of the frozen blocks plus the active block. The output is `final_input(block_states, hidden) → norm → lm_head`, with logit softcap 30.
  - Source: `transformers/models/inkling2_loop/modeling_inkling2_loop.py`, `Inkling2LoopTextModel.forward`, `Inkling2LoopBlockAttentionResidual`.
- **Passes 2 and 3 restart from the prelude's state.** The previous pass reaches them only through four *messages* per layer (query, key, value, MoE), blended in by rank-64 tanh gates. Lens readouts therefore drop at the start of each pass and rise at its end.
  - Source: the same file, and <https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/>.
- **A layer's call is `layer(hidden, block_states, [messages], ...)`,** so on NDIF `hidden, blocks = layer.inputs[0][:2]`. `layer.output` at visit v equals the input of visit v+1 within a pass. The probe checked this at 43 of 43 visits.
- **Some NDIF replicas split the model over two GPUs.** `block_states` and `hidden` can then be on different devices, so move tensors to one device before `torch.cat`.
  - Seen: a probe request failed with `Expected all tensors to be on the same device … cuda:1 … cuda:0`.
- **At the chat read position (`<|message_model|>`), the model's next token is `<|content_thinking|>`.** The `final_input`-mixed readout late in the network fills with that format prediction. On reasoning prompts the last visits show the topic (`arithmetic`, `week`) but not the answer.

## The rules (clarified by the organizers, 2026-11-24; see LENS_RULES.md)

- **One position per prompt:** every readout of a prompt at the same token.
- **One visit per readout:** no sums or differences across visits.
- **No prompt text** in building, filtering or ranking readouts.
- **Status of our designs:**
  - v1, v2 and v3 read visit 46 at the `?` and the rest at the read position, which **breaks the first rule**;
  - our first entry summed two positions, which breaks it too;
  - both scored entries were flagged (the email reached the user);
  - v4, on branch `idea/one-position`, reads every visit at the read position.
- **What the `?` findings below are now worth:** they still describe the model, but an entry may only use them by reading *all* its visits at the `?`.

## The scoring

- **Each scored dataset is one family** (`evaluator/evaluate.py`, `_questions`).
- **A question's readouts need distinct layer labels,** so there are at most 10 readouts, one per visit. Each can come from any computation at that visit, at any position.
- **Practice (10 questions) is a poor guide.** A lens at 0.90 on practice scored 0.660 on the hidden sets (see [leaderboard.md](leaderboard.md)).

## Lenses

Evidence for this section:
- data: `research/data/probe/`, from the full probe of the dev set (434 questions, 48 visits × 4 readouts × the last 12 positions);
- analysis: `research/analyze_probe.py`;
- an ad-hoc search over designs on 2026-11-24.

Pass rates are by exact match, over the mean of families.

- **The starter on our dev set (chat):** 0.550. By family: arith1 0.13, arith2 0.53, country 0.80, hop_country 0.90, instrument 0.63, shift1 0.07, shift2 0.33, sport 1.00. In plain format it scores 0.382.
- **The question's `?` is where calendar answers live.**
  - For "What is the day after Friday?", the read position (`<|message_model|>`) gives 0.03–0.10 under every readout.
  - At the `?` token, visits 45 and 46 give **1.00** under the active, mixed and MoE-input readouts alike.
  - The instruction that follows (" Answer in one word.") washes the answer out.
- **At the read position, choosing the readout or the visits does not help.** A greedy choice of up to 10 readouts at the read position scores 0.458 on held-out prompts, against 0.462 for the starter. The gain has to come from position.
- **The `final_input`-mixed readout is faithful but not better here.** Late in the network it fills with what the model will *write* next (`<|content_thinking|>`), which is no intermediate. Active-block and MoE-input readouts show more content.
- **Arithmetic intermediates are not readable** by any of the 4 readouts at any of the 12 positions. The best single config reaches 0.13 on arith1 (chat). This needs a trained lens or a different idea.
- **Two-hop country questions are easy** (0.90–1.00 at the read position). The intermediate, the country, shows up next to the answer.
- **Plain format:** the read position is the final `.`. The subject's last token and the `?` read much better.
- **Search over late visits (40–47), active lens:** visits 44, 45 and 47 at the read position plus 46 at the `?` maximize the worse of the two held-out halves. The halves score 0.676 and 0.727, all chat prompts 0.703, against 0.550 for the starter.
  - By family: shift1 1.00, instrument 0.79, country 0.93, hop_country 1.00, sport 1.00, arith2 0.47, shift2 0.30, arith1 0.13.
  - This is lens v1 (`submission/main.py`, branch `version1`).
