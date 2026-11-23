"""A token lens, written as functions that run inside an NDIF trace.

A lens here takes ``h``, the state entering one layer visit at the read position (a tensor of
shape [hidden]), and returns scores over the vocabulary ([vocabulary]). Everything in these
functions executes on NDIF; only the top ten token ids per visit are sent back.
"""


def layers(model):
    """The decoder layers: 24 of them, of which 6 to 17 run three times per forward pass."""
    return model.model.language_model.layers


def visit_address(config, visit: int) -> tuple[int, int]:
    """Where a layer visit happens: which layer, and which of that layer's runs (0, 1 or 2).

    One token passes through 48 visits, numbered 0 to 47 in the order they run: layers 0-5 once,
    then layers 6-17 three times over, then layers 18-23 once. ``config`` is the model's text
    configuration, which holds those numbers.
    """
    start, end, passes = config.loop_layer_index_start, config.loop_layer_index_end, config.loop_num_passes
    looped = end - start
    if not 0 <= visit < config.num_hidden_layers + looped * (passes - 1):
        raise ValueError(f"there is no layer visit {visit}")
    if visit < start:
        return visit, 0
    if visit < start + looped * passes:
        return start + (visit - start) % looped, (visit - start) // looped
    return visit - looped * (passes - 1), 0


def logit_lens(model, h):
    """The model's own final norm and unembedding, applied early. No trained weights.

    The model also soft-caps its logits, which does not change their order, so it is left out.
    """
    return model.lm_head(model.model.language_model.norm(h))


def dual_logit_lens(model, h_read, h_anchor, anchor_weight: float = 1.5):
    """Weighted sum of logit lenses at the read position and one earlier anchor token.

    Calendar and arithmetic intermediates often surface more clearly on the prompt's last
    contentful token (e.g. the day or month named in the question) than on ``<|message_model|>``,
    while factoids still read cleanly at the official read position. A slightly higher weight on
    the anchor keeps those intermediates in the top-10 without drowning factoid votes.
    """
    return logit_lens(model, h_read).float() + anchor_weight * logit_lens(model, h_anchor).float()


def fetch(repository: str, filename: str, token: str | None = None) -> dict:
    """A lens's published weights, downloaded by NDIF itself: a dict of tensors.

    The starter lens has no weights and does not call this. A lens that does publishes them to
    the Hugging Face Hub as a safetensors file and calls this at the top of the session, before
    any trace, so that they are fetched once per request and never travel in it. The file is read
    straight into memory and nothing is left on NDIF's disk. NDIF has no Hugging Face token of
    its own, so a private repository needs yours passed as ``token``.
    """
    import urllib.request

    import safetensors.torch

    request = urllib.request.Request(f"https://huggingface.co/{repository}/resolve/main/{filename}")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=600) as response:
        return safetensors.torch.load(response.read())
