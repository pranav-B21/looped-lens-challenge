"""Fit a tuned lens: for each visit, an affine map from its state to the model's final state.

    .venv/bin/python research/train_lens.py [--acts research/data/acts] [--out research/data/tuned_lens.safetensors]

Closed-form ridge regression, per visit, of ``y`` (the final norm's input) on ``x`` (the state
entering the visit), both from research/collect_acts.py: W = (X'X + lambda I)^-1 X'Y, with a bias
column. Lambda is chosen per visit on a held-out tenth of the positions by the cosine similarity
of prediction and target, after the same unweighted RMS normalization the final norm applies. It
prints that similarity next to the identity map's, which is what the plain logit lens assumes.

The weights are saved as safetensors in bf16: ``v{visit}.weight`` [hidden, hidden], applied as
``x @ weight + bias``, and ``v{visit}.bias`` [hidden].
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


def rms(t):
    return t / t.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(rms(a), rms(b), dim=-1).mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    parser.add_argument("--acts", type=Path, default=here / "data" / "acts")
    parser.add_argument("--out", type=Path, default=here / "data" / "tuned_lens.safetensors")
    parser.add_argument("--visits", type=int, nargs="*", default=None, help="only these visits")
    args = parser.parse_args()

    batches = [torch.load(p) for p in sorted(args.acts.glob("batch-*.pt"))]
    visits = batches[0]["visits"]
    x = torch.cat([b["x"] for b in batches], dim=1).float()  # [visits, N, hidden]
    y = torch.cat([b["y"] for b in batches], dim=0).float()  # [N, hidden]
    n = y.shape[0]
    generator = torch.Generator().manual_seed(0)
    order = torch.randperm(n, generator=generator)
    held, fit = order[: n // 10], order[n // 10 :]
    print(f"{n} positions ({len(fit)} fit, {len(held)} held out), visits {visits}")

    tensors = {}
    for k, visit in enumerate(visits):
        if args.visits and visit not in args.visits:
            continue
        X = torch.cat([x[k], torch.ones(n, 1)], dim=1).double()
        Y = y.double()
        gram = X[fit].T @ X[fit]
        cross = X[fit].T @ Y[fit]
        scale = gram.diagonal().mean()
        best = None
        for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            W = torch.linalg.solve(gram + lam * scale * torch.eye(gram.shape[0], dtype=gram.dtype), cross)
            score = cosine((X[held] @ W).float(), y[held])
            if best is None or score > best[0]:
                best = (score, lam, W)
        score, lam, W = best
        identity = cosine(x[k][held], y[held])
        print(f"  visit {visit:>2}: held-out cosine {score:.3f} (identity {identity:.3f}), lambda {lam:g}")
        tensors[f"v{visit}.weight"] = W[:-1].float().to(torch.bfloat16).contiguous()
        tensors[f"v{visit}.bias"] = W[-1].float().to(torch.bfloat16).contiguous()
    save_file(tensors, str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB, {len(tensors) // 2} visits)")


if __name__ == "__main__":
    main()
