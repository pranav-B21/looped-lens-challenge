"""Download the model's unembedding and final norm weights, only those bytes of the checkpoint.

    .venv/bin/python research/fetch_unembed.py [--out research/data/unembed.safetensors]

The checkpoint is one 20 GB safetensors file. Its header gives each tensor's byte range, so two
HTTP range requests fetch ``model.llm.unembed.weight`` [201024, 2304] (bf16, ~0.93 GB) and
``model.llm.norm.weight`` [2304]. With them a lens can be trained and scored on this machine:
logits = unembed @ rmsnorm(state) * norm_weight, as the model's own head does (softcap aside,
which keeps the order).
"""

import argparse
import json
import struct
from pathlib import Path

import requests
import torch
from huggingface_hub import get_token, hf_hub_url
from safetensors.torch import save_file

REPO = "thinkingmachines/Inkling-2-Mini"
FILE = "model-00001-of-00001.safetensors"
WANTED = {"model.llm.unembed.weight": "unembed", "model.llm.norm.weight": "norm"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data" / "unembed.safetensors")
    args = parser.parse_args()
    url = hf_hub_url(REPO, FILE)
    headers = {"Authorization": f"Bearer {get_token()}"} if get_token() else {}
    session = requests.Session()

    def get_range(start, end):
        response = session.get(url, headers={**headers, "Range": f"bytes={start}-{end - 1}"}, timeout=600, allow_redirects=True)
        response.raise_for_status()
        assert len(response.content) == end - start, (len(response.content), end - start)
        return response.content

    (header_size,) = struct.unpack("<Q", get_range(0, 8))
    header = json.loads(get_range(8, 8 + header_size))
    base = 8 + header_size
    tensors = {}
    for name, short in WANTED.items():
        info = header[name]
        assert info["dtype"] == "BF16", info
        start, end = info["data_offsets"]
        print(f"fetching {name} {info['shape']} ({(end - start) / 1e6:.0f} MB)", flush=True)
        raw = bytearray(get_range(base + start, base + end))
        tensors[short] = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(info["shape"]).clone()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
