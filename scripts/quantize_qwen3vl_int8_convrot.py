#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
import re
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from comfy.quant_ops import QuantizedTensor


LANGUAGE_LINEAR_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"
    r"\.weight$"
)


def quant_conf(convrot_groupsize: int) -> torch.Tensor:
    payload = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": convrot_groupsize,
    }
    return torch.tensor(list(json.dumps(payload).encode("utf-8")), dtype=torch.uint8)


def should_quantize(key: str, shape: tuple[int, ...]) -> bool:
    return len(shape) == 2 and LANGUAGE_LINEAR_RE.match(key) is not None


def quantize_file(input_path: Path, output_path: Path, convrot_groupsize: int, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=str(output_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    out = {}
    quantized = 0
    copied = 0

    try:
        with safe_open(input_path, framework="pt", device="cpu") as src:
            for key in src.keys():
                tensor = src.get_tensor(key)
                shape = tuple(tensor.shape)

                if should_quantize(key, shape):
                    module_key = key[: -len(".weight")]
                    q = QuantizedTensor.from_float(
                        tensor,
                        "TensorWiseINT8Layout",
                        per_channel=True,
                        convrot=True,
                        convrot_groupsize=convrot_groupsize,
                    )
                    out[key] = q._qdata.contiguous().cpu()
                    out[f"{module_key}.weight_scale"] = q._params.scale.contiguous().cpu()
                    out[f"{module_key}.comfy_quant"] = quant_conf(convrot_groupsize)
                    quantized += 1
                else:
                    out[key] = tensor.contiguous().cpu()
                    copied += 1

                if (quantized + copied) % 50 == 0:
                    print(f"processed={quantized + copied} quantized={quantized} copied={copied}", flush=True)

        metadata = {
            "format": "pt",
            "quantization": (
                "ComfyUI int8_tensorwise ConvRot; language_model layer projection "
                "weights quantized with per-channel scales; embeddings, norms, and visual tower kept BF16"
            ),
            "convrot_groupsize": str(convrot_groupsize),
            "source": str(input_path),
        }
        print(f"saving {tmp_path}", flush=True)
        save_file(out, str(tmp_path), metadata=metadata)
        os.replace(tmp_path, output_path)
    except Exception:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    print(f"wrote {output_path}")
    print(f"quantized={quantized} copied={copied}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--convrot-groupsize", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    quantize_file(args.input, args.output, args.convrot_groupsize, args.overwrite)


if __name__ == "__main__":
    main()
