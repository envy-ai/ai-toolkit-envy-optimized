"""Iris's exact chat template and 12 layer taps, without a vision tower or LM head."""
import warnings
import json
from pathlib import Path
import torch
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextOnlyEncoder
from toolkit.prompt_utils import PromptEmbeds

TEXT_ENCODER_REPO = "Qwen/Qwen3-VL-4B-Instruct"
HIDDEN_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
PREFIX = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TEXT_LENGTH = 300


class IrisTextEncoder(Qwen3VLTextOnlyEncoder):
    aitk_subfolder = ""
    aitk_config_repo = TEXT_ENCODER_REPO
    aitk_cast_quantized_load = True

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        config = super().aitk_load_config(path, subfolder)
        return getattr(config, "text_config", config)

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        # TextModel.from_pretrained does not strip the VL checkpoint's nested
        # language_model namespace. Filter before loading, and never materialize
        # the unused vision tower or output vocabulary projection.
        def resolve(filename):
            if Path(path).is_dir():
                file = Path(path) / filename
                if not file.is_file():
                    raise FileNotFoundError(file)
                return str(file)
            try:
                return hf_hub_download(path, filename, local_files_only=True)
            except Exception:
                return hf_hub_download(path, filename)

        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
        try:
            index = json.loads(Path(resolve("model.safetensors.index.json")).read_text())
            filenames = sorted(set(index["weight_map"].values()))
        except (FileNotFoundError, EntryNotFoundError, LocalEntryNotFoundError):
            filenames = ["model.safetensors"]
        state = {}
        files = []
        for filename in filenames:
            file = resolve(filename)
            files.append(file)
            with safe_open(file, framework="pt", device="cpu") as checkpoint:
                for key in checkpoint.keys():
                    if key.startswith(("model.language_model.", "language_model.", "layers.", "embed_tokens.", "norm.")):
                        state[key] = checkpoint.get_tensor(key)
        model = cls.load_from_state_dict(state, dtype, config=cls.aitk_load_config(path))
        model.aitk_load_files = files
        return model

    @classmethod
    def convert_state_dict_on_load(cls, state_dict):
        for prefix in ("model.language_model.", "language_model.", "model."):
            if prefix + "embed_tokens.weight" in state_dict:
                return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        return {key: value for key, value in state_dict.items() if not key.startswith("lm_head.")}


@torch.no_grad()
def encode_iris_prompts(encoder, tokenizer, prompts, dtype):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    budget = TEXT_LENGTH - len(suffix)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 151643
    ids = torch.full((len(prompts), len(prefix) + TEXT_LENGTH), pad, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, prompt in enumerate(prompts):
        caption = tokenizer.encode(prompt, add_special_tokens=False)
        if len(caption) > budget:
            warnings.warn(f"Iris caption truncated from {len(caption)} to {budget} tokens; chat suffix preserved.")
        tokens = prefix + caption[:budget] + suffix
        ids[row, :len(tokens)] = torch.tensor(tokens)
        mask[row, :len(tokens)] = 1
    ids, mask = ids.to(encoder.device), mask.to(encoder.device)
    result = encoder(input_ids=ids, attention_mask=mask, use_cache=False,
                     output_hidden_states=True, return_dict=True)
    mask = mask[:, len(prefix):].bool()
    states = torch.stack([result.hidden_states[i][:, len(prefix):] for i in HIDDEN_LAYERS], dim=2)
    states = states.to(dtype) * mask[:, :, None, None]
    # Keep the generic prompt cache/batching API's [batch, tokens, features] shape.
    return PromptEmbeds(states.flatten(2), attention_mask=mask)
