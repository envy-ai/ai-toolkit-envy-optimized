from typing import TYPE_CHECKING
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.lora_key_format import peft_key_to_internal_key
from toolkit.print import print_acc
from safetensors.torch import load_file

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel
    from toolkit.stable_diffusion_model import StableDiffusion


def _get_assistant_lora_layer_names(lora_state_dict) -> list[str]:
    suffixes = (
        ".lora_A.weight",
        ".lora_B.weight",
        ".lora_down.weight",
        ".lora_up.weight",
        ".alpha",
    )
    layer_names = set()
    for key in lora_state_dict.keys():
        layer_name = key
        for suffix in suffixes:
            if key.endswith(suffix):
                layer_name = key[: -len(suffix)]
                break
        layer_names.add(layer_name)
    return sorted(layer_names)


def _canonicalize_assistant_lora_state_dict(lora_state_dict, sd):
    if hasattr(sd, "convert_assistant_lora_weights_before_load"):
        return sd.convert_assistant_lora_weights_before_load(lora_state_dict)
    if hasattr(sd, "convert_lora_weights_before_load"):
        converted_state_dict = sd.convert_lora_weights_before_load(lora_state_dict)
        if converted_state_dict is not None:
            return converted_state_dict
    return lora_state_dict


def _find_assistant_rank_key(lora_state_dict) -> str | None:
    candidate_keys = (
        "transformer.double_blocks.0.img_attn.qkv.lora_down.weight",
        "transformer.double_blocks.0.img_attn.qkv.lora_A.weight",
        "transformer.single_transformer_blocks.0.attn.to_k.lora_down.weight",
        "transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight",
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight",
        "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight",
        "diffusion_model.single_transformer_blocks.0.attn.to_k.lora_down.weight",
        "diffusion_model.single_transformer_blocks.0.attn.to_k.lora_A.weight",
    )
    for key in candidate_keys:
        if key in lora_state_dict:
            return key
    for key in lora_state_dict.keys():
        if key.endswith(".lora_down.weight") or key.endswith(".lora_A.weight"):
            return key
    return None


def _assistant_lora_is_transformer_only(layer_names: list[str]) -> bool:
    full_model_markers = (
        ".img_in",
        ".txt_in",
        ".time_in",
        ".guidance_in",
        ".double_stream_modulation_",
        ".single_stream_modulation",
        ".final_layer",
        ".proj_out",
    )
    for layer_name in layer_names:
        if any(marker in layer_name for marker in full_model_markers):
            return False
    return True


def load_assistant_lora_from_path(
    adapter_path, sd: 'StableDiffusion | BaseModel', strict: bool = False
) -> LoRASpecialNetwork:
    is_flux_style_model = sd.is_flux or getattr(sd, "is_transformer", False)
    if not is_flux_style_model:
        raise ValueError(
            "Only Flux-style transformer models can load assistant adapters currently."
        )
    pipe = sd.pipeline
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        text_encoder = sd.text_encoder
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        transformer = sd.unet
    print(f"Loading assistant adapter from {adapter_path}")
    adapter_name = adapter_path.split("/")[-1].split(".")[0]
    raw_lora_state_dict = load_file(adapter_path)
    raw_layer_names = _get_assistant_lora_layer_names(raw_lora_state_dict)

    print_acc(f"Assistant adapter layers found: {len(raw_layer_names)}")
    for layer_name in raw_layer_names:
        print_acc(f"  {layer_name}")

    lora_state_dict = _canonicalize_assistant_lora_state_dict(raw_lora_state_dict, sd)
    if strict:
        for key, value in lora_state_dict.items():
            if not key.endswith(".lora_A.weight"):
                continue
            up_key = key.removesuffix(".lora_A.weight") + ".lora_B.weight"
            up = lora_state_dict.get(up_key)
            if up is None or value.ndim != 2 or up.ndim != 2 or up.shape[1] != value.shape[0]:
                raise ValueError(f"Training adapter has an incomplete or invalid LoRA pair: {key}")
    layer_names = _get_assistant_lora_layer_names(lora_state_dict)
    if list(raw_lora_state_dict.keys()) != list(lora_state_dict.keys()):
        print_acc("Canonicalized assistant adapter layer names for load")

    rank_key = _find_assistant_rank_key(lora_state_dict)
    if rank_key is None:
        raise ValueError(
            f"Assistant adapter format is not supported. Could not find a LoRA rank key in {adapter_path}."
        )

    linear_dim = int(lora_state_dict[rank_key].shape[0])
    alpha_key = rank_key.rsplit(".", 2)[0] + ".alpha"
    # linear_alpha = int(lora_state_dict['lora_transformer_single_transformer_blocks_0_attn_to_k.alpha'].item())
    if alpha_key in lora_state_dict:
        linear_alpha = int(lora_state_dict[alpha_key].item())
    else:
        linear_alpha = linear_dim
    transformer_only = _assistant_lora_is_transformer_only(layer_names)
    # get dim and scale
    network_config = NetworkConfig(
        linear=linear_dim,
        linear_alpha=linear_alpha,
        transformer_only=transformer_only,
    )

    network_kwargs = dict(
        text_encoder=text_encoder,
        unet=transformer,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        is_assistant_adapter=True,
        base_model=sd,
    )
    # A fused projection can require a different rank from the other layers.
    modules_dim = {}
    modules_alpha = {}
    for key, value in lora_state_dict.items():
        suffix = next((suffix for suffix in (".lora_A.weight", ".lora_down.weight") if key.endswith(suffix)), None)
        if suffix is None:
            continue
        layer_name = key[:-len(suffix)]
        module_name = peft_key_to_internal_key(layer_name + ".lora_A.weight").removesuffix(".lora_down.weight")
        modules_dim[module_name] = int(value.shape[0])
        alpha = lora_state_dict.get(layer_name + ".alpha")
        modules_alpha[module_name] = float(alpha.item()) if alpha is not None else int(value.shape[0])
    if strict:
        network_kwargs["modules_dim"] = modules_dim
        network_kwargs["modules_alpha"] = modules_alpha
    if getattr(sd, "is_transformer", False):
        network_kwargs["is_transformer"] = True
        if getattr(sd, "target_lora_modules", None):
            network_kwargs["target_lin_modules"] = list(sd.target_lora_modules)
    else:
        network_kwargs["is_flux"] = True
    network = LoRASpecialNetwork(**network_kwargs)
    network.apply_to(
        text_encoder,
        transformer,
        apply_text_encoder=False,
        apply_unet=True
    )
    network.force_to(sd.device_torch, dtype=sd.torch_dtype)
    network.eval()
    network._update_torch_multiplier()
    extra_weights = network.load_weights(lora_state_dict)
    if strict and extra_weights:
        unmatched = [key for key in extra_weights if key.endswith((".lora_down.weight", ".lora_up.weight"))]
        if unmatched:
            raise ValueError(f"Training adapter has unmatched LoRA weights: {unmatched[:5]}")
    network.requires_grad_(False)
    network.is_active = True

    return network
