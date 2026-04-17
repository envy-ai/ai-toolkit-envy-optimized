from typing import TYPE_CHECKING
from toolkit.config_modules import NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.print import print_acc
from safetensors.torch import load_file

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel
    from toolkit.stable_diffusion_model import StableDiffusion


def _get_assistant_lora_layer_names(lora_state_dict) -> list[str]:
    suffixes = (
        ".lora_A.weight",
        ".lora_B.weight",
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
    adapter_path, sd: 'StableDiffusion | BaseModel'
) -> LoRASpecialNetwork:
    is_flux_style_model = sd.is_flux or getattr(sd, "is_transformer", False)
    if not is_flux_style_model:
        raise ValueError(
            "Only Flux-style transformer models can load assistant adapters currently."
        )
    pipe = sd.pipeline
    print(f"Loading assistant adapter from {adapter_path}")
    adapter_name = adapter_path.split("/")[-1].split(".")[0]
    raw_lora_state_dict = load_file(adapter_path)
    raw_layer_names = _get_assistant_lora_layer_names(raw_lora_state_dict)

    print_acc(f"Assistant adapter layers found: {len(raw_layer_names)}")
    for layer_name in raw_layer_names:
        print_acc(f"  {layer_name}")

    lora_state_dict = _canonicalize_assistant_lora_state_dict(raw_lora_state_dict, sd)
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
        text_encoder=pipe.text_encoder,
        unet=pipe.transformer,
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
    if getattr(sd, "is_transformer", False):
        network_kwargs["is_transformer"] = True
        if getattr(sd, "target_lora_modules", None):
            network_kwargs["target_lin_modules"] = sd.target_lora_modules
    else:
        network_kwargs["is_flux"] = True
    network = LoRASpecialNetwork(**network_kwargs)
    network.apply_to(
        pipe.text_encoder,
        pipe.transformer,
        apply_text_encoder=False,
        apply_unet=True
    )
    network.force_to(sd.device_torch, dtype=sd.torch_dtype)
    network.eval()
    network._update_torch_multiplier()
    network.load_weights(lora_state_dict)
    network.is_active = True

    return network
