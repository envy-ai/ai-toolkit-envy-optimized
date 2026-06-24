from fnmatch import fnmatch
from typing import List, Optional, Union, TYPE_CHECKING
from types import MethodType
import torch
import torch.nn.functional as F

from optimum.quanto.quantize import _quantize_submodule
from optimum.quanto.tensor import Optimizer, qtype, qtypes
from torchao.quantization.quant_api import (
    quantize_ as torchao_quantize_,
    Float8WeightOnlyConfig,
    UIntXWeightOnlyConfig,
    Int8WeightOnlyConfig
)
from optimum.quanto import freeze
from tqdm import tqdm
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

from toolkit.print import print_acc
import os

NUCLEUS_MOE_EXPERT_CHUNK_SIZE = int(os.environ.get("AI_TOOLKIT_NUCLEUS_MOE_EXPERT_CHUNK_SIZE", "8"))

if TYPE_CHECKING:
    from toolkit.models.base_model import BaseModel

# the quantize function in quanto had a bug where it was using exclude instead of include

Q_MODULES = [
    "QLinear",
    "QConv2d",
    "QEmbedding",
    "QBatchNorm2d",
    "QLayerNorm",
    "QConvTranspose2d",
    "QEmbeddingBag",
]


class QuantizedNucleusMoEWeight:
    def __init__(
        self,
        qweight: torch.Tensor,
        scale: torch.Tensor,
        original_dtype: torch.dtype,
        keep_on_cpu: bool = False,
    ):
        self.qweight = qweight
        self.scale = scale
        self.original_dtype = original_dtype
        self.keep_on_cpu = keep_on_cpu

    @classmethod
    @torch.no_grad()
    def from_tensor(cls, tensor: torch.Tensor, keep_on_cpu: bool = False):
        original_dtype = tensor.dtype
        qweight_chunks = []
        scale_chunks = []

        # Quantize per expert and per output column. This keeps temporary peak
        # memory bounded to one expert slice instead of the full packed tensor.
        for expert_weight in tensor.detach():
            expert_weight = expert_weight.to(device="cpu", dtype=torch.float32)
            max_abs = expert_weight.abs().amax(dim=0, keepdim=True)
            scale = torch.where(
                max_abs > 0,
                max_abs / 127.0,
                torch.ones_like(max_abs),
            )
            qweight = torch.round(expert_weight / scale).clamp(-127, 127).to(torch.int8)
            qweight_chunks.append(qweight.contiguous())
            scale_chunks.append(scale.contiguous())

        return cls(
            qweight=torch.stack(qweight_chunks, dim=0).contiguous(),
            scale=torch.stack(scale_chunks, dim=0).contiguous(),
            original_dtype=original_dtype,
            keep_on_cpu=keep_on_cpu,
        )

    def dequantize(
        self,
        expert_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = self.original_dtype
        qweight = self.qweight if expert_idx is None else self.qweight[expert_idx]
        scale = self.scale if expert_idx is None else self.scale[expert_idx]
        return qweight.to(device=device, dtype=dtype) * scale.to(device=device, dtype=dtype)

    def storage_nbytes(self) -> int:
        return (
            self.qweight.numel() * self.qweight.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def apply_(self, fn):
        if self.keep_on_cpu:
            # Low-VRAM Nucleus training cannot afford to move all packed MoE
            # expert weights to CUDA during Module.to(...). Keep storage on CPU
            # and copy only the active expert slice inside dequantize().
            self.scale = fn(self.scale).to("cpu")
            return self

        self.qweight = fn(self.qweight)
        self.scale = fn(self.scale)
        return self


def _run_quantized_nucleus_moe_for_loop(
    self,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if (
        num_tokens_per_expert.numel() == self.num_experts
        and x.shape[0] % self.num_experts == 0
        and bool(torch.all(num_tokens_per_expert == num_tokens_per_expert[0]).item())
    ):
        tokens_per_expert = x.shape[0] // self.num_experts
        x_per_expert = x.reshape(self.num_experts, tokens_per_expert, x.shape[-1])
        expert_outputs = []
        chunk_size = max(1, NUCLEUS_MOE_EXPERT_CHUNK_SIZE)

        for start_idx in range(0, self.num_experts, chunk_size):
            end_idx = min(start_idx + chunk_size, self.num_experts)
            expert_slice = slice(start_idx, end_idx)
            x_chunk = x_per_expert[expert_slice]

            gate_up_proj = self.gate_up_proj.dequantize(
                expert_slice,
                device=x.device,
                dtype=x.dtype,
            )
            gate_up = torch.bmm(x_chunk, gate_up_proj)
            del gate_up_proj

            gate, up = gate_up.chunk(2, dim=-1)
            hidden_chunk = F.silu(gate) * up
            del gate_up, gate, up

            down_proj = self.down_proj.dequantize(
                expert_slice,
                device=x.device,
                dtype=x.dtype,
            )
            expert_outputs.append(torch.bmm(hidden_chunk, down_proj))
            del hidden_chunk, down_proj

        return torch.cat(expert_outputs, dim=0).reshape(x.shape[0], -1)

    num_tokens_per_expert_list = num_tokens_per_expert.tolist()
    num_real_tokens = sum(num_tokens_per_expert_list)
    num_padding = x.shape[0] - num_real_tokens
    x_per_expert = torch.split(
        x[:num_real_tokens],
        split_size_or_sections=num_tokens_per_expert_list,
        dim=0,
    )

    expert_outputs = []
    for expert_idx, x_expert in enumerate(x_per_expert):
        gate_up_proj = self.gate_up_proj.dequantize(
            expert_idx,
            device=x_expert.device,
            dtype=x_expert.dtype,
        )
        gate_up = torch.matmul(x_expert, gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        down_proj = self.down_proj.dequantize(
            expert_idx,
            device=x_expert.device,
            dtype=x_expert.dtype,
        )
        out_expert = torch.matmul(F.silu(gate) * up, down_proj)
        expert_outputs.append(out_expert)

    out = torch.cat(expert_outputs, dim=0)
    if num_padding > 0:
        out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


def _apply_quantized_nucleus_moe(self, fn, recurse=True):
    try:
        result = self._nucleus_moe_orig_apply(fn, recurse=recurse)
    except TypeError:
        result = self._nucleus_moe_orig_apply(fn)

    if isinstance(self.gate_up_proj, QuantizedNucleusMoEWeight):
        self.gate_up_proj.apply_(fn)
    if isinstance(self.down_proj, QuantizedNucleusMoEWeight):
        self.down_proj.apply_(fn)
    return result


def _is_nucleus_moe_experts_module(module: torch.nn.Module) -> bool:
    return (
        module.__class__.__name__ == "SwiGLUExperts"
        and hasattr(module, "gate_up_proj")
        and hasattr(module, "down_proj")
        and isinstance(module.gate_up_proj, torch.nn.Parameter)
        and isinstance(module.down_proj, torch.nn.Parameter)
        and module.gate_up_proj.ndim == 3
        and module.down_proj.ndim == 3
    )


def quantize_nucleus_moe_experts(
    model: torch.nn.Module,
    keep_on_cpu: bool = False,
) -> int:
    quantized_count = 0
    for module in model.modules():
        if not _is_nucleus_moe_experts_module(module):
            continue

        gate_up_proj = module._parameters.pop("gate_up_proj")
        down_proj = module._parameters.pop("down_proj")
        module.gate_up_proj = QuantizedNucleusMoEWeight.from_tensor(
            gate_up_proj,
            keep_on_cpu=keep_on_cpu,
        )
        module.down_proj = QuantizedNucleusMoEWeight.from_tensor(
            down_proj,
            keep_on_cpu=keep_on_cpu,
        )
        module.use_grouped_mm = False
        module._run_experts_for_loop = MethodType(
            _run_quantized_nucleus_moe_for_loop,
            module,
        )
        if not hasattr(module, "_nucleus_moe_orig_apply"):
            module._nucleus_moe_orig_apply = module._apply
            module._apply = MethodType(_apply_quantized_nucleus_moe, module)

        del gate_up_proj
        del down_proj
        quantized_count += 1

    return quantized_count


def move_nucleus_moe_quantized_weights(
    model: torch.nn.Module,
    device: Union[str, torch.device],
    dtype: Optional[torch.dtype] = None,
    non_blocking: bool = True,
) -> int:
    moved_count = 0
    device = torch.device(device)

    for module in model.modules():
        for attr_name in ("gate_up_proj", "down_proj"):
            packed_weight = getattr(module, attr_name, None)
            if not isinstance(packed_weight, QuantizedNucleusMoEWeight):
                continue

            packed_weight.keep_on_cpu = False
            packed_weight.qweight = packed_weight.qweight.to(
                device=device,
                non_blocking=non_blocking,
            )

            scale_kwargs = {
                "device": device,
                "non_blocking": non_blocking,
            }
            if dtype is not None:
                scale_kwargs["dtype"] = dtype
            packed_weight.scale = packed_weight.scale.to(**scale_kwargs)
            moved_count += 1

    return moved_count

torchao_qtypes = {
    # "int4": Int4WeightOnlyConfig(),
    "uint2": UIntXWeightOnlyConfig(torch.uint2),
    "uint3": UIntXWeightOnlyConfig(torch.uint3),
    "uint4": UIntXWeightOnlyConfig(torch.uint4),
    "uint5": UIntXWeightOnlyConfig(torch.uint5),
    "uint6": UIntXWeightOnlyConfig(torch.uint6),
    "uint7": UIntXWeightOnlyConfig(torch.uint7),
    "uint8": UIntXWeightOnlyConfig(torch.uint8),
    "int8": Int8WeightOnlyConfig(),
    "float8": Float8WeightOnlyConfig(),
}


class aotype:
    def __init__(self, name: str):
        self.name = name
        self.config = torchao_qtypes[name]


def get_qtype(qtype: Union[str, qtype]) -> qtype:
    if qtype in torchao_qtypes:
        return aotype(qtype)
    if isinstance(qtype, str):
        return qtypes[qtype]
    else:
        return qtype


def quantize(
    model: torch.nn.Module,
    weights: Optional[Union[str, qtype, aotype]] = None,
    activations: Optional[Union[str, qtype]] = None,
    optimizer: Optional[Optimizer] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
):
    """Quantize the specified model submodules

    Recursively quantize the submodules of the specified parent model.

    Only modules that have quantized counterparts will be quantized.

    If include patterns are specified, the submodule name must match one of them.

    If exclude patterns are specified, the submodule must not match one of them.

    Include or exclude patterns are Unix shell-style wildcards which are NOT regular expressions. See
    https://docs.python.org/3/library/fnmatch.html for more details.

    Note: quantization happens in-place and modifies the original model and its descendants.

    Args:
        model (`torch.nn.Module`): the model whose submodules will be quantized.
        weights (`Optional[Union[str, qtype]]`): the qtype for weights quantization.
        activations (`Optional[Union[str, qtype]]`): the qtype for activations quantization.
        include (`Optional[Union[str, List[str]]]`):
            Patterns constituting the allowlist. If provided, module names must match at
            least one pattern from the allowlist.
        exclude (`Optional[Union[str, List[str]]]`):
            Patterns constituting the denylist. If provided, module names must not match
            any patterns from the denylist.
    """
    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude
    for name, m in model.named_modules():
        if include is not None and not any(
            fnmatch(name, pattern) for pattern in include
        ):
            continue
        if exclude is not None and any(fnmatch(name, pattern) for pattern in exclude):
            continue
        try:
            # check if m is QLinear or QConv2d
            if m.__class__.__name__ in Q_MODULES:
                continue
            else:
                if isinstance(weights, aotype):
                    torchao_quantize_(m, weights.config)
                else:
                    _quantize_submodule(
                        model,
                        name,
                        m,
                        weights=weights,
                        activations=activations,
                        optimizer=optimizer,
                    )
        except Exception as e:
            print(f"Failed to quantize {name}: {e}")
            # raise e


def quantize_model(
    base_model: "BaseModel",
    model_to_quantize: torch.nn.Module,
):
    from toolkit.dequantize import patch_dequantization_on_save

    if not hasattr(base_model, "get_transformer_block_names"):
        raise ValueError(
            "The model to quantize must have a method `get_transformer_block_names`."
        )

    # patch the state dict method
    patch_dequantization_on_save(model_to_quantize)
    keep_nucleus_moe_on_cpu = bool(getattr(base_model.model_config, "low_vram", False))

    if base_model.model_config.accuracy_recovery_adapter is not None:
        from toolkit.config_modules import NetworkConfig
        from toolkit.lora_special import LoRASpecialNetwork

        # we need to load and quantize with an accuracy recovery adapter
        # todo handle hf repos
        load_lora_path = base_model.model_config.accuracy_recovery_adapter

        if not os.path.exists(load_lora_path):
            # not local file, grab from the hub

            path_split = load_lora_path.split("/")
            if len(path_split) > 3:
                raise ValueError(
                    "The accuracy recovery adapter path must be a local path or for a hf repo, 'username/repo_name/filename.safetensors'."
                )
            repo_id = f"{path_split[0]}/{path_split[1]}"
            print_acc(f"Grabbing lora from the hub: {load_lora_path}")
            new_lora_path = hf_hub_download(
                repo_id,
                filename=path_split[-1],
            )
            # replace the path
            load_lora_path = new_lora_path

        # build the lora config based on the lora weights
        lora_state_dict = load_file(load_lora_path)
        
        if hasattr(base_model, "convert_lora_weights_before_load"):
            lora_state_dict = base_model.convert_lora_weights_before_load(lora_state_dict)
        
        network_config = {
            "type": "lora",
            "network_kwargs": {"only_if_contains": []},
            "transformer_only": False,
        }
        first_key = list(lora_state_dict.keys())[0]
        first_weight = lora_state_dict[first_key]
        # if it starts with lycoris and includes lokr
        if first_key.startswith("lycoris") and any(
            "lokr" in key for key in lora_state_dict.keys()
        ):
            network_config["type"] = "lokr"
        
        network_kwargs = {}

        # find firse loraA weight
        if network_config["type"] == "lora":
            linear_dim = None
            for key, value in lora_state_dict.items():
                if "lora_A" in key:
                    linear_dim = int(value.shape[0])
                    break
            linear_alpha = linear_dim
            network_config["linear"] = linear_dim
            network_config["linear_alpha"] = linear_alpha

            # we build the keys to match every key
            only_if_contains = []
            for key in lora_state_dict.keys():
                contains_key = key.split(".lora_")[0]
                if contains_key not in only_if_contains:
                    only_if_contains.append(contains_key)

            network_kwargs["only_if_contains"] = only_if_contains
        elif network_config["type"] == "lokr":
            # find the factor
            largest_factor = 0
            for key, value in lora_state_dict.items():
                if "lokr_w1" in key:
                    factor = int(value.shape[0])
                    if factor > largest_factor:
                        largest_factor = factor
            network_config["lokr_full_rank"] = True
            network_config["lokr_factor"] = largest_factor

            only_if_contains = []
            for key in lora_state_dict.keys():
                if "lokr_w1" in key:
                    contains_key = key.split(".lokr_w1")[0]
                    contains_key = contains_key.replace("lycoris_", "")
                    if contains_key not in only_if_contains:
                        only_if_contains.append(contains_key)
            network_kwargs["only_if_contains"] = only_if_contains
        
        if hasattr(base_model, 'target_lora_modules'):
            network_kwargs['target_lin_modules'] = base_model.target_lora_modules

        # todo auto grab these
        # get dim and scale
        network_config = NetworkConfig(**network_config)

        network = LoRASpecialNetwork(
            text_encoder=None,
            unet=model_to_quantize,
            lora_dim=network_config.linear,
            multiplier=1.0,
            alpha=network_config.linear_alpha,
            # conv_lora_dim=self.network_config.conv,
            # conv_alpha=self.network_config.conv_alpha,
            train_unet=True,
            train_text_encoder=False,
            network_config=network_config,
            network_type=network_config.type,
            transformer_only=network_config.transformer_only,
            is_transformer=base_model.is_transformer,
            base_model=base_model,
            is_ara=True,
            **network_kwargs
        )
        network.apply_to(
            None, model_to_quantize, apply_text_encoder=False, apply_unet=True
        )
        network.force_to(base_model.device_torch, dtype=base_model.torch_dtype)
        network._update_torch_multiplier()
        network.load_weights(lora_state_dict)
        network.eval()
        network.is_active = True
        network.can_merge_in = False
        base_model.accuracy_recovery_adapter = network

        # quantize it
        lora_exclude_modules = []
        quantization_type = get_qtype(base_model.model_config.qtype)
        for lora_module in tqdm(network.unet_loras, desc="Attaching quantization"):
            # the lora has already hijacked the original module
            orig_module = lora_module.org_module[0]
            orig_module.to(base_model.torch_dtype)
            # make the params not require gradients
            for param in orig_module.parameters():
                param.requires_grad = False
            quantize(orig_module, weights=quantization_type)
            freeze(orig_module)
            module_name = lora_module.lora_name.replace('$$', '.').replace('transformer.', '')
            lora_exclude_modules.append(module_name)
            if base_model.model_config.low_vram:
                # move it back to cpu
                orig_module.to("cpu")
        pass
        # quantize additional layers
        print_acc(" - quantizing additional layers")
        quantization_type = get_qtype('uint8')
        quantize(
            model_to_quantize,
            weights=quantization_type,
            exclude=lora_exclude_modules
        )
        raw_expert_count = quantize_nucleus_moe_experts(
            model_to_quantize,
            keep_on_cpu=keep_nucleus_moe_on_cpu,
        )
        if raw_expert_count > 0:
            print_acc(f" - quantized {raw_expert_count} Nucleus MoE expert modules")
    else:
        # quantize model the original way without an accuracy recovery adapter
        # move and quantize only certain pieces at a time.
        quantization_type = get_qtype(base_model.model_config.qtype)
        # all_blocks = list(model_to_quantize.transformer_blocks)
        all_blocks: List[torch.nn.Module] = []
        transformer_block_names = base_model.get_transformer_block_names()
        for name in transformer_block_names:
            # name may be a dotted path for models that nest their blocks
            # (e.g. hidream_o1's "model.language_model.layers").
            block_list = model_to_quantize
            for part in name.split('.'):
                block_list = getattr(block_list, part, None)
                if block_list is None:
                    break
            if block_list is not None:
                all_blocks += list(block_list)
        base_model.print_and_status_update(
            f" - quantizing {len(all_blocks)} transformer blocks"
        )
        raw_expert_count = 0
        for block in tqdm(all_blocks):
            block.to(base_model.device_torch, dtype=base_model.torch_dtype, non_blocking=True)
            quantize(block, weights=quantization_type)
            raw_expert_count += quantize_nucleus_moe_experts(
                block,
                keep_on_cpu=keep_nucleus_moe_on_cpu,
            )
            freeze(block)
            block.to("cpu", non_blocking=True)

        # todo, on extras find a universal way to quantize them on device and move them back to their original
        # device without having to move the transformer blocks to the device first
        base_model.print_and_status_update(" - quantizing extras")
        # model_to_quantize.to(base_model.device_torch, dtype=base_model.torch_dtype)
        quantize(model_to_quantize, weights=quantization_type)
        raw_expert_count += quantize_nucleus_moe_experts(
            model_to_quantize,
            keep_on_cpu=keep_nucleus_moe_on_cpu,
        )
        if raw_expert_count > 0:
            print_acc(f" - quantized {raw_expert_count} Nucleus MoE expert modules")
        freeze(model_to_quantize)
