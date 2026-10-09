"""Pinned, immutable Base/Instruct contract and fail-fast training validation."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

from .system_prompt import UNIFIED_SYSTEM_PROMPT_EN

TENCENT_REVISION = '2ec2c78bee7d4b94157341fba86c4c2c7b1858b2'
BASE_REVISION = '36f21fe74b65614451cc50ffd8a35a5f662dac70'
PEDRO_REVISION = '84ad3a3e2a54472e69e195269729f774b242d120'
CONDITIONING_VERSION = 'hunyuan3_joint_frozen_v2'
MAPPING_VERSION = 1
TARGETS = {
    'attention': ('self_attn.qkv_proj', 'self_attn.o_proj'),
    'attention_shared_mlp': ('self_attn.qkv_proj', 'self_attn.o_proj',
                            'mlp.shared_mlp.gate_and_up_proj', 'mlp.shared_mlp.down_proj'),
}


@dataclass(frozen=True)
class Variant:
    name: str
    repository: str
    sequence_template: str
    system_prompt: str
    guidance_scale: float
    supports_edit: bool
    steps: int = 50
    shift: float = 3.0
    max_positions: int = 22800


INSTRUCT = Variant('instruct', 'tencent/HunyuanImage-3.0-Instruct', 'instruct',
                   UNIFIED_SYSTEM_PROMPT_EN, 2.5, True)
BASE = Variant('base', 'tencent/HunyuanImage-3.0', 'pretrain', '', 5.0, False)


def pinned_config(variant=INSTRUCT):
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    config['model_type'] = variant.name
    config['sequence_template'] = variant.sequence_template
    config['cfg_distilled'] = False
    config['use_meanflow'] = False
    return config


def validate_variant(variant, source, metadata=None, keys=()):
    identity = str(source).lower()
    metadata = metadata or {}
    known = metadata.get('hunyuan_variant') or metadata.get('variant')
    if 'distil' in identity or any('guidance_emb.' in k or 'timestep_r_emb.' in k for k in keys):
        raise ValueError('HunyuanImage 3 Instruct-Distil/MeanFlow is unsupported for this trainer')
    named = ('instruct' if 'instruct' in identity else
             'base' if source == BASE.repository or 'hunyuan_image_3_base' in identity else None)
    if known and named and known != named:
        raise ValueError('Checkpoint variant metadata conflicts with its source identity')
    known = known or named
    if known and known != variant.name:
        raise ValueError(f'Checkpoint variant {known!r} conflicts with selected {variant.name!r}')


def validate_training(holder, process):
    train, network = process.train_config, process.network_config
    # DiffusionTrainer is the ordinary UI/API wrapper around SDTrainer; its
    # status/database hooks do not change the training objective. Keep this
    # explicit allowlist so specialized SDTrainer subclasses remain rejected.
    if process.get_conf('type') not in ('sd_trainer', 'diffusion_trainer'):
        raise ValueError('HunyuanImage 3 supports ordinary sd_trainer/diffusion_trainer image LoRA only')
    if network is None or network.type.lower() != 'lora':
        raise ValueError('HunyuanImage 3 supports LoRA training only')
    if train.train_text_encoder or not train.train_unet:
        raise ValueError('Train only Hunyuan attention/shared-MLP LoRA; text/vision/VAE training is unsupported')
    if train.batch_size != 1:
        raise ValueError('HunyuanImage 3 requires batch_size: 1; use gradient_accumulation for larger effective batches')
    if getattr(train, 'frequency_loss_enabled', False):
        raise ValueError('Hunyuan differentiable VAE/image-space losses require separate validation')
    for config in (getattr(process, 'sample_config', None), getattr(process, 'first_sample_config', None)):
        comfy = getattr(config, 'comfy', None)
        if comfy is not None and getattr(comfy, 'enabled', False):
            if (not getattr(holder.model_config, 'shared_weights', {}).get('enabled')
                    or getattr(comfy, 'provider', 'ordinary') != 'aitk_shared_models'):
                raise ValueError('Hunyuan Comfy previews require the negotiated aitk_shared_models provider; ordinary two-backbone previews remain unsupported')
            if comfy.run_in_background or comfy.send_prompts_as_batch or comfy.inference_lora:
                raise ValueError('Shared Hunyuan previews require synchronous single-image snapshots without a second inference adapter')
            from toolkit.comfy_sample import ComfyApiClient
            ComfyApiClient(comfy.api_url, comfy.timeout).require_shared_provider(holder.model_config.shared_weights)
    preset = holder.model_config.model_kwargs.get('lora_targets', 'attention')
    if preset not in TARGETS:
        raise ValueError(f'Unsupported Hunyuan LoRA target preset {preset!r}')
    targets = list(TARGETS[preset])
    explicit = network.network_kwargs.get('only_if_contains') or holder.model_config.only_if_contains
    if explicit is not None and set(explicit) != set(targets):
        raise ValueError('Explicit network.only_if_contains conflicts with model.model_kwargs.lora_targets')
    if (getattr(network, 'all_layers', False) or getattr(network, 'full_if_contains', None)
            or network.network_kwargs.get('all_layers') or network.network_kwargs.get('full_if_contains')):
        raise ValueError('Hunyuan routed experts, routers and full-weight adapters are unsupported')
    network.network_kwargs['only_if_contains'] = targets
    for dataset in process.dataset_configs:
        controls = getattr(dataset, 'control_path', None)
        if controls and not holder.variant.supports_edit:
            raise ValueError('HunyuanImage 3 Base does not support edit datasets')
        if controls and (getattr(dataset, 'flip_x', False) or getattr(dataset, 'flip_y', False)
                         or getattr(dataset, 'random_crop', False)):
            raise ValueError('Hunyuan edit random crops/flips require coordinated target/reference augmentation')
        if controls:
            if not holder.model_config.model_kwargs.get('vision_path'):
                raise ValueError('Edit datasets require model.model_kwargs.vision_path')
            folders = controls if isinstance(controls, list) else [controls]
            if len(folders) > 3:
                raise ValueError('Hunyuan edit supports up to three ordered reference folders')
            extensions = {'.jpg', '.jpeg', '.png', '.webp', '.jxl'}
            targets = Path(dataset.folder_path or dataset.dataset_path)
            if not targets.is_dir():
                raise ValueError('Hunyuan edit preflight requires a target image folder')
            # Match AiToolkitDataset's folder scan: UI .thumbs/.tmp trees and
            # hidden files are not training images. Cache tensors have no image
            # extension, and images directly in _controls are also excluded.
            for root, dirs, files in os.walk(targets):
                dirs[:] = [name for name in dirs if not name.startswith('.') and
                           (getattr(dataset, 'kto_label', None) is None or name != '_controls')]
                if Path(root).name == '_controls':
                    continue
                for name in files:
                    target = Path(root) / name
                    if name.startswith('.') or target.suffix.lower() not in extensions:
                        continue
                    for folder in folders:
                        source = Path(folder)
                        matches = [item for item in source.glob(target.stem + '.*')
                                   if item.is_file() and item.suffix.lower() in extensions]
                        if len(matches) != 1:
                            raise ValueError(f'Edit pair {target.name}: expected one source in {folder}, found {len(matches)}')
    holder.validate_model_config()
