"""Experimental flow-velocity adaptation of Diffusion-KTO's sigmoid utility.

This is objective math, not a standalone training mode. Model-native time/noising
and per-image velocity errors belong to the shared flow/model boundary. The
detached, clamped error-difference mean follows the official default estimator;
a pooled replay window computes it over ALL examples, not a mean of batch means.
"""
from dataclasses import dataclass
import math
import os

import torch


@dataclass(frozen=True)
class FlowKTOSettings:
    beta: float = 1.0
    liked_weight: float = 1.0
    disliked_weight: float = 1.0
    reference_estimator: str = 'batch_mean'
    score_window_size: int = 4

    @classmethod
    def parse(cls, values=None):
        values = {} if values is None else values
        if not isinstance(values, dict):
            raise ValueError('diffusion_kto must be a configuration object')
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown Diffusion-KTO settings: {", ".join(sorted(unknown))}')
        settings = cls(**values)
        for name in ('beta', 'liked_weight', 'disliked_weight'):
            value = getattr(settings, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'Diffusion-KTO {name} must be finite and positive')
        if settings.reference_estimator not in ('batch_mean', 'score_window'):
            raise ValueError('Diffusion-KTO reference_estimator must be batch_mean or score_window')
        if type(settings.score_window_size) is not int or settings.score_window_size < 1:
            raise ValueError('Diffusion-KTO score_window_size must be a positive integer')
        if settings.reference_estimator == 'score_window' and settings.score_window_size < 2:
            raise ValueError('Diffusion-KTO score_window requires at least two accumulation items')
        return settings


def kto_reference_point(error_differences):
    """Clamped detached empirical reference point, pooled by example count."""
    groups = [error_differences] if isinstance(error_differences, torch.Tensor) else list(error_differences)
    if not groups or any(group.numel() == 0 for group in groups):
        raise ValueError('Diffusion-KTO reference estimation requires nonempty scores')
    values = torch.cat([group.detach().float().reshape(-1) for group in groups])
    return values.mean().clamp_min(0).detach()


def flow_kto_terms(policy_error, reference_error, liked, settings, *, reference_point=None):
    """Return per-image utility loss, detached exact error coefficient and z.

    Liked labels are bool/0/1 (not signed strengths). A positive coefficient
    reduces policy error; a negative coefficient increases it. Reference errors
    and the reference point NEVER receive gradients. Score/replay callers detach
    the returned coefficient and backprop coefficient * recomputed policy_error.
    """
    if policy_error.ndim != 1 or policy_error.numel() == 0 or reference_error.shape != policy_error.shape:
        raise ValueError('Diffusion-KTO requires matching nonempty per-image error vectors')
    liked = torch.as_tensor(liked, device=policy_error.device)
    if liked.shape != policy_error.shape or not torch.all((liked == 0) | (liked == 1)):
        raise ValueError('Diffusion-KTO labels must be per-image boolean or 0/1 values')
    difference = reference_error.detach().float() - policy_error.float()
    if reference_point is None:
        reference_point = kto_reference_point(difference)
    point = torch.as_tensor(reference_point, device=policy_error.device, dtype=torch.float32).detach()
    if point.numel() != 1 or not torch.isfinite(point).all() or (point < 0).any():
        raise ValueError('Diffusion-KTO reference point must be a finite nonnegative scalar')
    point = point.reshape(())
    liked = liked.bool()
    sign = torch.where(liked, 1., -1.)
    weight = torch.where(liked, float(settings.liked_weight), float(settings.disliked_weight))
    utility = torch.sigmoid(sign * settings.beta * (difference - point))
    loss = weight * (1 - utility)
    coefficient = (weight * sign * settings.beta * utility * (1 - utility)).detach()
    return loss, coefficient, point


def validate_kto_config(config):
    """Objective restrictions and label counts, before any diffusion weights load.

    Returns settings and raw valid image counts (not multiplied by repeats).
    No pair matching, resolution equality or second/reward model is required.
    """
    from toolkit.training_capabilities import validate_specialized_model
    from PIL import Image
    validate_specialized_model(config, 'diffusion_kto')
    settings = FlowKTOSettings.parse(config.get('diffusion_kto'))
    model, network = config['model'], config['network']
    kwargs = model.get('model_kwargs') or {}
    auxiliary_paths = ('inference_lora_path', 'unconditional_lora_path') if model['arch'] == 'qwen_image_2' else (
        'assistant_lora_path', 'inference_lora_path', 'unconditional_lora_path')
    if any(model.get(key) for key in auxiliary_paths):
        raise ValueError('Diffusion-KTO supports frozen Qwen training helpers, but no other auxiliary adapters')
    if kwargs.get('edit') or kwargs.get('kv_cache'):
        raise ValueError('Diffusion-KTO initially supports text-to-image only; disable edit/kv_cache')
    if network.get('pretrained_lora_path'):
        raise ValueError('Diffusion-KTO cannot change its base reference with a pretrained LoRA')
    network_kwargs = network.get('network_kwargs') or {}
    if not isinstance(network_kwargs, dict):
        raise ValueError('Diffusion-KTO network_kwargs must be an object')
    if any(values.get(key) not in (None, 0, 0.) for values in (network, network_kwargs)
           for key in ('dropout', 'rank_dropout', 'module_dropout')):
        raise ValueError('Diffusion-KTO requires zero adapter dropout for exact replay')
    if network.get('all_layers') or any(network_kwargs.get(key) for key in ('full_train_in_out', 'full_if_contains', 'is_ara')):
        raise ValueError('Diffusion-KTO requires ordinary LoRA-only parameters, not full-layer adapters')
    train = config.get('train') or {}
    if train.get('noise_scheduler') != 'flowmatch' or train.get('timestep_type', 'shift') != 'shift':
        raise ValueError('Diffusion-KTO requires flowmatch with model-native shifted times')
    if train.get('train_text_encoder') or train.get('train_unet') is False:
        raise ValueError('Diffusion-KTO trains only a transformer LoRA')
    if not train.get('cache_text_embeddings') or train.get('unload_text_encoder'):
        raise ValueError('Diffusion-KTO requires Cache Text Embeddings, not blank-prompt Unload TE')
    if train.get('loss_type', 'mse') != 'mse':
        raise ValueError('Diffusion-KTO uses unweighted per-image velocity MSE as its experimental surrogate')
    conflicts = ('do_cfg', 'do_random_cfg', 'do_guidance_loss', 'do_differential_guidance',
                 'diff_output_preservation', 'blank_prompt_preservation', 'do_prior_divergence',
                 'train_turbo', 'inverted_mask_prior', 'do_fft_loss', 'correct_pred_norm',
                 'learnable_snr_gos')
    if any(train.get(key) for key in conflicts) or train.get('frequency_loss_type', 'none') != 'none':
        raise ValueError('Diffusion-KTO cannot be combined with other training objectives')
    if any(train.get(key) not in (None, 0, 0.) for key in ('min_snr_gamma', 'snr_gamma', 'snr_weight', 'noise_offset')):
        raise ValueError('Diffusion-KTO initially uses the unweighted velocity-error surrogate without SNR/noise offsets')
    if any(config.get(key) for key in ('adapter', 'embedding', 'decorator')) or train.get('diffusion_feature_extractor_path'):
        raise ValueError('Diffusion-KTO does not support additional trainable adapters or feature losses')
    if (train.get('ema_config') or {}).get('use_ema'):
        raise ValueError('Diffusion-KTO does not support EMA')
    if train.get('gradient_accumulation_steps', 1) != 1:
        raise ValueError('Diffusion-KTO uses gradient_accumulation within one step, not gradient_accumulation_steps')
    accumulation = train.get('gradient_accumulation', 1)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError('Diffusion-KTO gradient_accumulation must be a positive integer')
    if settings.reference_estimator == 'score_window' and accumulation < settings.score_window_size:
        raise ValueError(f'Diffusion-KTO score_window requires train.gradient_accumulation >= {settings.score_window_size}')
    low, high = train.get('min_denoising_steps', 0), train.get('max_denoising_steps', 999)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (low, high)) or not 0 <= low < high <= 1000:
        raise ValueError('Diffusion-KTO requires 0 <= min timestep < max timestep <= 1000')
    datasets = config.get('datasets')
    if not isinstance(datasets, list) or not datasets:
        raise ValueError('Diffusion-KTO requires individually labeled image/caption datasets')
    counts = {'liked': 0, 'disliked': 0}
    for dataset in datasets:
        if not isinstance(dataset, dict) or dataset.get('kto_label') not in counts:
            raise ValueError('Each Diffusion-KTO dataset needs kto_label: liked or disliked')
        folder = dataset.get('folder_path')
        if not isinstance(folder, str) or not os.path.isdir(folder):
            raise ValueError(f'Diffusion-KTO dataset does not exist: {folder}')
        if not dataset.get('buckets', True) or dataset.get('num_frames', 1) != 1 or dataset.get('auto_frame_count'):
            raise ValueError('Diffusion-KTO requires bucketed still-image datasets')
        if not (dataset.get('cache_latents_to_disk') or dataset.get('cache_latents')):
            raise ValueError('Diffusion-KTO requires Cache Latents on every dataset')
        rejected = ('control_path', 'control_path_1', 'control_path_2', 'control_path_3', 'controls',
                    'control_from_same_folder', 'inpaint_path', 'unconditional_path', 'mask_path', 'alpha_mask',
                    'is_reg', 'prior_reg', 'random_scale', 'random_crop', 'augmentations', 'augments',
                    'shuffle_tokens', 'token_dropout_rate', 'random_triggers')
        if any(dataset.get(key) for key in rejected):
            raise ValueError('Diffusion-KTO needs fixed, unmasked captions/images without paired/edit/reg controls')
        if float(dataset.get('network_weight', 1)) != 1:
            raise ValueError('Diffusion-KTO requires dataset LoRA Weight = 1')
        weight = dataset.get('loss_multiplier', 1.)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('Diffusion-KTO dataset loss_multiplier must be finite and nonnegative')
        repeats = dataset.get('num_repeats', 1)
        if type(repeats) is not int or repeats < 1:
            raise ValueError('Diffusion-KTO dataset num_repeats must be a positive integer')
        dataset_count = 0
        for root, dirs, files in os.walk(folder):
            dirs[:] = [name for name in dirs if not name.startswith('.') and name != '_controls']
            for filename in files:
                # Same enabled formats as toolkit.data_loader.image_extensions.
                if filename.startswith('.') or not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.jxl')):
                    continue
                try:
                    with Image.open(os.path.join(root, filename)) as image:
                        image.verify()
                    dataset_count += 1
                except (OSError, ValueError):
                    continue
        if not dataset_count:
            raise ValueError(f'Diffusion-KTO dataset has no valid enabled images: {folder}')
        counts[dataset['kto_label']] += dataset_count
    return settings, counts
