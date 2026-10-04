"""SliderSpace's configuration, semantic objective, and discovery utilities.

Independent implementation of the method in arXiv:2502.01639, adapted to flow
matching; no code from the upstream research-licensed training scripts.
"""

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from toolkit.flow_training import flow_clean_prediction


def parse_auto_sample_strengths(value):
    """Ordered finite CSV strengths; zero is the separate base comparison."""
    message = 'Auto sample strengths must be finite numbers separated by commas'
    if not isinstance(value, str):
        raise ValueError(message)
    strengths = []
    for token in value.split(','):
        token = token.strip()
        if not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', token, flags=re.ASCII):
            raise ValueError(message)
        strength = float(token)
        if not math.isfinite(strength):
            raise ValueError(message)
        if strength != 0 and strength not in strengths:
            strengths.append(strength)
    return strengths


@dataclass
class SliderSpaceConfig:
    discovery_mode: str = 'generated'
    discovery_datasets: list = field(default_factory=list)
    discovery_buckets: bool = False
    concept_prompts: list = field(default_factory=lambda: [''])
    num_directions: int = 4
    discovery_samples: int = 128
    resolution: int = 512
    discovery_steps: int = 20
    cfg_scale: float = 4.0
    negative_prompt: str = ''
    seed: int = 42
    feature_encoder: str = 'openai/clip-vit-base-patch32'
    feature_device: str = 'cpu'
    loss_weight: float = 1.0
    preview_direction: int = 1
    preview_strength: float = 1.0
    preview_auto: bool = False
    preview_auto_strengths: str = '-1, 1'

    @classmethod
    def parse(cls, value):
        if not isinstance(value, dict):
            raise ValueError('SliderSpace settings must be an object')
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown SliderSpace settings: {sorted(unknown)}')
        config = cls(**value)
        if type(config.preview_auto) is not bool:
            raise ValueError('SliderSpace preview_auto must be a boolean')
        parse_auto_sample_strengths(config.preview_auto_strengths)
        if type(config.discovery_buckets) is not bool:
            raise ValueError('SliderSpace discovery_buckets must be a boolean')
        if config.discovery_mode not in ('generated', 'provided', 'both'):
            raise ValueError('SliderSpace discovery mode must be generated, provided, or both')
        generates = config.discovery_mode != 'provided'
        if (not isinstance(config.concept_prompts, list)
                or any(not isinstance(prompt, str) for prompt in config.concept_prompts)
                or (generates and (not config.concept_prompts or any(not prompt.strip() for prompt in config.concept_prompts)))):
            raise ValueError('SliderSpace requires at least one nonempty concept prompt')
        config.concept_prompts = [prompt.strip() for prompt in config.concept_prompts]
        if not isinstance(config.discovery_datasets, list):
            raise ValueError('SliderSpace discovery datasets must be a list of image folders')
        for dataset in config.discovery_datasets:
            if (not isinstance(dataset, dict) or set(dataset) - {'folder_path', 'default_caption'}
                    or not isinstance(dataset.get('folder_path'), str)
                    or (config.discovery_mode != 'generated' and not dataset['folder_path'].strip())
                    or not isinstance(dataset.get('default_caption', ''), str)):
                raise ValueError('Each SliderSpace discovery dataset needs a folder_path and optional default_caption')
        if config.discovery_mode != 'generated' and not config.discovery_datasets:
            raise ValueError('SliderSpace provided/both discovery requires at least one image folder')
        for key in ('num_directions', 'discovery_samples', 'resolution', 'discovery_steps', 'seed', 'preview_direction'):
            if type(getattr(config, key)) is not int:
                raise ValueError(f'SliderSpace {key} must be an integer')
        if not 1 <= config.num_directions <= 64:
            raise ValueError('SliderSpace directions must be between 1 and 64')
        minimum = (max(config.num_directions + 1, len(config.concept_prompts)) if config.discovery_mode == 'generated'
                   else len(config.concept_prompts) if config.discovery_mode == 'both' else 0)
        if config.discovery_samples < minimum:
            raise ValueError('SliderSpace generated discovery images must cover the concept prompts '
                             '(and exceed directions in generated-only mode)')
        if config.resolution < 128 or config.resolution % 32:
            raise ValueError('SliderSpace resolution must be at least 128 pixels and a multiple of 32')
        if config.discovery_steps < 1 or not 0 <= config.seed <= 2**32 - 1:
            raise ValueError('SliderSpace generation steps must be positive and seed must be in [0, 4294967295]')
        for key in ('cfg_scale', 'loss_weight', 'preview_strength'):
            number = getattr(config, key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f'SliderSpace {key} must be finite')
        if config.cfg_scale < 1 or config.loss_weight <= 0:
            raise ValueError('SliderSpace CFG must be at least 1 and loss weight must be positive')
        if not 1 <= config.preview_direction <= config.num_directions:
            raise ValueError('SliderSpace preview direction is outside the trained direction range')
        if not isinstance(config.negative_prompt, str):
            raise ValueError('SliderSpace negative prompt must be text')
        if not isinstance(config.feature_encoder, str) or not config.feature_encoder.strip():
            raise ValueError('SliderSpace feature encoder name or local path is required')
        if config.feature_device not in ('cpu', 'cuda'):
            raise ValueError('SliderSpace feature device must be cpu or cuda')
        return config


def scan_discovery_images(config, divisibility=32):
    """Deterministic recursive scan; overlapping folders/symlink files deduplicate."""
    if config.discovery_mode == 'generated':
        return []
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.jxl', '.bmp', '.tif', '.tiff'}
    entries, seen = [], set()
    fallback = config.concept_prompts[0] if config.concept_prompts else ''
    for dataset in config.discovery_datasets:
        root = Path(dataset['folder_path']).expanduser()
        if not root.is_dir():
            raise ValueError(f'SliderSpace discovery image folder does not exist: {root}')
        images = sorted(path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in extensions)
        if not images:
            raise ValueError(f'SliderSpace discovery image folder has no supported images: {root}')
        for image in images:
            resolved = str(image.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            caption_path = image.with_suffix('.txt')
            try:
                caption = caption_path.read_text(encoding='utf-8-sig').strip() if caption_path.is_file() else ''
            except (OSError, UnicodeError) as error:
                raise ValueError(f'Cannot read SliderSpace caption: {caption_path}') from error
            caption = caption or dataset.get('default_caption', '').strip() or fallback
            entries.append({'source': 'provided', 'image': resolved, 'caption': caption,
                            'size': discovery_image_size(image, config.resolution, config.discovery_buckets, divisibility),
                            'caption_file': str(caption_path.absolute()),
                            'image_identity': _local_identity(str(image)),
                            'caption_identity': _local_identity(str(caption_path))})
    return entries


def verify_discovery_sources(entries):
    for entry in entries:
        if entry['source'] == 'provided' and (
                _local_identity(entry['image']) != entry['image_identity']
                or _local_identity(entry['caption_file']) != entry['caption_identity']):
            raise ValueError(f'SliderSpace discovery source changed after scanning: {entry["image"]}; restart the job')


def discovery_entries(config, provided=None):
    entries = list(scan_discovery_images(config) if provided is None else provided)
    if config.discovery_mode != 'provided':
        entries += [{'source': 'generated', 'caption': config.concept_prompts[i % len(config.concept_prompts)],
                     'seed': (config.seed + i) % 2**32} for i in range(config.discovery_samples)]
    if len(entries) <= config.num_directions:
        raise ValueError(f'SliderSpace needs at least {config.num_directions + 1} total discovery images; found {len(entries)}')
    return entries


def discovery_image_size(filename, resolution, buckets=False, divisibility=32):
    from PIL import Image
    from toolkit.buckets import get_bucket_for_image_size
    if not buckets:
        return [resolution, resolution]
    with Image.open(filename) as image:
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
    bucket = get_bucket_for_image_size(width, height, resolution, divisibility=divisibility)
    # The shared helper's fallback can exceed the budget on extremely thin images.
    width, height = bucket['width'], bucket['height']
    maximum = resolution * resolution
    if width * height > maximum:
        if width >= height:
            width = max(divisibility, maximum // height // divisibility * divisibility)
        else:
            height = max(divisibility, maximum // width // divisibility * divisibility)
    return [width, height]


def prepare_discovery_image(filename, resolution, buckets=False, divisibility=32):
    """Match image/latent/feature presentation: EXIF, Mitchell cover resize, center crop."""
    from PIL import Image, ImageOps
    from toolkit.image_resampling import resize_mitchell
    try:
        with Image.open(filename) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            width, height = discovery_image_size(filename, resolution, buckets, divisibility)
            scale = max(width / image.width, height / image.height)
            size = (max(width, math.ceil(image.width * scale)),
                    max(height, math.ceil(image.height * scale)))
            image = resize_mitchell(image, size)
            left, top = (image.width - width) // 2, (image.height - height) // 2
            return image.crop((left, top, left + width, top + height))
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot prepare SliderSpace discovery image: {filename}') from error


def semantic_direction_loss(adapted_features, base_features, direction, eps=1e-5):
    delta = adapted_features.float() - base_features.detach().float()
    return (1 - F.cosine_similarity(delta, direction.to(delta).reshape(1, -1), dim=-1, eps=eps)).mean()


def discover_directions(features, count):
    features = features.detach().to(device='cpu', dtype=torch.float64)
    if features.ndim != 2 or features.shape[0] <= count or features.shape[1] < count:
        raise ValueError('Too few discovery samples/feature dimensions for the requested PCA directions')
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite SliderSpace discovery features')
    centered = features - features.mean(dim=0)
    _, singular, vectors = torch.linalg.svd(centered, full_matrices=False)
    tolerance = singular[0] * max(centered.shape) * torch.finfo(torch.float32).eps
    if torch.count_nonzero(singular > tolerance) < count:
        raise ValueError('Discovery bank has too little visual variation for this many directions; '
                         'use more diverse prompts/images or fewer directions')
    directions = vectors[:count]
    pivot = directions.abs().argmax(dim=1)
    sign = directions[torch.arange(count), pivot].sign()
    directions = directions * sign[:, None]
    variance = singular.square() / (features.shape[0] - 1)
    ratio = variance[:count] / variance.sum()
    scores = centered @ directions.T
    return directions.float().contiguous(), ratio.float().contiguous(), scores.float().contiguous()


def _local_identity(value):
    if not isinstance(value, str):
        return None
    path = Path(value).expanduser()
    if path.is_file():
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    if path.is_dir():
        return [_local_identity(str(child)) for child in sorted(path.rglob('*'))
                if child.is_file() and child.suffix in ('.safetensors', '.bin', '.json', '.yaml')]
    return None


def discovery_signature(config, model, provided=None):
    from toolkit.training_capabilities import FLOW_TRAINING_MODELS, cfg_reference_mode
    capability = FLOW_TRAINING_MODELS.get(model.get('arch'))
    divisibility = capability.bucket_divisibility if capability else 32
    settings = asdict(config)
    for key in ('preview_direction', 'preview_strength', 'preview_auto', 'preview_auto_strengths', 'loss_weight'):
        settings.pop(key)
    if config.discovery_mode == 'generated':
        # Preserve signatures/resume compatibility for existing generated-only jobs.
        settings.pop('discovery_mode')
        settings.pop('discovery_datasets')
        settings.pop('discovery_buckets')
    identity = {}
    for key, value in model.items():
        local = _local_identity(value)
        if local is not None:
            identity[key] = local
    identity['feature_encoder'] = _local_identity(config.feature_encoder)
    payload = {'version': 1, 'sliderspace': settings, 'model': model, 'local_identity': identity}
    if config.discovery_mode != 'generated':
        from toolkit.image_resampling import MITCHELL_RESIZE_VERSION
        payload.update(version=2, provided=scan_discovery_images(config, divisibility) if provided is None else provided,
                       presentation=f'exif_rgb_center_cover_{MITCHELL_RESIZE_VERSION}',
                       feature_presentation='mean_letterbox_v1' if config.discovery_buckets else 'stretch_v1')
    if model.get('arch') in ('krea2', 'anima', 'ideogram4'):
        from toolkit.flow_training import FlowTrainingProfile
        payload.update(version=3, flow_profile=FlowTrainingProfile.from_model_config(model).identity(),
                       cfg_reference=cfg_reference_mode(model), bucket_divisibility=divisibility,
                       latent_layout='ideogram_patchified_norm_v1' if model['arch'] == 'ideogram4' else 'bchw_v1')
        if model['arch'] == 'ideogram4':
            payload['caption_digest'] = _local_identity(str(Path(__file__).parent / 'ideogram_caption.py'))
            payload['latent_norm'] = _local_identity(str(Path(__file__).parent.parent /
                'extensions_built_in/diffusion_models/ideogram4/src/latent_norm.py'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def atomic_json(filename, value):
    temporary = str(filename) + '.tmp'
    with open(temporary, 'w', encoding='utf8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temporary, filename)


class SemanticEncoder(nn.Module):
    """Frozen CLIP vision backbone; differentiable preprocessing, no PIL in loss."""
    def __init__(self, name_or_path, device='cpu', preserve_aspect_ratio=False):
        super().__init__()
        self.preserve_aspect_ratio = preserve_aspect_ratio
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('SliderSpace feature_device=cuda requires a visible CUDA GPU')
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        self.model = CLIPVisionModelWithProjection.from_pretrained(name_or_path).to(device, dtype=torch.float32)
        self.model.eval().requires_grad_(False)
        processor = CLIPImageProcessor.from_pretrained(name_or_path)
        from toolkit.flow_cache_identity import resolved_semantic_components
        self.cache_identity = resolved_semantic_components(self.model, processor, name_or_path)
        self.image_size = self.model.config.image_size
        self.register_buffer('mean', torch.tensor(processor.image_mean, device=device).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(processor.image_std, device=device).reshape(1, 3, 1, 1))

    def _encode(self, pixels):
        height, width = pixels.shape[-2:]
        if getattr(self, 'preserve_aspect_ratio', False):
            scale = self.image_size / max(height, width)
            size = (max(1, round(height * scale)), max(1, round(width * scale)))
        else:
            size = (self.image_size, self.image_size)
        pixels = F.interpolate(pixels[:, :3].float(), size=size,
                               mode='bicubic', align_corners=False, antialias=True)
        pixels = pixels.to(self.mean.device)
        normalized = ((pixels + 1) * 0.5 - self.mean) / self.std
        # Mean-color padding is zero in normalized CLIP space. The full image
        # remains visible, with the same differentiable transform for PCA/loss.
        left, top = (self.image_size - size[1]) // 2, (self.image_size - size[0]) // 2
        normalized = F.pad(normalized, (left, self.image_size - size[1] - left,
                                       top, self.image_size - size[0] - top))
        features = self.model(pixel_values=normalized).image_embeds
        return F.normalize(features.float(), dim=-1)

    def forward(self, pixels):
        if torch.is_grad_enabled() and pixels.requires_grad:
            return checkpoint(self._encode, pixels, use_reentrant=False)
        return self._encode(pixels)
