"""Model-specific boundaries for shared flow objectives, not model internals.

All prediction wrappers expose toolkit velocity (noise - clean) and BCHW latents.
Ideogram's model-time reversal/negation must remain ONLY inside its wrapper.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import torch

from toolkit.training_capabilities import FLOW_TRAINING_MODELS, cfg_reference_mode


@dataclass(frozen=True)
class FlowTrainingProfile:
    arch: str
    fixed_mu: float | None = None
    min_resolution: int = 256
    max_resolution: int = 1280
    min_shift: float = .5
    max_shift: float = 1.15

    @classmethod
    def from_model_config(cls, model):
        arch = model.get('arch')
        if arch not in FLOW_TRAINING_MODELS:
            raise ValueError(f'Unsupported flow training architecture: {arch}')
        kwargs = model.get('model_kwargs') or {}
        if arch == 'krea2':
            value = cls(arch, kwargs.get('schedule_mu'), kwargs.get('schedule_min_res', 256),
                        kwargs.get('schedule_max_res', 1280), kwargs.get('schedule_y1', .5),
                        kwargs.get('schedule_y2', 1.15))
            if (type(value.min_resolution) is not int or type(value.max_resolution) is not int
                    or value.min_resolution < 16 or value.max_resolution <= value.min_resolution
                    or value.min_resolution % 16 or value.max_resolution % 16):
                raise ValueError('Krea schedule resolutions must be increasing multiples of 16')
            for number in (value.fixed_mu, value.min_shift, value.max_shift):
                if number is not None and (isinstance(number, bool)
                        or not isinstance(number, (int, float)) or not math.isfinite(number)):
                    raise ValueError('Krea schedule shifts must be finite numbers')
            return value
        return cls(arch, math.log(3) if arch == 'anima' else 0 if arch == 'ideogram4' else None)

    def shift(self, height, width):
        if self.fixed_mu is not None:
            return self.fixed_mu
        if self.arch == 'qwen_image_2':
            # EXACT legacy Qwen arithmetic and raw latent-area interpretation.
            slope = (.9 - .5) / (8192 - 256)
            return height * width * slope + .5 - slope * 256
        if height % 2 or width % 2:
            raise ValueError('Krea latent dimensions must be divisible by its 2x2 patch size')
        tokens = (height // 2) * (width // 2)
        minimum = (self.min_resolution // 16) ** 2
        maximum = (self.max_resolution // 16) ** 2
        slope = (self.max_shift - self.min_shift) / (maximum - minimum)
        return slope * tokens + (self.min_shift - slope * minimum)

    def sample_time(self, clean, *, generator=None, device=None, minimum=0, maximum=999):
        if not 0 <= minimum < maximum <= 1000:
            raise ValueError('Flow training requires 0 <= minimum timestep < maximum <= 1000')
        if clean.ndim != 4:
            raise ValueError('Flow training requires BCHW still-image latents')
        normal = torch.randn(clean.shape[0], generator=generator, device=device, dtype=torch.float32)
        time = torch.sigmoid(normal + self.shift(*clean.shape[-2:]))
        low, high = minimum / 1000, maximum / 1000
        return low + (high - low) * time

    def identity(self):
        from dataclasses import asdict
        return {'version': 1, **asdict(self)}


def noised_flow_state(clean, time, noise):
    weight = time.float().reshape(-1, 1, 1, 1)
    return ((1 - weight) * clean.float() + weight * noise.float()).to(clean.dtype)


def trainer_flow_profile(trainer):
    """Resolve initialized trainers and lightweight extension/test fixtures."""
    profile = getattr(trainer, 'flow_profile', None)
    if profile is not None:
        return profile
    model = getattr(trainer, 'sd', None)
    config = getattr(model, 'model_config', None)
    return FlowTrainingProfile.from_model_config({'arch': getattr(model, 'arch', 'qwen_image_2'),
                                                  'model_kwargs': getattr(config, 'model_kwargs', {})})


def flow_training_metadata(trainer):
    """Versioned new-model metadata; legacy Qwen exports stay unchanged."""
    profile = trainer_flow_profile(trainer)
    if profile.arch == 'qwen_image_2':
        return {}
    import json
    return {'ss_flow_training_arch': profile.arch,
            'ss_flow_training_profile': json.dumps(profile.identity(), sort_keys=True),
            'ss_cfg_reference': getattr(trainer, 'cfg_reference', 'negative_prompt')}


def log_flow_training_profile(trainer):
    metadata = flow_training_metadata(trainer)
    if metadata:
        trainer.print(f"Resolved {metadata['ss_flow_training_arch']} training flow profile: "
                      f"{metadata['ss_flow_training_profile']}; CFG reference: {metadata['ss_cfg_reference']}")


def flow_clean_prediction(noisy, velocity, timesteps):
    weight = (timesteps.float() / 1000).reshape(-1, *([1] * (noisy.ndim - 1)))
    return noisy.float() - weight * velocity.float()


def image_only_prompt_embeds(model, positive):
    """True no-text conditioning; encoding an empty caption is NOT equivalent."""
    from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
    if getattr(model, 'arch', None) != 'ideogram4':
        raise ValueError('Image-only CFG conditioning is supported only for Ideogram 4')
    if not isinstance(positive, AdvancedPromptEmbeds):
        raise ValueError('Ideogram image-only conditioning requires AdvancedPromptEmbeds')
    return AdvancedPromptEmbeds(text_embeds=[features.new_empty((0, features.shape[-1]))
                                             for features in positive.text_embeds])


def resolve_cfg_reference(model, positive, negative):
    model_config = getattr(model, 'model_config', None)
    kwargs = getattr(model_config, 'model_kwargs', {}) or {}
    if cfg_reference_mode({'arch': getattr(model, 'arch', None), 'model_kwargs': kwargs}) == 'image_only':
        return image_only_prompt_embeds(model, positive)
    return negative


def predict_flow_branch(model, noisy, timesteps, embeds, batch=None):
    return model.predict_noise(latents=noisy, timestep=timesteps, conditional_embeddings=embeds,
                               unconditional_embeddings=None, guidance_scale=1.0,
                               guidance_embedding_scale=1.0, batch=batch).float()


def guided_flow_prediction(model, noisy, timesteps, positive, negative, cfg, batch=None):
    conditional = predict_flow_branch(model, noisy, timesteps, positive, batch)
    if cfg <= 1:
        return conditional
    reference = resolve_cfg_reference(model, positive, negative)
    if reference is None:
        raise ValueError('Guided flow prediction requires negative/reference conditioning at CFG > 1')
    unconditional = predict_flow_branch(model, noisy, timesteps, reference, batch)
    return unconditional + cfg * (conditional - unconditional)


def scoped_training_references(function):
    """Share detached edit inputs across specialized passes, not adapted K/V.

    Qwen/Anima/Ideogram and non-edit Krea keep their existing prediction path.
    The scope includes backward so checkpoint recomputation uses the same refs.
    """
    @wraps(function)
    def wrapped(self, batch, *args, **kwargs):
        model = getattr(self, 'sd', None)
        if getattr(model, 'arch', None) != 'krea2' or not getattr(model, 'is_edit', False):
            return function(self, batch, *args, **kwargs)
        with model.training_reference_context(batch):
            return function(self, batch, *args, **kwargs)
    return wrapped


@torch.no_grad()
def render_flow_bank_image(model, pipeline, positive, negative, *, width, height, steps, cfg, seed):
    """Render without preview file writes, duplicate TE encoding or CFG offsets."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    arch = getattr(model, 'arch', 'qwen_image_2')
    if arch == 'qwen_image_2':
        # Keep legacy discovery/practice behavior and lightweight test doubles.
        return pipeline(positive, unconditional_embeds=negative, height=height, width=width,
                        num_inference_steps=steps, guidance_scale=cfg, generator=generator)[0].convert('RGB')
    from toolkit.config_modules import GenerateImageConfig
    if arch == 'anima' and negative is None:
        # Anima's generation wrapper dereferences this object even without CFG.
        negative = positive
    config = GenerateImageConfig(width=width, height=height, num_inference_steps=steps,
                                  guidance_scale=cfg, seed=seed, output_path='bank.png')
    return model.generate_single_image(pipeline, config, positive, negative, generator, {}).convert('RGB')


@contextmanager
def training_decode_context(model):
    """Retain VAE weights until checkpoint backward is done, then restore placement.

Inference decode offload/tiling stays unchanged outside this scoped context.
"""
    vae = getattr(model, 'vae', None)
    depth = getattr(model, '_training_decode_depth', 0)
    previous_device = getattr(vae, 'device', None)
    model._training_decode_depth = depth + 1
    try:
        yield
    finally:
        model._training_decode_depth = depth
        if vae is not None and hasattr(vae, 'clear_cache'):
            vae.clear_cache()
        if not depth and vae is not None and previous_device is not None:
            vae.to(previous_device)


def decode_training_latents(model, latents):
    try:
        return model.decode_latents(latents)
    finally:
        vae = getattr(model, 'vae', None)
        if vae is not None and hasattr(vae, 'clear_cache'):
            vae.clear_cache()


def scoped_training_decode(function):
    @wraps(function)
    def wrapped(trainer, *args, **kwargs):
        if getattr(trainer.sd, 'arch', 'qwen_image_2') == 'qwen_image_2':
            return function(trainer, *args, **kwargs)
        with training_decode_context(trainer.sd):
            return function(trainer, *args, **kwargs)
    return wrapped
