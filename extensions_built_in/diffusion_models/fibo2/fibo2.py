"""FIBO2 Dev: native training with explicit Comfy/HF component paths and frozen VLM."""
import hashlib
import json
import os
from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from toolkit.accelerator import unwrap_model
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.assistant_lora import load_assistant_lora_from_path
from toolkit.basic import flush
from toolkit.flow_cache_identity import local_component_identity, _component_info
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.text_encoders.qwen3_vl import Qwen3VLTextEncoder
from toolkit.models.v2.vae.flux2_kl import convert_diffusers_state_dict
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler

from .src.conditioning import BUNDLE_LAYERS, TAP_LAYERS, flow_mu, hidden_state_bundles, prompt_inputs
from .src.pipeline import Fibo2Pipeline
from .src.text_images import load_font
from .src.transformer import Fibo2Transformer, adapter_strengths
from .src.vae import Fibo2AutoEncoder
from .src.adapters import canonical_adapter

BASE_REPO = 'briaai/Fibo-2.0-dev'
TE_REPO = 'Qwen/Qwen3-VL-4B-Instruct'


class Fibo2TextEncoder(Qwen3VLTextEncoder):
    aitk_config_repo = TE_REPO
    aitk_processor_repo = TE_REPO
    aitk_processor_subfolder = ''
    aitk_tokenizer_subfolder = ''


def resolve_sources(config):
    """Explicit component selection; an unrelated checkpoint never supplies TE defaults."""
    return {
        'transformer': config.name_or_path,
        'text_encoder': config.te_name_or_path or config.text_encoder_path or config.model_kwargs.get('text_encoder_path') or TE_REPO,
        'processor': config.model_kwargs.get('processor_path') or (
            getattr(config, 'extras_name_or_path', None) if getattr(config, 'extras_name_or_path', None) != config.name_or_path else None) or TE_REPO,
        'vae': config.vae_path,
    }


class Fibo2Model(BaseModel):
    arch = 'fibo2'
    lora_keys_use_comfy_prefix = True
    lora_accept_bare_transformer_keys = True
    unconditional_text_uses_control_images = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_flow_matching = self.is_transformer = True
        self.target_lora_modules = ['Fibo2Transformer']
        self.supports_model_paths = True
        self.use_old_lokr_format = False
        self.vae_scale_factor = 16
        self.latent_space_version = 'fibo2_flux2_packed_bn_v1'
        self.encode_control_in_text_embeddings = self.has_multiple_control_images = True
        self.use_raw_control_images = True
        self.cache_processed_control_text_embeddings = True
        self.caption_dropout_keeps_control_images = True
        self.is_edit = True  # one model; actual references determine whether a prediction edits
        self.supports_quantized_text_encoder_cache = True
        self.generation_cpu_offload_modules = {'text_encoder', 'vae'}
        self.max_text_length = int(self.model_config.model_kwargs.get('max_text_length', 4096))
        self.font_paths = tuple(self.model_config.model_kwargs.get('glyph_font_paths', ()))
        self.fonts = ()
        self._sources = resolve_sources(self.model_config)

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, use_dynamic_shifting=True,
            base_image_seq_len=256, max_image_seq_len=4096, base_shift=.5, max_shift=1.15)

    def get_bucket_divisibility(self):
        return 16

    @property
    def text_embedding_uses_target_size(self):
        return True

    def training_preflight(self, process):
        self._validate_sources()
        if getattr(self.model_config, 'shared_weights', {}).get('enabled'):
            raise ValueError('FIBO2 does not yet support the Hunyuan immutable shared-weight handoff')
        name = str(self.model_config.name_or_path).lower()
        if 'turbo' in name or self.model_config.model_kwargs.get('turbo') or self.model_config.model_kwargs.get('is_distilled'):
            raise ValueError('FIBO2 training requires the non-distilled Dev base, not Edit Turbo')
        if process.is_fine_tuning and self.model_config.quantize:
            raise ValueError('FIBO2 full finetuning requires unquantized trainable base weights')
        if process.is_fine_tuning and os.path.isfile(self._sources['transformer']):
            from safetensors import safe_open
            with safe_open(self._sources['transformer'], framework='pt', device='cpu') as header:
                if any(key.endswith('.comfy_quant') for key in header.keys()):
                    raise ValueError('FIBO2 full finetuning requires a floating-point checkpoint, not prequantized weights')
        if process.train_config.train_text_encoder:
            raise ValueError('FIBO2 training uses a frozen Qwen3-VL text encoder')

    def _validate_sources(self):
        mc = self.model_config
        ara = mc.accuracy_recovery_adapter or '|' in (mc.qtype or '') or '|' in (mc.qtype_te or '')
        if mc.cache_quantized_models and ara:
            raise ValueError('FIBO2 persistent quantized caching cannot be combined with an accuracy-recovery adapter; disable Cache Quantized Models')
        if not self._sources['vae']:
            raise ValueError('Select the FLUX.2 VAE in model.vae_path for FIBO2')
        paths = {**self._sources, 'training helper': self.model_config.assistant_lora_path,
                 'inference adapter': self.model_config.inference_lora_path}
        for role, source in paths.items():
            if not source:
                continue
            local = os.path.expanduser(source)
            if (os.path.isabs(local) or source.startswith(('./', '../', '~'))
                    or (source.endswith('.safetensors') and source.count('/') < 2)) and not os.path.exists(local):
                raise FileNotFoundError(f'FIBO2 {role} component not found: {source}')
        for source in self.font_paths:
            if not os.path.isfile(os.path.expanduser(source)):
                raise FileNotFoundError(f'FIBO2 glyph font not found: {source}')

    def _resolve_sources(self):
        """Resolve snapshots/files before quant-cache lookup, so moving Hub refs cannot reuse stale modules."""
        from huggingface_hub import snapshot_download
        from extensions_built_in.diffusion_models.moss_image.src.checkpoints import resolve_hub_file
        for role, source in list(self._sources.items()):
            source = os.path.expanduser(source)
            if os.path.exists(source):
                self._sources[role] = os.path.realpath(source)
            elif source.endswith('.safetensors'):
                self._sources[role] = resolve_hub_file(source, hf_token=os.getenv('HF_TOKEN'))
            elif role == 'transformer':
                self._sources[role] = snapshot_download(source, allow_patterns=['transformer/*.safetensors', 'transformer/*.json'])
            elif role == 'vae':
                self._sources[role] = snapshot_download(source, allow_patterns=['vae/*.safetensors', 'vae/*.json', '*.safetensors', '*.json'])
            else:
                patterns = ['*.json', '*.txt', '*.model', '*.jinja', 'processor/*']
                if role == 'text_encoder':
                    patterns += ['*.safetensors']
                self._sources[role] = snapshot_download(source, allow_patterns=patterns)

    def _load_transformer(self):
        return Fibo2Transformer.load_model(self._sources['transformer'], dtype=self.torch_dtype,
            qtype=self.model_config.qtype if self.model_config.quantize else None, quantize_on_load=False)

    def _load_text_encoder(self):
        source = self._sources['text_encoder']
        if os.path.isdir(os.path.join(source, 'text_encoder')):
            source = os.path.join(source, 'text_encoder')
        return Fibo2TextEncoder.load_model(source, dtype=self.torch_dtype,
            config_path=TE_REPO if source.endswith('.safetensors') else None,
            subfolder='', quantize_on_load=False,
            qtype=self.model_config.qtype_te if self.model_config.quantize_te else None)

    def _load_processor(self):
        from transformers import Qwen3VLProcessor
        source = self._sources['processor']
        if os.path.isdir(os.path.join(source, 'processor')):
            source = os.path.join(source, 'processor')
        processor = Qwen3VLProcessor.from_pretrained(source)
        processor.aitk_load_source = {'path': source}
        return processor

    def _load_vae(self):
        source = self._sources['vae']
        if not source:
            raise ValueError('Select the FLUX.2 VAE in model.vae_path for FIBO2')
        if os.path.isdir(source):
            folder = os.path.join(source, 'vae') if os.path.isdir(os.path.join(source, 'vae')) else source
            files = [os.path.join(folder, name) for name in sorted(os.listdir(folder)) if name.endswith('.safetensors')]
            state = {}
            for filename in files:
                for key, value in load_file(filename).items():
                    if key in state:
                        raise ValueError(f'Duplicate FIBO2 VAE tensor: {key}')
                    state[key] = value
            if not state:
                raise FileNotFoundError(f'No FIBO2 VAE safetensors in {folder}')
        else:
            files = [source]
        if not os.path.isfile(source) and not os.path.isdir(source):
            from extensions_built_in.diffusion_models.moss_image.src.checkpoints import resolve_hub_file
            source = resolve_hub_file(source, hf_token=os.getenv('HF_TOKEN'))
            files = [source]
        if not os.path.isdir(source):
            state = load_file(source)
        if any(key.startswith('encoder.down_blocks.') for key in state):
            state = convert_diffusers_state_dict(state)
        vae = Fibo2AutoEncoder.load_from_state_dict(state, self.vae_torch_dtype).eval().requires_grad_(False)
        vae.aitk_load_source = {'path': os.path.realpath(source)}
        vae.aitk_load_files = [os.path.realpath(filename) for filename in files]
        return vae

    def load_model(self):
        self._validate_sources()
        self._resolve_sources()
        self.print_and_status_update('Loading FIBO2 transformer')
        qtype = self.model_config.qtype if self.model_config.quantize else None
        cache = self.get_quantized_module_cache_path('transformer', qtype or 'none',
            source_ref=self._sources['transformer'], extra_cache_key={
                'layout': 'fibo2_comfy_v1', 'local': local_component_identity(self._sources['transformer'])}) if qtype else None
        self.model = self.load_quantized_module_cache(cache, 'transformer', strict=True)
        fresh = self.model is None
        if fresh:
            self.model = self._load_transformer()
        self.text_encoder = []
        self.pipeline = Fibo2Pipeline(self)
        placement = self.component_load_kwargs('transformer')
        helper = self.model_config.assistant_lora_path
        offload = placement.pop('offload', 0.)
        final_device = placement['device']
        if offload or cache:
            placement['device'] = 'cpu'
        self.model.aitk_post_load(offload=0., **placement)
        if fresh:
            self.save_quantized_module_cache(self.model, cache, 'transformer', strict=True)
        if helper:
            if not os.path.isfile(helper):
                from extensions_built_in.diffusion_models.moss_image.src.checkpoints import resolve_hub_file
                helper = resolve_hub_file(helper, hf_token=os.getenv('HF_TOKEN'))
            self.assistant_lora = load_assistant_lora_from_path(helper, self, strict=True)
            self._resolved_helper_path = helper
            self.invert_assistant_lora = False
        inference = self.model_config.inference_lora_path
        if inference:
            if not os.path.isfile(inference):
                from extensions_built_in.diffusion_models.moss_image.src.checkpoints import resolve_hub_file
                inference = resolve_hub_file(inference, hf_token=os.getenv('HF_TOKEN'))
            self.inference_lora_network = load_assistant_lora_from_path(inference, self, strict=True)
            self.inference_lora_network.is_active = False
            self.inference_lora_network.force_to('cpu', self.torch_dtype)
        if offload:
            from toolkit.memory_management import MemoryManager
            MemoryManager.attach(self.model, self.device_torch, offload_percent=offload,
                ignore_modules=[self.model.perceiver.latent_queries])
        elif self.model.device != torch.device(final_device):
            self.model.to(final_device)
        flush()
        self.print_and_status_update('Loading FIBO2 Qwen3-VL-4B encoder')
        te_type = self.model_config.qtype_te if self.model_config.quantize_te else None
        te_cache = self.get_quantized_module_cache_path('text_encoder', te_type or 'none',
            source_ref=self._sources['text_encoder'], extra_cache_key={
                'format': 'fibo2_qwen3vl4b_v1', 'local': local_component_identity(self._sources['text_encoder'])}) if te_type else None
        self.text_encoder = self.load_quantized_module_cache(te_cache, 'text_encoder', strict=True)
        fresh_te = self.text_encoder is None
        if fresh_te:
            self.text_encoder = self._load_text_encoder()
        self.text_encoder.eval().requires_grad_(False)
        if fresh_te and te_cache:
            parking = self.component_load_kwargs('te')
            parking.update(device='cpu', offload=0.)
            self.text_encoder.aitk_post_load(**parking)
            self.save_quantized_module_cache(self.text_encoder, te_cache, 'text_encoder', strict=True)
        self.text_encoder.patch_vision_patch_embed()
        self.text_encoder.aitk_post_load(**self.component_load_kwargs('te'))
        self.processor = self._load_processor()
        self.tokenizer = self.processor.tokenizer
        self.fonts = tuple(load_font(source) for source in self.font_paths)
        self.vae = self._load_vae().to(self.vae_device_torch)
        self.vae.enable_gradient_checkpointing()
        self.noise_scheduler = self.get_train_scheduler()
        self._cache_identity = self._component_identity()
        self._latent_cache_version = self._make_latent_space_version()
        self.print_and_status_update('FIBO2 model loaded')

    def _component_identity(self):
        import PIL
        return hashlib.sha256(json.dumps({'version': 1,
            'sources': {name: value for name, value in self._sources.items() if name in ('text_encoder', 'processor')},
            'files': {name: local_component_identity(value) for name, value in self._sources.items() if name in ('text_encoder', 'processor')},
            'loaded': {name: _component_info(getattr(self, name, None)) for name in ('text_encoder', 'processor', 'tokenizer')},
            'processor_config': getattr(getattr(self, 'processor', None), 'image_processor', None).to_dict() if getattr(getattr(self, 'processor', None), 'image_processor', None) is not None else None,
            'chat_template': getattr(getattr(self, 'tokenizer', None), 'chat_template', None),
            'layers': BUNDLE_LAYERS, 'fonts': [local_component_identity(value) for value in self.font_paths],
            'pillow': PIL.__version__, 'glyph_code': local_component_identity(os.path.join(os.path.dirname(__file__), 'src', 'text_images.py')),
            'presentation_code': [local_component_identity(os.path.join(os.path.dirname(__file__), 'src', name))
                                  for name in ('conditioning.py', 'reference.py')],
            'max_length': self.max_text_length, 'dtype': str(self.torch_dtype),
            'qtype_te': self.model_config.qtype_te if self.model_config.quantize_te else None,
            'conditioning': 'fibo2_v1_null_dual_glyph_rgb_exif_ref768'}, sort_keys=True).encode()).hexdigest()

    def get_text_embedding_space_version(self):
        identity = getattr(self, '_cache_identity', None)
        return 'fibo2_text_v1_' + (identity if identity is not None else self._component_identity())

    def get_latent_space_version(self):
        return getattr(self, '_latent_cache_version', None) or self._make_latent_space_version()

    def _make_latent_space_version(self):
        identity = {'layout': self.latent_space_version, 'vae': local_component_identity(self._sources['vae']),
                    'loaded': _component_info(getattr(self, 'vae', None)),
                    'dtype': str(self.vae_torch_dtype)}
        return 'fibo2_latents_v1_' + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def _pil(image):
        if isinstance(image, Image.Image):
            from PIL import ImageOps
            return ImageOps.exif_transpose(image).convert('RGB')
        image = image.detach().float().cpu()
        if image.ndim == 4:
            image = image[0]
        return Image.fromarray((image[:3].permute(1, 2, 0).clamp(0, 1).numpy()*255).round().astype('uint8'))

    def _encode_text(self, prompt, images, text_images, *, is_unconditional=False):
        text, pictures = prompt_inputs(prompt, images, text_images=text_images, fonts=self.fonts,
                                      is_unconditional=is_unconditional)
        inputs = self.processor(text=[text], images=pictures or None, return_tensors='pt', padding=False)
        if inputs['input_ids'].shape[-1] > self.max_text_length:
            raise ValueError('FIBO2 caption/reference tokens exceed max_text_length; shorten the caption or references')
        inputs = {key: value.to(self.device_torch) for key, value in inputs.items()}
        captured, handles = {}, []
        layers = self.text_encoder.model.language_model.layers
        for index in TAP_LAYERS:
            def capture(_module, _args, output, index=index):
                captured[index] = output[0] if isinstance(output, tuple) else output
            handles.append(layers[index-1].register_forward_hook(capture))
        try:
            # Base VLM only: never materialize full vocabulary logits.
            self.text_encoder.model(**inputs, use_cache=False)
            return hidden_state_bundles(captured)[0].to(self.torch_dtype)
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def get_prompt_embeds(self, prompt, control_images=None, target_size=None, is_unconditional=False):
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if self.text_encoder.device != self.device_torch:
            self.text_encoder.to(self.device_torch)
        if control_images is None or is_unconditional:
            controls = [[] for _ in prompts]
        elif isinstance(control_images, torch.Tensor):
            controls = [[image] for image in control_images]
        elif control_images and not isinstance(control_images[0], (list, tuple)):
            controls = [control_images]
        else:
            controls = control_images
        if len(controls) == 1 and len(prompts) > 1:
            controls = controls * len(prompts)
        if len(controls) != len(prompts):
            raise ValueError('FIBO2 prompts and reference image sets must have matching lengths')
        positive, base = [], []
        from .src.reference import prepare_references
        for text, images in zip(prompts, controls):
            images = prepare_references([self._pil(image) for image in (images or [])], target_size)
            states = self._encode_text(text, images, not is_unconditional, is_unconditional=is_unconditional)
            positive.append(states)
            # Edits use glyphs at every sigma; T2I selects both variants.
            if not is_unconditional and not images and 'text_render' in text:
                base.append(self._encode_text(text, [], False))
            else:
                base.append(states)
        return AdvancedPromptEmbeds(text_embeds=positive, base_text_embeds=base)

    def encode_unconditional_prompt(self, prompt, prompt2=None, **kwargs):
        return self.get_prompt_embeds(prompt, control_images=None, target_size=kwargs.get('target_size'), is_unconditional=True)

    def predict_embeddings(self, latents, timesteps, embeds, refs=None):
        outputs = []
        batch = latents.shape[0]
        time = timesteps.float().reshape(-1).expand(batch)
        for index in range(batch):
            key = 'base_text_embeds' if time[index] > 900 and 'base_text_embeds' in embeds else 'text_embeds'
            values = embeds[key]
            item = values[index if len(values) > 1 else 0].to(self.device_torch, self.torch_dtype)
            context = item[None]
            references = None if refs is None else [refs[index]]
            states = []
            networks = []
            for network in (self.network, getattr(self, 'assistant_lora', None)):
                if network is not None:
                    network = unwrap_model(network)
                    networks.extend(list(network.directions) if hasattr(network, 'directions') else [network])
            for network in networks:
                if network is None or not getattr(network, 'is_active', False):
                    continue
                network = unwrap_model(network)
                strengths = network.torch_multiplier
                if strengths is not None and strengths.numel() > 1:
                    if strengths.ndim != 1 or strengths.shape[0] != batch:
                        raise ValueError('FIBO2 per-example adapter strengths must match the prediction batch')
                    selected = strengths[index:index+1]
                    scalar = float(selected.item())
                    states.append((network, scalar, selected))
            previous = getattr(self.transformer, '_adapter_replay_states', ())
            self.transformer._adapter_replay_states = tuple(states)
            try:
                with adapter_strengths(states):
                    outputs.append(self.transformer(latents[index:index+1].to(self.device_torch, self.torch_dtype),
                        time[index:index+1].to(self.device_torch)/1000, context, ref_latents=references))
            finally:
                self.transformer._adapter_replay_states = previous
        return torch.cat(outputs)

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, batch=None, **kwargs):
        refs = None
        if batch is not None:
            from toolkit.flow_training import batch_training_reference_latents
            previous = getattr(self, '_reference_target_size', None)
            self._reference_target_size = (latent_model_input.shape[-1]*16, latent_model_input.shape[-2]*16)
            try:
                refs = batch_training_reference_latents(self, batch, latent_model_input.shape[0],
                    latent_model_input.shape[-2]*latent_model_input.shape[-1]*256)
            finally:
                self._reference_target_size = previous
        return self.predict_embeddings(latent_model_input, timestep, text_embeddings, refs)

    @contextmanager
    def training_reference_context(self, batch):
        from toolkit.flow_training import training_reference_context
        with training_reference_context(self, batch):
            yield

    def _encode_ref_latents(self, controls, target_pixels=None, target_size=None):
        from torchvision.transforms.functional import to_tensor
        from .src.reference import prepare_references
        images = [to_tensor(image) for image in prepare_references([self._pil(image) for image in controls],
            target_size or getattr(self, '_reference_target_size', None))]
        return [self.encode_images([image*2-1], self.device_torch, self.torch_dtype)[0].detach() for image in images]

    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        device, dtype = device or self.vae_device_torch, dtype or self.vae_torch_dtype
        self.vae.to(device)
        return torch.cat([self.vae.encode(image[None].to(device, self.vae.dtype)).to(dtype) for image in image_list])

    def decode_latents(self, latents, device=None, dtype=None):
        device = device or self.vae_device_torch
        self.vae.to(device)
        latents = latents.to(device, self.vae.dtype)
        tiled = not torch.is_grad_enabled() and (self.model_config.low_vram or latents.shape[-2]*latents.shape[-1]*256 > 1024*1024)
        decoder = self.vae.decode_tiled if tiled else self.vae.decode
        # The gradient path retains residency through training_decode_context.
        return torch.cat([decoder(image[None]) for image in latents]).to(dtype or self.vae_torch_dtype)

    def get_loss_target(self, **kwargs):
        return (kwargs['noise']-kwargs['batch'].latents).detach()

    def get_model_has_grad(self):
        return any(value.requires_grad for value in self.model.parameters())

    def get_te_has_grad(self):
        return False

    def get_generation_pipeline(self):
        return Fibo2Pipeline(self)

    def generate_single_image(self, pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra):
        references = self.get_sample_control_image_paths(gen_config)
        refs = None
        if references:
            images = []
            for source in references:
                with Image.open(source) as image:
                    images.append(image.copy())
            refs = [self._encode_ref_latents(images, target_size=(gen_config.width, gen_config.height))]
        return pipeline(conditional_embeds, unconditional_embeds, height=gen_config.height, width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps, guidance_scale=gen_config.guidance_scale,
            generator=generator, latents=gen_config.latents, ref_latents=refs)[0]

    def save_model(self, output_path, meta, save_dtype):
        unwrap_model(self.model).save_model(output_path if output_path.endswith('.safetensors') else output_path+'.safetensors',
            dtype=save_dtype, metadata=get_meta_for_safetensors(meta, name=self.arch), quantized=False)

    def get_base_model_version(self):
        return self.arch

    def get_transformer_block_names(self):
        return Fibo2Transformer.get_transformer_block_names()

    def get_quantization_exclude_modules(self):
        return Fibo2Transformer.get_quantization_exclude_modules()

    def convert_lora_weights_before_load(self, state_dict):
        return canonical_adapter(super().convert_lora_weights_before_load(state_dict))

    def convert_assistant_lora_weights_before_load(self, state_dict):
        return self.convert_lora_weights_before_load(state_dict)
