"""HunyuanImage 3 Base and non-distilled Instruct through the standard trainer."""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from toolkit.models.base_model import BaseModel
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds
from toolkit.config_modules import GenerateImageConfig
from toolkit.samplers.custom_flowmatch_sampler import CustomFlowMatchEulerDiscreteScheduler
from .src.config import BASE, INSTRUCT, CONDITIONING_VERSION, TARGETS, TENCENT_REVISION, BASE_REVISION, pinned_config, validate_training, validate_variant


class HunyuanImage3Model(BaseModel):
    variant = INSTRUCT
    disable_dataloader_workers = True
    lora_keys_use_comfy_prefix = True
    lora_accept_bare_transformer_keys = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_flow_matching = self.is_transformer = True
        self.target_lora_modules = ['HunyuanImage3Attention']
        preset = self.model_config.model_kwargs.get('lora_targets', 'attention')
        if preset == 'attention_shared_mlp':
            # Explicit network filter installed by preflight excludes routed experts.
            self.target_lora_modules += ['HunyuanImage3MLP']
        self.vae_scale_factor = 16
        self.encode_control_in_text_embeddings = True
        self.has_multiple_control_images = True
        self.caption_dropout_keeps_control_images = True
        self.use_raw_control_images = True
        self.conditioning_component = None
        self._reference_cache = None
        self._release_vae_after_conditioning = False
        self.generation_cpu_offload_modules = {'vae', 'text_encoder'}
        self.validate_model_config()

    def validate_model_config(self):
        mc = self.model_config
        validate_variant(self.variant, mc.name_or_path)
        if os.path.isabs(mc.name_or_path) and not os.path.exists(mc.name_or_path):
            raise ValueError(f'Hunyuan backbone checkpoint does not exist: {mc.name_or_path}')
        if mc.model_kwargs.get('lora_targets', 'attention') not in TARGETS:
            raise ValueError('Hunyuan LoRA targets must be attention or attention_shared_mlp')
        if mc.quantize and mc.qtype not in ('convrot8', 'comfy_w4a8'):
            raise ValueError('Hunyuan requires qtype convrot8 or comfy_w4a8')
        if mc.quantize_te:
            raise ValueError('Hunyuan tokenizer/vision conditioning is not a separate quantized language backbone')
        if int(mc.model_kwargs.get('reference_max_pixels', 262144)) < 256:
            raise ValueError('reference_max_pixels must be at least 256')
        for label, source in [('VAE', mc.vae_path), ('vision', mc.model_kwargs.get('vision_path'))]:
            if source and os.path.isabs(source) and not os.path.exists(source):
                raise ValueError(f'Hunyuan {label} checkpoint does not exist: {source}')
        if os.path.isfile(mc.name_or_path) and not mc.vae_path:
            from safetensors import safe_open
            with safe_open(mc.name_or_path, framework='pt', device='cpu') as checkpoint:
                has_vae = any(key.startswith('vae.') for key in checkpoint.keys())
            if not has_vae:
                raise ValueError('Split Comfy checkpoints require model.vae_path; select hunyuan_image_3_vae_fp16.safetensors')
        assets, _ = self._conditioning_source()
        if os.path.isdir(assets) and not (Path(assets) / 'tokenizer.json').is_file():
            raise ValueError(f'Hunyuan conditioning assets lack tokenizer.json: {assets}')
        if self.device_torch.type == 'cuda' and torch.cuda.is_available():
            frozen_gib = {'comfy_w4a8': 44., 'convrot8': 77.}.get(mc.qtype if mc.quantize else None, 151.)
            fraction = mc.layer_offloading_transformer_percent if mc.layer_offloading else 0.
            available_gib = torch.cuda.get_device_properties(self.device_torch).total_memory / 2**30
            if frozen_gib * (1 - fraction) + 4 > available_gib:
                raise ValueError('Hunyuan frozen weights exceed this GPU budget; increase transformer layer offloading')

    def training_preflight(self, process):
        validate_training(self, process)
        self._release_vae_after_conditioning = (
            process.train_config.disable_sampling and bool(process.dataset_configs)
            and all(dataset.cache_latents or dataset.cache_latents_to_disk for dataset in process.dataset_configs))

    def get_adapter_metadata(self, network=None):
        preset = self.model_config.model_kwargs.get('lora_targets', 'attention')
        matched = [name for name, module in self.model.named_modules()
                   if name.endswith(TARGETS[preset])]
        return {'variant': self.variant.name, 'source': self.model.aitk_load_source,
                'preset': preset, 'modules': matched, 'conditioning_version': CONDITIONING_VERSION,
                'rank': None if network is None else network.lora_dim,
                'alpha': None if network is None else network.alpha,
                'qtype': getattr(self.model, 'aitk_qtype', None)}

    def validate_adapter_metadata(self, metadata):
        recorded = (metadata or {}).get('hunyuan_training')
        if recorded is None:
            raise ValueError('Hunyuan training resume requires variant/source/target metadata')
        if isinstance(recorded, str):
            recorded = json.loads(recorded)
        expected = self.get_adapter_metadata(self.network)
        for key in ('variant', 'source', 'preset', 'conditioning_version', 'rank', 'alpha', 'qtype'):
            if recorded.get(key) != expected.get(key):
                raise ValueError(f'Hunyuan adapter resume mismatch for {key}')

    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0, use_dynamic_shifting=False)

    def get_bucket_divisibility(self):
        return 16

    @property
    def text_embedding_uses_target_size(self):
        return False

    @property
    def text_embedding_space_version(self):
        return self.get_text_embedding_space_version()

    def get_text_embedding_space_version(self):
        mc = self.model_config
        assets, revision = self._conditioning_source()
        tokenizer_file = Path(assets) / 'tokenizer.json'
        sources = [mc.name_or_path, str(tokenizer_file) if tokenizer_file.is_file() else assets,
                   mc.vae_path, mc.model_kwargs.get('vision_path'), revision]
        identities = []
        for source in sources:
            if source and os.path.isfile(source):
                stat = os.stat(source)
                identities.append([os.path.realpath(source), stat.st_size, stat.st_mtime_ns])
            else:
                identities.append(source)
        payload = [CONDITIONING_VERSION, self.variant.name, self.variant.sequence_template,
                   self.variant.system_prompt, identities,
                   mc.model_kwargs.get('reference_max_pixels', 262144), 'rgb_white_alpha',
                   'aspect_bilinear_16grid', 'vae_mean', 'vae_fp32_autocast_fp16', .562679178327931]
        return CONDITIONING_VERSION + '_' + hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:24]

    def _conditioning_source(self):
        mc = self.model_config
        assets = mc.extras_name_or_path
        if assets == mc.name_or_path and not os.path.isdir(assets):
            assets = self.variant.repository
        revision = mc.model_kwargs.get('tokenizer_revision')
        if assets == INSTRUCT.repository and revision is None:
            revision = TENCENT_REVISION
        if assets == BASE.repository and revision is None:
            revision = BASE_REVISION
        return assets, revision

    def load_model(self):
        # Imports stay lazy: registering this model does not load Hunyuan runtimes.
        from toolkit.models.v2.diffusion_models.hunyuan_image_3 import HunyuanImage3Transformer
        from toolkit.models.v2.vae.hunyuan_image_3 import HunyuanImage3VAE
        from toolkit.models.v2.text_encoders.hunyuan_image_3 import HunyuanImage3Conditioning
        from .src.pipeline import HunyuanImage3Pipeline
        mc = self.model_config
        assets, asset_revision = self._conditioning_source()
        source = mc.name_or_path
        qtype = mc.qtype if mc.quantize else None
        cache_path = self.get_quantized_module_cache_path(
            'transformer', qtype or 'none', source_ref=source,
            extra_cache_key={'variant': self.variant.name, 'format': 1, 'mapping': 1,
                             'stat': None if not os.path.isfile(source) else [os.stat(source).st_size, os.stat(source).st_mtime_ns]})
        transformer = self.load_quantized_module_cache(cache_path, 'transformer')
        if transformer is None:
            self.print_and_status_update('Streaming HunyuanImage 3 weights')
            transformer = HunyuanImage3Transformer.load_model(
                source, dtype=self.torch_dtype, qtype=qtype, variant=self.variant,
                quantize_device=self.device_torch, config=pinned_config(self.variant))
            self.save_quantized_module_cache(transformer, cache_path, 'transformer')
        transformer.aitk_post_load(**self.component_load_kwargs('transformer'))
        if mc.layer_offloading:
            # Large embedding gathers stay on CPU and deliberately are not pinned.
            # A safetensors view of this table otherwise retains the mapping of the
            # entire 44–76 GiB checkpoint even after packed buffers were pinned.
            # Copy just the table once, allowing the original file mapping to close.
            table = transformer.model.wte.weight
            if table.device.type == 'cpu' and not table.is_pinned():
                table.data = table.detach().clone()
        self.print_and_status_update('Loading Hunyuan conditioning assets')
        conditioning = HunyuanImage3Conditioning.create(
            self.variant, assets, mc.model_kwargs.get('vision_path'),
            revision=asset_revision)
        conditioning.requires_grad_(False).eval()
        self.conditioning_component = conditioning
        self._conditioning_assets = assets
        self._conditioning_revision = asset_revision
        self.text_encoder = [conditioning]
        self.tokenizer = [conditioning.tokenizer]
        self.print_and_status_update('Loading Hunyuan image VAE')
        vae_source = mc.vae_path or source
        # Exact Tencent precision policy: fp32 VAE parameters, fp16 operation autocast.
        self.vae = HunyuanImage3VAE.load_model(vae_source, dtype=torch.float32,
                                             config=pinned_config(self.variant)['vae'], use_comfy_weights=False)
        self.vae.requires_grad_(False).eval()
        self.model = transformer
        self.noise_scheduler = self.get_train_scheduler()
        self.pipeline = HunyuanImage3Pipeline(self)
        self.print_and_status_update('HunyuanImage 3 loaded')

    @staticmethod
    def _normalize_controls(images, batch_size):
        if images is None or (isinstance(images, (list, tuple)) and len(images) == 0):
            return [[] for _ in range(batch_size)]
        if isinstance(images, torch.Tensor):
            if images.dim() == 3:
                images = [[images]]
            elif images.dim() == 4:
                images = [[sample] for sample in images]
            elif images.dim() == 5:
                images = [list(sample) for sample in images]
            else:
                raise ValueError('Unsupported reference tensor shape')
        elif images and not isinstance(images[0], (list, tuple)):
            images = [list(images)]
        if len(images) != batch_size:
            raise ValueError('Reference batch count differs from prompt batch')
        if any(len(sample) > 3 for sample in images):
            raise ValueError('Hunyuan edit supports at most three ordered references')
        # Never deduplicate by pixels: repeated reference slots are semantic.
        return images

    def _prepare_reference(self, image):
        if isinstance(image, Image.Image):
            if image.mode == 'RGBA':
                backdrop = Image.new('RGBA', image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(backdrop, image).convert('RGB')
            image = torch.from_numpy(np.array(image.convert('RGB'), dtype=np.float32) / 255).permute(2, 0, 1)
        if image.dim() == 4:
            image = image[0]
        image = image.detach().float().cpu()
        if image.shape[0] == 4:
            image = image[:3] * image[3:] + 1 - image[3:]
        if image.shape[0] != 3 or image.min() < 0 or image.max() > 1:
            raise ValueError('Hunyuan reference presentation requires RGB tensors in [0,1]')
        height, width = image.shape[-2:]
        budget = int(self.model_config.model_kwargs.get('reference_max_pixels', 262144))
        scale = min(1., math.sqrt(budget / (height * width)))
        h, w = max(16, int(height * scale) // 16 * 16), max(16, int(width * scale) // 16 * 16)
        while h * w > budget:
            if h >= w: h -= 16
            else: w -= 16
        return F.interpolate(image[None], size=(h, w), mode='bilinear', align_corners=False)[0]

    @torch.no_grad()
    def get_prompt_embeds(self, prompt, control_images=None, target_size=None):
        from .src.tokenizer import build_sequence
        prompts = [prompt] if isinstance(prompt, str) else prompt
        controls = self._normalize_controls(control_images, len(prompts))
        if any(controls) and not self.variant.supports_edit:
            raise ValueError('Base only supports text-to-image')
        target_size = target_size or (512, 512)
        component = self.conditioning_component
        if component is None:
            raise RuntimeError('Hunyuan conditioning component is not loaded')
        store = {'ids': [], 'text_embeds': [], 'reference_count': [], 'reference_grids': []}
        for slot in range(3):
            store[f'reference_latent_{slot}'] = []
            store[f'reference_vision_{slot}'] = []
        for caption, references in zip(prompts, controls):
            latents, visions, grids, sizes = [], [], [], []
            presentations = [self._prepare_reference(image) for image in references]
            identity = tuple((tuple(image.shape), hashlib.sha256(image.numpy().tobytes()).hexdigest()) for image in presentations)
            if presentations and self._reference_cache is not None and self._reference_cache[0] == identity:
                latents, visions, grids, sizes = self._reference_cache[1]
            elif presentations:
                # Reload a destroyed vision tower only on a frozen-feature miss.
                if component.vision_model is not None and next(component.vision_model.parameters()).is_meta:
                    from toolkit.models.v2.text_encoders.hunyuan_image_3 import HunyuanImage3Conditioning
                    replacement = HunyuanImage3Conditioning.create(
                        self.variant, self._conditioning_assets,
                        self.model_config.model_kwargs.get('vision_path'),
                        self._conditioning_revision)
                    component.vision_model = replacement.vision_model
                    component._placement = replacement._placement
                    self.text_encoder = [component]
                self._ensure_vae()
                vae_device = self.vae.device
                try:
                    for presented in presentations:
                        latent = self.encode_images([presented * 2 - 1], dtype=self.torch_dtype).detach().cpu()
                        pil = Image.fromarray((presented.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))
                        vision, grid = component.encode_reference(pil, self.device_torch, self.torch_dtype)
                        latents.append(latent); visions.append(vision); grids.append(grid[0]); sizes.append(tuple(presented.shape[-2:]))
                finally:
                    self.vae.to(vae_device)
                    if component.vision_model is not None:
                        component.vision_model.to('cpu')
                if presentations:
                    self._reference_cache = (identity, (latents, visions, grids, sizes))
            sequence = build_sequence(component.tokenizer, caption, (target_size[1], target_size[0]),
                self.variant.system_prompt, cfg_distilled=False, use_meanflow=False,
                sequence_template=self.variant.sequence_template, extra_rows=self.variant.supports_edit,
                reference_vit_padding=True, cond_images=list(zip(sizes, [g.tolist() for g in grids])))
            store['ids'].append(sequence['ids'])
            # The standard trainer derives batch count from this 2D tensor.
            # It carries real token IDs, whose embeddings belong to the shared backbone.
            store['text_embeds'].append(sequence['ids'].unsqueeze(-1).clone())
            store['reference_count'].append(torch.tensor(len(references), dtype=torch.long))
            store['reference_grids'].append(torch.stack(grids) if grids else torch.empty((0,2), dtype=torch.long))
            for slot in range(3):
                store[f'reference_latent_{slot}'].append(latents[slot] if slot < len(latents) else torch.empty(0))
                store[f'reference_vision_{slot}'].append(visions[slot] if slot < len(visions) else torch.empty(0))
        embeds = AdvancedPromptEmbeds(**store)
        embeds.frozen_dtype_keys = ['ids', 'text_embeds', 'reference_count', 'reference_grids']
        return embeds.detach()

    def condition_noisy_latents(self, latents, batch):
        return latents.detach()

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, batch=None, **kwargs):
        from .src.conditioning import rebuild_target_ids
        count = int(text_embeddings.reference_count[0])
        if len(text_embeddings.ids) != 1 or latent_model_input.shape[0] != 1:
            raise ValueError('Hunyuan currently requires microbatch one')
        dtype, device = self.torch_dtype, self.device_torch
        return self.model(latent_model_input.to(device, dtype), timestep.to(device),
            ids=rebuild_target_ids(text_embeddings.ids[0], latent_model_input.shape[-2:],
                                   self.tokenizer[0], self.variant).to(device),
            cond_latent=[text_embeddings[f'reference_latent_{i}'][0].to(device, dtype) for i in range(count)],
            cond_vit=[text_embeddings[f'reference_vision_{i}'][0].to(device, dtype) for i in range(count)],
            cond_vit_grid=text_embeddings.reference_grids[0].tolist())

    def get_loss_target(self, *args, **kwargs):
        return (kwargs['noise'] - kwargs['batch'].latents).detach()

    def get_model_has_grad(self): return False
    def get_te_has_grad(self): return False
    def get_transformer_block_names(self): return ['model.layers']

    @torch.no_grad()
    def encode_images(self, image_list, device=None, dtype=None):
        # The shared latent cacher may call with grad mode enabled. This VAE
        # is frozen and differentiable image losses are rejected in preflight,
        # so explicitly disable gradients before selecting its low-VRAM path.
        self._ensure_vae()
        device, dtype = device or self.vae_device_torch, dtype or self.torch_dtype
        self.vae.to(device=device, dtype=torch.float32)
        tiled = self.model_config.low_vram
        if tiled: self.vae.enable_spatial_tiling()
        try:
            result = []
            for image in image_list:
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=torch.device(device).type == 'cuda'):
                    latent = self.vae.encode(image[None].to(device, torch.float32)).latent_dist.mode()
                result.append((latent[:, :, 0] * self.vae.scaling_factor).to(dtype))
            return torch.cat(result)
        finally:
            if tiled: self.vae.disable_spatial_tiling()

    @torch.no_grad()
    def decode_latents(self, latents, device=None, dtype=None):
        self._ensure_vae()
        device = device or self.vae_device_torch
        self.vae.to(device=device, dtype=torch.float32)
        tiled = self.model_config.low_vram
        if tiled: self.vae.enable_spatial_tiling()
        try:
            decoded = []
            for latent in latents.split(1):
                with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=torch.device(device).type == 'cuda'):
                    image = self.vae.decode(latent.to(device, torch.float32).unsqueeze(2) / self.vae.scaling_factor).sample
                decoded.append(image[:, :, 0])
            return torch.cat(decoded)
        finally:
            if tiled: self.vae.disable_spatial_tiling()

    def get_generation_pipeline(self): return self.pipeline

    def _ensure_vae(self):
        if self.vae is None:
            from toolkit.models.v2.vae.hunyuan_image_3 import HunyuanImage3VAE
            self.vae = HunyuanImage3VAE.load_model(
                self.model_config.vae_path or self.model_config.name_or_path,
                dtype=torch.float32, config=pinned_config(self.variant)['vae'],
                use_comfy_weights=False)
            self.vae.requires_grad_(False).eval()
            if getattr(self, 'pipeline', None) is not None:
                self.pipeline.vae = self.vae

    def generate_images(self, *args, **kwargs):
        # The shared sampler restores placement on success. Also restore it on
        # decoding/conditioning failure, so the next optimizer step remains usable.
        unloaded_vae = self.vae is None
        multiplier = getattr(self.network, 'multiplier', None)
        active = getattr(self.network, 'is_active', None)
        rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        try:
            return super().generate_images(*args, **kwargs)
        finally:
            self.restore_device_state()
            if self.vae is not None:
                self.vae.to('cpu')
            component = self.conditioning_component
            if component is not None and component.vision_model is not None:
                if not next(component.vision_model.parameters()).is_meta:
                    component.vision_model.to('cpu')
            if self.network is not None:
                self.network.train()
                if self.network.is_merged_in:
                    self.network.merge_out()
                if multiplier is not None:
                    self.network.multiplier = multiplier
                if active is not None:
                    self.network.is_active = active
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng)
            if unloaded_vae and self.vae is not None:
                self.unload_vae_after_caching()

    def unload_vae_after_caching(self):
        from toolkit.memory_management import MemoryManager
        MemoryManager.free(self.vae)
        self.vae = None
        self.pipeline.vae = None
        MemoryManager.release_cached_memory()

    def on_text_encoder_unloaded(self):
        # Novel sample references may have reloaded the VAE after the target
        # latent cache was completed. Discard it once frozen conditioning ends.
        if self._release_vae_after_conditioning and self.vae is not None:
            self.unload_vae_after_caching()

    def get_sample_control_image_paths(self, gen_config):
        # ctrl_img_1 is the normalized legacy ctrl_img alias, not another slot.
        return [path for path in (gen_config.ctrl_img_1 or gen_config.ctrl_img,
                                  gen_config.ctrl_img_2, gen_config.ctrl_img_3) if path is not None]

    def decode_to_images(self, latents):
        return [Image.fromarray(((image.detach().float().clamp(-1, 1) + 1) / 2
                                * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy())
                for image in self.decode_latents(latents)]

    def generate_single_image(self, pipeline, gen_config: GenerateImageConfig, conditional_embeds,
                              unconditional_embeds, generator, extra):
        width = gen_config.width // 16 * 16
        height = gen_config.height // 16 * 16
        return pipeline(conditional_embeds=conditional_embeds, unconditional_embeds=unconditional_embeds,
                        width=width, height=height, num_inference_steps=gen_config.num_inference_steps,
                        guidance_scale=gen_config.guidance_scale, generator=generator,
                        latents=gen_config.latents)[0]

    def get_base_model_version(self): return self.arch


class HunyuanImage3InstructModel(HunyuanImage3Model):
    arch = 'hunyuan_image_3_instruct'
    variant = INSTRUCT


class HunyuanImage3BaseModel(HunyuanImage3Model):
    arch = 'hunyuan_image_3_base'
    variant = BASE
