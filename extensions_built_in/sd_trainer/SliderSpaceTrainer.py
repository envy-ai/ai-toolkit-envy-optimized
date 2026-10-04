"""Semantic PCA direction LoRAs with model-native flow and conditioning."""

import copy
import json
import math
import os
import re
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from torchvision.transforms.functional import to_tensor

from toolkit.flow_training import (FlowTrainingProfile, trainer_flow_profile, render_flow_bank_image,
                                   guided_flow_prediction, scoped_training_decode)
from toolkit.training_capabilities import validate_specialized_model, cfg_reference_mode
from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.basic import flush
from toolkit.metadata import get_meta_for_safetensors, load_metadata_from_safetensors
from toolkit.models.sliderspace_network import SliderSpaceNetwork
from toolkit.sliderspace import (SliderSpaceConfig, SemanticEncoder, atomic_json, parse_auto_sample_strengths,
                                discovery_signature, discover_directions,
                                flow_clean_prediction, semantic_direction_loss,
                                scan_discovery_images, discovery_entries, prepare_discovery_image, verify_discovery_sources)
from toolkit.train_tools import get_torch_dtype


class SliderSpaceTrainer(DiffusionTrainer):
    require_optimizer_state = True

    def __init__(self, process_id, job, config, **kwargs):
        config = copy.deepcopy(config)
        self.sliderspace = SliderSpaceConfig.parse(config.get('sliderspace'))
        model, network = config.get('model', {}), config.get('network', {}) or {}
        train = config.setdefault('train', {})
        capability = validate_specialized_model(config, 'sliderspace')
        self.flow_profile = FlowTrainingProfile.from_model_config(model)
        self.cfg_reference = cfg_reference_mode(model)
        self.bucket_divisibility = capability.bucket_divisibility
        if network.get('pretrained_lora_path'):
            raise ValueError('SliderSpace starts a new adapter bank; pretrained initialization is unsupported')
        if (network.get('network_kwargs') or {}).get('full_train_in_out'):
            raise ValueError('SliderSpace cannot train shared input/output layers')
        if config.get('datasets'):
            raise ValueError('SliderSpace uses sliderspace.discovery_datasets, not ordinary training datasets')
        if train.get('batch_size', 1) != 1 or any(train.get(key, 1) != 1 for key in
                                                ('gradient_accumulation', 'gradient_accumulation_steps')):
            raise ValueError('SliderSpace requires batch size and gradient accumulation of 1')
        if type(train.get('steps', 1000)) is not int or train.get('steps', 1000) < self.sliderspace.num_directions:
            raise ValueError('SliderSpace total steps must include at least one update per direction')
        if train.get('train_text_encoder') or train.get('train_refiner') or train.get('train_unet') is False:
            raise ValueError('SliderSpace trains only transformer LoRAs')
        if (train.get('noise_scheduler', 'flowmatch') != 'flowmatch'
                or train.get('timestep_type', 'shift') != 'shift'):
            raise ValueError('SliderSpace requires shifted flow-matching timesteps')
        conflicts = ('merge_network_on_save', 'do_cfg', 'do_random_cfg', 'do_guidance_loss',
                     'do_differential_guidance', 'diff_output_preservation', 'blank_prompt_preservation',
                     'do_prior_divergence', 'train_turbo', 'do_fft_loss', 'learnable_snr_gos',
                     'diffusion_feature_extractor_path', 'validation_config', 'inverted_mask_prior',
                     'correct_pred_norm', 'free_u', 'do_paramiter_swapping')
        if (any(train.get(key) for key in conflicts) or train.get('loss_type', 'mse') != 'mse'
                or train.get('frequency_loss_type', 'none') != 'none'
                or (train.get('ema_config') or {}).get('use_ema')):
            raise ValueError('SliderSpace cannot combine its semantic objective with other objectives, EMA, or merging')
        if any(config.get(key) for key in ('adapter', 'decorator', 'embedding')) or any(
                model.get(key) for key in ('inference_lora_path', 'unconditional_lora_path')):
            raise ValueError('SliderSpace does not support additional trainable, inference or unconditional adapters')
        if config.get('trigger_word'):
            raise ValueError('SliderSpace uses exact concept prompts; remove trigger_word')
        minimum, maximum = train.get('min_denoising_steps', 0), train.get('max_denoising_steps', 999)
        if not (isinstance(minimum, (int, float)) and isinstance(maximum, (int, float))
                and 0 <= minimum < maximum <= 1000):
            raise ValueError('SliderSpace requires 0 <= minimum timestep < maximum timestep <= 1000')
        if (config.get('save', {}).get('save_format', 'safetensors') != 'safetensors'
                or config.get('save', {}).get('push_to_hub')):
            raise ValueError('SliderSpace exports local safetensors; automatic Hub upload is unsupported')
        provided = scan_discovery_images(self.sliderspace, self.bucket_divisibility)
        self.discovery_entries = discovery_entries(self.sliderspace, provided)
        self.discovery_id = discovery_signature(self.sliderspace, model, provided)
        self.resume_bank_path = None
        self.discovery_components = None
        self.resume_components = None
        self.direction_vectors = None
        self.variance_ratio = None
        self.semantic_encoder = None
        # Own conditioning is cached in RAM on CPU. The generic "unload TE"
        # path also caches preview prompts, without freeing the encoder object.
        train.update(cache_text_embeddings=False, unload_text_encoder=True,
                     noise_scheduler='flowmatch', timestep_type='shift', train_unet=True)
        config.setdefault('save', {}).update(record_low_start_step=train.get('steps', 1000) + 1,
                                             sample_on_record_low=False)
        config['datasets'] = []
        super().__init__(process_id, job, config, **kwargs)
        if self.accelerator.num_processes != 1:
            raise ValueError('SliderSpace currently supports one training process only')
        self.retain_vae_after_caching = True
        self.needs_vae_at_train_time = True

    def get_network_class(self):
        return partial(SliderSpaceNetwork, num_directions=self.sliderspace.num_directions)

    def hook_after_model_load(self):
        super().hook_after_model_load()
        if self.flow_profile.arch != 'qwen_image_2':
            from toolkit.flow_cache_identity import resolved_flow_components
            self.discovery_components = {'version': 1, 'flow': resolved_flow_components(self.sd)}

    def _resolve_feature_identity(self):
        if trainer_flow_profile(self).arch == 'qwen_image_2':
            return
        if self.discovery_components is None or not getattr(self.semantic_encoder, 'cache_identity', None):
            raise ValueError('SliderSpace requires resolved model and feature-encoder identities before discovery')
        self.discovery_components['feature'] = self.semantic_encoder.cache_identity
        if getattr(self, 'resume_components', None) is not None and self.resume_components != self.discovery_components:
            raise ValueError('SliderSpace loaded components/feature encoder changed on resume; use a new job name')

    def _stage(self, message):
        self.print(message)
        self.update_status('running', message)
        if getattr(self, 'is_ui_trainer', False):
            self.maybe_stop()

    def _cache_concepts(self):
        self._stage('Caching SliderSpace concept prompts')
        self.sd.set_device_state_preset('cache_text_encoder')
        try:
            with torch.no_grad():
                self.static_embeds = {}
                for prompt in ('', self.train_config.unconditional_prompt):
                    self.static_embeds[prompt] = self.sd.encode_prompt([prompt]).detach().to('cpu')
                self.concept_embeds = [self.sd.encode_prompt([prompt]).detach().to('cpu')
                                       for prompt in self.sliderspace.concept_prompts]
                by_caption = dict(zip(self.sliderspace.concept_prompts, self.concept_embeds))
                for index, entry in enumerate(self.discovery_entries):
                    if entry['caption'] not in by_caption:
                        self._stage(f'Caching discovery captions ({index + 1}/{len(self.discovery_entries)})')
                        by_caption[entry['caption']] = self.sd.encode_prompt([entry['caption']]).detach().to('cpu')
                self.bank_embeds = [by_caption[entry['caption']] for entry in self.discovery_entries]
                self.negative_embeds = (self.sd.encode_prompt([self.sliderspace.negative_prompt]).detach().to('cpu')
                                        if self.sliderspace.cfg_scale > 1 and getattr(self, 'cfg_reference', None) != 'image_only' else None)
                self.cache_sample_prompts()
        finally:
            # Do not put the DiT back onto GPU while TE occupies it.
            self.sd.text_encoder_to('cpu')
            flush()

    def encode_static_prompt(self, prompt, **kwargs):
        key = prompt[0] if isinstance(prompt, list) and len(prompt) == 1 else prompt
        if isinstance(key, str) and key in getattr(self, 'static_embeds', {}):
            return self.static_embeds[key].detach()
        return super().encode_static_prompt(prompt, **kwargs)

    def hook_before_train_loop(self):
        from toolkit.flow_training import log_flow_training_profile
        log_flow_training_profile(self)
        self._run_with_optimizer_state_offload(self._cache_concepts)
        # Prompt caches were prepared with only TE resident. Avoid the generic
        # hook reloading TE alongside the VAE required by our pixel-space loss.
        self.train_config.unload_text_encoder = False
        try:
            super().hook_before_train_loop()
        finally:
            self.train_config.unload_text_encoder = True
        self.semantic_encoder = SemanticEncoder(self.sliderspace.feature_encoder, self.sliderspace.feature_device,
                                                preserve_aspect_ratio=self.sliderspace.discovery_buckets
                                                and self.sliderspace.discovery_mode != 'generated')
        self._resolve_feature_identity()
        self._run_with_optimizer_state_offload(self._build_discovery_bank)

    def _discovery_root(self):
        root = Path(self.save_root) / 'sliderspace_discovery' / self.discovery_id
        components = getattr(self, 'discovery_components', None)
        if components is not None:
            from toolkit.flow_cache_identity import component_identity_digest
            root = root / component_identity_digest(components)
        return root

    def _bank_paths(self, index):
        stem = self._discovery_root() / f'{index:06d}'
        return stem.with_suffix('.png'), stem.with_suffix('.safetensors')

    @staticmethod
    def _file_stamp(path):
        stat = path.stat()
        return [stat.st_size, stat.st_mtime_ns]

    def _build_discovery_bank(self):
        verify_discovery_sources(self.discovery_entries)
        components = getattr(self, 'discovery_components', None)
        if components is not None and 'feature' not in components:
            raise ValueError('SliderSpace feature identity must be resolved before discovery bank reuse')
        root = self._discovery_root()
        root.mkdir(parents=True, exist_ok=True)
        manifest_path, tensor_path = root / 'manifest.json', root / 'features.safetensors'
        if manifest_path.exists() and tensor_path.exists():
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            expected = {path.name: self._file_stamp(path)
                        for i in range(len(self.discovery_entries)) for path in self._bank_paths(i)
                        if path.exists()}
            if (manifest.get('signature') != self.discovery_id or manifest.get('files') != expected
                    or len(expected) != 2 * len(self.discovery_entries)
                    or manifest.get('feature_file') != self._file_stamp(tensor_path)
                    or (components is not None and manifest.get('components') != components)):
                raise ValueError('SliderSpace discovery cache changed or is incomplete; restore it or use a new job')
            tensors = load_file(str(tensor_path))
            if (tensors['directions'].shape[0] != self.sliderspace.num_directions
                    or tensors['features'].shape[0] != len(self.discovery_entries)
                    or not all(torch.isfinite(value).all() for value in tensors.values())):
                raise ValueError('Invalid SliderSpace discovery feature cache')
            self.direction_vectors, self.variance_ratio = tensors['directions'], tensors['variance_ratio']
            self._stage('Reusing SliderSpace discovery bank and PCA directions')
            return

        self.sd.save_device_state()
        previous = self.network.multiplier, self.network.is_active, self.sd.unet.training
        try:
            self.sd.text_encoder_to('cpu')
            generates = self.sliderspace.discovery_mode != 'provided'
            # Provided-only ingestion needs VAE and features, not a GPU DiT.
            self.sd.unet.to(self.device_torch if generates else 'cpu')
            self.sd.unet.eval()
            self.sd.vae.to(self.sd.vae_device_torch)
            self.network.is_active = False
            with torch.no_grad():
                pipeline = self.sd.get_generation_pipeline() if generates else None
                features = []
                for i, entry in enumerate(self.discovery_entries):
                    image_path, latent_path = self._bank_paths(i)
                    self._stage(f'{"Importing" if entry["source"] == "provided" else "Generating"} SliderSpace discovery images '
                                f'({i + 1}/{len(self.discovery_entries)})')
                    if not image_path.exists():
                        if entry['source'] == 'provided':
                            image = prepare_discovery_image(entry['image'], self.sliderspace.resolution,
                                                            self.sliderspace.discovery_buckets,
                                                            getattr(self, 'bucket_divisibility', 32))
                        else:
                            positive = self.bank_embeds[i].detach().to(self.device_torch, dtype=self.sd.torch_dtype)
                            negative = self.negative_embeds.detach().to(self.device_torch, dtype=self.sd.torch_dtype) if self.negative_embeds else None
                            image = render_flow_bank_image(self.sd, pipeline, positive, negative,
                                             height=self.sliderspace.resolution, width=self.sliderspace.resolution,
                                             steps=self.sliderspace.discovery_steps,
                                             cfg=self.sliderspace.cfg_scale, seed=entry['seed'])
                        image.save(str(image_path) + '.tmp', format='PNG')
                        os.replace(str(image_path) + '.tmp', image_path)
                    with Image.open(image_path) as image:
                        pixels = to_tensor(image.convert('RGB')).mul(2).sub(1).unsqueeze(0)
                    expected_size = entry.get('size', [self.sliderspace.resolution] * 2)
                    if pixels.shape[-2:] != tuple(reversed(expected_size)):
                        raise ValueError('Wrong SliderSpace discovery image resolution')
                    if not latent_path.exists():
                        latent = self.sd.encode_images(pixels.to(self.sd.vae_device_torch, dtype=self.sd.vae_torch_dtype)).detach().cpu()
                        save_file({'latent': latent.contiguous()}, str(latent_path) + '.tmp')
                        os.replace(str(latent_path) + '.tmp', latent_path)
                    verify_discovery_sources([entry])
                    self._stage(f'Extracting SliderSpace features ({i + 1}/{len(self.discovery_entries)})')
                    features.append(self.semantic_encoder(pixels).cpu())
                features = torch.cat(features)
                self._stage(f'Finding {self.sliderspace.num_directions} SliderSpace PCA directions')
                directions, ratios, scores = discover_directions(features, self.sliderspace.num_directions)
                save_file({'features': features, 'directions': directions, 'variance_ratio': ratios,
                           'scores': scores}, str(tensor_path) + '.tmp')
                os.replace(str(tensor_path) + '.tmp', tensor_path)
                atomic_json(manifest_path, {'signature': self.discovery_id, 'feature_file': self._file_stamp(tensor_path),
                            **({'components': components} if components is not None else {}),
                            'entries': self.discovery_entries,
                            'files': {path.name: self._file_stamp(path) for i in range(len(self.discovery_entries))
                                      for path in self._bank_paths(i)}})
                self.direction_vectors, self.variance_ratio = directions, ratios
        finally:
            self.network.multiplier, self.network.is_active = previous[:2]
            self.sd.unet.train(previous[2])
            self.sd.restore_device_state()
            flush()

    def _guided_prediction(self, noisy, timestep, positive, negative):
        if getattr(self.sd, 'arch', 'qwen_image_2') != 'qwen_image_2':
            return guided_flow_prediction(self.sd, noisy, timestep, positive, negative, self.sliderspace.cfg_scale)
        def predict(embeds):
            return self.sd.predict_noise(latents=noisy, timestep=timestep,
                                         conditional_embeddings=embeds, unconditional_embeddings=None,
                                         guidance_scale=1.0, guidance_embedding_scale=1.0, batch=None).float()
        conditional = predict(positive)
        if negative is None:
            return conditional
        unconditional = predict(negative)
        return unconditional + self.sliderspace.cfg_scale * (conditional - unconditional)

    def _decode_training_prediction(self, latent):
        try:
            return self.sd.decode_latents(latent)
        finally:
            # Nonreentrant checkpoint recomputation can stop before the VAE's
            # own end-of-forward cache clear. Do not retain causal feature maps
            # (or their graphs) between steps or after an interrupted decode.
            vae = getattr(self.sd, 'vae', None)
            if vae is not None and hasattr(vae, 'clear_cache'):
                vae.clear_cache()

    @scoped_training_decode
    def train_single_accumulation(self, batch, accum_scale=1.0):
        direction = self.step_num % self.sliderspace.num_directions
        generator = torch.Generator(device='cpu').manual_seed((self.sliderspace.seed + self.step_num * 104729) % 2**63)
        index = torch.randint(len(self.discovery_entries), (1,), generator=generator).item()
        clean = load_file(str(self._bank_paths(index)[1]))['latent'].to(self.device_torch, dtype=self.sd.torch_dtype)
        t = torch.sigmoid(torch.randn(1, generator=generator) + trainer_flow_profile(self).shift(*clean.shape[-2:]))
        low, high = self.train_config.min_denoising_steps / 1000, self.train_config.max_denoising_steps / 1000
        t = (low + (high - low) * t).to(self.device_torch)
        noise = torch.randn(clean.shape, generator=generator).to(self.device_torch)
        noisy = ((1 - t.view(-1, 1, 1, 1)) * clean.float() + t.view(-1, 1, 1, 1) * noise).to(clean.dtype)
        timestep = t * 1000
        positive = self.bank_embeds[index].detach().to(self.device_torch, dtype=self.sd.torch_dtype)
        negative = self.negative_embeds.detach().to(self.device_torch, dtype=self.sd.torch_dtype) if self.negative_embeds else None
        previous = self.network.active_direction, self.network.multiplier, self.network.is_active, self.sd.unet.training
        try:
            self.network.is_active = False
            self.sd.unet.eval()
            with torch.no_grad():
                teacher = self._guided_prediction(noisy, timestep, positive, negative)
                # Use the same one-shot decode in both branches, without a
                # teacher graph: detached input and frozen VAE require no grads.
                # Qwen otherwise tiles no-grad decodes but not student decodes.
                with torch.enable_grad():
                    base_pixels = self._decode_training_prediction(flow_clean_prediction(noisy, teacher, timestep).to(clean.dtype).detach())
                base_features = self.semantic_encoder(base_pixels).detach()
                del teacher, base_pixels
            self.network.select_direction(direction)
            self.network.multiplier = 1.0
            self.network.is_active = True
            self.sd.unet.train()
            context = torch.autograd.graph.save_on_cpu(pin_memory=True) if noisy.is_cuda else nullcontext()
            with context:
                student = self._guided_prediction(noisy, timestep, positive, negative)
                clean_prediction = flow_clean_prediction(noisy, student, timestep).to(clean.dtype)
                pixels = checkpoint(self._decode_training_prediction, clean_prediction, use_reentrant=False)
                features = self.semantic_encoder(pixels)
                loss = semantic_direction_loss(features, base_features, self.direction_vectors[direction]) * self.sliderspace.loss_weight
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite SliderSpace semantic loss')
            # Keep this direction active until backward has recomputed checkpoints.
            self.accelerator.backward(loss * accum_scale)
            return loss.detach()
        finally:
            self.network.select_direction(previous[0])
            self.network.multiplier, self.network.is_active = previous[1:3]
            self.sd.unet.train(previous[3])

    def _state_root(self):
        return Path(self.save_root) / 'sliderspace_state'

    def update_training_metadata(self):
        super().update_training_metadata()
        from toolkit.flow_training import flow_training_metadata
        self.add_meta(flow_training_metadata(self))

    def hook_train_loop(self, batch):
        result = super().hook_train_loop(batch)
        # The base loop's step_num is zero-based. Resume from the next update,
        # rather than repeating the saved update and its direction.
        self.completed_updates = self.step_num + 1
        self.update_status('running', f'Training SliderSpace direction {self.step_num % self.sliderspace.num_directions + 1}'
                           f'/{self.sliderspace.num_directions} · {self.completed_updates}/{self.train_config.steps} total updates')
        return result

    def get_latest_save_path(self, name=None, post='', include_pretrained_lora=True):
        candidates = []
        for path in self._state_root().glob('checkpoint_*.json'):
            with path.open() as handle:
                record = json.load(handle)
            step = record['step']
            if type(step) is not int or step < 0 or (path.name != 'checkpoint_final.json'
                    and path.name != f'checkpoint_{step:09d}.json'):
                raise ValueError('Invalid SliderSpace checkpoint step')
            bank = self._state_root() / f'bank_{step:09d}.safetensors'
            candidates.append((step, path.stat().st_mtime_ns, str(bank), record))
        if not candidates:
            self.resume_bank_path = None
            return None
        step, _, bank_path, record = max(candidates, key=lambda item: item[:2])
        bank = Path(bank_path)
        optimizer = bank.with_suffix('.optimizer.pt')
        if not bank.is_file() or not optimizer.is_file():
            raise ValueError('SliderSpace published checkpoint is incomplete: complete bank and optimizer are required')
        if record.get('signature') != self.discovery_id:
            raise ValueError('SliderSpace discovery settings/model changed on resume; use a new job name')
        files = record.get('files')
        if files is not None:
            exports = record.get('exports')
            if (not isinstance(files, dict) or not isinstance(exports, list)
                    or len(exports) != self.sliderspace.num_directions
                    or any(not isinstance(name, str) or Path(name).name != name for name in exports)):
                raise ValueError('Invalid SliderSpace checkpoint file manifest')
            paths = [bank, optimizer, *(Path(self.save_root) / name for name in exports)]
            actual = {path.name: self._file_stamp(path) for path in paths if path.is_file()}
            if len(actual) != len(paths) or actual != files:
                raise ValueError('SliderSpace checkpoint files changed or are incomplete; restore the published generation')
        elif getattr(self, 'discovery_components', None) is not None:
            raise ValueError('SliderSpace checkpoint is missing resolved-component file verification; use a new job name')
        self.resume_bank_path = bank_path
        return self.resume_bank_path

    def get_optimizer_state_path(self):
        if self.resume_bank_path:
            path = self.resume_bank_path.removesuffix('.safetensors') + '.optimizer.pt'
            if not os.path.isfile(path):
                raise ValueError('SliderSpace resume requires the optimizer paired with its complete bank checkpoint')
            return path
        return str(self._state_root() / 'no_resume.optimizer.pt')

    def load_weights(self, path):
        metadata = load_metadata_from_safetensors(path)
        if metadata.get('ss_sliderspace_signature') != self.discovery_id:
            raise ValueError('SliderSpace discovery settings/model changed on resume; use a new job name')
        components = getattr(self, 'discovery_components', None)
        if components is not None:
            saved = metadata.get('ss_sliderspace_components')
            if isinstance(saved, str):
                try:
                    saved = json.loads(saved)
                except (ValueError, TypeError):
                    saved = None
            if not isinstance(saved, dict) or saved.get('flow') != components.get('flow') or 'feature' not in saved:
                raise ValueError('SliderSpace resolved components changed or are missing on resume; use a new job name')
            self.resume_components = saved
        return super().load_weights(path)

    def save(self, step=None, record_low_loss=None, scheduled_save=False):
        if not self.accelerator.is_main_process:
            return
        if self.optimizer is None:
            raise ValueError('SliderSpace complete bank checkpoints require optimizer state')
        current_step = getattr(self, 'completed_updates', self.step_num if step is None else step)
        self._stage('Saving SliderSpace direction LoRAs and complete bank')
        root = self._state_root()
        root.mkdir(parents=True, exist_ok=True)
        old_final_step = None
        if step is None and (root / 'checkpoint_final.json').exists():
            with (root / 'checkpoint_final.json').open() as handle:
                old_final_step = json.load(handle)['step']
        self.update_training_metadata()
        self.add_meta({'ss_sliderspace_signature': self.discovery_id,
                       'ss_sliderspace_directions': self.sliderspace.num_directions})
        if getattr(self, 'discovery_components', None) is not None:
            if 'feature' not in self.discovery_components:
                raise ValueError('SliderSpace cannot save before feature identity is resolved')
            self.add_meta({'ss_sliderspace_components': json.dumps(self.discovery_components, sort_keys=True)})
        metadata = get_meta_for_safetensors(copy.deepcopy(self.meta), self.job.name)
        metadata['training_info'] = json.dumps({'step': current_step, 'epoch': self.epoch_num})
        exported = []
        suffix = f'_{current_step:09d}' if step is not None else ''
        for i, direction in enumerate(self.network.directions):
            path = Path(self.save_root) / f'{self.job.name}_direction_{i + 1:02d}{suffix}.safetensors'
            child_metadata = dict(metadata, ss_sliderspace_direction=str(i + 1))
            # Stage below the job root so downloads never list half-written
            # independent direction exports while this bank is being saved.
            temporary = str(root / f'export_direction_{i + 1:02d}.tmp.safetensors')
            direction.save_weights(temporary, dtype=get_torch_dtype(self.save_config.dtype), metadata=child_metadata)
            os.replace(temporary, path)
            exported.append(path.name)
        bank = root / f'bank_{current_step:09d}.safetensors'
        if self.optimizer is not None:
            optimizer_path = bank.with_suffix('.optimizer.pt')
            torch.save(self.optimizer.state_dict(), str(optimizer_path) + '.tmp')
            os.replace(str(optimizer_path) + '.tmp', optimizer_path)
        self.network.save_bank(str(bank), metadata)
        # This marker publishes a complete checkpoint generation last.
        marker = 'checkpoint_final.json' if step is None else f'checkpoint_{current_step:09d}.json'
        record = {'step': current_step, 'exports': exported, 'signature': self.discovery_id}
        if getattr(self, 'discovery_components', None) is not None:
            record['files'] = {path.name: self._file_stamp(path) for path in
                [bank, bank.with_suffix('.optimizer.pt'), *(Path(self.save_root) / name for name in exported)]}
        atomic_json(root / marker, record)
        if (type(old_final_step) is int and old_final_step != current_step
                and not (root / f'checkpoint_{old_final_step:09d}.json').exists()):
            (root / f'bank_{old_final_step:09d}.safetensors').unlink(missing_ok=True)
            (root / f'bank_{old_final_step:09d}.optimizer.pt').unlink(missing_ok=True)
        atomic_json(Path(self.save_root) / 'sliderspace.json', {'step': current_step, 'exports': exported,
                    'signature': self.discovery_id, 'variance_ratio': self.variance_ratio.tolist() if self.variance_ratio is not None else []})
        self.last_save_path = str(bank)
        self._stage(f'Saved {len(exported)} SliderSpace direction LoRAs and complete bank')
        self.clean_up_saves()
        self.post_save_hook(str(bank))

    def clean_up_saves(self):
        if not self.accelerator.is_main_process:
            return
        root = self._state_root()
        markers = sorted(path for path in root.glob('checkpoint_*.json') if path.stem != 'checkpoint_final')
        protected = None
        final = root / 'checkpoint_final.json'
        if final.exists():
            with final.open() as handle:
                protected = json.load(handle)['step']
        keep = max(1, self.save_config.max_step_saves_to_keep)
        for marker in markers[:-keep]:
            with marker.open() as handle:
                step = json.load(handle)['step']
            if type(step) is not int or marker.name != f'checkpoint_{step:09d}.json':
                raise ValueError('Invalid SliderSpace retention marker')
            paths = [Path(self.save_root) / f'{self.job.name}_direction_{i + 1:02d}_{step:09d}.safetensors'
                     for i in range(self.sliderspace.num_directions)]
            if step != protected:
                paths += [root / f'bank_{step:09d}.safetensors', root / f'bank_{step:09d}.optimizer.pt']
            for path in paths:
                path.unlink(missing_ok=True)
            marker.unlink()
        # Separate preview exports never participate in bank resume/retention.
        # Keep them while background Comfy workers may still need any snapshot.
        if not getattr(self, '_comfy_background_threads', []):
            pending = False
        else:
            pending = any(thread.is_alive() for thread in self._comfy_background_threads)
        if not pending:
            pattern = re.compile(re.escape(self.job.name) + r'_comfy_direction_\d+_(\d{9})\.safetensors')
            snapshots = [(int(match[1]), path) for path in Path(self.save_root).iterdir()
                         if path.is_file() and (match := pattern.fullmatch(path.name))]
            steps = sorted({step for step, _ in snapshots})
            old_steps = set(steps[:-keep])
            for step, path in snapshots:
                if step in old_steps:
                    path.unlink()

    def _refresh_live_sample_config(self):
        # Keep the entire auto comparison on one snapshot of the live settings.
        if getattr(self, '_auto_sample_preview', None) is not None:
            return
        super()._refresh_live_sample_config()
        path = getattr(self.job, 'config_path', None)
        if path:
            from toolkit.config import get_config
            value = get_config(path)['config']['process'][self.process_id].get('sliderspace', {})
            direction, strength = value.get('preview_direction', self.sliderspace.preview_direction), value.get('preview_strength', self.sliderspace.preview_strength)
            if type(direction) is not int or not 1 <= direction <= self.sliderspace.num_directions:
                raise ValueError('Invalid live SliderSpace preview direction')
            if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not math.isfinite(strength):
                raise ValueError('Invalid live SliderSpace preview strength')
            auto = value.get('preview_auto', self.sliderspace.preview_auto)
            if type(auto) is not bool:
                raise ValueError('Invalid live SliderSpace auto sampling setting')
            strengths = value.get('preview_auto_strengths', self.sliderspace.preview_auto_strengths)
            parse_auto_sample_strengths(strengths)
            self.sliderspace.preview_direction, self.sliderspace.preview_strength = direction, strength
            self.sliderspace.preview_auto = auto
            self.sliderspace.preview_auto_strengths = strengths

    def _sample_preview(self):
        preview = getattr(self, '_auto_sample_preview', None)
        return preview if preview is not None else (self.sliderspace.preview_direction, self.sliderspace.preview_strength)

    def _get_comfy_lora_display_filename(self, step=None):
        suffix = f'_{step:09d}' if step is not None else ''
        direction, _ = self._sample_preview()
        # The base comparison uses a zero-strength snapshot of direction 1.
        return f'{self.job.name}_comfy_direction_{max(1, direction):02d}{suffix}.safetensors'

    def post_process_generate_image_config_list(self, configs):
        direction, strength = self._sample_preview()
        token = ('minus' if strength < 0 else 'plus') + str(abs(strength)).removesuffix('.0')
        identity = f'_direction_{direction:02d}_strength_{token}'
        auto = getattr(self, '_auto_sample_preview', None) is not None
        if auto:
            identity = '_auto' + identity
            configs = configs[:1]
        for config in configs:
            config.network_multiplier = strength
            count = str(self._auto_sample_index) if auto else '[count]'
            config.output_path = config.output_path.replace('_[count]', identity + '_' + count)
            config.output_filename_no_ext = config.output_filename_no_ext.replace('_[count]', identity + '_' + count)
            if auto:
                # Resolve a random seed once, then reuse it for every direction.
                if self._auto_sample_seed is None:
                    self._auto_sample_seed = config.seed
                config.seed = self._auto_sample_seed
        return configs

    def sample(self, step=None, is_first=False):
        self._refresh_live_sample_config()
        previous = self.network.active_direction, self.network.multiplier, self.network.is_active
        try:
            if not self.sliderspace.preview_auto:
                self.network.select_direction(self.sliderspace.preview_direction - 1)
                self.network.is_active = True
                return super().sample(step=step, is_first=is_first)
            # Reuse the normal native/Comfy renderer with only one direction
            # active at a time; never allocate another base model or bank.
            strengths = parse_auto_sample_strengths(self.sliderspace.preview_auto_strengths)
            previews = [(0, 0.0)] + [(direction, strength)
                for direction in range(1, self.sliderspace.num_directions + 1)
                for strength in strengths]
            self._auto_sample_seed = None
            for index, (direction, strength) in enumerate(previews):
                self._auto_sample_preview = (direction, strength)
                self._auto_sample_index = index
                self.network.select_direction(max(1, direction) - 1)
                self.network.multiplier = strength
                self.network.is_active = True
                super().sample(step=step, is_first=is_first)
        finally:
            self._auto_sample_preview = None
            self._auto_sample_seed = None
            self.network.select_direction(previous[0])
            self.network.multiplier, self.network.is_active = previous[1:]
