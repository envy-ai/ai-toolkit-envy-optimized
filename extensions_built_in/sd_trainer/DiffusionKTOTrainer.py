"""Experimental binary-feedback alignment using a flow-velocity MSE surrogate.

The reference and policy share one base model. A replay window scores detached
errors first, then sequentially recomputes exact policy gradients at unchanged
weights. Checkpoints are allowed ONLY at completed-window/optimizer boundaries.
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict
import json

import torch

from extensions_built_in.sd_trainer.DiffusionTrainer import DiffusionTrainer
from toolkit.flow_kto import validate_kto_config, flow_kto_terms, kto_reference_point
from toolkit.flow_training import (FlowTrainingProfile, flow_training_metadata,
                                  log_flow_training_profile, noised_flow_state, predict_flow_branch)
from toolkit.train_tools import get_torch_dtype


class DiffusionKTOTrainer(DiffusionTrainer):
    def __init__(self, process_id, job, config, **kwargs):
        self.kto_settings, self.kto_source_counts = validate_kto_config(config)
        self.flow_profile = FlowTrainingProfile.from_model_config(config['model'])
        from toolkit.training_capabilities import cfg_reference_mode
        self.cfg_reference = cfg_reference_mode(config['model'])
        for dataset in config['datasets']:
            dataset['caption_dropout_rate'] = 0.
        super().__init__(process_id, job, config, **kwargs)
        self.needs_vae_at_train_time = False
        self._kto_replay_queue = deque()
        self._kto_preparing = False
        self.print(f'Diffusion-KTO valid source images (before repeats): {self.kto_source_counts}')
        if not all(self.kto_source_counts.values()):
            self.print('WARNING: Diffusion-KTO has only one feedback class; monitor drift and reference-point estimates.')
        self.print('Diffusion-KTO is experimental: unweighted per-image flow-velocity MSE, not a validated DDPM likelihood.')
        self.print(f'Diffusion-KTO estimator: {self.kto_settings.reference_estimator}; '
                   f'window size {self.kto_settings.score_window_size}; short tails use actual counts. '
                   'No estimator state carries across optimizer steps.')

    def update_training_metadata(self):
        super().update_training_metadata()
        self.add_meta({**flow_training_metadata(self),
            'ss_diffusion_kto': json.dumps(asdict(self.kto_settings), sort_keys=True),
            'ss_diffusion_kto_surrogate': 'flow_velocity_mse_unweighted_v1',
            'ss_diffusion_kto_checkpoint_boundary': 'completed_optimizer_window_v1'})

    def hook_before_train_loop(self):
        log_flow_training_profile(self)
        super().hook_before_train_loop()

    @contextmanager
    def _prediction_rng(self, record):
        # Reference, detached policy score and replay use identical stochastic
        # state. fork_rng also restores caller RNG on exceptions/backward exits.
        cuda = record['cuda_device']
        with torch.random.fork_rng(devices=[] if cuda is None else [cuda]):
            torch.set_rng_state(record['cpu_rng'])
            if cuda is not None:
                torch.cuda.set_rng_state(record['cuda_rng'], cuda)
            yield

    def _record_error(self, record):
        dtype = get_torch_dtype(self.train_config.dtype)
        noisy = record['noisy'].to(self.device_torch, dtype=dtype)
        target = record['target'].to(self.device_torch, dtype=torch.float32)
        timestep = record['timestep'].to(self.device_torch)
        # .to() mutates prompt objects. Keep the master (including Anima's
        # integer fields) on CPU rather than accumulating a GPU prompt bank.
        embeds = record['embeds'].detach().to(self.device_torch, dtype=dtype)
        prediction = predict_flow_branch(self.sd, noisy, timestep, embeds, record['batch'])
        return (prediction - target).square().flatten(1).mean(1)

    @torch.no_grad()
    def _score_batch(self, batch):
        if batch is None or batch.latents is None or batch.prompt_embeds is None:
            raise ValueError('Diffusion-KTO requires cached image latents and prompt embeddings')
        dtype = get_torch_dtype(self.train_config.dtype)
        clean = batch.latents.detach().to(self.device_torch, dtype=dtype)
        labels = [getattr(item.dataset_config, 'kto_label', None) for item in batch.file_items]
        if len(labels) != clean.shape[0] or any(label not in ('liked', 'disliked') for label in labels):
            raise ValueError('Diffusion-KTO batch lost its per-image liked/disliked labels')
        weights = torch.as_tensor(batch.loss_multiplier_list, dtype=torch.float32)
        if weights.shape != (clean.shape[0],) or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Diffusion-KTO needs finite nonnegative per-image dataset weights')
        time = self.flow_profile.sample_time(clean, device=self.device_torch,
            minimum=self.train_config.min_denoising_steps, maximum=self.train_config.max_denoising_steps)
        noise = torch.randn(clean.shape, device=self.device_torch, dtype=torch.float32)
        cuda = None
        if torch.device(self.device_torch).type == 'cuda':
            cuda = torch.device(self.device_torch).index
            if cuda is None:
                cuda = torch.cuda.current_device()
        record = {'batch': batch, 'noisy': noised_flow_state(clean, time, noise).detach().cpu(),
            'target': (noise - clean.float()).detach().cpu(), 'timestep': (time * 1000).cpu(),
            'embeds': batch.prompt_embeds.detach().to('cpu'), 'weights': weights.cpu(),
            'liked': torch.tensor([label == 'liked' for label in labels]),
            'cpu_rng': torch.get_rng_state(), 'cuda_device': cuda,
            'cuda_rng': None if cuda is None else torch.cuda.get_rng_state(cuda)}
        del clean, noise, time
        self.network.is_active = False
        with self._prediction_rng(record):
            record['reference_error'] = self._record_error(record).detach().cpu()
        self.network.is_active = True
        with self._prediction_rng(record):
            record['policy_error'] = self._record_error(record).detach().cpu()
        if not torch.isfinite(record['policy_error']).all() or not torch.isfinite(record['reference_error']).all():
            raise ValueError('Diffusion-KTO encountered non-finite score errors')
        return record

    def _prepare_replay(self, batches):
        records = [self._score_batch(batch) for batch in batches]
        size = self.kto_settings.score_window_size if self.kto_settings.reference_estimator == 'score_window' else 1
        points = []
        for start in range(0, len(records), size):
            window = records[start:start + size]
            point = kto_reference_point([record['reference_error'] - record['policy_error'] for record in window])
            points.append(point.item())
            for record in window:
                record['loss'], record['coefficient'], _ = flow_kto_terms(record['policy_error'],
                    record['reference_error'], record['liked'], self.kto_settings, reference_point=point)
        count = sum(record['liked'].numel() for record in records)
        for record in records:
            # SDTrainer multiplies each replay by 1/number_of_batches. Account
            # for unequal bucket batch sizes so this is a per-EXAMPLE average.
            record['normalization'] = len(records) / count
        self._kto_replay_queue = deque(records)
        labels = torch.cat([record['liked'] for record in records])
        differences = torch.cat([record['reference_error'] - record['policy_error'] for record in records])
        utility_losses = torch.cat([record['loss'] for record in records])
        weighted_losses = torch.cat([record['loss'] * record['weights'] for record in records])
        self.additional_logs.update({'kto/reference_point': sum(points) / len(points),
            'kto/liked_count': int(labels.sum()), 'kto/disliked_count': int((~labels).sum()),
            'kto/window_count': len(points), 'kto/short_tail_batches': len(records) % size,
            'kto/saturation_fraction': float(torch.cat([record['coefficient'].abs() < 1e-8 for record in records]).float().mean())})
        for label, mask in [('liked', labels), ('disliked', ~labels)]:
            if mask.any():
                self.additional_logs[f'kto/{label}_score'] = differences[mask].mean().item()
                self.additional_logs[f'kto/{label}_utility_loss'] = utility_losses[mask].mean().item()
        return weighted_losses.mean().item()

    def train_single_accumulation(self, batch, accum_scale=1.):
        if not self._kto_replay_queue:
            if self.kto_settings.reference_estimator != 'batch_mean':
                raise ValueError('Diffusion-KTO score windows must run through the complete training hook')
            # Also supports ordinary extension/test callers with a single batch.
            active, multiplier = self.network.is_active, self.network.multiplier
            try:
                self.network.multiplier = 1.
                self._prepare_replay([batch])
                return self.train_single_accumulation(batch, accum_scale)
            finally:
                self._kto_replay_queue.clear()
                self.network.is_active, self.network.multiplier = active, multiplier
        record = self._kto_replay_queue[0]
        if record['batch'] is not batch:
            raise ValueError('Diffusion-KTO replay order changed after detached scoring')
        self.network.is_active = True
        with self._prediction_rng(record):
            error = self._record_error(record)
            coefficient = record['coefficient'].to(error.device)
            weight = record['weights'].to(error.device)
            self.accelerator.backward((coefficient * weight * error).sum() * record['normalization'] * accum_scale)
        self._kto_replay_queue.popleft()
        from toolkit.training_examples import log_image_losses
        log_image_losses(self, batch, record['loss'], record['loss'] * record['weights'], record['timestep'],
                         scope='per_image', extras={'kto_reference_estimator': self.kto_settings.reference_estimator})
        return (record['loss'] * record['weights']).mean().detach()

    def hook_train_loop(self, batch):
        batches = batch if isinstance(batch, list) else [batch]
        if not batches:
            raise ValueError('Diffusion-KTO cannot train an empty replay window')
        if self.kto_settings.reference_estimator == 'score_window' and len(batches) < self.kto_settings.score_window_size:
            raise ValueError('Diffusion-KTO received fewer accumulation batches than its configured score window')
        active, multiplier = self.network.is_active, self.network.multiplier
        was_training = self.sd.unet.training
        try:
            self._kto_preparing = True
            self.network.multiplier = 1.
            # Keep base train/eval mode identical in score and replay. In
            # particular, eval() would turn OFF several models' checkpointing.
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._prepare_replay(batches)
            self._kto_preparing = False
            result = super().hook_train_loop(batches)
            if self._kto_replay_queue:
                raise RuntimeError('Diffusion-KTO optimizer hook left an incomplete replay window')
            result['loss'] = loss
            return result
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            self._kto_replay_queue.clear()
            self._kto_preparing = False
            self.network.is_active, self.network.multiplier = active, multiplier
            self.sd.unet.train(was_training)

    def save(self, *args, **kwargs):
        if self._kto_preparing or self._kto_replay_queue:
            raise RuntimeError('Diffusion-KTO checkpoints require a completed score/replay window')
        return super().save(*args, **kwargs)
