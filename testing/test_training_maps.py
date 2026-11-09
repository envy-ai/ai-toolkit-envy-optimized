import json
import random
from types import SimpleNamespace as NS

import numpy as np
import torch

from toolkit.training_maps import TrainingMaps, _grid


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_active = False
        self.multiplier = .7

    def __enter__(self):
        self.is_active = True
        return self

    def __exit__(self, *args):
        self.is_active = False


class Scheduler:
    def set_train_timesteps(self, *args, **kwargs):
        self.sigmas = torch.tensor([.25, .5, .75, 0])
        self.timesteps = torch.tensor([1, 2, 3])


class Embeds:
    def clone(self): return self
    def to(self, *args, **kwargs): return self


def make_trainer(tmp_path):
    network = Network()
    scheduler = Scheduler()
    unet = torch.nn.Linear(1, 1)
    class Model:
        noise_scheduler = scheduler
        def add_noise(self, latent, noise, timestep): return latent + noise
        def predict_noise(self, **kwargs):
            return torch.ones_like(kwargs['latents']) * network.multiplier
    model = Model()
    model.unet = unet
    config = NS(dtype='float32', timestep_type='linear', linear_timesteps=False, linear_timesteps2=False,
                num_train_timesteps=10, noise_scheduler='flowmatch', min_denoising_steps=0, max_denoising_steps=1000, cfg_scale=1., bypass_guidance_embedding=False)
    return NS(_run_with_optimizer_state_offload=lambda func: func(), save_root=str(tmp_path), model_config=NS(arch='moss_image', model_kwargs={}, name_or_path='fixture'), start_step=0, sd=model, network=network,
              train_config=config, device_torch=torch.device('cpu'), ema=None)


def test_capture_baseline_improvement_effect_and_restore(tmp_path):
    trainer = make_trainer(tmp_path)
    maps = TrainingMaps(trainer)
    maps.probes = [{'key': 'p', 'latents': torch.zeros(1, 2, 2, 2),
                    'noise': torch.ones(1, 2, 2, 2), 'embeds': Embeds()}]
    old_scheduler = trainer.sd.noise_scheduler
    torch.manual_seed(77); random.seed(77); np.random.seed(77)
    torch_state = torch.random.get_rng_state().clone()
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    maps.capture(trainer, 0)
    assert not maps.failed
    initial = json.loads((tmp_path / 'training_maps/step-000000000.json').read_text())['maps']['p']['0.25']
    assert initial['error']['v'] == [0.] * 4
    assert initial['effect']['v'] == [1.] * 4
    assert 'improvement' not in initial
    maps.capture(trainer, 2)
    later = json.loads((tmp_path / 'training_maps/step-000000002.json').read_text())['maps']['p']['0.25']
    assert later['improvement']['v'] == [0.] * 4
    assert maps.manifest['steps'] == [0, 2]
    assert maps.manifest['scales']['p']['0.25']['effect'] == 1
    assert trainer.sd.noise_scheduler is old_scheduler
    assert trainer.sd.unet.training and trainer.network.training and not trainer.network.is_active
    assert trainer.network.multiplier == .7
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    assert random.getstate() == random_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])


def test_grid_pooling_and_failure_persisted(tmp_path, monkeypatch):
    logged = []
    monkeypatch.setattr('toolkit.print.print_acc', lambda message: logged.append(message))
    assert _grid(torch.ones(49, 97))['w'] <= 48
    trainer = make_trainer(tmp_path)
    maps = TrainingMaps(trainer)
    maps.probes = [{'key': 'p', 'latents': torch.zeros(1, 2, 2, 2),
                    'noise': torch.ones(1, 2, 2, 2), 'embeds': Embeds()}]
    trainer.sd.predict_noise = lambda **kwargs: torch.zeros(1)
    maps.capture(trainer, 1)
    assert maps.failed
    manifest = json.loads((tmp_path / 'training_maps/manifest.json').read_text())
    assert manifest['status'] == 'error'
    assert 'prediction shape' in manifest['error']['message']
    assert 'Traceback' in manifest['error']['stack']
    assert 'prediction shape' in logged[0]
    assert trainer.network.multiplier == .7 and not trainer.network.is_active


def test_setup_cached_probe_bfloat16_and_resume(tmp_path, monkeypatch):
    from PIL import Image
    from toolkit.prompt_utils import PromptEmbeds
    import toolkit.training_maps as module

    class CachedItem:
        path = '/dataset/cat.png'
        is_reg = is_video = has_augmentations = False
        augments = []
        is_latent_cached = is_text_embedding_cached = True
        extra_values = []
        num_frames = 1
        loss_multiplier = 1.
        te_padding_side = 'right'
        text_embedding_space_version = 'moss_image'
        caption = 'cat'
        crop_x = crop_y = 0
        crop_width = crop_height = 8
        dataset_config = NS(random_crop=False, random_scale=False, load_image_when_caching_latents=False,
                            cache_tensors_to_disk=False)
        prompt_embeds = PromptEmbeds(torch.tensor([[[1., 2.]]], dtype=torch.bfloat16))
        def __getattr__(self, name): return None
        def get_latent(self): return torch.ones(2, 2, 2)
        def load_and_process_image(self, *args, **kwargs): self.tensor = Image.new('RGB', (8, 8), (1, 2, 3))

    item = CachedItem()
    dataset = NS(file_list=[item], _get_single_item=lambda index: item)
    trainer = make_trainer(tmp_path)
    trainer.data_loader = NS(dataset=dataset)
    trainer.logging_config = NS(training_maps_images=1)
    trainer.trigger_word = None
    monkeypatch.setattr(module, '_check_supported', lambda trainer: None)
    maps = TrainingMaps(trainer)
    maps.setup(trainer)
    assert maps.probes[0]['embeds'].text_embeds.dtype == torch.bfloat16
    assert maps.manifest['probes'][0]['crop'] == [0, 0, 8, 8]
    assert (tmp_path / 'training_maps/probe-0.png').exists()
    maps.capture(trainer, 0)
    maps.capture(trainer, 10)
    trainer.start_step = 10
    resumed = TrainingMaps(trainer)
    resumed.setup(trainer)
    assert resumed.manifest['steps'] == [0, 10]
    assert resumed.baseline_step == 0
    assert ('probe-0', '0.25') in resumed.baseline
    resumed.capture(trainer, 20)
    resumed.manifest['scales']['probe-0']['0.25']['effect'] = 1000
    (tmp_path / 'training_maps/manifest.json').write_text(json.dumps(resumed.manifest))
    rewind = TrainingMaps(trainer)
    rewind.setup(trainer)
    assert rewind.manifest['steps'] == [0, 10]
    assert not (tmp_path / 'training_maps/step-000000020.json').exists()
    assert rewind.manifest['scales']['probe-0']['0.25']['effect'] < 1000
    trainer.model_config.name_or_path = 'different-model'
    changed = TrainingMaps(trainer)
    changed.setup(trainer)
    assert changed.manifest['steps'] == []
    assert not (tmp_path / 'training_maps/step-000000000.json').exists()
    changed.capture(trainer, 10)
    trainer.start_step = 0
    fresh = TrainingMaps(trainer)
    fresh.setup(trainer)
    assert fresh.manifest['steps'] == []
    assert not (tmp_path / 'training_maps/step-000000010.json').exists()


def test_request_guards_reject_specialized_and_multi_process_before_load():
    import pytest
    from toolkit.training_maps import validate_request

    with pytest.raises(ValueError, match='ordinary diffusion'):
        validate_request(NS())
    trainer = type('SDTrainer', (), {})()
    trainer.accelerator = NS(num_processes=2)
    with pytest.raises(ValueError, match='single training process'):
        validate_request(trainer)
