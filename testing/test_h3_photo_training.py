import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest
import torch

from testing.test_h3_quant_cache import H3_MODULE, make_h3_model


def validation_function():
    # Importing the whole process imports all diffusion plugins, including CUDA-only
    # Triton code. Execute the actual isolated guard without loading those models.
    path = Path(__file__).resolve().parents[1] / 'jobs/process/BaseSDTrainProcess.py'
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_validate_h3_photo_training')
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function.name]


validate = validation_function()


def process():
    return NS(get_conf=lambda key: 'diffusion_trainer', model_config=NS(arch='minimax_h3', model_kwargs={'h3_photo_training': True}, assistant_lora_path='helper.safetensors'),
              network_config=NS(type='lora', transformer_only=True, all_layers=False, network_kwargs={}),
              train_config=NS(batch_size=1, train_unet=True, train_text_encoder=False, train_refiner=False,
                              noise_scheduler='flowmatch', timestep_type='shift', content_or_style='balanced', content_or_style_reg='balanced', linear_timesteps=False, linear_timesteps2=False, first_timestep_chance=0, min_denoising_steps=0, max_denoising_steps=999, num_train_timesteps=1000, loss_target='noise', do_prior_divergence=False, train_turbo=False, inverted_mask_prior=False, correct_pred_norm=False, do_cfg=False, do_random_cfg=False, blank_prompt_preservation=False, do_guidance_loss=False,
                              do_differential_guidance=False, frequency_loss_enabled=False, diff_output_preservation=False),
              adapter_config=None, guidance_config=None,
              dataset_configs=[NS(type='image', num_frames=1, auto_frame_count=False, batch_size=1,
                                  do_i2v=False, do_audio=False, h3_v2v_path=None, h3_reference_path=None,
                                  control_path=None, control_from_same_folder=False, controls=[])])


def test_opt_in_filter_and_disabled_preserves_custom_settings():
    p = process()
    validate(p)
    assert p.network_config.network_kwargs['only_if_contains'] == [f'blocks.{i}.' for i in range(20, 50)]
    assert not any('refiner' in name for name in p.network_config.network_kwargs['only_if_contains'])
    p.model_config.model_kwargs['h3_photo_training'] = False
    p.network_config.network_kwargs = {'only_if_contains': ['custom']}
    before = copy.deepcopy(p)
    validate(p)
    assert p == before


@pytest.mark.parametrize('component,field,value', [
    ('train_config', 'batch_size', 2), ('train_config', 'train_refiner', True),
    ('train_config', 'content_or_style', 'style'), ('train_config', 'timestep_type', 'linear'), ('network_config', 'type', 'lokr'),
    ('model_config', 'arch', 'qwen_image_2'), ('model_config', 'assistant_lora_path', None),
    ('dataset', 'num_frames', 17), ('dataset', 'h3_reference_path', '/reference'),
    ('dataset', 'do_audio', True), ('dataset', 'control_path', '/control'),
])
def test_invalid_photo_jobs_rejected(component, field, value):
    p = process()
    target = p.dataset_configs[0] if component == 'dataset' else getattr(p, component)
    setattr(target, field, value)
    with pytest.raises(ValueError, match='h3_photo_training'): validate(p)


def test_scheduler_density_default_and_inference_constants():
    model = H3_MODULE.MinimaxH3Model
    photo = model.get_train_scheduler(2 / 3)
    regular = model.get_train_scheduler()
    for scheduler in (photo, regular): scheduler.set_train_timesteps(10000, device='cpu', timestep_type='shift')
    fraction = (photo.timesteps < 500).float().mean().item()
    assert abs(fraction - .6) < .002
    assert regular.config.shift == 12
    assert model.video_sigma_shift == 12
    assert H3_MODULE.packing.AUDIO_SIGMA_SHIFT == 3
    assert (regular.timesteps < 500).float().mean().item() < .1


def test_photo_loader_requires_strict_frozen_helper(tmp_path):
    path = tmp_path / 'helper.safetensors'
    path.write_bytes(b'placeholder')
    model = make_h3_model()
    del model.load_training_adapter
    model.model_config.assistant_lora_path = str(path)
    model.model_config.model_kwargs['h3_photo_training'] = True
    transformer = torch.nn.Linear(2, 2)
    helper = NS()
    with patch('toolkit.assistant_lora.load_assistant_lora_from_path', return_value=helper) as load:
        model.load_training_adapter(transformer)
    assert model.assistant_lora is helper
    assert load.call_args.kwargs['strict'] is True
    assert load.call_args.kwargs['is_ara'] is True


def test_strict_h3_helper_real_factors_alpha_and_rejects_unmatched(tmp_path):
    from safetensors.torch import save_file
    from toolkit.assistant_lora import load_assistant_lora_from_path

    class MiniMaxH3Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].proj = torch.nn.Linear(2, 2, bias=False)
            self.token_refiner = torch.nn.Linear(2, 2, bias=False)
            self.requires_grad_(False)
            self.blocks[0].proj.weight.data.zero_()
            self.token_refiner.weight.data.zero_()

    model = make_h3_model()
    model.model = MiniMaxH3Transformer()
    model.is_transformer = True
    model.arch = 'minimax_h3'
    model.pipeline = NS()
    model.text_encoder = []
    p, q = 'diffusion_model.blocks.0.proj', 'diffusion_model.token_refiner'
    weights = {p + '.lora_A.weight': torch.ones(1, 2), p + '.lora_B.weight': torch.ones(2, 1), p + '.alpha': torch.tensor(.5),
               q + '.lora_A.weight': torch.eye(2), q + '.lora_B.weight': torch.eye(2), q + '.alpha': torch.tensor(4.)}
    path = tmp_path / 'helper.safetensors'
    save_file(weights, str(path))
    with patch('toolkit.assistant_lora.print_acc'):
        helper = load_assistant_lora_from_path(str(path), model, strict=True, is_ara=True)
    assert sorted(lora.lora_dim for lora in helper.unet_loras) == [1, 2]
    torch.testing.assert_close(model.model.blocks[0].proj(torch.ones(1, 2)), torch.ones(1, 2))
    torch.testing.assert_close(model.model.token_refiner(torch.ones(1, 2)), torch.full((1, 2), 2.))
    assert all(not p.requires_grad for p in helper.parameters())
    helper.is_active = False
    torch.testing.assert_close(model.model.blocks[0].proj(torch.ones(1, 2)), torch.zeros(1, 2))
    weights['diffusion_model.blocks.0.unknown.lora_A.weight'] = torch.ones(1, 2)
    weights['diffusion_model.blocks.0.unknown.lora_B.weight'] = torch.ones(2, 1)
    save_file(weights, str(path))
    model.model = MiniMaxH3Transformer()
    with patch('toolkit.assistant_lora.print_acc'), pytest.raises(ValueError, match='unmatched'):
        load_assistant_lora_from_path(str(path), model, strict=True, is_ara=True)


def test_published_yaml_recipe_and_ordinary_process_types():
    import yaml
    from toolkit.config_modules import DatasetConfig, ModelConfig, NetworkConfig, TrainConfig

    config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/examples/train_lora_minimax_h3_image.yaml').read_text())['config']['process'][0]
    p = NS(model_config=ModelConfig(**config['model']), network_config=NetworkConfig(**config['network']),
           train_config=TrainConfig(**config['train']), dataset_configs=[DatasetConfig(**item) for item in config['datasets']],
           adapter_config=None, guidance_config=None)
    for kind in ('diffusion_trainer', 'sd_trainer', 'ui_trainer'):
        p.get_conf = lambda key, kind=kind: kind
        validate(p)
    p.get_conf = lambda key: 'flow_dpo'
    with pytest.raises(ValueError, match="ordinary"): validate(p)
    p.model_config.model_kwargs['h3_photo_training'] = False
    validate(p)
