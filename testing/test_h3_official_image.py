"""CPU contracts for the official H3 Image model, VAE, sampling and workflow."""
import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Avoid executing the global model registry, which imports unrelated CUDA-only
# OmniGen Triton kernels. These are the real image/video VAE source files.
video = load_file('h3_image_test_video_vae', ROOT / 'extensions_built_in/diffusion_models/minimax_h3/src/vae.py')
with patch.dict(sys.modules, {'extensions_built_in.diffusion_models.minimax_h3.src.vae': video}):
    image = load_file('h3_image_test_vae', ROOT / 'extensions_built_in/diffusion_models/minimax_h3_image/src/vae.py')


def minimal_vae():
    vae = image.MiniMaxH3ImageVAE.__new__(image.MiniMaxH3ImageVAE)
    torch.nn.Module.__init__(vae)
    vae.latent_channels = 2
    vae.register_parameter('anchor', torch.nn.Parameter(torch.zeros(())))
    vae.register_buffer('latents_mean', torch.tensor([.5, -1.]))
    vae.register_buffer('latents_std', torch.tensor([2., 4.]))
    vae.register_buffer('pixel_mean', torch.tensor(video.IMAGENET_MEAN).view(1, 3, 1, 1, 1))
    vae.register_buffer('pixel_std', torch.tensor(video.IMAGENET_STD).view(1, 3, 1, 1, 1))
    return vae


def test_2d_rope_and_image_decoder_spatial_shape_with_gradients():
    decoder = video.ViTDecoder3d(in_channels=2, out_channels=3, patch_size=2, patch_size_t=1,
                                  num_layers=1, heads=2, head_dim=8, rope_dim_ratio=.5, rope_axes=2)
    coords = torch.tensor([[[.25, -.5]]])
    cos, sin = decoder.rope(coords)
    assert cos.shape == sin.shape == (1, 1, 1, 4)
    z = torch.randn(1, 2, 1, 2, 3, requires_grad=True)
    result = decoder(z)
    assert result.shape == (1, 3, 1, 4, 6)
    result.square().mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_image_normalization_and_reference_fixed_posterior_seed():
    vae = minimal_vae()
    seen = []
    def encode_clip(x):
        seen.append(x.clone())
        return torch.cat((torch.ones(1, 2, 1, 2, 2), torch.zeros(1, 2, 1, 2, 2)), 1)
    vae._encode_clip = encode_clip
    pixels = torch.zeros(1, 3, 2, 2)
    mean = vae.encode(pixels, sample=False)
    torch.testing.assert_close(seen[0][0, :, 0, 0, 0], (torch.tensor([.5] * 3) - torch.tensor(video.IMAGENET_MEAN)) / torch.tensor(video.IMAGENET_STD))
    torch.testing.assert_close(mean[0, :, 0, 0, 0], torch.tensor([.25, .5]))
    one = vae.encode_reference(pixels)
    two = vae.encode_reference(pixels)
    expected_noise = torch.randn((1, 2, 1, 2, 2), generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(one, (1 + expected_noise - vae.latents_mean.view(1, 2, 1, 1, 1)) / vae.latents_std.view(1, 2, 1, 1, 1))
    torch.testing.assert_close(one, two)
    vae._decode_clip = lambda z: torch.zeros(1, 3, 1, 2, 2)
    decoded = vae.decode(mean)
    torch.testing.assert_close(decoded[0, :, 0, 0, 0], torch.tensor(video.IMAGENET_MEAN) * 2 - 1)
    with pytest.raises(ValueError, match='expects RGB'): vae.encode(torch.zeros(1, 3, 2, 2, 2))
    with pytest.raises(ValueError, match='expects latents'): vae.decode(torch.zeros(1, 2, 2, 2, 2))


def test_tile_blending_preserves_constant_and_expected_spatial_size():
    vae = minimal_vae()
    vae.spatial_compression = 2
    vae.tile_size = 8
    vae.tile_overlap_min = 2
    vae.use_tiling = True
    vae._decode_pixels = lambda z: torch.nn.functional.interpolate(z, scale_factor=(1, 2, 2), mode='nearest')
    result = vae._decode_clip(torch.ones(1, 2, 1, 6, 7))
    assert result.shape == (1, 2, 1, 12, 14)
    torch.testing.assert_close(result, torch.ones_like(result))


def isolated_h3_image_modules():
    base = ROOT / 'extensions_built_in/diffusion_models'
    packages = {
        'h3_test': base, 'h3_test.diffusion_models': base,
        'h3_test.diffusion_models.minimax_h3': base / 'minimax_h3',
        'h3_test.diffusion_models.minimax_h3.src': base / 'minimax_h3/src',
        'h3_test.diffusion_models.minimax_h3_image': base / 'minimax_h3_image',
        'h3_test.diffusion_models.minimax_h3_image.src': base / 'minimax_h3_image/src',
    }
    for name, path in packages.items():
        package = types.ModuleType(name); package.__path__ = [str(path)]; sys.modules[name] = package
    original = load_file('h3_test.diffusion_models.minimax_h3.src.transformer', base / 'minimax_h3/src/transformer.py')
    load_file('h3_test.diffusion_models.minimax_h3.src.packing', base / 'minimax_h3/src/packing.py')
    load_file('h3_test.diffusion_models.minimax_h3.src.text_encoder', base / 'minimax_h3/src/text_encoder.py')
    transformer = load_file('h3_test.diffusion_models.minimax_h3_image.src.transformer', base / 'minimax_h3_image/src/transformer.py')
    pipeline = load_file('h3_test.diffusion_models.minimax_h3_image.src.pipeline', base / 'minimax_h3_image/src/pipeline.py')
    return original, transformer, pipeline


original, image_transformer, image_pipeline = isolated_h3_image_modules()


def tiny_image_model(pdd_steps=1):
    p = original.MiniMaxH3TransformerParams(hidden_size=24, num_layers=1, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=12, ffn_hidden_size=32, latents_dim=2,
        audio_latents_dim=2, text_dim=8, timestep_input_dim=8, time_embed_hidden_size=16,
        time_embed_dim=8, rope_inv_freq_len=2)
    p.pdd_steps = pdd_steps
    return image_transformer.MiniMaxH3ImageTransformer(p)


def test_pdd_fused_full_and_packed_offset_head_math():
    model = tiny_image_model(6)
    rows = model.params.latents_dim * 4
    with torch.no_grad():
        for i in range(6):
            model.final_layer.video_out.weight[i*rows:(i+1)*rows].fill_(i + 1)
            model.final_layer.video_out.bias[i*rows:(i+1)*rows].fill_(i / 10)
    hidden = torch.ones(1, 1, model.params.hidden_size)
    train = model._head(hidden, None)
    torch.testing.assert_close(train, torch.full_like(train, 24.))
    for i in range(6):
        expected = 24. if i == 0 else 24. + 24. * (i + 1) + i / 10
        torch.testing.assert_close(model._head(hidden, i), torch.full_like(train, expected))
    weight = torch.stack([torch.full((rows, 24), float(i + 1)) for i in range(6)])
    bias = torch.stack([torch.full((rows,), float(i)) for i in range(6)])
    model.attach_pdd_fused_heads(weight, bias)
    for i in range(6): torch.testing.assert_close(model._head(hidden, i), torch.full_like(train, 24. * (i + 1) + i))
    torch.testing.assert_close(model._head(hidden, None), train)
    with pytest.raises(ValueError, match='step'): model._head(hidden, 6)
    with pytest.raises(ValueError, match='dimensions'): model.attach_pdd_fused_heads(weight[:-1], bias)


def test_tiny_image_transformer_checkpoint_forward_gradient_no_audio():
    model = tiny_image_model()
    model.gradient_checkpointing = True
    kwargs = dict(hidden_states=torch.randn(1, 2, 8, requires_grad=True), encoder_hidden_states=torch.randn(1, 2, 8),
                  row_timesteps=torch.tensor([[.4, .4, .4, .4]]), token_tags=torch.tensor([[1, 1, 0, 0]]),
                  position_ids=torch.zeros(1, 4, 3), video_indices=torch.tensor([2, 3]), text_indices=torch.tensor([0, 1]))
    result = model(**kwargs)
    assert result.shape == (1, 2, 8)
    result.square().mean().backward()
    assert kwargs['hidden_states'].grad.abs().sum() > 0
    assert model.blocks[0].attn.qkv_proj.weight.grad is not None
    kwargs['token_tags'] = torch.tensor([[1, 1, 2, 0]])
    with pytest.raises(ValueError, match='audio'): model(**kwargs)


def test_six_step_pdd_grid_and_sign_of_euler_update():
    from types import SimpleNamespace as NS
    calls = []
    class StubTransformer:
        def __call__(self, **kwargs):
            calls.append((kwargs['pdd_step'], kwargs['row_timesteps'].clone()))
            return torch.ones_like(kwargs['hidden_states']) * .25
    model = NS(device_torch=torch.device('cpu'), torch_dtype=torch.float32, max_text_length=100,
               transformer=StubTransformer(), decode_latents=lambda z: z[:, :3])
    pipe = image_pipeline.MiniMaxH3ImagePipeline(model)
    pipe.progress_bar = lambda x: x
    embeds = NS(text_embeds=[torch.zeros(2, 8)], text_token_tags=[torch.ones(2, dtype=torch.long)])
    captured = []
    model.decode_latents = lambda z: (captured.append(z.clone()) or z[:, :3])
    result = pipe(embeds, height=32, width=32, latents=torch.zeros(1, 24, 1, 2, 2))
    assert len(result) == 1 and result[0].size == (2, 2)
    assert [item[0] for item in calls] == list(range(6))
    assert image_pipeline.PDD_SIGMAS == (1., .9675236940383911, .8875485062599182, .75, .5327739119529724, .23200887441635132, 0.)
    torch.testing.assert_close(captured[0], torch.full_like(captured[0], .25))
    with pytest.raises(ValueError, match='six PDD steps'): pipe(embeds, height=32, width=32, num_inference_steps=4)


def test_checkpoint_header_rejects_audio_and_invalid_head_rows():
    cls = image_transformer.MiniMaxH3ImageTransformer
    for rows, count in [(96, 1), (576, 6)]:
        params = cls.aitk_config_from_state_dict({'final_layer.video_out.weight': torch.empty(rows, 4)})
        assert params.pdd_steps == count
    with pytest.raises(ValueError, match='audio-free'):
        cls.aitk_config_from_state_dict({'audio_patch_proj.weight': torch.empty(1), 'final_layer.video_out.weight': torch.empty(96, 4)})
    with pytest.raises(ValueError, match='expected'):
        cls.aitk_config_from_state_dict({'final_layer.video_out.weight': torch.empty(97, 4)})


def test_tiny_lora_train_save_load_optimizer_resume():
    from toolkit.lora_special import LoRAModule
    from testing.test_loha import _Network
    torch.manual_seed(21)
    model = tiny_image_model().requires_grad_(False)
    network = _Network()
    network.network_type, network.is_lorm = 'lora', False
    adapter = LoRAModule('fixture', model.blocks[0].attn.qkv_proj, network=network, lora_dim=2, alpha=2)
    adapter.apply_to()
    opt = torch.optim.AdamW(adapter.parameters(), lr=.01)
    inputs = dict(hidden_states=torch.randn(1, 2, 8), encoder_hidden_states=torch.randn(1, 2, 8),
                  row_timesteps=torch.full((1, 4), .4), token_tags=torch.tensor([[1, 1, 0, 0]]),
                  position_ids=torch.zeros(1, 4, 3), video_indices=torch.tensor([2, 3]), text_indices=torch.tensor([0, 1]))
    model(**inputs).square().mean().backward()
    assert adapter.lora_up.weight.grad is not None
    opt.step(); opt.zero_grad()
    import copy
    checkpoint = {'weights': copy.deepcopy(adapter.state_dict()), 'optimizer': copy.deepcopy(opt.state_dict())}
    expected = model(**inputs).detach()
    with torch.no_grad(): adapter.lora_up.weight.zero_()
    adapter.load_state_dict(checkpoint['weights'])
    opt.load_state_dict(checkpoint['optimizer'])
    torch.testing.assert_close(model(**inputs), expected)
    model(**inputs).square().mean().backward(); opt.step()
    assert not torch.equal(adapter.lora_up.weight, checkpoint['weights']['lora_up.weight'])


def test_image_strict_helper_frozen_and_separate_from_trainable_adapter(tmp_path):
    from types import SimpleNamespace as NS
    from safetensors.torch import save_file
    from toolkit.assistant_lora import load_assistant_lora_from_path
    from toolkit.lora_special import LoRAModule
    from testing.test_loha import _Network

    base = tiny_image_model().requires_grad_(False)
    prefix = 'transformer.blocks.0.attn.qkv_proj'
    path = tmp_path / 'helper.safetensors'
    save_file({prefix + '.lora_A.weight': torch.ones(2, 24), prefix + '.lora_B.weight': torch.ones(72, 2)}, str(path))
    class ModelStub: pass
    sd = ModelStub()
    sd.convert_lora_weights_before_load = lambda weights: weights
    sd.__dict__.update(dict(use_old_lokr_format=False, is_flux=False, is_transformer=True, unet=base, model=base, pipeline=NS(), text_encoder=[],
            target_lora_modules=['MiniMaxH3ImageTransformer'], device_torch=torch.device('cpu'), torch_dtype=torch.float32,
            arch='minimax_h3_image'))
    with patch('toolkit.assistant_lora.print_acc'):
        helper = load_assistant_lora_from_path(str(path), sd, strict=True, is_ara=True)
    assert len(helper.unet_loras) == 1 and all(not p.requires_grad for p in helper.parameters())
    network = _Network(); network.network_type, network.is_lorm = 'lora', False
    train = LoRAModule('train', base.blocks[0].attn.qkv_proj, network=network, lora_dim=2, alpha=2)
    train.apply_to()
    assert set(map(id, train.parameters())).isdisjoint(map(id, helper.parameters()))
    assert all('helper' not in key for key in train.state_dict())
    save_file({prefix + '.lora_A.weight': torch.ones(2, 24)}, str(path))
    sd.unet = tiny_image_model().requires_grad_(False)
    with patch('toolkit.assistant_lora.print_acc'), pytest.raises(ValueError, match='incomplete'):
        load_assistant_lora_from_path(str(path), sd, strict=True, is_ara=True)


def test_comfy_image_workflow_render_and_rejects_incompatible_requests():
    from dataclasses import replace
    from toolkit.comfy_sample import ComfySampleRequest, get_workflow_for_sample, DEFAULT_COMFY_MINIMAX_H3_IMAGE_WORKFLOW_PATH
    req = ComfySampleRequest(prompt='a "photo"', width=1024, height=1024, steps=6, cfg=1, seed=42,
        model='h3-image.safetensors', vae='h3-image-vae.safetensors', text_encoder='qwen.safetensors',
        sampler='euler', scheduler='simple', inference_lora='', inference_lora_strength=1., output_format='png',
        output_quality='high', training_lora_path='/tmp/my.safetensors', training_lora_filename='my.safetensors',
        filename_prefix='fixture', model_arch='minimax_h3_image')
    graph = get_workflow_for_sample(DEFAULT_COMFY_MINIMAX_H3_IMAGE_WORKFLOW_PATH, req)
    assert graph['6']['class_type'] == 'MiniMaxH3ImageConditioning'
    assert graph['6']['inputs']['prompt'] == 'a "photo"'
    assert graph['6']['inputs']['clip'] == ['4', 1] and graph['6']['inputs']['vae'] == ['3', 0]
    assert graph['4']['inputs']['absolute_path'] == '/tmp/my.safetensors'
    assert graph['8']['inputs']['steps'] == 6 and graph['8']['inputs']['cfg'] == 1
    assert graph['8']['inputs']['model'] == ['4', 0]
    assert graph['9']['inputs']['samples'] == ['8', 0]
    for changed in [dict(steps=4), dict(cfg=2), dict(num_frames=5), dict(control_image='/ref'),
                    dict(sampler='heun'), dict(model_arch='minimax_h3')]:
        with pytest.raises(ValueError): get_workflow_for_sample(DEFAULT_COMFY_MINIMAX_H3_IMAGE_WORKFLOW_PATH, replace(req, **changed))
    other = get_workflow_for_sample(DEFAULT_COMFY_MINIMAX_H3_IMAGE_WORKFLOW_PATH, replace(req, inference_lora='extra.safetensors'))
    assert other['8']['inputs']['model'] == ['5', 0]


def test_checkpoint_header_metadata_validation(tmp_path):
    import ast
    import json
    import os
    import math
    from safetensors import safe_open
    from safetensors.torch import save_file
    from types import SimpleNamespace as NS

    path = ROOT / 'extensions_built_in/diffusion_models/minimax_h3_image/minimax_h3_image.py'
    tree = ast.parse(path.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MinimaxH3ImageModel')
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '_load_transformer')
    method.returns = None
    fake_loader = NS(load_model=lambda filename, dtype: ('loaded', filename, dtype))
    env = {'safe_open': safe_open, 'json': json, 'os': os, 'math': math, 'PDD_SIGMAS': image_pipeline.PDD_SIGMAS, 'MiniMaxH3ImageTransformer': fake_loader}
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_validate_pdd_grid', '_read_pdd_config')]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(path), 'exec'), env)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), env)
    load = env['_load_transformer']
    model = NS(torch_dtype=torch.float32, model_config=NS(model_kwargs={}), print_and_status_update=lambda msg: None)
    checkpoint = tmp_path / 'model.safetensors'
    def write(rows, metadata=None, **extra):
        save_file({'final_layer.video_out.weight': torch.zeros(rows, 1), **extra}, str(checkpoint),
                  metadata={'config': json.dumps({'transformer': metadata})} if metadata is not None else None)
    write(576, {'pdd_steps': 6, 'pdd_sigmas': list(image_pipeline.PDD_SIGMAS)})
    assert load(model, str(checkpoint))[0] == 'loaded'
    write(576, {'pdd_steps': 1})
    with pytest.raises(ValueError, match='conflicting PDD metadata'): load(model, str(checkpoint))
    write(576, {'pdd_steps': 6, 'pdd_sigmas': [1., 0.]})
    with pytest.raises(ValueError, match='sigma grid'): load(model, str(checkpoint))
    for invalid in [float('nan'), float('inf'), 'bad']:
        malformed = list(image_pipeline.PDD_SIGMAS); malformed[1] = invalid
        write(576, {'pdd_steps': 6, 'pdd_sigmas': malformed})
        with pytest.raises(ValueError, match='non-finite|nonnumeric'): load(model, str(checkpoint))
    write(96, None, **{'audio_patch_proj.weight': torch.zeros(1)})
    with pytest.raises(ValueError, match='audio-free'): load(model, str(checkpoint))


def test_local_comfy_conditioning_schema_accepts_rendered_inputs():
    import ast
    path = Path('/home/bart/ComfyUI/comfy_extras/nodes_minimax_h3.py')
    if not path.exists(): pytest.skip('local ComfyUI checkout unavailable')
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MiniMaxH3ImageConditioning')
    define = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'define_schema')
    names = {node.args[0].value for node in ast.walk(define)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'Input'
             and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)}
    assert {'clip', 'vae', 'prompt', 'width', 'height'} <= names


def test_official_image_vae_architecture_on_meta_device():
    with torch.device('meta'):
        vae = image.MiniMaxH3ImageVAE()
    assert len(vae.decoder.transformer_blocks) == 24
    assert vae.decoder.rope_axes == 2
    assert vae.decoder.patch_size_t == 1
    assert vae.decoder.transformer_blocks[0].attn.heads == 16
    assert isinstance(vae.post_quant_conv, torch.nn.Conv2d)
    assert vae.token_drop == 0


def test_published_recipe_guard_and_single_image_contract():
    import ast
    from types import SimpleNamespace as NS
    import yaml
    from toolkit.config_modules import DatasetConfig, ModelConfig, NetworkConfig, TrainConfig
    path = ROOT / 'jobs/process/BaseSDTrainProcess.py'
    tree = ast.parse(path.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('_validate_h3_image_training', '_validate_h3_photo_training')]
    env = {}; exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), env)
    config = yaml.safe_load((ROOT / 'config/examples/train_lora_minimax_h3_official_image.yaml').read_text())['config']['process'][0]
    p = NS(model_config=ModelConfig(**config['model']), network_config=NetworkConfig(**config['network']),
        train_config=TrainConfig(**config['train']), dataset_configs=[DatasetConfig(**x) for x in config['datasets']],
        adapter_config=None, guidance_config=None, get_conf=lambda name: 'diffusion_trainer')
    env['_validate_h3_image_training'](p); env['_validate_h3_photo_training'](p)
    p.dataset_configs[0].num_frames = 5
    with pytest.raises(ValueError, match='single-frame'): env['_validate_h3_image_training'](p)
    p.dataset_configs[0].num_frames = 1; p.dataset_configs[0].h3_reference_path = '/ref'
    with pytest.raises(ValueError, match='reference'): env['_validate_h3_image_training'](p)

    wrapper_path = ROOT / 'extensions_built_in/diffusion_models/minimax_h3_image/minimax_h3_image.py'
    tree = ast.parse(wrapper_path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MinimaxH3ImageModel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'generate_single_image')
    scope = {}; exec(compile(ast.Module(body=[method], type_ignores=[]), str(wrapper_path), 'exec'), scope)
    cfg = NS(num_frames=1, guidance_scale=1, num_inference_steps=6, height=32, width=32, latents=None)
    fake = NS(get_sample_control_image_paths=lambda cfg: [])
    sentinel = object()
    assert scope['generate_single_image'](fake, lambda **kwargs: [sentinel], cfg, None, None, None, None) is sentinel
    cfg.num_frames = 5
    with pytest.raises(ValueError, match='one image frame'):
        scope['generate_single_image'](fake, lambda **kwargs: [sentinel], cfg, None, None, None, None)


def test_reference_noise_uses_seeded_packed_fp32_rows():
    from PIL import Image
    from types import SimpleNamespace as NS
    packed_inputs = []
    class StubTransformer:
        def __call__(self, **kwargs):
            packed_inputs.append(kwargs['hidden_states'].clone())
            return torch.zeros_like(kwargs['hidden_states'])
    z = torch.arange(24 * 4, dtype=torch.float32).reshape(1, 24, 1, 2, 2) / 100
    model = NS(device_torch=torch.device('cpu'), torch_dtype=torch.bfloat16, max_text_length=100,
               transformer=StubTransformer(), encode_keyframe_latents=lambda pixels: z,
               decode_latents=lambda latents: latents[:, :3])
    pipe = image_pipeline.MiniMaxH3ImagePipeline(model); pipe.progress_bar = lambda x: x
    embeds = NS(text_embeds=[torch.zeros(2, 8)], text_token_tags=[torch.tensor([0, 1])])
    pipe(embeds, height=32, width=32, ctrl_img=Image.new('RGB', (32, 32)),
         generator=torch.Generator().manual_seed(123), latents=torch.zeros_like(z))
    rows = image_pipeline.patchify_video_latents(z)
    noise = torch.randn(rows.shape, generator=torch.Generator().manual_seed(123), dtype=torch.float32)
    expected = image_pipeline.KEYFRAME_NOISE_AUG_T * rows + (1 - image_pipeline.KEYFRAME_NOISE_AUG_T) * noise
    assert packed_inputs[0].dtype == torch.float32
    torch.testing.assert_close(packed_inputs[0][:, :rows.shape[1]], expected)


def test_downloaded_checkpoint_headers_match_meta_architectures():
    from safetensors import safe_open
    import json
    vae_path = ROOT / 'models/vae/minimax_h3_image_vae_fp16.safetensors'
    dit_path = ROOT / 'models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors'
    if not vae_path.exists() or not dit_path.exists(): pytest.skip('official checkpoint assets not present')
    with torch.device('meta'):
        vae = image.MiniMaxH3ImageVAE()
    with safe_open(str(vae_path), framework='pt', device='cpu') as header:
        file_keys = set(header.keys())
        meta = vae.state_dict()
        missing = set(meta) - file_keys
        extra = file_keys - set(meta) - {'latents_mean', 'latents_std'}
        assert not missing and not extra, (list(missing)[:5], list(extra)[:5])
        mismatches = [(key, tuple(t.shape), tuple(header.get_slice(key).get_shape())) for key, t in meta.items()
                      if tuple(t.shape) != tuple(header.get_slice(key).get_shape())]
        assert not mismatches, mismatches[:5]
    with safe_open(str(dit_path), framework='pt', device='cpu') as header:
        keys = set(header.keys())
        assert not {'audio_patch_proj.weight', 'final_layer.audio_out.weight'} & keys
        assert header.get_slice('final_layer.video_out.weight').get_shape() == [576, 5376]
        cfg = json.loads(header.metadata()['config'])['transformer']
        assert cfg['pdd_steps'] == 6
        assert cfg['pdd_sigmas'] == list(image_pipeline.PDD_SIGMAS)


def test_image_unconditional_empty_sentinels_and_user_text():
    import ast
    path = ROOT / 'extensions_built_in/diffusion_models/minimax_h3_image/minimax_h3_image.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MinimaxH3ImageModel')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'encode_unconditional_prompt')
    class Base:
        def encode_unconditional_prompt(self, prompt, *args, **kwargs): return prompt, args, kwargs
    subject = ast.ClassDef(name='Subject', bases=[ast.Name(id='Base', ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
    env = {'Base': Base}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[subject], type_ignores=[])), str(path), 'exec'), env)
    model = env['Subject']()
    for empty in (False, None): assert model.encode_unconditional_prompt(empty, 3, flag=True) == ('', (3,), {'flag': True})
    assert model.encode_unconditional_prompt('avoid text') == ('avoid text', (), {})
