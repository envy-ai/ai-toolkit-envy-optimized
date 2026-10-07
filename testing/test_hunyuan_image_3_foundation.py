"""Small canonical fixtures; real checkpoint qualification is separate."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save_file

from extensions_built_in.diffusion_models.hunyuan_image_3.src.config import INSTRUCT, BASE, pinned_config
from extensions_built_in.diffusion_models.hunyuan_image_3.src.transformer import HunyuanImage3, params_from_config, _split_qkv
from extensions_built_in.diffusion_models.hunyuan_image_3.src.tokenizer import build_sequence
from extensions_built_in.diffusion_models.hunyuan_image_3.src.conditioning import rebuild_target_ids, build_attention_mask
from toolkit.models.v2.diffusion_models.hunyuan_image_3 import HunyuanImage3Transformer
from toolkit.util.comfy_quant_import import import_comfy_quantized_layers
from toolkit.util.comfy_quant_export import export_comfy_quantized_layers, comfy_quant_marker
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds


def small_config():
    c = pinned_config()
    c.update(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, attention_head_dim=8, num_experts=3,
             moe_topk=2, moe_intermediate_size=16, num_shared_expert=1,
             patch_embed_hidden_dim=32, vocab_size=133120)
    c['image_token_id'] = FixtureTokenizer().token_to_id('<img>')
    c['vit_aligner'] = dict(projector_type='mlp_gelu', input_dim=8, n_embed=32, depth=2)
    return c


class FixtureTokenizer:
    def __init__(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.src.tokenizer import META_TOKENS
        self.tokens = {text: index + 500 for index, text in enumerate(META_TOKENS.values())}
        self.tokens.update({f'<img_size_{size}>': 600 + index for index, size in enumerate([256,512,768,1024,1536,2048,3072,4096,8192])})
        self.tokens.update({f'<img_ratio_{index}>': 700 + index for index in range(37)})
    def token_to_id(self, text): return self.tokens.get(text)
    def encode(self, text, add_special_tokens=False):
        if text in self.tokens: return SimpleNamespace(ids=[self.tokens[text]])
        return SimpleNamespace(ids=[ord(char) for char in text])


class HunyuanFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = FixtureTokenizer()

    def test_qkv_interleaved_by_kv_head(self):
        q, k, v = _split_qkv(torch.arange(16.).reshape(1,1,16), 2, 2, 2)
        self.assertEqual(q.flatten().tolist(), [0,1,2,3,8,9,10,11])
        self.assertEqual(k.flatten().tolist(), [4,5,12,13])
        self.assertEqual(v.flatten().tolist(), [6,7,14,15])

    def sequence(self, refs=(), variant=INSTRUCT):
        return build_sequence(self.tokenizer, 'a red circle', (32,48), variant.system_prompt,
                              cfg_distilled=False, use_meanflow=False, cond_images=refs,
                              reference_vit_padding=True, sequence_template=variant.sequence_template,
                              extra_rows=variant.supports_edit)

    def test_target_rebuild_and_variants(self):
        for variant in (INSTRUCT, BASE):
            original = self.sequence(variant=variant)
            ids = rebuild_target_ids(original['ids'], (3,4), self.tokenizer, variant)
            expected = build_sequence(self.tokenizer, 'a red circle', (48,64), variant.system_prompt,
                                      cfg_distilled=False, use_meanflow=False,
                                      sequence_template=variant.sequence_template, extra_rows=variant.supports_edit)['ids']
            torch.testing.assert_close(ids, expected)
        self.assertGreater(len(self.sequence()['ids']), len(self.sequence(variant=BASE)['ids']))

    def test_reference_order_mask_and_checkpoint_gradients(self):
        torch.manual_seed(10)
        model = HunyuanImage3(params_from_config(small_config()), dtype=torch.float32)
        model.requires_grad_(False)
        model.model.layers[0].self_attn.o_proj.weight.requires_grad_(True)
        refs = [((32,48),(1,2)), ((48,32),(2,1))]
        seq = self.sequence(refs)
        from extensions_built_in.diffusion_models.hunyuan_image_3.src.transformer import _sequence_from_ids
        geometry = _sequence_from_ids(seq['ids'], torch.zeros(1,32,2,3), model.config,
                                      [torch.zeros(1,32,2,3),torch.zeros(1,32,3,2)], [(1,2),(2,1)])
        mask = build_attention_mask(geometry, len(seq['ids']), torch.float32, 'cpu')
        a,b = geometry['full_attention_slices'][:2]
        self.assertTrue(torch.isneginf(mask[0,0,a,b]).all())
        self.assertTrue((mask[0,0,b,a] == 0).all())
        kwargs = dict(ids=seq['ids'], cond_latent=[torch.randn(1,32,2,3),torch.randn(1,32,3,2)],
                      cond_vit=[torch.randn(1,1024,8),torch.randn(1,1024,8)],cond_vit_grid=[(1,2),(2,1)])
        # Exercise actual reference embedding assembly at mixed time precision.
        model.to(torch.bfloat16)
        self.assertTrue(model(torch.randn(1,32,2,3,dtype=torch.bfloat16),torch.tensor([500.]),**kwargs).isfinite().all())
        model.to(torch.float32)
        # Long system/reference scaffolding is costly on CPU; shorten only for the
        # gradient fixture, preserving the same block geometry in an actual builder.
        short = build_sequence(self.tokenizer, 'red', (32,48), '', cfg_distilled=False,
                               use_meanflow=False, reference_vit_padding=False)
        kwargs = dict(ids=short['ids'])
        latent = torch.randn(1,32,2,3)
        plain = model(latent, torch.tensor([500.]), **kwargs)
        plain.square().mean().backward()
        gradient = model.model.layers[0].self_attn.o_proj.weight.grad.clone()
        model.zero_grad(set_to_none=True)
        model.enable_gradient_checkpointing()
        replay = model(latent, torch.tensor([500.]), **kwargs)
        replay.square().mean().backward()
        torch.testing.assert_close(plain, replay)
        torch.testing.assert_close(gradient, model.model.layers[0].self_attn.o_proj.weight.grad)
        self.assertTrue(gradient.isfinite().all()); self.assertGreater(gradient.abs().sum(),0)

    def test_streamed_bf16_and_quantized_canonical_loader(self):
        torch.manual_seed(2)
        c=small_config()
        model=HunyuanImage3(params_from_config(c),dtype=torch.float32)
        with tempfile.TemporaryDirectory() as directory:
            file=str(Path(directory)/'tiny.safetensors')
            save_file(model.state_dict(),file)
            from toolkit.models.v2.pool import ComponentPool
            pool=ComponentPool()
            with patch.object(ComponentPool,'current',pool):
                loaded=HunyuanImage3Transformer.load_model(file,config=c,dtype=torch.float32)
                self.assertIs(loaded,HunyuanImage3Transformer.load_model(file,config=c,dtype=torch.float32))
                self.assertEqual(pool.hits,1)
                self.assertFalse(any(p.is_meta for p in loaded.parameters()))
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,loaded.state_dict()[key])
                pool.clear()

    def test_w4a8_import_export_and_surrogate(self):
        root=torch.nn.Sequential(torch.nn.Linear(256,16,bias=False))
        book=torch.linspace(-1,1,16)
        state={'0.weight':torch.randint(-128,127,(16,128),dtype=torch.int8),
               '0.weight_s_rel':torch.ones(16,16,dtype=torch.float8_e4m3fn),
               '0.weight_s_channel':torch.ones(16)*.03,
               '0.weight_codebook':book,
               '0.comfy_quant':comfy_quant_marker(dict(format='asym_w4a8_int8',group_size=16,convrot_groupsize=256))}
        remaining,count=import_comfy_quantized_layers(root,state,orig_dtype=torch.float32)
        self.assertFalse(remaining);self.assertEqual(count,1)
        x=torch.randn(3,256,requires_grad=True)
        root(x).sum().backward()
        expected=torch.ones(3,16) @ root[0].dequantize_weight()
        torch.testing.assert_close(x.grad,expected,atol=1e-5,rtol=1e-5)
        exported,names,bad=export_comfy_quantized_layers(root)
        self.assertFalse(bad); self.assertEqual(names,['0'])
        for key in state:
            if not key.endswith('comfy_quant'):
                torch.testing.assert_close(exported[key].view(torch.uint8),state[key].view(torch.uint8))

    def test_cache_roundtrip_keeps_ids_and_rebuilds_nonsquare_target(self):
        sequence = build_sequence(self.tokenizer, 'red', (512,512), '',
                                  cfg_distilled=False, use_meanflow=False)
        pe = AdvancedPromptEmbeds(ids=sequence['ids'], text_embeds=sequence['ids'][:,None].clone(),
                                  reference_count=torch.tensor(0),
                                  reference_grids=torch.empty(0,2,dtype=torch.long))
        pe.frozen_dtype_keys = pe.keys()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/'cache.safetensors')
            pe.save(path)
            cached = AdvancedPromptEmbeds.load(path).to(dtype=torch.bfloat16)
        self.assertEqual(cached.ids[0].dtype, torch.long)
        self.assertEqual(cached.reference_grids[0].dtype, torch.long)
        rebuilt = rebuild_target_ids(cached.ids[0], (32,48), self.tokenizer, INSTRUCT)
        expected = build_sequence(self.tokenizer, 'red', (512,768), '',
                                  cfg_distilled=False,use_meanflow=False)['ids']
        torch.testing.assert_close(rebuilt, expected)

    def test_zero_to_three_ordered_references_and_blank_keep_slots(self):
        from PIL import Image
        from extensions_built_in.diffusion_models.hunyuan_image_3.hunyuan_image_3 import HunyuanImage3InstructModel
        holder = object.__new__(HunyuanImage3InstructModel)
        holder.model_config = SimpleNamespace(model_kwargs={'reference_max_pixels':256})
        holder.device_torch, holder.torch_dtype = torch.device('cpu'), torch.float32
        holder.vae = SimpleNamespace(device=torch.device('cpu'), to=Mock())
        holder.conditioning_component = SimpleNamespace(tokenizer=self.tokenizer, vision_model=None)
        holder._reference_cache = None
        holder.encode_images = Mock(side_effect=lambda images, **kwargs: images[0].mean().expand(1,32,1,1).clone())
        holder.conditioning_component.encode_reference = Mock(side_effect=lambda image,*args:
            (torch.tensor(list(image.getpixel((0,0))),dtype=torch.float32).mean().expand(1,1024,8).clone(),torch.tensor([[1,1]])))
        refs=[Image.new('RGB',(16,16), color) for color in ('red','white','blue')]
        for count in range(4):
            positive = holder.get_prompt_embeds('red', control_images=refs[:count])
            blank = holder.get_prompt_embeds('', control_images=refs[:count])
            self.assertEqual(int(positive.reference_count[0]),count)
            self.assertEqual(int(blank.reference_count[0]),count)
            for slot in range(count):
                torch.testing.assert_close(positive[f'reference_latent_{slot}'][0],blank[f'reference_latent_{slot}'][0])
                torch.testing.assert_close(positive[f'reference_vision_{slot}'][0],blank[f'reference_vision_{slot}'][0])
        # Repeated source slots are retained, and reversing sources reverses outputs.
        duplicate = holder.get_prompt_embeds('red', control_images=[refs[0],refs[0]])
        self.assertEqual(int(duplicate.reference_count[0]),2)
        forward = holder.get_prompt_embeds('red',control_images=refs[:2])
        reverse = holder.get_prompt_embeds('red',control_images=list(reversed(refs[:2])))
        torch.testing.assert_close(forward.reference_latent_0[0],reverse.reference_latent_1[0])
        self.assertFalse(torch.equal(forward.reference_latent_0[0],reverse.reference_latent_0[0]))

        # The shared unloader destroys all tensor storage, including placement.
        from toolkit.models.v2.text_encoders.hunyuan_image_3 import HunyuanImage3Conditioning
        destroyed=HunyuanImage3Conditioning(self.tokenizer,vision=torch.nn.Linear(1,1)).to('meta')
        destroyed.encode_reference=holder.conditioning_component.encode_reference
        holder.conditioning_component=destroyed;holder._reference_cache=None
        holder._conditioning_assets='fixture';holder._conditioning_revision='fixture_revision'
        replacement=HunyuanImage3Conditioning(self.tokenizer,vision=torch.nn.Linear(1,1))
        with patch.object(HunyuanImage3Conditioning,'create',return_value=replacement):
            holder.get_prompt_embeds('new reference',control_images=[refs[1]])
        holder.text_encoder[0].to('cpu')
        self.assertEqual(holder.text_encoder[0].device.type,'cpu')

    def test_native_pipeline_cpu_generator_and_euler_cfg(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.src.pipeline import HunyuanImage3Pipeline
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        holder=SimpleNamespace(model=None,vae=None,text_encoder=[None],noise_scheduler=None,
                               device_torch=device,variant=INSTRUCT,
                               get_noise_prediction=lambda latents,t,branch:torch.ones_like(latents)*branch,
                               decode_to_images=lambda latents:latents,_emit_sample_step=Mock())
        output=HunyuanImage3Pipeline(holder)(2,1,16,16,num_inference_steps=2,guidance_scale=2.5,
                                           generator=torch.Generator('cpu').manual_seed(42))
        initial=torch.randn(1,32,1,1,generator=torch.Generator('cpu').manual_seed(42))
        torch.testing.assert_close(output.cpu(),initial-3.5)
        self.assertEqual(holder._emit_sample_step.call_count,2)

    def test_sampling_failure_restores_placement_and_rng(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.hunyuan_image_3 import HunyuanImage3InstructModel
        from toolkit.models.base_model import BaseModel
        holder=object.__new__(HunyuanImage3InstructModel)
        holder.vae=SimpleNamespace(to=Mock());holder.conditioning_component=None
        holder.network=SimpleNamespace(train=Mock(),is_merged_in=False)
        holder.restore_device_state=Mock()
        state=torch.get_rng_state().clone()
        def fail(*args,**kwargs):
            torch.randn(1)
            raise RuntimeError('injected sample failure')
        with patch.object(BaseModel,'generate_images',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                holder.generate_images([])
        torch.testing.assert_close(torch.get_rng_state(),state)
        holder.vae.to.assert_called_with('cpu');holder.restore_device_state.assert_called_once()
        holder.network.train.assert_called_once()

    def test_conditioning_unload_discards_reloaded_unused_vae(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.hunyuan_image_3 import HunyuanImage3InstructModel
        from toolkit.models.v2.text_encoders.hunyuan_image_3 import HunyuanImage3Conditioning
        from toolkit.unloader import unload_text_encoder
        holder=object.__new__(HunyuanImage3InstructModel)
        component=HunyuanImage3Conditioning(self.tokenizer)
        holder.text_encoder=[component];holder.conditioning_component=component
        vae=torch.nn.Linear(1,1)
        holder.vae=vae;holder.pipeline=SimpleNamespace(text_encoder=component,vae=vae)
        holder.device_torch=torch.device('cpu');holder.torch_dtype=torch.float32
        holder._release_vae_after_conditioning=True
        unload_text_encoder(holder)
        self.assertTrue(vae.weight.is_meta)
        self.assertIsNone(holder.vae);self.assertIsNone(holder.pipeline.vae)

    def test_banked_w4a8_codebooks_shared_and_per_expert(self):
        c = small_config(); c['num_hidden_layers']=1
        state = HunyuanImage3(params_from_config(c),dtype=torch.float32).state_dict()
        for projection, out_features, in_features in [('gate_and_up_proj',32,32),('down_proj',32,16)]:
            bank='model.layers.0.mlp.'+('experts_gate_up_proj' if projection=='gate_and_up_proj' else 'experts_down_proj')
            for expert in range(3):
                del state[f'model.layers.0.mlp.experts.{expert}.{projection}.weight']
            state[bank+'.weight']=torch.zeros(3,out_features,in_features//2,dtype=torch.int8)
            state[bank+'.weight_s_rel']=torch.ones(3,out_features,in_features//16,dtype=torch.float8_e4m3fn)
            state[bank+'.weight_s_channel']=torch.ones(3,out_features)
            state[bank+'.comfy_quant']=comfy_quant_marker(dict(format='asym_w4a8_int8',num_experts=3,group_size=16,convrot_groupsize=4))
            state[bank+'.weight_codebook'] = (torch.arange(16,dtype=torch.float32).repeat(3,1)+torch.arange(3)[:,None]
                                            if projection=='gate_and_up_proj' else torch.arange(16,dtype=torch.float32))
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'bank.safetensors'); save_file(state,path)
            model=HunyuanImage3Transformer.load_model(path,config=c,dtype=torch.float32)
        for index, expert in enumerate(model.model.layers[0].mlp.experts):
            self.assertEqual(expert.gate_and_up_proj.cw4_codebook.view(torch.float32)[0].item(),index)
            self.assertEqual(expert.down_proj.cw4_codebook.view(torch.float32)[0].item(),0)

    def test_adapter_resume_identity_rejects_mismatches(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.hunyuan_image_3 import HunyuanImage3InstructModel
        holder=object.__new__(HunyuanImage3InstructModel)
        holder.model_config=SimpleNamespace(model_kwargs={})
        holder.network=SimpleNamespace(lora_dim=16,alpha=16)
        holder.model=SimpleNamespace(aitk_load_source={'fingerprint':[123,456]},aitk_qtype='comfy_w4a8',named_modules=lambda: [])
        metadata=holder.get_adapter_metadata(holder.network)
        holder.validate_adapter_metadata({'hunyuan_training':metadata})
        for key,value in [('variant','base'),('preset','attention_shared_mlp'),('rank',32),('source',{}),('qtype','convrot8')]:
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,key):
                holder.validate_adapter_metadata({'hunyuan_training':dict(metadata,**{key:value})})

    def test_local_tokenizer_changes_conditioning_identity(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.hunyuan_image_3 import HunyuanImage3InstructModel
        holder=object.__new__(HunyuanImage3InstructModel)
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'tokenizer.json';file.write_text('{}')
            holder.model_config=SimpleNamespace(name_or_path='fixture',extras_name_or_path=directory,
                                                vae_path=None,model_kwargs={})
            first=holder.get_text_embedding_space_version()
            file.write_text('{"updated":true}')
            self.assertNotEqual(first,holder.get_text_embedding_space_version())

    def test_edit_preflight_rejects_missing_pairs_and_base_controls(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.src.config import validate_training
        train=SimpleNamespace(train_text_encoder=False,train_unet=True,batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); targets=root/'targets'; targets.mkdir(); (targets/'one.png').touch()
            refs=root/'sources';refs.mkdir()
            dataset=SimpleNamespace(folder_path=str(targets),control_path=[str(refs)])
            process=SimpleNamespace(train_config=train,network_config=SimpleNamespace(type='lora',network_kwargs={}),
                                    dataset_configs=[dataset],get_conf=lambda key:'sd_trainer')
            holder=SimpleNamespace(variant=INSTRUCT,model_config=SimpleNamespace(model_kwargs={'vision_path':'vision'},only_if_contains=None),validate_model_config=lambda:None)
            with self.assertRaisesRegex(ValueError,'expected one source'):
                validate_training(holder,process)
            (refs/'one.png').touch(); validate_training(holder,process)
            holder.variant=BASE
            with self.assertRaisesRegex(ValueError,'Base does not support edit'):
                validate_training(holder,process)

    def test_preflight_accepts_ordinary_ui_trainer_and_rejects_specialized_modes(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.src.config import validate_training
        train = SimpleNamespace(train_text_encoder=False, train_unet=True, batch_size=1)
        holder = SimpleNamespace(variant=INSTRUCT,
                                 model_config=SimpleNamespace(model_kwargs={}, only_if_contains=None),
                                 validate_model_config=Mock())
        for mode in ('sd_trainer', 'diffusion_trainer', 'flow_dpo',
                     'qwen_guidance_distillation', 'diffusion_kto'):
            with self.subTest(mode=mode):
                network = SimpleNamespace(type='lora', network_kwargs={})
                process = SimpleNamespace(train_config=train, network_config=network,
                                          dataset_configs=[], get_conf=lambda key, mode=mode: mode)
                if mode in ('sd_trainer', 'diffusion_trainer'):
                    validate_training(holder, process)
                    self.assertEqual(network.network_kwargs['only_if_contains'],
                                     ['self_attn.qkv_proj', 'self_attn.o_proj'])
                else:
                    with self.assertRaisesRegex(ValueError, 'ordinary'):
                        validate_training(holder, process)
                    self.assertEqual(network.network_kwargs, {})
        self.assertEqual(holder.validate_model_config.call_count, 2)

    def test_edit_preflight_ignores_ui_thumbnails_hidden_files_and_controls(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.src.config import validate_training
        train = SimpleNamespace(train_text_encoder=False, train_unet=True, batch_size=1)
        holder = SimpleNamespace(variant=INSTRUCT,
                                 model_config=SimpleNamespace(model_kwargs={'vision_path': 'vision'},
                                                              only_if_contains=None),
                                 validate_model_config=Mock())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); targets = root / 'targets'; refs = root / 'sources'
            targets.mkdir(); refs.mkdir()
            # UI datasets allow source/target extension differences and create
            # thumbnails by appending a second extension to the original name.
            for name in ('1.jpg', '2.png'):
                (targets / name).touch()
            for name in ('1.png', '2.png'):
                (refs / name).touch()
            for parent in (targets, refs):
                (parent / '.thumbs').mkdir()
            (targets / '.thumbs' / '1.jpg.jpg').touch()
            (refs / '.thumbs' / '1.png.png').touch()
            (targets / '.hidden.png').touch()
            (targets / '3.png.disabled').touch()
            (targets / '_latent_cache').mkdir()
            (targets / '_latent_cache' / 'cache.safetensors').touch()
            (targets / '_controls').mkdir()
            (targets / '_controls' / 'unpaired.png').touch()
            dataset = SimpleNamespace(folder_path=str(targets), control_path=[str(refs)])
            process = SimpleNamespace(train_config=train,
                                      network_config=SimpleNamespace(type='lora', network_kwargs={}),
                                      dataset_configs=[dataset], get_conf=lambda key: 'diffusion_trainer')
            validate_training(holder, process)
            holder.validate_model_config.assert_called_once()
            # Real images in ordinary subfolders still require a source pair.
            (targets / 'visible').mkdir()
            (targets / 'visible' / 'missing.png').touch()
            with self.assertRaisesRegex(ValueError, 'Edit pair missing.png'):
                validate_training(holder, process)

    def test_frozen_vae_tiling_when_caller_has_grad_enabled(self):
        from extensions_built_in.diffusion_models.hunyuan_image_3.hunyuan_image_3 import HunyuanImage3InstructModel

        class FrozenVAE(torch.nn.Module):
            scaling_factor = 0.5
            use_spatial_tiling = False

            def enable_spatial_tiling(self): self.use_spatial_tiling = True
            def disable_spatial_tiling(self): self.use_spatial_tiling = False

            def encode(self, image):
                self.assert_phase()
                return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: image.unsqueeze(2)))

            def decode(self, latent):
                self.assert_phase()
                return SimpleNamespace(sample=latent)

            def assert_phase(self):
                if torch.is_grad_enabled() or not self.use_spatial_tiling:
                    raise AssertionError('frozen VAE must use no-grad spatial tiling')

        holder = object.__new__(HunyuanImage3InstructModel)
        holder.vae = FrozenVAE()
        holder.model_config = SimpleNamespace(low_vram=True)
        holder._ensure_vae = Mock()
        image = torch.ones(3, 8, 12, requires_grad=True)
        with torch.enable_grad():
            encoded = holder.encode_images([image], device='cpu', dtype=torch.float32)
            self.assertFalse(encoded.requires_grad)
            self.assertFalse(holder.vae.use_spatial_tiling)
            decoded = holder.decode_latents(encoded.requires_grad_(), device='cpu')
            self.assertFalse(decoded.requires_grad)
            self.assertFalse(holder.vae.use_spatial_tiling)
            self.assertTrue(torch.is_grad_enabled())
        torch.testing.assert_close(decoded, image.unsqueeze(0))
        with patch.object(holder.vae, 'encode', side_effect=RuntimeError('encode failure')):
            with self.assertRaisesRegex(RuntimeError, 'encode failure'):
                holder.encode_images([image], device='cpu', dtype=torch.float32)
        self.assertFalse(holder.vae.use_spatial_tiling)

    def test_packed_module_cache_hit_bypasses_checkpoint_loader(self):
        from toolkit.config_modules import ModelConfig
        from extensions_built_in.diffusion_models.hunyuan_image_3.hunyuan_image_3 import HunyuanImage3InstructModel
        from toolkit.models.v2.text_encoders.hunyuan_image_3 import HunyuanImage3Conditioning
        from toolkit.models.v2.vae.hunyuan_image_3 import HunyuanImage3VAE
        model=HunyuanImage3Transformer(params_from_config(small_config()),dtype=torch.float32)
        name='model.layers.0.self_attn.qkv_proj'
        state={name+'.weight':torch.zeros(64,16,dtype=torch.int8),
               name+'.weight_s_rel':torch.ones(64,2,dtype=torch.float8_e4m3fn),
               name+'.weight_s_channel':torch.ones(64),
               name+'.weight_codebook':torch.arange(16,dtype=torch.float32),
               name+'.comfy_quant':comfy_quant_marker(dict(format='asym_w4a8_int8',group_size=16,convrot_groupsize=4))}
        import_comfy_quantized_layers(model,state,orig_dtype=torch.float32)
        model.aitk_is_quantized=True;model.aitk_qtype='comfy_w4a8'
        model.aitk_load_source={'path':'fixture','backend_version':1}
        conditioner=HunyuanImage3Conditioning(self.tokenizer)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'tokenizer.json').write_text('{}')
            source=root/'instruct.safetensors';save_file({'dummy':torch.ones(1)},str(source))
            config=ModelConfig(arch='hunyuan_image_3_instruct',name_or_path=str(source),
                               extras_name_or_path=str(root),vae_path=str(source),quantize=True,
                               qtype='comfy_w4a8',quantize_te=False,cache_quantized_models=True,
                               quantized_model_cache_dir=str(root/'cache'))
            first=HunyuanImage3InstructModel('cpu',config,dtype='float32')
            with patch.object(HunyuanImage3Transformer,'load_model',return_value=model) as load, \
                 patch.object(HunyuanImage3Conditioning,'create',return_value=conditioner), \
                 patch.object(HunyuanImage3VAE,'load_model',return_value=torch.nn.Identity()):
                first.load_model();load.assert_called_once()
                load.reset_mock();load.side_effect=AssertionError('original loader called on cache hit')
                second=HunyuanImage3InstructModel('cpu',config,dtype='float32');second.load_model()
                load.assert_not_called()
            restored=second.model.get_submodule(name)
            self.assertEqual(restored.cw4_packed.dtype,torch.uint8)
            torch.testing.assert_close(restored.cw4_packed,model.get_submodule(name).cw4_packed)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for asynchronous offload comparison')
    def test_w4a8_resident_and_offloaded_input_gradients(self):
        from toolkit.memory_management import MemoryManager
        import copy
        root=torch.nn.Sequential(torch.nn.Linear(256,16,bias=False))
        state={'0.weight':torch.randint(-128,127,(16,128),dtype=torch.int8),
               '0.weight_s_rel':torch.ones(16,16,dtype=torch.float8_e4m3fn),
               '0.weight_s_channel':torch.ones(16)*.02,
               '0.weight_codebook':torch.linspace(-1,1,16),
               '0.comfy_quant':comfy_quant_marker(dict(format='asym_w4a8_int8',group_size=16,convrot_groupsize=256))}
        import_comfy_quantized_layers(root,state,orig_dtype=torch.float32)
        resident=copy.deepcopy(root).cuda()
        MemoryManager.attach(root,torch.device('cuda'),offload_percent=1.)
        try:
            x=torch.randn(5,256,device='cuda',requires_grad=True)
            y=x.detach().clone().requires_grad_(True)
            a,b=resident(x),root(y)
            a.square().sum().backward();b.square().sum().backward()
            torch.cuda.synchronize()
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            self.assertTrue(root[0].cw4_packed.is_pinned())
            self.assertEqual(root[0].cw4_packed.device.type,'cpu')
        finally:
            MemoryManager.detach(root)

if __name__ == '__main__': unittest.main()
