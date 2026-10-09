"""CPU Iris integration tests. GPU/large-checkpoint tests run separately."""
import json
import weakref
import gc
import ast
import logging
import io
from typing import Optional
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from dataclasses import asdict, replace

import torch
from safetensors.torch import load_file, save_file
from transformers import Qwen3VLTextConfig

from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.lora_special import LoRASpecialNetwork
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.sample_progress import configure_sample_progress
from extensions_built_in.diffusion_models.iris import IrisModel
from extensions_built_in.diffusion_models.iris.transformer import IrisTransformer, architecture_config
from extensions_built_in.diffusion_models.iris.text_encoder import IrisTextEncoder, encode_iris_prompts, PREFIX, SUFFIX
from extensions_built_in.diffusion_models.iris.pipeline import IrisPipeline
from extensions_built_in.diffusion_models.iris.src.flow.solver import FlowDPMSolver


def tiny_config():
    return architecture_config(dict(hidden_size=32, depth=2, dual_depth=1, num_heads=4, num_kv_heads=2,
        text_dim=32, text_len=6, text_lap_num_layers=12, text_lap_num_heads=4, patch_size=4,
        pixel=dict(depth=1, hidden_size=4, attn_hidden_size=32, num_heads=4)))


def backbone():
    model = IrisTransformer(tiny_config())
    # Released weights have a nonzero output head and shared modulation.
    torch.nn.init.normal_(model.final_layer.linear.weight, std=.1)
    for core in model.modulation_cores.values():
        torch.nn.init.normal_(core.weight, std=.01)
    return model


def holder(model=None, **config):
    result = IrisModel('cpu', ModelConfig(name_or_path='unused', arch='iris', **config), dtype='fp32')
    result.model = model or backbone()
    result.pipeline = SimpleNamespace(transformer=result.model, text_encoder=None)
    return result


def attach_lora(sd):
    config = NetworkConfig(type='lora', linear=2, linear_alpha=2)
    network = LoRASpecialNetwork(text_encoder=None, unet=sd.model, lora_dim=2, alpha=2,
        train_unet=True, train_text_encoder=False, network_config=config, is_transformer=True,
        target_lin_modules=sd.target_lora_modules, peft_format=True, base_model=sd)
    network.apply_to(None, sd.model, apply_text_encoder=False, apply_unet=True)
    network.force_to('cpu', torch.float32)
    network.is_active = True
    network._update_torch_multiplier()
    return network


class IrisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def conditioning(self):
        return PromptEmbeds(torch.randn(1, 6, 12 * 32), attention_mask=torch.tensor([[1, 1, 1, 0, 0, 0]]))

    def test_native_single_file_and_comfy_prefix_roundtrip(self):
        model = backbone().eval()
        with tempfile.TemporaryDirectory() as d:
            for prefix in ('', 'diffusion_model.', 'model.diffusion_model.'):
                file = str(Path(d, 'model.safetensors'))
                save_file({prefix+k: v for k,v in model.state_dict().items()}, file)
                loaded = IrisTransformer.load(file, config=tiny_config(), dtype=torch.float32, device='cpu')
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
                self.assertIs(loaded.blocks[0].adaln_img.core, loaded.modulation_cores['adaln_img'])

    def test_lora_backward_save_reload(self):
        sd = holder()
        original = {key: v.clone() for key,v in sd.model.state_dict().items()}
        sd.model.requires_grad_(False).train()
        sd.model.enable_gradient_checkpointing()
        network = attach_lora(sd)
        optimizer = torch.optim.AdamW(network.parameters(), lr=.01)
        x, time, embeds = torch.randn(1,3,8,8), torch.tensor([500.]), self.conditioning()
        pred = sd.get_noise_prediction(x, time, embeds)
        pred.square().mean().backward()
        grads = [p.grad for p in network.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))
        self.assertGreater(sum(float(grad.abs().sum()) for grad in grads), 0)
        self.assertTrue(all(p.grad is None for p in sd.model.parameters()))
        optimizer.step()
        expected = sd.get_noise_prediction(x, time, embeds).detach()
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d, 'iris_lora.safetensors'))
            network.save_weights(path, dtype=torch.float32)
            state = load_file(path)
            self.assertTrue(any(k.startswith('diffusion_model.blocks.0.') and k.endswith('lora_A.weight') for k in state))
            self.assertTrue(any(k.startswith('diffusion_model.pixel_blocks.') for k in state))
            copy = holder()
            copy.model.load_state_dict(original)
            other = attach_lora(copy)
            other.load_weights(path)
            actual = copy.get_noise_prediction(x, time, embeds).detach()
            torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)

    def test_checkpoint_policies_preserve_lora_gradients(self):
        sd = holder()
        sd.model.requires_grad_(False).train()
        network = attach_lora(sd)
        x, time, embeds = torch.randn(1,3,8,8), torch.tensor([300.]), self.conditioning()
        baseline = None
        for policy in ('none', 'full', 'selective_op', 'selective_layer'):
            network.zero_grad(set_to_none=True)
            sd.model.activation_checkpointing = policy
            pred = sd.get_noise_prediction(x, time, embeds)
            pred.square().mean().backward()
            grads = {n:p.grad.clone() for n,p in network.named_parameters() if p.grad is not None}
            if baseline is None:
                baseline = grads
            else:
                self.assertEqual(baseline.keys(), grads.keys())
                for key in grads:
                    torch.testing.assert_close(grads[key], baseline[key], rtol=1e-5, atol=1e-7)

    def test_comfy_int8_checkpoint_backward(self):
        model = backbone()
        state = dict(model.state_dict())
        name = 'blocks.0.attn.q_proj_x'
        weight = state[name+'.weight']
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 127
        state[name+'.weight'] = (weight / scale).round().clamp(-127,127).to(torch.int8)
        state[name+'.weight_scale'] = scale
        state[name+'.comfy_quant'] = torch.tensor(list(json.dumps({'format':'int8_tensorwise'}).encode()), dtype=torch.uint8)
        loaded = IrisTransformer.load_from_state_dict(state, torch.float32, config=tiny_config())
        self.assertTrue(loaded.aitk_is_quantized)
        sd = holder(loaded)
        sd.model.requires_grad_(False).train()
        network = attach_lora(sd)
        sd.get_noise_prediction(torch.randn(1,3,8,8), torch.tensor([400.]), self.conditioning()).square().mean().backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters()))

    def test_pixel_velocity_target_and_schedule(self):
        sd = holder()
        x, noise = torch.randn(2,3,8,8), torch.randn(2,3,8,8)
        times = torch.tensor([0.,1000.])
        scheduler = sd.get_train_scheduler()
        self.assertEqual(scheduler.config.shift, 4)
        noisy = scheduler.add_noise(x, noise, times[:,None,None,None])
        torch.testing.assert_close(noisy[0],x[0])
        torch.testing.assert_close(noisy[1],noise[1])
        target = sd.get_loss_target(noise=noise,batch=SimpleNamespace(latents=x))
        torch.testing.assert_close(target,noise-x)
        torch.manual_seed(4)
        expected = torch.sort(1-torch.sigmoid(torch.randn(1000)),descending=True).values
        expected = 4*expected/(1+3*expected)*1000
        torch.manual_seed(4)
        actual = scheduler.set_train_timesteps(1000,'cpu','sigmoid')
        torch.testing.assert_close(actual,expected)

    def test_text_tower_hf_and_comfy_loading(self):
        config = Qwen3VLTextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
            num_hidden_layers=36,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        te = IrisTextEncoder(config)
        with tempfile.TemporaryDirectory() as d:
            Path(d,'config.json').write_text(json.dumps({'model_type':'qwen3_vl','text_config':config.to_dict()}))
            state = {'model.language_model.'+k:v for k,v in te.state_dict().items()}
            state['model.visual.unused.weight'] = torch.ones(2)
            save_file(state,str(Path(d,'model.safetensors')))
            directory = IrisTextEncoder.load(d,dtype=torch.float32,device='cpu')
            single = IrisTextEncoder.load(str(Path(d,'model.safetensors')),config=config,dtype=torch.float32,device='cpu')
            for key,value in te.state_dict().items():
                torch.testing.assert_close(directory.state_dict()[key],value,rtol=0,atol=0)
                torch.testing.assert_close(single.state_dict()[key],value,rtol=0,atol=0)
            self.assertFalse(hasattr(directory,'visual'))

    def test_prompt_template_suffix_mask_and_cache(self):
        class Tokenizer:
            pad_token_id=0
            def encode(self,text,**kwargs):
                if text==PREFIX:return [1,2,3]
                if text==SUFFIX:return [4,5]
                return [6]*len(text)
        class Encoder:
            device='cpu'
            def __call__(self,input_ids,attention_mask,**kwargs):
                self.ids,self.mask=input_ids,attention_mask
                states=tuple(torch.full((*input_ids.shape,32),float(i)) for i in range(37))
                return SimpleNamespace(hidden_states=states)
        encoder=Encoder()
        with self.assertWarns(UserWarning):
            embeds=encode_iris_prompts(encoder,Tokenizer(),['abc','x'*400],torch.float32)
        self.assertEqual(embeds.text_embeds.shape,(2,300,384))
        self.assertEqual(encoder.ids[0,:8].tolist(),[1,2,3,6,6,6,4,5])
        self.assertEqual(encoder.ids[1,-2:].tolist(),[4,5])
        self.assertEqual(embeds.attention_mask[0].sum().item(),5)
        self.assertEqual(embeds.text_embeds[0,6].abs().sum().item(),0)
        self.assertEqual(embeds.text_embeds[0,0].reshape(12,32)[:,0].tolist(),[2,5,8,11,14,17,20,23,26,29,32,35])
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d,'embeds.safetensors'))
            embeds.save(path)
            restored=PromptEmbeds.load(path)
            torch.testing.assert_close(restored.text_embeds,embeds.text_embeds)
            batched=concat_prompt_embeds([restored,embeds])
            self.assertEqual(batched.text_embeds.shape,(4,300,384))
            self.assertEqual(restored.attention_mask.shape,(2,300))

    def test_sampler_matches_native_solver(self):
        sd = holder()
        sd.model.eval()
        embeds=self.conditioning()
        noise=torch.randn(1,3,8,8)
        reference=FlowDPMSolver(lambda x,t,y:sd.get_noise_prediction(x,t,PromptEmbeds(y,attention_mask=embeds.attention_mask))).sample(
            noise,embeds.text_embeds,steps=4,shift=4)
        expected=((reference.clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(0,2,3,1).numpy()[0]
        pipeline=IrisPipeline(sd)
        pipeline.set_progress_bar_config(disable=True)
        actual=pipeline(embeds,None,8,8,4,1,torch.Generator().manual_seed(42),latents=noise)[0]
        torch.testing.assert_close(torch.tensor(__import__('numpy').array(actual)),torch.tensor(expected),rtol=0,atol=0)

    def test_sampler_progress_per_image_preserves_step_hooks(self):
        stream = io.StringIO()
        steps = []
        sd = SimpleNamespace(
            device_torch=torch.device("cpu"),
            get_noise_prediction=lambda x, t, embeds: torch.zeros_like(x),
            sample_step_hook=lambda index, total, x: steps.append((index, total)),
        )
        pipeline = IrisPipeline(sd)
        pipeline.set_progress_bar_config(file=stream, disable=True, mininterval=0)
        for index in range(2):
            configure_sample_progress(pipeline, index, 2)
            images = pipeline(None, None, 8, 8, 20, 1, torch.Generator(),
                              latents=torch.zeros(1, 3, 8, 8))
            self.assertEqual(images[0].size, (8, 8))
        self.assertEqual(steps, list(zip(range(20), [20] * 20)) * 2)
        lines = [line for line in stream.getvalue().split("\n") if line]
        self.assertEqual(len(lines), 2)
        for index, line in enumerate(lines):
            self.assertIn(f"Sample {index + 1}/2", line)
            self.assertIn("20/20", line)

    def test_model_cache_pickle_preserves_shared_cores(self):
        sd = holder()
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d,'model.pt'))
            sd.save_quantized_module_cache(sd.model,path,'transformer')
            restored=sd.load_quantized_module_cache(path,'transformer')
            self.assertIsNotNone(restored)
            self.assertIs(restored.blocks[0].adaln_img.core,restored.modulation_cores['adaln_img'])

    def test_cached_text_tower_retains_layer_capture(self):
        config=Qwen3VLTextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        sd=holder()
        with tempfile.TemporaryDirectory() as d:
            path=str(Path(d,'text.pt'))
            sd.save_quantized_module_cache(IrisTextEncoder(config),path,'text_encoder')
            encoder=sd.load_quantized_module_cache(path,'text_encoder')
            output=encoder(input_ids=torch.tensor([[1,2,3]]),output_hidden_states=True,use_cache=False)
            self.assertEqual(len(output.hidden_states),3)

    def test_patch_size_is_enforced(self):
        sd=holder()
        with self.assertRaisesRegex(ValueError,'not divisible'):
            sd.get_noise_prediction(torch.randn(1,3,7,8),torch.tensor([500.]),self.conditioning())

    def test_full_load_prompt_encode_and_lora_update(self):
        class Tokenizer:
            pad_token_id=0
            def encode(self,text,**kwargs):
                if text==PREFIX:return [1,2,3]
                if text==SUFFIX:return [4,5]
                return [6]*len(text)
        config=Qwen3VLTextConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=36,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8)
        with tempfile.TemporaryDirectory() as d:
            file=str(Path(d,'iris.safetensors'))
            save_file(backbone().state_dict(),file)
            text_dir=Path(d,'encoder');text_dir.mkdir()
            Path(text_dir,'config.json').write_text(json.dumps({'model_type':'qwen3_vl','text_config':config.to_dict()}))
            text=IrisTextEncoder(config)
            save_file({'model.language_model.'+k:v for k,v in text.state_dict().items()},str(text_dir/'model.safetensors'))
            mc=ModelConfig(name_or_path=file,arch='iris',quantize=False,quantize_te=False,low_vram=True,
                model_kwargs={'transformer_config':asdict(tiny_config()),'text_encoder_path':str(text_dir)})
            sd=IrisModel('cpu',mc,dtype='fp32')
            with patch('extensions_built_in.diffusion_models.iris.iris.AutoTokenizer.from_pretrained',return_value=Tokenizer()):
                sd.load_model()
            encoded=sd.get_prompt_embeds('fox')
            self.assertEqual(encoded.text_embeds.shape,(1,300,384))
            self.assertTrue(torch.isfinite(encoded.text_embeds).all())
            self.assertEqual(sd.vae.config.latent_channels,3)
            network=attach_lora(sd)
            sd.model.enable_gradient_checkpointing();sd.model.train()
            pred=sd.get_noise_prediction(torch.randn(1,3,16,16),torch.tensor([600.]),encoded)
            pred.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in network.parameters()))

    def test_pipeline_does_not_retain_unloaded_encoder(self):
        sd=holder()
        sd.text_encoder=torch.nn.Linear(4,4)
        ref=weakref.ref(sd.text_encoder)
        sd.pipeline=IrisPipeline(sd)
        sd.text_encoder=None
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(sd.pipeline.text_encoder)

    def test_comfy_workflow_without_vae_and_with_inference_lora(self):
        from toolkit.comfy_sample import get_workflow_for_sample
        from testing.test_cross_model_comfy_templates import CrossModelComfyTemplateTests
        request=CrossModelComfyTemplateTests().request('iris',cfg=3)
        for inference in ('','additional.safetensors'):
            workflow=get_workflow_for_sample('config/comfy_templates/iris_lora_sample.json.njk',replace(request,inference_lora=inference))
            self.assertEqual(workflow['1']['class_type'],'IrisModelLoader')
            self.assertEqual(workflow['2']['class_type'],'IrisCLIPLoader')
            self.assertEqual(workflow['7']['class_type'],'IrisKSampler')
            self.assertEqual(workflow['7']['inputs']['scheduler'],'iris')
            self.assertEqual(workflow['3']['inputs']['lora_strength'],-.5)
            self.assertEqual(workflow['7']['inputs']['model'],['4' if inference else '3',0])
            self.assertFalse(any('VAE' in node['class_type'] for node in workflow.values()))

    def test_installed_comfy_lora_parser_accepts_every_layer(self):
        # Execute the installed parser bodies unchanged. Importing the full
        # Comfy package initializes its Triton kernels even in CPU mode.
        root=Path('/home/bart/ComfyUI/comfy')
        if not (root/'weight_adapter/lora.py').is_file():
            self.skipTest('Installed Comfy LoRA parser unavailable')
        class ModelTypes:
            def __getattr__(self,name):
                return type(name,(),{})
        native=ast.parse((root/'lora.py').read_text())
        adapter=ast.parse((root/'weight_adapter/lora.py').read_text())
        functions=[node for node in native.body if isinstance(node,ast.FunctionDef) and node.name in ('load_lora','model_lora_keys_unet')]
        cls=next(node for node in adapter.body if isinstance(node,ast.ClassDef) and node.name=='LoRAAdapter')
        namespace={'torch':torch,'Optional':Optional,'logging':logging,'WeightAdapterBase':object,
            'comfy':SimpleNamespace(utils=SimpleNamespace(unet_to_diffusers=lambda _:{}),model_base=ModelTypes())}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(root/'weight_adapter/lora.py'),'exec'),namespace)
        namespace['weight_adapter']=SimpleNamespace(adapters=[namespace['LoRAAdapter']])
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(root/'lora.py'),'exec'),namespace)
        sd=holder()
        network=attach_lora(sd)
        state=network.get_state_dict(dtype=torch.float32)
        wrapper=torch.nn.Module()
        wrapper.diffusion_model=sd.model
        wrapper.model_config=SimpleNamespace(unet_config={})
        keys=namespace['model_lora_keys_unet'](wrapper,{})
        patches=namespace['load_lora'](state,keys)
        expected={key.removesuffix('.lora_A.weight')+'.weight' for key in state if key.endswith('.lora_A.weight')}
        self.assertEqual(set(patches),expected)
        self.assertTrue(patches)

    def test_strict_inference_lora_loads_block_and_input_layers(self):
        from toolkit.assistant_lora import load_assistant_lora_from_path
        for layer in ('blocks.0.attn.q_proj_x', 's_embedder.proj'):
            with self.subTest(layer=layer), tempfile.TemporaryDirectory() as directory:
                sd=holder()
                module=sd.model.get_submodule(layer)
                state={f'diffusion_model.{layer}.lora_A.weight':torch.randn(2,module.in_features),
                    f'diffusion_model.{layer}.lora_B.weight':torch.randn(module.out_features,2),
                    f'diffusion_model.{layer}.alpha':torch.tensor(2.)}
                path=str(Path(directory,'iris-inference.safetensors'))
                save_file(state,path)
                network=load_assistant_lora_from_path(path,sd,strict=True)
                self.assertEqual(len(network.get_all_modules()),1)
                self.assertTrue(network.is_active)
                self.assertTrue(all(not parameter.requires_grad for parameter in network.parameters()))
                x=torch.randn(1,module.in_features)
                expected=torch.nn.functional.linear(x,module.weight,module.bias)
                expected=expected+torch.nn.functional.linear(torch.nn.functional.linear(x,state[f'diffusion_model.{layer}.lora_A.weight']),state[f'diffusion_model.{layer}.lora_B.weight'])
                torch.testing.assert_close(module(x),expected,rtol=1e-5,atol=1e-5)

    def test_foreign_inference_lora_names_the_path_and_arch(self):
        from toolkit.assistant_lora import load_assistant_lora_from_path
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory,'krea2-adapter.safetensors'))
            save_file({'transformer.blocks.0.attn.wq.lora_A.weight':torch.randn(2,32),
                       'transformer.blocks.0.attn.wq.lora_B.weight':torch.randn(32,2)},path)
            sd=holder()
            original=sd.model.blocks[0].attn.q_proj_x.forward
            with self.assertRaisesRegex(ValueError,"krea2-adapter.*no matching LoRA layers.*iris"):
                load_assistant_lora_from_path(path,sd,strict=True)
            self.assertEqual(sd.model.blocks[0].attn.q_proj_x.forward,original)


if __name__=='__main__':
    unittest.main()
