"""Canonical 2D Hunyuan backbone with bounded mmap/sharded loading."""
import contextlib
import fnmatch
import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open

from .._mixin import OstrisModelMixin
from extensions_built_in.diffusion_models.hunyuan_image_3.src.transformer import HunyuanImage3, params_from_config
from extensions_built_in.diffusion_models.hunyuan_image_3.src.config import INSTRUCT, pinned_config, validate_variant

_BANK = re.compile(r'^(model\.layers\.\d+\.mlp)\.(experts_gate_up_proj|experts_down_proj)$')
SKIPPED_PREFIXES = ('vae.', 'vision_model.', 'lm_head.', 'model.ln_f.')


class CheckpointReader:
    """Open each shard once, and slice expert banks before materializing tensors."""
    def __init__(self, path):
        self.shared = False
        self.stack = contextlib.ExitStack()
        self.handles = {}
        if os.path.isdir(path):
            index = Path(path) / 'model.safetensors.index.json'
            data = json.loads(index.read_text(), object_pairs_hook=self._unique)
            self.mapping = {k: str(Path(path) / v) for k, v in data['weight_map'].items()}
        else:
            f = self.stack.enter_context(safe_open(path, framework='pt', device='cpu'))
            self.handles[path] = f
            self.mapping = {k: path for k in f.keys()}
        self.keys = set(self.mapping)
        self.consumed = set()

    @staticmethod
    def _unique(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f'Duplicate checkpoint/index key {key}')
            out[key] = value
        return out

    def handle(self, key):
        path = self.mapping[key]
        if path not in self.handles:
            self.handles[path] = self.stack.enter_context(safe_open(path, framework='pt', device='cpu'))
        return self.handles[path]

    def shape(self, key):
        return self.handle(key).get_slice(key).get_shape()

    def tensor(self, key, expert=None):
        self.consumed.add(key)
        handle = self.handle(key)
        return handle.get_tensor(key) if expert is None else handle.get_slice(key)[expert]

    def metadata(self):
        return next(iter(self.handles.values())).metadata() if self.handles else {}

    def close(self):
        self.stack.close()


def canonical_source(name, reader, experts):
    """Return (checkpoint prefix, bank index), without expanding a bank."""
    match = re.match(r'^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_and_up_proj|down_proj)$', name)
    if match:
        root, index, proj = match.groups()
        bank = root + ('.experts_gate_up_proj' if proj == 'gate_and_up_proj' else '.experts_down_proj')
        if bank + '.weight' in reader.keys:
            if name + '.weight' in reader.keys:
                raise ValueError(f'Checkpoint contains both bank and canonical expert {name}')
            if reader.shape(bank + '.weight')[0] != experts:
                raise ValueError(f'{bank}: wrong expert count')
            return bank, int(index)
    return name, None


class HunyuanImage3Transformer(HunyuanImage3, OstrisModelMixin):
    aitk_subfolder = None
    aitk_config_repo = INSTRUCT.repository
    aitk_cast_on_load = False
    aitk_cast_quantized_load = False

    @classmethod
    def load(cls, name_or_path, qtype=None, offload=0., dtype=torch.bfloat16,
             device=None, base_model=None, quantize_device=None,
             exclude_quant_modules=None, **kwargs):
        if qtype and '|' in qtype:
            raise ValueError('Hunyuan accuracy-recovery requantization is unsupported; use factorized LoRA')
        # The custom streamed loader quantizes each projection as it is read.
        # Forward the selected conversion device rather than silently doing
        # CPU conversion through the generic deferred-quantization entry point.
        model = cls.load_model(name_or_path, dtype=dtype,
                               qtype=(qtype or '').split('|')[0] or None,
                               quantize_device=quantize_device or device, **kwargs)
        return model.aitk_post_load(qtype=qtype, offload=offload, dtype=dtype,
                                   device=device, base_model=base_model,
                                   quantize_device=quantize_device,
                                   exclude_quant_modules=exclude_quant_modules)

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        file = Path(path) / 'config.json'
        if file.is_file():
            return json.loads(file.read_text())
        from huggingface_hub import hf_hub_download
        return json.loads(Path(hf_hub_download(path, 'config.json')).read_text())

    @classmethod
    def aitk_from_config(cls, config):
        with torch.device('meta'):
            return cls(params_from_config(config))

    @classmethod
    def get_transformer_block_names(cls):
        return ['model.layers']

    @classmethod
    def get_quantization_exclude_modules(cls):
        # Published repacks quantize only attention, shared MLP, and experts.
        return ['*gate.wg', 'model.wte', 'vision_aligner*', 'patch_embed*',
                'final_layer*', 'time_embed*', 'timestep_emb*']

    @classmethod
    def load_model(cls, name_or_path, dtype=torch.bfloat16, qtype=None, config=None,
                   config_path=None, quantize_device=None, variant=INSTRUCT,
                   quantize_on_load=True, device=None, tensor_source=None, shared_store=None, **kwargs):
        validate_variant(variant, name_or_path)
        if config is None:
            config = cls.aitk_load_config(config_path or name_or_path) if os.path.isdir(config_path or name_or_path) else pinned_config(variant)
        if config.get('cfg_distilled') or config.get('use_meanflow'):
            raise ValueError('Distil/MeanFlow config is unsupported')
        config = dict(config, model_type=variant.name, sequence_template=variant.sequence_template)
        from toolkit.models.v2.pool import ComponentPool
        pool = ComponentPool.current
        pool_key = None
        if pool is not None:
            source_stat = os.stat(name_or_path) if os.path.isfile(name_or_path) else None
            pool_key = ComponentPool.make_key(
                cls, name_or_path, dtype=dtype, qtype=qtype, variant=variant.name,
                config=config, revision=kwargs.get('revision'),
                fingerprint=None if source_stat is None else [source_stat.st_size, source_stat.st_mtime_ns],
                mapping_version=1, backend_version=1,
                shared_store=None if shared_store is None else shared_store['identity'])
            resident = pool.get(pool_key)
            if resident is not None:
                return resident
        path = name_or_path
        if tensor_source is None and not os.path.exists(path):
            if str(path).endswith('.safetensors'):
                path = cls._resolve_single_file(path)
            else:
                from huggingface_hub import snapshot_download
                path = snapshot_download(path, revision=kwargs.get('revision'),
                                         allow_patterns=['config.json', '*.safetensors', '*.safetensors.index.json'])
        requested = (qtype or '').split('|')[0]
        reader = tensor_source if tensor_source is not None else CheckpointReader(path)
        if getattr(reader, 'shared', False) and (variant != INSTRUCT or requested != 'convrot8'):
            raise ValueError('Shared Hunyuan attachment supports Instruct convrot8 only')
        try:
            validate_variant(variant, name_or_path, reader.metadata(), reader.keys)
            model = cls.aitk_from_config(config)
            model.dtype = dtype
            from toolkit.util.comfy_quant_import import import_comfy_quantized_layers, parse_comfy_quant_blob
            from toolkit.util.ostris_quant import convert_linear_to_ostris, get_ostris_quantizer
            exclusions = cls.get_quantization_exclude_modules()
            shipped = set()
            for name, module in list(model.named_modules()):
                if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    continue
                prefix, expert = canonical_source(name, reader, model.config.num_experts)
                weight_key = prefix + '.weight'
                if weight_key not in reader.keys:
                    raise ValueError(f'Missing checkpoint weight {weight_key}')
                marker_key = prefix + '.comfy_quant'
                if marker_key in reader.keys:
                    marker = reader.tensor(marker_key)
                    conf = parse_comfy_quant_blob(marker)
                    if conf.get('num_experts', model.config.num_experts) != model.config.num_experts:
                        raise ValueError(f'{prefix}: marker expert count mismatch')
                    fmt = conf['format']
                    backend = {'asym_w4a8_int8':'comfy_w4a8', 'int8_tensorwise':'convrot8'}.get(fmt)
                    if backend is None:
                        raise ValueError(f'Hunyuan checkpoint format {fmt!r} unsupported')
                    shipped.add(backend)
                    if requested and requested != backend:
                        raise ValueError(f'Checkpoint ships {backend}; select qtype: {backend}; whole-model conversion is forbidden')
                    local = {name + '.comfy_quant': marker}
                    suffixes = ['weight','weight_scale','weight_s_rel','weight_s_channel','weight_codebook','weight_correction','bias','input_scale']
                    for suffix in suffixes:
                        key = prefix + '.' + suffix
                        if key not in reader.keys:
                            continue
                        shape = reader.shape(key)
                        # Codebooks may be shared [16] or per expert [E,16].
                        slice_index = expert
                        if expert is not None and suffix == 'weight_codebook' and shape == [16]:
                            slice_index = None
                        local[name + '.' + suffix] = reader.tensor(key, slice_index)
                    remaining, _ = import_comfy_quantized_layers(
                        model, local, orig_dtype=dtype, strict_shared=getattr(reader, 'shared', False))
                    if remaining:
                        raise ValueError(f'Unconsumed quantization tensors: {list(remaining)}')
                    continue
                value = reader.tensor(weight_key, expert)
                if tuple(value.shape) != tuple(module.weight.shape):
                    raise ValueError(f'{name}: shape {tuple(value.shape)} != {tuple(module.weight.shape)}')
                module.weight = torch.nn.Parameter(value, requires_grad=False)
                bias_key = prefix + '.bias'
                if getattr(module, 'bias', None) is not None:
                    if bias_key not in reader.keys:
                        raise ValueError(f'Missing bias {bias_key}')
                    module.bias = torch.nn.Parameter(reader.tensor(bias_key, expert), requires_grad=False)
                if requested and isinstance(module, torch.nn.Linear) and not any(fnmatch.fnmatchcase(name, p) for p in exclusions):
                    quantizer = get_ostris_quantizer(requested)
                    if quantizer is None:
                        raise ValueError('Hunyuan fresh quantization supports convrot8 only')
                    if quantizer.can_quantize(module):
                        # Bound conversion to one projection, never an 80B block/model.
                        original = module.weight.device
                        if quantize_device is not None:
                            module.to(quantize_device)
                        convert_linear_to_ostris(module, quantizer)
                        module.to(original)
            for name, parameter in list(model.named_parameters()):
                if not parameter.is_meta:
                    continue
                if name not in reader.keys:
                    raise ValueError(f'Missing non-linear parameter {name}')
                value = reader.tensor(name)
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f'Invalid shape for {name}')
                parent, _, leaf = name.rpartition('.')
                model.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
            unknown = reader.keys - reader.consumed
            unknown = {k for k in unknown if not k.startswith(SKIPPED_PREFIXES) and not k.endswith('.comfy_attention.config')}
            if unknown:
                raise ValueError(f'Unconsumed Hunyuan checkpoint tensors: {sorted(unknown)[:20]}')
            model.aitk_is_quantized = bool(shipped or requested)
            model.aitk_qtype = requested or next(iter(shipped), None)
            stat = os.stat(path) if os.path.isfile(path) else None
            model.aitk_load_source = {'path': os.path.realpath(path), 'variant': variant.name,
                                      'fingerprint': None if stat is None else [stat.st_size, stat.st_mtime_ns],
                                      'mapping_version': 1, 'backend_version': 1}
            model.requires_grad_(False)
            if getattr(reader, 'shared', False):
                from toolkit.shared_models import capture_shared_masters
                model._shared_store = shared_store
                model._shared_arena = reader.arena
                capture_shared_masters(model)
            if device is not None:
                model.to(device)
            if pool is not None:
                pool.put(pool_key, model)
            return model
        finally:
            reader.close()

    def aitk_post_load(self, qtype=None, **kwargs):
        if self.aitk_is_quantized:
            qtype = qtype or self.aitk_qtype
        return super().aitk_post_load(qtype=qtype, **kwargs)
