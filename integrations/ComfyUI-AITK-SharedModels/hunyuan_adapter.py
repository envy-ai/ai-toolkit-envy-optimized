"""Build Pedro model definitions on meta, then bind descriptor views directly."""
import torch

from aitk_shared_models.ownership import capture_masters, storage_leaves, verify_masters
from aitk_shared_models.protocol import SharedModelError, digest_json, require_compatible
from aitk_shared_models.tensors import owner_of

from .compatibility import check_source, pedro_modules
from .shared_ops import shared_operations, shared_quantized_tensor
from .shared_patcher import SharedModelPatcher


def build_hunyuan(arena, client, identity):
    import comfy.ops
    import comfy.model_management as management
    import comfy.model_patcher
    from comfy.quant_ops import QUANT_ALGOS, QuantizedTensor, get_layout_class
    from tokenizers import Tokenizer

    check_source(comfy.ops, 'comfy.ops')
    check_source(comfy.model_patcher, 'comfy.model_patcher')
    loader, definitions, model_base = pedro_modules()
    manifest = arena.manifest
    require_compatible(manifest, {'variant': 'instruct', 'quantization': 'int8_tensorwise'})
    config = dict(manifest['config'])
    if config.get('cfg_distilled') or config.get('use_meanflow') or config.get('model_type') != 'instruct':
        raise SharedModelError('Shared model sampling contract must be non-distilled Instruct')
    if digest_json(config) != manifest['config_digest']:
        raise SharedModelError('Model config digest changed')
    # No Comfy checkpoint loader, detect_layer_quantization or CPU normalization.
    markers = manifest['quantization_markers']
    quant_config = dict(markers)
    dtype = management.unet_dtype(supported_dtypes=[torch.bfloat16, torch.float16, torch.float32])
    operations = shared_operations(dtype, quant_config)
    import importlib
    latent = importlib.import_module(loader.__package__.rsplit('.', 1)[0] + '.latent_formats').HunyuanImage3
    model_config = model_base.HunyuanImage3ModelConfig(
        definitions.params_from_config(config), latent(), {'shift': loader.SHIFT},
        quant_config=quant_config, dtype=dtype, custom_operations=operations)
    with torch.device('meta'):
        model = model_base.HunyuanImage3Model(model_config, device=torch.device('meta'))
    consumed = set()
    for name, module in model.diffusion_model.named_modules():
        conf = markers.get(name)
        if conf is None:
            if isinstance(module, (operations.Linear, operations.MoEExperts)):
                key = name + '.weight'
                if key not in manifest['tensors']:
                    raise SharedModelError(f'Missing dense shared projection {key}')
                value = arena.tensor(key)
                if tuple(value.shape) != tuple(module._orig_shape) or not value.dtype.is_floating_point:
                    raise SharedModelError(f'{name}: dense projection shape/dtype mismatch')
                module.weight = torch.nn.Parameter(value, requires_grad=False)
                module.weight_function, module.bias_function = [], []
                module.factory_kwargs['device'] = torch.device('cpu')
                consumed.add(key)
            continue
        if not isinstance(module, (operations.Linear, operations.MoEExperts)):
            raise SharedModelError(f'{name}: quantized shared embeddings/other ops are unsupported')
        if conf.get('format') != 'int8_tensorwise':
            raise SharedModelError('Only int8 is qualified for descriptor attachment')
        qdata = arena.tensor(name + '.weight')
        scale = arena.tensor(name + '.weight_scale')
        if qdata.dtype != torch.int8 or scale.dtype != torch.float32 or not qdata.is_contiguous() or not scale.is_contiguous():
            raise SharedModelError(f'{name}: int8/FP32 contiguous bank layout required')
        if (tuple(qdata.shape) != tuple(module._orig_shape)
                or scale.numel() != qdata.numel() // qdata.shape[-1]):
            raise SharedModelError(f'{name}: checkpoint bank/projection shape differs from model config')
        algo = QUANT_ALGOS['int8_tensorwise']
        layout = get_layout_class(algo['comfy_tensor_layout'])
        if layout is None:
            raise SharedModelError('Comfy Kitchen INT8 layout unavailable')
        options = {'scale': scale, 'orig_dtype': dtype, 'orig_shape': tuple(qdata.shape)}
        if conf.get('convrot'):
            options.update(convrot=True, convrot_groupsize=int(conf.get('convrot_groupsize', 256)))
        params = layout.Params(**options)
        module.weight = torch.nn.Parameter(shared_quantized_tensor(qdata, algo['comfy_tensor_layout'], params), requires_grad=False)
        module.quant_format = 'int8_tensorwise'
        module.layout_type = algo['comfy_tensor_layout']
        module._full_precision_mm_config = bool(conf.get('full_precision_matrix_mult', False))
        module._full_precision_mm = module._full_precision_mm_config
        module.weight_function, module.bias_function = [], []
        module.factory_kwargs['device'] = torch.device('cpu')
        # Validate all real storage, including layout scale tensors, after wrapping.
        leaves = list(storage_leaves(module.weight))
        if not leaves or any(owner_of(value) is None for value in leaves):
            raise SharedModelError(f'{name}: quantized wrapper copied shared storage')
        consumed.update((name + '.weight', name + '.weight_scale', name + '.comfy_quant'))
        # Calibration is tiny and optional; attach it as a view rather than silently discard.
        calibration = name + '.input_scale'
        if calibration in manifest['tensors']:
            module.register_buffer('input_scale', arena.tensor(calibration), persistent=False)
            consumed.add(calibration)
    for name, parameter in list(model.diffusion_model.named_parameters()):
        if not parameter.is_meta:
            continue
        if name not in manifest['tensors']:
            raise SharedModelError(f'Missing frozen model tensor {name}')
        value = arena.tensor(name)
        if tuple(value.shape) != tuple(parameter.shape):
            raise SharedModelError(f'{name}: dense shape mismatch')
        parent, _, leaf = name.rpartition('.')
        model.diffusion_model.get_submodule(parent)._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        consumed.add(name)
    # Non-checkpoint sampling tables may be created on CPU; no backbone is initialized.
    model.model_sampling = model_base.HunyuanImage3Sampling(model_config)
    missing = set(manifest['tensors']) - consumed
    if missing:
        raise SharedModelError(f'Unconsumed shared tensors: {sorted(missing)[:12]}')
    meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
    if meta_buffers:
        raise SharedModelError(f'Unsupported generated meta buffers: {meta_buffers[:8]}')
    model.requires_grad_(False)
    capture_masters(model, strict=True)
    verify_masters(model)
    model.tokenizer = Tokenizer.from_file(loader.TOKENIZER_PATH)
    model._aitk_store = {'arena': arena, 'client': client, 'identity': identity}
    # Model-owned non-dynamic ops preserve bank-resident kernels. Dynamic AIMDO
    # packing/lookahead is deliberately unavailable and requires separate measurements.
    patcher = SharedModelPatcher(model, management.get_torch_device(), torch.device('cpu'))
    patcher.model_options.setdefault('transformer_options', {})['prefetch_dynamic_vbars'] = False
    return patcher
