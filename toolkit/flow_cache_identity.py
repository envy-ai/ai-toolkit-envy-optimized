"""Versioned component identity for NEW specialized models' text caches only.

No tensor hashing, GPU work or modification of ordinary/legacy Qwen cache keys.
File identities use size/mtime; remote components should be pinned to revisions.
"""
import hashlib
import json
from pathlib import Path
from toolkit.control_image import KREA_EDIT_CONTROL_PRESENTATION


def local_component_identity(value):
    if not isinstance(value, str):
        return None
    path = Path(value).expanduser()
    if path.is_file():
        stat = path.stat()
        return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    if path.is_dir():
        return [local_component_identity(str(child)) for child in sorted(path.rglob('*'))
                if child.is_file() and child.suffix in ('.safetensors', '.bin', '.json', '.yaml', '.model', '.txt')]
    return None


def _component_info(component):
    if component is None:
        return None
    config = getattr(component, 'config', None)
    source = getattr(config, '_name_or_path', None) or getattr(component, 'name_or_path', None)
    loaded = getattr(component, 'aitk_load_source', None)
    loaded = loaded if isinstance(loaded, dict) else {}
    return {'class': f'{type(component).__module__}.{type(component).__name__}',
            'source': source, 'local': local_component_identity(source),
            'loaded': loaded, 'loaded_local': {key: local_component_identity(value) for key, value in loaded.items()},
            'loaded_files': [local_component_identity(value) for value in getattr(component, 'aitk_load_files', ())],
            'commit': getattr(config, '_commit_hash', None),
            'model_max_length': getattr(component, 'model_max_length', None),
            'padding_side': getattr(component, 'padding_side', None)}


def resolved_flow_components(model):
    """Stat actual loaded components without copies, tensor hashes or GPU work.

    Used by new-model SliderSpace resume/discovery AFTER model loading. Loader
    provenance disambiguates implicit defaults and Comfy-ranked single files.
    """
    config = model.model_config
    transformer = model.transformer
    if model.arch == 'anima':
        transformer = model.trainable_model.transformer
    payload = {'version': 1, 'arch': model.arch,
        'text_identity': specialized_text_cache_identity(model),
        'transformer': _component_info(transformer), 'vae': _component_info(model.vae),
        'paths': {key: getattr(config, key, None) for key in ('name_or_path', 'vae_path')},
        'local': {key: local_component_identity(getattr(config, key, None))
                  for key in ('name_or_path', 'vae_path')},
        'precision': {key: getattr(config, key, None) for key in ('quantize', 'qtype', 'quantize_te', 'qtype_te')},
        'dtype': str(getattr(model, 'torch_dtype', None)),
        'vae_dtype': str(getattr(model, 'vae_torch_dtype', None))}
    return payload


def component_identity_digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def resolved_semantic_components(model, processor, name_or_path):
    return {'version': 1, 'source': name_or_path, 'model': _component_info(model),
            'processor': _component_info(processor), 'local': local_component_identity(name_or_path),
            'image_size': model.config.image_size, 'image_mean': list(processor.image_mean),
            'image_std': list(processor.image_std)}


def specialized_text_cache_identity(model, *, include_qwen=False):
    """Call once after loading components, before constructing dataset cache keys."""
    arch = getattr(model, 'arch', None)
    if arch not in ('krea2', 'anima', 'ideogram4') and not (include_qwen and arch == 'qwen_image_2'):
        return None
    config = model.model_config
    kwargs = getattr(config, 'model_kwargs', {}) or {}
    paths = {key: getattr(config, key, None) for key in ('name_or_path', 'text_encoder_path')}
    # These affect text/reference presentation, unlike CFG, inference schedules,
    # rank, loss weight and preview/discovery counts, which must not enter a TE key.
    encoding_kwargs = {key: value for key, value in kwargs.items() if key in (
        'text_encoder_path', 'text_encoder_revision', 'tokenizer_path', 'tokenizer_revision',
        'max_text_length', 'max_sequence_length', 'edit', 'instruction', 'match_target_res',
        'vlm_max_pixels')}
    encoders = getattr(model, 'text_encoder', None)
    encoders = encoders if isinstance(encoders, (list, tuple)) else [encoders]
    tokenizers = getattr(model, 'tokenizer', None)
    tokenizers = tokenizers if isinstance(tokenizers, (list, tuple)) else [tokenizers]
    if getattr(model, 't5_tokenizer', None) is not None:
        tokenizers = [*tokenizers, model.t5_tokenizer]
    payload = {'version': 1, 'arch': arch, 'paths': paths,
               'local': {key: local_component_identity(value) for key, value in paths.items()},
               'encoding_kwargs': encoding_kwargs,
               'kwargs_local': {key: local_component_identity(value) for key, value in encoding_kwargs.items()},
               'max_text_length': getattr(model, 'max_text_length', None),
               'max_sequence_length': getattr(model, 'max_sequence_length', None),
               'te_dtype': str(getattr(model, 'te_torch_dtype', None)),
               'quantize_te': getattr(config, 'quantize_te', False), 'qtype_te': getattr(config, 'qtype_te', None),
               'encoders': [_component_info(value) for value in encoders],
               'tokenizers': [_component_info(value) for value in tokenizers],
               'processors': [_component_info(getattr(model, key, None))
                              for key in ('processor', 'vl_processor')]}
    presentation = getattr(model, '_specialized_control_presentation', None)
    if presentation is not None:
        payload['control_presentation'] = presentation
        payload['control_loader'] = local_component_identity(str(Path(__file__).parent / 'control_image.py'))
    if arch == 'anima':
        payload['conditioner'] = _component_info(getattr(getattr(model, 'trainable_model', None), 'text_conditioner', None))
    source_root = Path(__file__).resolve().parents[1]
    payload['wrapper'] = local_component_identity(str(source_root / f'extensions_built_in/diffusion_models/{arch}/{arch}.py'))
    if arch == 'ideogram4':
        payload['caption_digest'] = local_component_identity(str(Path(__file__).parent / 'ideogram_caption.py'))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def prepare_specialized_conditioning(model):
    """Called ONLY for new specialized modes, before dataset/cache construction.

    Raw Krea edit refs still retain their own dimensions, but the VLM must see
    the same transparency-composited pixels as the latent-reference loader.
    Ordinary training and legacy Qwen retain their current flags/cache identities.
    """
    if getattr(model, 'arch', None) == 'krea2' and getattr(model, 'is_edit', False):
        model.cache_processed_control_text_embeddings = True
        model._specialized_control_presentation = KREA_EDIT_CONTROL_PRESENTATION
