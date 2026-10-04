"""Pure configuration contracts for shared flow-training objectives.

No model imports: invalid jobs must fail before allocating/downloading weights.
Availability in the UI is enabled separately, after trainer integration tests.
"""

import re
from dataclasses import dataclass


MODE_ALIASES = {'qwen_flow_dpo': 'flow_dpo',
                'qwen_guidance_distillation': 'guidance_distillation'}
SPECIALIZED_MODES = ('fizgig_image_slider', 'fizgig_prompt_slider', 'flow_dpo',
                     'guidance_distillation', 'sliderspace', 'diffusion_kto')


@dataclass(frozen=True)
class FlowTrainingCapability:
    name: str
    bucket_divisibility: int
    edit_references: bool = False
    image_only_cfg: bool = False


FLOW_TRAINING_MODELS = {
    'qwen_image_2': FlowTrainingCapability('Qwen Image 2.1', 32, edit_references=True),
    'krea2': FlowTrainingCapability('Krea 2', 16, edit_references=True),
    'anima': FlowTrainingCapability('Anima', 32),
    'ideogram4': FlowTrainingCapability('Ideogram 4', 16, image_only_cfg=True),
}


def canonical_training_mode(mode):
    return MODE_ALIASES.get(mode, mode)


def cfg_reference_mode(model):
    if model.get('arch') != 'ideogram4':
        return 'negative_prompt'
    mode = (model.get('model_kwargs') or {}).get('ideogram_cfg_reference', 'image_only')
    if mode not in ('image_only', 'negative_prompt'):
        raise ValueError('Ideogram CFG reference must be image_only or negative_prompt')
    return mode


def validate_specialized_model(config, mode=None):
    """Model/network compatibility only; objective-specific validation stays local."""
    mode = canonical_training_mode(mode or config.get('type'))
    if mode not in SPECIALIZED_MODES:
        raise ValueError(f'Unknown specialized flow training mode: {mode}')
    model = config.get('model') or {}
    arch = model.get('arch')
    capability = FLOW_TRAINING_MODELS.get(arch)
    if capability is None:
        raise ValueError(f'{mode} supports Qwen Image 2.1, Krea 2, Anima and Ideogram 4')
    networks = ('lora', 'dora') if mode.startswith('fizgig_') else ('lora',)
    if (config.get('network') or {}).get('type') not in networks:
        labels = {'lora': 'LoRA', 'dora': 'DoRA'}
        suffix = ' only' if len(networks) == 1 else ''
        raise ValueError(f'{mode} requires {" or ".join(labels[item] for item in networks)}{suffix}')
    kwargs = model.get('model_kwargs') or {}
    if arch == 'anima' and kwargs.get('train_text_conditioner'):
        raise ValueError(f'{mode} requires a frozen Anima text conditioner; disable train_text_conditioner')
    cfg_reference_mode(model)
    comfy = ((config.get('sample') or {}).get('comfy') or {})
    if comfy.get('enabled') and arch in ('anima', 'ideogram4'):
        workflow = str(comfy.get('workflow_path', '')).replace('\\', '/').split('/')[-1]
        known = {'krea2_lora_sample.json.njk': 'krea2', 'krea2_lora_sample_batch_easy_use.json.njk': 'krea2',
                 'qwen_image_2_lora_sample.json.njk': 'qwen_image_2',
                 'qwen_image_2_lora_sample_batch_easy_use.json.njk': 'qwen_image_2',
                 'anima_lora_sample.json.njk': 'anima', 'ideogram4_lora_sample.json.njk': 'ideogram4'}
        if workflow in known and known[workflow] != arch:
            raise ValueError(f'Select {arch}_lora_sample.json.njk for {capability.name} Comfy previews, not {known[workflow]}')
    # Do not introduce new restrictions on old Qwen jobs during the port.
    if arch != 'qwen_image_2' or mode == 'diffusion_kto':
        # A frozen Qwen helper is part of both the policy and its reference.
        auxiliary_paths = ('inference_lora_path', 'unconditional_lora_path') if arch == 'qwen_image_2' else (
            'assistant_lora_path', 'inference_lora_path', 'unconditional_lora_path')
        if any(model.get(key) for key in auxiliary_paths):
            if arch == 'qwen_image_2':
                raise ValueError(f'{mode} does not support inference or unconditional LoRA paths; frozen training helpers are supported')
            raise ValueError(f'{mode} requires an unadapted base without auxiliary LoRA paths')
        name = str(model.get('name_or_path', ''))
        if kwargs.get('is_distilled') or re.search(r'(?i)(?:^|[\W_])(?:turbo|lightning|distilled)(?:$|[\W_])', name):
            raise ValueError(f'{mode} currently requires a base checkpoint, not a turbo/step-distilled variant')
        if arch == 'krea2' and mode in ('fizgig_image_slider', 'fizgig_prompt_slider', 'sliderspace'):
            if kwargs.get('edit') or kwargs.get('kv_cache'):
                raise ValueError(f'{mode} is text-to-image; disable Krea edit and kv_cache')
    return capability


def validate_edit_references(config, datasets, *, paired=False):
    """Control 1 can be a target/rejected image, never an actual edit reference."""
    model = config.get('model') or {}
    capability = FLOW_TRAINING_MODELS[model['arch']]
    keys = ('control_path_2', 'control_path_3') if paired else (
        'control_path', 'control_path_1', 'control_path_2', 'control_path_3')
    has_references = any(dataset.get(key) for dataset in datasets for key in keys)
    if has_references and not capability.edit_references:
        raise ValueError(f'{capability.name} does not support edit-source image conditioning')
    if has_references and model['arch'] == 'krea2' and not (model.get('model_kwargs') or {}).get('edit'):
        raise ValueError('Krea edit-source images require model_kwargs.edit=true')
    return has_references
