import React from 'react';
import type { ModelArch } from '@/app/jobs/new/options';
import { defaultComfySampleConfig } from '@/helpers/defaultSamples';

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [{
  name: 'krea2_edit',
  label: 'Krea 2 Identity/Edit (krea2edit extension)',
  group: 'experimental',
  generate: { disabled: true },
  defaults: {
    'config.process[0].model.name_or_path': ['krea/Krea-2-Raw', ''],
    'config.process[0].model.quantize': [true, false],
    'config.process[0].model.qtype': ['convrotint6', 'qfloat8'],
    'config.process[0].model.quantize_te': [true, false],
    'config.process[0].model.qtype_te': ['convrot8', 'qfloat8'],
    'config.process[0].model.low_vram': [true, false],
    'config.process[0].model.model_kwargs': [{ fit_refs: true, multi_ref: false, checkpoint_every_n: 1 }, {}],
    'config.process[0].network.conv': [undefined, 16],
    'config.process[0].network.conv_alpha': [undefined, 16],
    'config.process[0].train.batch_size': [1, 1],
    'config.process[0].train.timestep_type': ['weighted', 'sigmoid'],
    'config.process[0].train.cache_text_embeddings': [true, false],
    'config.process[0].train.unload_text_encoder': [false, false],
    'config.process[0].datasets[x].flip_x': [false, false],
    'config.process[0].datasets[x].flip_y': [false, false],
    'config.process[0].sample.comfy.enabled': [true, false],
    'config.process[0].sample.comfy.workflow_path': [
      'config/comfy_templates/krea2_edit_lora_sample.json.njk', defaultComfySampleConfig.workflow_path,
    ],
  },
  disableSections: ['network.conv', 'train.unload_text_encoder'],
  additionalSections: ['sample.ctrl_img', 'model.low_vram', 'model.layer_offloading',
    'model.assistant_lora_path', 'model.text_encoder_path', 'model.vae_path'],
  datasetControlPathListMax: 2,
  customModelSelectOptions: [
    {
      type: 'checkbox', label: 'Fit References to Target',
      getValue: config => config.config.process[0].model.model_kwargs?.fit_refs ?? true,
      onChange: (value, _config, set) => set(value, 'config.process[0].model.model_kwargs.fit_refs'),
    },
    {
      type: 'checkbox', label: 'Train with Two References',
      getValue: config => config.config.process[0].model.model_kwargs?.multi_ref ?? false,
      onChange: (value, _config, set) => set(value, 'config.process[0].model.model_kwargs.multi_ref'),
    },
    {
      type: 'number', label: 'Checkpoint Every N Transformer Blocks', min: 1, step: 1,
      getValue: config => config.config.process[0].model.model_kwargs?.checkpoint_every_n ?? 1,
      onChange: (value, _config, set) => set(value, 'config.process[0].model.model_kwargs.checkpoint_every_n'),
    },
  ],
  modelNotes: <>
    <p>Uses the krea2edit extension's grounded identity/edit conditioning and fitted references.
      Batch size must be 1; disable horizontal/vertical flips and leave Unload TE off.</p>
    <p>Select one reference dataset, or enable Train with Two References and select two ordered folders.
      Cache Text Embeddings reduces memory on 24 GB GPUs and freezes grounding jitter in the cache.</p>
    <p>Samples require ComfyUI and the Krea 2 Edit LoRA (one reference) workflow. Set one source image per sample.
      The Helper LoRA Path configures the frozen training helper; Comfy Inference LoRA is configured separately.</p>
  </>,
}];
