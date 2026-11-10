import type { JobConfig, DatasetConfig, DiffusionKTOConfig, SampleItem } from '@/types';

export const FIBO2_WORKFLOW_PATH = 'config/comfy_templates/fibo2_lora_sample.json.njk';
export const FIBO2_TURBO_WORKFLOW_PATH = 'config/comfy_templates/fibo2_edit_turbo_lora_sample.json.njk';
export const fibo2WorkflowPaths = [FIBO2_WORKFLOW_PATH, FIBO2_TURBO_WORKFLOW_PATH];

/** Apply graph defaults only on an explicit Turbo preview selection. */
export function applyComfyWorkflowSelection(config: JobConfig, workflowPath: string): JobConfig {
  const process = config.config.process[0];
  const turboPreview = process.model.arch === 'fibo2' && workflowPath === FIBO2_TURBO_WORKFLOW_PATH;
  const sample = {
    ...process.sample,
    ...(turboPreview ? { sample_steps: 4, guidance_scale: 1 } : {}),
    comfy: {
      ...process.sample.comfy,
      workflow_path: workflowPath,
      ...(turboPreview ? {
        model: 'd/fibo2_edit_turbo_int8_convrot.safetensors',
        text_encoder: 'd/qwen3vl_4b_instruct_heretic_7refusal_int8_convrot.safetensors',
        vae: 'flux2-vae.safetensors',
        sampler: 'euler',
        scheduler: 'simple',
      } : {}),
    },
  };
  return {
    ...config,
    config: {
      ...config.config,
      process: config.config.process.map((entry, index) => index === 0 ? { ...entry, sample } : entry),
    },
  };
}

export const architectureModelSourceFields = ['text_encoder_path', 'te_name_or_path', 'vae_path',
  'extras_name_or_path', 'assistant_lora_path', 'inference_lora_path', 'unconditional_lora_path',
  'lora_path', 'unet_path', 'refiner_name_or_path', 'model_paths'] as const;
export const architectureComfyAdapterSourceFields = ['inference_lora_absolute_path', 'inference_lora_path', 'inference_lora_enabled'] as const;

/** Keep preview prompt/settings while removing references to the old architecture. */
export function clearArchitectureSampleReferences(sample: SampleItem): SampleItem {
  const next = { ...sample };
  for (const key of ['ctrl_img', 'ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3'] as const) delete next[key];
  return next;
}

/** Mirrored by toolkit/training_capabilities.py; tested against that table. */
export const flowTrainingModels: Record<string, { label: string; bucketDivisibility: number; editReferences: boolean }> = {
  qwen_image_2: { label: 'Qwen Image 2.1', bucketDivisibility: 32, editReferences: true },
  krea2: { label: 'Krea 2', bucketDivisibility: 16, editReferences: true },
  anima: { label: 'Anima', bucketDivisibility: 32, editReferences: false },
  ideogram4: { label: 'Ideogram 4', bucketDivisibility: 16, editReferences: false },
  moss_image: { label: 'MOSS-Image', bucketDivisibility: 16, editReferences: true },
  fibo2: { label: 'FIBO2 Dev', bucketDivisibility: 16, editReferences: true },
};
const mossVariants = ['moss_image', 'moss_image:base', 'moss_image:edit', 'moss_image:base_edit'];
export const canonicalFlowArch = (arch: string) => mossVariants.includes(arch) ? 'moss_image' : arch;
export const mossVariantKwargs = (arch: string): Record<string, unknown> => mossVariants.includes(arch)
  ? { uncond_skip_layers: [22], ...(['moss_image:edit', 'moss_image:base_edit'].includes(arch)
    ? { edit: true, kv_cache: true, match_target_res: true } : {}) } : {};
export const effectiveFlowModel = <T extends { arch: string; model_kwargs?: Record<string, any> }>(model: T) => ({
  ...model, arch: canonicalFlowArch(model.arch), model_kwargs: { ...mossVariantKwargs(model.arch), ...model.model_kwargs },
});
export const flowTrainingModelOptions = [
  ...Object.entries(flowTrainingModels).map(([value, model]) => ({ value, label: value === 'moss_image' ? 'MOSS-Image SFT' : model.label })),
  { value: 'moss_image:base', label: 'MOSS-Image Base' },
  { value: 'moss_image:edit', label: 'MOSS-Image SFT [Edit Training]' },
  { value: 'moss_image:base_edit', label: 'MOSS-Image Base [Edit Training]' },
];
export const trainingModeAliases: Record<string, string> = {
  qwen_flow_dpo: 'flow_dpo', qwen_guidance_distillation: 'guidance_distillation',
};
export const canonicalTrainingMode = (mode: string) => trainingModeAliases[mode] ?? mode;
export const specializedTrainingModes = ['fizgig_image_slider', 'fizgig_prompt_slider', 'flow_dpo', 'guidance_distillation', 'sliderspace', 'diffusion_kto'];
export const defaultDiffusionKTOConfig: DiffusionKTOConfig = {
  beta: 1, liked_weight: 1, disliked_weight: 1, reference_estimator: 'batch_mean', score_window_size: 4,
};
export const isSpecializedTrainingMode = (mode: string) => specializedTrainingModes.includes(canonicalTrainingMode(mode));
export const trainingModeSupported = (arch: string, mode: string) => !isSpecializedTrainingMode(mode)
  || (!!flowTrainingModels[canonicalFlowArch(arch)] && (!['moss_image:edit', 'moss_image:base_edit'].includes(arch)
    || ['flow_dpo', 'guidance_distillation'].includes(canonicalTrainingMode(mode))));
export const trainingModeVisible = (arch: string, mode: string, selected: string) => mode === selected
  || (!trainingModeAliases[mode] && trainingModeSupported(arch, mode));
export const cfgNegativeTextEnabled = (model: { arch: string; model_kwargs?: Record<string, unknown> }) =>
  model.arch !== 'ideogram4' || model.model_kwargs?.ideogram_cfg_reference === 'negative_prompt';

/** Installed TextEncodeFibo2 has no switch to disable glyph images in negatives. */
const fibo2ComfyNegativeHasGlyphs = (prompt: string) => {
  try {
    const parsed = JSON.parse(prompt);
    const caption = parsed && Object.prototype.hasOwnProperty.call(parsed, 'structured_caption') ? parsed.structured_caption : parsed;
    return Array.isArray(caption?.text_render) && caption.text_render.some((entry: unknown) =>
      !!entry && typeof entry === 'object' && typeof (entry as { text?: unknown }).text === 'string'
      && (entry as { text: string }).text.length > 0);
  } catch {
    // Invalid JSON is a supported plain prompt and contains no glyph entries.
    return false;
  }
};

/** Pure pre-load checks used by imports, both save buttons and the API. */
export function validateTrainingCapabilities(config: JobConfig): string[] {
  const process = config?.config?.process?.[0];
  if (!process) return ['Job configuration must contain a training process.'];
  const mode = canonicalTrainingMode(process.type);
  const errors: string[] = [];
  for (const key of ['record_training_examples', 'record_training_rng', 'training_maps'] as const) {
    const value = process.logging?.[key];
    if (value !== undefined && typeof value !== 'boolean') errors.push(`Logging ${key} must be a boolean.`);
  }
  if (process.logging?.record_training_rng && process.logging?.record_training_examples === false) errors.push('Recording noise RNG state requires per-image loss logging.');
  for (const key of ['training_maps_every', 'training_maps_images'] as const) {
    const value = process.logging?.[key];
    if (value !== undefined && (!Number.isSafeInteger(value) || value <= 0)) errors.push(`Logging ${key} must be a positive integer.`);
  }
  if (process.logging?.training_maps) {
    if (!['sd_trainer', 'ui_trainer', 'diffusion_trainer'].includes(mode)
        || !['qwen_image', 'qwen_image_2', 'krea2', 'moss_image'].includes(canonicalFlowArch(process.model?.arch ?? '')))
      errors.push('Training maps support ordinary Qwen Image 2.1, Krea 2 and MOSS-Image training.');
    if (!['lora', 'dora'].includes(process.network?.type ?? '')) errors.push('Training maps require LoRA or DoRA.');
    if ((process.model && effectiveFlowModel(process.model).model_kwargs?.edit) || process.train?.train_text_encoder)
      errors.push('Training maps require text-to-image training with a frozen text encoder.');
  }
  const h3PhotoFlag = process.model?.model_kwargs?.h3_photo_training;
  if (h3PhotoFlag !== undefined && typeof h3PhotoFlag !== 'boolean') {
    errors.push('MiniMax-H3 photo training flag must be true or false.');
  }
  if (h3PhotoFlag === true) {
    if (!['minimax_h3', 'minimax_h3_image'].includes(process.model.arch) ||
        !['sd_trainer', 'ui_trainer', 'diffusion_trainer'].includes(mode)) {
      errors.push('MiniMax-H3 photo training requires an ordinary MiniMax-H3 training job.');
    }
    if (process.model.arch === 'minimax_h3' &&
        !['fl2va', 'fl2va_pruned'].includes(process.model.model_kwargs?.partition ?? 'fl2va_pruned')) {
      errors.push('MiniMax-H3 photo training requires the fl2va or fl2va_pruned base partition.');
    }
    if (process.model.arch === 'minimax_h3_image' &&
        ![undefined, 'image'].includes(process.model.model_kwargs?.partition)) {
      errors.push('MiniMax-H3-Image photo training requires the image partition.');
    }
    if (['do_cfg', 'do_random_cfg', 'do_guidance_loss', 'do_differential_guidance',
      'diff_output_preservation', 'blank_prompt_preservation', 'do_prior_divergence',
      'train_turbo', 'inverted_mask_prior', 'correct_pred_norm'].some(key =>
      !!(process.train as unknown as Record<string, unknown>)[key]) ||
      (process.train.frequency_loss_type ?? 'none') !== 'none') {
      errors.push('MiniMax-H3 photo training cannot combine CFG, guidance or preservation objectives with the image training adapter.');
    }
  }
  if (process.model?.arch === 'minimax_h3_image' && process.sample?.comfy?.enabled) {
    const workflow = (process.sample.comfy.workflow_path ?? '').replaceAll('\\', '/').split('/').pop();
    if (workflow !== 'minimax_h3_image_lora_sample.json.njk') {
      errors.push('MiniMax-H3-Image Comfy previews require the MiniMax-H3-Image LoRA still-image workflow.');
    }
    if (process.sample.samples?.some(item => item.ctrl_img || item.ctrl_img_1 || item.ctrl_img_2 || item.ctrl_img_3)) {
      errors.push('The bundled MiniMax-H3-Image Comfy workflow supports text-to-image only; remove sample reference images or use native samples.');
    }
  }
  if (process.model?.arch === 'krea2_edit') errors.push(...validateKrea2Edit(config));
  if (process.model?.arch === 'fibo2') {
    const kwargs = process.model.model_kwargs ?? {};
    if (!process.model.vae_path) errors.push('FIBO2 requires an explicit FLUX.2 VAE checkpoint in model.vae_path.');
    if (process.train?.train_text_encoder) errors.push('FIBO2 training requires a frozen Qwen3-VL-4B text encoder.');
    if (kwargs.turbo || kwargs.is_distilled || /(?:^|[\W_])(?:turbo|lightning|distilled)(?:$|[\W_])/i.test(process.model.name_or_path ?? '')) {
      errors.push('FIBO2 training requires the non-distilled Dev checkpoint. Edit Turbo is inference-only; select it only in an inference workflow.');
    }
    if (process.sample?.comfy?.enabled && !fibo2WorkflowPaths.some(path => path.split('/').pop()
      === (process.sample.comfy.workflow_path ?? '').replaceAll('\\', '/').split('/').pop())) {
      errors.push('FIBO2 Comfy previews require the FIBO2 LoRA image / edit or FIBO2 Edit Turbo LoRA image / edit workflow and compatible FIBO2, Qwen3-VL-4B and FLUX.2 VAE files.');
    }
    if (process.sample?.comfy?.enabled && fibo2ComfyNegativeHasGlyphs(process.sample.comfy.negative_prompt ?? '')) {
      errors.push('Installed ComfyUI FIBO2 cannot keep glyph-bearing structured negatives text-only. Use an empty/plain Comfy negative prompt or native previews.');
    }
  }
  if (canonicalFlowArch(process.model?.arch ?? '') === 'moss_image' && process.sample?.comfy?.enabled) {
    const workflow = (process.sample.comfy.workflow_path ?? 'config/comfy_templates/krea2_lora_sample.json.njk')
      .replaceAll('\\', '/').split('/').pop() ?? '';
    if (workflow.startsWith('krea2_') && workflow.endsWith('.json.njk')) {
      errors.push('Krea 2 ComfyUI workflows are incompatible with MOSS-Image. Select MOSS-Image LoRA image for text-to-image previews, or disable Use ComfyUI Renderer to use native MOSS previews.');
    }
  }
  if (!isSpecializedTrainingMode(mode)) return errors;
  const model = process.model && effectiveFlowModel(process.model);
  if (!model || !flowTrainingModels[model.arch]) return [...errors, `${mode} supports Qwen Image 2.1, Krea 2, Anima, Ideogram 4, MOSS-Image and FIBO2 Dev.`];
  const networks = mode.startsWith('fizgig_') ? ['lora', 'dora'] : ['lora'];
  if (!networks.includes(process.network?.type ?? '')) errors.push(`${mode} requires ${networks.length === 1 ? 'LoRA only' : 'LoRA or DoRA'}.`);
  const kwargs = model.model_kwargs ?? {};
  if (model.arch === 'moss_image') {
    if (kwargs.kv_cache && !kwargs.edit) errors.push('MOSS-Image kv_cache requires model_kwargs.edit=true.');
    if ((kwargs.edit || kwargs.kv_cache) && !['flow_dpo', 'guidance_distillation'].includes(mode)) errors.push(`${mode} is text-to-image. Disable MOSS-Image edit and kv_cache.`);
    const skipped = kwargs.uncond_skip_layers ?? [22];
    if (!Array.isArray(skipped) || skipped.some((index: unknown) => !Number.isInteger(index) || Number(index) < 0 || Number(index) > 41)) errors.push('MOSS-Image uncond_skip_layers must contain zero-based block indices in [0, 41].');
  }
  if (mode.startsWith('fizgig_') && process.train?.do_differential_guidance) {
    const scale = process.train.differential_guidance_scale ?? 3;
    if (typeof scale !== 'number' || !Number.isFinite(scale) || scale < 0) {
      errors.push('Differential Guidance Scale must be finite and nonnegative.');
    }
  }
  if (process.sample?.comfy?.enabled && ['anima', 'ideogram4'].includes(model.arch)) {
    const workflow = (process.sample.comfy.workflow_path ?? '').replaceAll('\\', '/').split('/').pop() ?? '';
    const known: Record<string, string> = { 'krea2_lora_sample.json.njk': 'krea2', 'krea2_lora_sample_batch_easy_use.json.njk': 'krea2',
      'qwen_image_2_lora_sample.json.njk': 'qwen_image_2', 'qwen_image_2_lora_sample_batch_easy_use.json.njk': 'qwen_image_2',
      'anima_lora_sample.json.njk': 'anima', 'ideogram4_lora_sample.json.njk': 'ideogram4',
      'moss_image_lora_sample.json.njk': 'moss_image' };
    if (known[workflow] && known[workflow] !== model.arch) errors.push(`Select ${model.arch}_lora_sample.json.njk and compatible model/VAE/text encoder files for Comfy previews.`);
  }
  if (model.arch === 'anima' && kwargs.train_text_conditioner) errors.push('Specialized training requires a frozen Anima text conditioner. Disable train_text_conditioner.');
  if (model.arch === 'ideogram4' && !['image_only', 'negative_prompt'].includes(String(kwargs.ideogram_cfg_reference ?? 'image_only'))) errors.push('Choose image-only or text-negative Ideogram CFG reference.');
  if (model.arch !== 'qwen_image_2' || mode === 'diffusion_kto') {
    const unsupportedPaths = ['unconditional_lora_path',
      ...(!['qwen_image_2', 'fibo2'].includes(model.arch) ? ['assistant_lora_path'] : [])] as const;
    const configuredPaths = unsupportedPaths.filter(key => model[key as keyof typeof model]);
    if (configuredPaths.length) errors.push(`Unsupported auxiliary LoRA paths: ${configuredPaths.map(key => `model.${key}`).join(', ')}. Frozen Qwen/FIBO2 training helpers and preview-only inference LoRAs are supported.`);
    if (kwargs.is_distilled || /(?:^|[\W_])(?:turbo|lightning|distilled)(?:$|[\W_])/i.test(model.name_or_path ?? '')) errors.push('Specialized training currently requires a base checkpoint, not a turbo/step-distilled variant.');
    if (model.arch === 'krea2' && (mode.startsWith('fizgig_') || mode === 'sliderspace') && (kwargs.edit || kwargs.kv_cache)) errors.push('Fizgig and SliderSpace are text-to-image objectives. Disable Krea edit and kv_cache.');
  }
  const hasEditReferences = flowTrainingModels[model.arch].editReferences && (!['krea2', 'moss_image'].includes(model.arch) || !!kwargs.edit);
  if (process.datasets != null && !Array.isArray(process.datasets)) errors.push('Training datasets must be a list.');
  for (const dataset of Array.isArray(process.datasets) ? process.datasets : []) {
    if (!dataset || typeof dataset !== 'object') { errors.push('Each dataset must be a configuration object.'); continue; }
    const paired = mode === 'flow_dpo' || mode === 'fizgig_image_slider';
    const references = paired ? [dataset.control_path_2, dataset.control_path_3] :
      [dataset.control_path, dataset.control_path_1, dataset.control_path_2, dataset.control_path_3];
    if (references.some(Boolean) && (mode.startsWith('fizgig_') || !hasEditReferences)) errors.push('This mode/model does not support the configured edit-source images. Control Dataset 1 is a paired target in image sliders/DPO, not an edit reference.');
    if (model.arch === 'moss_image' && kwargs.edit && !references.some(Boolean)) errors.push('MOSS-Image edit training requires fixed edit-source images for every dataset.');
  }
  if (mode === 'guidance_distillation' && model.arch === 'ideogram4' && process.guidance_distillation?.objective === 'negative_only'
    && (!cfgNegativeTextEnabled(model) || !process.guidance_distillation.negative_prompt?.trim())) errors.push('Ideogram negative-only distillation requires text-negative CFG and a nonempty teacher negative prompt.');
  if (mode === 'diffusion_kto') errors.push(...validateDiffusionKTO(config));
  return [...new Set(errors)];
}

/** The external identity/edit extension has its own dataset and preview contract. */
export function validateKrea2Edit(config: JobConfig): string[] {
  const process = config.config.process[0];
  const kwargs = process.model.model_kwargs ?? {};
  const errors: string[] = [];
  if (process.train.batch_size !== 1) errors.push('Krea 2 Identity/Edit requires Batch Size 1.');
  if (process.train.unload_text_encoder) errors.push('Krea 2 Identity/Edit requires Unload TE off.');
  for (const key of ['fit_refs', 'multi_ref']) {
    if (kwargs[key] !== undefined && typeof kwargs[key] !== 'boolean') errors.push(`Krea 2 Identity/Edit ${key} must be a boolean.`);
  }
  const checkpointEvery = kwargs.checkpoint_every_n ?? 1;
  if (!Number.isInteger(checkpointEvery) || checkpointEvery < 1) errors.push('Checkpoint Every N Transformer Blocks must be a positive integer.');
  if (kwargs.edit || kwargs.kv_cache) errors.push('Krea 2 Identity/Edit uses fit_refs/multi_ref; remove the experimental edit/kv_cache settings.');
  if (!process.train.disable_sampling) {
    const comfy = process.sample?.comfy;
    if (!comfy?.enabled) errors.push('Krea 2 Identity/Edit samples require Use ComfyUI Renderer, or Disable Sampling.');
    else if ((comfy.workflow_path ?? '').replaceAll('\\', '/').split('/').pop() !== 'krea2_edit_lora_sample.json.njk') {
      errors.push('Krea 2 Identity/Edit samples require the Krea 2 Edit LoRA (one reference) workflow.');
    }
  }
  for (const dataset of Array.isArray(process.datasets) ? process.datasets : []) {
    if (!dataset || typeof dataset !== 'object') continue;
    if (dataset.flip_x || dataset.flip_y) errors.push('Krea 2 Identity/Edit requires horizontal and vertical flips off.');
    if (dataset.control_path_1 || dataset.control_path_2 || dataset.control_path_3) errors.push('Krea 2 Identity/Edit reference folders must use the ordered control_path list.');
    const refs = Array.isArray(dataset.control_path) ? dataset.control_path : dataset.control_path ? [dataset.control_path] : [];
    if (refs.length > 2 || refs.some(path => typeof path !== 'string' || !path.trim())) {
      errors.push('Krea 2 Identity/Edit supports at most two consecutive reference dataset folders.');
    } else if (refs.length > 1 && !kwargs.multi_ref) errors.push('Enable Train with Two References to configure a second reference dataset.');
  }
  return errors;
}

/** Runtime-shape checks also run on imported JSON/YAML and server requests. */
export function validateDiffusionKTO(config: JobConfig): string[] {
  const process = config.config.process[0];
  const raw = process as unknown as Record<string, any>;
  const train = raw.train ?? {};
  const supplied = raw.diffusion_kto ?? {};
  const errors: string[] = [];
  if (typeof supplied !== 'object' || Array.isArray(supplied)) return ['Diffusion-KTO settings must be an object.'];
  if (Object.keys(supplied).some(key => !(key in defaultDiffusionKTOConfig))) errors.push('Unknown Diffusion-KTO settings.');
  const settings = { ...defaultDiffusionKTOConfig, ...supplied };
  for (const field of ['beta', 'liked_weight', 'disliked_weight'] as const) {
    if (typeof settings[field] !== 'number' || !Number.isFinite(settings[field]) || settings[field] <= 0) errors.push(`Diffusion-KTO ${field} must be finite and positive.`);
  }
  if (!['batch_mean', 'score_window'].includes(settings.reference_estimator)) errors.push('Choose batch mean or score window for the KTO reference point.');
  if (!Number.isInteger(settings.score_window_size) || settings.score_window_size < 1) errors.push('KTO score window must be a positive integer.');
  const accumulation = train.gradient_accumulation ?? 1;
  if (!Number.isInteger(accumulation) || accumulation < 1) errors.push('KTO gradient accumulation must be a positive integer.');
  if (settings.reference_estimator === 'score_window' && (settings.score_window_size < 2 || accumulation < settings.score_window_size)) errors.push(`KTO score windows require at least 2 batches and Gradient Accumulation >= ${settings.score_window_size}.`);
  if ((train.gradient_accumulation_steps ?? 1) !== 1) errors.push('KTO uses Gradient Accumulation within one optimizer step, not gradient_accumulation_steps.');
  const network = raw.network ?? {};
  if (network.pretrained_lora_path || (!['qwen_image_2', 'fibo2'].includes(process.model.arch) && process.model.assistant_lora_path) || process.model.unconditional_lora_path) errors.push('KTO supports frozen Qwen/FIBO2 training helpers and preview-only inference LoRAs, but no pretrained or other auxiliary LoRAs.');
  const networkOptions = network.network_kwargs ?? {};
  if (typeof networkOptions !== 'object' || Array.isArray(networkOptions)) errors.push('KTO network_kwargs must be an object.');
  if ([network, networkOptions].some(values => ['dropout', 'rank_dropout', 'module_dropout'].some(key => values[key] != null && values[key] !== 0))) errors.push('KTO requires zero adapter dropout for exact replay.');
  if (network.all_layers || ['full_train_in_out', 'full_if_contains', 'is_ara'].some(key => networkOptions[key])) errors.push('KTO requires ordinary LoRA-only parameters, not full-layer adapters.');
  const effectiveModel = effectiveFlowModel(process.model);
  if (effectiveModel.model_kwargs.edit || effectiveModel.model_kwargs.kv_cache) errors.push('KTO initially supports text-to-image only; disable edit/kv_cache.');
  if (!train.cache_text_embeddings || train.unload_text_encoder) errors.push('KTO requires Cache Text Embeddings, not Unload TE.');
  if (train.train_text_encoder || train.train_unet === false || train.noise_scheduler !== 'flowmatch' || (train.timestep_type ?? 'shift') !== 'shift' || (train.loss_type ?? 'mse') !== 'mse') errors.push('KTO requires transformer-only LoRA training with flowmatch, shifted times and MSE.');
  if (['do_cfg', 'do_random_cfg', 'do_guidance_loss', 'do_differential_guidance', 'diff_output_preservation',
    'blank_prompt_preservation', 'do_prior_divergence', 'train_turbo', 'inverted_mask_prior', 'do_fft_loss',
    'correct_pred_norm', 'learnable_snr_gos'].some(key => train[key]) || (train.frequency_loss_type ?? 'none') !== 'none'
    || train.ema_config?.use_ema || train.diffusion_feature_extractor_path || raw.adapter || raw.embedding || raw.decorator) errors.push('KTO cannot be combined with other training objectives, EMA or trainable adapters.');
  if (['min_snr_gamma', 'snr_gamma', 'snr_weight', 'noise_offset'].some(key => train[key] != null && train[key] !== 0)) errors.push('KTO uses the unweighted velocity-error surrogate without SNR/noise offsets.');
  const low = train.min_denoising_steps ?? 0, high = train.max_denoising_steps ?? 999;
  if (typeof low !== 'number' || typeof high !== 'number' || !Number.isFinite(low) || !Number.isFinite(high) || !(0 <= low && low < high && high <= 1000)) errors.push('KTO requires 0 <= min timestep < max timestep <= 1000.');
  if (!Array.isArray(process.datasets) || !process.datasets.length) errors.push('KTO requires individually labeled image/caption datasets.');
  for (const dataset of Array.isArray(process.datasets) ? process.datasets : []) {
    if (!dataset || typeof dataset !== 'object') continue;
    const data = dataset as unknown as Record<string, any>;
    if (!['liked', 'disliked'].includes(data.kto_label)) errors.push('Every KTO image folder needs a Liked or Disliked label.');
    if (!data.folder_path) errors.push('Select a KTO image folder.');
    if (!data.cache_latents_to_disk && !data.cache_latents) errors.push('KTO requires Cache Latents on every dataset.');
    if (data.buckets === false || (data.num_frames ?? 1) !== 1 || data.auto_frame_count) errors.push('KTO requires bucketed still images.');
    if (['control_path', 'control_path_1', 'control_path_2', 'control_path_3', 'control_from_same_folder', 'inpaint_path',
      'unconditional_path', 'mask_path', 'alpha_mask', 'is_reg', 'prior_reg', 'random_scale', 'random_crop',
      'shuffle_tokens', 'token_dropout_rate'].some(key => data[key]) || ['controls', 'augmentations', 'augments', 'random_triggers'].some(key => data[key]?.length)) errors.push('KTO requires fixed unmasked images/captions, not paired/edit/reg controls.');
    if ((data.network_weight ?? 1) !== 1) errors.push('KTO dataset LoRA Weight must be 1.');
    if (typeof (data.loss_multiplier ?? 1) !== 'number' || !Number.isFinite(data.loss_multiplier ?? 1) || (data.loss_multiplier ?? 1) < 0) errors.push('KTO dataset loss multiplier must be finite and nonnegative.');
    if (!Number.isInteger(data.num_repeats ?? 1) || (data.num_repeats ?? 1) < 1) errors.push('KTO dataset repeats must be a positive integer.');
  }
  return [...new Set(errors)];
}

export function activateDiffusionKTO(config: JobConfig, defaultDataset: DatasetConfig): JobConfig {
  const process = config.config.process[0];
  process.diffusion_kto = { ...defaultDiffusionKTOConfig };
  process.network ??= { type: 'lora', linear: 32, linear_alpha: 32, conv: 16, conv_alpha: 16,
    lokr_full_rank: true, lokr_factor: -1, network_kwargs: {} };
  process.network.type = 'lora';
  const train = process.train as unknown as Record<string, any>;
  Object.assign(train, { train_unet: true, train_text_encoder: false, noise_scheduler: 'flowmatch',
    timestep_type: 'shift', cache_text_embeddings: true, unload_text_encoder: false, loss_type: 'mse',
    content_or_style: 'balanced', gradient_accumulation_steps: 1, frequency_loss_type: 'none',
    min_snr_gamma: 0, snr_gamma: 0, snr_weight: 0, noise_offset: 0 });
  for (const key of ['do_cfg', 'do_random_cfg', 'do_guidance_loss', 'do_differential_guidance',
    'diff_output_preservation', 'blank_prompt_preservation', 'do_prior_divergence', 'train_turbo',
    'inverted_mask_prior', 'do_fft_loss', 'correct_pred_norm', 'learnable_snr_gos']) train[key] = false;
  if (process.train.ema_config) process.train.ema_config.use_ema = false;
  // Keep explicit pretrained/auxiliary paths/edit overrides so validation can
  // explain the conflict; never silently use a different base teacher.
  process.trigger_word = null;
  if (!process.datasets.length) process.datasets = [structuredClone(defaultDataset)];
  for (const dataset of process.datasets) {
    Object.assign(dataset, { cache_latents_to_disk: true, caption_dropout_rate: 0, network_weight: 1,
      shuffle_tokens: false, is_reg: false, controls: [], mask_path: null });
    dataset.kto_label ??= 'liked';
    for (const key of ['control_path', 'control_path_1', 'control_path_2', 'control_path_3', 'anchor_path', 'multipoint_images']) delete dataset[key as keyof DatasetConfig];
  }
  return config;
}

/** Specialized architecture switches never silently choose a new checkpoint. */
export function switchSpecializedArchitecture(config: JobConfig, arch: string): { config: JobConfig; notice: string } {
  const next = structuredClone(config);
  const process = next.config.process[0];
  if (!isSpecializedTrainingMode(process.type) || !trainingModeSupported(arch, process.type)) throw new Error('Unsupported specialized training architecture.');
  process.model.arch = arch;
  process.model.name_or_path = '';
  for (const key of architectureModelSourceFields) delete (process.model as unknown as Record<string, unknown>)[key];
  process.model.model_kwargs = mossVariantKwargs(arch);
  process.sample.samples = process.sample.samples.map(clearArchitectureSampleReferences);
  if (process.sample.comfy) {
    process.sample.comfy.inference_lora = '';
    for (const key of architectureComfyAdapterSourceFields) delete (process.sample.comfy as unknown as Record<string, unknown>)[key];
  }
  if (arch === 'fibo2') {
    process.train.noise_scheduler = 'flowmatch';
    process.train.timestep_type = 'shift';
    process.sample.sampler = 'flowmatch';
    if (process.sample.comfy) process.sample.comfy.workflow_path = 'config/comfy_templates/fibo2_lora_sample.json.njk';
  }
  if (process.network) delete process.network.pretrained_lora_path;
  const paired = ['flow_dpo', 'fizgig_image_slider'].includes(canonicalTrainingMode(process.type));
  for (const dataset of process.datasets ?? []) {
    delete dataset.control_path;
    if (!paired) delete dataset.control_path_1;
    delete dataset.control_path_2;
    delete dataset.control_path_3;
    dataset.controls = [];
  }
  const source = canonicalFlowArch(arch) === 'moss_image'
    ? ` For this variant select OpenMOSS-Team/MOSS-Image-${['moss_image:base', 'moss_image:base_edit'].includes(arch) ? 'Base' : 'SFT'} or compatible weights.`
    : arch === 'fibo2' ? ' Select FIBO2 Dev, a Qwen3-VL-4B text encoder and a FLUX.2 VAE. Edit Turbo is inference-only.' : '';
  return { config: next, notice: 'Architecture changed. Select compatible base, text encoder and VAE paths; model-specific overrides, pretrained/auxiliary adapters and edit references were cleared. Prompts, paired targets, anchors, ranks and memory settings were retained.' + source };
}
