import type { JobConfig, DatasetConfig, DiffusionKTOConfig } from '@/types';

/** Mirrored by toolkit/training_capabilities.py; tested against that table. */
export const flowTrainingModels: Record<string, { label: string; bucketDivisibility: number; editReferences: boolean }> = {
  qwen_image_2: { label: 'Qwen Image 2.1', bucketDivisibility: 32, editReferences: true },
  krea2: { label: 'Krea 2', bucketDivisibility: 16, editReferences: true },
  anima: { label: 'Anima', bucketDivisibility: 32, editReferences: false },
  ideogram4: { label: 'Ideogram 4', bucketDivisibility: 16, editReferences: false },
};
export const trainingModeAliases: Record<string, string> = {
  qwen_flow_dpo: 'flow_dpo', qwen_guidance_distillation: 'guidance_distillation',
};
export const canonicalTrainingMode = (mode: string) => trainingModeAliases[mode] ?? mode;
export const specializedTrainingModes = ['fizgig_image_slider', 'fizgig_prompt_slider', 'flow_dpo', 'guidance_distillation', 'sliderspace', 'diffusion_kto'];
export const defaultDiffusionKTOConfig: DiffusionKTOConfig = {
  beta: 1, liked_weight: 1, disliked_weight: 1, reference_estimator: 'batch_mean', score_window_size: 4,
};
export const isSpecializedTrainingMode = (mode: string) => specializedTrainingModes.includes(canonicalTrainingMode(mode));
export const trainingModeSupported = (arch: string, mode: string) => !isSpecializedTrainingMode(mode) || !!flowTrainingModels[arch];
export const trainingModeVisible = (arch: string, mode: string, selected: string) => mode === selected
  || (!trainingModeAliases[mode] && trainingModeSupported(arch, mode));
export const cfgNegativeTextEnabled = (model: { arch: string; model_kwargs?: Record<string, unknown> }) =>
  model.arch !== 'ideogram4' || model.model_kwargs?.ideogram_cfg_reference === 'negative_prompt';

/** Pure pre-load checks used by imports, both save buttons and the API. */
export function validateTrainingCapabilities(config: JobConfig): string[] {
  const process = config?.config?.process?.[0];
  if (!process) return ['Job configuration must contain a training process.'];
  const mode = canonicalTrainingMode(process.type);
  const errors: string[] = [];
  for (const key of ['record_training_examples', 'record_training_rng'] as const) {
    const value = process.logging?.[key];
    if (value !== undefined && typeof value !== 'boolean') errors.push(`Logging ${key} must be a boolean.`);
  }
  if (process.logging?.record_training_rng && process.logging?.record_training_examples === false) errors.push('Recording noise RNG state requires per-image loss logging.');
  if (!isSpecializedTrainingMode(mode)) return errors;
  const model = process.model;
  if (!model || !trainingModeSupported(model.arch, mode)) return [...errors, `${mode} supports Qwen Image 2.1, Krea 2, Anima and Ideogram 4.`];
  const networks = mode.startsWith('fizgig_') ? ['lora', 'dora'] : ['lora'];
  if (!networks.includes(process.network?.type ?? '')) errors.push(`${mode} requires ${networks.length === 1 ? 'LoRA only' : 'LoRA or DoRA'}.`);
  const kwargs = model.model_kwargs ?? {};
  if (process.sample?.comfy?.enabled && ['anima', 'ideogram4'].includes(model.arch)) {
    const workflow = (process.sample.comfy.workflow_path ?? '').replaceAll('\\', '/').split('/').pop() ?? '';
    const known: Record<string, string> = { 'krea2_lora_sample.json.njk': 'krea2', 'krea2_lora_sample_batch_easy_use.json.njk': 'krea2',
      'qwen_image_2_lora_sample.json.njk': 'qwen_image_2', 'qwen_image_2_lora_sample_batch_easy_use.json.njk': 'qwen_image_2',
      'anima_lora_sample.json.njk': 'anima', 'ideogram4_lora_sample.json.njk': 'ideogram4' };
    if (known[workflow] && known[workflow] !== model.arch) errors.push(`Select ${model.arch}_lora_sample.json.njk and compatible model/VAE/text encoder files for Comfy previews.`);
  }
  if (model.arch === 'anima' && kwargs.train_text_conditioner) errors.push('Specialized training requires a frozen Anima text conditioner. Disable train_text_conditioner.');
  if (model.arch === 'ideogram4' && !['image_only', 'negative_prompt'].includes(String(kwargs.ideogram_cfg_reference ?? 'image_only'))) errors.push('Choose image-only or text-negative Ideogram CFG reference.');
  if (model.arch !== 'qwen_image_2' || mode === 'diffusion_kto') {
    const unsupportedPaths = ['unconditional_lora_path',
      ...(model.arch !== 'qwen_image_2' ? ['assistant_lora_path'] : [])] as const;
    const configuredPaths = unsupportedPaths.filter(key => model[key as keyof typeof model]);
    if (configuredPaths.length) errors.push(`Unsupported auxiliary LoRA paths: ${configuredPaths.map(key => `model.${key}`).join(', ')}. Frozen Qwen training helpers and preview-only inference LoRAs are supported.`);
    if (kwargs.is_distilled || /(?:^|[\W_])(?:turbo|lightning|distilled)(?:$|[\W_])/i.test(model.name_or_path ?? '')) errors.push('Specialized training currently requires a base checkpoint, not a turbo/step-distilled variant.');
    if (model.arch === 'krea2' && (mode.startsWith('fizgig_') || mode === 'sliderspace') && (kwargs.edit || kwargs.kv_cache)) errors.push('Fizgig and SliderSpace are text-to-image objectives. Disable Krea edit and kv_cache.');
  }
  const hasEditReferences = flowTrainingModels[model.arch].editReferences && (model.arch !== 'krea2' || !!kwargs.edit);
  if (process.datasets != null && !Array.isArray(process.datasets)) errors.push('Training datasets must be a list.');
  for (const dataset of Array.isArray(process.datasets) ? process.datasets : []) {
    if (!dataset || typeof dataset !== 'object') { errors.push('Each dataset must be a configuration object.'); continue; }
    const paired = mode === 'flow_dpo' || mode === 'fizgig_image_slider';
    const references = paired ? [dataset.control_path_2, dataset.control_path_3] :
      [dataset.control_path, dataset.control_path_1, dataset.control_path_2, dataset.control_path_3];
    if (references.some(Boolean) && (mode.startsWith('fizgig_') || !hasEditReferences)) errors.push('This mode/model does not support the configured edit-source images. Control Dataset 1 is a paired target in image sliders/DPO, not an edit reference.');
  }
  if (mode === 'guidance_distillation' && model.arch === 'ideogram4' && process.guidance_distillation?.objective === 'negative_only'
    && (!cfgNegativeTextEnabled(model) || !process.guidance_distillation.negative_prompt?.trim())) errors.push('Ideogram negative-only distillation requires text-negative CFG and a nonempty teacher negative prompt.');
  if (mode === 'diffusion_kto') errors.push(...validateDiffusionKTO(config));
  return [...new Set(errors)];
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
  if (network.pretrained_lora_path || (process.model.arch !== 'qwen_image_2' && process.model.assistant_lora_path) || process.model.unconditional_lora_path) errors.push('KTO supports frozen Qwen training helpers and preview-only inference LoRAs, but no pretrained or other auxiliary LoRAs.');
  const networkOptions = network.network_kwargs ?? {};
  if (typeof networkOptions !== 'object' || Array.isArray(networkOptions)) errors.push('KTO network_kwargs must be an object.');
  if ([network, networkOptions].some(values => ['dropout', 'rank_dropout', 'module_dropout'].some(key => values[key] != null && values[key] !== 0))) errors.push('KTO requires zero adapter dropout for exact replay.');
  if (network.all_layers || ['full_train_in_out', 'full_if_contains', 'is_ara'].some(key => networkOptions[key])) errors.push('KTO requires ordinary LoRA-only parameters, not full-layer adapters.');
  if (process.model.model_kwargs?.edit || process.model.model_kwargs?.kv_cache) errors.push('KTO initially supports text-to-image only; disable edit/kv_cache.');
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
  if (!isSpecializedTrainingMode(process.type) || !flowTrainingModels[arch]) throw new Error('Unsupported specialized training architecture.');
  process.model.arch = arch;
  process.model.name_or_path = '';
  delete process.model.text_encoder_path;
  delete process.model.vae_path;
  delete process.model.assistant_lora_path;
  delete process.model.unconditional_lora_path;
  process.model.model_kwargs = {};
  if (process.network) delete process.network.pretrained_lora_path;
  const paired = ['flow_dpo', 'fizgig_image_slider'].includes(canonicalTrainingMode(process.type));
  for (const dataset of process.datasets ?? []) {
    delete dataset.control_path;
    if (!paired) delete dataset.control_path_1;
    delete dataset.control_path_2;
    delete dataset.control_path_3;
    dataset.controls = [];
  }
  return { config: next, notice: 'Architecture changed. Select compatible base, text encoder and VAE paths; model-specific overrides, pretrained/auxiliary adapters and edit references were cleared. Prompts, paired targets, anchors, ranks and memory settings were retained.' };
}
