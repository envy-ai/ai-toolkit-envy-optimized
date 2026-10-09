import type { JobConfig } from '@/types';
import { validateTrainingCapabilities } from './trainingCapabilities';

/** Zero is rendered separately; retain the order of unique nonzero strengths. */
export function parseSliderSpaceAutoStrengths(value: unknown): number[] | null {
  if (typeof value !== 'string') return null;
  const strengths: number[] = [];
  for (const token of value.split(',').map(item => item.trim())) {
    if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(token)) return null;
    const strength = Number(token);
    if (!Number.isFinite(strength)) return null;
    if (strength !== 0 && !strengths.includes(strength)) strengths.push(strength);
  }
  return strengths;
}

/** Used by the top-bar save action as well as the form; native constraints alone are insufficient. */
export function validateSliderSpace(config: JobConfig): string[] {
  const process = config?.config?.process?.[0];
  if (!process) return ['Job configuration must contain a training process.'];
  if (process?.type !== 'sliderspace') return [];
  return validateSliderSpaceSettings(config);
}

/** Matching the backend contract, including imported and advanced YAML configurations. */
export function validateSliderSpaceSettings(config: JobConfig): string[] {
  const process = config?.config?.process?.[0];
  if (process?.type !== 'sliderspace') return [];
  const value = process.sliderspace;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ['SliderSpace settings must be an object.'];
  if (!process.train || typeof process.train !== 'object' || Array.isArray(process.train)
      || !process.model || typeof process.model !== 'object' || Array.isArray(process.model)
      || !process.save || typeof process.save !== 'object' || Array.isArray(process.save)) {
    return ['SliderSpace requires model, train, and save configuration objects. Use Advanced view to correct the imported configuration.'];
  }
  const errors: string[] = [];
  const knownSettings = ['discovery_mode', 'discovery_datasets', 'discovery_buckets', 'concept_prompts', 'num_directions', 'discovery_samples', 'resolution', 'discovery_steps',
    'cfg_scale', 'negative_prompt', 'seed', 'feature_encoder', 'feature_device', 'loss_weight', 'preview_direction', 'preview_strength', 'preview_auto', 'preview_auto_strengths'];
  if (Object.keys(value).some(key => !knownSettings.includes(key))) errors.push('Remove unknown SliderSpace settings; check the supported configuration fields.');
  const integer = (value: number, min: number, max = Number.MAX_SAFE_INTEGER) =>
    Number.isInteger(value) && value >= min && value <= max;
  const mode = value.discovery_mode ?? 'generated';
  const generates = mode !== 'provided';
  const provided = mode !== 'generated';
  if (value.discovery_buckets !== undefined && typeof value.discovery_buckets !== 'boolean') errors.push('Discovery bucketing must be enabled or disabled.');
  if (!['provided', 'generated', 'both'].includes(mode)) errors.push('Choose provided, generated, or both for discovery.');
  const promptsValid = Array.isArray(value.concept_prompts)
    && value.concept_prompts.every(prompt => typeof prompt === 'string' && (!generates || prompt.trim().length > 0))
    && (!generates || value.concept_prompts.length > 0);
  if (!promptsValid) errors.push('Enter at least one concept prompt and fill or remove each empty prompt.');
  if (!integer(value.num_directions, 1, 64)) errors.push('Directions to train must be a whole number from 1 to 64.');
  const minimumImages = !generates ? 0 : Math.max(mode === 'generated' && integer(value.num_directions, 1, 64) ? value.num_directions + 1 : 1,
    Array.isArray(value.concept_prompts) ? value.concept_prompts.length : 1);
  if (!integer(value.discovery_samples, minimumImages)) {
    errors.push(`Generated discovery images must be a whole number of at least ${minimumImages}${mode === 'generated' ? ' (more than the directions, and at least one per prompt)' : ''}.`);
  }
  const folders = value.discovery_datasets ?? [];
  if (!Array.isArray(folders)) errors.push('Discovery image folders must be a list.');
  else {
    if (provided && !folders.length) errors.push('Add at least one discovery image folder.');
    folders.forEach((folder, index) => {
      if (!folder || typeof folder !== 'object' || Array.isArray(folder)
          || Object.keys(folder).some(key => !['folder_path', 'default_caption'].includes(key))
          || typeof folder.folder_path !== 'string' || (provided && !folder.folder_path.trim())
          || (folder.default_caption !== undefined && typeof folder.default_caption !== 'string')) {
        errors.push(`Discovery image folder ${index + 1} needs a folder path and optional text default caption.`);
      }
    });
  }
  if (!integer(value.resolution, 128) || value.resolution % 32 !== 0) {
    errors.push('Discovery resolution must be at least 128 pixels and a multiple of 32.');
  }
  if (!integer(value.discovery_steps, 1)) errors.push('Discovery generation steps must be a positive whole number.');
  if (!Number.isFinite(value.cfg_scale) || value.cfg_scale < 1) errors.push('Discovery / training CFG must be at least 1.');
  if (!integer(value.seed, 0, 4294967295)) errors.push('Discovery seed must be a whole number from 0 to 4294967295.');
  if (typeof value.negative_prompt !== 'string') errors.push('Discovery negative prompt must be text.');
  if (typeof value.feature_encoder !== 'string' || !value.feature_encoder.trim()) errors.push('Enter a feature model name or path.');
  if (!['cpu', 'cuda'].includes(value.feature_device)) errors.push('Feature model device must be CPU or CUDA.');
  if (!Number.isFinite(value.loss_weight) || value.loss_weight <= 0) errors.push('SliderSpace loss weight must be greater than zero.');
  errors.push(...validateSliderSpacePreview(config));
  errors.push(...validateTrainingCapabilities(config));
  if (process.network?.type !== 'lora') errors.push('SliderSpace requires a LoRA network.');
  if (process.network?.pretrained_lora_path) errors.push('Clear the pretrained LoRA path; SliderSpace starts a separate new LoRA for each direction.');
  if (process.network?.network_kwargs?.full_train_in_out) errors.push('Disable shared input/output layer training for SliderSpace.');
  if (!integer(process.train.steps ?? 1000, value.num_directions)) errors.push('Total training steps must be a whole number with at least one step per direction.');
  if ((process.train.batch_size ?? 1) !== 1 || (process.train.gradient_accumulation ?? 1) !== 1) errors.push('SliderSpace requires batch size 1 and gradient accumulation 1.');
  const train = process.train as unknown as Record<string, unknown>;
  if (train.gradient_accumulation_steps !== undefined && train.gradient_accumulation_steps !== 1) errors.push('SliderSpace requires gradient accumulation steps of 1.');
  if ((process.train.noise_scheduler ?? 'flowmatch') !== 'flowmatch' || (process.train.timestep_type ?? 'shift') !== 'shift') errors.push('SliderSpace requires shifted flow-matching timesteps.');
  if (process.train.train_unet === false || process.train.train_text_encoder || train.train_refiner) errors.push('SliderSpace trains the transformer only; disable text encoder and refiner training.');
  if (process.train.ema_config?.use_ema) errors.push('Disable EMA for SliderSpace.');
  if (train.merge_network_on_save) errors.push('Disable merging the network on save; SliderSpace exports separate direction LoRAs.');
  if ((process.train.loss_type ?? 'mse') !== 'mse' || (process.train.frequency_loss_type ?? 'none') !== 'none'
    || ['do_cfg', 'do_random_cfg', 'diff_output_preservation', 'blank_prompt_preservation', 'do_guidance_loss', 'do_differential_guidance',
      'do_prior_divergence', 'train_turbo', 'do_fft_loss', 'learnable_snr_gos', 'diffusion_feature_extractor_path', 'inverted_mask_prior', 'correct_pred_norm', 'free_u', 'do_paramiter_swapping'].some(key => train[key])) {
    errors.push('SliderSpace uses its own semantic objective. Disable generic loss, preservation, and training guidance options.');
  }
  if (process.train.validation_config) errors.push('Disable dataset validation for SliderSpace; use direction previews instead.');
  if ((process.model.arch !== 'qwen_image_2' && process.model.assistant_lora_path) || process.model.unconditional_lora_path) {
    errors.push('Clear auxiliary model LoRA paths before running SliderSpace.');
  }
  const extra = process as unknown as Record<string, unknown>;
  if (process.trigger_word) errors.push('Clear the trigger word; SliderSpace uses the exact concept prompts.');
  if (extra.adapter || extra.decorator || extra.embedding) errors.push('SliderSpace does not support additional adapter, decorator, or embedding configurations.');
  if (process.datasets != null && (!Array.isArray(process.datasets) || process.datasets.length)) errors.push('Use discovery image folders in the SliderSpace card instead of ordinary training datasets.');
  const minimum = train.min_denoising_steps ?? 0;
  const maximum = train.max_denoising_steps ?? 999;
  if (typeof minimum !== 'number' || typeof maximum !== 'number' || !(0 <= minimum && minimum < maximum && maximum <= 1000)) {
    errors.push('SliderSpace requires 0 ≤ minimum timestep < maximum timestep ≤ 1000.');
  }
  if ((process.save.save_format ?? 'safetensors') !== 'safetensors' || process.save.push_to_hub) errors.push('SliderSpace exports local safetensors; disable automatic Hub upload.');
  return errors;
}

export function validateSliderSpacePreview(config: JobConfig): string[] {
  const process = config?.config?.process?.[0];
  if (process?.type !== 'sliderspace') return [];
  const value = process.sliderspace;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return ['SliderSpace settings must be an object.'];
  const errors: string[] = [];
  if (value.preview_auto !== undefined && typeof value.preview_auto !== 'boolean') errors.push('Auto sampling must be enabled or disabled.');
  if (parseSliderSpaceAutoStrengths(value.preview_auto_strengths === undefined ? '-1, 1' : value.preview_auto_strengths) === null) {
    errors.push('Auto sample strengths must be finite numbers separated by commas.');
  }
  if (value.preview_auto) {
    const samples = process.sample?.samples;
    if (!Array.isArray(samples) || samples.length !== 1 || typeof samples[0]?.prompt !== 'string' || !samples[0].prompt.trim()) {
      errors.push('Auto sampling requires one nonempty sample prompt.');
    }
  }
  if (!Number.isInteger(value.preview_direction) || value.preview_direction < 1 || value.preview_direction > value.num_directions) {
    errors.push('Choose a preview direction within the number of directions to train.');
  }
  if (!Number.isFinite(value.preview_strength)) errors.push('Preview strength must be a finite number.');
  return errors;
}

export function sliderSpaceSampleLabel(path: string): string | null {
  const match = path.split(/[\\/]/).pop()?.match(/_direction_(\d+)_strength_(plus|minus)(\d+(?:\.\d+)?(?:e[+-]?\d+)?)_/i);
  if (!match) return null;
  if (Number(match[1]) === 0) return 'Base model · strength 0';
  const strength = Number(match[3]) * (match[2].toLowerCase() === 'minus' ? -1 : 1);
  return `Direction ${Number(match[1])} · strength ${strength > 0 ? '+' : ''}${strength}`;
}

export function updateSliderSpaceConcepts(config: JobConfig, prompts: string[]): JobConfig {
  const next = structuredClone(config);
  const process = next.config.process[0];
  if (!process.sliderspace) return next;
  const previous = Array.isArray(process.sliderspace.concept_prompts) ? process.sliderspace.concept_prompts[0] ?? '' : '';
  process.sliderspace.concept_prompts = prompts;
  const sample = process.sample.samples?.[0];
  if (sample && sample.prompt === previous) sample.prompt = prompts[0] ?? '';
  return next;
}

/** Apply only live preview choices; training/discovery must remain immutable during a run. */
export function mergeSliderSpacePreview(existing: JobConfig, incoming: JobConfig): void {
  const target = existing.config.process[0];
  const source = incoming.config.process[0].sliderspace;
  if (target.type !== 'sliderspace' || !target.sliderspace || !source) return;
  if (!Number.isInteger(source.preview_direction) || source.preview_direction < 1
      || source.preview_direction > target.sliderspace.num_directions || !Number.isFinite(source.preview_strength)) {
    throw new Error('Choose a trained SliderSpace direction and a finite preview strength.');
  }
  if (source.preview_auto !== undefined && typeof source.preview_auto !== 'boolean') {
    throw new Error('Auto sampling must be enabled or disabled.');
  }
  const strengths = source.preview_auto_strengths === undefined ? '-1, 1' : source.preview_auto_strengths;
  if (parseSliderSpaceAutoStrengths(strengths) === null) {
    throw new Error('Auto sample strengths must be finite numbers separated by commas.');
  }
  target.sliderspace.preview_direction = source.preview_direction;
  target.sliderspace.preview_strength = source.preview_strength;
  target.sliderspace.preview_auto = source.preview_auto ?? false;
  target.sliderspace.preview_auto_strengths = strengths;
}

export function sampleFilenameInfo(path: string) {
  const filename = path.split(/[\\/]/).pop() ?? '';
  const match = filename.match(/^\d+_+(\d+)(?:_auto)?(?:_direction_\d+_strength_(?:plus|minus)\d+(?:\.\d+)?(?:e[+-]?\d+)?)?_(\d+)\.[^.]+$/i);
  return { filename, step: match ? Number(match[1]) : 0, promptIdx: match ? Number(match[2]) : 0 };
}

/** Preserve preview iterations even after prompt-count edits or an interrupted render. */
export function groupSliderSpaceSamples(paths: string[]): string[][] {
  const groups: string[][] = [];
  let previousIdentity = '';
  let previousIndex = -1;
  for (const path of paths) {
    const { filename, promptIdx } = sampleFilenameInfo(path);
    // Generation timestamps belong to individual images, not the whole iteration.
    let identity = filename.replace(/^\d+_+/, '').replace(/_\d+\.[^.]+$/, '');
    if (identity.includes('_auto_direction_')) identity = identity.replace(/_direction_.*$/, '');
    if (!groups.length || identity !== previousIdentity || promptIdx <= previousIndex) groups.push([]);
    groups[groups.length - 1].push(path);
    previousIdentity = identity;
    previousIndex = promptIdx;
  }
  return groups;
}

export function adjacentSliderSpaceSample(rows: string[][], path: string, rowDelta: number, columnDelta: number): string | null {
  const rowIndex = rows.findIndex(row => row.includes(path));
  if (rowIndex < 0) return null;
  const target = rows[rowIndex + rowDelta];
  if (!target) return null;
  if (columnDelta) return target[target.indexOf(path) + columnDelta] ?? null;
  const promptIdx = sampleFilenameInfo(path).promptIdx;
  return target.find(candidate => sampleFilenameInfo(candidate).promptIdx === promptIdx) ?? null;
}
