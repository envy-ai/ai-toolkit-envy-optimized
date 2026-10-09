'use client';
import { isMac } from '@/helpers/basic';
import { defaultComfySampleConfig, defaultSampleConfig } from '@/helpers/defaultSamples';
import { JobConfig, SampleConfig, DatasetConfig, SliderConfig, SliderSpaceConfig } from '@/types';
import { defaultDiffusionKTOConfig } from './trainingCapabilities';

export const defaultSliderSpaceConfig: SliderSpaceConfig = {
  discovery_mode: 'generated',
  discovery_datasets: [],
  discovery_buckets: true,
  concept_prompts: [''],
  num_directions: 4,
  discovery_samples: 128,
  resolution: 512,
  discovery_steps: 20,
  cfg_scale: 4,
  negative_prompt: '',
  seed: 42,
  feature_encoder: 'openai/clip-vit-base-patch32',
  feature_device: 'cpu',
  loss_weight: 1,
  preview_direction: 1,
  preview_strength: 1,
  preview_auto: false,
  preview_auto_strengths: '-1, 1',
};

export const activateSliderSpace = (config: JobConfig): JobConfig => {
  const process = config.config.process[0];
  process.sliderspace = { ...defaultSliderSpaceConfig, concept_prompts: [''], discovery_datasets: [] };
  process.datasets = [];
  process.trigger_word = null;
  process.save.save_format = 'safetensors';
  process.save.push_to_hub = false;
  process.network = {
    ...process.network,
    type: 'lora', linear: 4, linear_alpha: 4, conv: 0, conv_alpha: 0,
    lokr_full_rank: true, lokr_factor: -1,
    network_kwargs: { ...process.network?.network_kwargs },
  };
  delete process.network.pretrained_lora_path;
  delete process.network.network_kwargs.full_train_in_out;
  if (process.model.arch !== 'qwen_image_2') delete process.model.assistant_lora_path;
  delete process.model.unconditional_lora_path;
  Object.assign(process.train, {
    batch_size: 1, gradient_accumulation: 1, steps: 1000,
    train_unet: true, train_text_encoder: false,
    noise_scheduler: 'flowmatch', timestep_type: 'shift', content_or_style: 'balanced',
    cache_text_embeddings: false, unload_text_encoder: true,
    loss_type: 'mse', frequency_loss_type: 'none',
    diff_output_preservation: false, blank_prompt_preservation: false,
    do_cfg: false, do_random_cfg: false, do_guidance_loss: false, do_differential_guidance: false,
    train_refiner: false, merge_network_on_save: false, lr: 0.0001,
  });
  if (process.train.ema_config) process.train.ema_config.use_ema = false;
  delete process.train.validation_config;
  const extraTrain = process.train as unknown as Record<string, unknown>;
  for (const key of ['gradient_accumulation_steps', 'do_prior_divergence', 'train_turbo', 'do_fft_loss',
    'learnable_snr_gos', 'diffusion_feature_extractor_path', 'inverted_mask_prior', 'correct_pred_norm', 'free_u', 'do_paramiter_swapping']) {
    delete extraTrain[key];
  }
  const extraProcess = process as unknown as Record<string, unknown>;
  for (const key of ['adapter', 'decorator', 'embedding']) delete extraProcess[key];
  if (!process.sample.samples?.length || JSON.stringify(process.sample.samples) === JSON.stringify(defaultSampleConfig.samples)) {
    process.sample.samples = [{ prompt: '' }];
  }
  process.sample.walk_seed = false;
  return config;
};

export const defaultDatasetConfig: DatasetConfig = {
  folder_path: '/path/to/images/folder',
  mask_path: null,
  mask_min_value: 0.1,
  default_caption: '',
  caption_ext: 'txt',
  caption_dropout_rate: 0.05,
  cache_latents_to_disk: false,
  is_reg: false,
  network_weight: 1,
  resolution: [512, 768, 1024],
  controls: [],
  shrink_video_to_frames: true,
  num_frames: 1,
  flip_x: false,
  flip_y: false,
  num_repeats: 1,
};

export const defaultSliderConfig: SliderConfig = {
  guidance_strength: 3.0,
  anchor_strength: 1.0,
  positive_prompt: 'person who is happy',
  negative_prompt: 'person who is sad',
  target_class: 'person',
  anchor_class: '',
  resolutions: [[1024, 1024]],
  batch_full_slide: false,
  targets: [
    {
      target_class: 'person',
      positive: 'person who is happy',
      negative: 'person who is sad',
      weight: 1.0,
      shuffle: false,
    },
  ],
};

export const defaultCompileOptions = {
  block_compile: true,
};

export const defaultJobConfig: JobConfig = {
  job: 'extension',
  config: {
    name: 'my_first_lora_v1',
    process: [
      {
        type: 'diffusion_trainer',
        training_folder: 'output',
        sqlite_db_path: './aitk_db.db',
        device: 'cuda',
        trigger_word: null,
        performance_log_every: 10,
        network: {
          type: 'lora',
          linear: 32,
          linear_alpha: 32,
          conv: 16,
          conv_alpha: 16,
          lokr_full_rank: true,
          lokr_factor: -1,
          save_magnitude_less_lora: false,
          loha_dora: false,
          network_kwargs: {
            ignore_if_contains: [],
          },
        },
        save: {
          dtype: 'bf16',
          save_every: 250,
          max_step_saves_to_keep: 4,
          record_low_enabled: false,
          record_low_window_size: 3000,
          record_low_start_step: 50,
          sample_on_record_low: true,
          max_record_low_saves_to_keep: 5,
          save_format: 'diffusers',
          push_to_hub: false,
        },
        datasets: [defaultDatasetConfig],
        train: {
          batch_size: 1,
          bypass_guidance_embedding: true,
          steps: 3000,
          gradient_accumulation: 1,
          train_unet: true,
          train_text_encoder: false,
          gradient_checkpointing: true,
          noise_scheduler: 'flowmatch',
          optimizer: 'adamw8bit',
          timestep_type: 'sigmoid',
          content_or_style: 'balanced',
          optimizer_params: {
            weight_decay: 1e-4,
          },
          min_snr_gamma: 5.0,
          unload_text_encoder: false,
          cache_text_embeddings: false,
          lr: 0.0001,
          ema_config: {
            use_ema: false,
            ema_decay: 0.99,
          },
          skip_first_sample: false,
          force_first_sample: false,
          disable_sampling: false,
          dtype: 'bf16',
          diff_output_preservation: false,
          diff_output_preservation_multiplier: 1.0,
          diff_output_preservation_class: 'person',
          switch_boundary_every: 1,
          loss_type: 'mse',
          frequency_loss_type: 'none',
          frequency_loss_weight: 0.1,
          frequency_loss_cutoff: 18,
          frequency_loss_min_period: 14,
          frequency_loss_max_period: 28,
          frequency_loss_transition: 4,
          frequency_loss_patch_size: 384,
          frequency_loss_activation_offload: true,
        },
        logging: {
          log_every: 1,
          use_ui_logger: true,
          record_training_examples: true,
          record_training_rng: false,
        },
        model: {
          name_or_path: 'ostris/Flex.1-alpha',
          quantize: true,
          qtype: 'qfloat8',
          quantize_te: true,
          qtype_te: 'qfloat8',
          arch: 'flex1',
          low_vram: false,
          model_kwargs: {},
          compile: false,
        },
        sample: defaultSampleConfig,
      },
    ],
  },
  meta: {
    name: '[name]',
    version: '1.0',
  },
};

export const migrateJobConfig = (jobConfig: JobConfig): JobConfig => {
  if (jobConfig.config.process[0]?.type === 'sliderspace') {
    const process = jobConfig.config.process[0];
    if ((process.sliderspace !== undefined && (process.sliderspace === null || typeof process.sliderspace !== 'object' || Array.isArray(process.sliderspace)))
      || !process.train || typeof process.train !== 'object' || Array.isArray(process.train)
      || !process.model || typeof process.model !== 'object' || Array.isArray(process.model)
      || !process.save || typeof process.save !== 'object' || Array.isArray(process.save)) return jobConfig;
    // Fill absent settings without overwriting imported choices or hiding invalid values.
    process.sliderspace = {
      ...defaultSliderSpaceConfig,
      concept_prompts: [''],
      discovery_datasets: [],
      // Old configurations used square cropping; do not silently change resume.
      discovery_buckets: false,
      ...process.sliderspace,
    };
    process.datasets ??= [];
  }
  if (['qwen_guidance_distillation', 'guidance_distillation'].includes(jobConfig.config.process[0]?.type)) {
    jobConfig.config.process[0].guidance_distillation = {
      teacher_cfg_scale: 4,
      negative_prompt: '',
      objective: 'full_guidance',
      ...jobConfig.config.process[0].guidance_distillation,
    };
  }
  if (jobConfig.config.process[0]?.type === 'diffusion_kto') {
    const process = jobConfig.config.process[0];
    const settings = process.diffusion_kto;
    // Fill missing objective settings, but never invent feedback labels or
    // silently repair malformed imports; shared validation reports those.
    if (settings === undefined || (settings !== null && typeof settings === 'object' && !Array.isArray(settings))) {
      process.diffusion_kto = { ...defaultDiffusionKTOConfig, ...settings };
    }
  }
  // upgrade prompt strings to samples
  if (
    jobConfig?.config?.process &&
    jobConfig.config.process[0]?.sample &&
    Array.isArray(jobConfig.config.process[0].sample.prompts) &&
    jobConfig.config.process[0].sample.prompts.length > 0
  ) {
    let newSamples = [];
    for (const prompt of jobConfig.config.process[0].sample.prompts) {
      newSamples.push({
        prompt: prompt,
      });
    }
    jobConfig.config.process[0].sample.samples = newSamples;
    delete jobConfig.config.process[0].sample.prompts;
  }

  // upgrade job from ui_trainer to diffusion_trainer
  if (jobConfig?.config?.process && jobConfig.config.process[0]?.type === 'ui_trainer') {
    jobConfig.config.process[0].type = 'diffusion_trainer';
  }

  if ('auto_memory' in jobConfig.config.process[0].model) {
    jobConfig.config.process[0].model.layer_offloading = (jobConfig.config.process[0].model.auto_memory ||
      false) as boolean;
    delete jobConfig.config.process[0].model.auto_memory;
  }

  if (!('logging' in jobConfig.config.process[0])) {
    //@ts-ignore
    jobConfig.config.process[0].logging = {
      log_every: 1,
      use_ui_logger: true,
    };
  }
  jobConfig.config.process[0].logging.record_training_examples ??= true;
  jobConfig.config.process[0].logging.record_training_rng ??= false;
  if (jobConfig.config.process[0].save) {
    jobConfig.config.process[0].save.record_low_enabled ??= false;
  }
  if (isMac()) {
    jobConfig.config.process[0].device = 'mps';
  }

  if (jobConfig.config.process[0]?.train?.optimizer === 'prodigyopt') {
    jobConfig.config.process[0].train.optimizer = 'prodigy';
  }

  if (jobConfig.config.process[0]?.train?.min_snr_gamma === undefined) {
    jobConfig.config.process[0].train.min_snr_gamma = 5.0;
  }

  if (jobConfig.config.process[0]?.slider && !jobConfig.config.process[0].slider?.targets) {
    jobConfig.config.process[0].slider.targets = [
      {
        target_class: jobConfig.config.process[0].slider.target_class ?? '',
        positive: jobConfig.config.process[0].slider.positive_prompt ?? '',
        negative: jobConfig.config.process[0].slider.negative_prompt ?? '',
        weight: 1.0,
        shuffle: false,
      },
    ];
  }

  if (jobConfig.config.process[0]?.sample && jobConfig.config.process[0].sample.comfy === undefined) {
    jobConfig.config.process[0].sample.comfy = { ...defaultComfySampleConfig };
  } else if (jobConfig.config.process[0]?.sample?.comfy) {
    jobConfig.config.process[0].sample.comfy = {
      ...defaultComfySampleConfig,
      ...jobConfig.config.process[0].sample.comfy,
    };
  }

  return jobConfig;
};
