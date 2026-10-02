/**
 * GPU API response
 */

export interface GpuUtilization {
  gpu: number;
  memory: number;
}

export interface GpuMemory {
  total: number;
  free: number;
  used: number;
  // unified memory (e.g. GB10): figures are the system RAM pool the GPU shares
  shared?: boolean;
}

export interface GpuPower {
  draw: number;
  limit: number;
}

export interface GpuClocks {
  graphics: number;
  memory: number;
}

export interface GpuFan {
  speed: number;
}

export interface GpuInfo {
  index: number;
  name: string;
  driverVersion: string;
  temperature: number;
  utilization: GpuUtilization;
  memory: GpuMemory;
  power: GpuPower;
  clocks: GpuClocks;
  fan: GpuFan;
}

export interface CpuInfo {
  name: string;
  cores: number;
  temperature: number;
  totalMemory: number;
  freeMemory: number;
  availableMemory: number;
  currentLoad: number;
}

export interface GPUApiResponse {
  hasNvidiaSmi: boolean;
  isMac: boolean;
  gpus: GpuInfo[];
  error?: string;
}

/**
 * System monitor stream (SSE at /api/monitor)
 */

// Rolling history only logs load + memory; everything else (temps, fans,
// power, clocks) is instantaneous-only via MonitorSample.
export interface MonitorHistoryPoint {
  t: number; // epoch ms
  cpu: { load: number; memUsedMb: number };
  // one entry per GPU, same order as MonitorSample.gpu.gpus (sorted by index)
  gpus: { load: number; memUsedMb: number }[];
}

export interface MonitorSample {
  t: number;
  cpu: CpuInfo | null;
  gpu: GPUApiResponse;
}

export interface MonitorInit extends MonitorSample {
  history: MonitorHistoryPoint[];
}

/**
 * Training configuration
 */

export interface LayerLrMultiplierConfig {
  match?: string | string[];
  contains?: string | string[];
  pattern?: string | string[];
  multiplier: number;
  regex?: boolean;
}

export interface NetworkConfig {
  type: string;
  pretrained_lora_path?: string;
  linear: number;
  linear_alpha: number;
  conv: number;
  conv_alpha: number;
  lokr_full_rank: boolean;
  lokr_factor: number;
  save_magnitude_less_lora?: boolean;
  network_kwargs: {
    ignore_if_contains?: string[];
    only_if_contains?: string[];
    layer_lr_multipliers?: LayerLrMultiplierConfig[] | Record<string, number>;
    lr_multipliers?: LayerLrMultiplierConfig[] | Record<string, number>;
    block_lr_multipliers?: LayerLrMultiplierConfig[] | Record<string, number>;
    [key: string]: any;
  };
  transformer_only?: boolean;
}

export interface SaveConfig {
  dtype: string;
  save_every: number;
  max_step_saves_to_keep: number;
  record_low_window_size?: number;
  record_low_start_step?: number;
  sample_on_record_low?: boolean;
  max_record_low_saves_to_keep?: number;
  save_format: string;
  push_to_hub: boolean;
}

export interface DatasetConfig {
  batch_size?: number;
  folder_path: string;
  mask_path: string | null;
  mask_min_value: number;
  default_caption: string;
  caption_ext: string;
  caption_dropout_rate: number;
  shuffle_tokens?: boolean;
  is_reg: boolean;
  network_weight: number;
  cache_latents_to_disk?: boolean;
  resolution: number[];
  controls: string[];
  control_path?: string | null;
  num_frames: number;
  shrink_video_to_frames: boolean;
  do_i2v?: boolean;
  do_audio?: boolean;
  audio_normalize?: boolean;
  audio_preserve_pitch?: boolean;
  fps?: number;
  flip_x: boolean;
  flip_y: boolean;
  num_repeats?: number;
  control_path_1?: string | null;
  control_path_2?: string | null;
  control_path_3?: string | null;
  auto_frame_count?: boolean;
}

export interface EMAConfig {
  use_ema: boolean;
  ema_decay: number;
}

export interface ValidationItem {
  image_path: string;
  prompt: string;
}

export interface ValidationConfig {
  validation_items: ValidationItem[];
  resolution: number;
  validate_every_n_steps: number;
  validation_sigmas?: number[];
}

export interface TrainConfig {
  batch_size: number;
  bypass_guidance_embedding?: boolean;
  steps: number;
  gradient_accumulation: number;
  train_unet: boolean;
  train_text_encoder: boolean;
  do_cfg?: boolean;
  do_random_cfg?: boolean;
  gradient_checkpointing: boolean;
  noise_scheduler: string;
  timestep_type: string;
  content_or_style: string;
  optimizer: string;
  lr: number;
  ema_config?: EMAConfig;
  dtype: string;
  unload_text_encoder: boolean;
  cache_text_embeddings: boolean;
  optimizer_params: {
    weight_decay: number;
    d0?: number;
    d_coef?: number;
  };
  skip_first_sample: boolean;
  force_first_sample: boolean;
  disable_sampling: boolean;
  diff_output_preservation: boolean;
  diff_output_preservation_multiplier: number;
  diff_output_preservation_class: string;
  blank_prompt_preservation?: boolean;
  blank_prompt_preservation_multiplier?: number;
  switch_boundary_every: number;
  loss_type: 'mse' | 'mae' | 'wavelet' | 'stepped';
  frequency_loss_type?: 'none' | 'low_pass' | 'high_pass' | 'band_pass' | 'notch';
  frequency_loss_weight?: number;
  frequency_loss_cutoff?: number;
  frequency_loss_min_period?: number;
  frequency_loss_max_period?: number;
  frequency_loss_transition?: number;
  frequency_loss_patch_size?: number;
  frequency_loss_activation_offload?: boolean;
  min_snr_gamma?: number;
  max_denoising_steps?: number;
  do_differential_guidance?: boolean;
  differential_guidance_scale?: number;
  audio_loss_multiplier?: number;
  max_loss?: number | null;
  validation_config?: ValidationConfig;
  do_guidance_loss?: boolean;
  guidance_loss_target?: number;
}

export interface QuantizeKwargsConfig {
  exclude: string[];
}

export interface ModelConfig {
  name_or_path: string;
  quantize: boolean;
  quantize_te: boolean;
  qtype: string;
  qtype_te: string;
  quantize_kwargs?: QuantizeKwargsConfig;
  arch: string;
  low_vram: boolean;
  low_vram_layer_streaming?: boolean;
  model_kwargs: { [key: string]: any };
  layer_offloading?: boolean;
  layer_offloading_transformer_percent?: number;
  layer_offloading_text_encoder_percent?: number;
  assistant_lora_path?: string;
  text_encoder_path?: string;
  inference_lora_path?: string;
  unconditional_lora_path?: string;
  compile?: boolean;
  block_compile?: boolean;
  compile_mode?: 'default' | 'max-autotune' | 'fastest';
  compile_fullgraph?: boolean;
  compile_dynamic?: boolean;
  cache_size_limit?: number;
}

export interface SampleItem {
  prompt: string;
  width?: number;
  height?: number;
  neg?: string;
  seed?: number;
  guidance_scale?: number;
  sample_steps?: number;
  fps?: number;
  num_frames?: number;
  duration?: number;
  ctrl_img?: string | null;
  ctrl_idx?: number;
  network_multiplier?: number;
  ctrl_img_1?: string | null;
  ctrl_img_2?: string | null;
  ctrl_img_3?: string | null;
}

export interface ComfySampleConfig {
  enabled: boolean;
  api_url: string;
  negative_prompt: string;
  workflow_path: string;
  model: string;
  vae: string;
  audio_vae: string;
  text_encoder: string;
  sampler: string;
  scheduler: string;
  inference_lora: string;
  inference_lora_strength: number;
  send_prompts_as_batch: boolean;
  run_in_background: boolean;
  training_lora_path_replace_from: string;
  training_lora_path_replace_to: string;
  output_format: string;
  output_quality: string;
}

export interface SampleConfig {
  sampler: string;
  sample_every: number;
  sample_start_step: number;
  width: number;
  height: number;
  prompts?: string[];
  samples: SampleItem[];
  neg: string;
  seed: number;
  walk_seed: boolean;
  guidance_scale: number;
  sample_steps: number;
  num_frames: number;
  fps: number;
  comfy: ComfySampleConfig;
  duration?: number;
}

export interface LoggingConfig {
  log_every: number;
  use_ui_logger: boolean;
}

export interface SliderTargetConfig {
  target_class: string;
  positive: string;
  negative: string;
  weight?: number;
  shuffle?: boolean;
}

export interface SliderConfig {
  guidance_strength?: number;
  anchor_strength?: number;
  positive_prompt?: string;
  negative_prompt?: string;
  target_class?: string;
  anchor_class?: string | null;
  resolutions?: number[][];
  targets?: SliderTargetConfig[];
  prompt_file?: string | null;
  prompt_tensors?: string | null;
  batch_full_slide?: boolean;
}

export interface FizgigPromptTriplet {
  neutral_prompt: string;
  positive_prompt: string;
  negative_prompt: string;
  cfg_negative_prompt?: string;
  cfg_negative_prompt_positive?: string;
  cfg_negative_prompt_negative?: string;
}

export type FizgigPromptEntry =
  | { kind: 'simple'; prompt: string }
  | ({ kind: 'specific' } & FizgigPromptTriplet);

export interface FizgigSliderConfig {
  diff_weight?: number;
  positive_prefix?: string;
  negative_prefix?: string;
  cfg_scale?: number;
  cfg_negative_prefix?: string;
  cfg_negative_prefix_positive?: string;
  cfg_negative_prefix_negative?: string;
  cfg_negative_prompt?: string;
  cfg_negative_prompt_positive?: string;
  cfg_negative_prompt_negative?: string;
  prompt_entries?: FizgigPromptEntry[];
  // Backward compatibility with earlier prompt-slider jobs.
  prompt_triplets?: FizgigPromptTriplet[];
  // Existing single-triplet jobs use these fields until edited in the form.
  neutral_prompt?: string;
  positive_prompt?: string;
  negative_prompt?: string;
  guidance?: number;
  bank_size?: number;
  bank_resolution?: number;
  bank_steps?: number;
}

export interface FlowDPOConfig {
  beta: number;
  sft_weight: number;
}

export interface ProcessConfig {
  type: string;
  sqlite_db_path?: string;
  training_folder: string;
  performance_log_every: number;
  trigger_word: string | null;
  device: string;
  network?: NetworkConfig;
  slider?: SliderConfig;
  fizgig_slider?: FizgigSliderConfig;
  flow_dpo?: FlowDPOConfig;
  save: SaveConfig;
  datasets: DatasetConfig[];
  train: TrainConfig;
  logging: LoggingConfig;
  model: ModelConfig;
  sample: SampleConfig;
}

export interface ConfigObject {
  name: string;
  process: ProcessConfig[];
}

export interface MetaConfig {
  name: string;
  version: string;
}

export interface JobConfig {
  job: string;
  config: ConfigObject;
  meta: MetaConfig;
}

// A LoRA published on the hub, offered for a specific model option. `path` is a
// 'org/repo/path_to/file.safetensors' reference; the backend looks for it under
// the models folder first and downloads it into MODELS_PATH/loras if missing.
export interface CloudLora {
  path: string;
  name: string;
  description?: string;
}

export interface CaptionLora {
  path: string;
  name: string;
  strength: number;
}

export interface CaptionProcessConfig {
  type: string;
  sqlite_db_path?: string;
  device: string;
  caption: {
    model_name_or_path: string;
    model_name_or_path2?: string;
    dtype: string;
    quantize: boolean;
    qtype: string;
    low_vram: boolean;
    extensions: string[];
    path_to_caption: string;
    recaption: boolean;
    compile?: boolean;
    caption_prompt?: string;
    max_res?: number;
    max_new_tokens?: number;
    fixed_caption?: string;
    caption_format?: string;
    extract_vocals_before_transcribe?: boolean;
    keep_timestamps?: boolean;
    caption_extension?: string;
    thinking?: boolean;
    batch_size?: number;
    layer_offloading?: boolean;
    layer_offloading_percent?: number;
    loras?: CaptionLora[];
  }
}

export interface CaptionConfigObject {
  name: string;
  process: CaptionProcessConfig[];
}

export interface CaptionJobConfig {
  job: string;
  config: CaptionConfigObject;
}

export interface ConfigDoc {
  title: string | React.ReactNode;
  description: React.ReactNode;
}

export interface SelectOption {
  readonly value: string;
  readonly label: string;
}
export interface GroupedSelectOption {
  readonly label: string;
  readonly options: SelectOption[];
}

export type JobStatus = 'queued' | 'running' | 'stopping' | 'stopped' | 'completed' | 'error';
