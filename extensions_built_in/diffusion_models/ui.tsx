// UI entries (training form + Generate page) for the models this package
// registers in AI_TOOLKIT_MODELS. Loaded at runtime by the UI, not bundled:
// see ui/src/extensions/README.md for the convention and the allowed imports.
import Link from "next/link";
import type { CustomModelOption, ModelArch } from "@/app/jobs/new/options";
import { effectiveNetworkFilter } from "@/app/jobs/new/options";
import type { JobConfig } from "@/types";
import {
  defaultComfySampleConfig,
  defaultSampleConfig,
  defaultIdeogramSamplesConfig,
  defaultMinimaxH3ImageSampleConfig,
} from "@/helpers/defaultSamples";

const defaultNameOrPath = "";
const defaultLinearRank = 32;
const h3PhotoTrainingAdapter =
  "circlestone-labs/MiniMax-H3-Image-Training-Adapter/minimax_h3_image_training_adapter.safetensors";
const h3VideoTrainingAdapter =
  "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v3.safetensors";
const h3PhotoBlocks = Array.from({ length: 30 }, (_, index) => `blocks.${index + 20}.`);
const h3PhotoIgnoredModules = ["token_refiner", "adaln_proj"];

/** An explicit filter edit replaces its process-level alias atomically. */
function setH3NetworkFilter(config: JobConfig, key: "only_if_contains" | "ignore_if_contains", value: string[] | undefined,
  set: (value: any, key: string) => void) {
  const process = structuredClone(config.config.process[0]);
  if (!process.network) throw new Error("Module filters require an adapter network. Select a Target Type first.");
  delete process[key];
  process.network = { ...process.network, network_kwargs: { ...process.network?.network_kwargs } };
  if (value === undefined) delete process.network.network_kwargs[key];
  else process.network.network_kwargs[key] = value;
  set(process, "config.process[0]");
}

const h3ImageAdvancedModelOptions: CustomModelOption[] = [
  {
    type: "text", label: "DiT path (optional)",
    getValue: config => config.config.process[0].model.model_kwargs?.dit_image_path,
    onChange: (value, _config, set) => set(value.trim() || undefined, "config.process[0].model.model_kwargs.dit_image_path"),
    placeholder: "/absolute/path/to/minimax_h3_image_pruned_int8_convrot.safetensors",
    help: "Leave blank to resolve the official image DiT from Name or Path.",
  },
  {
    type: "text", label: "Text encoder path (optional)",
    getValue: config => {
      const model = config.config.process[0].model;
      return model.te_name_or_path ?? (model.model_kwargs?.text_encoder_path || model.text_encoder_path);
    },
    onChange: (value, config, set) => {
      const model = structuredClone(config.config.process[0].model);
      delete model.te_name_or_path;
      delete model.text_encoder_path;
      if (model.model_kwargs) delete model.model_kwargs.text_encoder_path;
      if (value.trim()) model.te_name_or_path = value.trim();
      set(model, "config.process[0].model");
    },
    placeholder: "/absolute/path/to/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    help: "Shows the active text encoder override. Editing replaces its path aliases; clearing restores automatic resolution.",
  },
  {
    type: "text", label: "VAE path (optional)",
    getValue: config => config.config.process[0].model.vae_path || config.config.process[0].model.model_kwargs?.image_vae_path,
    onChange: (value, config, set) => {
      const model = structuredClone(config.config.process[0].model);
      delete model.vae_path;
      if (model.model_kwargs) delete model.model_kwargs.image_vae_path;
      if (value.trim()) model.vae_path = value.trim();
      set(model, "config.process[0].model");
    },
    placeholder: "/absolute/path/to/minimax_h3_image_vae_fp16.safetensors",
    help: "Leave blank to resolve the official image VAE. Editing replaces an imported image_vae_path alias.",
  },
  {
    type: "checkbox", label: "Cache quantized models",
    getValue: config => config.config.process[0].model.cache_quantized_models ?? true,
    onChange: (value, _config, set) => set(value, "config.process[0].model.cache_quantized_models"),
    help: "Reuse compressed model components between launches. Creating the first cache can involve substantial disk I/O.",
  },
  {
    type: "text", label: "Quantized cache directory (optional)",
    getValue: config => config.config.process[0].model.quantized_model_cache_dir,
    onChange: (value, _config, set) => set(value.trim() || undefined, "config.process[0].model.quantized_model_cache_dir"),
    placeholder: "models/.quantized_training_cache",
    help: "Use a writable directory. Blank uses models/.quantized_training_cache; existing cache symlinks are not replaced.",
  },
  {
    type: "number", label: "Caption token limit", min: 0, step: 1, allowEmpty: true, placeholder: "4096 (model default)",
    getValue: config => config.config.process[0].model.model_kwargs?.max_text_length ?? null,
    onChange: (value, _config, set) => set(value ?? undefined, "config.process[0].model.model_kwargs.max_text_length"),
    help: "0 = unlimited. Blank uses 4096. Reference vision tokens are never truncated. Changing the limit uses a separate image text cache.",
  },
  {
    type: "checkbox", label: "Cache dataset text embeddings to disk",
    getValue: config => {
      const datasets = config.config.process[0].datasets;
      return datasets.length > 0 && datasets.every(dataset => dataset.cache_text_embeddings === true);
    },
    onChange: (value, config, set) => set(config.config.process[0].datasets.map(dataset => ({ ...dataset, cache_text_embeddings: value })),
      "config.process[0].datasets"),
    help: "Applies to every dataset; mixed imported choices stay unchanged until toggled. Separate from Training → Cache Text Embeddings. Newly added datasets inherit a common disk-cache choice.",
  },
];

const h3ImageAdvancedTargetOptions: CustomModelOption[] = [
  {
    type: "multiselect", label: "Include modules", allowCustomValues: true,
    options: Array.from({ length: 50 }, (_, index) => ({ value: `blocks.${index}.`, label: `#${index}` })),
    getValue: config => effectiveNetworkFilter(config.config.process[0], "only_if_contains") ?? [],
    onChange: (value, config, set) => setH3NetworkFilter(config, "only_if_contains", value.length ? value : undefined, set),
    placeholder: "Add module…",
    help: "Numbered chips refer to transformer blocks. The photo recipe accepts any nonempty selection of blocks #0–#49, including scattered blocks. Select chips or type a canonical block filter such as blocks.7. and press Enter. Clearing uses the default blocks 20–49 in photo mode; otherwise it includes all eligible modules. A saved empty array matches no modules and is invalid in photo mode. Choose Image mode (keep current settings) before using non-block module targets.",
  },
  {
    type: "multiselect", label: "Exclude modules", allowCustomValues: true,
    options: h3PhotoIgnoredModules.map(value => ({ value, label: value })),
    getValue: config => effectiveNetworkFilter(config.config.process[0], "ignore_if_contains") ?? [],
    onChange: (value, config, set) => setH3NetworkFilter(config, "ignore_if_contains", value, set),
    placeholder: "Add exclusion…",
    help: "Matches module-name substrings. Empty excludes nothing. The photo recipe accepts no exclusions or both token_refiner and adaln_proj; use Image mode for other filters. Remove a focused chip with Backspace or its × button.",
  },
];

/** An explicit still-photo preset. Loading a job or changing model arch never calls this. */
export function applyMinimaxH3PhotoRecipe(config: JobConfig): JobConfig {
  const next = structuredClone(config);
  const process = next.config.process[0];
  const isOfficialImage = process.model.arch === "minimax_h3_image";
  if (!["minimax_h3", "minimax_h3_image"].includes(process.model.arch) ||
      !["sd_trainer", "ui_trainer", "diffusion_trainer"].includes(process.type) ||
      !process.network) {
    throw new Error("MiniMax-H3 photo training requires an ordinary MiniMax-H3 LoRA job.");
  }
  if (!isOfficialImage && !["fl2va", "fl2va_pruned"].includes(process.model.model_kwargs?.partition ?? "fl2va_pruned")) {
    throw new Error("MiniMax-H3 photo training requires the fl2va or fl2va_pruned base partition; choose that base before applying the recipe.");
  }
  if (isOfficialImage && ![undefined, "image"].includes(process.model.model_kwargs?.partition)) {
    throw new Error("MiniMax-H3-Image requires the image partition; choose the official image base before applying the recipe.");
  }
  process.model.model_kwargs = {
    ...(process.model.model_kwargs ?? {}),
    h3_photo_training: true,
    image_refs_as_video: false,
  };
  process.model.assistant_lora_path = h3PhotoTrainingAdapter;
  Object.assign(process.network, {
    type: "lora",
    linear: 8,
    linear_alpha: 8,
    transformer_only: true,
    network_kwargs: {
      ...(process.network.network_kwargs ?? {}),
      only_if_contains: [...h3PhotoBlocks],
      ignore_if_contains: [...h3PhotoIgnoredModules],
    },
  });
  delete (process.network as unknown as Record<string, unknown>).all_layers;
  delete (process.network as unknown as Record<string, unknown>).full_if_contains;
  delete process.network.network_kwargs.full_if_contains;
  Object.assign(process.train, {
    batch_size: 1,
    gradient_accumulation: 1,
    train_unet: true,
    train_text_encoder: false,
    train_refiner: false,
    noise_scheduler: "flowmatch",
    timestep_type: "shift",
    content_or_style: "balanced",
    content_or_style_reg: "balanced",
    linear_timesteps: false,
    linear_timesteps2: false,
    first_timestep_chance: 0,
    min_denoising_steps: 0,
    max_denoising_steps: 999,
    num_train_timesteps: 1000,
    min_snr_gamma: 0,
    snr_gamma: 0,
    loss_type: "mse",
    loss_target: "noise",
    frequency_loss_type: "none",
    optimizer: "automagic3",
    lr: 1e-6,
    lr_scheduler: "constant",
    cache_text_embeddings: true,
    audio_loss_multiplier: 0,
    do_cfg: false,
    do_random_cfg: false,
    do_guidance_loss: false,
    do_differential_guidance: false,
    diff_output_preservation: false,
    blank_prompt_preservation: false,
    do_prior_divergence: false,
    train_turbo: false,
    learnable_snr_gos: false,
    inverted_mask_prior: false,
    correct_pred_norm: false,
    ema_config: { ...(process.train.ema_config ?? {}), use_ema: true, ema_decay: 0.98 },
  });
  for (const dataset of process.datasets) {
    Object.assign(dataset, {
      batch_size: 1,
      resolution: [512],
      native_resolution: false,
      buckets: true,
      num_frames: 1,
      auto_frame_count: false,
      do_i2v: false,
      do_audio: false,
      cache_latents_to_disk: true,
      cache_text_embeddings: true,
      caption_dropout_rate: 0.05,
      controls: [],
    });
    for (const key of ["control_path", "control_path_1", "control_path_2", "control_path_3", "anchor_path", "multipoint_images",
      "h3_reference_path", "h3_v2v_path", "control_from_same_folder"] as const) {
      delete (dataset as unknown as Record<string, unknown>)[key];
    }
  }
  Object.assign(process.sample, {
    sampler: "flowmatch",
    width: isOfficialImage ? defaultMinimaxH3ImageSampleConfig.width : 512,
    height: isOfficialImage ? defaultMinimaxH3ImageSampleConfig.height : 512,
    num_frames: 1,
    fps: 1,
    guidance_scale: 1,
    sample_steps: isOfficialImage ? defaultMinimaxH3ImageSampleConfig.sample_steps : 20,
  });
  process.sample.samples = process.sample.samples.map(sample => {
    const nextSample = { ...sample };
    for (const key of ["ctrl_img", "ctrl_img_1", "ctrl_img_2", "ctrl_img_3", "ctrl_idx",
      "width", "height", "num_frames", "fps", "duration", "guidance_scale", "sample_steps"] as const) {
      delete nextSample[key];
    }
    return nextSample;
  });
  if (process.sample.comfy) process.sample.comfy.enabled = false;
  return next;
}

/** Report actual saved settings; a modified photo job never claims to match the preset. */
export function getMinimaxH3TrainingRecipe(config: JobConfig): "video" | "photo" | "custom_photo" {
  const process = config.config.process[0];
  if (process.model.model_kwargs?.h3_photo_training !== true) return "video";
  const isOfficialImage = process.model.arch === "minimax_h3_image";
  const network = process.network;
  const train = process.train;
  const sample = process.sample;
  const networkOptions = network?.network_kwargs;
  const exactBlocks = effectiveNetworkFilter(process, "only_if_contains");
  const ignored = effectiveNetworkFilter(process, "ignore_if_contains");
  const trainOptions = train as unknown as Record<string, unknown>;
  const matches =
    ["sd_trainer", "ui_trainer", "diffusion_trainer"].includes(process.type) &&
    (isOfficialImage
      ? [undefined, "image"].includes(process.model.model_kwargs?.partition)
      : ["fl2va", "fl2va_pruned"].includes(process.model.model_kwargs?.partition ?? "fl2va_pruned")) &&
    !process.model.model_kwargs?.image_refs_as_video &&
    network?.type === "lora" && network.linear === 8 && network.linear_alpha === 8 &&
    network.transformer_only === true &&
    !(network as unknown as Record<string, unknown>).all_layers &&
    !(network as unknown as Record<string, unknown>).full_if_contains &&
    !networkOptions?.full_if_contains &&
    Array.isArray(exactBlocks) && exactBlocks.length === h3PhotoBlocks.length &&
    exactBlocks.every((block, index) => block === h3PhotoBlocks[index]) &&
    Array.isArray(ignored) && ignored.length === h3PhotoIgnoredModules.length &&
    ignored.every((module, index) => module === h3PhotoIgnoredModules[index]) &&
    process.model.assistant_lora_path === h3PhotoTrainingAdapter &&
    train.batch_size === 1 && train.gradient_accumulation === 1 &&
    train.train_unet === true && train.train_text_encoder === false &&
    trainOptions.train_refiner === false &&
    train.noise_scheduler === "flowmatch" && train.timestep_type === "shift" &&
    train.content_or_style === "balanced" && trainOptions.content_or_style_reg === "balanced" &&
    train.min_snr_gamma === 0 && trainOptions.snr_gamma === 0 &&
    trainOptions.linear_timesteps === false && trainOptions.linear_timesteps2 === false &&
    trainOptions.first_timestep_chance === 0 && trainOptions.min_denoising_steps === 0 &&
    train.max_denoising_steps === 999 && trainOptions.num_train_timesteps === 1000 &&
    trainOptions.loss_target === "noise" &&
    train.loss_type === "mse" && train.frequency_loss_type === "none" &&
    train.audio_loss_multiplier === 0 &&
    train.optimizer === "automagic3" && train.lr === 1e-6 &&
    trainOptions.lr_scheduler === "constant" &&
    train.cache_text_embeddings === true && train.ema_config?.use_ema === true &&
    train.ema_config.ema_decay === 0.98 &&
    !train.do_cfg && !train.do_random_cfg && !train.do_guidance_loss &&
    !train.do_differential_guidance && !train.diff_output_preservation &&
    !train.blank_prompt_preservation && !trainOptions.do_prior_divergence &&
    !trainOptions.train_turbo && !trainOptions.inverted_mask_prior &&
    !trainOptions.correct_pred_norm && !trainOptions.learnable_snr_gos &&
    sample.sampler === "flowmatch" &&
    sample.width === (isOfficialImage ? defaultMinimaxH3ImageSampleConfig.width : 512) &&
    sample.height === (isOfficialImage ? defaultMinimaxH3ImageSampleConfig.height : 512) &&
    sample.num_frames === 1 && sample.fps === 1 &&
    sample.guidance_scale === 1 &&
    sample.sample_steps === (isOfficialImage ? defaultMinimaxH3ImageSampleConfig.sample_steps : 20) &&
    !sample.comfy?.enabled &&
    sample.samples.every(item => !item.width && !item.height && !item.num_frames &&
      !item.fps && !item.duration && !item.guidance_scale && !item.sample_steps &&
      !item.ctrl_img && !item.ctrl_img_1 && !item.ctrl_img_2 && !item.ctrl_img_3 &&
      item.ctrl_idx == null) &&
    process.datasets.every(dataset =>
      (dataset.batch_size ?? 1) === 1 &&
      dataset.resolution?.length === 1 && dataset.resolution[0] === 512 &&
      (dataset as unknown as Record<string, unknown>).buckets === true &&
      (dataset as unknown as Record<string, unknown>).cache_text_embeddings === true &&
      !dataset.native_resolution && dataset.num_frames === 1 && !dataset.auto_frame_count &&
      !dataset.do_i2v && !dataset.do_audio && dataset.cache_latents_to_disk === true &&
      dataset.caption_dropout_rate === 0.05 && !dataset.controls?.length &&
      !(dataset as unknown as Record<string, unknown>).h3_reference_path &&
      !(dataset as unknown as Record<string, unknown>).h3_v2v_path &&
      !(dataset as unknown as Record<string, unknown>).control_from_same_folder &&
      !dataset.control_path &&
      !dataset.control_path_1 && !dataset.control_path_2 && !dataset.control_path_3);
  return matches ? "photo" : "custom_photo";
}

// MOSS-Image edit LoRAs need the companion custom nodes in ComfyUI
const mossImageEditNotes = (
  <p>
    Using these LoRAs in ComfyUI requires{" "}
    <a
      href="https://github.com/ostris/ComfyUI-MOSS-Image-Ostris-Edit"
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-400 hover:text-blue-300 underline"
    >
      ComfyUI-MOSS-Image-Ostris-Edit
    </a>
    .
  </p>
);

// Qwen-Image-2.1 and its turbo distill share the RGBA VAE, so both entries
// carry this option
const qwenImage21TransparencyOption: CustomModelOption[] = [
  {
    type: "checkbox",
    label: "Transparency (RGBA)",
    getValue: (config: JobConfig) =>
      config?.config?.process?.[0]?.model?.model_kwargs?.rgba ?? false,
    onChange: (
      value: boolean,
      config: JobConfig,
      setJobConfig: (value: any, key: string) => void,
    ) => {
      const kwargs = {
        ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}),
      };
      if (value) {
        kwargs.rgba = true;
      } else {
        delete kwargs.rgba;
      }
      setJobConfig(kwargs, "config.process[0].model.model_kwargs");
    },
    doc: {
      title: "Transparency (RGBA)",
      description: (
        <div className="space-y-2">
          <p>
            This model&apos;s VAE is natively RGBA, so alpha can be carried
            end to end. When on, dataset images and reference images load
            with their alpha channel, the VAE encodes all four channels, and
            samples are saved as PNGs with their transparency intact.
          </p>
          <p>
            Images with no alpha of their own get a fully opaque one, so a
            mixed dataset is fine. Turn this off to train and sample flat
            RGB: alpha is dropped on the way in and added back as opaque.
          </p>
          <p>
            Changing this re-caches latents, and it cannot be combined with
            a dataset using <code>alpha_mask</code>, which consumes the
            alpha channel as a loss mask instead.
          </p>
        </div>
      ),
    },
  },
];

export const AI_TOOLKIT_UI_MODELS: ModelArch[] = [
  ...([['instruct', 2.5], ['base', 5.0]] as const).map(([variant, guidance]): ModelArch => ({
    name: `hunyuan_image_3_${variant}`,
    label: `HunyuanImage 3.0 ${variant === 'instruct' ? 'Instruct' : 'Base'}`,
    group: 'image',
    defaults: {
      'config.process[0].model.name_or_path': [`/path/to/hunyuan_image_3_${variant}_w4a8.safetensors`, defaultNameOrPath],
      'config.process[0].model.extras_name_or_path': [variant === 'instruct' ? 'tencent/HunyuanImage-3.0-Instruct' : 'tencent/HunyuanImage-3.0', undefined],
      'config.process[0].model.vae_path': ['/path/to/hunyuan_image_3_vae_fp16.safetensors', undefined],
      'config.process[0].model.quantize': [true, false],
      'config.process[0].model.quantize_te': [false, false],
      'config.process[0].model.qtype': ['comfy_w4a8', 'qfloat8'],
      'config.process[0].model.low_vram': [true, false],
      'config.process[0].model.layer_offloading': [true, false],
      'config.process[0].model.layer_offloading_transformer_percent': [1, 1],
      'config.process[0].model.cache_quantized_models': [false, true],
      'config.process[0].model.model_kwargs': [{lora_targets: 'attention', reference_max_pixels: 262144}, {}],
      'config.process[0].train.cache_text_embeddings': [true, false],
      'config.process[0].train.batch_size': [1, 1],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.sampler': ['flowmatch', 'flowmatch'],
      'config.process[0].sample.guidance_scale': [guidance, 4],
      'config.process[0].sample.sample_steps': [50, 25],
      'config.process[0].sample.width': [512, 1024],
      'config.process[0].sample.height': [512, 1024],
      'config.process[0].network.linear': [16, defaultLinearRank],
      'config.process[0].network.linear_alpha': [16, defaultLinearRank],
      'config.process[0].datasets[x].num_workers': [0, 0],
      'config.process[0].datasets[x].pin_memory': [false, false],
    },
    disableSections: ['network.conv', 'model.quantize_te'],
    additionalSections: variant === 'instruct'
      ? ['datasets.multi_control_paths', 'sample.multi_ctrl_imgs', 'model.low_vram', 'model.layer_offloading', 'model.vae_path', 'model.vision_path']
      : ['model.low_vram', 'model.layer_offloading', 'model.vae_path'],
    customModelSelectOptions: [
      {
        label: 'Checkpoint format',
        options: [{label: 'Comfy W4A8', value: 'comfy_w4a8'}, {label: 'Int8 ConvRot', value: 'convrot8'}],
        getValue: config => config.config.process[0].model.qtype,
        onChange: (value, config, setJobConfig) => setJobConfig(value, 'config.process[0].model.qtype'),
      },
      {
        label: 'LoRA targets',
        options: [{label: 'Attention', value: 'attention'}, {label: 'Attention and shared MLP', value: 'attention_shared_mlp'}],
        getValue: config => config.config.process[0].model.model_kwargs?.lora_targets ?? 'attention',
        onChange: (value, config, setJobConfig) => setJobConfig(value, 'config.process[0].model.model_kwargs.lora_targets'),
      },
      ...(variant === 'instruct' ? [{
        label: 'Reference pixel budget',
        options: [{label: '512 × 512 area', value: '262144'}, {label: '768 × 768 area', value: '589824'}, {label: '1024 × 1024 area', value: '1048576'}],
        getValue: (config: JobConfig) => String(config.config.process[0].model.model_kwargs?.reference_max_pixels ?? 262144),
        onChange: (value: string, config: JobConfig, setJobConfig: (value: any, key: string) => void) => setJobConfig(Number(value), 'config.process[0].model.model_kwargs.reference_max_pixels'),
      }] : []),
    ],
  })),
  {
    name: "anima",
    label: "Anima",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "circlestone-labs/Anima-Base-v1.0-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].model.qtype": ["", "qfloat8"],
      "config.process[0].model.qtype_te": ["", "qfloat8"],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].sample.neg": [
        "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, signature, artist name",
        "",
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "flux",
    label: "FLUX.1",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "black-forest-labs/FLUX.1-dev",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.1-dev",
  },
  {
    name: "flux_kontext",
    label: "FLUX.1-Kontext-dev",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "black-forest-labs/FLUX.1-Kontext-dev",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["datasets.control_path", "sample.ctrl_img"],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev",
  },
  {
    name: "flex1",
    label: "Flex.1",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ostris/Flex.1-alpha",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.bypass_guidance_embedding": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "chroma",
    label: "Chroma",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "lodestones/Chroma1-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "zeta_chroma",
    label: "Zeta Chroma",
    group: "experimental",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "lodestones/Zeta-Chroma/zeta-chroma-base-x0-pixel-dino-distance.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.extras_name_or_path": [
        "Tongyi-MAI/Z-Image-Turbo",
        undefined,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "wan21:1b",
    label: "Wan 2.1 (1.3B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan21_i2v:14b480p",
    label: "Wan 2.1 I2V (14B-480P)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan21_i2v:14b",
    label: "Wan 2.1 I2V (14B-720P)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan21:14b",
    label: "Wan 2.1 (14B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].datasets[x].fps": [16, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.num_frames",
      "model.low_vram",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "wan22_14b:t2v",
    label: "Wan 2.2 (14B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ai-toolkit/Wan2.2-T2V-A14B-Diffusers-bf16",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
      "config.process[0].model.model_kwargs": [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.num_frames",
      "model.low_vram",
      "model.multistage",
      "model.layer_offloading",
      "datasets.auto_frame_count",
    ],
    accuracyRecoveryAdapters: {
      // '3 bit with ARA': 'uint3|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint3.safetensors',
      "4 bit with ARA":
        "uint4|ostris/accuracy_recovery_adapters/wan22_14b_t2i_torchao_uint4.safetensors",
    },
  },
  {
    name: "wan22_14b_i2v",
    label: "Wan 2.2 I2V (14B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ai-toolkit/Wan2.2-I2V-A14B-Diffusers-bf16",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [41, 1],
      "config.process[0].sample.fps": [16, 1],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].datasets[x].fps": [16, undefined],
      "config.process[0].model.model_kwargs": [
        {
          train_high_noise: true,
          train_low_noise: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "model.multistage",
      "model.layer_offloading",
      "datasets.auto_frame_count",
    ],
    accuracyRecoveryAdapters: {
      "4 bit with ARA":
        "uint4|ostris/accuracy_recovery_adapters/wan22_14b_i2v_torchao_uint4.safetensors",
    },
  },
  {
    name: "wan22_5b",
    label: "Wan 2.2 TI2V (5B)",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].do_i2v": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.low_vram",
      "datasets.do_i2v",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "lumina2",
    label: "Lumina2",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Alpha-VLLM/Lumina-Image-2.0",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
    },
    disableSections: ["network.conv"],
  },
  {
    name: "qwen_image",
    label: "Qwen-Image",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Qwen/Qwen-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image:2512",
    label: "Qwen-Image-2512",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Qwen/Qwen-Image-2512",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
    // Training an ARA now, the other one will not work
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint3.safetensors",
      "4 bit with ARA":
        "uint4|ostris/accuracy_recovery_adapters/qwen_image_2512_torchao_uint4.safetensors",
    },
  },
  {
    name: "qwen_image_edit",
    label: "Qwen-Image-Edit",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Qwen/Qwen-Image-Edit",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.control_path",
      "sample.ctrl_img",
      "model.low_vram",
      "model.layer_offloading",
    ],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image_edit_plus",
    label: "Qwen-Image-Edit-2509",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Qwen/Qwen-Image-Edit-2509",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2509_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image_edit_plus:2511",
    label: "Qwen-Image-Edit-2511",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Qwen/Qwen-Image-Edit-2511",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/qwen_image_edit_2511_torchao_uint3.safetensors",
    },
  },
  {
    name: "qwen_image_2",
    label: "Qwen-Image-2.1",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Comfy-Org/Qwen-Image-2.1",
        defaultNameOrPath,
      ],
      "config.process[0].model.assistant_lora_path": [undefined, undefined],
      "config.process[0].model.text_encoder_path": [undefined, undefined],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      // the Comfy-Org weights are pre-quantized int8 convrot; these qtypes
      // match the checkpoints exactly, so the load is unchanged. Picking a
      // different qtype re-quantizes layer by layer into that format.
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].sample.guidance_scale": [3.0, 4.0],
      "config.process[0].sample.comfy.workflow_path": [
        "config/comfy_templates/qwen_image_2_lora_sample.json.njk",
        defaultComfySampleConfig.workflow_path,
      ],
      // the VAE is RGBA: images load, encode and decode with their alpha
      "config.process[0].model.model_kwargs": [
        {
          rgba: false,
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    // one model: it edits when the dataset has control paths, and is plain
    // text to image when it does not
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.assistant_lora_path",
      "model.text_encoder_path",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    customModelSelectOptions: qwenImage21TransparencyOption,
  },
  {
    name: "fibo2",
    label: "FIBO2 Dev",
    group: "image",
    hasMultiLinePrompts: true,
    defaults: {
      "config.process[0].model.name_or_path": ["briaai/Fibo-2.0-dev", defaultNameOrPath],
      "config.process[0].model.text_encoder_path": [undefined, undefined],
      "config.process[0].model.vae_path": [undefined, undefined],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].sample.guidance_scale": [4, 4],
      "config.process[0].sample.comfy.workflow_path": [
        "config/comfy_templates/fibo2_lora_sample.json.njk",
        defaultComfySampleConfig.workflow_path,
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths", "sample.multi_ctrl_imgs",
      "model.text_encoder_path", "model.vae_path", "model.assistant_lora_path",
      "model.low_vram", "model.layer_offloading",
    ],
    modelNotes: <p>
      Train the non-distilled FIBO2 Dev model with Qwen3-VL-4B and the FLUX.2 VAE.
      Edit Turbo is inference-only. A helper LoRA is optional and frozen during training;
      preview-only inference LoRAs are configured separately under Samples.
      Structured JSON captions and plain prompts are supported; do not prepend text to a JSON caption.
    </p>,
  },
  {
    name: "qwen_image_2:turbo",
    label: "Qwen-Image-2.1 Turbo (w/ Training Adapter)",
    generateNameOverride: "Qwen-Image-2.1 Turbo",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      // the turbo weight file in the repack (text encoder and VAE come from
      // the same repack); model_kwargs.turbo selects the turbo schedule
      "config.process[0].model.name_or_path": [
        "Comfy-Org/Qwen-Image-2.1/diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      // training adapter: live on the DiT while training, off while sampling
      "config.process[0].model.assistant_lora_path": [
        'ostris/qwen_image_2_turbo_training_adapter/qwen_image_2_turbo_training_adapter_v1.safetensors',
        undefined
      ],
      // the turbo checkpoint samples with its own fixed 8-step schedule at
      // CFG 1; the step count here is informational
      "config.process[0].sample.guidance_scale": [1.0, 4.0],
      "config.process[0].sample.sample_steps": [8, 25],
      "config.process[0].model.model_kwargs": [
        {
          turbo: true,
          rgba: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
    ],
    customModelSelectOptions: qwenImage21TransparencyOption,
  },
  {
    name: "ming_image",
    label: "Ming-Image 0.1 Design (w/ Training Adapter)",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      // the ComfyUI repack (local copies in the comfy models folder win over
      // the download); the vendor repo or a fine-tune here loads as named
      "config.process[0].model.name_or_path": [
        "Comfy-Org/Ming-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      // the text encoder is a 16B MoE (34 GB in bf16): quantize it, and
      // prefer caching text embeddings so it can be unloaded before training
      "config.process[0].model.quantize_te": [true, false],
      // the ComfyUI repack ships int8 convrot for both; these qtypes match it
      // exactly, so the files attach as-is (another qtype re-quantizes)
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      // training adapter: merged in for training, inverted for sampling
      "config.process[0].model.assistant_lora_path": [
        "ostris/ming_image_training_adapter/ming_image_01_design_training_adapter_v1.safetensors",
        undefined,
      ],
      // recommended: 12 steps, no guidance (the negative is zero conditioning)
      "config.process[0].sample.guidance_scale": [1.0, 4.0],
      "config.process[0].sample.sample_steps": [12, 25],
      // the VAE is RGBA: images load, encode and decode with their alpha
      "config.process[0].model.model_kwargs": [
        {
          rgba: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    // the model also edits (a dataset control path + reference image); the
    // UI does not expose that yet
    additionalSections: [
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
    ],
    customModelSelectOptions: [
      {
        type: "checkbox",
        label: "Transparency (RGBA)",
        getValue: (config: JobConfig) =>
          config?.config?.process?.[0]?.model?.model_kwargs?.rgba ?? false,
        onChange: (
          value: boolean,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          const kwargs = {
            ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}),
          };
          if (value) {
            kwargs.rgba = true;
          } else {
            delete kwargs.rgba;
          }
          setJobConfig(kwargs, "config.process[0].model.model_kwargs");
        },
        doc: {
          title: "Transparency (RGBA)",
          description: (
            <div className="space-y-2">
              <p>
                Ming-Image&apos;s VAE is natively RGBA, so alpha can be carried
                end to end. When on, dataset images and reference images load
                with their alpha channel, the VAE encodes all four channels, and
                samples are saved as PNGs with their transparency intact.
              </p>
              <p>
                Images with no alpha of their own get a fully opaque one, so a
                mixed dataset is fine. Turn this off to train and sample flat
                RGB: alpha is dropped on the way in and added back as opaque.
              </p>
              <p>
                To sample a transparent image, start the prompt with one of the
                model&apos;s fixed trigger phrases, e.g.{" "}
                <code>RGBA, 4-channel, transparent background</code> or{" "}
                <code>带透明通道，4通道RGBA图像</code>, and sample at 2048x2048:
                the same prompts come out opaque at 1024.
              </p>
              <p>
                Changing this re-caches latents, and it cannot be combined with
                a dataset using <code>alpha_mask</code>, which consumes the
                alpha channel as a loss mask instead.
              </p>
            </div>
          ),
        },
      },
    ],
  },
  {
    name: "hidream",
    label: "HiDream",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "HiDream-ai/HiDream-I1-Full",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.lr": [0.0002, 0.0001],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["ff_i.experts", "ff_i.gate"],
        [],
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram"],
    accuracyRecoveryAdapters: {
      "3 bit with ARA":
        "uint3|ostris/accuracy_recovery_adapters/hidream_i1_full_torchao_uint3.safetensors",
    },
  },
  {
    name: "hidream_e1",
    label: "HiDream E1",
    group: "instruction",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "HiDream-ai/HiDream-E1-1",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.lr": [0.0001, 0.0001],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["ff_i.experts", "ff_i.gate"],
        [],
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.control_path",
      "sample.ctrl_img",
      "model.low_vram",
    ],
  },
  {
    name: "sdxl",
    label: "SDXL",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "stabilityai/stable-diffusion-xl-base-1.0",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].sample.sampler": ["ddpm", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["ddpm", "flowmatch"],
      "config.process[0].sample.guidance_scale": [6, 4],
    },
    disableSections: ["model.quantize", "train.timestep_type"],
  },
  {
    name: "sd15",
    label: "SD 1.5",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        defaultNameOrPath,
      ],
      "config.process[0].sample.sampler": ["ddpm", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["ddpm", "flowmatch"],
      "config.process[0].sample.width": [512, 1024],
      "config.process[0].sample.height": [512, 1024],
      "config.process[0].sample.guidance_scale": [6, 4],
    },
    disableSections: ["model.quantize", "train.timestep_type"],
  },
  {
    name: "omnigen2",
    label: "OmniGen2",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "OmniGen2/OmniGen2",
        defaultNameOrPath,
      ],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].model.quantize": [false, false],
      "config.process[0].model.quantize_te": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["datasets.control_path", "sample.ctrl_img"],
  },
  {
    name: "flux2",
    label: "FLUX.2",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "black-forest-labs/FLUX.2-dev",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.2-dev",
  },
  {
    name: "zimage:turbo",
    label: "Z-Image Turbo (w/ Training Adapter)",
    generateNameOverride: "Z-Image Turbo",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Tongyi-MAI/Z-Image-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.assistant_lora_path": [
        "ostris/zimage_turbo_training_adapter/zimage_turbo_training_adapter_v2.safetensors",
        undefined,
      ],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [9, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
    ],
  },
  {
    name: "zimage",
    label: "Z-Image",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Tongyi-MAI/Z-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample.sample_steps": [30, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "zimage:deturbo",
    label: "Z-Image De-Turbo (De-Distilled)",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "ostris/Z-Image-De-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.extras_name_or_path": [
        "Tongyi-MAI/Z-Image-Turbo",
        undefined,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [25, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "minimax_h3_image",
    label: "MiniMax-H3-Image",
    group: "image",
    generate: { model: { model_kwargs: {} }, needsControlImage: false },
    defaults: {
      "config.process[0].model.name_or_path": ["Comfy-Org/MiniMax-H3-Image", defaultNameOrPath],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["nvfp4", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].model.low_vram_layer_streaming": [true, true],
      "config.process[0].model.model_kwargs": [{ partition: "image", h3_photo_training: true }, {}],
      "config.process[0].model.assistant_lora_path": [h3PhotoTrainingAdapter, undefined],
      "config.process[0].network.type": ["lora", "lora"],
      "config.process[0].network.linear": [8, defaultLinearRank],
      "config.process[0].network.linear_alpha": [8, defaultLinearRank],
      "config.process[0].network.transformer_only": [true, undefined],
      "config.process[0].network.network_kwargs.only_if_contains": [[...h3PhotoBlocks], undefined],
      "config.process[0].network.network_kwargs.ignore_if_contains": [[...h3PhotoIgnoredModules], []],
      "config.process[0].train.batch_size": [1, 1],
      "config.process[0].train.gradient_accumulation": [1, 1],
      "config.process[0].train.train_unet": [true, true],
      "config.process[0].train.train_text_encoder": [false, false],
      "config.process[0].train.train_refiner": [false, undefined],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].train.content_or_style": ["balanced", "balanced"],
      "config.process[0].train.content_or_style_reg": ["balanced", undefined],
      "config.process[0].train.linear_timesteps": [false, undefined],
      "config.process[0].train.linear_timesteps2": [false, undefined],
      "config.process[0].train.first_timestep_chance": [0, undefined],
      "config.process[0].train.min_denoising_steps": [0, undefined],
      "config.process[0].train.max_denoising_steps": [999, undefined],
      "config.process[0].train.num_train_timesteps": [1000, undefined],
      "config.process[0].train.min_snr_gamma": [0, 5],
      "config.process[0].train.snr_gamma": [0, undefined],
      "config.process[0].train.loss_type": ["mse", "mse"],
      "config.process[0].train.loss_target": ["noise", undefined],
      "config.process[0].train.frequency_loss_type": ["none", "none"],
      "config.process[0].train.optimizer": ["automagic3", "adamw8bit"],
      "config.process[0].train.lr": [1e-6, 1e-4],
      "config.process[0].train.lr_scheduler": ["constant", undefined],
      "config.process[0].train.cache_text_embeddings": [true, false],
      "config.process[0].train.audio_loss_multiplier": [0, undefined],
      "config.process[0].train.ema_config": [{ use_ema: true, ema_decay: 0.98 }, { use_ema: false, ema_decay: 0.99 }],
      "config.process[0].datasets[x].batch_size": [1, undefined],
      "config.process[0].datasets[x].resolution": [[512], [512, 768, 1024]],
      "config.process[0].datasets[x].native_resolution": [false, false],
      "config.process[0].datasets[x].buckets": [true, undefined],
      "config.process[0].datasets[x].num_frames": [1, 1],
      "config.process[0].datasets[x].auto_frame_count": [false, undefined],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [false, undefined],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].cache_text_embeddings": [true, undefined],
      "config.process[0].datasets[x].caption_dropout_rate": [0.05, 0.05],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.width": [defaultMinimaxH3ImageSampleConfig.width, defaultSampleConfig.width],
      "config.process[0].sample.height": [defaultMinimaxH3ImageSampleConfig.height, defaultSampleConfig.height],
      "config.process[0].sample.num_frames": [1, 1],
      "config.process[0].sample.fps": [1, 1],
      "config.process[0].sample.guidance_scale": [1, defaultSampleConfig.guidance_scale],
      "config.process[0].sample.sample_steps": [6, defaultSampleConfig.sample_steps],
      "config.process[0].sample.comfy": [defaultMinimaxH3ImageSampleConfig.comfy, defaultComfySampleConfig],
    },
    disableSections: ["network.conv"],
    additionalSections: ["sample.multi_ctrl_imgs", "model.low_vram", "model.low_vram_layer_streaming", "model.layer_offloading", "model.assistant_lora_path"],
    advancedModelOptions: h3ImageAdvancedModelOptions,
    advancedTargetOptions: h3ImageAdvancedTargetOptions,
    initializeDataset: (dataset, config) => {
      const choices = config.config.process[0].datasets.map(item => item.cache_text_embeddings ?? false);
      return choices.length && choices.every(choice => choice === choices[0])
        ? { ...dataset, cache_text_embeddings: choices[0] } : dataset;
    },
    customModelSelectOptions: [{
      label: "Training recipe",
      options: [
        { value: "video", label: "Image mode (keep current settings)" },
        { value: "photo", label: "Apply experimental photo recipe" },
        { value: "custom_photo", label: "Image photo recipe — customized settings", isDisabled: true },
      ],
      getValue: getMinimaxH3TrainingRecipe,
      onChange: (value, config, setJobConfig) => {
        if (value === "photo") {
          setJobConfig(applyMinimaxH3PhotoRecipe(config).config.process[0], "config.process[0]");
        } else if (value === "video") {
          setJobConfig(false, "config.process[0].model.model_kwargs.h3_photo_training");
        }
      },
      doc: {
        title: "MiniMax-H3-Image experimental photo recipe",
        description: (
          <div className="space-y-2">
            <p>The official image model samples at six PDD steps, CFG 1, with a 2048 × 2048 preview by default. Selecting the recipe changes training settings; opening a saved job does not.</p>
            <p>Initial training settings follow the Fizgig still-photo hypothesis: 512-pixel buckets, rank/alpha 8 on blocks 20–49, frozen Circlestone helper, Automagic v3 at 1e-6, EMA 0.98 and 5% caption dropout. This has not been qualified for image quality on this official image checkpoint.</p>
            <p>Dataset folders and captions, prompts, total steps, model component overrides, quantization and memory choices are retained. Reference controls and conflicting guidance objectives are cleared. Native sampling is selected; the separate image Comfy graph remains available in the workflow selector.</p>
          </div>
        ),
      },
    }],
    modelNotes: (
      <div className="space-y-2">
        <p>Official MiniMax-H3-Image uses an audio-free image DiT and image VAE. Missing files are resolved from <code>Comfy-Org/MiniMax-H3-Image</code>; its Qwen3-VL text encoder is shared with <code>Comfy-Org/MiniMax-H3</code>.</p>
        <p>Native previews use the fixed six-step PDD schedule at guidance 1. The 2048 × 2048 sample canvas follows the official image setting; the 512-pixel training bucket is an initial recipe assumption and can be edited.</p>
      </div>
    ),
  },
  {
    name: "minimax_h3",
    label: "MiniMax-H3",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Comfy-Org/MiniMax-H3",
        defaultNameOrPath,
      ],
      // the Comfy-Org weights are pre-quantized (int8 convrot DiT, nvfp4 TE); these
      // qtypes match the checkpoints exactly, so the load is unchanged. Picking a
      // different qtype re-quantizes layer by layer into that format.
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["nvfp4", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].model.low_vram_layer_streaming": [true, true],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.cache_text_embeddings": [true, false],
      "config.process[0].model.assistant_lora_path": [
        "ostris/minimax_h3_training_adapter/minimax_h3_training_adapter_v3.safetensors",
        undefined,
      ],
      "config.process[0].network.linear": [16, defaultLinearRank],
      "config.process[0].network.linear_alpha": [16, defaultLinearRank],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["adaln_proj"],
        [],
      ],
      "config.process[0].sample.num_frames": [107, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [28, 25],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].num_frames": [39, undefined],
      "config.process[0].datasets[x].auto_frame_count": [true, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "model.low_vram_layer_streaming",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
      "model.assistant_lora_path",
    ],
    customModelSelectOptions: [
      {
        label: "Training recipe",
        options: [
          { value: "video", label: "Video mode (keep current settings)" },
          { value: "photo", label: "Apply still-photo recipe" },
          { value: "custom_photo", label: "Photo mode — customized settings", isDisabled: true },
        ],
        getValue: getMinimaxH3TrainingRecipe,
        onChange: (value, config, setJobConfig) => {
          if (value === "photo") {
            setJobConfig(applyMinimaxH3PhotoRecipe(config).config.process[0], "config.process[0]");
          } else if (value === "video") {
            // This changes the H3 training schedule; existing user choices stay as they are.
            setJobConfig(false, "config.process[0].model.model_kwargs.h3_photo_training");
          }
        },
        doc: {
          title: "MiniMax-H3 training recipe",
          description: (
            <div className="space-y-2">
              <p>Apply still-photo recipe is an explicit action. Opening a job or choosing MiniMax-H3 leaves its saved video settings alone.</p>
              <p>The recipe sets 512-pixel stills, rank/alpha 8 LoRA on blocks 20–49, excludes the token refiner and AdaLN projection, and selects the Circlestone image training adapter. It uses Automagic v3 at 1e-6 with constant LR, EMA 0.98, 5% caption dropout, and a single-image native sample at guidance 1.</p>
              <p>It keeps dataset folders and captions, sample prompts, total steps, model component paths, quantization, and memory settings. It clears image/video references and incompatible guidance or preservation objectives, and turns off ComfyUI sampling because the supplied H3 graph is for video. The saved ComfyUI workflow remains available if you later configure an image-compatible graph.</p>
              <p>The disabled “customized settings” entry reports that a photo-mode config differs from this recipe. Choosing Video mode changes the H3 schedule back to video behavior while retaining the other current settings.</p>
            </div>
          ),
        },
      },
      {
        label: "Distillation Handling Method",
        options: [
          { value: "cg", label: "Contrastive Guidance" },
          { value: "ta", label: "Training Adapter (default)" },
          {
            value: "both",
            label: "Contrastive Guidance + Training Adapter",
          },
          { value: "none", label: "None" },
        ],
        getValue: (config: JobConfig) => {
          const assistantLoraPath =
            config?.config?.process?.[0]?.model?.assistant_lora_path;
          const hasAssistantLoraPath =
            assistantLoraPath && assistantLoraPath.trim() !== "";
          const hasContrastiveGuidance =
            config?.config?.process?.[0]?.train?.do_guidance_loss;
          if (hasAssistantLoraPath && hasContrastiveGuidance) {
            return "both";
          }
          if (hasAssistantLoraPath) {
            return "ta";
          }
          if (hasContrastiveGuidance) {
            return "cg";
          }
          return "none";
        },
        onChange: (
          value: string,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          if (value === "cg") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "ta") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              config.config.process[0].model.model_kwargs?.h3_photo_training ? h3PhotoTrainingAdapter : h3VideoTrainingAdapter,
              "config.process[0].model.assistant_lora_path",
            );
          } else if (value === "both") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              config.config.process[0].model.model_kwargs?.h3_photo_training ? h3PhotoTrainingAdapter : h3VideoTrainingAdapter,
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "none") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
          }
        },
        doc: {
          title: "MiniMax-H3 Distillation Handling",
          description: (
            <div>
              MiniMax H3 is a guidance distilled model, so training on it
              directly will make the guidance distillation break down. There are
              two different ways to train on this model without breaking the
              guidance distillation: Contrastive Guidance and Training Adapter.
              Both have their pros and cons. The adapter is faster, but will
              still break down over a long run. Contrastive Guidance is slower,
              but is less likely to break down.
            </div>
          ),
        },
      },
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          Weights load from the{" "}
          <Link href="/settings" className="text-blue-400 hover:underline">
            Models Folder Path
          </Link>{" "}
          set in settings. Anything missing is downloaded there from{" "}
          <code>Comfy-Org/MiniMax-H3</code> on first load (~43GB total). Files
          used:
        </p>
        <pre className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs overflow-x-auto">
          <code>{`<MODELS_PATH>/
├── diffusion_models/
│   └── minimax_h3_fl2va_pruned_int8_convrot.safetensors
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors`}</code>
        </pre>
        <p>
          The checkpoints are pre-quantized and load directly: int8 ConvRot DiT
          (~21GB) and nvfp4 Qwen3-VL text encoder (~16GB). The default qtypes (
          <code>convrot8</code> / <code>nvfp4</code>) match the files exactly,
          so nothing is re-quantized on load. Picking a different quantization
          re-quantizes the pre-quantized layers into that format, one layer at a
          time.
        </p>
        <p>
          Supports t2v and first-frame i2v (ctrl img / i2v datasets) with joint
          audio. The model is guidance-distilled — keep guidance scale at 1.
          Video is fixed 24 fps and frame counts snap down to the 17n+5 grid (5,
          22, 39, 56, ..., 107, 124 ≈ 5s). Image datasets (num_frames 1) train
          as single frames, and a sample with num_frames 1 renders a single
          image.
        </p>
      </div>
    ),
  },
  {
    name: "minimax_h3_ref2va",
    label: "MiniMax-H3 Ref2V",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Comfy-Org/MiniMax-H3",
        defaultNameOrPath,
      ],
      // pre-quantized weights: matching qtypes keep the load unchanged
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["nvfp4", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].model.low_vram_layer_streaming": [true, true],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.cache_text_embeddings": [true, false],
      "config.process[0].model.assistant_lora_path": [
        "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v2.safetensors",
        undefined,
      ],
      "config.process[0].network.linear": [16, defaultLinearRank],
      "config.process[0].network.linear_alpha": [16, defaultLinearRank],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["adaln_proj"],
        [],
      ],
      "config.process[0].sample.num_frames": [107, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [28, 25],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].num_frames": [39, undefined],
      "config.process[0].datasets[x].auto_frame_count": [true, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.multi_ctrl_imgs",
      "datasets.multi_control_paths",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "model.low_vram_layer_streaming",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
      "model.assistant_lora_path",
    ],
    customModelSelectOptions: [
      {
        label: "Distillation Handling Method",
        options: [
          { value: "cg", label: "Contrastive Guidance" },
          { value: "ta", label: "Training Adapter (default)" },
          {
            value: "both",
            label: "Contrastive Guidance + Training Adapter",
          },
          { value: "dopsd", label: "D-OPSD" },
          { value: "none", label: "None" },
        ],
        getValue: (config: JobConfig) => {
          if (config?.config?.process?.[0]?.model?.model_kwargs?.dopsd) {
            return "dopsd";
          }
          const assistantLoraPath =
            config?.config?.process?.[0]?.model?.assistant_lora_path;
          const hasAssistantLoraPath =
            assistantLoraPath && assistantLoraPath.trim() !== "";
          const hasContrastiveGuidance =
            config?.config?.process?.[0]?.train?.do_guidance_loss;
          if (hasAssistantLoraPath && hasContrastiveGuidance) {
            return "both";
          }
          if (hasAssistantLoraPath) {
            return "ta";
          }
          if (hasContrastiveGuidance) {
            return "cg";
          }
          return "none";
        },
        onChange: (
          value: string,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          const kwargs = {
            ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}),
          };
          if (value === "dopsd") {
            kwargs.dopsd = true;
            kwargs.dopsd_bleed_strength = 1.0;
          } else {
            delete kwargs.dopsd;
            delete kwargs.dopsd_bleed_strength;
          }
          setJobConfig(kwargs, "config.process[0].model.model_kwargs");
          if (value === "cg") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "ta") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v2.safetensors",
              "config.process[0].model.assistant_lora_path",
            );
          } else if (value === "both") {
            setJobConfig(true, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v2.safetensors",
              "config.process[0].model.assistant_lora_path",
            );
            if (!config?.config?.process?.[0]?.train?.guidance_loss_target) {
              setJobConfig(3.5, "config.process[0].train.guidance_loss_target");
            }
          } else if (value === "dopsd" || value === "none") {
            setJobConfig(undefined, "config.process[0].train.do_guidance_loss");
            setJobConfig(
              undefined,
              "config.process[0].train.guidance_loss_target",
            );
            setJobConfig(
              undefined,
              "config.process[0].model.assistant_lora_path",
            );
          }
        },
        doc: {
          title: "MiniMax-H3 Distillation Handling",
          description: (
            <div>
              MiniMax H3 is a guidance distilled model, so training on it
              directly will make the guidance distillation break down. There are
              two different ways to train on this model without breaking the
              guidance distillation: Contrastive Guidance and Training Adapter.
              Both have their pros and cons. The adapter is faster, but will
              still break down over a long run. Contrastive Guidance is slower,
              but is less likely to break down. D-OPSD instead self-distills: a
              no-grad teacher pass sees the training target as its own reference
              and its prediction becomes the training target for a
              reference-free pass, baking the reference into your trigger word
              (or the caption itself when no trigger word is set). Re-caches
              latents with pixel tensors.
            </div>
          ),
        },
      },
      {
        label: "Image Reference Presentation",
        options: [
          { value: "picture", label: "Picture (default)" },
          { value: "video", label: "Static video clip" },
        ],
        getValue: (config: JobConfig) => {
          return config?.config?.process?.[0]?.model?.model_kwargs
            ?.image_refs_as_video
            ? "video"
            : "picture";
        },
        onChange: (
          value: string,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          const kwargs = {
            ...(config?.config?.process?.[0]?.model?.model_kwargs ?? {}),
          };
          if (value === "video") {
            kwargs.image_refs_as_video = true;
          } else {
            delete kwargs.image_refs_as_video;
            delete kwargs.image_ref_video_frames;
          }
          setJobConfig(kwargs, "config.process[0].model.model_kwargs");
        },
        doc: {
          title: "MiniMax-H3 Image Reference Presentation",
          description: (
            <div className="space-y-2">
              <p>
                How still-image references (dataset control images and sample
                ctrl images) are shown to the model. Video references always use
                the video path.
              </p>
              <p>
                <strong>Picture</strong>: the native ref2va recipe — a
                single-frame reference block, shown to Qwen3-VL as a{" "}
                <code>&lt;Picture i&gt;</code> block, scaled down only.
              </p>
              <p>
                <strong>Static video clip</strong>: the image is held for 5
                frames (2 latent frames) and routed through the exact path a
                reference VIDEO takes — video sizing (matched to the target's
                pixel area), multi-frame reference block,{" "}
                <code>&lt;Video k&gt;</code> timestamped presentation. Use this
                when training on image references but sampling with video
                references, so the LoRA learns the pathway it will be used
                through. Adds a handful of rows per reference. Frame count is
                adjustable with <code>model_kwargs.image_ref_video_frames</code>{" "}
                (17n+5). Changing this re-caches text embeddings.
              </p>
            </div>
          ),
        },
      },
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          Reference-to-video: control images and videos condition the output as
          subject/style references (never as a first frame). References keep
          their own aspect and are matched to the target's pixel area (images
          scale down only, never up; a same-aspect video reference is exactly
          the target size). Each rides into the packed sequence as a reference
          block, and is also shown to the Qwen3-VL conditioner as a{" "}
          <code>&lt;Picture i&gt;</code> (image) or timestamped{" "}
          <code>&lt;Video k&gt;</code> (video) vision block. Training references
          come from the dataset control path(s); sampling uses the sample ctrl
          images — always as references. The Image Reference Presentation option
          can route still images through the video-reference path as short
          static clips.
        </p>
        <p>
          Weights load like MiniMax-H3 (see that arch's notes) from the{" "}
          <Link href="/settings" className="text-blue-400 hover:underline">
            Models Folder Path
          </Link>
          , using{" "}
          <code>
            diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
          </code>{" "}
          — the ref2va partition of the same release; text encoder and VAEs are
          shared with the fl2va arch. Everything else (pre-quantized load, 24
          fps, 17n+5 frame grid, guidance scale 1, single-image mode) matches
          MiniMax-H3.
        </p>
      </div>
    ),
  },
  {
    name: "minimax_h3_vsa",
    label: "FastH3 8-Step V2",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "FastVideo/FastVideo-FastH3-Comfy",
        defaultNameOrPath,
      ],
      // pre-quantized weights: matching qtypes keep the load unchanged
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["nvfp4", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.cache_text_embeddings": [true, false],
      "config.process[0].model.assistant_lora_path": [
        "ostris/minimax_h3_training_adapter/fastvideo_fasth3_8step_v2_training_adapter_v1.safetensors",
        undefined,
      ],
      "config.process[0].network.linear": [16, defaultLinearRank],
      "config.process[0].network.linear_alpha": [16, defaultLinearRank],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["adaln_proj"],
        [],
      ],
      "config.process[0].sample.num_frames": [107, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [8, 25],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].num_frames": [39, undefined],
      "config.process[0].datasets[x].auto_frame_count": [true, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
      "model.assistant_lora_path",
    ],
    customModelSelectOptions: [
      {
        label: "Distillation Handling Method",
        options: [
          { value: "ta", label: "Training Adapter (default)" },
          { value: "none", label: "None" },
        ],
        getValue: (config: JobConfig) => {
          const assistantLoraPath =
            config?.config?.process?.[0]?.model?.assistant_lora_path;
          return assistantLoraPath && assistantLoraPath.trim() !== ""
            ? "ta"
            : "none";
        },
        onChange: (
          value: string,
          config: JobConfig,
          setJobConfig: (value: any, key: string) => void,
        ) => {
          setJobConfig(
            value === "ta"
              ? "ostris/minimax_h3_training_adapter/fastvideo_fasth3_8step_v2_training_adapter_v1.safetensors"
              : undefined,
            "config.process[0].model.assistant_lora_path",
          );
        },
        doc: {
          title: "FastH3 Distillation Handling",
          description: (
            <div>
              FastH3 is a step-distilled model, so training on it directly will
              make the distillation break down. The Training Adapter (the
              MiniMax-H3 adapter) absorbs that drift during training and is
              removed at inference, but can still break down over a long run.
            </div>
          ),
        },
      },
    ],
    modelNotes: (
      <div className="space-y-2">
        <p>
          FastVideo's FastH3 8-Step V2: MiniMax-H3 DMD2-distilled to 8 steps
          with Video Sparse Attention (80% sparsity). Text-to-video with joint
          audio only — no first-frame or reference conditioning.
        </p>
        <p>
          Weights load from the{" "}
          <Link href="/settings" className="text-blue-400 hover:underline">
            Models Folder Path
          </Link>{" "}
          set in settings. Anything missing is downloaded there from{" "}
          <code>FastVideo/FastVideo-FastH3-Comfy</code> on first load. The text
          encoder and VAEs are the same files MiniMax-H3 uses, so an existing
          MiniMax-H3 setup only needs the new DiT. Files used:
        </p>
        <pre className="bg-gray-900 border border-gray-700 rounded-lg p-3 text-xs overflow-x-auto">
          <code>{`<MODELS_PATH>/
├── diffusion_models/
│   └── fastvideo_fasth3_8step_v2_pruned_int8_convrot.safetensors
├── text_encoders/
│   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
└── vae/
    ├── minimax_h3_video_vae_fp16.safetensors
    └── minimax_h3_audio_vae_fp32.safetensors`}</code>
        </pre>
        <p>
          Sample with 8 steps and guidance scale 1 (the trained schedule). Video
          is fixed 24 fps and frame counts snap down to the 17n+5 grid.
        </p>
      </div>
    ),
  },
  {
    name: "ltx2",
    label: "LTX-2",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Lightricks/LTX-2",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].auto_frame_count": [false, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "ltx2.3",
    label: "LTX-2.3",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].auto_frame_count": [false, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "ltx2.5",
    label: "LTX-2.5",
    gateUrl: "https://huggingface.co/Lightricks/LTX-2.5",
    group: "video",
    isVideoModel: true,
    defaults: {
      // default updates when [selected, unselected] in the UI
      // comfy-style split files resolve from/download to the models folder;
      // the int8 ConvRot dev transformer is the default
      "config.process[0].model.name_or_path": [
        "Lightricks/LTX-2.5",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.num_frames": [121, 1],
      "config.process[0].sample.fps": [24, 1],
      "config.process[0].sample.width": [768, 1024],
      "config.process[0].sample.height": [768, 1024],
      "config.process[0].train.audio_loss_multiplier": [1.0, undefined],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].datasets[x].cache_latents_to_disk": [true, false],
      "config.process[0].datasets[x].do_i2v": [false, undefined],
      "config.process[0].datasets[x].do_audio": [true, undefined],
      "config.process[0].datasets[x].fps": [24, undefined],
      "config.process[0].datasets[x].auto_frame_count": [false, undefined],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "sample.ctrl_img",
      "datasets.num_frames",
      "model.layer_offloading",
      "model.low_vram",
      "datasets.do_audio",
      "datasets.audio_normalize",
      "datasets.audio_preserve_pitch",
      "datasets.do_i2v",
      "train.audio_loss_multiplier",
      "datasets.auto_frame_count",
    ],
  },
  {
    name: "flux3",
    label: "FLUX.3 Image Klein (base)",
    group: "image",
    gateUrl: "https://huggingface.co/black-forest-labs/flux-3-image-klein-base",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "black-forest-labs/flux-3-image-klein-base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.width": [1024, 1024],
      "config.process[0].sample.height": [1024, 1024],
      "config.process[0].sample.guidance_scale": [3.5, 4],
      "config.process[0].sample.sample_steps": [28, 25],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.layer_offloading",
      "model.low_vram",
      "model.qie.match_target_res",
    ],
  },
  {
    name: "moss_image",
    label: "MOSS-Image SFT",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      // the vendor repo: its transformer is merged once into the comfy models
      // folder; a .safetensors path (fine-tune / int8 convrot repack) loads as named
      "config.process[0].model.name_or_path": [
        "OpenMOSS-Team/MOSS-Image-SFT",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      // Qwen3.5-9B text encoder
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.width": [1024, 1024],
      "config.process[0].sample.height": [1024, 1024],
      // vendor defaults: CFG 3 with block 22 skipped on the negative branch
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [30, 25],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      // sampling: the CFG negative branch skips these DiT blocks (zero-based)
      "config.process[0].model.model_kwargs": [
        {
          uncond_skip_layers: [22],
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.layer_offloading", "model.low_vram", "model.text_encoder_path"],
  },
  {
    name: "moss_image:base",
    label: "MOSS-Image Base",
    generateNameOverride: "MOSS-Image Base",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      // the vendor repo: Comfy-Org/MOSS-Image base files on disk win, else the
      // repack's bf16 (identical weights) is downloaded
      "config.process[0].model.name_or_path": [
        "OpenMOSS-Team/MOSS-Image-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      // Qwen3.5-9B text encoder
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.width": [1024, 1024],
      "config.process[0].sample.height": [1024, 1024],
      // vendor defaults: CFG 3 with block 22 skipped on the negative branch
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [30, 25],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      // sampling: the CFG negative branch skips these DiT blocks (zero-based)
      "config.process[0].model.model_kwargs": [
        {
          uncond_skip_layers: [22],
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.layer_offloading", "model.low_vram", "model.text_encoder_path"],
  },
  {
    name: "moss_image:edit",
    label: "MOSS-Image SFT [Edit Training]",
    generateNameOverride: "MOSS-Image SFT Edit",
    group: "experimental",
    modelNotes: mossImageEditNotes,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "OpenMOSS-Team/MOSS-Image-SFT",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      // Qwen3.5-9B text encoder + vision tower (references ride in the prompt)
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      // prompt embeds depend on the control images, so the TE stays loaded
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.width": [1024, 1024],
      "config.process[0].sample.height": [1024, 1024],
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [30, 25],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      // edit: references as clean t=0 latents + in the prompt; kv_cache:
      // references attend only to themselves, so sampling computes their K/V
      // once (must be trained that way)
      "config.process[0].model.model_kwargs": [
        {
          edit: true,
          kv_cache: true,
          match_target_res: true,
          uncond_skip_layers: [22],
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.text_encoder_path",
      "model.qie.match_target_res",
      "model.model_kwargs.kv_cache",
    ],
  },
  {
    name: "moss_image:base_edit",
    label: "MOSS-Image Base [Edit Training]",
    generateNameOverride: "MOSS-Image Base Edit",
    group: "experimental",
    modelNotes: mossImageEditNotes,
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "OpenMOSS-Team/MOSS-Image-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      // Qwen3.5-9B text encoder + vision tower (references ride in the prompt)
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].model.low_vram": [true, false],
      // prompt embeds depend on the control images, so the TE stays loaded
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].sample.width": [1024, 1024],
      "config.process[0].sample.height": [1024, 1024],
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [30, 25],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      // edit: references as clean t=0 latents + in the prompt; kv_cache:
      // references attend only to themselves, so sampling computes their K/V
      // once (must be trained that way)
      "config.process[0].model.model_kwargs": [
        {
          edit: true,
          kv_cache: true,
          match_target_res: true,
          uncond_skip_layers: [22],
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.text_encoder_path",
      "model.qie.match_target_res",
      "model.model_kwargs.kv_cache",
    ],
  },
  {
    name: "flux2_klein_4b",
    label: "FLUX.2-klein-base-4B",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "black-forest-labs/FLUX.2-klein-base-4B",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
  },
  {
    name: "ernie_image",
    label: "ERNIE-Image",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "baidu/ERNIE-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "flux2_klein_9b",
    label: "FLUX.2-klein-base-9B",
    group: "image",
    defaults: {
      // default updates when [selected, unselected] in the UI
      "config.process[0].model.name_or_path": [
        "black-forest-labs/FLUX.2-klein-base-9B",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.sampler": ["flowmatch", "flowmatch"],
      "config.process[0].train.noise_scheduler": ["flowmatch", "flowmatch"],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].model.qtype": ["qfloat8", "qfloat8"],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
    gateUrl: "https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B",
  },
  {
    name: "nucleus_image",
    label: "Nucleus-Image",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "NucleusAI/Nucleus-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["img_mlp.experts", "img_mlp.gate"],
        [],
      ],
      "config.process[0].network.linear": [128, defaultLinearRank],
      "config.process[0].network.linear_alpha": [128, defaultLinearRank],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram"],
  },
  {
    name: "hidream_o1",
    label: "HiDream-O1",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "HiDream-ai/HiDream-O1-Image",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [false, false],
      "config.process[0].train.timestep_type": ["weighted", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].train.max_loss": [1.0, undefined],
      "config.process[0].network.network_kwargs.ignore_if_contains": [
        ["lm_head", "patch_embed", "visual"],
        [],
      ],
      "config.process[0].network.transformer_only": [false, undefined],
      "config.process[0].sample.width": [2048, 1024],
      "config.process[0].sample.height": [2048, 1024],
      "config.process[0].model.model_kwargs": [
        {
          noise_scale_inference: 8.0,
          noise_scale: 8.0,
        },
        {},
      ],
    },
    disableSections: [
      "network.conv",
      "model.quantize_te",
      "train.unload_text_encoder",
    ],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "zimage_l2p",
    label: "Z-Image L2P (pixel space)",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "zhen-nan/L2P/model-1k-merge.safetensors",
        defaultNameOrPath,
      ],
      "config.process[0].model.extras_name_or_path": [
        "Tongyi-MAI/Z-Image-Turbo",
        undefined,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "ideogram4",
    label: "Ideogram4",
    group: "experimental",
    defaults: {
      "config.process[0].model.name_or_path": [
        "ideogram-ai/ideogram-4-fp8",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample": [
        defaultIdeogramSamplesConfig,
        defaultSampleConfig,
      ],
      "config.process[0].model.unconditional_lora_path": [
        "ostris/ideogram_4_unconditional_lora/ideogram_4_unconditional_lora_r16.safetensors",
        undefined,
      ],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "model.low_vram",
      "model.layer_offloading",
      "ideogram_4_prompt",
      "model.unconditional_lora_path",
    ],
    hasMultiLinePrompts: true,
    gateUrl: "https://huggingface.co/ideogram-ai/ideogram-4-fp8",
  },
  {
    name: "prx_pixel",
    label: "PRXPixel (pixel space)",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "Photoroom/prxpixel-t2i",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "iris",
    label: "Iris 3B (pixel space)",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": ["speridlabs/iris-3b", defaultNameOrPath],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.timestep_type": ["sigmoid", "sigmoid"],
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [100, 25],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading", "model.text_encoder_path"],
  },
  {
    name: "iris3b",
    label: "Iris-3B (pixel space)",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "speridlabs/iris-3b",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].model.qtype": ["convrot8", "qfloat8"],
      "config.process[0].model.qtype_te": ["convrot8", "qfloat8"],
      "config.process[0].train.timestep_type": ["shift", "sigmoid"],
      "config.process[0].sample.guidance_scale": [3, 4],
      "config.process[0].sample.sample_steps": [30, 25],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "krea2",
    label: "Krea 2 (raw)",
    group: "image",
    gateUrl: "https://huggingface.co/krea/Krea-2-Raw",
    defaults: {
      "config.process[0].model.name_or_path": [
        "krea/Krea-2-Raw",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "krea2:turbo",
    label: "Krea 2 Turbo (w/ Training Adapter)",
    generateNameOverride: "Krea 2 Turbo",
    group: "image",
    gateUrl: "https://huggingface.co/krea/Krea-2-Turbo",
    defaults: {
      "config.process[0].model.name_or_path": [
        "krea/Krea-2-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].model.assistant_lora_path": [
        "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
        undefined,
      ],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [9, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: [
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
    ],
  },
  {
    name: "krea2:o_edit",
    label: "Krea 2 (raw) [Edit Training]",
    gateUrl: "https://huggingface.co/krea/Krea-2-Raw",
    group: "experimental",
    defaults: {
      "config.process[0].model.name_or_path": [
        "krea/Krea-2-Raw",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.comfy.workflow_path": [
        "config/comfy_templates/krea2_ostris_edit_lora_sample.json.njk",
        defaultComfySampleConfig.workflow_path,
      ],
      "config.process[0].model.model_kwargs": [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
      "model.qie.match_target_res",
      "model.model_kwargs.kv_cache",
    ],
  },
  {
    name: "krea2:o_edit_turbo",
    label: "Krea 2 Turbo (w/ Training Adapter) [Edit Training]",
    generateNameOverride: "Krea 2 Turbo (Edit)",
    gateUrl: "https://huggingface.co/krea/Krea-2-Turbo",
    group: "experimental",
    defaults: {
      "config.process[0].model.name_or_path": [
        "krea/Krea-2-Turbo",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].model.assistant_lora_path": [
        "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
        undefined,
      ],
      "config.process[0].sample.guidance_scale": [1, 4],
      "config.process[0].sample.sample_steps": [8, 25],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].sample.comfy.workflow_path": [
        "config/comfy_templates/krea2_ostris_edit_lora_sample.json.njk",
        defaultComfySampleConfig.workflow_path,
      ],
      "config.process[0].model.model_kwargs": [
        {
          edit: true,
          match_target_res: true,
          kv_cache: true,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.assistant_lora_path",
      "model.qie.match_target_res",
      "model.model_kwargs.kv_cache",
    ],
  },
  {
    name: "mageflow",
    label: "Mage-Flow",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "microsoft/Mage-Flow-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.guidance_scale": [4, 4],
      "config.process[0].sample.sample_steps": [25, 25],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "mageflow_edit",
    label: "Mage-Flow Edit",
    group: "instruction",
    defaults: {
      "config.process[0].model.name_or_path": [
        "microsoft/Mage-Flow-Edit-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].sample.guidance_scale": [4, 4],
      "config.process[0].sample.sample_steps": [25, 25],
      "config.process[0].train.unload_text_encoder": [false, false],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
    ],
  },
  {
    name: "boogu_image",
    label: "Boogu Image",
    group: "image",
    defaults: {
      "config.process[0].model.name_or_path": [
        "Boogu/Boogu-Image-0.1-Base",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
    },
    disableSections: ["network.conv"],
    additionalSections: ["model.low_vram", "model.layer_offloading"],
  },
  {
    name: "boogu_image_edit",
    label: "Boogu Image Edit",
    group: "instruction",
    defaults: {
      "config.process[0].model.name_or_path": [
        "Boogu/Boogu-Image-0.1-Edit",
        defaultNameOrPath,
      ],
      "config.process[0].model.quantize": [true, false],
      "config.process[0].model.quantize_te": [true, false],
      "config.process[0].train.timestep_type": ["linear", "sigmoid"],
      "config.process[0].network.conv": [undefined, 16],
      "config.process[0].network.conv_alpha": [undefined, 16],
      "config.process[0].model.low_vram": [true, false],
      "config.process[0].train.unload_text_encoder": [false, false],
      "config.process[0].model.model_kwargs": [
        {
          match_target_res: false,
        },
        {},
      ],
    },
    disableSections: ["network.conv", "train.unload_text_encoder"],
    additionalSections: [
      "datasets.multi_control_paths",
      "sample.multi_ctrl_imgs",
      "model.low_vram",
      "model.layer_offloading",
      "model.qie.match_target_res",
    ],
  },
];
