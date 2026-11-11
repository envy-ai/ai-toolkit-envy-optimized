const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const test = require('node:test');

function load(file, overrides = {}, jsx = false) {
  const filename = path.resolve(__dirname, file);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: jsx ? ts.JsxEmit.ReactJSX : undefined,
    },
  }).outputText;
  const result = new Module(filename, module);
  result.filename = filename;
  result.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = result.require.bind(result);
  result.require = id => Object.hasOwn(overrides, id) ? overrides[id] : originalRequire(id);
  result._compile(compiled, filename);
  return result.exports;
}
const caps = load('../src/app/jobs/new/trainingCapabilities.ts');
const modes = caps.specializedTrainingModes;
function config(arch, type, network = 'lora') {
  const model = { arch, name_or_path: '/models/base', model_kwargs: {} };
  if (arch === 'fibo2') model.vae_path = '/fixture/flux2-vae.safetensors';
  const job = { config: { process: [{ type, model,
    network: { type: network, linear: 4 }, datasets: [], train: {}, sample: { samples: [], comfy: {} } }] } };
  if (type === 'diffusion_kto') {
    job.config.process[0].train = { cache_text_embeddings: true, noise_scheduler: 'flowmatch' };
    job.config.process[0].datasets = [{ folder_path: '/liked', kto_label: 'liked', cache_latents_to_disk: true }];
  }
  return job;
}

test('ordinary MOSS jobs reject the bundled Krea preview workflow before saving', () => {
  const job = config('moss_image:base', 'diffusion_trainer', 'dora');
  job.config.process[0].sample = { comfy: { enabled: true, workflow_path: 'config/comfy_templates/krea2_lora_sample.json.njk' } };
  assert.match(caps.validateTrainingCapabilities(job).join('\n'), /Krea 2 ComfyUI workflows are incompatible/);
  job.config.process[0].sample.comfy.enabled = false;
  assert.deepEqual(caps.validateTrainingCapabilities(job), []);
  job.config.process[0].sample.comfy = { enabled: true, workflow_path: 'config/comfy_templates/moss_image_lora_sample.json.njk' };
  assert.deepEqual(caps.validateTrainingCapabilities(job), []);
  job.config.process[0].sample.comfy = { enabled: true, workflow_path: 'custom/moss_sample.json' };
  assert.deepEqual(caps.validateTrainingCapabilities(job), []);
  const simpleJob = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  assert.match(simpleJob, /value: 'config\/comfy_templates\/moss_image_lora_sample\.json\.njk', label: 'MOSS-Image LoRA image'/);
});

test('ordinary LoKr and DoKr jobs allow compatible previews and enforce workflow compatibility', () => {
  for (const dora of [false, true]) {
    const job = config('moss_image:base', 'diffusion_trainer', 'lokr');
    const process = job.config.process[0];
    process.network.lokr_dora = dora;
    process.sample.comfy = { enabled: true, workflow_path: 'config/comfy_templates/moss_image_lora_sample.json.njk' };
    assert.deepEqual(caps.validateTrainingCapabilities(job), []);
    process.sample.comfy.workflow_path = 'config/comfy_templates/krea2_lora_sample.json.njk';
    assert.match(caps.validateTrainingCapabilities(job).join('\n'), /Krea 2 ComfyUI workflows are incompatible/);
  }
});

test('all specialized MOSS modes and aliases accept ComfyUI previews while retaining workflow validation', () => {
  for (const mode of [...modes, ...Object.keys(caps.trainingModeAliases)]) {
    const job = config('moss_image:base', caps.canonicalTrainingMode(mode));
    const process = job.config.process[0];
    process.type = mode;
    for (const workflow of ['config/comfy_templates/moss_image_lora_sample.json.njk', 'custom/moss_sample.json']) {
      process.sample.comfy = { enabled: true, workflow_path: workflow };
      assert.deepEqual(caps.validateTrainingCapabilities(job), [], `${mode}: ${workflow}`);
    }
    process.sample.comfy.workflow_path = 'config/comfy_templates/krea2_lora_sample.json.njk';
    assert.match(caps.validateTrainingCapabilities(job).join('\n'), /Krea 2 ComfyUI workflows are incompatible/, mode);
  }
});

test('capability matrix agrees with Python and supports all objectives without duplicate legacy choices', () => {
  const backend = fs.readFileSync(path.resolve(__dirname, '../../toolkit/training_capabilities.py'), 'utf8');
  const declarations = [...backend.matchAll(/'([^']+)': FlowTrainingCapability\('([^']+)', (\d+)([^\n]*)/g)];
  assert.equal(declarations.length, 6);
  for (const [, arch, label, bucket, options] of declarations) {
    assert.deepEqual(caps.flowTrainingModels[arch], { label, bucketDivisibility: Number(bucket), editReferences: options.includes('edit_references=True') });
    for (const mode of modes) {
      assert.deepEqual(caps.validateTrainingCapabilities(config(arch, mode)), []);
      assert.equal(caps.trainingModeVisible(arch, mode, 'diffusion_trainer'), true);
      assert.deepEqual(caps.validateTrainingCapabilities(config(arch, mode, 'dora')).length, mode.startsWith('fizgig_') ? 0 : 1);
    }
  }
  assert.equal(caps.trainingModeVisible('flux', 'sliderspace', 'diffusion_trainer'), false);
  assert.equal(caps.trainingModeVisible('flux', 'sliderspace', 'sliderspace'), true);
  for (const [alias, canonical] of Object.entries(caps.trainingModeAliases)) {
    assert.equal(caps.canonicalTrainingMode(alias), canonical);
    assert.equal(caps.trainingModeVisible('anima', alias, 'diffusion_trainer'), false);
    assert.equal(caps.trainingModeVisible('anima', alias, alias), true);
    assert.deepEqual(caps.validateTrainingCapabilities(config('anima', alias)), []);
  }
});

test('all Qwen modes and aliases allow frozen helpers; other model restrictions remain', () => {
  for (const arch of Object.keys(caps.flowTrainingModels)) {
    for (const mode of [...modes, ...Object.keys(caps.trainingModeAliases)]) {
      const job = config(arch, caps.canonicalTrainingMode(mode));
      job.config.process[0].type = mode;
      job.config.process[0].model.assistant_lora_path = '/helper.safetensors';
      const errors = caps.validateTrainingCapabilities(job);
      if (['qwen_image_2', 'fibo2'].includes(arch)) assert.deepEqual(errors, [], mode);
      else assert.match(errors.join('\n'), /auxiliary/, `${arch}: ${mode}`);
    }
  }
});

test('specialized modes allow preview-only inference LoRAs and identify rejected training adapters', () => {
  for (const arch of Object.keys(caps.flowTrainingModels)) {
    for (const mode of [...modes, ...Object.keys(caps.trainingModeAliases)]) {
      const job = config(arch, caps.canonicalTrainingMode(mode));
      job.config.process[0].type = mode;
      job.config.process[0].model.inference_lora_path = '/models/preview.safetensors';
      assert.deepEqual(caps.validateTrainingCapabilities(job), [], `${arch}: ${mode}`);
    }
  }
  for (const mode of [...modes, ...Object.keys(caps.trainingModeAliases)]) {
    const job = config('krea2', caps.canonicalTrainingMode(mode));
    const process = job.config.process[0];
    process.type = mode;
    process.model.inference_lora_path = '/models/raw_to_turbo.safetensors';
    assert.deepEqual(caps.validateTrainingCapabilities(job), [], mode);
    for (const key of ['assistant_lora_path', 'unconditional_lora_path']) {
      process.model[key] = '/training-adapter.safetensors';
      assert.match(caps.validateTrainingCapabilities(job).join('\n'), new RegExp(`model\\.${key}`));
      delete process.model[key];
    }
  }
});

test('Ideogram default reference disables negatives without deleting drafts; negative-only validation is explicit', () => {
  const job = config('ideogram4', 'guidance_distillation');
  const process = job.config.process[0];
  process.guidance_distillation = { objective: 'negative_only', negative_prompt: 'haze' };
  assert.equal(caps.cfgNegativeTextEnabled(process.model), false);
  assert.match(caps.validateTrainingCapabilities(job).join('\n'), /text-negative CFG/);
  process.model.model_kwargs.ideogram_cfg_reference = 'negative_prompt';
  assert.equal(caps.cfgNegativeTextEnabled(process.model), true);
  assert.deepEqual(caps.validateTrainingCapabilities(job), []);
  assert.equal(process.guidance_distillation.negative_prompt, 'haze');
  process.guidance_distillation.negative_prompt = ' ';
  assert.match(caps.validateTrainingCapabilities(job).join('\n'), /nonempty/);
});

test('paired targets are not edit references and unsupported actual edit fields are rejected', () => {
  for (const arch of Object.keys(caps.flowTrainingModels)) {
    const job = config(arch, 'flow_dpo');
    const process = job.config.process[0];
    process.datasets = [{ folder_path: '/preferred', control_path_1: '/rejected' }];
    assert.deepEqual(caps.validateTrainingCapabilities(job), []);
    process.datasets[0].control_path_2 = '/source';
    if (arch === 'krea2' || arch === 'moss_image') process.model.model_kwargs.edit = true;
    assert.equal(caps.validateTrainingCapabilities(job).length, ['krea2', 'qwen_image_2', 'moss_image', 'fibo2'].includes(arch) ? 0 : 1);
  }
});

test('MOSS variants normalize correctly and architecture switching retains shared settings', () => {
  const variants = {
    moss_image: [false, false],
    'moss_image:base': [false, false],
    'moss_image:edit': [true, true],
    'moss_image:base_edit': [true, true],
  };
  for (const [variant, [edit, kvCache]] of Object.entries(variants)) {
    const effective = caps.effectiveFlowModel({ arch: variant, model_kwargs: { custom: true } });
    assert.equal(effective.arch, 'moss_image');
    assert.equal(effective.model_kwargs.edit ?? false, edit);
    assert.equal(effective.model_kwargs.kv_cache ?? false, kvCache);
    assert.deepEqual(effective.model_kwargs.uncond_skip_layers, [22]);
    assert.equal(caps.flowTrainingModelOptions.some(option => option.value === variant), true);

    for (const mode of modes) {
      const job = config(variant, mode);
      const errors = caps.validateTrainingCapabilities(job);
      assert.equal(errors.length === 0, !edit || ['flow_dpo', 'guidance_distillation'].includes(mode), `${variant}: ${mode}: ${errors}`);
    }
  }

  const editJob = config('moss_image:edit', 'flow_dpo');
  editJob.config.process[0].datasets = [{ folder_path: '/preferred', control_path_1: '/rejected', control_path_2: '/source' }];
  assert.deepEqual(caps.validateTrainingCapabilities(editJob), []);
  const { config: switched } = caps.switchSpecializedArchitecture(editJob, 'moss_image:base_edit');
  const process = switched.config.process[0];
  assert.equal(process.model.arch, 'moss_image:base_edit');
  assert.equal(process.model.model_kwargs.edit, true);
  assert.equal(process.model.model_kwargs.kv_cache, true);
  assert.equal(process.model.name_or_path, '');
  assert.equal(process.datasets[0].control_path_1, '/rejected');
  assert.equal(process.datasets[0].control_path_2, undefined);
});

test('MOSS form defaults keep negative CFG block skip and omit LoRA block-22 exclusions', () => {
  const modelUi = load('../../extensions_built_in/diffusion_models/ui.tsx', {
    'next/link': { __esModule: true, default: () => null },
    '@/app/jobs/new/options': caps,
    '@/helpers/defaultSamples': {
      defaultComfySampleConfig: { workflow_path: 'workflow.json' },
      defaultSampleConfig: {},
      defaultIdeogramSamplesConfig: {},
      defaultMinimaxH3ImageSampleConfig: load('../src/helpers/defaultSamples.ts').defaultMinimaxH3ImageSampleConfig,
    },
    react: { __esModule: true, default: { Fragment: Symbol.for('react.fragment') } },
    'react/jsx-runtime': {
      jsx: () => null,
      jsxs: () => null,
      Fragment: Symbol.for('react.fragment'),
    },
  }, true);
  const variants = ['moss_image', 'moss_image:base', 'moss_image:edit', 'moss_image:base_edit'];
  const entries = modelUi.AI_TOOLKIT_UI_MODELS.filter(model => variants.includes(model.name));
  assert.deepEqual(entries.map(model => model.name).sort(), [...variants].sort());
  for (const entry of entries) {
    assert.equal(entry.defaults['config.process[0].network.network_kwargs.ignore_if_contains'], undefined,
      `${entry.name} should not set LoRA ignore filters`);
    assert.deepEqual(entry.defaults['config.process[0].model.model_kwargs'][0].uncond_skip_layers, [22]);
  }
});

test('switching MOSS architecture does not inject LoRA exclusions and preserves explicit user filters', () => {
  const variants = ['moss_image', 'moss_image:base', 'moss_image:edit', 'moss_image:base_edit'];
  for (const arch of variants) {
    const job = config('moss_image', 'flow_dpo');
    const process = job.config.process[0];
    process.network.network_kwargs = { alpha: 3 };
    const { config: switched } = caps.switchSpecializedArchitecture(job, arch);
    assert.deepEqual(switched.config.process[0].network.network_kwargs, { alpha: 3 }, arch);
    assert.deepEqual(switched.config.process[0].model.model_kwargs.uncond_skip_layers, [22], arch);

    process.network.network_kwargs.ignore_if_contains = ['layers.22.', 'user.custom'];
    const { config: explicit } = caps.switchSpecializedArchitecture(job, arch);
    assert.deepEqual(explicit.config.process[0].network.network_kwargs.ignore_if_contains,
      ['layers.22.', 'user.custom'], arch);
  }
});

test('architecture switching preserves pairs, prompts and memory options but never swaps in a checkpoint', () => {
  const job = config('qwen_image_2', 'fizgig_image_slider', 'dora');
  const process = job.config.process[0];
  process.model = { ...process.model, quantize: true, qtype: 'convrotint4', low_vram: true, layer_offloading: true,
    text_encoder_path: '/qwen/te', vae_path: '/qwen/vae', inference_lora_path: '/preview.safetensors', model_kwargs: { kv_cache: true } };
  process.datasets = [{ folder_path: '/positive', control_path_1: '/negative', anchor_path: '/anchors', control_path_2: '/source' }];
  process.fizgig_slider = { positive_prefix: 'large', negative_prefix: 'small', anchor_prompts: [{ prompt: 'dog' }] };
  const before = structuredClone(job);
  const { config: next, notice } = caps.switchSpecializedArchitecture(job, 'anima');
  assert.deepEqual(job, before);
  const updated = next.config.process[0];
  assert.equal(updated.model.name_or_path, '');
  for (const key of ['quantize', 'qtype', 'low_vram', 'layer_offloading']) assert.equal(updated.model[key], process.model[key]);
  assert.deepEqual(updated.network, process.network);
  assert.deepEqual(updated.fizgig_slider, process.fizgig_slider);
  assert.equal(updated.datasets[0].control_path_1, '/negative');
  assert.equal(updated.datasets[0].anchor_path, '/anchors');
  assert.equal(updated.datasets[0].control_path_2, undefined);
  assert.equal(updated.model.text_encoder_path, undefined);
  assert.equal(updated.model.inference_lora_path, undefined);
  assert.match(notice, /cleared/);
});

test('ordinary switch to FIBO2 retains user choices and prompts while clearing architecture-bound sources', () => {
  const utils = load('../src/app/jobs/new/utils.ts', {
    '@/utils/basic': { objectCopy: value => structuredClone(value) },
    '@/helpers/datasetControls': { getDatasetControlPaths: () => [] },
    './trainingCapabilities': caps,
  });
  const sample = { prompt: 'keep prompt', negative_prompt: 'keep negative', seed: 123,
    network_multiplier: -0.5, ctrl_img: '/old/ref.png', ctrl_img_1: '/old/ref1.png' };
  const job = { config: { process: [{ type: 'diffusion_trainer',
    model: { arch: 'legacy', model_kwargs: { legacy_option: true }, quantize: true, qtype: 'int8', cache_quantized_models: false,
      low_vram: true, layer_offloading: true, text_encoder_path: '/old/te', vae_path: '/old/vae',
      inference_lora_path: '/old/inference.safetensors' },
    network: { type: 'lora', linear: 16, linear_alpha: 8 },
    train: { cache_text_embeddings: true, noise_scheduler: 'ddpm' }, datasets: [{ folder_path: '/data', control_path_1: '/old/control' }],
    sample: { samples: [sample], comfy: { inference_lora: '/old/comfy.safetensors',
      inference_lora_path: '/old/comfy.safetensors', inference_lora_enabled: true } } }] } };
  const before = structuredClone(job);
  const modelArchs = [
    { name: 'legacy', defaults: { 'config.process[0].model.model_kwargs.legacy_option': [true, false] }, additionalSections: [] },
    { name: 'fibo2', defaults: {
      'config.process[0].model.model_kwargs.fibo_default': [true, false],
      'config.process[0].train.noise_scheduler': ['flowmatch', 'ddpm'],
      'config.process[0].train.timestep_type': ['shift', 'linear'],
    }, additionalSections: [] },
  ];
  function set(value, dottedPath) {
    const parts = dottedPath.replaceAll('[0]', '.0').split('.');
    let target = job;
    for (const key of parts.slice(0, -1)) target = target[key];
    target[parts.at(-1)] = value;
  }
  utils.handleModelArchChange(modelArchs, 'legacy', 'fibo2', job, set, { preserveTrainingChoices: true });
  const updated = job.config.process[0];
  assert.notDeepEqual(job, before);
  assert.equal(updated.network.type, 'lora');
  assert.equal(updated.network.linear, 16);
  assert.equal(updated.network.linear_alpha, 8);
  assert.equal(updated.network.pretrained_lora_path, undefined);
  assert.deepEqual(updated.model.quantize, true);
  assert.equal(updated.model.qtype, 'int8');
  assert.equal(updated.model.cache_quantized_models, false);
  assert.equal(updated.model.low_vram, true);
  assert.equal(updated.model.layer_offloading, true);
  assert.equal(updated.model.name_or_path, '');
  for (const field of ['text_encoder_path', 'te_name_or_path', 'vae_path', 'inference_lora_path']) {
    assert.equal(updated.model[field], undefined, field);
  }
  assert.deepEqual(updated.model.model_kwargs, { fibo_default: true });
  assert.equal(updated.train.cache_text_embeddings, true);
  assert.equal(updated.train.noise_scheduler, 'flowmatch');
  assert.equal(updated.train.timestep_type, 'shift');
  assert.equal(updated.datasets[0].control_path_1, undefined);
  assert.deepEqual(updated.sample.samples[0], { prompt: 'keep prompt', negative_prompt: 'keep negative',
    seed: 123, network_multiplier: -0.5 });
  assert.equal(updated.sample.comfy.inference_lora, '');
  for (const field of ['inference_lora_path', 'inference_lora_enabled']) {
    assert.equal(updated.sample.comfy[field], undefined, field);
  }

  const baseline = structuredClone(before);
  const setBaseline = (value, dottedPath) => {
    const parts = dottedPath.replaceAll('[0]', '.0').split('.');
    let target = baseline;
    for (const key of parts.slice(0, -1)) target = target[key];
    target[parts.at(-1)] = value;
  };
  utils.handleModelArchChange(modelArchs, 'legacy', 'fibo2', baseline, setBaseline);
  assert.equal(baseline.config.process[0].model.low_vram, false);
  assert.equal(baseline.config.process[0].model.quantize, true);
});

test('FIBO2 Edit Turbo is selectable as a preview workflow while Dev remains the required training base', () => {
  const workflows = caps;
  const form = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  const turboPath = 'config/comfy_templates/fibo2_edit_turbo_lora_sample.json.njk';
  assert.deepEqual(workflows.fibo2WorkflowPaths, [
    'config/comfy_templates/fibo2_lora_sample.json.njk',
    turboPath,
  ]);
  assert.match(form, /FIBO2 Edit Turbo LoRA image \/ edit/);

  const job = config('fibo2', 'diffusion_trainer');
  const process = job.config.process[0];
  process.model.name_or_path = '/models/fibo2_dev_int8_convrot.safetensors';
  process.model.vae_path = '/models/flux2-vae.safetensors';
  process.sample = {
    samples: [{ prompt: 'keep positive', negative_prompt: 'keep negative', seed: 73, width: 768, height: 1024 }],
    sample_steps: 9,
    guidance_scale: 2.5,
    comfy: {
      enabled: true,
      workflow_path: 'config/comfy_templates/fibo2_lora_sample.json.njk',
      model: 'old-dev.safetensors',
      text_encoder: 'old-encoder.safetensors',
      vae: 'old-vae.safetensors',
      sampler: 'dpmpp_2m',
      scheduler: 'karras',
    },
  };
  const before = structuredClone(job);
  const selected = workflows.applyComfyWorkflowSelection(job, turboPath);
  assert.deepEqual(job, before, 'selecting a preview must not mutate the source config');
  const updated = selected.config.process[0];
  assert.equal(updated.model.name_or_path, '/models/fibo2_dev_int8_convrot.safetensors');
  assert.equal(updated.sample.comfy.workflow_path, turboPath);
  assert.equal(updated.sample.comfy.model, 'd/fibo2_edit_turbo_int8_convrot.safetensors');
  assert.equal(updated.sample.comfy.text_encoder, 'd/qwen3vl_4b_instruct_heretic_7refusal_int8_convrot.safetensors');
  assert.equal(updated.sample.comfy.vae, 'flux2-vae.safetensors');
  assert.equal(updated.sample.comfy.sampler, 'euler');
  assert.equal(updated.sample.comfy.scheduler, 'simple');
  assert.equal(updated.sample.sample_steps, 4);
  assert.equal(updated.sample.guidance_scale, 1);
  assert.deepEqual(updated.sample.samples, before.config.process[0].sample.samples);
  assert.deepEqual(caps.validateTrainingCapabilities(selected), []);

  const unrelated = structuredClone(selected);
  unrelated.config.process[0].sample.comfy.workflow_path = 'config/comfy_templates/krea2_lora_sample.json.njk';
  assert.match(caps.validateTrainingCapabilities(unrelated).join('\n'), /FIBO2 Comfy previews require/);
  const turboTrainingBase = structuredClone(selected);
  turboTrainingBase.config.process[0].model.name_or_path = '/models/fibo2_edit_turbo_int8_convrot.safetensors';
  assert.match(caps.validateTrainingCapabilities(turboTrainingBase).join('\n'), /requires the non-distilled Dev checkpoint/);

  const switchedBack = workflows.applyComfyWorkflowSelection(selected, 'config/comfy_templates/fibo2_lora_sample.json.njk');
  assert.equal(switchedBack.config.process[0].sample.comfy.workflow_path, 'config/comfy_templates/fibo2_lora_sample.json.njk');
  assert.equal(switchedBack.config.process[0].model.name_or_path, '/models/fibo2_dev_int8_convrot.safetensors');
  assert.equal(switchedBack.config.process[0].sample.samples[0].negative_prompt, 'keep negative');
});

test('imports and both selectors/save paths use shared checks; server protects ordinary and sample-only paths', () => {
  const page = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/page.tsx'), 'utf8');
  const form = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  const api = fs.readFileSync(path.resolve(__dirname, '../src/app/api/jobs/route.ts'), 'utf8');
  assert.ok(page.includes('validateTrainingCapabilities(parsed)'));
  assert.ok(page.includes('validateTrainingCapabilities(jobConfig)'));
  for (const source of [page, form]) assert.ok(source.includes('trainingModeVisible('));
  assert.ok(api.includes('if (!sample_only)'));
  assert.ok(api.includes('validateTrainingCapabilities(job_config)'));
  assert.ok(api.includes('mergeSampleOnlyJobConfig(existingConfig, job_config)'));
});

test('H3 photo standalone differential guidance is accepted while conflicting objectives and invalid scales are rejected', () => {
  for (const arch of ['minimax_h3', 'minimax_h3_image']) {
    const job = config(arch, 'diffusion_trainer');
    const p = job.config.process[0];
    p.model.model_kwargs.h3_photo_training = true;
    p.train.do_differential_guidance = true;
    for (const scale of [1, 3]) {
      p.train.differential_guidance_scale = scale;
      assert.deepEqual(caps.validateTrainingCapabilities(job), [], `${arch} scale ${scale}`);
    }
    for (const scale of [0, -1, NaN, Infinity, '3', true, null]) {
      p.train.differential_guidance_scale = scale;
      assert.match(caps.validateTrainingCapabilities(job).join('\n'), /Differential Guidance Scale/i);
    }
    p.train.differential_guidance_scale = 3;
    p.train.loss_type = 't0';
    assert.match(caps.validateTrainingCapabilities(job).join('\n'), /(?:MSE|mean.squared)/i);
    p.train.do_differential_guidance = false;
    assert.deepEqual(caps.validateTrainingCapabilities(job), []);
    p.train.loss_type = 'mse';
    p.train.do_differential_guidance = true;
    p.train.t0_loss_target = true;
    assert.match(caps.validateTrainingCapabilities(job).join('\n'), /(?:MSE|t0)/i);
    p.train.t0_loss_target = false;
    for (const key of ['do_cfg', 'do_random_cfg', 'do_guidance_loss', 'diff_output_preservation', 'blank_prompt_preservation']) {
      p.train[key] = true;
      assert.match(caps.validateTrainingCapabilities(job).join('\n'), /photo training.*(?:CFG|guidance|preservation)/i, key);
      p.train[key] = false;
    }
  }
});
