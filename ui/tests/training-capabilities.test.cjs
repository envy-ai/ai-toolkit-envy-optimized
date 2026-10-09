const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const test = require('node:test');

function load(file) {
  const filename = path.resolve(__dirname, file);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const result = new Module(filename, module);
  result._compile(compiled, filename);
  return result.exports;
}
const caps = load('../src/app/jobs/new/trainingCapabilities.ts');
const modes = caps.specializedTrainingModes;
function config(arch, type, network = 'lora') {
  const job = { config: { process: [{ type, model: { arch, name_or_path: '/models/base', model_kwargs: {} },
    network: { type: network, linear: 4 }, datasets: [] }] } };
  if (type === 'diffusion_kto') {
    job.config.process[0].train = { cache_text_embeddings: true, noise_scheduler: 'flowmatch' };
    job.config.process[0].datasets = [{ folder_path: '/liked', kto_label: 'liked', cache_latents_to_disk: true }];
  }
  return job;
}

test('capability matrix agrees with Python and supports all objectives without duplicate legacy choices', () => {
  const backend = fs.readFileSync(path.resolve(__dirname, '../../toolkit/training_capabilities.py'), 'utf8');
  const declarations = [...backend.matchAll(/'([^']+)': FlowTrainingCapability\('([^']+)', (\d+)([^\n]*)/g)];
  assert.equal(declarations.length, 4);
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
      if (arch === 'qwen_image_2') assert.deepEqual(errors, [], mode);
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
    if (arch === 'krea2') process.model.model_kwargs.edit = true;
    assert.equal(caps.validateTrainingCapabilities(job).length, ['krea2', 'qwen_image_2'].includes(arch) ? 0 : 1);
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
  assert.equal(updated.model.inference_lora_path, '/preview.safetensors');
  assert.match(notice, /cleared/);
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
