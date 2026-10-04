const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

function loadTypescript(relativePath, overrides = {}) {
  const filename = path.resolve(__dirname, relativePath);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.React,
      esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const compiledModule = new Module(filename, module);
  compiledModule.filename = filename;
  compiledModule.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = compiledModule.require.bind(compiledModule);
  compiledModule.require = id => Object.hasOwn(overrides, id) ? overrides[id] :
    id === './trainingCapabilities' ? loadTypescript('../src/app/jobs/new/trainingCapabilities.ts') : originalRequire(id);
  compiledModule._compile(compiled, filename);
  return compiledModule.exports;
}

const defaults = loadTypescript('../src/app/jobs/new/jobConfig.ts', {
  '@/helpers/basic': { isMac: () => false },
  '@/helpers/defaultSamples': loadTypescript('../src/helpers/defaultSamples.ts'),
});
const { jobTypeOptions } = loadTypescript('../src/app/jobs/new/options.tsx', { './jobConfig': defaults });
const mode = jobTypeOptions.find(option => option.value === 'qwen_guidance_distillation');

test('guidance distillation activation creates a valid CFG-1 LoRA setup, retaining memory settings', () => {
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  process.type = mode.value;
  process.model.arch = 'qwen_image_2';
  process.model.assistant_lora_path = '/helper.safetensors';
  process.model.quantize = true;
  process.model.quantize_te = true;
  process.model.qtype = 'convrotint4';
  process.model.low_vram = true;
  process.model.layer_offloading = true;
  process.model.low_vram_layer_streaming = true;
  const beforeModel = structuredClone(process.model);
  process.train.gradient_checkpointing = true;
  process.train.ema_config.use_ema = true;
  process.train.frequency_loss_type = 'notch';
  process.train.do_guidance_loss = true;
  process.sample.guidance_scale = 4;
  process.sample.neg = 'noise, haze';
  process.sample.samples = [{ prompt: 'a cat', neg: 'blur', guidance_scale: 5, network_multiplier: -1 }];
  process.network.type = 'dora';
  mode.onActivate(config);
  assert.deepEqual(process.guidance_distillation, {
    teacher_cfg_scale: 4, negative_prompt: 'noise, haze', objective: 'full_guidance',
  });
  assert.equal(process.network.type, 'lora');
  assert.equal(process.train.cache_text_embeddings, true);
  assert.equal(process.train.unload_text_encoder, false);
  assert.equal(process.train.gradient_checkpointing, true);
  assert.equal(process.train.timestep_type, 'shift');
  assert.equal(process.train.frequency_loss_type, 'none');
  assert.equal(process.train.ema_config.use_ema, false);
  assert.equal(process.train.do_guidance_loss, false);
  assert.deepEqual(process.model, beforeModel);
  assert.equal(process.datasets[0].cache_latents_to_disk, true);
  assert.equal(process.datasets[0].caption_dropout_rate, 0);
  assert.equal(process.datasets[0].network_weight, 1);
  assert.equal(process.sample.guidance_scale, 1);
  assert.equal(process.sample.neg, '');
  assert.deepEqual(process.sample.samples, [{ prompt: 'a cat', neg: '', guidance_scale: 1, network_multiplier: 1 }]);
  mode.onDeactivate(config);
  assert.equal(process.guidance_distillation, undefined);
});

test('activation from full finetuning creates a LoRA and ordinary dataset, transfers Comfy negative', () => {
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  delete process.network;
  process.datasets = [];
  process.sample.comfy = { enabled: true, negative_prompt: 'artifacts' };
  mode.onActivate(config);
  assert.equal(process.network.type, 'lora');
  assert.equal(process.datasets.length, 1);
  assert.equal(process.datasets[0].cache_latents_to_disk, true);
  assert.equal(process.guidance_distillation.negative_prompt, 'artifacts');
  assert.equal(process.sample.comfy.negative_prompt, '');
});

test('import defaults preserve teacher settings and do not reset sample choices', () => {
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  process.type = mode.value;
  process.guidance_distillation = { teacher_cfg_scale: 3, negative_prompt: 'noise' };
  process.sample.guidance_scale = 2;
  defaults.migrateJobConfig(config);
  assert.deepEqual(process.guidance_distillation, { teacher_cfg_scale: 3, negative_prompt: 'noise', objective: 'full_guidance' });
  assert.equal(process.sample.guidance_scale, 2);
});

test('shared capability selector and all teacher controls are present on the form', () => {
  const form = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  const page = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/page.tsx'), 'utf8');
  assert.ok(page.includes('trainingModeVisible('));
  for (const field of ['teacher_cfg_scale', 'negative_prompt', 'objective']) {
    assert.ok(form.includes(`guidance_distillation.${field}`));
  }
  assert.ok(form.includes('Cache Latents (required)'));
  assert.ok(form.includes('Teacher Negative Prompt'));
});

test('Fizgig slider anchors start optional, and image selection leaves memory settings intact', () => {
  for (const value of ['fizgig_prompt_slider', 'fizgig_image_slider']) {
    const config = structuredClone(defaults.defaultJobConfig);
    const process = config.config.process[0];
    const slider = jobTypeOptions.find(option => option.value === value);
    process.model.arch = 'qwen_image_2';
    process.datasets[0].anchor_path = '/datasets/dogs';
    const beforeModel = structuredClone(process.model);
    slider.onActivate(config);
    assert.deepEqual(process.fizgig_slider.anchor_prompts, []);
    assert.equal(process.fizgig_slider.preservation_weight, 1);
    assert.equal(process.datasets[0].anchor_path, '/datasets/dogs');
    assert.deepEqual(process.model, beforeModel);
  }
  const form = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  assert.ok(form.includes('Add Anchor Prompt'));
  assert.ok(form.includes('Remove Anchor'));
  assert.ok(form.includes('label="Anchor Negative Prompt (optional)"'));
  const negativeField = form.slice(form.indexOf('label="Anchor Negative Prompt (optional)"'), form.indexOf('label="Anchor Negative Prompt (optional)"') + 230);
  assert.ok(negativeField.includes('disabled={!sliderCfgEnabled}'));
  const legacyNegative = form.indexOf('Control Dataset 1 (−1 Images)');
  assert.ok(form.indexOf('label="Anchor Images (optional)"', legacyNegative) > legacyNegative);
  const pointFolders = form.indexOf('label="Image Folder"');
  assert.ok(form.indexOf('label="Anchor Images (optional)"', pointFolders) > pointFolders);
  assert.ok(form.includes('anchor_prompts: promptSet.anchor_prompts ?? []'));
});
