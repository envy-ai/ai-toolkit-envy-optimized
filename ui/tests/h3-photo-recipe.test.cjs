const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const test = require('node:test');

function load(file) {
  const filename = path.resolve(__dirname, file);
  const mod = new Module(filename, module);
  mod.filename = filename;
  mod.paths = [path.resolve(__dirname, '../node_modules'), ...Module._nodeModulePaths(path.dirname(filename))];
  const original = mod.require.bind(mod);
  mod.require = id => id === 'next/link' ? () => null :
    id === '@/helpers/defaultSamples' ? load('../src/helpers/defaultSamples.ts') :
    id === '@/app/jobs/new/options' ? load('../src/app/jobs/new/trainingCapabilities.ts') : original(id);
  mod._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true },
  }).outputText, filename);
  return mod.exports;
}
const { applyMinimaxH3PhotoRecipe: apply, getMinimaxH3TrainingRecipe: selection } = load('../../extensions_built_in/diffusion_models/ui.tsx');
function job() {
  return { config: { process: [{ type: 'diffusion_trainer', model: { arch: 'minimax_h3', name_or_path: '/model', low_vram: true, model_kwargs: { partition: 'fl2va_pruned' } },
    network: { type: 'lora', network_kwargs: {} }, train: { ema_config: {}, low_vram: true },
    datasets: [{ folder_path: '/photos', caption_ext: 'txt' }], sample: { samples: [{ prompt: 'hello' }], comfy: { enabled: true } } }] } };
}

test('explicit selection applies still-photo recipe without mutating saved job or memory settings', () => {
  const original = job();
  Object.assign(original.config.process[0].train, { unload_text_encoder: true, gradient_checkpointing: false, steps: 678 });
  Object.assign(original.config.process[0].sample.samples[0], { width: 900, sample_steps: 77, seed: 42 });
  const before = structuredClone(original);
  assert.equal(selection(original), 'video');
  const result = apply(original);
  assert.deepEqual(original, before);
  const p = result.config.process[0];
  assert.equal(p.model.model_kwargs.h3_photo_training, true);
  assert.match(p.model.assistant_lora_path, /circlestone-labs\/MiniMax-H3-Image-Training-Adapter/);
  assert.equal(p.model.low_vram, true);
  assert.equal(p.train.unload_text_encoder, true);
  assert.equal(p.train.gradient_checkpointing, false);
  assert.equal(p.train.steps, 678);
  assert.deepEqual(p.sample.samples[0], { prompt: 'hello', seed: 42 });
  assert.equal(p.datasets[0].folder_path, '/photos');
  assert.deepEqual(p.network.network_kwargs.only_if_contains, Array.from({ length: 30 }, (_, i) => `blocks.${i + 20}.`));
  assert.equal(p.train.batch_size, 1);
  assert.equal(p.sample.comfy.enabled, false);
  assert.equal(selection(JSON.parse(JSON.stringify(result))), 'photo');
  p.train.lr = 2e-6;
  assert.equal(selection(result), 'custom_photo');
});

test('incompatible jobs are rejected without mutation', () => {
  for (const mutate of [p => p.model.arch = 'moss_image', p => p.type = 'flow_dpo', p => p.network = null]) {
    const original = job(); mutate(original.config.process[0]);
    const before = structuredClone(original);
    assert.throws(() => apply(original), /ordinary MiniMax-H3 LoRA/);
    assert.deepEqual(original, before);
  }
});

test('photo transition clears conflicting network and dataset options and refuses Ref2VA', () => {
  const input = job();
  const p = input.config.process[0];
  p.model.model_kwargs.image_refs_as_video = true;
  p.network.all_layers = true;
  p.network.network_kwargs.full_if_contains = ['everything'];
  Object.assign(p.datasets[0], { batch_size: 2, h3_reference_path: '/ref', h3_v2v_path: '/video', control_from_same_folder: true, control_path: '/control' });
  const result = apply(input).config.process[0];
  assert.equal(result.model.model_kwargs.image_refs_as_video, false);
  assert.ok(!result.network.all_layers);
  assert.ok(!result.network.network_kwargs.full_if_contains);
  assert.ok(result.datasets[0].batch_size == null || result.datasets[0].batch_size === 1);
  for (const key of ['h3_reference_path', 'h3_v2v_path', 'control_from_same_folder', 'control_path']) assert.ok(!result.datasets[0][key]);
  p.model.model_kwargs.partition = 'ref2va';
  assert.throws(() => apply(input), /FL2VA|reference|Ref2VA/i);
});
