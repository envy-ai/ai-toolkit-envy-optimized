const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const test = require('node:test');
function load(file, overrides = {}) {
  const filename = path.resolve(__dirname, file), mod = new Module(filename, module);
  mod.filename = filename;
  mod.paths = [path.resolve(__dirname, '../node_modules'), ...Module._nodeModulePaths(path.dirname(filename))];
  const req = mod.require.bind(mod);
  mod.require = id => Object.hasOwn(overrides, id) ? overrides[id] : req(id);
  mod._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText, filename);
  return mod.exports;
}
const defaults = load('../src/helpers/defaultSamples.ts');
const capabilities = load('../src/app/jobs/new/trainingCapabilities.ts');
const ui = load('../../extensions_built_in/diffusion_models/ui.tsx', { 'next/link': () => null, '@/helpers/defaultSamples': defaults, '@/app/jobs/new/options': capabilities });
function job(arch) {
  return { config: { process: [{ type: 'diffusion_trainer', model: { arch, name_or_path: '/my/model', low_vram: true,
    model_kwargs: { partition: arch === 'minimax_h3_image' ? 'image' : 'fl2va_pruned' } }, network: { type: 'lora', network_kwargs: {} },
    train: { steps: 678, gradient_checkpointing: false, unload_text_encoder: true, ema_config: {} },
    datasets: [{ folder_path: '/my/photos', caption_ext: 'txt' }], sample: { samples: [{ prompt: 'cat', seed: 42 }], comfy: { enabled: false } } }] } };
}
test('official image selectable with explicit still recipe and safe save/load', () => {
  assert.ok(ui.AI_TOOLKIT_UI_MODELS.some(m => m.name === 'minimax_h3_image'));
  const input = job('minimax_h3_image'), before = structuredClone(input);
  assert.equal(ui.getMinimaxH3TrainingRecipe(input), 'video');
  const result = ui.applyMinimaxH3PhotoRecipe(input), p = result.config.process[0];
  assert.deepEqual(input, before);
  assert.equal(p.model.name_or_path, '/my/model'); assert.equal(p.model.low_vram, true);
  assert.equal(p.train.steps, 678); assert.equal(p.train.gradient_checkpointing, false); assert.equal(p.train.unload_text_encoder, true);
  assert.equal(p.datasets[0].folder_path, '/my/photos'); assert.deepEqual(p.datasets[0].resolution, [512]);
  assert.equal(p.sample.width, 2048); assert.equal(p.sample.height, 2048); assert.equal(p.sample.sample_steps, 6);
  assert.deepEqual(p.sample.samples, [{ prompt: 'cat', seed: 42 }]);
  assert.equal(ui.getMinimaxH3TrainingRecipe(JSON.parse(JSON.stringify(result))), 'photo');
  p.train.lr = 2e-6; assert.equal(ui.getMinimaxH3TrainingRecipe(result), 'custom_photo');
});
test('existing video recipe remains separate and prior job is not silently rewritten', () => {
  const input = job('minimax_h3'), before = structuredClone(input);
  assert.equal(ui.getMinimaxH3TrainingRecipe(input), 'video'); assert.deepEqual(input, before);
  const p = ui.applyMinimaxH3PhotoRecipe(input).config.process[0];
  assert.equal(p.model.arch, 'minimax_h3'); assert.equal(p.sample.width, 512); assert.equal(p.sample.sample_steps, 20);
});
test('effective network filters preserve legacy aliases and prefer populated nested values', () => {
  const p = { only_if_contains: ['legacy'], network: { network_kwargs: {} } };
  assert.deepEqual(capabilities.effectiveNetworkFilter(p, 'only_if_contains'), ['legacy']);
  p.network.network_kwargs.only_if_contains = [];
  assert.deepEqual(capabilities.effectiveNetworkFilter(p, 'only_if_contains'), ['legacy']);
  p.network.network_kwargs.only_if_contains = ['nested'];
  assert.deepEqual(capabilities.effectiveNetworkFilter(p, 'only_if_contains'), ['nested']);
});
test('H3 dataset cache controls preserve imports and inherit a common choice', () => {
  const entry = ui.AI_TOOLKIT_UI_MODELS.find(model => model.name === 'minimax_h3_image');
  const diskCache = entry.advancedModelOptions.find(option => option.label === 'Cache dataset text embeddings to disk');
  const quantCache = entry.advancedModelOptions.find(option => option.label === 'Cache quantized models');
  const config = job('minimax_h3_image');
  const process = config.config.process[0];
  const newDataset = { folder_path: '/new', cache_text_embeddings: false, caption_ext: 'txt' };
  for (const choice of [true, false]) {
    process.datasets = [{ folder_path: '/a', cache_text_embeddings: choice }, { folder_path: '/b', cache_text_embeddings: choice }];
    const before = structuredClone(config);
    assert.deepEqual(entry.initializeDataset(newDataset, config), { ...newDataset, cache_text_embeddings: choice });
    assert.equal(diskCache.getValue(config), choice);
    assert.deepEqual(config, before);
  }
  process.datasets = [{ folder_path: '/a', cache_text_embeddings: true }, { folder_path: '/b', cache_text_embeddings: false }];
  const before = structuredClone(config), newBefore = structuredClone(newDataset);
  assert.deepEqual(entry.initializeDataset(newDataset, config), newBefore);
  assert.equal(diskCache.getValue(config), false);
  let result, key;
  diskCache.onChange(true, config, (value, path) => { result = value; key = path; });
  assert.equal(key, 'config.process[0].datasets');
  assert.deepEqual(result, process.datasets.map(dataset => ({ ...dataset, cache_text_embeddings: true })));
  assert.deepEqual(config, before); assert.deepEqual(newDataset, newBefore);
  assert.equal(quantCache.getValue(config), true);
  process.model.cache_quantized_models = false; assert.equal(quantCache.getValue(config), false);
  quantCache.onChange(true, config, (value, path) => { result = value; key = path; });
  assert.equal(result, true); assert.equal(key, 'config.process[0].model.cache_quantized_models');
});

test('photo block filter accepts only default or all 50 blocks for both architectures', () => {
  for (const arch of ['minimax_h3', 'minimax_h3_image']) {
    const result = ui.applyMinimaxH3PhotoRecipe(job(arch));
    const network = result.config.process[0].network.network_kwargs;
    for (const start of [20, 0]) {
      network.only_if_contains = Array.from({length: 50 - start}, (_, i) => `blocks.${start + i}.`);
      assert.deepEqual(capabilities.validateTrainingCapabilities(result), [], `${arch} from ${start}`);
    }
    for (const filters of [[], ['blocks.0.'], Array.from({length: 49}, (_, i) => `blocks.${i + 1}.`),
      [...Array.from({length: 30}, (_, i) => `blocks.${i + 20}.`), 'custom']]) {
      network.only_if_contains = filters;
      assert.match(capabilities.validateTrainingCapabilities(result).join('\n'), /Include modules/i, `${arch}: ${filters}`);
    }
    delete network.only_if_contains;
    assert.deepEqual(capabilities.validateTrainingCapabilities(result), []);
    network.only_if_contains = Array.from({length: 50}, (_, i) => `blocks.${i}.`);
    network.ignore_if_contains = ['custom'];
    assert.match(capabilities.validateTrainingCapabilities(result).join('\n'), /Exclude modules/i);
  }
});
