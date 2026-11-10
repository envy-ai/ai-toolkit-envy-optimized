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
const ui = load('../../extensions_built_in/diffusion_models/ui.tsx', { 'next/link': () => null, '@/helpers/defaultSamples': defaults });
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
