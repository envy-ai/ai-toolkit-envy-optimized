const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

function load(relativePath, overrides = {}, jsx = false) {
  const filename = path.resolve(__dirname, relativePath);
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

const datasets = load('../src/helpers/datasetControls.ts');
const capabilities = load('../src/app/jobs/new/trainingCapabilities.ts');
const editModels = load('../../extensions/krea2_edit/ui.tsx', {
  '@/helpers/defaultSamples': {
    defaultComfySampleConfig: { workflow_path: 'config/comfy_templates/krea2_lora_sample.json.njk' },
  },
  react: { __esModule: true, default: { Fragment: Symbol.for('react.fragment') } },
  'react/jsx-runtime': {
    jsx: () => null,
    jsxs: () => null,
    Fragment: Symbol.for('react.fragment'),
  },
}, true);
const editModel = editModels.AI_TOOLKIT_UI_MODELS[0];

function validJob() {
  return {
    config: { process: [{
      type: 'diffusion_trainer',
      model: { arch: 'krea2_edit', name_or_path: 'krea/Krea-2-Raw', model_kwargs: {} },
      train: { batch_size: 1, unload_text_encoder: false, disable_sampling: false },
      sample: { comfy: { enabled: true, workflow_path: 'config/comfy_templates/krea2_edit_lora_sample.json.njk' } },
      datasets: [{ control_path: ['/refs/scene'], flip_x: false, flip_y: false }],
    }] },
  };
}

test('Krea edit UI metadata keeps the existing Krea 2 architecture registered', () => {
  assert.equal(editModel.name, 'krea2_edit');
  assert.equal(editModel.datasetControlPathListMax, 2);
  assert.deepEqual(editModel.defaults['config.process[0].model.model_kwargs'], [
    { fit_refs: true, multi_ref: false, checkpoint_every_n: 1 }, {},
  ]);
  const builtInRegistry = fs.readFileSync(
    path.resolve(__dirname, '../../extensions_built_in/diffusion_models/ui.tsx'), 'utf8',
  );
  assert.match(builtInRegistry, /name:\s*"krea2"/);
});

test('ordered dataset reference editing preserves interior gaps and removes trailing blanks', () => {
  const dataset = { control_path: ['/scene'] };
  assert.deepEqual(datasets.setDatasetControlPath(dataset, 1, '/subject'), ['/scene', '/subject']);
  assert.deepEqual(datasets.setDatasetControlPath(dataset, 1, ''), ['/scene']);
  assert.deepEqual(datasets.setDatasetControlPath({ control_path: ['/scene'] }, 2, '/third'), [
    '/scene', '', '/third',
  ]);
});

test('Krea edit validation enforces batch, flip, reference count and Comfy sampling contract', () => {
  const config = validJob();
  assert.deepEqual(capabilities.validateKrea2Edit(config), []);

  const process = config.config.process[0];
  process.train.batch_size = 2;
  process.datasets[0].flip_x = true;
  process.datasets[0].control_path = ['/one', '/two', '/three'];
  process.sample.comfy.enabled = false;
  const errors = capabilities.validateKrea2Edit(config).join('\n');
  assert.match(errors, /Batch Size 1/);
  assert.match(errors, /flips off/);
  assert.match(errors, /at most two consecutive/);
  assert.match(errors, /Use ComfyUI Renderer/);

  process.train.disable_sampling = true;
  assert.doesNotMatch(capabilities.validateKrea2Edit(config).join('\n'), /Use ComfyUI Renderer/);
  process.train.disable_sampling = false;
  process.sample.comfy.enabled = true;
  process.sample.comfy.workflow_path = 'config/comfy_templates/krea2_lora_sample.json.njk';
  assert.match(capabilities.validateKrea2Edit(config).join('\n'), /Krea 2 Edit LoRA/);
});

test('switching to Krea edit keeps all pre-existing ordered references without truncation', () => {
  const modelArchs = [
    { name: 'legacy', defaults: {}, additionalSections: ['datasets.multi_control_paths'] },
    editModel,
  ];
  const config = {
    config: { process: [{
      model: { arch: 'legacy', model_kwargs: {} },
      network: {},
      train: {},
      datasets: [{ control_path_1: '/scene', control_path_2: '/subject', control_path_3: '/extra' }],
      sample: { samples: [], comfy: {} },
    }] },
  };
  function set(value, dottedPath) {
    const parts = dottedPath.replaceAll('[0]', '.0').split('.');
    let parent = config;
    for (const key of parts.slice(0, -1)) parent = parent[key];
    parent[parts.at(-1)] = value;
  }
  const utils = load('../src/app/jobs/new/utils.ts', {
    '@/utils/basic': { objectCopy: value => structuredClone(value) },
    '@/helpers/datasetControls': datasets,
    './trainingCapabilities': capabilities,
  });

  utils.handleModelArchChange(modelArchs, 'legacy', 'krea2_edit', config, set);

  assert.deepEqual(config.config.process[0].datasets[0].control_path,
    ['/scene', '/subject', '/extra']);
  assert.equal(config.config.process[0].model.arch, 'krea2_edit');
});
