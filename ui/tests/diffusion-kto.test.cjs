const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const YAML = require('yaml');

function load(relativePath, overrides = {}) {
  const filename = path.resolve(__dirname, relativePath);
  const result = new Module(filename, module);
  result.filename = filename;
  result.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = result.require.bind(result);
  result.require = id => Object.hasOwn(overrides, id) ? overrides[id] :
    id === './trainingCapabilities' ? capabilities : originalRequire(id);
  result._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true, target: ts.ScriptTarget.ES2022 },
  }).outputText, filename);
  return result.exports;
}
const capabilities = load('../src/app/jobs/new/trainingCapabilities.ts');
const defaults = load('../src/app/jobs/new/jobConfig.ts', {
  '@/helpers/basic': { isMac: () => false },
  '@/helpers/defaultSamples': load('../src/helpers/defaultSamples.ts'),
});
const { jobTypeOptions } = load('../src/app/jobs/new/options.tsx', { './jobConfig': defaults });
const mode = jobTypeOptions.find(item => item.value === 'diffusion_kto');

function validJob(arch = 'qwen_image_2') {
  const job = structuredClone(defaults.defaultJobConfig);
  const process = job.config.process[0];
  process.type = 'diffusion_kto';
  process.model.arch = arch;
  process.model.name_or_path = '/models/base';
  process.model.model_kwargs = {};
  process.datasets[0].folder_path = '/images/liked';
  mode.onActivate(job);
  return job;
}

test('KTO activation on all four models fixes objectives but preserves ranks, memory and preview configuration', () => {
  for (const arch of Object.keys(capabilities.flowTrainingModels)) {
    const job = structuredClone(defaults.defaultJobConfig), process = job.config.process[0];
    process.type = mode.value;
    Object.assign(process.model, { arch, name_or_path: '/models/base', quantize: true,
      qtype: 'convrotint4', low_vram: true, layer_offloading: true, low_vram_layer_streaming: true });
    process.model.model_kwargs = {};
    process.network.type = 'dora';
    process.network.linear = 16;
    process.network.linear_alpha = 8;
    process.train.gradient_checkpointing = true;
    process.train.ema_config.use_ema = true;
    process.train.frequency_loss_type = 'notch';
    process.train.do_guidance_loss = true;
    process.datasets[0].folder_path = '/images/liked';
    process.datasets[0].control_path_1 = '/old/rejected';
    process.datasets[0].anchor_path = '/old/anchors';
    const model = structuredClone(process.model), sample = structuredClone(process.sample);
    mode.onActivate(job);
    assert.deepEqual(process.model, model);
    assert.deepEqual(process.sample, sample);
    assert.equal(process.network.type, 'lora');
    assert.equal(process.network.linear, 16);
    assert.equal(process.network.linear_alpha, 8);
    assert.equal(process.train.gradient_checkpointing, true);
    assert.equal(process.train.frequency_loss_type, 'none');
    assert.equal(process.train.do_guidance_loss, false);
    assert.equal(process.train.ema_config.use_ema, false);
    assert.equal(process.datasets[0].kto_label, 'liked');
    assert.equal(process.datasets[0].control_path_1, undefined);
    assert.equal(process.datasets[0].anchor_path, undefined);
    assert.deepEqual(capabilities.validateTrainingCapabilities(job), []);
    mode.onDeactivate(job);
    assert.equal(process.diffusion_kto, undefined);
    assert.equal(process.datasets[0].kto_label, undefined);
  }
});

test('activation from full fine-tuning creates an ordinary LoRA and dataset but never hides teacher conflicts', () => {
  const job = validJob();
  const process = job.config.process[0];
  delete process.network;
  process.datasets = [];
  mode.onActivate(job);
  assert.equal(process.network.type, 'lora');
  assert.equal(process.datasets.length, 1);
  process.datasets[0].folder_path = '/images/liked';
  assert.deepEqual(capabilities.validateTrainingCapabilities(job), []);
  process.network.pretrained_lora_path = '/old/lora';
  process.model.model_kwargs.edit = true;
  mode.onActivate(job);
  assert.equal(process.network.pretrained_lora_path, '/old/lora');
  assert.equal(process.model.model_kwargs.edit, true);
  assert.match(capabilities.validateTrainingCapabilities(job).join('\n'), /pretrained.*auxiliary/);
  assert.match(capabilities.validateTrainingCapabilities(job).join('\n'), /text-to-image/);
});

test('YAML import defaults preserve objective choices and never infer labels, pair independent folders or repair invalid settings', () => {
  const job = validJob();
  const process = job.config.process[0];
  process.diffusion_kto = { beta: 2, reference_estimator: 'score_window', score_window_size: 3 };
  process.train.gradient_accumulation = 3;
  process.datasets.push({ ...process.datasets[0], folder_path: '/other/independent', kto_label: 'disliked', resolution: [768] });
  const imported = defaults.migrateJobConfig(YAML.parse(YAML.stringify(job)));
  assert.deepEqual(imported.config.process[0].diffusion_kto, { ...capabilities.defaultDiffusionKTOConfig,
    beta: 2, reference_estimator: 'score_window', score_window_size: 3 });
  assert.deepEqual(capabilities.validateTrainingCapabilities(imported), []);
  delete imported.config.process[0].datasets[1].kto_label;
  defaults.migrateJobConfig(imported);
  assert.match(capabilities.validateTrainingCapabilities(imported).join('\n'), /Liked or Disliked/);
  imported.config.process[0].diffusion_kto = 'bad';
  defaults.migrateJobConfig(imported);
  assert.equal(imported.config.process[0].diffusion_kto, 'bad');
  assert.match(capabilities.validateTrainingCapabilities(imported).join('\n'), /must be an object/);
});

test('shared import/client/server validation rejects incomplete windows, labels and replay/objective conflicts', () => {
  const cases = [
    ['diffusion_kto', 'beta', 0], ['diffusion_kto', 'liked_weight', Infinity],
    ['diffusion_kto', 'reference_estimator', 'ema'], ['diffusion_kto', 'score_window_size', 1.5],
    ['train', 'cache_text_embeddings', false], ['train', 'gradient_accumulation_steps', 2],
    ['train', 'frequency_loss_type', 'notch'], ['network', 'dropout', 0.1],
    ['train', 'min_denoising_steps', -1],
    ['network', 'network_kwargs', { rank_dropout: .1 }],
    ['network', 'network_kwargs', { full_train_in_out: true }],
    ['model', 'name_or_path', '/models/qwen-turbo'],
  ];
  for (const [section, key, value] of cases) {
    const job = validJob();
    job.config.process[0][section][key] = value;
    assert.ok(capabilities.validateTrainingCapabilities(job).length, `${section}.${key}`);
  }
  const job = validJob(), process = job.config.process[0];
  process.diffusion_kto.reference_estimator = 'score_window';
  assert.match(capabilities.validateTrainingCapabilities(job).join('\n'), /Gradient Accumulation >= 4/);
  process.train.gradient_accumulation = 4;
  assert.deepEqual(capabilities.validateTrainingCapabilities(job), []);
  process.datasets[0].control_path_1 = '/paired';
  assert.match(capabilities.validateTrainingCapabilities(job).join('\n'), /paired\/edit\/reg/);
});

test('objective card renders both estimators and its controls update their persisted fields', () => {
  const fields = [], updates = [];
  const input = props => { fields.push(props); return React.createElement('label', null, props.label); };
  const { default: Editor } = load('../src/app/jobs/new/DiffusionKTOEditor.tsx', {
    '@/components/Card': props => React.createElement('section', null, props.title, props.children),
    '@/components/formInputs': { NumberInput: input, SelectInput: input },
  });
  const job = validJob();
  const props = { jobConfig: job, setJobConfig: (value, key) => updates.push({ value, key }) };
  let markup = renderToStaticMarkup(React.createElement(Editor, props));
  assert.match(markup, /single-image estimate/);
  assert.doesNotMatch(markup, /Score Window \(accumulation batches\)/);
  fields.find(field => field.label === 'Beta').onChange(0.5);
  assert.deepEqual(updates.pop(), { value: 0.5, key: 'config.process[0].diffusion_kto.beta' });
  job.config.process[0].diffusion_kto.reference_estimator = 'score_window';
  fields.length = 0;
  markup = renderToStaticMarkup(React.createElement(Editor, props));
  assert.match(markup, /Gradient Accumulation below to at least 4/);
  const window = fields.find(field => field.label === 'Score Window (accumulation batches)');
  assert.equal(window.min, 2);
  window.onChange(8);
  assert.deepEqual(updates.pop(), { value: 8, key: 'config.process[0].diffusion_kto.score_window_size' });
  const source = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  assert.match(source, /isDiffusionKTO && <SelectInput label="Feedback Label"/);
  assert.match(source, /isDiffusionKTO && <div className=\{sampleOnlyLockedClass\}/);
  assert.match(source, /if \(isDiffusionKTO\) newDataset.kto_label = 'liked'/);
});
