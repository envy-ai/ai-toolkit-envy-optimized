const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');
const YAML = require('yaml');

function loadTypescript(relativePath, overrides = {}) {
  const filename = path.resolve(__dirname, relativePath);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const result = new Module(filename, module);
  result.filename = filename;
  result.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = result.require.bind(result);
  result.require = id => Object.hasOwn(overrides, id) ? overrides[id] :
    id === './trainingCapabilities' ? loadTypescript('../src/app/jobs/new/trainingCapabilities.ts') : originalRequire(id);
  result._compile(compiled, filename);
  return result.exports;
}

const defaults = loadTypescript('../src/app/jobs/new/jobConfig.ts', {
  '@/helpers/basic': { isMac: () => false },
  '@/helpers/defaultSamples': loadTypescript('../src/helpers/defaultSamples.ts'),
});
const capabilities = loadTypescript('../src/app/jobs/new/trainingCapabilities.ts');
const helpers = loadTypescript('../src/app/jobs/new/sliderspace.ts', { './trainingCapabilities': capabilities });
const options = loadTypescript('../src/app/jobs/new/options.tsx', { './jobConfig': defaults });
const { jobTypeOptions } = options;
const mode = jobTypeOptions.find(option => option.value === 'sliderspace');
function validJob() {
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  process.type = 'sliderspace';
  process.model.arch = 'qwen_image_2';
  mode.onActivate(config);
  process.sliderspace.concept_prompts = ['a spaceship'];
  return config;
}

test('activation creates independent LoRA discovery setup and retains memory/model settings', () => {
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  process.type = 'sliderspace';
  process.model.arch = 'qwen_image_2';
  Object.assign(process.model, { name_or_path: '/models/qwen.safetensors', text_encoder_path: '/models/te',
    low_vram: true, layer_offloading: true, low_vram_layer_streaming: true, quantize: true, qtype: 'convrotint4' });
  const model = structuredClone(process.model);
  process.train.gradient_checkpointing = true;
  process.train.ema_config.use_ema = true;
  process.train.frequency_loss_type = 'notch';
  process.network.type = 'loha';
  process.sample.comfy.enabled = true;
  mode.onActivate(config);
  assert.deepEqual(process.model, model);
  assert.deepEqual(process.datasets, []);
  assert.equal(process.train.gradient_checkpointing, true);
  assert.equal(process.train.ema_config.use_ema, false);
  assert.equal(process.train.frequency_loss_type, 'none');
  assert.equal(process.train.steps, 1000);
  assert.equal(process.train.batch_size, 1);
  assert.equal(process.train.gradient_accumulation, 1);
  assert.equal(process.network.type, 'lora');
  assert.equal(process.network.linear, 4);
  assert.equal(process.save.save_format, 'safetensors');
  assert.equal(process.save.push_to_hub, false);
  assert.equal(process.train.cache_text_embeddings, false);
  assert.equal(process.train.unload_text_encoder, true);
  assert.equal(process.sample.comfy.enabled, true);
  assert.deepEqual(process.sample.samples, [{ prompt: '' }]);
  assert.equal(process.sample.walk_seed, false);
  assert.match(helpers.validateSliderSpace(config)[0], /concept prompt/);
  process.sliderspace.concept_prompts = ['a spaceship'];
  assert.deepEqual(helpers.validateSliderSpace(config), []);
});

test('activation preserves custom previews and mode switching restores normal datasets', () => {
  const config = validJob();
  const process = config.config.process[0];
  process.sample.samples = [{ prompt: 'my custom scene', seed: 7 }];
  delete process.network;
  mode.onActivate(config);
  assert.deepEqual(process.sample.samples, [{ prompt: 'my custom scene', seed: 7 }]);
  assert.equal(process.network.type, 'lora');
  mode.onDeactivate(config);
  assert.equal(process.sliderspace, undefined);
  assert.equal(process.datasets.length, 1);
  process.type = 'diffusion_trainer';
  assert.deepEqual(helpers.validateSliderSpace(config), []);
  assert.ok(mode.disableSections.includes('datasets'));
});

test('activation clears unsupported imported objectives and auxiliary adapters without touching memory controls', () => {
  const config = validJob();
  const process = config.config.process[0];
  const unsupported = ['do_prior_divergence', 'train_turbo', 'do_fft_loss', 'learnable_snr_gos',
    'diffusion_feature_extractor_path', 'inverted_mask_prior', 'correct_pred_norm', 'free_u', 'do_paramiter_swapping'];
  for (const key of unsupported) process.train[key] = true;
  process.train.gradient_accumulation_steps = 4;
  process.network.network_kwargs.full_train_in_out = true;
  process.adapter = { type: 'controlnet' };
  process.embedding = {};
  process.decorator = {};
  process.trigger_word = 'token';
  process.model.low_vram_layer_streaming = true;
  process.model.assistant_lora_path = '/helper.safetensors';
  process.model.inference_lora_path = '/preview.safetensors';
  process.model.unconditional_lora_path = '/unconditional.safetensors';
  process.train.gradient_checkpointing = true;
  mode.onActivate(config);
  process.sliderspace.concept_prompts = ['a spaceship'];
  assert.deepEqual(helpers.validateSliderSpace(config), []);
  for (const key of unsupported) assert.equal(process.train[key], undefined);
  for (const key of ['adapter', 'embedding', 'decorator']) assert.equal(process[key], undefined);
  assert.equal(process.model.low_vram_layer_streaming, true);
  assert.equal(process.model.assistant_lora_path, '/helper.safetensors');
  assert.equal(process.model.inference_lora_path, '/preview.safetensors');
  assert.equal(process.model.unconditional_lora_path, undefined);
  assert.equal(process.train.gradient_checkpointing, true);
});

test('SliderSpace activation retains preview-only inference LoRAs on all supported models', () => {
  for (const arch of Object.keys(capabilities.flowTrainingModels)) {
    const config = validJob();
    const process = config.config.process[0];
    process.model.arch = arch;
    process.model.name_or_path = '/models/base.safetensors';
    process.model.inference_lora_path = '/models/preview.safetensors';
    mode.onActivate(config);
    process.sliderspace.concept_prompts = ['a spaceship'];
    assert.equal(process.model.inference_lora_path, '/models/preview.safetensors');
    assert.deepEqual(helpers.validateSliderSpace(config), [], arch);
  }
});

test('Generate defaults retain inference adapters while excluding training adapters', () => {
  const result = options.getGenerateDefaults({
    name: 'krea2', label: 'Krea 2', group: 'image',
    defaults: {
      'config.process[0].model.inference_lora_path': ['/preview.safetensors'],
      'config.process[0].model.assistant_lora_path': ['/training.safetensors'],
      'config.process[0].model.unconditional_lora_path': ['/unconditional.safetensors'],
    },
  });
  assert.equal(result.model.inference_lora_path, '/preview.safetensors');
  assert.equal(result.model.assistant_lora_path, undefined);
  assert.equal(result.model.unconditional_lora_path, undefined);
});

test('concept synchronization follows automatic preview only and does not mutate previous state', () => {
  const config = validJob();
  config.config.process[0].sample.samples = [{ prompt: 'a spaceship' }];
  const next = helpers.updateSliderSpaceConcepts(config, ['a new spaceship', 'an alien ship']);
  assert.equal(next.config.process[0].sample.samples[0].prompt, 'a new spaceship');
  assert.equal(config.config.process[0].sample.samples[0].prompt, 'a spaceship');
  next.config.process[0].sample.samples[0].prompt = 'custom scene';
  assert.equal(helpers.updateSliderSpaceConcepts(next, ['a boat']).config.process[0].sample.samples[0].prompt, 'custom scene');
});

test('import and YAML roundtrip fill missing defaults without replacing explicit choices', () => {
  const config = validJob();
  config.config.process[0].sliderspace = { concept_prompts: ['a cat', 'a dog'], num_directions: 8, preview_direction: 7, preview_strength: -0.5 };
  const next = defaults.migrateJobConfig(YAML.parse(YAML.stringify(config)));
  assert.deepEqual(next.config.process[0].sliderspace, { ...defaults.defaultSliderSpaceConfig,
    discovery_buckets: false, concept_prompts: ['a cat', 'a dog'], num_directions: 8, preview_direction: 7, preview_strength: -0.5 });
  assert.deepEqual(helpers.validateSliderSpace(next), []);
});

test('central save validation rejects malformed numbers, PCA limits, unsupported model and objectives', () => {
  const cases = [
    [p => p.sliderspace.concept_prompts = ['a cat', '  '], /concept prompt/],
    [p => p.sliderspace.concept_prompts = 'a cat', /concept prompt/],
    [p => p.sliderspace.num_directions = 65, /1 to 64/],
    [p => p.sliderspace.num_directions = 1.5, /whole number/],
    [p => p.sliderspace.discovery_samples = 4, /at least 5/],
    [p => { p.sliderspace.concept_prompts = Array(10).fill('ship'); p.sliderspace.discovery_samples = 9; }, /at least 10/],
    [p => p.sliderspace.resolution = 144, /multiple of 32/],
    [p => p.sliderspace.discovery_steps = 0, /generation steps/],
    [p => p.sliderspace.cfg_scale = NaN, /CFG/],
    [p => p.sliderspace.seed = -1, /seed/],
    [p => p.sliderspace.feature_device = 'mps', /device/],
    [p => p.sliderspace.feature_encoder = '', /model name/],
    [p => p.sliderspace.loss_weight = Infinity, /loss weight/],
    [p => p.sliderspace.preview_direction = 5, /preview direction/],
    [p => p.sliderspace.preview_strength = Infinity, /finite/],
    [p => p.model.arch = 'qwen_image', /Qwen Image 2.1/],
    [p => p.network.type = 'loha', /LoRA network/],
    [p => p.network.pretrained_lora_path = '/lora.safetensors', /pretrained/],
    [p => p.network.network_kwargs.full_train_in_out = true, /shared input/],
    [p => p.train.timestep_type = 'sigmoid', /shifted flow/],
    [p => p.train.min_denoising_steps = 1001, /minimum timestep/],
    [p => p.train.gradient_accumulation_steps = 2, /accumulation steps/],
    [p => p.save.save_format = 'diffusers', /local safetensors/],
    [p => p.save.push_to_hub = true, /Hub upload/],
    [p => p.sliderspace.typo = 7, /unknown SliderSpace settings/],
    [p => p.train.steps = 3, /at least one step/],
    [p => p.train.gradient_accumulation = 2, /accumulation 1/],
    [p => p.train.ema_config.use_ema = true, /EMA/],
    [p => p.train.train_text_encoder = true, /transformer only/],
    [p => p.train.do_cfg = true, /semantic objective/],
    [p => p.train.validation_config = {}, /dataset validation/],
    [p => p.model.unconditional_lora_path = '/adapter', /auxiliary/],
    [p => p.datasets = [{}], /discovery image folders/],
    [p => p.datasets = {}, /discovery image folders/],
    [p => p.sliderspace.discovery_buckets = 'yes', /bucketing/],
    [p => p.train.free_u = true, /semantic objective/],
    [p => p.train.do_paramiter_swapping = true, /semantic objective/],
    [p => p.trigger_word = 'token', /trigger word/],
  ];
  for (const [mutate, message] of cases) {
    const config = validJob();
    mutate(config.config.process[0]);
    assert.match(helpers.validateSliderSpace(config).join('\n'), message);
  }
  const config = validJob();
  config.config.process[0].sliderspace.preview_strength = 0;
  assert.deepEqual(helpers.validateSliderSpace(config), []);
  const page = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/page.tsx'), 'utf8');
  assert.match(page, /const saveJob = async \(\) => \{[\s\S]*?validateSliderSpace\(jobConfig\)/);
});

test('malformed imported objects produce validation errors without throwing or resetting their values', () => {
  for (const [key, value] of [['sliderspace', null], ['sliderspace', []], ['sliderspace', 'oops'], ['train', null], ['model', []], ['save', false]]) {
    const config = validJob();
    config.config.process[0][key] = value;
    const migrated = defaults.migrateJobConfig(config);
    assert.deepEqual(migrated.config.process[0][key], value);
    assert.ok(helpers.validateSliderSpace(migrated).length > 0);
  }
  assert.ok(helpers.validateSliderSpace({}).length > 0);
  const config = validJob();
  config.config.process[0].train.steps = 0;
  assert.ok(helpers.validateSliderSpace(config).length > 0);
  assert.deepEqual(helpers.validateSliderSpacePreview(config), []);
});

test('sample-only preview merge rejects out-of-range choices and preserves discovery/training', () => {
  const existing = validJob();
  const incoming = structuredClone(existing);
  incoming.config.process[0].sliderspace.preview_direction = 3;
  incoming.config.process[0].sliderspace.preview_strength = -0.75;
  incoming.config.process[0].sliderspace.num_directions = 20;
  incoming.config.process[0].sliderspace.concept_prompts = ['changed'];
  incoming.config.process[0].train.steps = 99;
  helpers.mergeSliderSpacePreview(existing, incoming);
  assert.equal(existing.config.process[0].sliderspace.preview_direction, 3);
  assert.equal(existing.config.process[0].sliderspace.preview_strength, -0.75);
  assert.equal(existing.config.process[0].sliderspace.num_directions, 4);
  assert.deepEqual(existing.config.process[0].sliderspace.concept_prompts, ['a spaceship']);
  assert.equal(existing.config.process[0].train.steps, 1000);
  incoming.config.process[0].sliderspace.preview_direction = 20;
  assert.throws(() => helpers.mergeSliderSpacePreview(existing, incoming), /trained SliderSpace direction/);
});

test('gallery identity parses direction/strength without mistaking decimal strength for extension', () => {
  for (const [strength, label] of [['plus1', '+1'], ['minus0.5', '-0.5'], ['plus0', '0'], ['minus1e-05', '-0.00001']]) {
    const filename = `20261002120000__000000250_direction_02_strength_${strength}_3.webp`;
    assert.equal(helpers.sliderSpaceSampleLabel(`/samples/${filename}`), `Direction 2 · strength ${label}`);
    assert.deepEqual(helpers.sampleFilenameInfo(`C:\\samples\\${filename}`), { filename, step: 250, promptIdx: 3 });
  }
  assert.deepEqual(helpers.sampleFilenameInfo('20261002120000_000000100_2.png'), {
    filename: '20261002120000_000000100_2.png', step: 100, promptIdx: 2,
  });
  assert.deepEqual(helpers.sampleFilenameInfo('1763563000704__000004000_0.jpg'), {
    filename: '1763563000704__000004000_0.jpg', step: 4000, promptIdx: 0,
  });
  assert.equal(helpers.sliderSpaceSampleLabel('20261002120000_100_0.png'), null);
});

test('gallery groups changing preview counts using step, direction and strength, with independent image timestamps', () => {
  const paths = [
    '1000__000000100_direction_01_strength_plus1_0.png',
    '1100__000000100_direction_01_strength_plus1_1.png',
    '1200__000000100_direction_02_strength_minus0.5_0.png',
    '1300__000000100_direction_02_strength_minus0.5_1.png',
    '1400__000000100_direction_02_strength_minus0.5_2.png',
    '1500__000000100_direction_02_strength_minus0.5_0.png',
    '1600__000000200_direction_02_strength_plus0_0.png',
  ];
  const rows = helpers.groupSliderSpaceSamples(paths);
  assert.deepEqual(rows, [paths.slice(0, 2), paths.slice(2, 5), paths.slice(5, 6), paths.slice(6)]);
  assert.equal(helpers.adjacentSliderSpaceSample(rows, paths[0], 0, 1), paths[1]);
  assert.equal(helpers.adjacentSliderSpaceSample(rows, paths[1], 0, 1), null);
  assert.equal(helpers.adjacentSliderSpaceSample(rows, paths[1], 1, 0), paths[3]);
  assert.equal(helpers.adjacentSliderSpaceSample(rows, paths[3], 1, 0), null);
});

test('running-job merge whitelists only supported SliderSpace previews while preserving ordinary job behavior', () => {
  const { mergeSampleOnlyJobConfig } = loadTypescript('../src/helpers/sampleOnlyJobConfig.ts', {
    '@/app/jobs/new/sliderspace': helpers,
  });
  const existing = validJob();
  const incoming = structuredClone(existing);
  Object.assign(incoming.config.process[0].sliderspace, { concept_prompts: ['changed'], num_directions: 10, preview_direction: 3, preview_strength: -1 });
  Object.assign(incoming.config.process[0].train, { steps: 10, disable_sampling: true });
  incoming.config.process[0].model.inference_lora_path = '/preview.safetensors';
  incoming.config.process[0].save.sample_on_record_low = false;
  incoming.config.process[0].sample.samples = [{ prompt: 'preview', guidance_scale: 3 }];
  incoming.config.process[0].sliderspace.preview_auto = true;
  incoming.config.process[0].sliderspace.preview_auto_strengths = '-0.5, 0.5, 2';
  const result = mergeSampleOnlyJobConfig(existing, incoming).config.process[0];
  assert.equal(result.sliderspace.preview_direction, 3);
  assert.equal(result.sliderspace.preview_strength, -1);
  assert.equal(result.sliderspace.preview_auto, true);
  assert.equal(result.sliderspace.preview_auto_strengths, '-0.5, 0.5, 2');
  assert.equal(result.sliderspace.num_directions, 4);
  assert.deepEqual(result.sliderspace.concept_prompts, ['a spaceship']);
  assert.equal(result.train.steps, 1000);
  assert.equal(result.train.disable_sampling, true);
  assert.equal(result.model.inference_lora_path, '/preview.safetensors');
  assert.equal(result.save.sample_on_record_low, existing.config.process[0].save.sample_on_record_low);
  assert.deepEqual(result.sample.samples, [{ prompt: 'preview', guidance_scale: 3 }]);
  assert.equal(existing.config.process[0].sliderspace.preview_direction, 1);
  existing.config.process[0].type = 'diffusion_trainer';
  const ordinary = mergeSampleOnlyJobConfig(existing, incoming).config.process[0];
  assert.equal(ordinary.model.inference_lora_path, '/preview.safetensors');
  assert.equal(ordinary.save.sample_on_record_low, false);
});

const Input = props => React.createElement('label', null, props.label,
  React.createElement('input', { value: props.value, disabled: props.disabled, readOnly: true }));
const editor = loadTypescript('../src/app/jobs/new/SliderSpaceEditor.tsx', {
  '@/components/Card': ({ title, children }) => React.createElement('section', null, title, children),
  '@/components/formInputs': { Checkbox: Input, NumberInput: Input, SelectInput: Input, TextAreaInput: Input, TextInput: Input },
  './sliderspace': helpers,
  './trainingCapabilities': capabilities,
});

test('provided/both discovery validates folders, optional fallback and generated counts independently', () => {
  const config = validJob();
  const value = config.config.process[0].sliderspace;
  assert.equal(value.discovery_buckets, true);
  value.discovery_mode = 'provided';
  value.concept_prompts = [];
  value.discovery_samples = 0;
  assert.match(helpers.validateSliderSpace(config).join('\n'), /Add at least one discovery image folder/);
  value.discovery_datasets = [{ folder_path: '/images', default_caption: 'cat' }, { folder_path: '/other' }];
  assert.deepEqual(helpers.validateSliderSpace(config), []);
  value.discovery_datasets[1].folder_path = '';
  assert.match(helpers.validateSliderSpace(config).join('\n'), /folder 2 needs a folder path/);
  value.discovery_datasets[1].folder_path = '/other';
  value.discovery_mode = 'both';
  assert.match(helpers.validateSliderSpace(config).join('\n'), /concept prompt/);
  value.concept_prompts = ['a cat'];
  value.discovery_samples = 1; // Supplied images contribute the remainder at runtime.
  assert.deepEqual(helpers.validateSliderSpace(config), []);
  value.discovery_mode = 'generated';
  assert.match(helpers.validateSliderSpace(config).join('\n'), /at least 5/);
});

function walkElements(element, predicate) {
  if (!React.isValidElement(element)) return [];
  return [...(predicate(element) ? [element] : []),
    ...React.Children.toArray(element.props.children).flatMap(child => walkElements(child, predicate))];
}

test('discovery editor supports add/remove folders, mode preservation and bucket selection', () => {
  let config = validJob();
  const setJobConfig = (next, key) => {
    if (!key) config = next;
    else config.config.process[0].sliderspace[key.split('.').at(-1)] = next;
  };
  const tree = () => editor.default({ jobConfig: config, setJobConfig,
    datasetOptions: [{ value: '/images', label: 'My images' }] });
  const control = label => walkElements(tree(), el => el.props.label === label)[0];
  const button = text => walkElements(tree(), el => el.type === 'button' && el.props.children === text)[0];
  control('Discovery images').props.onChange('provided');
  assert.deepEqual(config.config.process[0].sliderspace.discovery_datasets, [{ folder_path: '', default_caption: '' }]);
  control('Dataset folder').props.onChange('/images');
  control('Default caption (optional)').props.onChange('an animal');
  assert.equal(control('Dataset folder').props.options[1].value, '/images');
  button('Add Discovery Image Folder').props.onClick();
  assert.equal(config.config.process[0].sliderspace.discovery_datasets.length, 2);
  const remove = walkElements(tree(), el => el.props['aria-label'] === 'Remove discovery image folder 2')[0];
  remove.props.onClick();
  control('Provided image sizing').props.onChange('square');
  assert.equal(config.config.process[0].sliderspace.discovery_buckets, false);
  control('Provided image sizing').props.onChange('buckets');
  control('Discovery images').props.onChange('generated');
  assert.equal(control('Provided image sizing'), undefined);
  control('Discovery images').props.onChange('both');
  assert.deepEqual(config.config.process[0].sliderspace.discovery_datasets, [{ folder_path: '/images', default_caption: 'an animal' }]);
  assert.equal(control('Provided image sizing').props.value, 'buckets');
  const markup = renderToStaticMarkup(React.createElement(editor.default, { jobConfig: config, setJobConfig }));
  assert.match(markup, /subfolders/);
  assert.match(markup, /resolution²/);
  assert.match(markup, /Generated images remain square/);
});

test('provided discovery survives YAML import and sample-only edits unchanged', () => {
  const config = validJob();
  Object.assign(config.config.process[0].sliderspace, { discovery_mode: 'provided', discovery_buckets: true,
    discovery_datasets: [{ folder_path: '/images', default_caption: 'scene' }], concept_prompts: [] });
  const imported = defaults.migrateJobConfig(YAML.parse(YAML.stringify(config)));
  assert.deepEqual(imported.config.process[0].sliderspace, config.config.process[0].sliderspace);
  const incoming = structuredClone(config);
  Object.assign(incoming.config.process[0].sliderspace, { discovery_mode: 'generated', discovery_buckets: false,
    discovery_datasets: [], preview_direction: 2 });
  helpers.mergeSliderSpacePreview(config, incoming);
  // The helper updates only preview keys on the existing config.
  assert.equal(config.config.process[0].sliderspace.discovery_mode, 'provided');
  assert.equal(config.config.process[0].sliderspace.discovery_buckets, true);
  assert.equal(config.config.process[0].sliderspace.discovery_datasets[0].folder_path, '/images');
  const markup = renderToStaticMarkup(React.createElement(editor.default, { jobConfig: config, setJobConfig() {}, disabled: true }));
  assert.match(markup, /fieldset disabled=""/);
  assert.match(markup, /Fallback concept prompt \(optional\)/);
  assert.doesNotMatch(markup, /Add Concept Prompt/);
  assert.doesNotMatch(markup, /Generated discovery images/);
});

test('discovery is locked in sample-only mode while preview controls remain editable', () => {
  const jobConfig = validJob();
  const markup = renderToStaticMarkup(React.createElement(editor.default, { jobConfig, setJobConfig() {}, disabled: true }));
  assert.match(markup, /fieldset disabled=""/);
  assert.match(markup, /Add Concept Prompt/);
  assert.match(markup, /Discovery settings/);
  assert.match(markup, /Feature device/);
  assert.match(markup, /Semantic loss weight/);
  assert.match(markup, /Discovery \/ training CFG/);
  assert.match(markup, /grid-cols-1 md:grid-cols-3/);
  const preview = renderToStaticMarkup(React.createElement(editor.SliderSpacePreview, { jobConfig, setJobConfig() {}, disabled: true }));
  assert.match(preview, /Preview direction/);
  assert.match(preview, /Preview strength/);
  assert.doesNotMatch(preview, /disabled/);
  assert.match(preview, /All directions are exported/);
});

test('preview controls serialize into SliderSpace settings and use one-based direction options', () => {
  const changes = [];
  const element = editor.SliderSpacePreview({ jobConfig: validJob(), setJobConfig: (...args) => changes.push(args) });
  const controls = ['Preview direction', 'Preview strength'].map(label => walkElements(element, el => el.props.label === label)[0]);
  assert.deepEqual(controls[0].props.options.map(item => item.value), ['1', '2', '3', '4']);
  controls[0].props.onChange('3');
  controls[1].props.onChange(-1);
  assert.deepEqual(changes, [[3, 'config.process[0].sliderspace.preview_direction'], [-1, 'config.process[0].sliderspace.preview_strength']]);
});

test('auto switch keeps one prompt, renders a single editor and persists in live preview edits', () => {
  let config = validJob();
  config.config.process[0].sample.samples = [{ prompt: 'a spaceship', seed: 123 }, { prompt: 'another scene' }];
  const changes = [];
  const setJobConfig = (next, key) => {
    changes.push([next, key]);
    if (!key) config = next;
  };
  const tree = () => editor.SliderSpacePreview({ jobConfig: config, setJobConfig });
  walkElements(tree(), el => el.props.label === 'Auto sampling')[0].props.onChange(true);
  assert.equal(config.config.process[0].sliderspace.preview_auto, true);
  assert.deepEqual(config.config.process[0].sample.samples, [{ prompt: 'a spaceship', seed: 123 }]);
  const markup = renderToStaticMarkup(tree());
  assert.match(markup, /Auto sample prompt/);
  assert.match(markup, /Auto sample strengths/);
  assert.match(markup, /9 images per sampling round/);
  assert.doesNotMatch(markup, /Preview direction|Preview strength/);
  assert.deepEqual(helpers.validateSliderSpacePreview(config), []);
  walkElements(tree(), el => el.props.label === 'Auto sample prompt')[0].props.onChange('a portrait');
  assert.deepEqual(changes.at(-1), [[{ prompt: 'a portrait', seed: 123 }], 'config.process[0].sample.samples']);
  const existing = validJob();
  helpers.mergeSliderSpacePreview(existing, config);
  assert.equal(existing.config.process[0].sliderspace.preview_auto, true);
  walkElements(tree(), el => el.props.label === 'Auto sample strengths')[0].props.onChange('-0.5, 0.5, 2');
  assert.deepEqual(changes.at(-1), ['-0.5, 0.5, 2', 'config.process[0].sliderspace.preview_auto_strengths']);
  config.config.process[0].sliderspace.preview_auto_strengths = '-0.5, 0.5, 2';
  assert.match(renderToStaticMarkup(tree()), /13 images per sampling round/);
  helpers.mergeSliderSpacePreview(existing, config);
  assert.equal(existing.config.process[0].sliderspace.preview_auto_strengths, '-0.5, 0.5, 2');
  config.config.process[0].sample.samples.push({ prompt: 'extra' });
  assert.match(helpers.validateSliderSpacePreview(config).join('\n'), /one nonempty sample prompt/);
  config.config.process[0].sample.samples = [{ prompt: ' ' }];
  assert.match(helpers.validateSliderSpacePreview(config).join('\n'), /one nonempty sample prompt/);
  config.config.process[0].sliderspace.preview_auto = 'yes';
  assert.match(helpers.validateSliderSpacePreview(config).join('\n'), /enabled or disabled/);
});

test('auto strengths accept finite CSV, preserve ordering, skip zero/duplicates and reject malformed entries', () => {
  assert.deepEqual(helpers.parseSliderSpaceAutoStrengths('-1, 1'), [-1, 1]);
  assert.deepEqual(helpers.parseSliderSpaceAutoStrengths(' -.5, 0, +.25, 1.5, .25, -0, 2e-2 '), [-.5, .25, 1.5, .02]);
  assert.deepEqual(helpers.parseSliderSpaceAutoStrengths('0'), []);
  const config = validJob();
  config.config.process[0].sliderspace.preview_auto = true;
  config.config.process[0].sample.samples = [{ prompt: 'a spaceship' }];
  for (const value of ['', ' ', '1,', ',1', '1,,2', 'cat', 'NaN', 'Infinity', '1e309', '0x10', null, [1], true]) {
    assert.equal(helpers.parseSliderSpaceAutoStrengths(value), null, String(value));
    config.config.process[0].sliderspace.preview_auto_strengths = value;
    assert.match(helpers.validateSliderSpacePreview(config).join('\n'), /finite numbers separated by commas/);
    assert.throws(() => helpers.mergeSliderSpacePreview(validJob(), config), /finite numbers separated by commas/);
  }
  config.config.process[0].sliderspace.preview_auto_strengths = '';
  const markup = renderToStaticMarkup(React.createElement(editor.SliderSpacePreview, { jobConfig: config, setJobConfig() {} }));
  assert.match(markup, /role="alert"/);
});

test('auto sample filenames label the base and group all signed directions in one round', () => {
  const round = (time, step) => [
    `${time}_${step}_auto_direction_00_strength_plus0_0.jpg`,
    `${time + 1}_${step}_auto_direction_01_strength_minus1_1.jpg`,
    `${time + 2}_${step}_auto_direction_01_strength_plus1_2.jpg`,
    `${time + 3}_${step}_auto_direction_02_strength_minus1_3.jpg`,
    `${time + 4}_${step}_auto_direction_02_strength_plus1_4.jpg`,
  ];
  const first = round(100, '000000020');
  const second = round(200, '000000040');
  assert.equal(helpers.sliderSpaceSampleLabel(first[0]), 'Base model · strength 0');
  assert.equal(helpers.sliderSpaceSampleLabel(first[1]), 'Direction 1 · strength -1');
  assert.equal(helpers.sampleFilenameInfo(first[4]).step, 20);
  assert.equal(helpers.sampleFilenameInfo(first[4]).promptIdx, 4);
  assert.deepEqual(helpers.groupSliderSpaceSamples([...first, ...second]), [first, second]);
  assert.deepEqual(helpers.groupSliderSpaceSamples([...first, ...round(300, '000000020')]), [first, round(300, '000000020')]);
  assert.equal(helpers.adjacentSliderSpaceSample([first, second], first[1], 1, 0), second[1]);
});

test('whole training form exposes supported fields, mobile mode selection and full-width discovery', () => {
  const Null = () => null;
  const Card = ({ title, children }) => React.createElement('section', { 'data-card': title }, title, children);
  const FormGroup = ({ label, children }) => React.createElement('div', null, label, children);
  const controls = { TextInput: Input, TextAreaInput: Input, NumberInput: Input, SelectInput: Input,
    Checkbox: Input, SliderInput: Input, CreatableSelectInput: Input, FormGroup };
  const { default: SimpleJob } = loadTypescript('../src/app/jobs/new/SimpleJob.tsx', {
    './options': options,
    './jobConfig': defaults,
    '@/extensions/modelArchs': { useModelArchs: () => ({ archs: [{ name: 'qwen_image_2', group: 'image',
      additionalSections: ['model.assistant_lora_path', 'model.text_encoder_path', 'model.low_vram', 'model.layer_offloading', 'model.low_vram_layer_streaming'] }], groupedModelOptions: [] }) },
    '@/utils/basic': { objectCopy: structuredClone },
    '@/components/formInputs': controls,
    '@/components/Card': Card,
    'lucide-react': { X: Null, Copy: Null, Wand2: Null, SquareDashed: Null, Info: Null, FlipHorizontal2: Null, FlipVertical2: Null },
    'react-icons/io5': { IoFlaskSharp: Null },
    '@/components/DocModal': {},
    '@/components/UpsamplePromptsModal': {},
    '@/components/PromptBoxEditorModal': {},
    '@/components/AddSingleImageModal': Null,
    '@/components/SampleControlImage': Null,
    './utils': {},
    '@/helpers/basic': { isMac: () => false },
    '@/utils/api': {},
    '@/components/Modal': { Modal: Null },
    './FizgigMultipointEditor': Null,
    './SliderSpaceEditor': editor,
    './DiffusionKTOEditor': loadTypescript('../src/app/jobs/new/DiffusionKTOEditor.tsx', {
      '@/components/Card': Card, '@/components/formInputs': controls,
    }).default,
    './fizgigMultipoint': {},
    './trainingCapabilities': capabilities,
  });
  const props = { jobConfig: validJob(), setJobConfig() {}, handleSubmit() {}, status: 'idle', runId: null,
    gpuIDs: '0', setGpuIDs() {}, gpuList: [], datasetOptions: [] };
  const markup = renderToStaticMarkup(React.createElement(SimpleJob, props));
  for (const text of ['Training mode', 'Concept prompt', 'Directions to train', 'Total training steps', 'Learning Rate',
    'Linear Alpha', 'Preview direction', 'Preview strength', 'Low VRAM', 'Text Encoder Safetensors Path', 'Helper LoRA Path', 'Inference LoRA Path']) assert.ok(markup.includes(text), text);
  for (const text of ['data-card="Datasets"', 'data-card="Validation"', 'data-card="Advanced"', 'Use EMA', 'Loss Type',
    'Pixel Frequency Loss', 'Contrastive Guidance Loss', 'Trigger Word', 'LoRA Scale', 'Record Low Window']) {
    assert.ok(!markup.includes(text), text);
  }
  assert.ok(markup.indexOf('Directions to train') > markup.indexOf('data-card="Save"'));
  assert.ok(markup.indexOf('Directions to train') < markup.indexOf('data-card="Training"'));
  const running = renderToStaticMarkup(React.createElement(SimpleJob, { ...props, sampleOnlyMode: true }));
  assert.match(running, /fieldset disabled=""/);
  assert.match(running, /Preview direction/);
  const automatic = structuredClone(props.jobConfig);
  automatic.config.process[0].sliderspace.preview_auto = true;
  const autoMarkup = renderToStaticMarkup(React.createElement(SimpleJob, { ...props, jobConfig: automatic, sampleOnlyMode: true }));
  assert.match(autoMarkup, /Auto sampling/);
  assert.match(autoMarkup, /Auto sample prompt/);
  assert.doesNotMatch(autoMarkup, /Sample Prompts|Add Prompt|Preview direction|Preview strength/);
  const kto = structuredClone(defaults.defaultJobConfig);
  kto.config.process[0].type = 'diffusion_kto';
  kto.config.process[0].model.arch = 'qwen_image_2';
  options.jobTypeOptions.find(item => item.value === 'diffusion_kto').onActivate(kto);
  const ktoMarkup = renderToStaticMarkup(React.createElement(SimpleJob, { ...props, jobConfig: kto }));
  for (const text of ['Diffusion-KTO (experimental)', 'Feedback Label', 'Image Folder', 'Reference-point Estimator',
    'Beta', 'Liked Weight', 'Disliked Weight', 'Dataset Loss Weight', 'Gradient Accumulation', 'Cache Latents (required)', 'Linear Alpha']) {
    assert.ok(ktoMarkup.includes(text), text);
  }
  for (const text of ['Control Dataset', 'Anchor Images', 'Caption Dropout Rate', 'Loss Type',
    'Contrastive Guidance Loss', 'LoRA Weight', 'Is Regularization']) assert.ok(!ktoMarkup.includes(text), text);
});
