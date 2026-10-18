const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

function load(relativePath, overrides = {}) {
  const filename = path.resolve(__dirname, relativePath);
  const result = new Module(filename, module);
  result.filename = filename;
  result.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = result.require.bind(result);
  result.require = id => Object.hasOwn(overrides, id) ? overrides[id] : originalRequire(id);
  result._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText, filename);
  return result.exports;
}

const capabilities = load('../src/app/jobs/new/trainingCapabilities.ts');
const defaults = load('../src/app/jobs/new/jobConfig.ts', {
  '@/helpers/basic': { isMac: () => false },
  '@/helpers/defaultSamples': load('../src/helpers/defaultSamples.ts'),
  './trainingCapabilities': capabilities,
});
const options = load('../src/app/jobs/new/options.tsx', {
  './jobConfig': defaults,
  './trainingCapabilities': capabilities,
});
const Null = () => null;
const Card = ({ title, children }) => React.createElement('section', { 'data-card': title }, children);
const FormGroup = ({ label, children }) => React.createElement('div', null, label, children);

function renderJob(jobConfig) {
  const fields = [];
  const updates = [];
  const Input = props => {
    fields.push(props);
    return React.createElement('label', null, typeof props.label === 'string' ? props.label : null);
  };
  const controls = {
    TextInput: Input,
    TextAreaInput: Input,
    NumberInput: Input,
    SelectInput: Input,
    Checkbox: Input,
    SliderInput: Input,
    CreatableSelectInput: Input,
    FormGroup,
  };
  const { default: SimpleJob } = load('../src/app/jobs/new/SimpleJob.tsx', {
    './options': options,
    './jobConfig': defaults,
    '@/extensions/modelArchs': {
      useModelArchs: () => ({ archs: [{ name: 'qwen_image_2', group: 'image', additionalSections: [] }], groupedModelOptions: [] }),
    },
    '@/utils/basic': { objectCopy: structuredClone },
    '@/helpers/datasetControls': {
      getDatasetControlPaths: () => [],
      setDatasetControlPath: () => ({}),
    },
    '@/components/formInputs': controls,
    '@/components/Card': Card,
    'lucide-react': {
      X: Null, Copy: Null, Wand2: Null, SquareDashed: Null, Info: Null,
      FlipHorizontal2: Null, FlipVertical2: Null,
    },
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
    './SliderSpaceEditor': Null,
    './DiffusionKTOEditor': Null,
    './fizgigMultipoint': {},
    './trainingCapabilities': capabilities,
  });
  const props = {
    jobConfig,
    setJobConfig: (value, key) => updates.push({ value, key }),
    handleSubmit() {},
    status: 'idle',
    runId: null,
    gpuIDs: '0',
    setGpuIDs() {},
    gpuList: [],
    datasetOptions: [],
  };
  const markup = renderToStaticMarkup(React.createElement(SimpleJob, props));
  return { fields, markup, updates };
}

function jobWithResolutionPresets(nativeResolution) {
  const job = structuredClone(defaults.defaultJobConfig);
  const process = job.config.process[0];
  process.model.arch = 'qwen_image_2';
  process.datasets[0].folder_path = '/images';
  process.datasets[0].resolution = [512, 1024];
  process.datasets[0].native_resolution = nativeResolution;
  return job;
}

test('native resolution toggle preserves selected presets and disables them only while enabled', () => {
  const initial = renderJob(jobWithResolutionPresets(false));
  const nativeToggle = initial.fields.find(field => field.label === 'Native resolution (no scaling)');
  assert.ok(nativeToggle);
  assert.equal(nativeToggle.checked, false);
  nativeToggle.onChange(true);
  assert.deepEqual(initial.updates, [{
    value: true,
    key: 'config.process[0].datasets[0].native_resolution',
  }]);

  const enabledConfig = jobWithResolutionPresets(true);
  const enabled = renderJob(enabledConfig);
  const enabledToggle = enabled.fields.find(field => field.label === 'Native resolution (no scaling)');
  assert.equal(enabledToggle.checked, true);
  for (const resolution of ['512', '1024']) {
    const preset = enabled.fields.find(field => field.label === resolution);
    assert.ok(preset, `resolution ${resolution} remains visible`);
    assert.equal(preset.checked, true);
    assert.equal(preset.disabled, true);
  }
  assert.match(enabled.markup, /Resolution presets are ignored/);
  enabledToggle.onChange(false);
  assert.deepEqual(enabled.updates, [{
    value: false,
    key: 'config.process[0].datasets[0].native_resolution',
  }]);

  const disabled = renderJob(jobWithResolutionPresets(false));
  for (const resolution of ['512', '1024']) {
    const preset = disabled.fields.find(field => field.label === resolution);
    assert.equal(preset.checked, true);
    assert.equal(preset.disabled, false);
  }
});
