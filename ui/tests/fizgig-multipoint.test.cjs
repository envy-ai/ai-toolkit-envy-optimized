const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

const filename = path.resolve(__dirname, '../src/app/jobs/new/fizgigMultipoint.ts');
const instance = new Module(filename, module);
instance._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText, filename);
const { toggleMultipoint, updateMultipointPoints, validateMultipointConfig, multipointNeutralPrompt } = instance.exports;

function config() {
  return { config: { process: [{ type: 'fizgig_prompt_slider', model: { qtype: 'convrotint4', low_vram: true },
    train: { gradient_checkpointing: true }, network: { type: 'dora' },
    fizgig_slider: { guidance: 3, positive_prefix: 'large', negative_prefix: 'small', cfg_negative_prefix: 'noise',
      prompt_entries: [{ kind: 'simple', prompt: 'cat' }, { kind: 'specific', neutral_prompt: 'cat', positive_prompt: 'adult',
        negative_prompt: 'baby', cfg_negative_prompt_positive: 'baby', cfg_negative_prompt_negative: 'adult' }] },
    datasets: [{ folder_path: '/adult', control_path_1: '/baby', anchor_path: '/dogs' }],
    sample: { walk_seed: true, samples: [{ prompt: 'cat', network_multiplier: .7 }] },
  }] } };
}

test('first enable seeds -1/+1 and blank +2; toggles preserve both drafts and memory settings', () => {
  const original = config();
  const enabled = toggleMultipoint(original, true);
  const process = enabled.config.process[0];
  assert.equal(original.config.process[0].fizgig_slider.multipoint, undefined);
  assert.deepEqual(process.fizgig_slider.multipoint_config.points.map(p => p.strength), [-1, 1, 2]);
  assert.deepEqual(process.fizgig_slider.multipoint_config.points.map(p => p.prefix), ['small', 'large', '']);
  assert.deepEqual(process.fizgig_slider.multipoint_config.prompt_entries[1].targets.map(t => t.prompt), ['baby', 'adult', '']);
  assert.deepEqual(process.datasets[0].multipoint_images.map(p => p.folder_path), ['/baby', '/adult', '']);
  assert.equal(process.datasets[0].anchor_path, '/dogs');
  assert.deepEqual(process.sample.samples.map(p => p.network_multiplier), [-1, 0, 1, 2]);
  for (const key of ['model', 'train', 'network']) assert.deepEqual(process[key], original.config.process[0][key]);
  process.fizgig_slider.multipoint_config.points[2].prefix = 'mega';
  process.sample.samples = [{ prompt: 'custom', network_multiplier: 1.8 }];
  const reenabled = toggleMultipoint(toggleMultipoint(enabled, false), true).config.process[0];
  assert.equal(reenabled.fizgig_slider.multipoint_config.points[2].prefix, 'mega');
  assert.deepEqual(reenabled.fizgig_slider.prompt_entries, original.config.process[0].fizgig_slider.prompt_entries);
  assert.deepEqual(reenabled.sample.samples, process.sample.samples);
  assert.equal(reenabled.datasets[0].folder_path, '/adult');
  assert.equal(reenabled.datasets[0].control_path_1, '/baby');
});

test('strength edits/reorder and adding/removing zero preserve associated data', () => {
  let job = toggleMultipoint(config(), true);
  let mp = job.config.process[0].fizgig_slider.multipoint_config;
  job = updateMultipointPoints(job, mp.points.map(p => p.id === 'plus' ? { ...p, strength: -2.5 } : p));
  mp = job.config.process[0].fizgig_slider.multipoint_config;
  assert.deepEqual(mp.points.map(p => p.id), ['plus', 'minus', 'extra']);
  assert.equal(mp.prompt_entries[1].targets[0].prompt, 'adult');
  assert.equal(job.config.process[0].datasets[0].multipoint_images[0].folder_path, '/adult');
  job = updateMultipointPoints(job, [...mp.points, { id: 'zero', strength: 0, prefix: '', negative_prefix: '' }]);
  mp = job.config.process[0].fizgig_slider.multipoint_config;
  const entry = mp.prompt_entries[1];
  assert.equal(entry.targets.find(t => t.point_id === 'zero').prompt, 'cat');
  entry.targets.find(t => t.point_id === 'zero').prompt = 'awkward cat';
  job = updateMultipointPoints(job, mp.points.filter(p => p.id !== 'zero'));
  assert.equal(job.config.process[0].fizgig_slider.multipoint_config.prompt_entries[1].neutral_prompt, 'awkward cat');
  assert.deepEqual(job.config.process[0].sample.samples.map(p => p.network_multiplier), [-1, 0, 1, 2]);
});

test('saved-set schema rejects duplicate numbers/IDs, zero-only/nonfinite and inconsistent targets', () => {
  const mp = toggleMultipoint(config(), true).config.process[0].fizgig_slider.multipoint_config;
  assert.deepEqual(validateMultipointConfig(mp), mp);
  for (const points of [
    [{ ...mp.points[0], strength: 0 }],
    [mp.points[0], { ...mp.points[1], strength: -1 }],
    [mp.points[0], { ...mp.points[1], id: 'minus' }],
    [{ ...mp.points[0], strength: Infinity }],
    [{ ...mp.points[0], strength: '1' }],
  ]) assert.equal(validateMultipointConfig({ ...mp, points }), null);
  assert.equal(validateMultipointConfig({ ...mp, prompt_entries: [{ ...mp.prompt_entries[1], targets: [] }] }), null);
});

test('preview neutral selection follows explicit zero for simple and specific entries', () => {
  const mp = toggleMultipoint(config(), true).config.process[0].fizgig_slider.multipoint_config;
  assert.equal(multipointNeutralPrompt(mp), 'cat');
  mp.points.push({ id: 'zero', strength: 0, prefix: 'awkward', negative_prefix: '' });
  assert.equal(multipointNeutralPrompt(mp), 'awkward\n\ncat');
  mp.prompt_entries = [{ kind: 'specific', neutral_prompt: 'implicit', neutral_negative_prompt: '',
    targets: [{ point_id: 'zero', prompt: 'explicit zero', negative_prompt: '' }] }];
  assert.equal(multipointNeutralPrompt(mp), 'explicit zero');
});

test('form has opt-in editors and disabled CFG negatives, keeping legacy UI behind the toggle', () => {
  const form = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  const editor = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/FizgigMultipointEditor.tsx'), 'utf8');
  assert.ok(form.includes('<Checkbox label="Multi-point" checked={isMultipoint}'));
  assert.ok(form.includes('{!isMultipoint && <NumberInput'));
  assert.ok(form.includes('dataset.multipoint_images'));
  assert.ok(editor.includes('Add Point'));
  assert.ok(editor.includes('Add Specific Prompts'));
  assert.ok(editor.includes('disabled={!cfgEnabled}'));
  assert.ok(editor.includes('!zero && <Card title="Neutral reference (base model)"'));
  assert.ok(form.includes('multipoint_config: multipointConfig'));
});

test('preservation button is beside the other add buttons in both prompt slider layouts', () => {
  const form = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  const editor = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/FizgigMultipointEditor.tsx'), 'utf8');
  for (const [source, filename] of [[form, 'SimpleJob.tsx'], [editor, 'FizgigMultipointEditor.tsx']]) {
    const tree = ts.createSourceFile(filename, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
    let preservationButton;
    function visit(node) {
      if (ts.isJsxElement(node) && node.openingElement.tagName.getText(tree) === 'button' &&
          node.children.some(child => ts.isJsxText(child) && child.getText(tree).trim() === 'Add Preservation Prompt')) {
        preservationButton = node;
      }
      ts.forEachChild(node, visit);
    }
    visit(tree);
    assert.ok(preservationButton, filename);
    const toolbar = preservationButton.parent.getText(tree);
    assert.ok(toolbar.includes('Add Simplified Prompt'), filename);
    assert.match(toolbar, /Add Specific (Triplet|Prompts)/, filename);
  }
  assert.ok(form.includes('onAddPreservationPrompt={addPreservationPrompt}'));
  assert.ok(form.indexOf('title="Preservation Anchors"') > form.indexOf('Add Preservation Prompt'));
  assert.ok(form.includes('title={`Preservation Prompt ${index + 1} (Anchor)`}'));
});

test('add preservation appends a separate anchor without replacing slider target prompts', () => {
  const form = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  const tree = ts.createSourceFile('SimpleJob.tsx', form, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  let initializer;
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(tree) === 'addPreservationPrompt') initializer = node.initializer;
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.ok(initializer);
  const anchors = [{ prompt: 'A dog stays the same size', negative_prompt: 'cat' }];
  const calls = [];
  const add = new Function('setJobConfig', 'anchorPrompts', `return (${initializer.getText(tree)})`)(
    (...args) => calls.push(args), anchors);
  add();
  assert.deepEqual(calls, [[[
    anchors[0], { prompt: '', negative_prompt: '' },
  ], 'config.process[0].fizgig_slider.anchor_prompts']]);
  assert.equal(anchors.length, 1);

  // Invoke the actual multipoint component and click its React button.
  const editorPath = path.resolve(__dirname, '../src/app/jobs/new/FizgigMultipointEditor.tsx');
  const componentModule = new Module(editorPath, module);
  componentModule.filename = editorPath;
  componentModule.paths = Module._nodeModulePaths(path.dirname(editorPath));
  const originalRequire = componentModule.require.bind(componentModule);
  componentModule.require = name => {
    if (name === '@/components/Card') return { __esModule: true, default: 'section' };
    if (name === '@/components/formInputs') return { NumberInput: 'input', TextAreaInput: 'textarea' };
    if (name === './fizgigMultipoint') return instance.exports;
    return originalRequire(name);
  };
  componentModule._compile(ts.transpileModule(fs.readFileSync(editorPath, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2020 },
  }).outputText, editorPath);
  const mp = toggleMultipoint(config(), true).config.process[0].fizgig_slider.multipoint_config;
  const before = structuredClone(mp);
  let clicks = 0;
  const element = componentModule.exports.default({ config: mp, cfgEnabled: false,
    onChange: () => assert.fail('preservation must not modify multipoint targets'),
    onAddPreservationPrompt: () => clicks++,
  });
  let button;
  function find(node) {
    if (Array.isArray(node)) return node.forEach(find);
    if (!node || typeof node !== 'object' || !node.props) return;
    if (node.type === 'button' && node.props.children === 'Add Preservation Prompt') button = node;
    find(node.props.children);
  }
  find(element);
  assert.ok(button);
  button.props.onClick();
  assert.equal(clicks, 1);
  assert.deepEqual(mp, before);
});
