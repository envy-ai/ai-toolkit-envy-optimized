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
const source = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
const tree = ts.createSourceFile('SimpleJob.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);

test('DoKr is opt-in and config serialization retains rank, alpha and memory options', () => {
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  assert.equal(process.network.lokr_dora, false);
  process.network.type = 'lokr';
  process.network.lokr_dora = true;
  process.network.linear = 16;
  process.network.linear_alpha = 8;
  process.model.qtype = 'convrotint4';
  process.model.low_vram = true;
  process.train.gradient_checkpointing = true;
  const decoded = JSON.parse(JSON.stringify(config)).config.process[0];
  assert.equal(decoded.network.type, 'lokr');
  assert.equal(decoded.network.lokr_dora, true);
  assert.equal(decoded.network.linear_alpha / decoded.network.linear, 0.5);
  assert.equal(decoded.model.qtype, 'convrotint4');
  assert.equal(decoded.model.low_vram, true);
  assert.equal(decoded.train.gradient_checkpointing, true);
});

test('form exposes LoKr and a checkbox wired to the serialized DoKr field', () => {
  assert.match(source, /value: 'lokr', label: 'LoKr'/);
  let checkbox;
  function visit(node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(tree) === 'Checkbox' &&
        node.attributes.getText(tree).includes('DoRA Weight Decomposition (DoKr)')) checkbox = node;
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.ok(checkbox);
  const attributes = new Map(checkbox.attributes.properties.filter(ts.isJsxAttribute)
    .map(attr => [attr.name.getText(tree), attr.initializer]));
  const handler = attributes.get('onChange').expression.getText(tree);
  const updates = [];
  new Function('setJobConfig', `return (${handler})`)((...args) => updates.push(args))(true);
  assert.deepEqual(updates, [[true, 'config.process[0].network.lokr_dora']]);
  assert.match(attributes.get('checked').getText(tree), /network\?\.lokr_dora \?\? false/);
  assert.match(source, /networkType == 'lokr' && \(\s*<Checkbox/);
  assert.match(source, /networkType == 'lora' \|\| networkType == 'dora' \|\| networkType == 'loha'/);
});

test('LoRA-only training modes still reset a previously selected LoKr', () => {
  const { jobTypeOptions } = loadTypescript('../src/app/jobs/new/options.tsx', { './jobConfig': defaults });
  for (const type of ['qwen_flow_dpo', 'qwen_guidance_distillation', 'fizgig_prompt_slider', 'fizgig_image_slider']) {
    const option = jobTypeOptions.find(option => option.value === type);
    assert.ok(option, type);
    const config = structuredClone(defaults.defaultJobConfig);
    config.config.process[0].network.type = 'lokr';
    config.config.process[0].model.arch = 'qwen_image_2';
    option.onActivate(config);
    assert.equal(config.config.process[0].network.type, 'lora', type);
  }
});

test('convolution controls remain available across adapter types and form hints', () => {
  const controls = new Map();
  function visit(node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(tree) === 'NumberInput') {
      const label = node.attributes.properties.find(a => ts.isJsxAttribute(a) && a.name.getText(tree) === 'label');
      if (label && ['Conv Rank', 'Conv Alpha'].includes(label.initializer.text)) controls.set(label.initializer.text, node);
    }
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.equal(controls.size, 2);
  for (const [label, node] of controls) {
    for (const networkType of ['lora', 'dora', 'loha', 'lokr', 'unknown']) {
      for (const isFizgigSlider of [false, true]) {
        for (const disableSections of [[], ['network.conv']]) {
          for (let parent = node.parent; parent && !ts.isJsxElement(parent); parent = parent.parent) {
            if (ts.isBinaryExpression(parent) && parent.operatorToken.kind === ts.SyntaxKind.AmpersandAmpersandToken &&
                parent.right.pos <= node.pos && parent.right.end >= node.end) {
              const condition = parent.left.getText(tree);
              const visible = new Function('networkType', 'isFizgigSlider', 'disableSections', `return (${condition})`)
                (networkType, isFizgigSlider, disableSections);
              assert.equal(!!visible, networkType !== 'unknown', `${label}: ${networkType}, fizgig=${isFizgigSlider}, ${condition}`);
            }
          }
        }
      }
    }
  }
  const handler = label => controls.get(label).attributes.properties
    .find(a => ts.isJsxAttribute(a) && a.name.getText(tree) === 'onChange').initializer.expression.getText(tree);
  for (const networkType of ['lora', 'loha', 'lokr']) {
    for (const [alpha, expected] of [[4, 1], [8, 2], [undefined, 2]]) {
      const updates = [];
      const jobConfig = { config: { process: [{ network: { conv: 8, conv_alpha: alpha } }] } };
      new Function('jobConfig', 'networkType', 'setJobConfig', `return (${handler('Conv Rank')})`)
        (jobConfig, networkType, (...args) => updates.push(args))(12);
      assert.equal(updates.length, expected);
      assert.deepEqual(updates[0], [12, 'config.process[0].network.conv']);
      if (expected === 2) assert.deepEqual(updates[1], [12, 'config.process[0].network.conv_alpha']);
    }
  }
  const updates = [];
  new Function('setJobConfig', `return (${handler('Conv Alpha')})`)((...args) => updates.push(args))(5);
  assert.deepEqual(updates, [[5, 'config.process[0].network.conv_alpha']]);
});

test('LoKr full rank can be switched off and survives serialization', () => {
  let checkbox;
  function visit(node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(tree) === 'Checkbox' &&
        node.attributes.getText(tree).includes('label="LoKr Full Rank"')) checkbox = node;
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.ok(checkbox);
  const attr = name => checkbox.attributes.properties.find(a => ts.isJsxAttribute(a) && a.name.getText(tree) === name);
  assert.match(attr('checked').getText(tree), /lokr_full_rank \?\? false/);
  const config = structuredClone(defaults.defaultJobConfig);
  assert.equal(config.config.process[0].network.lokr_full_rank, true);
  const setJobConfig = (value, field) => {
    assert.equal(field, 'config.process[0].network.lokr_full_rank');
    config.config.process[0].network.lokr_full_rank = value;
  };
  new Function('setJobConfig', `return (${attr('onChange').initializer.expression.getText(tree)})`)(setJobConfig)(false);
  assert.equal(JSON.parse(JSON.stringify(config)).config.process[0].network.lokr_full_rank, false);
});
