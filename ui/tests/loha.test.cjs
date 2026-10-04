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

test('DoHa is opt-in and config serialization retains rank, alpha and memory options', () => {
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  assert.equal(process.network.loha_dora, false);
  process.network.type = 'loha';
  process.network.loha_dora = true;
  process.network.linear = 16;
  process.network.linear_alpha = 8;
  process.model.qtype = 'convrotint4';
  process.model.low_vram = true;
  process.train.gradient_checkpointing = true;
  const decoded = JSON.parse(JSON.stringify(config)).config.process[0];
  assert.equal(decoded.network.type, 'loha');
  assert.equal(decoded.network.loha_dora, true);
  assert.equal(decoded.network.linear_alpha / decoded.network.linear, 0.5);
  assert.equal(decoded.model.qtype, 'convrotint4');
  assert.equal(decoded.model.low_vram, true);
  assert.equal(decoded.train.gradient_checkpointing, true);
});

test('form exposes LoHa and a checkbox wired to the serialized DoHa field', () => {
  assert.match(source, /value: 'loha', label: 'LoHa \(LyCORIS\)'/);
  let checkbox;
  function visit(node) {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(tree) === 'Checkbox' &&
        node.attributes.getText(tree).includes('DoRA Weight Decomposition (DoHa)')) checkbox = node;
    ts.forEachChild(node, visit);
  }
  visit(tree);
  assert.ok(checkbox);
  const attributes = new Map(checkbox.attributes.properties.filter(ts.isJsxAttribute)
    .map(attr => [attr.name.getText(tree), attr.initializer]));
  const handler = attributes.get('onChange').expression.getText(tree);
  const updates = [];
  new Function('setJobConfig', `return (${handler})`)((...args) => updates.push(args))(true);
  assert.deepEqual(updates, [[true, 'config.process[0].network.loha_dora']]);
  assert.match(attributes.get('checked').getText(tree), /network\?\.loha_dora \?\? false/);
  assert.match(source, /networkType == 'loha' && \(\s*<Checkbox/);
  assert.match(source, /networkType == 'lora' \|\| networkType == 'dora' \|\| networkType == 'loha'/);
});

test('LoRA-only training modes still reset a previously selected LoHa', () => {
  const { jobTypeOptions } = loadTypescript('../src/app/jobs/new/options.tsx', { './jobConfig': defaults });
  for (const type of ['qwen_flow_dpo', 'qwen_guidance_distillation', 'fizgig_prompt_slider', 'fizgig_image_slider']) {
    const option = jobTypeOptions.find(option => option.value === type);
    assert.ok(option, type);
    const config = structuredClone(defaults.defaultJobConfig);
    config.config.process[0].network.type = 'loha';
    config.config.process[0].model.arch = 'qwen_image_2';
    option.onActivate(config);
    assert.equal(config.config.process[0].network.type, 'lora', type);
  }
});
