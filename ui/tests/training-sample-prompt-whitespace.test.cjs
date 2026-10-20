const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const test = require('node:test');
const YAML = require('yaml');

function load(file, overrides = {}) {
  const filename = path.resolve(__dirname, file);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const result = new Module(filename, module);
  result.filename = filename;
  result.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = result.require.bind(result);
  result.require = id => Object.hasOwn(overrides, id) ? overrides[id] : originalRequire(id);
  result._compile(compiled, filename);
  return result.exports;
}

const { objToTags, tagsToObj } = load('../src/utils/basic.ts');

test('tagged training text fields retain newlines through job-config YAML transport and field edits', () => {
  const textFields = ['CAPTION', 'LYRICS'];
  const fields = {
    CAPTION: '\n123\n',
    LYRICS: 'verse one\n\n  verse two\n',
    BPM: 120,
  };
  const tagged = objToTags(fields);
  const parsed = tagsToObj(tagged, textFields);
  assert.equal(parsed.CAPTION, fields.CAPTION);
  assert.equal(parsed.LYRICS, fields.LYRICS);
  assert.equal(parsed.BPM, fields.BPM);

  parsed.BPM = 130;
  const editedTagged = objToTags(parsed);
  const configRoundTrip = YAML.parse(YAML.stringify({ sample: { samples: [{ prompt: editedTagged }] } }));
  assert.deepEqual(tagsToObj(configRoundTrip.sample.samples[0].prompt, textFields), { ...fields, BPM: 130 });
});

test('training Sample Prompts uses multiline text controls and keeps the sample prompt in job config', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  const sampleStart = source.indexOf('Sample Prompts (');
  assert.notEqual(sampleStart, -1);
  assert.match(source, /tagsToObj\(taggedPrompt,\s*textTags\)/);
  const sampleSection = source.slice(sampleStart, source.indexOf('{/* ----', sampleStart) > -1
    ? source.indexOf('{/* ----', sampleStart)
    : undefined);
  assert.match(sampleSection, /tag\.type === 'text'[\s\S]*?<TextAreaInput/);
  assert.match(sampleSection, /tag\.type === 'multiline'[\s\S]*?<TextAreaInput/);
  assert.match(sampleSection, /<TextAreaInput[\s\S]*?value=\{sample\.prompt\}/);
  assert.match(sampleSection, /config\.process\[0\]\.sample\.samples\[\$\{i\}\]\.prompt/);
});

test('Sample Prompt trigger prefix checkbox updates the sample config field', () => {
  const source = fs.readFileSync(path.resolve(__dirname, '../src/app/jobs/new/SimpleJob.tsx'), 'utf8');
  assert.match(source, /label="Prepend Trigger Word to Sample Prompts"[\s\S]*?checked=\{jobConfig\.config\.process\[0\]\.sample\.prepend_trigger_word \?\? false\}[\s\S]*?setJobConfig\(value, 'config\.process\[0\]\.sample\.prepend_trigger_word'\)/);
});

test('sample-only edits guard cached native prompt embeddings when prefix or renderer mode changes', () => {
  const { mergeSampleOnlyJobConfig } = load('../src/helpers/sampleOnlyJobConfig.ts', {
    '@/app/jobs/new/sliderspace': { mergeSliderSpacePreview: () => {} },
  });
  const job = ({ cached = false, cachedEmbeddings = false, prefix = false, comfy = false, datasetCached = false } = {}) => ({
    config: { process: [{
      type: 'diffusion_trainer',
      sample: { prepend_trigger_word: prefix, comfy: { enabled: comfy }, samples: [{ prompt: 'saved prompt' }] },
      train: { unload_text_encoder: cached, cache_text_embeddings: cachedEmbeddings },
      datasets: datasetCached ? [{ cache_text_embeddings: true }] : [],
      model: {}, save: {},
    }] },
  });
  const edit = (existing, changes = {}) => {
    const incoming = structuredClone(existing);
    Object.assign(incoming.config.process[0].sample, changes.sample || {});
    return mergeSampleOnlyJobConfig(existing, incoming);
  };

  assert.throws(
    () => edit(job({ cached: true }), { sample: { prepend_trigger_word: true } }),
    /cached while the text encoder is unloaded.*restart training/s,
  );
  assert.throws(
    () => edit(job({ datasetCached: true }), { sample: { prepend_trigger_word: true } }),
    /cached while the text encoder is unloaded/,
  );
  assert.throws(
    () => edit(job({ cachedEmbeddings: true }), { sample: { prepend_trigger_word: true } }),
    /cached while the text encoder is unloaded/,
  );
  assert.throws(
    () => edit(job({ cached: true, comfy: true }), { sample: { comfy: { enabled: false } } }),
    /cached while the text encoder is unloaded/,
  );

  assert.equal(edit(job({ cached: true })).config.process[0].sample.prepend_trigger_word, false);
  assert.equal(edit(job({ cached: true, comfy: true }), {
    sample: { prepend_trigger_word: true },
  }).config.process[0].sample.prepend_trigger_word, true);
  assert.equal(edit(job(), { sample: { prepend_trigger_word: true } })
    .config.process[0].sample.prepend_trigger_word, true);
});
