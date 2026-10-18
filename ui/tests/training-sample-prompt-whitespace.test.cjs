const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const test = require('node:test');
const YAML = require('yaml');

function load(file) {
  const filename = path.resolve(__dirname, file);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const result = new Module(filename, module);
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
