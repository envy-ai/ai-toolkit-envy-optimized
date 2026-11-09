const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

function load(file) {
  const filename = path.resolve(__dirname, file);
  const result = new Module(filename, module);
  result.filename = filename;
  result.paths = Module._nodeModulePaths(path.dirname(filename));
  result._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText, filename);
  return result.exports;
}
const reader = load('../src/server/trainingMaps.ts');
const colors = load('../src/utils/trainingMaps.ts');
const caps = load('../src/app/jobs/new/trainingCapabilities.ts');
const manifest = () => ({ version: 1, status: 'ready', probes: [{ key: 'p', image_path: 'p.png', caption: 'cat', width: 8, height: 8 }], levels: [.25], steps: [1, 10], scales: { p: { '0.25': { error: 2, effect: 1, improvement: 1 } } } });
const frame = step => ({ step, maps: { p: { '0.25': { error: { w: 2, h: 1, v: [1, 2] }, effect: { w: 2, h: 1, v: [1, 1] }, ...(step > 1 ? { improvement: { w: 2, h: 1, v: [0, 0] } } : {}) } } } });

test('filesystem reader chooses latest or requested frame and reports corruption', async t => {
  const dir = await fsp.mkdtemp(path.resolve(__dirname, '../../tmp/training-maps-test-'));
  t.after(() => fsp.rm(dir, { recursive: true, force: true }));
  assert.deepEqual(await reader.readTrainingMaps(dir, null), { available: false });
  await fsp.writeFile(path.join(dir, 'manifest.json'), JSON.stringify(manifest()));
  for (const step of [1, 10]) await fsp.writeFile(path.join(dir, `step-${String(step).padStart(9, '0')}.json`), JSON.stringify(frame(step)));
  assert.equal((await reader.readTrainingMaps(dir, null)).frame.step, 10);
  assert.equal((await reader.readTrainingMaps(dir, 1)).frame.step, 1);
  await assert.rejects(reader.readTrainingMaps(dir, 2), /not recorded/);
  await fsp.writeFile(path.join(dir, 'step-000000010.json'), JSON.stringify({ ...frame(10), maps: { p: { '0.25': { error: { w: 2, h: 2, v: [1] }, effect: { w: 1, h: 1, v: [1] }, improvement: { w: 1, h: 1, v: [0] } } } } }));
  await assert.rejects(reader.readTrainingMaps(dir, null), /Invalid training map grid/);
  await fsp.unlink(path.join(dir, 'step-000000001.json'));
  await assert.rejects(reader.readTrainingMaps(dir, 1), { code: 'ENOENT' });
  await fsp.writeFile(path.join(dir, 'manifest.json'), '{');
  await assert.rejects(reader.readTrainingMaps(dir, null), SyntaxError);
});

test('manifest, step and error validation', () => {
  assert.equal(reader.parseTrainingMapStep(null), null);
  assert.equal(reader.parseTrainingMapStep('001'), 1);
  for (const value of ['-1', '1.5', 'abc', '9007199254740992']) assert.throws(() => reader.parseTrainingMapStep(value));
  for (const patch of [{ steps: [10, 1] }, { steps: [1, 1] }, { levels: [1] }, { probes: [manifest().probes[0], manifest().probes[0]] }, { status: 'error' }]) {
    assert.throws(() => reader.validateTrainingMapManifest({ ...manifest(), ...patch }));
  }
  assert.equal(reader.validateTrainingMapManifest({ ...manifest(), status: 'error', error: { message: 'failed', stack: 'trace' } }).error.message, 'failed');
});

test('map colors use fixed shared scale and distinct signed colors', () => {
  assert.equal(colors.trainingMapScale([{ w: 2, h: 1, v: [-4, 1] }, { w: 1, h: 1, v: [2] }]), 4);
  assert.equal(colors.trainingMapColor(0, 4, true), 'transparent');
  assert.equal(colors.trainingMapColor(1, 4, true), 'rgba(34,197,94,0.2125)');
  assert.equal(colors.trainingMapColor(-1, 4, true), 'rgba(232,67,162,0.2125)');
  assert.match(colors.trainingMapColor(4, 4, false), /^rgba\(239,68,68,/);
});

test('map configuration validates positive arguments and supported mode', () => {
  const process = { type: 'diffusion_trainer', model: { arch: 'moss_image', model_kwargs: {} }, network: { type: 'lora' }, train: {}, datasets: [], sample: {}, logging: { training_maps: true, training_maps_every: 2, training_maps_images: 1 } };
  const job = { config: { process: [process] } };
  assert.deepEqual(caps.validateTrainingCapabilities(job), []);
  process.logging.training_maps_every = -1;
  assert.match(caps.validateTrainingCapabilities(job).join('\n'), /training_maps_every must be a positive integer/);
  process.logging.training_maps_every = 1;
  process.type = 'flow_dpo';
  assert.match(caps.validateTrainingCapabilities(job).join('\n'), /Training maps support ordinary/);
});

test('recorded maps reject omitted scales and required map kinds', () => {
  const good = manifest();
  assert.throws(() => reader.validateTrainingMapManifest({ ...good, scales: {} }), /scales/);
  assert.throws(() => reader.validateTrainingMapManifest({ ...good, scales: { p: { '0.25': { error: 1, effect: 1 } } } }), /scale/);
  const missingEffect = frame(1);
  delete missingEffect.maps.p['0.25'].effect;
  assert.throws(() => reader.validateTrainingMapFrame(missingEffect, good, 1), /incomplete/);
  const missingImprovement = frame(10);
  delete missingImprovement.maps.p['0.25'].improvement;
  assert.throws(() => reader.validateTrainingMapFrame(missingImprovement, good, 10), /incomplete/);
});
