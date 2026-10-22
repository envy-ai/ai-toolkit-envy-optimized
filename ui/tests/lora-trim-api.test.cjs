const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

const projectRoot = path.resolve(__dirname, '../..');
const artifactRoot = path.join(projectRoot, 'tmp', 'lora-trim');
const jobFolder = path.join(artifactRoot, 'e2e-output', 'trim_fixture');
const sourcePath = path.join(jobFolder, 'synthetic_lora.safetensors');
const unsupportedPath = path.join(jobFolder, 'unsupported_doha.safetensors');
const doraPath = path.join(jobFolder, 'synthetic_dora.safetensors');
const basePath = path.join(jobFolder, 'synthetic_base.safetensors');
const lohaPath = path.join(jobFolder, 'synthetic_loha.safetensors');
const outputPath = path.join(jobFolder, 'api_trimmed.safetensors');
const pythonPath = process.env.AITK_TEST_PYTHON || path.join(process.env.CONDA_PREFIX || '', 'bin', 'python');

function loadRoute() {
  const filename = path.resolve(__dirname, '../src/app/api/scripts/route.ts');
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      esModuleInterop: true,
      target: ts.ScriptTarget.ES2020,
    },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => {
    if (id === '@/paths') return { TOOLKIT_ROOT: projectRoot };
    if (id === '../../../../cron/pythonPath') return { resolvePythonPath: () => pythonPath };
    return originalRequire(id);
  };
  instance._compile(compiled, filename);
  return instance.exports;
}

const post = async (route, payload) => route.POST(new Request('http://localhost/api/scripts', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(payload),
}));

test('scripts route runs real CPU trimming CLI with preview binding and no-clobber writes', async () => {
  assert.ok(fs.existsSync(sourcePath), 'Run playwright_scripts/prepare_lora_trim_e2e.py first');
  assert.ok(fs.existsSync(unsupportedPath), 'Run playwright_scripts/prepare_lora_trim_e2e.py first');
  assert.ok(fs.existsSync(doraPath), 'Run playwright_scripts/prepare_lora_trim_e2e.py first');
  assert.ok(fs.existsSync(basePath), 'Run playwright_scripts/prepare_lora_trim_e2e.py first');
  assert.ok(fs.existsSync(lohaPath), 'Run playwright_scripts/prepare_lora_trim_e2e.py first');
  assert.ok(fs.existsSync(pythonPath), `Expected ai-toolkit Python interpreter at ${pythonPath}`);
  assert.equal(process.env.CUDA_VISIBLE_DEVICES, '', 'This integration test must not see a CUDA device');
  fs.rmSync(outputPath, { force: true });
  const sourceBefore = fs.readFileSync(sourcePath);
  const route = loadRoute();
  const baseArgs = {
    input: sourcePath,
    'retained-energy': 0.9999,
    'min-rank': 1,
    json: true,
  };

  const invalidScript = await post(route, { script: '../trim_lora.py', args: [] });
  assert.equal(invalidScript.status, 400);

  const previewResponse = await post(route, { script: 'trim_lora.py', args: baseArgs });
  assert.equal(previewResponse.status, 200);
  const previewEnvelope = await previewResponse.json();
  assert.equal(previewEnvelope.ok, true, previewEnvelope.stderr);
  assert.equal(previewEnvelope.result.written, false);
  assert.equal(previewEnvelope.result.layers[0].old_rank, 8);
  assert.equal(previewEnvelope.result.layers[0].new_rank, 3);
  assert.equal(fs.existsSync(outputPath), false, 'Preview must not create a file');

  const explicitPolicyPreview = await post(route, {
    script: 'trim_lora.py',
    args: { input: sourcePath, 'rank-policy': 'fixed', 'rank-value': 2, json: true },
  });
  assert.equal(explicitPolicyPreview.status, 200);
  const explicitPolicyEnvelope = await explicitPolicyPreview.json();
  assert.equal(explicitPolicyEnvelope.ok, true, explicitPolicyEnvelope.stderr);
  assert.equal(explicitPolicyEnvelope.result.options.rank_policy, 'fixed');
  assert.equal(explicitPolicyEnvelope.result.options.rank_value, 2);
  assert.equal(explicitPolicyEnvelope.result.layers[0].new_rank, 2);
  assert.notEqual(explicitPolicyEnvelope.result.preview_token, previewEnvelope.result.preview_token);

  const doraPreviewResponse = await post(route, {
    script: 'trim_lora.py',
    args: { input: doraPath, json: true },
  });
  assert.equal(doraPreviewResponse.status, 200);
  const doraPreviewEnvelope = await doraPreviewResponse.json();
  assert.equal(doraPreviewEnvelope.ok, true, doraPreviewEnvelope.stderr);
  assert.equal(doraPreviewEnvelope.result.layers[0].input_format, 'dora');
  assert.equal(doraPreviewEnvelope.result.layers[0].output_format, 'dora');
  assert.equal(doraPreviewEnvelope.result.layers[0].metric_scope, 'pre_normalization_direction');

  const doraBakeResponse = await post(route, {
    script: 'trim_lora.py',
    args: {
      input: doraPath,
      'bake-dora-to-lora': true,
      'base-model-path': basePath,
      'dora-convention': 'toolkit',
      'rank-policy': 'fixed',
      'rank-value': 64,
      json: true,
    },
  });
  assert.equal(doraBakeResponse.status, 200);
  const doraBakeEnvelope = await doraBakeResponse.json();
  assert.equal(doraBakeEnvelope.ok, true, doraBakeEnvelope.stderr);
  assert.equal(doraBakeEnvelope.result.layers[0].baked, true);
  assert.equal(doraBakeEnvelope.result.layers[0].output_format, 'lora');
  assert.equal(doraBakeEnvelope.result.layers[0].metric_scope, 'effective_update');
  assert.equal(doraBakeEnvelope.result.layers[0].dora_convention, 'toolkit');
  assert.equal(doraBakeEnvelope.result.base_model.path, basePath);

  const lohaPreviewResponse = await post(route, {
    script: 'trim_lora.py',
    args: {
      input: lohaPath,
      'convert-to-lora': true,
      'rank-policy': 'fixed',
      'rank-value': 4,
      json: true,
    },
  });
  assert.equal(lohaPreviewResponse.status, 200);
  const lohaPreviewEnvelope = await lohaPreviewResponse.json();
  assert.equal(lohaPreviewEnvelope.ok, true, lohaPreviewEnvelope.stderr);
  assert.equal(lohaPreviewEnvelope.result.layers[0].input_format, 'loha');
  assert.equal(lohaPreviewEnvelope.result.layers[0].output_format, 'lora');

  const unconfirmedWrite = await post(route, {
    script: 'trim_lora.py',
    args: { ...baseArgs, output: outputPath },
  });
  assert.equal(unconfirmedWrite.status, 500);
  assert.equal((await unconfirmedWrite.json()).ok, false);
  assert.equal(fs.existsSync(outputPath), false, 'UI bridge writes must carry a preview token');

  const writeResponse = await post(route, {
    script: 'trim_lora.py',
    args: {
      ...baseArgs,
      output: outputPath,
      'expected-preview': previewEnvelope.result.preview_token,
    },
  });
  assert.equal(writeResponse.status, 200);
  const writeEnvelope = await writeResponse.json();
  assert.equal(writeEnvelope.ok, true, writeEnvelope.stderr);
  assert.equal(writeEnvelope.result.written, true);
  assert.equal(writeEnvelope.result.output_path, outputPath);
  assert.ok(fs.statSync(outputPath).size < fs.statSync(sourcePath).size);
  assert.deepEqual(fs.readFileSync(sourcePath), sourceBefore, 'The original source must remain byte-identical');

  const publishedBeforeDuplicate = fs.readFileSync(outputPath);
  const duplicate = await post(route, {
    script: 'trim_lora.py',
    args: {
      ...baseArgs,
      output: outputPath,
      'expected-preview': previewEnvelope.result.preview_token,
    },
  });
  assert.equal(duplicate.status, 500);
  assert.equal((await duplicate.json()).ok, false);
  assert.deepEqual(fs.readFileSync(outputPath), publishedBeforeDuplicate, 'A duplicate write must not replace output');

  const invalidAdapter = await post(route, {
    script: 'trim_lora.py',
    args: { ...baseArgs, input: unsupportedPath },
  });
  assert.equal(invalidAdapter.status, 500);
  const invalidEnvelope = await invalidAdapter.json();
  assert.equal(invalidEnvelope.ok, false);
  assert.match(invalidEnvelope.stderr, /unsupported|Unrecognized|CP\/Tucker/i);
  assert.deepEqual(fs.readFileSync(sourcePath), sourceBefore);
});
