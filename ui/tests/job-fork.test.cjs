const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

const routePath = path.resolve(__dirname, '../src/app/api/jobs/route.ts');

function loadRoute(overrides) {
  const filename = routePath;
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
  instance.require = id => Object.hasOwn(overrides, id) ? overrides[id] : originalRequire(id);
  instance._compile(compiled, filename);
  return instance.exports;
}

function routeWithForkMocks({ createJobFork, previewJobFork, prisma } = {}) {
  class JobForkError extends Error {
    constructor(message, status = 400) {
      super(message);
      this.status = status;
    }
  }
  const client = prisma || {
    settings: { findFirst: async () => null },
    job: {
      aggregate: async () => ({ _max: { queue_position: 0 } }),
      create: async ({ data }) => ({ id: 'created', ...data }),
    },
  };
  return loadRoute({
    '@/server/prisma': { __esModule: true, default: client },
    '@/helpers/basic': { isMac: () => false },
    '@/server/apiCache': { cached: callback => callback() },
    '@/helpers/sampleOnlyJobConfig': { mergeSampleOnlyJobConfig: (_old, next) => next },
    '@/app/jobs/new/trainingCapabilities': { validateTrainingCapabilities: () => [] },
    '@/app/jobs/new/sliderspace': { validateSliderSpace: () => [] },
    '@/server/jobFork': {
      createJobFork: createJobFork || (async (_id, _name, _config, create) => create({ step: 0 })),
      previewJobFork: previewJobFork || (async id => ({ id, name: 'source_fork' })),
      JobForkError,
    },
  });
}

function loadJobFork() {
  const filename = path.resolve(__dirname, '../src/server/jobFork.ts');
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => {
    if (id === '@/server/prisma') return { __esModule: true, default: {} };
    if (id === '@/server/settings') return { getTrainingFolder: async () => '/tmp' };
    if (id === '@/paths') return { TOOLKIT_ROOT: path.resolve(__dirname, '../..') };
    if (id === '../../cron/pythonPath') return { resolvePythonPath: () => 'python' };
    return originalRequire(id);
  };
  instance._compile(compiled, filename);
  return instance.exports;
}

const bodyRequest = body => new Request('http://localhost/api/jobs', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

test('GET forkId returns the inspected fork preview', async () => {
  const calls = [];
  const route = routeWithForkMocks({
    previewJobFork: async id => {
      calls.push(id);
      return { id, name: 'source_fork', fork_snapshot: { step: 32, checkpoint: 'source_000000032.safetensors' } };
    },
  });
  const response = await route.GET(new Request('http://localhost/api/jobs?forkId=source-id'));
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), {
    id: 'source-id', name: 'source_fork',
    fork_snapshot: { step: 32, checkpoint: 'source_000000032.safetensors' },
  });
  assert.deepEqual(calls, ['source-id']);
});

test('POST fork_from_id creates a new stopped row at the inspected checkpoint step', async () => {
  let input;
  const route = routeWithForkMocks({
    createJobFork: async (id, name, config, create) => {
      input = { id, name, config };
      return create({ step: 32, checkpoint: 'source_000000032.safetensors' });
    },
  });
  const config = { config: { name: 'forked-job', process: [{ training_folder: '/tmp/training' }] } };
  const response = await route.POST(bodyRequest({
    fork_from_id: 'source-id', name: 'forked-job', job_config: config,
    gpu_ids: '0', job_type: 'train',
  }));
  assert.equal(response.status, 200);
  const created = await response.json();
  assert.equal(created.name, 'forked-job');
  assert.equal(created.step, 32);
  assert.equal(created.status, 'stopped');
  assert.equal(created.queue_position, 1000);
  assert.deepEqual(input, { id: 'source-id', name: 'forked-job', config });
});

test('fork requests cannot update an existing job or create a non-training job', async () => {
  let createCalls = 0;
  const route = routeWithForkMocks({
    createJobFork: async () => { createCalls += 1; return {}; },
  });
  const config = { config: { name: 'fork', process: [{}] } };
  let response = await route.POST(bodyRequest({
    id: 'existing-id', fork_from_id: 'source-id', name: 'fork', job_config: config,
  }));
  assert.equal(response.status, 400);
  response = await route.POST(bodyRequest({
    fork_from_id: 'source-id', name: 'fork', job_type: 'generate', job_config: config,
  }));
  assert.equal(response.status, 400);
  assert.equal(createCalls, 0);
});

test('source safety requires a stopped training job and rejects live or invalid process records', () => {
  const { assertForkSourceSafe, JobForkError } = loadJobFork();
  const temporaryRoot = path.resolve(__dirname, '../../tmp');
  fs.mkdirSync(temporaryRoot, { recursive: true });
  const folder = fs.mkdtempSync(path.join(temporaryRoot, 'aitk-job-fork-safe-'));
  try {
    for (const status of ['stopped', 'completed', 'error']) {
      assert.doesNotThrow(() => assertForkSourceSafe({ job_type: 'train', status, pid: null }, folder));
    }
    assert.throws(
      () => assertForkSourceSafe({ job_type: 'train', status: 'running', pid: null }, folder),
      error => error instanceof JobForkError && error.status === 409,
    );
    assert.throws(
      () => assertForkSourceSafe({ job_type: 'train', status: 'stopped', pid: process.pid }, folder),
      /still alive/,
    );
    fs.writeFileSync(path.join(folder, 'pid.txt'), 'not-a-pid');
    assert.throws(
      () => assertForkSourceSafe({ job_type: 'train', status: 'stopped', pid: null }, folder),
      /Invalid source process record/,
    );
  } finally {
    fs.rmSync(folder, { recursive: true, force: true });
  }
});
