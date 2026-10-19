const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const Module = require('node:module');
const EventEmitter = require('node:events');
const ts = require('typescript');
const { NextRequest } = require('next/server');

const routePath = path.resolve(__dirname, '../src/app/api/jobs/[jobID]/render_samples/route.ts');

function loadRoute({ job, startResult, readResult, startError } = {}) {
  const source = fs.readFileSync(routePath, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const routeModule = new Module(routePath, module);
  routeModule.filename = routePath;
  routeModule.paths = Module._nodeModulePaths(path.dirname(routePath));
  const prisma = { job: { findUnique: async () => job } };
  const jobSamples = {
    readSampleRender: async () => readResult,
    startSampleRender: async () => {
      if (startError) throw startError;
      return startResult;
    },
  };
  const originalRequire = routeModule.require.bind(routeModule);
  routeModule.require = id => id === '@/server/prisma' ? prisma
    : id === '@/server/jobSamples' ? jobSamples : originalRequire(id);
  routeModule._compile(compiled, routePath);
  return routeModule.exports;
}

function loadTypeScript(filename, customRequire) {
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => customRequire(id, originalRequire);
  instance._compile(compiled, filename);
  return instance.exports;
}

function loadLockModule() {
  return loadTypeScript(path.resolve(__dirname, '../cron/sampleRenderLocks.ts'), (id, originalRequire) => originalRequire(id));
}

function loadJobSamples({ root = '', jobRows = [], childProcess = {} } = {}) {
  const filename = path.resolve(__dirname, '../src/server/jobSamples.ts');
  const locks = loadLockModule();
  return loadTypeScript(filename, (id, originalRequire) => id === '@/server/prisma' ? { job: { findMany: async () => jobRows } }
    : id === '@/server/settings' ? {
      getTrainingFolder: async () => root, getHFToken: async () => '', getModelsPath: async () => '',
    }
      : id === '@/paths' ? { TOOLKIT_ROOT: root }
        : id === '../../cron/pythonPath' ? { resolveDetachedPythonPath: () => '/fake/python' }
          : id === '../../cron/sampleRenderLocks' ? locks
            : id === 'child_process' ? { ...originalRequire(id), ...childProcess }
              : originalRequire(id));
}

function renderJob(name, gpuIds = '0') {
  const config = {
    config: {
      name,
      process: [{ sample: { comfy: { api_url: 'http://comfy:8188' } } }],
    },
  };
  return {
    id: `id-${name}`, name, job_type: 'train', status: 'stopped', gpu_ids: gpuIds,
    job_config: JSON.stringify(config),
  };
}

function context(jobID = 'job-1') {
  return { params: Promise.resolve({ jobID }) };
}

function request() {
  return new NextRequest('http://localhost/api/jobs/job-1/render_samples');
}

test('saved sample route reports missing jobs as 404', async () => {
  const route = loadRoute({ job: null });

  const response = await route.POST(request(), context());

  assert.equal(response.status, 404);
  assert.deepEqual(await response.json(), { error: 'Job not found' });
});

test('saved sample route exposes start restrictions as a 409 error response', async () => {
  const route = loadRoute({ job: { id: 'job-1' }, startError: new Error('Render Samples is available for stopped, completed or failed training jobs.') });

  const response = await route.POST(request(), context());
  const body = await response.json();

  assert.equal(response.status, 409);
  assert.equal(body.status, 'error');
  assert.match(body.message, /stopped, completed or failed training jobs/);
  assert.match(body.stack, /Error/);
});

test('sample renderer rejects running, stopping and queued jobs before setup', async () => {
  const { startSampleRender } = loadJobSamples();

  for (const status of ['running', 'stopping', 'queued']) {
    await assert.rejects(
      startSampleRender({ job_type: 'train', status }),
      /stopped, completed or failed training jobs/,
    );
  }
  await assert.rejects(
    startSampleRender({ job_type: 'generate', status: 'stopped' }),
    /stopped, completed or failed training jobs/,
  );
});

test('saved sample rendering allows queued jobs to wait behind the render gate', async () => {
  const root = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'aitk-sample-queued-test-'));
  const job = renderJob('job-a', '0');
  fs.mkdirSync(path.join(root, job.name));
  const spawn = () => {
    const child = new EventEmitter();
    child.pid = process.pid;
    child.unref = () => {};
    process.nextTick(() => child.emit('spawn'));
    return child;
  };
  const execFile = (_python, _args, _options, callback) => callback(null, {
    stdout: '{"status":"running","checkpoint":"saved.safetensors","step":4}',
  });
  const queuedJob = { id: 'id-queued', name: 'queued-job', job_type: 'train', status: 'queued', pid: null };
  const { startSampleRender } = loadJobSamples({
    root,
    jobRows: [queuedJob],
    childProcess: { spawn, execFile },
  });
  try {
    const state = await startSampleRender(job);
    assert.equal(state.status, 'running');
    assert.equal(state.checkpoint, 'saved.safetensors');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('completed jobs with stale PIDs do not block saved sample rendering', async () => {
  const root = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'aitk-sample-stale-pid-test-'));
  const job = renderJob('job-a', '0');
  fs.mkdirSync(path.join(root, job.name));
  const spawn = () => {
    const child = new EventEmitter();
    child.pid = process.pid;
    child.unref = () => {};
    process.nextTick(() => child.emit('spawn'));
    return child;
  };
  const execFile = (_python, _args, _options, callback) => callback(null, {
    stdout: '{"status":"running","checkpoint":"saved.safetensors","step":4}',
  });
  const oldJob = { id: 'id-old', name: 'old-job', job_type: 'train', status: 'completed', pid: process.pid };
  const { startSampleRender } = loadJobSamples({ root, jobRows: [oldJob], childProcess: { spawn, execFile } });
  try {
    const state = await startSampleRender(job);
    assert.equal(state.status, 'running');
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('Comfy render gate serializes jobs even when training GPU IDs differ', async () => {
  const root = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'aitk-sample-gate-test-'));
  fs.mkdirSync(path.join(root, 'job-a'));
  fs.mkdirSync(path.join(root, 'job-b'));
  const spawn = () => {
    const child = new EventEmitter();
    child.pid = process.pid;
    child.unref = () => {};
    process.nextTick(() => child.emit('spawn'));
    return child;
  };
  const execFile = (_python, _args, _options, callback) => callback(null, {
    stdout: '{"status":"running","checkpoint":"saved.safetensors","step":4}',
  });
  const { startSampleRender } = loadJobSamples({ root, childProcess: { spawn, execFile } });
  try {
    await startSampleRender(renderJob('job-a', '0'));
    await assert.rejects(startSampleRender(renderJob('job-b', '1')), /busy/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('renderer launch failure persists an error state and releases GPU locks', async () => {
  const root = fs.mkdtempSync(path.join(require('node:os').tmpdir(), 'aitk-sample-launch-test-'));
  const job = renderJob('job-fail', '0');
  fs.mkdirSync(path.join(root, job.name));
  const spawn = () => {
    const child = new EventEmitter();
    process.nextTick(() => child.emit('error', new Error('spawn denied')));
    return child;
  };
  const execFile = (_python, _args, _options, callback) => callback(null, {
    stdout: '{"status":"running","checkpoint":"saved.safetensors","step":4}',
  });
  const { startSampleRender } = loadJobSamples({ root, childProcess: { spawn, execFile } });
  try {
    await assert.rejects(startSampleRender(job), /spawn denied/);
    const status = JSON.parse(fs.readFileSync(path.join(root, job.name, '.sample_render', 'status.json'), 'utf8'));
    assert.equal(status.status, 'error');
    assert.match(status.message, /spawn denied/);
    assert.match(status.stack, /Error: spawn denied/);
    assert.deepEqual(fs.readdirSync(path.join(root, 'data', 'gpu-launch-locks')), []);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('saved sample route returns the renderer state for status polling', async () => {
  const expected = { status: 'error', message: 'checkpoint metadata failed', error: 'checkpoint metadata failed', stack: 'trace' };
  const route = loadRoute({ job: { id: 'job-1' }, readResult: expected });

  const response = await route.GET(request(), context());

  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), expected);
});
