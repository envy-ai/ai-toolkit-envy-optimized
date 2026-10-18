const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

function loadStopRoute(job, updates) {
  const filename = path.resolve(__dirname, '../src/app/api/jobs/[jobID]/stop/route.ts');
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => {
    if (id === '@/server/prisma') {
      return {
        __esModule: true,
        default: {
          job: {
            findUnique: async () => job,
            update: async ({ data }) => { updates.push(data); return { ...job, ...data }; },
          },
        },
      };
    }
    return originalRequire(id);
  };
  instance._compile(compiled, filename);
  return instance.exports.GET;
}

async function stopJob(t, job, pause = true) {
  const updates = [];
  const signals = [];
  t.mock.method(process, 'kill', (pid, signal) => { signals.push([pid, signal]); return true; });
  const GET = loadStopRoute(job, updates);
  const response = await GET(
    { nextUrl: new URL(`http://example.test/api/jobs/test/stop${pause ? '?saveFirst=1' : ''}`) },
    { params: Promise.resolve({ jobID: 'test' }) },
  );
  return { updates, signals, response };
}

test('zero-step pause cancels pending save and uses immediate stop', async t => {
  const { updates, signals, response } = await stopJob(t, {
    id: 'test', job_type: 'train', status: 'running', step: 0, pid: 999999, save_now: true,
  });
  assert.equal(response.status, 200);
  assert.equal(updates[0].save_now, false);
  assert.equal(updates[0].stop, true);
  assert.equal(updates[1].status, 'stopped');
  assert.deepEqual(signals, [[999999, 'SIGINT']]);
});

test('positive-step pause requests a save without interrupting training', async t => {
  const { updates, signals, response } = await stopJob(t, {
    id: 'test', job_type: 'train', status: 'running', step: 1, pid: 999999,
  });
  assert.equal(response.status, 200);
  assert.equal(updates.length, 1);
  assert.equal(updates[0].save_now, true);
  assert.equal(updates[0].stop, true);
  assert.equal(updates[0].status, 'stopping');
  assert.deepEqual(signals, []);
});

test('zero-step pause still requires an active training job', async t => {
  const { updates, signals, response } = await stopJob(t, {
    id: 'test', job_type: 'train', status: 'stopped', step: 0, pid: null,
  });
  assert.equal(response.status, 409);
  assert.deepEqual(updates, []);
  assert.deepEqual(signals, []);
});

test('ordinary stop clears an old zero-step pause save request', async t => {
  const { updates } = await stopJob(t, {
    id: 'test', job_type: 'train', status: 'stopping', step: 0, pid: null, save_now: true,
  }, false);
  assert.equal(updates[0].save_now, false);
  assert.equal(updates[0].stop, true);
});
