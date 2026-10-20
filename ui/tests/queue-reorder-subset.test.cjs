const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');
const { NextRequest } = require('next/server');

const routePath = path.resolve(__dirname, '../src/app/api/queue/[queueID]/reorder/route.ts');

function cloneJobs(jobs) {
  return jobs.map(job => ({ ...job }));
}

function createRoute(initialJobs, { transitionId } = {}) {
  const jobs = cloneJobs(initialJobs);
  const writes = [];
  const makeClient = working => ({
    job: {
      findMany: async ({ where }) => working
        .filter(job => job.status === where.status && job.gpu_ids === where.gpu_ids)
        .sort((a, b) => a.queue_position - b.queue_position || a.id.localeCompare(b.id))
        .map(({ id, queue_position }) => ({ id, queue_position })),
      updateMany: async ({ where, data }) => {
        writes.push({ where: { ...where }, data: { ...data } });
        const job = working.find(item => item.id === where.id);
        if (job?.id === transitionId) job.status = 'running';
        if (
          !job ||
          job.status !== where.status ||
          job.gpu_ids !== where.gpu_ids ||
          job.queue_position !== where.queue_position ||
          job.id === transitionId
        ) return { count: 0 };
        job.queue_position = data.queue_position;
        return { count: 1 };
      },
    },
  });
  const prisma = {
    job: makeClient(jobs).job,
    $transaction: async callback => {
      const working = cloneJobs(jobs);
      const result = await callback(makeClient(working));
      for (const row of working) Object.assign(jobs.find(job => job.id === row.id), row);
      return result;
    },
  };
  const compiled = ts.transpileModule(fs.readFileSync(routePath, 'utf8'), {
    fileName: routePath,
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const instance = new Module(routePath, module);
  instance.filename = routePath;
  instance.paths = Module._nodeModulePaths(path.dirname(routePath));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => {
    if (id === '@/server/prisma') return { __esModule: true, default: prisma };
    if (id === '@/server/apiCache') return { invalidateCache: () => {} };
    return originalRequire(id);
  };
  instance._compile(compiled, routePath);
  return { ...instance.exports, jobs, writes };
}

function fixtureJobs() {
  return [
    { id: 'running', name: 'running', status: 'running', gpu_ids: '0', queue_position: 0 },
    { id: 'stopping', name: 'stopping', status: 'stopping', gpu_ids: '0', queue_position: 500 },
    { id: 'A', name: 'A', status: 'queued', gpu_ids: '0', queue_position: 1000 },
    { id: 'B', name: 'B', status: 'queued', gpu_ids: '0', queue_position: 2000 },
    { id: 'C', name: 'C', status: 'queued', gpu_ids: '0', queue_position: 3000 },
    { id: 'D', name: 'D', status: 'queued', gpu_ids: '0', queue_position: 4000 },
    { id: 'other-queue', name: 'other queue', status: 'queued', gpu_ids: '1', queue_position: 1000 },
  ];
}

async function patch(route, queueID, orderedJobIds) {
  const request = new NextRequest(`http://localhost/api/queue/${queueID}/reorder`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ orderedJobIds }),
  });
  const originalConsoleError = console.error;
  console.error = () => {};
  try {
    return await route.PATCH(request, { params: Promise.resolve({ queueID }) });
  } finally {
    console.error = originalConsoleError;
  }
}

function positions(jobs, ids) {
  return Object.fromEntries(ids.map(id => [id, jobs.find(job => job.id === id).queue_position]));
}

function loadQueueUtils(apiClient) {
  const queueUtilsPath = path.resolve(__dirname, '../src/utils/queue.ts');
  const compiled = ts.transpileModule(fs.readFileSync(queueUtilsPath, 'utf8'), {
    fileName: queueUtilsPath,
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const instance = new Module(queueUtilsPath, module);
  instance.filename = queueUtilsPath;
  instance.paths = Module._nodeModulePaths(path.dirname(queueUtilsPath));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => id === '@/utils/api' ? { apiClient } : originalRequire(id);
  instance._compile(compiled, queueUtilsPath);
  return instance.exports;
}

test('partial visible reorder merges into only queued slots and leaves running/stopping rows untouched', async () => {
  const route = createRoute(fixtureJobs());
  const response = await patch(route, '0', ['D', 'B']);
  assert.equal(response.status, 200);
  const body = await response.json();
  assert.equal(body.success, true);
  assert.deepEqual(positions(route.jobs, ['running', 'stopping', 'A', 'B', 'C', 'D', 'other-queue']), {
    running: 0,
    stopping: 500,
    A: 1000,
    D: 2000,
    C: 3000,
    B: 4000,
    'other-queue': 1000,
  });
  assert.deepEqual(route.writes.map(write => write.where.id), ['A', 'D', 'C', 'B']);
  assert.ok(route.writes.every(write => write.where.status === 'queued' && write.where.gpu_ids === '0'));
  assert.ok(route.writes.every(write => !['running', 'stopping'].includes(write.where.id)));
});

test('a complete queue order remains valid', async () => {
  const route = createRoute(fixtureJobs());
  const response = await patch(route, '0', ['C', 'D', 'A', 'B']);
  assert.equal(response.status, 200);
  assert.deepEqual(positions(route.jobs, ['A', 'B', 'C', 'D']), {
    C: 1000,
    D: 2000,
    A: 3000,
    B: 4000,
  });
});

test('tied starting positions use deterministic ID order before merging the visible subset', async () => {
  const tiedJobs = fixtureJobs().map(job => ({ ...job }));
  tiedJobs.find(job => job.id === 'B').queue_position = 1000;
  tiedJobs.find(job => job.id === 'D').queue_position = 3000;
  tiedJobs.find(job => job.id === 'C').queue_position = 3000;
  tiedJobs.reverse();
  const route = createRoute(tiedJobs);
  const response = await patch(route, '0', ['D', 'B']);
  assert.equal(response.status, 200);
  assert.deepEqual(positions(route.jobs, ['A', 'B', 'C', 'D']), {
    A: 1000,
    D: 2000,
    C: 3000,
    B: 4000,
  });
});

test('duplicates, nonqueued jobs and jobs from another queue are rejected without writes', async () => {
  for (const ids of [
    ['B', 'B'],
    ['B', 'running'],
    ['B', 'other-queue'],
    ['B', 'missing'],
  ]) {
    const route = createRoute(fixtureJobs());
    const before = positions(route.jobs, ['running', 'stopping', 'A', 'B', 'C', 'D', 'other-queue']);
    const response = await patch(route, '0', ids);
    assert.equal(response.status, 400, ids.join(','));
    assert.deepEqual(positions(route.jobs, Object.keys(before)), before);
    assert.equal(route.writes.length, 0);
  }
});

test('a concurrent queued-to-running transition aborts the transaction atomically', async () => {
  const route = createRoute(fixtureJobs(), { transitionId: 'B' });
  const before = positions(route.jobs, ['running', 'stopping', 'A', 'B', 'C', 'D']);
  const response = await patch(route, '0', ['D', 'B']);
  assert.equal(response.status, 409);
  assert.deepEqual(positions(route.jobs, Object.keys(before)), before);
  assert.ok(route.writes.every(write => write.where.status === 'queued'));
  assert.ok(route.writes.every(write => Number.isFinite(write.where.queue_position)));
});

test('client submits the visible subset and returns authoritative merged positions', async () => {
  const visibleIds = ['D', 'B'];
  const mergedPositions = [
    { id: 'A', queue_position: 1000 },
    { id: 'D', queue_position: 2000 },
    { id: 'C', queue_position: 3000 },
    { id: 'B', queue_position: 4000 },
  ];
  const calls = [];
  const queueUtils = loadQueueUtils({
    patch: async (url, body) => {
      calls.push({ url, body });
      return { data: { success: true, positions: mergedPositions } };
    },
  });
  assert.deepEqual(await queueUtils.reorderQueueJobs('0', visibleIds), mergedPositions);
  assert.deepEqual(calls, [{ url: '/api/queue/0/reorder', body: { orderedJobIds: visibleIds } }]);
});
