const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

const routePath = path.resolve(__dirname, '../src/app/api/jobs/route.ts');
const searchHelpers = require('../src/utils/search.ts');

function loadJobsHook(apiGet) {
  const slots = [];
  let index = 0;
  const react = {
    useState(initial) {
      const slotIndex = index++;
      if (!slots[slotIndex]) slots[slotIndex] = { value: typeof initial === 'function' ? initial() : initial };
      const slot = slots[slotIndex];
      return [slot.value, update => { slot.value = typeof update === 'function' ? update(slot.value) : update; }];
    },
    useRef(initial) {
      const slotIndex = index++;
      if (!slots[slotIndex]) slots[slotIndex] = { value: { current: initial } };
      return slots[slotIndex].value;
    },
    useEffect(effect, dependencies) {
      const slotIndex = index++;
      if (!slots[slotIndex]) slots[slotIndex] = {};
      const slot = slots[slotIndex];
      if (!slot.dependencies || dependencies.some((dependency, position) => dependency !== slot.dependencies[position])) {
        slot.cleanup?.();
        slot.cleanup = effect();
        slot.dependencies = [...dependencies];
      }
    },
  };
  const filename = path.resolve(__dirname, '../src/hooks/useJobsList.tsx');
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
    },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => {
    if (id === 'react') return react;
    if (id === '@/utils/api') return { apiClient: { get: apiGet } };
    if (id === '@/hooks/usePollLoop') return () => {};
    if (id === '@/utils/search') return searchHelpers;
    return originalRequire(id);
  };
  instance._compile(compiled, filename);
  return props => {
    index = 0;
    return instance.exports.default(props);
  };
}

function loadRoute(jobRows) {
  const calls = [];
  const prisma = {
    settings: { findFirst: async () => null },
    job: {
      findMany: async args => {
        calls.push(args);
        let rows = [...jobRows];
        const where = args.where || {};
        if (where.job_type) rows = rows.filter(job => job.job_type === where.job_type);
        if (where.status?.in) rows = rows.filter(job => where.status.in.includes(job.status));
        if (where.name?.contains) rows = rows.filter(job => job.name.includes(where.name.contains));
        if (where.id?.in) rows = rows.filter(job => where.id.in.includes(job.id));
        if (where.OR) {
          const [older, sameDateEarlierId] = where.OR;
          rows = rows.filter(job =>
            job.created_at < older.created_at.lt ||
            (job.created_at.getTime() === sameDateEarlierId.created_at.getTime() && job.id < sameDateEarlierId.id.lt),
          );
        }
        rows.sort((a, b) => b.created_at - a.created_at || b.id.localeCompare(a.id));
        if (args.take !== undefined) rows = rows.slice(0, args.take);
        if (args.select) rows = rows.map(({ id, name, created_at }) => ({ id, name, created_at }));
        return rows;
      },
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
    if (id === '@/helpers/basic') return { isMac: () => false };
    if (id === '@/server/apiCache') return { cached: callback => callback() };
    if (id === '@/helpers/sampleOnlyJobConfig') return { mergeSampleOnlyJobConfig: (_old, next) => next };
    if (id === '@/app/jobs/new/trainingCapabilities') return { validateTrainingCapabilities: () => [] };
    if (id === '@/app/jobs/new/sliderspace') return { validateSliderSpace: () => [] };
    if (id === '../../../../cron/captionApiConfig') return {
      API_CAPTIONER: 'APICaptioner', API_CAPTION_QUEUE: 'cpu',
      apiCaptionSettingsSchema: { parse: value => value },
      assertNoCaptionCredentials: () => {}, isApiCaptionJob: () => false,
    };
    if (id === '../../../../cron/captionApiCredentials') return { readCaptionApiCredential: async () => '' };
    if (id === '@/server/settings') return { getDatasetsRoot: async () => '' };
    if (id === '@/server/jobFork') return {
      createJobFork: async () => ({}),
      previewJobFork: async id => ({ id }),
      JobForkError: class JobForkError extends Error {},
    };
    if (id === '@/utils/search') return originalRequire(path.resolve(__dirname, '../src/utils/search.ts'));
    return originalRequire(id);
  };
  instance._compile(compiled, routePath);
  return { ...instance.exports, calls };
}

function seedJobs() {
  return Array.from({ length: 205 }, (_, index) => ({
    id: `job-${String(index).padStart(3, '0')}`,
    name: `ordinary job ${index}`,
    job_type: 'train',
    status: index % 2 ? 'stopped' : 'running',
    created_at: new Date(Date.UTC(2025, 0, 1) + Math.floor(index / 2) * 60_000),
    job_config: '{}',
  }));
}

function sortedIds(jobs) {
  return [...jobs].sort((a, b) => b.created_at - a.created_at || b.id.localeCompare(a.id)).map(job => job.id);
}

async function get(route, params = {}) {
  const url = new URL('http://fixture.local/api/jobs');
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
  return route.GET(new Request(url));
}

test('jobs pagination returns stable newest-first pages across timestamp ties and preserves the full-list API', async () => {
  const jobs = seedJobs();
  const route = loadRoute(jobs);
  const expected = sortedIds(jobs);

  const firstResponse = await get(route, { limit: '100' });
  assert.equal(firstResponse.status, 200);
  const first = await firstResponse.json();
  assert.deepEqual(first.jobs.map(job => job.id), expected.slice(0, 100));
  assert.ok(first.jobs.every(job => typeof job.job_config === 'string'));
  assert.equal(first.hasMore, true);
  assert.equal(typeof first.nextCursor, 'string');

  const secondResponse = await get(route, { limit: '100', cursor: first.nextCursor });
  const second = await secondResponse.json();
  assert.deepEqual(second.jobs.map(job => job.id), expected.slice(100, 200));
  assert.equal(second.hasMore, true);
  assert.equal(typeof second.nextCursor, 'string');

  const third = await (await get(route, { limit: '100', cursor: second.nextCursor })).json();
  assert.deepEqual(third.jobs.map(job => job.id), expected.slice(200));
  assert.equal(third.hasMore, false);
  assert.equal(third.nextCursor, null);
  assert.deepEqual([...first.jobs, ...second.jobs, ...third.jobs].map(job => job.id), expected);

  const legacy = await (await get(route)).json();
  assert.equal(legacy.jobs.length, 205);
  assert.equal('hasMore' in legacy, false);
  assert.deepEqual(legacy.jobs.map(job => job.id), expected);
});

test('paginated search matches all jobs with literal terms and AND semantics, including older matches', async () => {
  const jobs = seedJobs().map((job, index) => ({
    ...job,
    name:
      index <= 150 ? `Alpha archive %_needle ${index}`
        : index <= 199 ? `Alpha archive ${index}`
          : `archive %_needle ${index}`,
  }));
  const route = loadRoute(jobs);
  const search = 'NEEDLE   %_ ALPHA';

  const first = await (await get(route, { limit: '100', search })).json();
  const expectedMatches = sortedIds(jobs.filter(job => job.name.includes('%_needle') && job.name.includes('Alpha')));
  assert.equal(expectedMatches.length, 151);
  assert.deepEqual(first.jobs.map(job => job.id), expectedMatches.slice(0, 100));
  assert.equal(first.hasMore, true);
  assert.equal(route.calls.some(call => call.select?.id && call.select?.name && call.take === undefined), true);

  const second = await (await get(route, { limit: '100', cursor: first.nextCursor, search })).json();
  assert.deepEqual(second.jobs.map(job => job.id), expectedMatches.slice(100));
  assert.equal(second.hasMore, false);
  assert.ok(second.jobs.some(job => job.id === 'job-000'), 'the oldest match is outside the newest 100 jobs');

  const expectedNameMatches = sortedIds(jobs.filter(job => job.name.includes('%_needle')));
  const literalNamePage = await (await get(route, { limit: '100', name: '%_needle' })).json();
  assert.deepEqual(literalNamePage.jobs.map(job => job.id), expectedNameMatches.slice(0, 100));
  assert.equal(literalNamePage.hasMore, true);
  const legacyName = await (await get(route, { name: '%_needle' })).json();
  assert.deepEqual(legacyName.jobs.map(job => job.id), expectedNameMatches);
});

test('paged job hook loads another 100, refreshes the loaded depth, and leaves active lists unpaged', async () => {
  const requests = [];
  const rows = length => Array.from({ length }, (_, index) => ({ id: `job-${index}`, name: `job ${index}` }));
  const render = loadJobsHook(async (_url, options) => {
    requests.push(options);
    const limit = Number(options.params.limit || 1000);
    const offset = options.params.cursor ? 100 : 0;
    return {
      data: options.params.limit
        ? {
            jobs: Array.from({ length: Math.min(limit, 250 - offset) }, (_, index) => ({
              id: `job-${offset + index}`,
              name: `job ${offset + index}`,
            })),
            hasMore: limit + offset < 250,
            nextCursor: limit + offset < 250 ? `cursor-${limit + offset}` : null,
          }
        : { jobs: rows(250) },
    };
  });

  let state = render({ pageSize: 100, search: ' two  words ' });
  await state.refreshJobs();
  state = render({ pageSize: 100, search: 'two words' });
  assert.equal(requests[0].params.limit, '100');
  assert.equal(requests[0].params.search, 'two words');
  assert.equal(state.jobs.length, 100);
  assert.equal(state.hasMore, true);

  await state.loadMore();
  state = render({ pageSize: 100, search: 'two words' });
  assert.equal(requests[1].params.limit, '100');
  assert.equal(requests[1].params.cursor, 'cursor-100');
  assert.equal(state.jobs.length, 200);
  assert.equal(state.jobs[199].id, 'job-199');

  await state.refreshJobs();
  state = render({ pageSize: 100, search: 'two words' });
  assert.equal(requests[2].params.limit, '200');
  assert.equal(state.jobs.length, 200);

  const active = render({ onlyActive: true, reloadInterval: 5_000 });
  await active.refreshJobs();
  assert.equal('limit' in requests[3].params, false);
  assert.equal('search' in requests[3].params, false);
});

test('paged job hook aborts and ignores a stale search response', async () => {
  const pending = [];
  const render = loadJobsHook((_url, options) => new Promise(resolve => pending.push({ options, resolve })));
  let state = render({ pageSize: 100, search: 'first' });
  const oldRequest = state.refreshJobs();
  assert.equal(pending.length, 1);

  state = render({ pageSize: 100, search: 'second' });
  assert.equal(pending[0].options.signal.aborted, true);
  const currentRequest = state.refreshJobs();
  assert.equal(pending.length, 2);
  pending[1].resolve({ data: { jobs: [{ id: 'current', name: 'second result' }], hasMore: false, nextCursor: null } });
  await currentRequest;
  state = render({ pageSize: 100, search: 'second' });
  assert.deepEqual(state.jobs.map(job => job.id), ['current']);

  pending[0].resolve({ data: { jobs: [{ id: 'stale', name: 'first result' }], hasMore: false, nextCursor: null } });
  await oldRequest;
  state = render({ pageSize: 100, search: 'second' });
  assert.deepEqual(state.jobs.map(job => job.id), ['current']);
});

test('forced refresh cancels an older poll while preserving query and loaded depth', async () => {
  const requests = [];
  const render = loadJobsHook((_url, options) => new Promise(resolve => requests.push({ options, resolve })));
  const props = { pageSize: 100, search: '  qwen   portrait ' };
  let state = render(props);

  const firstPage = state.refreshJobs();
  assert.equal(requests[0].options.params.limit, '100');
  assert.equal(requests[0].options.params.search, 'qwen portrait');
  requests[0].resolve({
    data: {
      jobs: Array.from({ length: 100 }, (_, index) => ({ id: `page-one-${index}` })),
      hasMore: true,
      nextCursor: 'cursor-100',
    },
  });
  await firstPage;

  state = render(props);
  const secondPage = state.loadMore();
  assert.equal(requests[1].options.params.limit, '100');
  assert.equal(requests[1].options.params.cursor, 'cursor-100');
  requests[1].resolve({
    data: {
      jobs: Array.from({ length: 100 }, (_, index) => ({ id: `page-two-${index}` })),
      hasMore: true,
      nextCursor: 'cursor-200',
    },
  });
  await secondPage;

  state = render(props);
  const olderPoll = state.refreshJobs();
  assert.equal(requests[2].options.params.limit, '200');
  assert.equal(requests[2].options.params.search, 'qwen portrait');

  state = render(props);
  const forcedRefresh = state.refreshJobs({ cancelPending: true });
  assert.equal(requests[2].options.signal.aborted, true);
  assert.equal(requests.length, 4);
  assert.equal(requests[3].options.params.limit, '200');
  assert.equal(requests[3].options.params.search, 'qwen portrait');
  const authoritative = Array.from({ length: 200 }, (_, index) => ({ id: `fresh-${index}` }));
  requests[3].resolve({ data: { jobs: authoritative, hasMore: false, nextCursor: null } });
  await forcedRefresh;

  requests[2].resolve({ data: { jobs: [{ id: 'stale-poll-result' }], hasMore: false, nextCursor: null } });
  await olderPoll;
  state = render(props);
  assert.deepEqual(state.jobs, authoritative);
  assert.equal(state.hasMore, false);
});
