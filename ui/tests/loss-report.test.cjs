const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const Module = require('node:module');
const { spawnSync } = require('node:child_process');
const { test, before, after } = require('node:test');
const ts = require('typescript');
const sqlite3 = require('sqlite3');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

function load(relative, overrides = {}) {
  const filename = path.resolve(__dirname, relative);
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      esModuleInterop: true,
      target: ts.ScriptTarget.ES2020,
    },
  }).outputText;
  const mod = new Module(filename, module);
  mod.filename = filename;
  mod.paths = Module._nodeModulePaths(path.dirname(filename));
  const original = mod.require.bind(mod);
  mod.require = id => (Object.hasOwn(overrides, id) ? overrides[id] : original(id));
  mod._compile(compiled, filename);
  return mod.exports;
}
const {
  parseLossReportOptions: parse,
  queryLossReport: query,
  queryLossReportDetail: detail,
  all,
} = load('../src/server/lossReport.ts');
let root;
before(() => {
  root = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-loss-report-'));
  const result = spawnSync(
    'conda',
    ['run', '--no-capture-output', '-n', 'ai-toolkit', 'python', 'testing/loss_report_fixture.py', root],
    {
      cwd: path.resolve(__dirname, '../..'),
      encoding: 'utf8',
      timeout: 60000,
    },
  );
  assert.equal(result.status, 0, result.stderr + result.stdout);
});
after(() => fs.rmSync(root, { recursive: true, force: true }));
async function withDb(name, run) {
  const db = await new Promise((resolve, reject) => {
    const instance = new sqlite3.Database(path.join(root, name, 'loss_log.db'), err =>
      err ? reject(err) : resolve(instance),
    );
  });
  try {
    return await run(db);
  } finally {
    await new Promise(resolve => db.close(resolve));
  }
}
const options = queryString => parse(new URLSearchParams(`window=5&${queryString}`));

test('report options reject malformed values and inverted bounds; defaults and empty bounds work', () => {
  assert.deepEqual(parse(new URLSearchParams()), {
    mode: 'threshold',
    target: 'step',
    window: 100,
    threshold: 3,
    top: 20,
    start: null,
    end: null,
    offset: 0,
    limit: 50,
  });
  assert.equal(parse(new URLSearchParams('start=&end=')).start, null);
  for (const input of [
    'window=0',
    'window=1.5',
    'window=10001',
    'threshold=NaN',
    'start=-1',
    'start=8&end=7',
    'limit=201',
    'offset=Infinity',
    'target=bad',
    'mode=bad',
    'top=0',
  ]) {
    assert.throws(() => parse(new URLSearchParams(input)), undefined, input);
  }
});

test('shared job validation accepts default-on logging in normal and specialized modes and validates RNG dependencies', () => {
  const { validateTrainingCapabilities } = load('../src/app/jobs/new/trainingCapabilities.ts');
  const config = {
    config: {
      process: [
        {
          type: 'guidance_distillation',
          model: { arch: 'qwen_image_2' },
          network: { type: 'lora' },
          datasets: [],
          logging: {},
        },
      ],
    },
  };
  assert.deepEqual(validateTrainingCapabilities(config), []);
  config.config.process[0].logging = { record_training_examples: true, record_training_rng: true };
  assert.deepEqual(validateTrainingCapabilities(config), []);
  config.config.process[0].logging.record_training_examples = false;
  assert.match(validateTrainingCapabilities(config).join(' '), /requires per-image/);
  config.config.process[0].logging = { record_training_examples: 'false' };
  assert.match(validateTrainingCapabilities(config).join(' '), /must be a boolean/);
  config.config.process[0].type = 'sd_trainer';
  assert.match(validateTrainingCapabilities(config).join(' '), /must be a boolean/);
  config.config.process[0].logging = { record_training_rng: true };
  assert.deepEqual(validateTrainingCapabilities(config), []);
});
test('new and imported jobs default reporting on and RNG off, preserving explicit opt-outs', () => {
  const defaults = load('../src/app/jobs/new/jobConfig.ts', {
    '@/helpers/basic': { isMac: () => false },
    '@/helpers/defaultSamples': load('../src/helpers/defaultSamples.ts'),
    './trainingCapabilities': load('../src/app/jobs/new/trainingCapabilities.ts'),
  });
  const config = structuredClone(defaults.defaultJobConfig);
  const process = config.config.process[0];
  assert.equal(process.logging.record_training_examples, true);
  assert.equal(process.logging.record_training_rng, false);
  delete process.logging.record_training_examples;
  delete process.logging.record_training_rng;
  defaults.migrateJobConfig(config);
  assert.equal(process.logging.record_training_examples, true);
  assert.equal(process.logging.record_training_rng, false);
  process.logging.record_training_examples = false;
  defaults.migrateJobConfig(config);
  assert.equal(process.logging.record_training_examples, false);
});
test('shared-step inputs appear in step reports and are excluded from individual image spike filters', async () =>
  withDb('shared', async db => {
    const result = await query(db, options('start=20&end=20'));
    assert.equal(result.total, 2);
    assert.ok(result.entries.every(entry => entry.step_loss === 10 && entry.weighted_loss === null && entry.loss_kind === 'unattributed'));
    assert.equal((await query(db, options('target=image&start=20&end=20'))).total, 0);
    const row = await detail(db, 20, 0, 0);
    assert.equal(row.presentation.loss_attribution, 'shared_step');
    assert.equal(row.rng_state, null);
  }));
test('inclusive range retains prior history and step spikes include every image', async () =>
  withDb('spikes', async db => {
    const full = await query(db, options('threshold=3'));
    const range = await query(db, options('start=20&end=20&threshold=3'));
    assert.equal(range.total, 2);
    assert.deepEqual(
      range.entries.map(e => e.step),
      [20, 20],
    );
    assert.equal(range.entries[0].baseline, 1);
    assert.equal(range.entries[0].step_loss, 10);
    assert.equal(range.entries[0].spike_ratio, 10);
    assert.equal(range.entries[0].baseline, full.entries.find(e => e.step === 20).baseline);
    assert.equal(range.recorded, 25);
    assert.equal(range.entries[0].caption, undefined);
    assert.equal(range.entries[0].metadata_json, undefined);
  }));
test('warmup excludes early finite spikes and current step never enters the mean', async () =>
  withDb('spikes', async db => {
    const result = await query(db, parse(new URLSearchParams('start=5&end=5&mode=top')));
    assert.equal(result.total, 0);
    const current = await query(db, options('start=5&end=5&mode=top'));
    assert.equal(current.entries[0].baseline, 1);
    assert.equal(current.entries[0].spike_ratio, 100);
  }));
test('individual spikes filter weighted image loss rather than including the whole step', async () =>
  withDb('spikes', async db => {
    const result = await query(db, options('target=image&start=20&end=20&threshold=3'));
    assert.equal(result.total, 1);
    assert.equal(result.entries[0].item_index, 1);
    assert.equal(result.entries[0].spike_ratio, 18);
  }));
test('top X steps selects distinct steps; top X images selects exposures, with pagination', async () =>
  withDb('spikes', async db => {
    const steps = await query(db, options('mode=top&top=1&start=20'));
    assert.equal(steps.total, 2);
    const images = await query(db, options('mode=top&target=image&top=1&start=20'));
    assert.equal(images.total, 1);
    assert.equal(images.entries[0].item_index, 1);
    const page = await query(db, options('mode=top&top=1&start=20&limit=1&offset=1'));
    assert.equal(page.total, 2);
    assert.equal(page.entries[0].item_index, 1);
    const empty = await query(db, options('mode=top&top=1&start=20&limit=1&offset=2'));
    assert.equal(empty.total, 2);
    assert.deepEqual(empty.entries, []);
  }));
test('zero baselines and nonfinite values are explicit, including nonfinite during warmup', async () => {
  await withDb('zero', async db => {
    const result = await query(db, options('threshold=999999'));
    assert.equal(result.total, 1);
    assert.equal(result.entries[0].spike_kind, 'zero_baseline');
    assert.equal(result.entries[0].spike_ratio, null);
    assert.equal(result.entries[0].step, 20);
  });
  await withDb('nonfinite', async db => {
    const result = await query(db, options('target=image&threshold=999999'));
    assert.equal(result.total, 2);
    assert.deepEqual(
      result.entries.map(e => e.loss_kind),
      ['nan', '+inf'],
    );
    assert.ok(result.entries.every(e => e.spike_kind === 'nonfinite'));
    assert.doesNotThrow(() => JSON.parse(JSON.stringify(result)));
  });
});
test('full record contains caption snapshot and RNG once per batch, missing record returns null', async () =>
  withDb('spikes', async db => {
    const row = await detail(db, 20, 0, 1);
    assert.equal(row.metadata.caption, 'stored training caption');
    assert.equal(row.presentation.crop_width, 64);
    assert.deepEqual(row.rng_state, { cpu: 'example-state' });
    assert.equal(row.loss, 18);
    assert.equal(row.microbatch_size, 2);
    assert.equal(await detail(db, 20, 4, 1), null);
  }));
test('old job database returns a graceful empty report', async () =>
  withDb('legacy', async db => {
    assert.deepEqual(await query(db, options('')), { available: false, total: 0, entries: [], recorded: 0 });
    assert.equal(await detail(db, 0, 0, 0), null);
  }));
test('deleting loss steps cascades example and RNG rows, preserving unaffected history', async () =>
  withDb('spikes', async db => {
    await all(db, 'PRAGMA foreign_keys=ON');
    await all(db, 'DELETE FROM steps WHERE step=22');
    assert.deepEqual(await all(db, 'SELECT COUNT(*) AS n FROM training_examples WHERE step=22'), [{ n: 0 }]);
    assert.deepEqual(await all(db, 'SELECT COUNT(*) AS n FROM training_batches WHERE step=22'), [{ n: 0 }]);
    assert.equal((await query(db, options('start=20&end=20'))).total, 2);
  }));
test('table omits images/captions; grid uses lazy thumbnails and the same selection handler', () => {
  const noop = () => null;
  const { LossReportResults } = load('../src/components/JobLossReport.tsx', {
    '@/utils/api': { apiClient: {} },
    'yet-another-react-lightbox': noop,
    'yet-another-react-lightbox/plugins/captions': noop,
    'yet-another-react-lightbox/plugins/counter': noop,
    'yet-another-react-lightbox/plugins/zoom': noop,
  });
  const entry = {
    step: 20,
    microbatch: 0,
    item_index: 1,
    relative_path: 'dataset/a.png',
    image_url: '/api/img/a.png',
    loss: 18,
    weighted_loss: 18,
    step_loss: 10,
    baseline: 1,
    spike_ratio: 10,
    spike_kind: 'finite',
    presentation: { crop_width: 64, crop_height: 32 },
    caption: 'do not show me',
  };
  const table = renderToStaticMarkup(
    React.createElement(LossReportResults, { entries: [entry], grid: false, onSelect: noop }),
  );
  assert.match(table, /dataset\/a.png/);
  assert.match(table, /10.00×/);
  assert.doesNotMatch(table, /<img|do not show me|caption/i);
  let selected;
  const gridElement = LossReportResults({
    entries: [entry],
    grid: true,
    onSelect: index => {
      selected = index;
    },
  });
  gridElement.props.children[0].props.onClick();
  assert.equal(selected, 0);
  const grid = renderToStaticMarkup(gridElement);
  assert.match(grid, /loading="lazy"/);
  assert.match(grid, /a.png\?thumb=1/);
});

test('report API validates bounds, decorates relative paths, preserves missing-image details and handles missing jobs', async () => {
  const next = { NextResponse: { json: (body, init) => ({ body, status: init?.status ?? 200 }) } };
  const shared = load('../src/server/lossReport.ts');
  const { GET } = load('../src/app/api/jobs/[jobID]/loss-report/route.ts', {
    'next/server': next,
    '@/server/prisma': {
      job: { findUnique: async ({ where }) => (where.id === 'missing' ? null : { name: where.id }) },
    },
    '@/server/settings': {
      getTrainingFolder: async () => root,
      getDatasetsRoot: async () => path.join(root, 'datasets'),
      getDataRoot: async () => path.join(root, 'data'),
    },
    '@/server/lossReport': shared,
  });
  const request = search => ({ nextUrl: new URL(`http://example.test/api?${search}`) });
  const params = jobID => ({ params: Promise.resolve({ jobID }) });
  assert.equal((await GET(request(''), params('missing'))).status, 404);
  assert.equal((await GET(request('start=10&end=5'), params('spikes'))).status, 400);
  assert.equal((await GET(request('detail=1&step=20'), params('spikes'))).status, 400);
  const list = await GET(request('window=5&start=20&end=20'), params('spikes'));
  assert.equal(list.status, 200);
  assert.equal(list.body.total, 2);
  assert.equal(list.body.entries[0].relative_path, 'image-0.png');
  assert.match(list.body.entries[0].image_url, /^\/api\/img\//);
  const record = await GET(request('detail=1&step=20&microbatch=0&item=1'), params('spikes'));
  assert.equal(record.status, 200);
  assert.equal(record.body.source_available, false);
  assert.equal(record.body.metadata.caption, 'stored training caption');
  assert.equal((await GET(request('detail=1&step=100&microbatch=0&item=0'), params('spikes'))).status, 404);
  assert.equal((await GET(request(''), params('no-log'))).body.available, false);
});

test('loss DELETE endpoint explicitly removes example/RNG rows without SQLite FK enforcement, and unused metadata', async () => {
  const { DELETE } = load('../src/app/api/jobs/[jobID]/loss/route.ts', {
    'next/server': { NextResponse: { json: (body, init) => ({ body, status: init?.status ?? 200 }) } },
    '@/server/prisma': { job: { findUnique: async () => ({ name: 'spikes' }) } },
    '@/server/settings': { getTrainingFolder: async () => root },
  });
  const result = await DELETE(
    { json: async () => ({ min_step: 20, max_step: 20 }) },
    { params: Promise.resolve({ jobID: 'spikes' }) },
  );
  assert.equal(result.status, 200);
  await withDb('spikes', async db => {
    assert.equal((await all(db, 'SELECT COUNT(*) AS n FROM training_examples WHERE step=20'))[0].n, 0);
    assert.equal((await all(db, 'SELECT COUNT(*) AS n FROM training_batches WHERE step=20'))[0].n, 0);
    assert.equal((await all(db, 'SELECT COUNT(*) AS n FROM steps WHERE step=20'))[0].n, 0);
    assert.equal(
      (
        await all(
          db,
          'SELECT COUNT(*) AS n FROM training_example_metadata WHERE NOT EXISTS (SELECT 1 FROM training_examples WHERE metadata_id=training_example_metadata.id)',
        )
      )[0].n,
      0,
    );
  });
});
