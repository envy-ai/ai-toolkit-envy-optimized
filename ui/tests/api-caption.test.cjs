const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const Module = require('node:module');
const EventEmitter = require('node:events');
const test = require('node:test');
const ts = require('typescript');
const { NextRequest } = require('next/server');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

const projectRoot = path.resolve(__dirname, '../..');
const tempRoot = path.join(projectRoot, 'tmp');
fs.mkdirSync(tempRoot, { recursive: true });

function loadTypeScript(filename, customRequire) {
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => customRequire(id, originalRequire);
  instance._compile(compiled, filename);
  return instance.exports;
}

function loadConfig() {
  return loadTypeScript(path.resolve(__dirname, '../cron/captionApiConfig.ts'), (_id, originalRequire) => originalRequire(_id));
}

function loadCredentials(root) {
  const config = loadConfig();
  return loadTypeScript(path.resolve(__dirname, '../cron/captionApiCredentials.ts'), (id, originalRequire) =>
    id === './captionApiConfig' ? config : originalRequire(id));
}

function apiCaptionConfig(root, credentialId) {
  return {
    job: 'extension',
    config: { name: 'API caption fixture', process: [{
      type: 'APICaptioner', device: 'cpu',
      caption: {
        api_url: 'http://127.0.0.1:34343/v1', model_name_or_path: 'fixture-model',
        system_prompt: 'Fixture system', caption_prompt: 'Fixture prompt',
        max_new_tokens: 4096, api_timeout: 3, api_max_tokens_field: 'max_tokens',
        path_to_caption: root, caption_extension: 'txt', extensions: ['png'], recaption: false,
        ...(credentialId ? { api_credential_id: credentialId } : {}),
      },
    }] },
  };
}

test('caption settings accept the 4096 limit and reject invalid settings or credential fields', () => {
  const config = loadConfig();
  const valid = config.apiCaptionSettingsSchema.parse(apiCaptionConfig('/tmp').config.process[0].caption);
  assert.equal(valid.max_new_tokens, 4096);
  assert.equal(config.apiCaptionSettingsSchema.parse({ ...valid, max_new_tokens: 16384 }).max_new_tokens, 16384);
  assert.equal(config.captionApiEndpoint('http://localhost:5005/v1/'), 'http://localhost:5005/v1/chat/completions');
  for (const limit of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => config.apiCaptionSettingsSchema.parse({ ...valid, max_new_tokens: limit }));
  }
  assert.throws(() => config.apiCaptionSettingsSchema.parse({ ...valid, api_url: 'https://user:pass@example.test/v1' }));
  assert.throws(() => config.assertNoCaptionCredentials({ config: { process: [{ caption: { api_key: 'should-not-persist' } }] } }), /never in job configuration/);
});

test('private caption credential is owner-only, URL-bound, and invalidated by a host change', async () => {
  const root = fs.mkdtempSync(path.join(tempRoot, 'api-caption-key-test-'));
  const credentials = loadCredentials(root);
  const oldDirectory = process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
  delete process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
  const key = 'fixture-secret-never-in-job-config';
  try {
    const id = await credentials.createCaptionApiCredential(root, 'http://127.0.0.1:34343/v1', key);
    const directory = path.join(root, 'data', 'caption-api', 'credentials');
    const credentialPath = path.join(directory, `${id}.json`);
    assert.equal(await credentials.readCaptionApiCredential(root, id, 'http://127.0.0.1:34343/v1/'), key);
    await assert.rejects(credentials.readCaptionApiCredential(root, id, 'http://127.0.0.1:34344/v1'), /different API URL/);
    const stat = fs.statSync(credentialPath);
    assert.equal(stat.mode & 0o077, 0);
    assert.match(fs.readFileSync(credentialPath, 'utf8'), new RegExp(key));
    await assert.rejects(credentials.createCaptionApiCredential(root, 'http://localhost/v1', 'bad\nheader'), /valid API key/);
  } finally {
    if (oldDirectory === undefined) delete process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
    else process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR = oldDirectory;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('private credential endpoint returns only a reference and never logs the submitted key', async () => {
  const root = fs.mkdtempSync(path.join(tempRoot, 'api-caption-credential-route-test-'));
  const credentials = loadCredentials(root);
  const filename = path.resolve(__dirname, '../src/app/api/caption/credentials/route.ts');
  const route = loadTypeScript(filename, (id, originalRequire) => {
    if (id === '@/paths') return { TOOLKIT_ROOT: root };
    if (id === '../../../../../cron/captionApiCredentials') return credentials;
    if (id === '../../../../../cron/captionApiConfig') return loadConfig();
    return originalRequire(id);
  });
  const key = 'credential-route-secret-fixture';
  const oldDirectory = process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
  delete process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
  const logged = [];
  const originalError = console.error;
  console.error = (...args) => logged.push(args.map(String).join(' '));
  try {
    const request = new NextRequest('http://localhost/api/caption/credentials', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ apiUrl: 'http://127.0.0.1:34343/v1', apiKey: key }),
    });
    const response = await route.POST(request);
    const body = await response.json();
    assert.equal(response.status, 200);
    assert.match(body.id, /^[0-9a-f-]{36}$/i);
    assert.equal(response.headers.get('cache-control'), 'no-store');
    assert.doesNotMatch(JSON.stringify(body), new RegExp(key));
    assert.equal(await credentials.readCaptionApiCredential(root, body.id, 'http://127.0.0.1:34343/v1'), key);

    const invalid = await route.POST(new NextRequest('http://localhost/api/caption/credentials', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ apiUrl: 'http://127.0.0.1:34344/v1?token=do-not-log', apiKey: key }),
    }));
    assert.equal(invalid.status, 400);
    const errorBody = await invalid.json();
    assert.doesNotMatch(JSON.stringify(errorBody), /do-not-log|credential-route-secret-fixture/);
    assert.doesNotMatch(logged.join(' '), /do-not-log|credential-route-secret-fixture/);
  } finally {
    console.error = originalError;
    if (oldDirectory === undefined) delete process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
    else process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR = oldDirectory;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('API caption setup presents URL, key, prompts and collapsed configurable token controls', () => {
  const filename = path.resolve(__dirname, '../src/components/APICaptionSetup.tsx');
  const formInput = name => ({ label, value, ...props }) => React.createElement('label', null,
    label, React.createElement(name === 'TextAreaInput' ? 'textarea' : 'input', {
      value: value ?? '', type: props.type, max: props.max, min: props.min,
      onChange: () => {},
    }));
  const controls = {
    Checkbox: ({ label, checked }) => React.createElement('label', null, label, React.createElement('input', { type: 'checkbox', checked: Boolean(checked), readOnly: true })),
    CreatableSelectInput: formInput('CreatableSelectInput'), NumberInput: formInput('NumberInput'),
    SelectInput: formInput('SelectInput'), TextAreaInput: formInput('TextAreaInput'), TextInput: formInput('TextInput'),
  };
  const Setup = loadTypeScript(filename, (id, originalRequire) => id === './formInputs' ? controls : originalRequire(id)).default;
  const markup = renderToStaticMarkup(React.createElement(Setup, {
    caption: {
      api_url: 'http://caption.local/v1', model_name_or_path: 'fixture-model',
      system_prompt: 'system fixture', caption_prompt: 'caption fixture', max_new_tokens: 4096,
      api_timeout: 300, api_max_tokens_field: 'max_tokens', caption_extension: 'txt', recaption: false,
    },
    setCaption: () => {}, apiKey: '', onKeyChange: () => {}, clearKey: false, onClearKeyChange: () => {},
  }));
  assert.match(markup, /API URL/);
  assert.match(markup, /API key/);
  assert.match(markup, /Model/);
  assert.match(markup, /System prompt/);
  assert.match(markup, /Caption prompt/);
  assert.match(markup, /Recaption existing captions/);
  assert.match(markup, /Advanced options/);
  assert.match(markup, /configurable positive integer/);
  assert.doesNotMatch(markup, /max="4096"/);
  assert.doesNotMatch(markup, /<details[^>]*open=/);
});

test('captioner selection exposes API defaults while leaving recaption off for existing sidecars', () => {
  const apiConfig = loadConfig();
  const options = loadTypeScript(path.resolve(__dirname, '../src/helpers/captionOptions.ts'), (id, originalRequire) =>
    id === '../../cron/captionApiConfig' ? apiConfig : originalRequire(id));
  const configModule = loadTypeScript(path.resolve(__dirname, '../src/helpers/captionJobConfig.ts'), (id, originalRequire) =>
    id === './captionOptions' ? options : originalRequire(id));
  const initial = JSON.parse(JSON.stringify(configModule.defaultCaptionJobConfig));
  assert.equal(initial.config.process[0].caption.recaption, false);
  const set = {};
  configModule.handleCaptionerTypeChange('Qwen3OmniCaptioner', 'APICaptioner', initial, (value, key) => { set[key] = value; });
  assert.equal(set['config.process[0].type'], 'APICaptioner');
  assert.equal(set['config.process[0].caption.max_new_tokens'], 4096);
  assert.equal(set['config.process[0].caption.api_max_tokens_field'], 'max_tokens');
  assert.equal(set['config.process[0].caption.api_credential_id'], undefined);
  assert.ok(set['config.process[0].caption.caption_prompt'].length > 0);
});

test('caption job API forces CPU queue and never stores raw keys in job rows or responses', async () => {
  const root = fs.mkdtempSync(path.join(tempRoot, 'api-caption-route-test-'));
  const config = loadConfig();
  let saved;
  const prisma = { job: {
    aggregate: async () => ({ _max: { queue_position: 0 } }),
    create: async ({ data }) => {
      saved = { id: 'api-job', ...data };
      return saved;
    },
    findUnique: async () => saved,
    findMany: async () => saved ? [saved] : [],
  } };
  const filename = path.resolve(__dirname, '../src/app/api/jobs/route.ts');
  const route = loadTypeScript(filename, (id, originalRequire) => {
    if (id === '@/server/prisma') return { __esModule: true, default: prisma };
    if (id === '@/helpers/basic') return { isMac: () => false };
    if (id === '@/server/apiCache') return { cached: async (_key, getter) => getter() };
    if (id === '@/helpers/sampleOnlyJobConfig') return { mergeSampleOnlyJobConfig: value => value };
    if (id === '@/app/jobs/new/trainingCapabilities' || id === '@/app/jobs/new/sliderspace') return {
      validateTrainingCapabilities: () => [], validateSliderSpace: () => [],
    };
    if (id === '../../../../cron/captionApiConfig') return config;
    if (id === '../../../../cron/captionApiCredentials') return { readCaptionApiCredential: async () => 'fixture-secret' };
    if (id === '@/server/settings') return { getDatasetsRoot: async () => root };
    if (id === '@/server/jobFork') return { createJobFork: async () => {}, previewJobFork: async () => {}, JobForkError: class extends Error {} };
    if (id === '@/utils/search') return { matchesSearch: (name, query) => name.includes(query) };
    return originalRequire(id);
  });
  const createRequest = (jobConfig, extras = {}) => new NextRequest('http://localhost/api/jobs', {
    method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: 'caption-fixture', job_type: 'caption', gpu_ids: '0', job_config: jobConfig, ...extras }),
  });
  try {
    const configBody = apiCaptionConfig(root);
    const created = await route.POST(createRequest(configBody));
    assert.equal(created.status, 200);
    assert.equal(saved.gpu_ids, 'cpu');
    const safeText = `${saved.job_config} ${JSON.stringify(await created.json())}`;
    assert.doesNotMatch(safeText, /fixture-secret|api_key|authorization/i);

    const listResponse = await route.GET(new NextRequest('http://localhost/api/jobs'));
    assert.equal(listResponse.status, 200);
    assert.doesNotMatch(JSON.stringify(await listResponse.json()), /fixture-secret|api_key|authorization/i);

    const nonApiCpu = await route.POST(createRequest({ config: { process: [{ type: 'QwenCaptioner' }] } }, { gpu_ids: 'cpu', job_type: 'train' }));
    assert.equal(nonApiCpu.status, 400, 'the CPU queue is reserved for API caption jobs');

    const unsafe = apiCaptionConfig(root);
    unsafe.config.process[0].caption.api_key = 'fixture-secret';
    const logged = [];
    const originalError = console.error;
    console.error = (...args) => logged.push(args.map(String).join(' '));
    const rejected = await route.POST(createRequest(unsafe));
    console.error = originalError;
    assert.equal(rejected.status, 400);
    assert.doesNotMatch(JSON.stringify(await rejected.json()), /fixture-secret/);
    assert.equal(logged.length, 1);
    assert.doesNotMatch(logged.join(' '), /fixture-secret/);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test('API caption worker passes the URL-bound key only to the child environment and bypasses GPU locks', async () => {
  const root = fs.mkdtempSync(path.join(tempRoot, 'api-caption-worker-test-'));
  const credentials = loadCredentials(root);
  const oldDirectory = process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
  delete process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
  const key = 'worker-secret-fixture';
  const credentialId = await credentials.createCaptionApiCredential(root, 'http://127.0.0.1:34343/v1', key);
  const job = {
    id: 'api-job', name: 'api-caption-fixture', job_type: 'caption', gpu_ids: 'cpu', status: 'queued',
    job_config: JSON.stringify(apiCaptionConfig(root, credentialId)),
  };
  fs.writeFileSync(path.join(root, 'run.py'), '# fake entrypoint: never executed\n');
  let spawnOptions;
  let spawnArgs;
  let gpuLockRequests = 0;
  let resolveSpawn;
  const spawned = new Promise(resolve => { resolveSpawn = resolve; });
  const child = new EventEmitter();
  child.pid = 45678;
  child.unref = () => {};
  const prisma = { job: {
    findUnique: async () => job,
    updateMany: async () => ({ count: 1 }),
    update: async () => ({}),
  } };
  const config = loadConfig();
  const filename = path.resolve(__dirname, '../cron/actions/startJob.ts');
  const worker = loadTypeScript(filename, (id, originalRequire) => {
    if (id === '../prisma') return { __esModule: true, default: prisma };
    if (id === '../paths') return {
      TOOLKIT_ROOT: root, getTrainingFolder: async () => root,
      getHFToken: async () => '', getModelsPath: async () => '',
    };
    if (id === '../pythonPath') return { resolveDetachedPythonPath: () => '/synthetic/python' };
    if (id === '../sampleRenderLocks') return {
      acquireGpuLocks: () => { gpuLockRequests++; throw new Error('GPU lock should not be requested for CPU captioning'); },
      gpuKeys: () => ['gpu0'], GpuBusyError: class extends Error {}, SAMPLE_RENDER_GATE: 'samples',
    };
    if (id === '../captionApiConfig') return config;
    if (id === '../captionApiCredentials') return credentials;
    if (id === 'child_process') return { ...originalRequire(id), spawn: (_command, args, options) => {
      spawnArgs = args;
      spawnOptions = options;
      process.nextTick(() => child.emit('spawn'));
      resolveSpawn();
      return child;
    } };
    return originalRequire(id);
  });
  try {
    await worker.default('api-job');
    await spawned;
    const pidPath = path.join(root, job.name, 'pid.txt');
    for (let attempt = 0; attempt < 100 && !fs.existsSync(pidPath); attempt++)
      await new Promise(resolve => setTimeout(resolve, 10));
    assert.ok(fs.existsSync(pidPath), 'worker finished writing its fake child pid');
    assert.equal(gpuLockRequests, 0);
    assert.equal(spawnOptions.env.AITK_API_CAPTION_KEY, key);
    assert.equal(spawnOptions.env.CUDA_VISIBLE_DEVICES, '');
    const configPath = path.join(root, job.name, '.job_config.json');
    const logPath = path.join(root, job.name, 'log.txt');
    const serialized = `${fs.readFileSync(configPath, 'utf8')} ${fs.readFileSync(logPath, 'utf8')} ${spawnArgs.join(' ')}`;
    assert.doesNotMatch(serialized, new RegExp(key));
    assert.doesNotMatch(serialized, /Authorization|Bearer/i);
    assert.match(serialized, new RegExp(credentialId));
  } finally {
    if (oldDirectory === undefined) delete process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR;
    else process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR = oldDirectory;
    fs.rmSync(root, { recursive: true, force: true });
  }
});

