const assert = require('node:assert/strict');
const { test, before, after } = require('node:test');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const Module = require('node:module');
const ts = require('typescript');
const { randomUUID } = require('node:crypto');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

const sourceRoot = path.resolve(__dirname, '../src');
const cache = new Map();
let root;
let assistantRouteService;
const settings = {
  getDatasetsRoot: async () => path.join(root, 'datasets'),
  getTrainingFolder: async () => path.join(root, 'output'),
  getDataRoot: async () => path.join(root, 'data'),
  getModelsPath: async () => path.join(root, 'models'),
  getHFToken: async () => 'fixture-private-hf-token',
};
function load(filename) {
  filename = path.resolve(sourceRoot, filename);
  if (!path.extname(filename)) filename += fs.existsSync(filename + '.ts') ? '.ts' : '.tsx';
  if (cache.has(filename)) return cache.get(filename).exports;
  const mod = new Module(filename, module);
  mod.filename = filename;
  mod.paths = Module._nodeModulePaths(path.dirname(filename));
  cache.set(filename, mod);
  const original = mod.require.bind(mod);
  mod.require = id =>
    id === '@/server/settings'
      ? settings
      : id === '@/server/assistant/service'
        ? { getAssistantService: async () => assistantRouteService }
      : id.startsWith('@/')
        ? load(id.slice(2))
        : id.startsWith('.')
          ? load(path.resolve(path.dirname(filename), id))
          : original(id);
  mod._compile(
    ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
      compilerOptions: {
        module: ts.ModuleKind.CommonJS,
        target: ts.ScriptTarget.ES2022,
        jsx: ts.JsxEmit.ReactJSX,
        esModuleInterop: true,
      },
    }).outputText,
    filename,
  );
  return mod.exports;
}
function loadStandaloneTypeScript(filename) {
  const absolute = path.resolve(__dirname, filename);
  const mod = new Module(absolute, module);
  mod.filename = absolute;
  mod.paths = Module._nodeModulePaths(path.dirname(absolute));
  mod._compile(
    ts.transpileModule(fs.readFileSync(absolute, 'utf8'), {
      fileName: absolute,
      compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
    }).outputText,
    absolute,
  );
  return mod.exports;
}
function loadAssistantSessionsHook(apiClient) {
  const slots = [];
  let index = 0;
  const react = {
    useRef(initial) {
      const slot = index++;
      if (!slots[slot]) slots[slot] = { current: initial };
      return slots[slot];
    },
    useSyncExternalStore(subscribe, getSnapshot) {
      const slot = index++;
      if (!slots[slot]) slots[slot] = { unsubscribe: subscribe(() => {}) };
      return getSnapshot();
    },
    useEffect(effect, dependencies) {
      const slot = index++;
      const previous = slots[slot];
      if (!previous || dependencies.some((dependency, position) => dependency !== previous.dependencies[position])) {
        previous?.cleanup?.();
        slots[slot] = { dependencies: [...dependencies], cleanup: effect() };
      }
    },
  };
  const filename = path.resolve(__dirname, '../src/hooks/useAssistantSessions.ts');
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText;
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => {
    if (id === 'react') return react;
    if (id === '@/utils/api') return { apiClient };
    if (id.startsWith('@/')) return load(id.slice(2));
    return originalRequire(id);
  };
  instance._compile(compiled, filename);
  return () => {
    index = 0;
    return instance.exports.default();
  };
}
const { DEFAULT_ASSISTANT_CONFIGURATION, assistantEndpoint, assistantConfigurationSchema } =
  load('assistant/AssistantProvider');
const { NodeAssistantConfiguration } = load('server/assistant/NodeAssistantConfiguration');
const { AssistantSessionStore, SessionStoreError } = load('server/assistant/AssistantSessionStore');
const { AssistantSessionController } = load('assistant/AssistantSessionController');
const { POST: postAssistantSessions, GET: getAssistantSessions } = load('app/api/assistant/sessions/route');
const assistantSessionItemRoute = load('app/api/assistant/sessions/[id]/route');
const { sessionContentSchema } = load('assistant/sessions');
const { isAssistantSessionPath } = loadStandaloneTypeScript('../cron/assistantSessionPaths.ts');
const { browserSessionHost } = load('hooks/useAssistantSessions');
const AssistantHistory = load('components/assistant/AssistantHistory').default;
const { NodeAssistantProvider, parseAssistantResponse, readAssistantSse } = load(
  'server/assistant/NodeAssistantProvider',
);
const { createAssistantDiagnosticSink } = load('server/assistant/diagnostics');
const { executeAssistantTool, scopedPath, validateApiOperation } = load('server/assistant/toolExecutor');
const { ASSISTANT_TOOLS, isReadOnlyTool } = load('assistant/tools');
const { POST: postAssistantTurn } = load('app/api/assistant/turn/route');
const sharp = require(require.resolve('sharp', { paths: [path.dirname(require.resolve('next/package.json'))] }));
before(async () => {
  root = await fsp.mkdtemp(path.join(os.tmpdir(), 'aitk-assistant-test-'));
  for (const name of ['datasets', 'output', 'data', 'models']) await fsp.mkdir(path.join(root, name));
});
after(async () => {
  await fsp.rm(root, { recursive: true, force: true });
});
const completed = output => ({
  status: 'completed',
  output,
  usage: { input_tokens: 12, output_tokens: 4, total_tokens: 16 },
});
const message = text => ({ type: 'message', role: 'assistant', content: [{ type: 'output_text', text }] });
const jsonResponse = value => ({
  status: 200,
  contentType: 'application/json',
  body: (async function* () {
    yield Buffer.from(JSON.stringify(value));
  })(),
});
async function provider(
  options = {},
  transport = async () => jsonResponse(completed([message('Done')])),
  diagnostics,
  apiKey,
) {
  const configuration = new NodeAssistantConfiguration();
  await configuration.update({
    config: {
      ...DEFAULT_ASSISTANT_CONFIGURATION,
      baseUrl: 'https://fixture.invalid/v1',
      authMode: 'none',
      model: 'fixture',
      ...options,
    },
    ...(apiKey === undefined ? {} : { apiKey }),
  });
  return { service: new NodeAssistantProvider({ configuration, transport, diagnostics }), configuration };
}
const turn = revision => ({
  runId: randomUUID(),
  requestId: randomUUID(),
  configurationRevision: revision,
  messages: [{ role: 'user', content: 'Inspect fixture.' }],
  tools: ASSISTANT_TOOLS,
});
const context = async (options = {}) => ({
  origin: 'http://fixture.local',
  authorization: 'Bearer fixture-auth',
  provider: (await provider({ vision: true })).service,
  ...options,
});
const call = (name, args) => ({ name, arguments: args, callId: randomUUID() });
const sessionContent = text => ({
  entries: [{ id: randomUUID(), role: 'user', text }],
  draft: '',
  attachments: [],
});
const makeSession = (id, content, revision = 1, title = content.entries[0]?.text || 'Recovered draft') => ({
  version: 1,
  id,
  title,
  titleSource: 'auto',
  createdAt: '2026-01-01T00:00:00.000Z',
  updatedAt: `2026-01-01T00:00:0${Math.min(revision, 9)}.000Z`,
  revision,
  entryCount: content.entries.length,
  content,
});
function memorySessionHost({ legacy = null, legacyId = null, recovery = null, lastId = null } = {}) {
  return {
    rememberedId: lastId,
    importedId: legacyId,
    legacyValue: legacy,
    recoveryValue: recovery,
    completedLegacy: false,
    lastId() { return this.rememberedId; },
    remember(id) { this.rememberedId = id; },
    legacy() { return this.legacyValue; },
    legacyId() { return this.importedId; },
    rememberLegacy(id) { this.importedId = id; },
    completeLegacy() { this.completedLegacy = true; this.legacyValue = null; this.importedId = null; },
    recovery() { return this.recoveryValue; },
    recover(value) { this.recoveryValue = value; },
  };
}
async function withSessionDirectory(callback) {
  const parent = path.join(__dirname, '../../tmp');
  await fsp.mkdir(parent, { recursive: true });
  const directory = await fsp.mkdtemp(path.join(parent, 'assistant-sessions-test-'));
  try {
    return await callback(directory);
  } finally {
    await fsp.rm(directory, { recursive: true, force: true });
  }
}

test('configuration validates budgets and URL prefixes without selecting guessed models', () => {
  assert.equal(DEFAULT_ASSISTANT_CONFIGURATION.model, '');
  assert.equal(
    assistantEndpoint({ ...DEFAULT_ASSISTANT_CONFIGURATION, baseUrl: 'https://fixture.invalid/custom/v1/' }, 'models')
      .href,
    'https://fixture.invalid/custom/v1/models',
  );
  for (const config of [
    { baseUrl: 'https://user:pass@fixture.invalid' },
    { timeoutMs: 0 },
    { maxRunTokens: 4096 },
    { reasoningEffort: 'made-up' },
  ])
    assert.throws(() => assistantConfigurationSchema.parse({ ...DEFAULT_ASSISTANT_CONFIGURATION, ...config }));
  assert.equal(DEFAULT_ASSISTANT_CONFIGURATION.imageMegapixels, 1);
  assert.equal(
    assistantConfigurationSchema.parse({ ...DEFAULT_ASSISTANT_CONFIGURATION, imageMegapixels: 0.375 })
      .imageMegapixels,
    0.375,
  );
  for (const imageMegapixels of [0, -0.25, 16.01, Infinity, NaN])
    assert.throws(() => assistantConfigurationSchema.parse({ ...DEFAULT_ASSISTANT_CONFIGURATION, imageMegapixels }));
});
test('image megapixel setting defaults for older saved configurations and accepts fractions', async () => {
  const directory = path.join(root, 'legacy-image-settings');
  await fsp.mkdir(directory);
  const { imageMegapixels: _imageMegapixels, ...legacyConfig } = DEFAULT_ASSISTANT_CONFIGURATION;
  await fsp.writeFile(path.join(directory, 'assistant.json'), JSON.stringify({ version: 1, config: legacyConfig }));
  const store = await NodeAssistantConfiguration.create(directory);
  assert.equal(store.state().config.imageMegapixels, 1);

  const fractional = { ...store.state().config, imageMegapixels: 0.375 };
  assert.equal((await store.update({ config: fractional })).config.imageMegapixels, 0.375);
  for (const imageMegapixels of [0, -1, 16.01, Infinity, NaN])
    await assert.rejects(() => store.update({ config: { ...fractional, imageMegapixels } }));
});
test('stored credentials remain private and outside configuration responses', async () => {
  const directory = path.join(root, 'settings');
  const store = await NodeAssistantConfiguration.create(directory);
  await store.update({ config: DEFAULT_ASSISTANT_CONFIGURATION, apiKey: 'fixture-secret', persistCredential: true });
  assert.equal(store.state().credentialPersisted, true);
  assert.ok(!JSON.stringify(store.state()).includes('fixture-secret'));
  assert.equal((await fsp.stat(path.join(directory, 'assistant-credential.json'))).mode & 0o777, 0o600);
  const reloaded = await NodeAssistantConfiguration.create(directory);
  assert.equal(reloaded.key(), 'fixture-secret');
  await reloaded.update({ config: DEFAULT_ASSISTANT_CONFIGURATION, clearCredential: true });
  assert.equal(reloaded.state().credentialPresent, false);
  assert.equal(fs.existsSync(path.join(directory, 'assistant-credential.json')), false);
});
test('saved chats persist across reopen, allow duplicate titles, and keep manual renames', async () => {
  await withSessionDirectory(async directory => {
    const store = await AssistantSessionStore.create(directory);
    const firstId = randomUUID();
    const secondId = randomUUID();
    const first = await store.create(firstId, sessionContent('Same request title'));
    const second = await store.create(secondId, sessionContent('Same request title'));
    assert.equal(first.title, second.title);
    assert.notEqual(first.id, second.id);
    assert.equal(first.titleSource, 'auto');

    const renamed = await store.update(firstId, { revision: first.revision, title: 'Pinned title' });
    const changedContent = await store.update(firstId, {
      revision: renamed.revision,
      content: sessionContent('A different follow-up prompt'),
    });
    assert.equal(changedContent.title, 'Pinned title');
    assert.equal(changedContent.titleSource, 'manual');
    assert.equal(changedContent.revision, 3);

    const reopened = await AssistantSessionStore.create(directory);
    assert.equal((await reopened.get(firstId)).content.entries[0].text, 'A different follow-up prompt');
    const summaries = await reopened.list();
    assert.equal(summaries.length, 2);
    assert.ok(summaries.every(summary => !('content' in summary)));
    assert.equal(summaries.find(summary => summary.id === firstId).title, 'Pinned title');
  });
});
test('saved-chat writes validate revisions and tombstone deletes against stale retries', async () => {
  await withSessionDirectory(async directory => {
    const store = await AssistantSessionStore.create(directory);
    const id = randomUUID();
    const initial = await store.create(id, sessionContent('First prompt'));
    await assert.rejects(
      store.update(id, { revision: initial.revision - 1, title: 'stale rename' }),
      error => error instanceof SessionStoreError && error.status === 409,
    );
    const renamed = await store.update(id, { revision: initial.revision, title: 'Current title' });
    await assert.rejects(
      store.delete(id, initial.revision),
      error => error instanceof SessionStoreError && error.status === 409,
    );
    await store.delete(id, renamed.revision);
    await assert.rejects(store.get(id), error => error instanceof SessionStoreError && error.status === 404);
    await assert.rejects(
      store.create(id, sessionContent('late autosave')),
      error => error instanceof SessionStoreError && error.status === 410,
    );

    const reopened = await AssistantSessionStore.create(directory);
    await assert.rejects(
      reopened.create(id, sessionContent('late autosave after restart')),
      error => error instanceof SessionStoreError && error.status === 410,
    );
    assert.equal((await reopened.list()).length, 0);
  });
});
test('concurrent saved-chat edits accept one revision and reject the stale writer', async () => {
  await withSessionDirectory(async directory => {
    const store = await AssistantSessionStore.create(directory);
    const id = randomUUID();
    const initial = await store.create(id, sessionContent('Concurrent edits'));
    const edits = await Promise.allSettled([
      store.update(id, { revision: initial.revision, title: 'Tab A title' }),
      store.update(id, { revision: initial.revision, title: 'Tab B title' }),
    ]);
    assert.equal(edits.filter(result => result.status === 'fulfilled').length, 1);
    const rejected = edits.find(result => result.status === 'rejected');
    assert.ok(rejected.reason instanceof SessionStoreError);
    assert.equal(rejected.reason.status, 409);
    const current = await store.get(id);
    assert.equal(current.revision, initial.revision + 1);
    assert.ok(['Tab A title', 'Tab B title'].includes(current.title));
  });
});
test('saved-chat retry with the same ID is idempotent and private files reject unsafe modes', async () => {
  await withSessionDirectory(async directory => {
    const store = await AssistantSessionStore.create(directory);
    const id = randomUUID();
    const original = sessionContent('Original content');
    const saved = await store.create(id, original);
    const retry = await store.create(id, sessionContent('Retry payload must not overwrite saved content'));
    assert.equal(retry.revision, saved.revision);
    assert.deepEqual(retry.content, original);
    assert.equal((await fsp.stat(directory)).mode & 0o777, 0o700);
    const filename = path.join(directory, `${id}.json`);
    assert.equal((await fsp.stat(filename)).mode & 0o777, 0o600);

    if (process.platform !== 'win32') {
      await fsp.chmod(filename, 0o644);
      await assert.rejects(AssistantSessionStore.create(directory), /owner-only permissions/);
    }
    assert.throws(() => sessionContentSchema.parse({
      ...original,
      entries: [{ ...original.entries[0], preview: { path: '/secret', url: 'https://evil.invalid/image' } }],
    }));
  });
});
test('saved-chat storage paths stay private through direct and symlinked file access paths', async () => {
  await withSessionDirectory(async directory => {
    const priorDirectory = process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR;
    process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR = directory;
    const sessions = path.join(directory, 'sessions');
    await fsp.mkdir(sessions);
    const privateFile = path.join(sessions, `${randomUUID()}.json`);
    const publicFile = path.join(directory, 'ordinary.json');
    await fsp.writeFile(privateFile, '{}');
    await fsp.writeFile(publicFile, '{}');
    const alias = path.join(directory, 'session-alias.json');
    try {
      await fsp.symlink(privateFile, alias);
      assert.equal(await isAssistantSessionPath(privateFile, root), true);
      assert.equal(await isAssistantSessionPath(alias, root), true);
      assert.equal(await isAssistantSessionPath(publicFile, root), false);
    } finally {
      if (priorDirectory === undefined) delete process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR;
      else process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR = priorDirectory;
    }
  });
});
test('saved-chat HTTP routes validate input and preserve same-ID retry, conflict, and delete semantics', async () => {
  await withSessionDirectory(async directory => {
    const priorDirectory = process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR;
    const priorStore = globalThis.toolkitAssistantSessions;
    process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR = directory;
    globalThis.toolkitAssistantSessions = undefined;
    const loggedErrors = [];
    const priorConsoleError = console.error;
    console.error = (...args) => loggedErrors.push(args);
    const request = body => ({ json: async () => body });
    const params = id => ({ params: Promise.resolve({ id }) });
    try {
      const invalid = await postAssistantSessions(request({ id: 'not-a-uuid', content: sessionContent('x') }));
      assert.equal(invalid.status, 400);
      const invalidContent = await postAssistantSessions(request({ id: randomUUID(), content: { entries: [], draft: '', attachments: [], unexpected: true } }));
      assert.equal(invalidContent.status, 400);

      const id = randomUUID();
      const content = sessionContent('Persist through the route');
      const createdResponse = await postAssistantSessions(request({ id, content }));
      assert.equal(createdResponse.status, 200);
      const created = await createdResponse.json();
      const retriedResponse = await postAssistantSessions(request({ id, content: sessionContent('late retry') }));
      assert.equal(retriedResponse.status, 200);
      assert.equal((await retriedResponse.json()).content.entries[0].text, 'Persist through the route');

      const conflict = await assistantSessionItemRoute.PATCH(request({ revision: created.revision - 1, title: 'stale' }), params(id));
      assert.equal(conflict.status, 409);
      const renamedResponse = await assistantSessionItemRoute.PATCH(request({ revision: created.revision, title: 'Manual title' }), params(id));
      assert.equal(renamedResponse.status, 200);
      const renamed = await renamedResponse.json();
      assert.equal(renamed.title, 'Manual title');
      assert.equal((await (await getAssistantSessions({ nextUrl: new URL('http://fixture.local/api/assistant/sessions') })).json()).sessions.length, 1);

      const deleted = await assistantSessionItemRoute.DELETE(request({ revision: renamed.revision }), params(id));
      assert.equal(deleted.status, 200);
      const staleWrite = await assistantSessionItemRoute.PATCH(request({ revision: renamed.revision, content }), params(id));
      assert.equal(staleWrite.status, 404);
      const lateRetry = await postAssistantSessions(request({ id, content }));
      assert.equal(lateRetry.status, 410);
      assert.equal((await (await getAssistantSessions({ nextUrl: new URL('http://fixture.local/api/assistant/sessions') })).json()).sessions.length, 0);
    } finally {
      globalThis.toolkitAssistantSessions = priorStore;
      console.error = priorConsoleError;
      if (priorDirectory === undefined) delete process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR;
      else process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR = priorDirectory;
    }
    assert.equal(loggedErrors.length, 5);
  });
});
test('session controller serializes autosaves and flushes edits made during an earlier write', async () => {
  let releaseCreate;
  let signalCreateStarted;
  const createPending = new Promise(resolve => (releaseCreate = resolve));
  const createStarted = new Promise(resolve => (signalCreateStarted = resolve));
  const calls = [];
  const id = randomUUID();
  const api = {
    list: async () => [],
    get: async () => { throw new Error('not used'); },
    create: async (sessionId, content) => {
      calls.push({ method: 'create', sessionId, content: structuredClone(content) });
      signalCreateStarted();
      return createPending;
    },
    update: async (sessionId, value) => {
      calls.push({ method: 'update', sessionId, revision: value.revision, content: structuredClone(value.content) });
      return makeSession(sessionId, value.content, value.revision + 1);
    },
    delete: async () => {},
  };
  const controller = new AssistantSessionController(api, memorySessionHost(), () => id);
  await controller.initialize();
  const firstContent = sessionContent('First saved version');
  controller.updateContent(() => firstContent);
  const pendingFlush = controller.flush();
  await createStarted;
  controller.updateContent(previous => ({
    ...previous,
    entries: [...previous.entries, { id: randomUUID(), role: 'assistant', text: 'Reply arrived during save' }],
  }));
  const latestContent = controller.snapshot().content;
  releaseCreate(makeSession(id, firstContent, 1));
  assert.equal(await pendingFlush, true, JSON.stringify(controller.snapshot()));
  assert.deepEqual(calls.map(call => call.method), ['create', 'update']);
  assert.equal(calls[1].revision, 1);
  assert.deepEqual(calls[1].content, latestContent);
  assert.equal(controller.snapshot().metadata.revision, 2);
  assert.deepEqual(controller.snapshot().content, latestContent);
  assert.equal(controller.snapshot().saveState, 'saved');
});
test('a clean flush does not leave a resolved save promise latched across later edits', async () => {
  const id = randomUUID();
  const calls = [];
  const api = {
    list: async () => [],
    get: async () => { throw new Error('not used'); },
    create: async (sessionId, content) => {
      calls.push({ method: 'create', content: structuredClone(content) });
      return makeSession(sessionId, content);
    },
    update: async (sessionId, value) => {
      calls.push({ method: 'update', revision: value.revision, content: structuredClone(value.content) });
      return makeSession(sessionId, value.content, value.revision + 1);
    },
    delete: async () => {},
  };
  const controller = new AssistantSessionController(api, memorySessionHost(), () => id);
  await controller.initialize();
  controller.updateContent(() => sessionContent('Persist first entry'));
  assert.equal(await controller.flush(), true);
  assert.equal(await controller.flush(), true, 'the clean boundary is the latch regression');
  controller.updateContent(previous => ({
    ...previous,
    entries: [...previous.entries, { id: randomUUID(), role: 'assistant', text: 'Persist later reply' }],
  }));
  assert.equal(await controller.flush(), true);
  assert.deepEqual(calls.map(call => call.method), ['create', 'update']);
  assert.equal(controller.snapshot().metadata.revision, 2);
  assert.equal(controller.snapshot().saveState, 'saved');
});
test('failed session flush preserves conversation text and recovery snapshot for retry', async () => {
  const id = randomUUID();
  const content = sessionContent('Keep this text after storage failure');
  let attempts = 0;
  const host = memorySessionHost();
  const api = {
    list: async () => [],
    get: async () => { throw new Error('not used'); },
    create: async (sessionId, value) => {
      attempts += 1;
      if (attempts === 1) throw new Error('fixture disk failure');
      return makeSession(sessionId, value);
    },
    update: async () => { throw new Error('not used'); },
    delete: async () => {},
  };
  const controller = new AssistantSessionController(api, host, () => id);
  await controller.initialize();
  controller.updateContent(() => content);
  assert.equal(await controller.flush(), false);
  assert.deepEqual(controller.snapshot().content, content);
  assert.equal(controller.snapshot().error.message, 'fixture disk failure');
  assert.equal(host.recoveryValue.id, id);
  assert.deepEqual(host.recoveryValue.content, content);
  assert.equal(await controller.retry(), true);
  assert.equal(attempts, 2);
  assert.deepEqual(controller.snapshot().content, content);
  assert.equal(controller.snapshot().saveState, 'saved');
  assert.equal(host.recoveryValue, null);
});
test('legacy session migration retains a stable ID and source until an idempotent retry is acknowledged', async () => {
  const legacy = sessionContent('Legacy conversation text');
  const host = memorySessionHost({ legacy });
  const serverSessions = new Map();
  const createdIds = [];
  let loseFirstResponse = true;
  const api = {
    list: async () => [],
    get: async id => serverSessions.get(id),
    create: async (id, content) => {
      createdIds.push(id);
      if (!serverSessions.has(id)) serverSessions.set(id, makeSession(id, content));
      if (loseFirstResponse) {
        loseFirstResponse = false;
        throw new Error('simulated response loss after commit');
      }
      return serverSessions.get(id);
    },
    update: async () => { throw new Error('not used'); },
    delete: async () => {},
  };
  const controller = new AssistantSessionController(api, host, () => randomUUID());
  await controller.initialize();
  const stableId = host.importedId;
  assert.ok(stableId);
  assert.equal(host.legacyValue, legacy);
  assert.equal(controller.snapshot().saveState, 'error');
  assert.equal(await controller.retry(), true);
  assert.deepEqual(createdIds, [stableId, stableId]);
  assert.equal(host.completedLegacy, true);
  assert.equal(host.legacyValue, null);
  assert.deepEqual(controller.snapshot().content, legacy);
  assert.equal(controller.snapshot().metadata.revision, 1);
});
test('a changed-content migration retry surfaces a conflict and preserves both local copies', async () => {
  const legacy = sessionContent('Original legacy conversation');
  const localLatest = {
    ...legacy,
    entries: [...legacy.entries, { id: randomUUID(), role: 'assistant', text: 'Unsaved follow-up' }],
  };
  const id = randomUUID();
  const host = memorySessionHost({ legacy });
  let remoteSnapshot;
  const updates = [];
  let loseFirstResponse = true;
  const api = {
    list: async () => [],
    get: async () => remoteSnapshot,
    create: async (sessionId, content) => {
      if (!remoteSnapshot) remoteSnapshot = makeSession(sessionId, content);
      if (loseFirstResponse) {
        loseFirstResponse = false;
        throw new Error('simulated lost create response');
      }
      return remoteSnapshot;
    },
    update: async (sessionId, value) => {
      updates.push({ sessionId, value });
      return makeSession(sessionId, value.content, value.revision + 1);
    },
    delete: async () => {},
  };
  const controller = new AssistantSessionController(api, host, () => id);
  await controller.initialize();
  controller.updateContent(() => localLatest);
  assert.equal(await controller.retry(), false);
  assert.equal(updates.length, 0, 'an idempotent create mismatch must not be overwritten with a newly acquired revision');
  assert.deepEqual(remoteSnapshot.content, legacy);
  assert.deepEqual(controller.snapshot().content, localLatest);
  assert.deepEqual(host.legacyValue, legacy);
  assert.deepEqual(host.recoveryValue.content, localLatest);
  assert.equal(host.recoveryValue.id, id);
  assert.equal(host.completedLegacy, false);
});
test('missing recovery target keeps local entries ready for an explicit save copy', async () => {
  const recoveryContent = sessionContent('Only recoverable local copy');
  const missingId = randomUUID();
  const copyId = randomUUID();
  const host = memorySessionHost({ recovery: { id: missingId, revision: 4, content: recoveryContent } });
  const creates = [];
  const api = {
    list: async () => [],
    get: async () => { throw { response: { status: 404, data: { error: 'Chat was deleted.' } } }; },
    create: async (id, content) => {
      creates.push({ id, content });
      return makeSession(id, content);
    },
    update: async () => { throw new Error('not used'); },
    delete: async () => {},
  };
  const generatedIds = [randomUUID(), copyId];
  const controller = new AssistantSessionController(api, host, () => generatedIds.shift());
  await controller.initialize();
  assert.equal(controller.snapshot().ready, true);
  assert.deepEqual(controller.snapshot().content, recoveryContent);
  assert.match(controller.snapshot().error.message, /deleted/i);
  assert.equal(await controller.saveCopy(), true);
  assert.deepEqual(creates, [{ id: copyId, content: recoveryContent }]);
  assert.equal(controller.snapshot().id, copyId);
  assert.equal(controller.snapshot().metadata.revision, 1);
  assert.equal(host.recoveryValue, null);
});
test('history-list failure does not hide a recovery snapshot or block save copy', async () => {
  const recoveryContent = sessionContent('Recovered when history endpoint is offline');
  const recoveryId = randomUUID();
  const copyId = randomUUID();
  const host = memorySessionHost({ recovery: { id: recoveryId, revision: 0, content: recoveryContent } });
  const api = {
    list: async () => { throw new Error('history service unavailable'); },
    get: async () => { throw new Error('not used'); },
    create: async (id, content) => makeSession(id, content),
    update: async () => { throw new Error('not used'); },
    delete: async () => {},
  };
  const ids = [randomUUID(), copyId];
  const controller = new AssistantSessionController(api, host, () => ids.shift());
  await controller.initialize();
  assert.equal(controller.snapshot().ready, true);
  assert.deepEqual(controller.snapshot().content, recoveryContent);
  assert.match(controller.snapshot().historyError.message, /history service unavailable/);
  assert.equal(await controller.saveCopy(), true);
  assert.equal(controller.snapshot().metadata.id, copyId);
});
test('browser session host reads legacy entries and persists migration and recovery markers', () => {
  const oldLocal = globalThis.localStorage;
  const oldSession = globalThis.sessionStorage;
  const makeStorage = () => {
    const values = new Map();
    return {
      getItem: key => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, String(value)),
      removeItem: key => values.delete(key),
      values,
    };
  };
  globalThis.localStorage = makeStorage();
  globalThis.sessionStorage = makeStorage();
  const legacy = sessionContent('Legacy browser entry');
  try {
    globalThis.sessionStorage.setItem('toolkit:assistant:messages', JSON.stringify(legacy.entries));
    assert.deepEqual(browserSessionHost.legacy(), legacy);
    browserSessionHost.rememberLegacy('migration-id');
    assert.equal(browserSessionHost.legacyId(), 'migration-id');
    browserSessionHost.remember('saved-id');
    assert.equal(browserSessionHost.lastId(), 'saved-id');
    const recovery = { id: randomUUID(), revision: 4, content: legacy };
    browserSessionHost.recover(recovery);
    assert.deepEqual(browserSessionHost.recovery(), recovery);
    browserSessionHost.completeLegacy();
    assert.equal(browserSessionHost.legacy(), null);
    assert.equal(browserSessionHost.legacyId(), null);
    assert.deepEqual(browserSessionHost.recovery(), recovery, 'legacy cleanup must not discard a separate recovery snapshot');
  } finally {
    if (oldLocal === undefined) delete globalThis.localStorage;
    else globalThis.localStorage = oldLocal;
    if (oldSession === undefined) delete globalThis.sessionStorage;
    else globalThis.sessionStorage = oldSession;
  }
});
test('session hook does not schedule autosave while a recovery snapshot is hydrating', async () => {
  const oldLocal = globalThis.localStorage;
  const oldSession = globalThis.sessionStorage;
  const oldSetTimeout = globalThis.setTimeout;
  const oldClearTimeout = globalThis.clearTimeout;
  const makeStorage = () => {
    const values = new Map();
    return {
      getItem: key => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, String(value)),
      removeItem: key => values.delete(key),
      values,
    };
  };
  globalThis.localStorage = makeStorage();
  globalThis.sessionStorage = makeStorage();
  const content = sessionContent('Hydrating recovery conversation');
  const id = randomUUID();
  const saved = makeSession(id, content, 4);
  const { version: _version, content: _content, ...metadata } = saved;
  globalThis.sessionStorage.setItem('toolkit:assistant:recovery', JSON.stringify({ id, revision: 4, content }));
  let resolveSession;
  let signalGetStarted;
  const sessionPending = new Promise(resolve => (resolveSession = resolve));
  const getStarted = new Promise(resolve => (signalGetStarted = resolve));
  const requests = [];
  const timers = [];
  globalThis.setTimeout = (callback, delay) => { timers.push({ callback, delay }); return timers.length; };
  globalThis.clearTimeout = () => {};
  const render = loadAssistantSessionsHook({
    get: async url => {
      requests.push(url);
      if (url === '/api/assistant/sessions') return { data: { sessions: [metadata] } };
      signalGetStarted();
      return sessionPending;
    },
    post: async () => { throw new Error('autosave must not run during hydration'); },
    patch: async () => { throw new Error('autosave must not run during hydration'); },
    delete: async () => {},
  });
  try {
    let state = render();
    const controller = state.controller;
    const initialization = controller.initialize();
    await getStarted;
    state = render();
    assert.equal(state.ready, true);
    assert.equal(state.loading, true);
    assert.equal(state.saveState, 'unsaved');
    assert.deepEqual(state.content, content);
    assert.equal(timers.length, 0, 'recovery must not trigger the debounce save before its saved version is checked');

    resolveSession({ data: saved });
    await initialization;
    await new Promise(resolve => setImmediate(resolve));
    state = render();
    assert.equal(state.loading, false, JSON.stringify(state.controller.snapshot()));
    assert.equal(state.saveState, 'saved');
    assert.equal(requests.length, 2);
    assert.equal(timers.length, 0);
  } finally {
    globalThis.localStorage = oldLocal;
    globalThis.sessionStorage = oldSession;
    globalThis.setTimeout = oldSetTimeout;
    globalThis.clearTimeout = oldClearTimeout;
  }
});
test('saved-history actions are all disabled while an assistant run is active', () => {
  const callbacks = {
    onOpen: () => {}, onNew: () => {}, onRename: () => {}, onDelete: () => {}, onRefresh: () => {},
  };
  const sessions = [{
    id: randomUUID(), title: 'Fixture conversation', titleSource: 'auto',
    createdAt: '2026-01-01T00:00:00.000Z', updatedAt: '2026-01-01T00:00:00.000Z',
    revision: 1, entryCount: 2,
  }];
  const render = disabled => renderToStaticMarkup(React.createElement(AssistantHistory, {
    sessions, activeId: sessions[0].id, disabled, refreshing: false, error: null, ...callbacks,
  }));
  const activeRunMarkup = render(true);
  const idleMarkup = render(false);
  assert.equal((activeRunMarkup.match(/<button[^>]*disabled=""/g) || []).length, 4);
  assert.equal((idleMarkup.match(/<button[^>]*disabled=""/g) || []).length, 0);
});
test('assistant diagnostics explain preflight rejection and log only safe metadata', async () => {
  const directory = path.join(root, 'assistant-diagnostics');
  await fsp.mkdir(directory);
  const diagnostics = createAssistantDiagnosticSink(directory);
  const credential = 'DIAGNOSTIC_CREDENTIAL_MARKER';
  const promptSecret = 'PROMPT_TEXT_MARKER';
  const argsSecret = 'FUNCTION_ARGUMENT_MARKER';
  const ciphertextSecret = 'SYNTHETIC_CIPHERTEXT_MARKER_'.repeat(10_000);
  const imageBytes = await sharp({ create: { width: 2, height: 2, channels: 3, background: '#00ff00' } })
    .png()
    .toBuffer();
  const imageBase64 = imageBytes.toString('base64');
  const requests = [];
  const { service, configuration } = await provider(
    { authMode: 'api-key', vision: true },
    async (_url, options) => {
      requests.push(JSON.parse(options.body));
      return jsonResponse({
        status: 'completed',
        output: [
          { type: 'reasoning', encrypted_content: ciphertextSecret },
          {
            type: 'function_call',
            call_id: 'diagnostic-call',
            name: 'fixture_tool',
            arguments: JSON.stringify({ value: argsSecret }),
          },
        ],
        usage: {
          input_tokens: 12,
          output_tokens: 4,
          total_tokens: 16,
          input_tokens_details: { cached_tokens: 3 },
          output_tokens_details: { reasoning_tokens: 2 },
        },
      });
    },
    diagnostics,
    credential,
  );
  const request = {
    runId: randomUUID(),
    requestId: randomUUID(),
    configurationRevision: configuration.state().revision,
    messages: [
      {
        role: 'user',
        content: promptSecret,
        images: [
          {
            artifactId: 'diagnostic-image',
            source: 'visible-image',
            mimeType: 'image/png',
            width: 2,
            height: 2,
            dataBase64: imageBase64,
          },
        ],
      },
    ],
    tools: [
      {
        name: 'fixture_tool',
        description: 'Fixture tool.',
        parameters: { type: 'object', properties: { value: { type: 'string' } }, required: ['value'] },
        strict: false,
      },
    ],
  };
  const first = await service.createTurn(request);
  assert.equal(first.usage.totalTokens, 16);
  assert.equal(first.remainingTokens, 63_984);
  await assert.rejects(
    () =>
      service.createTurn({
        runId: request.runId,
        requestId: randomUUID(),
        configurationRevision: request.configurationRevision,
        toolResults: [{ callId: 'diagnostic-call', output: 'small result' }],
      }),
    { message: 'Assistant token budget is exhausted; no request was sent.' },
  );
  assert.equal(requests.length, 1);

  const filename = path.join(directory, 'assistant-diagnostics.jsonl');
  const rows = (await fsp.readFile(filename, 'utf8'))
    .trim()
    .split('\n')
    .map(line => JSON.parse(line));
  const firstBudget = rows.find(event => event.event === 'request_budget' && event.phase === 'initial');
  const rejectedBudget = rows.find(event => event.event === 'request_budget' && event.phase === 'continuation');
  const rejection = rows.find(event => event.event === 'request_rejected' && event.phase === 'continuation');
  assert.equal(firstBudget.admitted, true);
  assert.equal(rejectedBudget.admitted, false);
  assert.equal(rejectedBudget.characters.encryptedReasoning, ciphertextSecret.length);
  assert.ok(rejectedBudget.contextTokensWithoutEncryptedEstimate < rejectedBudget.contextEstimate);
  assert.equal(rejection.reason, 'token_budget');
  assert.equal(rejection.remainingTokens, 63_984);
  assert.equal(rejection.images.count, 1);
  assert.equal(rejection.images.bytes, imageBytes.length);
  const received = rows.find(event => event.event === 'provider_usage' && event.stage === 'received');
  const accounted = rows.find(event => event.event === 'provider_usage' && event.stage === 'accounted');
  assert.deepEqual(received.providerUsage, {
    inputTokens: 12,
    outputTokens: 4,
    totalTokens: 16,
    cachedInputTokens: 3,
    reasoningTokens: 2,
  });
  assert.deepEqual(accounted.usage, { inputTokens: 12, outputTokens: 4, totalTokens: 16, estimated: false });
  assert.equal(accounted.spentAfter, 16);
  assert.equal(accounted.remainingAfter, 63_984);
  assert.equal((await fsp.stat(filename)).mode & 0o777, 0o600);
  const serialized = JSON.stringify(rows);
  for (const sensitive of [credential, promptSecret, argsSecret, ciphertextSecret, imageBase64])
    assert.ok(!serialized.includes(sensitive), `diagnostics must not include ${sensitive.slice(0, 24)}`);
});
test('large tool results receive a preflight rejection with character counts', async () => {
  const events = [];
  const requests = [];
  const queue = [
    completed([{ type: 'function_call', call_id: 'large-result-call', name: 'fixture_tool', arguments: '{}' }]),
    completed([message('Done')]),
  ];
  const { service, configuration } = await provider(
    {},
    async (_url, options) => {
      requests.push(JSON.parse(options.body));
      return jsonResponse(queue.shift());
    },
    async event => events.push(event),
  );
  const first = {
    ...turn(configuration.state().revision),
    tools: [{ name: 'fixture_tool', description: 'Fixture tool.', parameters: { type: 'object', properties: {} }, strict: false }],
  };
  const result = await service.createTurn(first);
  assert.equal(result.remainingTokens, 63_984);
  const output = 'LARGE_TOOL_RESULT_MARKER_'.repeat(11_200);
  assert.ok(output.length < 512_000);
  await assert.rejects(
    () =>
      service.createTurn({
        runId: first.runId,
        requestId: randomUUID(),
        configurationRevision: first.configurationRevision,
        toolResults: [{ callId: 'large-result-call', output }],
      }),
    { message: 'Assistant token budget is exhausted; no request was sent.' },
  );
  assert.equal(requests.length, 1);
  const budget = events.find(event => event.event === 'request_budget' && event.phase === 'continuation');
  const rejection = events.find(event => event.event === 'request_rejected' && event.phase === 'continuation');
  assert.equal(budget.admitted, false);
  assert.equal(budget.characters.toolResultText, output.length);
  assert.equal(rejection.reason, 'token_budget');
  assert.ok(!JSON.stringify(events).includes(output));
});
test('assistant run invalidation during accounted diagnostic write withholds the provider result', async () => {
  let reachedAccounted;
  const atAccounted = new Promise(resolve => (reachedAccounted = resolve));
  let releaseAccounted;
  const accountedWrite = new Promise(resolve => (releaseAccounted = resolve));
  const { service, configuration } = await provider(
    {},
    async () => jsonResponse(completed([message('Private completion')])),
    async event => {
      if (event.event === 'provider_usage' && event.stage === 'accounted') {
        reachedAccounted();
        await accountedWrite;
      }
    },
  );
  const request = turn(configuration.state().revision);
  const pending = service.createTurn(request);
  await atAccounted;
  await service.cancelRun(request.runId);
  releaseAccounted();
  await assert.rejects(pending, { message: 'Assistant completion arrived after the run was invalidated.' });
});
test('vision setting controls route tools, system guidance and toolkit_help output', async () => {
  const captures = [];
  assistantRouteService = {
    getConfiguration: async () => ({ config: { vision: false } }),
    createTurn: async value => {
      captures.push(value);
      return { text: 'Fixture response.' };
    },
    cancelRun: async () => {},
  };
  const body = {
    runId: randomUUID(),
    requestId: randomUUID(),
    configurationRevision: 1,
    messages: [{ role: 'user', content: 'Describe this image.' }],
    screen: '/dashboard',
  };
  const response = await postAssistantTurn({
    json: async () => body,
    signal: new AbortController().signal,
  });
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { text: 'Fixture response.' });
  assert.ok(!captures[0].tools.some(tool => tool.name === 'inspect_image'));
  assert.match(captures[0].messages[0].content, /Visual observations are disabled/);

  assistantRouteService = {
    ...assistantRouteService,
    getConfiguration: async () => ({ config: { vision: true } }),
  };
  await postAssistantTurn({
    json: async () => ({ ...body, runId: randomUUID(), requestId: randomUUID() }),
    signal: new AbortController().signal,
  });
  assert.ok(captures[1].tools.some(tool => tool.name === 'inspect_image'));
  assert.match(captures[1].messages[0].content, /Visual observations are enabled/);

  for (const vision of [false, true]) {
    const fixtureProvider = (await provider({ vision })).service;
    const ctx = await context({ provider: fixtureProvider });
    const help = JSON.parse((await executeAssistantTool(call('toolkit_help', {}), ctx)).output);
    assert.equal(help.tools.some(tool => tool.name === 'inspect_image'), vision);
    assert.equal(help.notes.includes('Visual observations are disabled.'), !vision);
  }
  assistantRouteService = undefined;
});
for (const protocol of ['responses', 'chat-completions'])
  test(`${protocol} correlates tools and native images over complete turns`, async () => {
    const requests = [];
    const pixels = Buffer.alloc(1536 * 1536 * 3);
    let noise = 0x12345678;
    for (let index = 0; index < pixels.length; index++) {
      noise = (Math.imul(noise, 1664525) + 1013904223) >>> 0;
      pixels[index] = noise >>> 24;
    }
    const fixtureImage = await sharp(pixels, { raw: { width: 1536, height: 1536, channels: 3 } })
      .png({ compressionLevel: 0 })
      .toBuffer();
    assert.ok(fixtureImage.byteLength >= 6 * 1024 * 1024);
    assert.ok(fixtureImage.byteLength <= 8 * 1024 * 1024);
    const imageBase64 = fixtureImage.toString('base64');
    assert.ok(imageBase64.length >= 9_000_000);
    const toolCall = {
      id: 'tool-item',
      type: 'function_call',
      name: 'inspect_image',
      call_id: 'inspect-1',
      arguments: '{"path":"fixture.png"}',
    };
    const { service, configuration } = await provider({ protocol, vision: true, imageMegapixels: 16 }, async (_url, options) => {
      requests.push(JSON.parse(options.body));
      const output = requests.length === 1 ? [toolCall] : [message('Red square')];
      return jsonResponse(
        protocol === 'responses'
          ? completed(output)
          : {
              choices: [
                {
                  finish_reason: requests.length === 1 ? 'tool_calls' : 'stop',
                  message:
                    requests.length === 1
                      ? {
                          role: 'assistant',
                          content: null,
                          tool_calls: [
                            {
                              id: 'inspect-1',
                              type: 'function',
                              function: { name: 'inspect_image', arguments: toolCall.arguments },
                            },
                          ],
                        }
                      : { role: 'assistant', content: 'Red square' },
                },
              ],
              usage: { prompt_tokens: 12, completion_tokens: 4, total_tokens: 16 },
            },
      );
    });
    const first = turn(configuration.state().revision);
    const result = await service.createTurn(first);
    assert.equal(result.toolCalls.length, 1);
    await service.createTurn({
      runId: first.runId,
      requestId: randomUUID(),
      configurationRevision: first.configurationRevision,
      toolResults: [{ callId: 'inspect-1', output: '{"path":"fixture.png"}' }],
      attachments: [
        {
          callId: 'inspect-1',
          image: {
            artifactId: 'continuation-large-image',
            source: 'visible-image',
            mimeType: 'image/png',
            width: 1536,
            height: 1536,
            dataBase64: imageBase64,
          },
        },
      ],
    });
    assert.ok(JSON.stringify(requests[1]).includes(imageBase64));
    assert.ok(JSON.stringify(requests[1]).includes(protocol === 'responses' ? 'input_image' : 'image_url'));
    assert.ok(JSON.stringify(requests[1]).includes(protocol === 'responses' ? 'function_call_output' : 'tool_call_id'));
    assert.equal(requests[0].model, 'fixture');

    const initialImageTurn = {
      ...turn(configuration.state().revision),
      messages: [
        {
          role: 'user',
          content: 'Inspect this image.',
          images: [
            {
              artifactId: 'initial-large-image',
              source: 'visible-image',
              mimeType: 'image/png',
              width: 1536,
              height: 1536,
              dataBase64: imageBase64,
            },
          ],
        },
      ],
    };
    await service.createTurn(initialImageTurn);
    assert.ok(JSON.stringify(requests[2]).includes(imageBase64));
    assert.ok(JSON.stringify(requests[2]).includes(protocol === 'responses' ? 'input_image' : 'image_url'));
  });
test('malformed image Base64 is rejected before provider transport', async () => {
  const malformed = [
    'AA?=', // non-Base64 alphabet
    '-_==', // URL-safe alphabet
    'AQI=\n', // newline
    'AAAA===', // excess padding
    'A=AA', // padding before the end
    'AQI', // omitted required padding
    'A', // impossible Base64 length
    'AB==', // noncanonical pad bits (canonical form is AA==)
  ];
  for (const dataBase64 of malformed) {
    let requestCount = 0;
    const { service, configuration } = await provider({ vision: true }, async () => {
      requestCount++;
      return jsonResponse(completed([message('Unexpected provider call')]));
    });
    const request = {
      ...turn(configuration.state().revision),
      messages: [
        {
          role: 'user',
          content: 'Inspect this image.',
          images: [
            {
              artifactId: 'malformed-image',
              source: 'visible-image',
              mimeType: 'image/png',
              width: 1,
              height: 1,
              dataBase64,
            },
          ],
        },
      ],
    };
    await assert.rejects(
      () => service.createTurn(request),
      dataBase64 === 'AB==' ? /Assistant image encoding or byte size is invalid/ : { name: 'ZodError' },
      `should reject ${JSON.stringify(dataBase64)}`,
    );
    assert.equal(requestCount, 0, `must reject ${JSON.stringify(dataBase64)} before provider transport`);
  }
});
test('image observations sent to the provider respect megapixel limits, fractions, orientation and no-upscale', async () => {
  const width = 1800;
  const height = 1200;
  const pixels = Buffer.alloc(width * height * 3);
  let noise = 0x10203040;
  for (let index = 0; index < pixels.length; index++) {
    noise = (Math.imul(noise, 1664525) + 1013904223) >>> 0;
    pixels[index] = noise >>> 24;
  }
  const original = await sharp(pixels, { raw: { width, height, channels: 3 } })
    .jpeg({ quality: 82 })
    .withMetadata({ orientation: 6 })
    .toBuffer();
  const providerSource = await sharp(pixels, { raw: { width, height, channels: 3 } })
    .png({ compressionLevel: 0 })
    .toBuffer();
  const providerSourceSnapshot = Buffer.from(providerSource);
  assert.ok(providerSource.byteLength <= 8 * 1024 * 1024, 'provider fixture must stay within the existing byte limit');
  const originalPath = path.join(root, 'datasets', 'megapixel-source.jpg');
  await fsp.writeFile(originalPath, original);

  for (const imageMegapixels of [1, 0.25]) {
    const requests = [];
    const { service, configuration } = await provider({ vision: true, imageMegapixels }, async (_url, options) => {
      requests.push(JSON.parse(options.body));
      return jsonResponse(completed([message('Observed image')]));
    });
    const inspected = await executeAssistantTool(
      call('inspect_image', { path: originalPath }),
      await context({ provider: service }),
    );
    assert.deepEqual(await fsp.readFile(originalPath), original, 'inspection must leave the source image unchanged');

    const inspectedBytes = Buffer.from(inspected.image.dataBase64, 'base64');
    const observed = await sharp(inspectedBytes).metadata();
    const targetPixels = imageMegapixels * 1_000_000;
    assert.ok(observed.width * observed.height <= targetPixels, `${imageMegapixels} MP observation exceeded its area limit`);
    assert.ok(Math.abs(observed.width / observed.height - 2 / 3) < 0.01, 'EXIF orientation and aspect ratio should be honored');

    await service.createTurn({
      ...turn(configuration.state().revision),
      messages: [
        {
          role: 'user',
          content: 'Inspect this image.',
          images: [
            {
              artifactId: `megapixel-source-${imageMegapixels}`,
              source: 'visible-image',
              mimeType: 'image/png',
              width,
              height,
              dataBase64: providerSource.toString('base64'),
            },
          ],
        },
      ],
    });
    const wireBody = JSON.stringify(requests[0]);
    const encoded = wireBody.match(/data:image\/png;base64,([A-Za-z0-9+/=]+)/)?.[1];
    assert.ok(encoded, 'provider request should contain a native PNG image part');
    const sent = await sharp(Buffer.from(encoded, 'base64')).metadata();
    assert.ok(sent.width * sent.height <= targetPixels, `${imageMegapixels} MP provider image exceeded its area limit`);
    assert.ok(Math.abs(sent.width / sent.height - 3 / 2) < 0.01, 'provider image should retain its aspect ratio');
    assert.deepEqual(await fsp.readFile(originalPath), original, 'provider normalization must leave the source image unchanged');
    assert.deepEqual(providerSource, providerSourceSnapshot, 'provider normalization must leave the input bytes unchanged');
  }

  const small = await sharp({ create: { width: 80, height: 40, channels: 3, background: '#336699' } }).png().toBuffer();
  const smallPath = path.join(root, 'datasets', 'megapixel-small.png');
  await fsp.writeFile(smallPath, small);
  const { service } = await provider({ vision: true, imageMegapixels: 0.25 });
  const smallObservation = await executeAssistantTool(
    call('inspect_image', { path: smallPath }),
    await context({ provider: service }),
  );
  assert.equal(smallObservation.image.width, 80, 'the megapixel limit must not upscale smaller images');
  assert.equal(smallObservation.image.height, 40, 'the megapixel limit must not upscale smaller images');
  assert.deepEqual(await fsp.readFile(smallPath), small, 'small source image should remain unchanged');
});
test('a fresh run resets tool rounds and advertises tools again', async () => {
  const requests = [];
  const { service, configuration } = await provider(
    { maxToolRounds: 1 },
    async (_url, options) => {
      const body = JSON.parse(options.body);
      requests.push(body);
      return jsonResponse(
        completed(
          body.tools?.length
            ? [{ type: 'function_call', name: 'fixture_tool', call_id: `call-${requests.length}`, arguments: '{}' }]
            : [message('Done')],
        ),
      );
    },
  );
  const makeRun = () => ({
    ...turn(configuration.state().revision),
    tools: [
      {
        name: 'fixture_tool',
        description: 'Fixture tool.',
        parameters: { type: 'object', properties: {} },
        strict: false,
      },
    ],
  });
  const first = makeRun();
  const firstResult = await service.createTurn(first);
  assert.equal(firstResult.round, 1);
  assert.equal(firstResult.toolCalls.length, 1);
  const finalFirstRound = await service.createTurn({
    runId: first.runId,
    requestId: randomUUID(),
    configurationRevision: first.configurationRevision,
    toolResults: [{ callId: 'call-1', output: 'ok' }],
  });
  assert.equal(finalFirstRound.round, 2);
  assert.equal(requests[1].tools.length, 0, 'the run limit suppresses tools after its first tool round');

  const second = makeRun();
  const secondResult = await service.createTurn(second);
  assert.equal(secondResult.round, 1);
  assert.equal(requests[2].tools.length, 1, 'a new run starts with its tools available');
});
test('fragmented SSE preserves UTF-8 and requires completed provider responses', async () => {
  const data = Buffer.from(
    'data: ' + JSON.stringify({ type: 'response.completed', response: completed([message('café')]) }) + '\r\n\r\n',
  );
  const stream = async function* () {
    for (let i = 0; i < data.length; i++) yield data.subarray(i, i + 1);
  };
  assert.equal(parseAssistantResponse(await readAssistantSse(stream(), 'responses'), 'responses').text, 'café');
  await assert.rejects(() =>
    readAssistantSse(
      (async function* () {
        yield Buffer.from('data: {}\n\n');
      })(),
      'responses',
    ),
  );
});
test('unknown tool calls and malformed argument JSON are rejected before any execution', async () => {
  for (const output of [
    [{ type: 'function_call', name: 'shell', call_id: 'x', arguments: '{}' }],
    [{ type: 'function_call', name: 'inspect_image', call_id: 'x', arguments: '{"path":' }],
  ]) {
    const { service, configuration } = await provider({}, async () => jsonResponse(completed(output)));
    await assert.rejects(() => service.createTurn(turn(configuration.state().revision)));
  }
});
test('cancel aborts the provider request and prevents reusing its run', async () => {
  let requested;
  const started = new Promise(resolve => (requested = resolve));
  const { service, configuration } = await provider(
    {},
    async (_url, options) =>
      new Promise((_resolve, reject) => {
        requested();
        options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true });
      }),
  );
  const first = turn(configuration.state().revision);
  const pending = service.createTurn(first);
  await started;
  await service.cancelRun(first.runId);
  await assert.rejects(pending, /cancelled/);
  await assert.rejects(() => service.createTurn({ ...first, requestId: randomUUID() }), /ended/);
});
test('configuration change cancels active requests and old revisions cannot execute', async () => {
  let requested;
  const started = new Promise(resolve => (requested = resolve));
  const { service, configuration } = await provider(
    {},
    async (_url, options) =>
      new Promise((_resolve, reject) => {
        requested();
        options.signal.addEventListener('abort', () => reject(options.signal.reason), { once: true });
      }),
  );
  const first = turn(configuration.state().revision);
  const pending = service.createTurn(first);
  await started;
  await service.setConfiguration({ config: { ...configuration.state().config, model: 'another-fixture' } });
  await assert.rejects(pending, /cancelled/);
  await assert.rejects(() => service.createTurn({ ...first, requestId: randomUUID(), runId: randomUUID() }), /changed/);
});
test('API operations include real GET mutation routes and PATCH queue reorder without bypassing review', () => {
  for (const endpoint of [
    '/api/jobs/job-1/sample_now',
    '/api/jobs/job-1/stop',
    '/api/jobs/job-1/kill',
    '/api/queue/0/start',
  ]) {
    assert.doesNotThrow(() => validateApiOperation({ endpoint, method: 'GET' }));
    assert.equal(isReadOnlyTool('toolkit_api', { endpoint, method: 'GET' }), false);
  }
  assert.doesNotThrow(() =>
    validateApiOperation({ endpoint: '/api/queue/0/reorder', method: 'PATCH', body: { orderedJobIds: ['x'] } }),
  );
  assert.equal(isReadOnlyTool('toolkit_api', { endpoint: '/api/jobs', method: 'GET' }), true);
  for (const endpoint of [
    'https://attacker.invalid/api/jobs',
    '/api/auth',
    '/api/assistant/settings',
    '/api/inference/../settings',
    '/api/jobs/%2e%2e/start',
  ])
    assert.throws(() => validateApiOperation({ endpoint, method: 'GET' }));
});
test('API reuse forwards auth, preserves errors, and withholds credential fields', async () => {
  let captured;
  const ctx = await context({
    fetcher: async (url, options) => {
      captured = { url, options };
      return new Response(JSON.stringify({ HF_TOKEN: 'secret', DATASETS_FOLDER: '/fixture' }), { status: 200 });
    },
  });
  const result = await executeAssistantTool(call('toolkit_api', { endpoint: '/api/settings', method: 'GET' }), ctx);
  assert.equal(captured.options.headers.Authorization, 'Bearer fixture-auth');
  assert.ok(!result.output.includes('secret'));
  await assert.rejects(
    () =>
      executeAssistantTool(call('toolkit_api', { endpoint: '/api/jobs', method: 'GET' }), {
        ...ctx,
        fetcher: async () => new Response('{"error":"fixture failure"}', { status: 500 }),
      }),
    /fixture failure/,
  );
});
test('running jobs reject full configuration writes while preserving sample-only editing', async () => {
  const ctx = await context({ fetcher: async () => new Response('{"status":"running"}', { status: 200 }) });
  await assert.rejects(
    () =>
      executeAssistantTool(
        call('toolkit_api', { endpoint: '/api/jobs', method: 'POST', body: { id: 'job-1', job_config: {} } }),
        ctx,
      ),
    /sample-only/,
  );
});
test('API credentials are redacted in nested job configuration and error responses', async () => {
  const payload = {
    job_config: JSON.stringify({ config: { api_key: 'fixture-hidden', model: 'visible' } }),
    engine: { authorization: 'fixture-hidden' },
  };
  const ctx = await context({ fetcher: async () => new Response(JSON.stringify(payload)) });
  const result = await executeAssistantTool(call('toolkit_api', { endpoint: '/api/jobs', method: 'GET' }), ctx);
  assert.ok(!result.output.includes('fixture-hidden'));
  assert.ok(result.output.includes('visible'));
  await assert.rejects(
    () =>
      executeAssistantTool(call('toolkit_api', { endpoint: '/api/jobs', method: 'GET' }), {
        ...ctx,
        fetcher: async () => new Response(JSON.stringify(payload), { status: 500 }),
      }),
    error => !error.message.includes('fixture-hidden') && error.message.includes('REDACTED'),
  );
});
test('oversized API streams stop reading at the response budget', async () => {
  let canceled = false;
  const stream = new ReadableStream({
    pull(controller) {
      controller.enqueue(new Uint8Array(300_000));
    },
    cancel() {
      canceled = true;
    },
  });
  const ctx = await context({ fetcher: async () => new Response(stream) });
  await assert.rejects(
    () => executeAssistantTool(call('toolkit_api', { endpoint: '/api/jobs', method: 'GET' }), ctx),
    /too large/,
  );
  assert.equal(canceled, true);
});
test('private runtime metadata cannot be read while captions work before the output folder exists', async () => {
  const ctx = await context();
  const engine = path.join(root, 'output', 'engine.json');
  await fsp.writeFile(engine, '{"token":"private"}');
  await assert.rejects(
    () => executeAssistantTool(call('dataset_file', { operation: 'read', path: engine }), ctx),
    /Private runtime/,
  );
  const hidden = path.join(root, 'datasets', '.hidden.json');
  await fsp.writeFile(hidden, '{}');
  await assert.rejects(
    () => executeAssistantTool(call('dataset_file', { operation: 'read', path: hidden }), ctx),
    /Private runtime/,
  );
  const caption = path.join(root, 'datasets', 'early.txt');
  await fsp.writeFile(caption, 'A fixture.');
  const output = path.join(root, 'output');
  await fsp.rename(output, output + '.saved');
  try {
    const result = await executeAssistantTool(call('dataset_file', { operation: 'read', path: caption }), ctx);
    assert.equal(JSON.parse(result.output).text, 'A fixture.');
  } finally {
    await fsp.rename(output + '.saved', output);
  }
});
test('path boundaries reject traversal, prefix collisions and symlink escapes', async () => {
  const dataset = await settings.getDatasetsRoot();
  await fsp.symlink(await settings.getDataRoot(), path.join(dataset, 'escape'));
  for (const name of ['../data/secret.txt', dataset + '-other/secret.txt', 'escape/secret.txt'])
    await assert.rejects(() => scopedPath(name, [dataset], true));
  await assert.rejects(() => scopedPath(dataset, [dataset], true));
});
test('CPU image creation, transformation and inspection return native image observations', async () => {
  const ctx = await context();
  const image = path.join(root, 'datasets', 'cpu.png');
  await executeAssistantTool(
    call('edit_image', { destination: image, width: 16, height: 12, background: '#ff0000' }),
    ctx,
  );
  const viewed = await executeAssistantTool(call('inspect_image', { path: image }), ctx);
  assert.equal(viewed.image.mimeType, 'image/png');
  assert.equal(viewed.image.width, 16);
  assert.equal(viewed.image.height, 12);
  const edited = path.join(root, 'datasets', 'edited.png');
  await executeAssistantTool(call('edit_image', { source: image, destination: edited, rotate: 90 }), ctx);
  const rotated = await sharp(edited).metadata();
  assert.equal(rotated.width, 12);
  assert.equal(rotated.height, 16);
  await assert.rejects(
    () => executeAssistantTool(call('edit_image', { destination: image, width: 2, height: 2 }), ctx),
    /exists/,
  );
  await assert.rejects(
    () =>
      executeAssistantTool(
        call('edit_image', { destination: edited, svg: '<svg><image href="file:///etc/passwd"/></svg>' }),
        ctx,
      ),
    /self-contained/,
  );
});
test('vision off reports inability to inspect rather than sending image data', async () => {
  const ctx = await context({ provider: (await provider({ vision: false })).service });
  await assert.rejects(
    () => executeAssistantTool(call('inspect_image', { path: path.join(root, 'datasets', 'cpu.png') }), ctx),
    /visual observations/,
  );
});
test('caption edits require overwrite and image copies/renames preserve sidecars', async () => {
  const ctx = await context();
  const image = path.join(root, 'datasets', 'cpu.png');
  const caption = image.replace('.png', '.txt');
  await executeAssistantTool(
    call('dataset_file', { operation: 'write', path: caption, text: 'A red rectangle.' }),
    ctx,
  );
  await assert.rejects(
    () => executeAssistantTool(call('dataset_file', { operation: 'write', path: caption, text: 'changed' }), ctx),
    /overwrite/,
  );
  const copied = image.replace('cpu.png', 'copied.png');
  await executeAssistantTool(call('dataset_file', { operation: 'copy', path: image, destination: copied }), ctx);
  assert.equal(await fsp.readFile(copied.replace('.png', '.txt'), 'utf8'), 'A red rectangle.');
  const renamed = copied.replace('copied.png', 'renamed.png');
  await executeAssistantTool(call('dataset_file', { operation: 'rename', path: copied, destination: renamed }), ctx);
  assert.equal(fs.existsSync(copied), false);
  assert.equal(fs.existsSync(renamed), true);
  await assert.rejects(
    () =>
      executeAssistantTool(
        call('dataset_file', { operation: 'rename', path: renamed, destination: renamed, overwrite: true }),
        ctx,
      ),
    /must differ/,
  );
});
test('explicitly small batch limits reject oversized API requests without contacting routes', async () => {
  let called = false;
  const ctx = await context({
    provider: (await provider({ maxBatchSize: 1 })).service,
    fetcher: async () => {
      called = true;
      return new Response('{}');
    },
  });
  await assert.rejects(
    () =>
      executeAssistantTool(
        call('toolkit_api', { endpoint: '/api/caption/getBatch', method: 'POST', body: { imgPaths: ['a', 'b'] } }),
        ctx,
      ),
    /batch limit/,
  );
  assert.equal(called, false);
});
test('LoRA upload uses bounded chunks and the existing start/chunk/finish API', async () => {
  const source = path.join(root, 'models', 'fixture.safetensors');
  const handle = await fsp.open(source, 'w');
  await handle.truncate(6 * 1024 * 1024);
  await handle.close();
  const requests = [];
  const ctx = await context({
    fetcher: async (url, options) => {
      requests.push({ url: new URL(url), options });
      return new Response(
        JSON.stringify(
          new URL(url).searchParams.get('action') === 'start'
            ? { uploadId: 'fixture-upload' }
            : { path: '/fixture/loras/fixture.safetensors' },
        ),
      );
    },
  });
  const result = await executeAssistantTool(call('upload_file', { source, target: 'lora' }), ctx);
  assert.equal(result.changed, true);
  assert.deepEqual(
    requests.map(request => request.url.searchParams.get('action')),
    ['start', 'chunk', 'chunk', 'finish'],
  );
  assert.equal(requests[1].options.body.byteLength, 5 * 1024 * 1024);
  assert.equal(requests[2].options.body.byteLength, 1024 * 1024);
});
test('failed LoRA uploads cancel partial uploads before reporting failure', async () => {
  const actions = [];
  const ctx = await context({
    fetcher: async url => {
      const action = new URL(url).searchParams.get('action');
      actions.push(action);
      return new Response(
        JSON.stringify(action === 'start' ? { uploadId: 'fixture-upload' } : { error: 'Chunk rejected' }),
        { status: action === 'chunk' ? 500 : 200 },
      );
    },
  });
  await assert.rejects(
    () =>
      executeAssistantTool(
        call('upload_file', { source: path.join(root, 'models', 'fixture.safetensors'), target: 'lora' }),
        ctx,
      ),
    /Chunk rejected/,
  );
  assert.deepEqual(actions, ['start', 'chunk', 'cancel']);
});
test('dataset uploads forward multipart file and dataset fields through the UI API', async () => {
  let captured;
  const ctx = await context({
    fetcher: async (url, options) => {
      captured = { url: new URL(url), options };
      return new Response('{"files":["cpu.png"]}');
    },
  });
  await executeAssistantTool(
    call('upload_file', { source: path.join(root, 'datasets', 'cpu.png'), target: 'dataset', datasetName: 'fixture' }),
    ctx,
  );
  assert.equal(captured.url.pathname, '/api/datasets/upload');
  assert.equal(captured.options.body.get('datasetName'), 'fixture');
  assert.equal(captured.options.body.get('files').name, 'cpu.png');
});
