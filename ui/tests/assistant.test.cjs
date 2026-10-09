const assert = require('node:assert/strict');
const { test, before, after } = require('node:test');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const Module = require('node:module');
const ts = require('typescript');
const { randomUUID } = require('node:crypto');

const sourceRoot = path.resolve(__dirname, '../src');
const cache = new Map();
let root;
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
const { DEFAULT_ASSISTANT_CONFIGURATION, assistantEndpoint, assistantConfigurationSchema } =
  load('assistant/AssistantProvider');
const { NodeAssistantConfiguration } = load('server/assistant/NodeAssistantConfiguration');
const { NodeAssistantProvider, parseAssistantResponse, readAssistantSse } = load(
  'server/assistant/NodeAssistantProvider',
);
const { executeAssistantTool, scopedPath, validateApiOperation } = load('server/assistant/toolExecutor');
const { ASSISTANT_TOOLS, isReadOnlyTool } = load('assistant/tools');
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
async function provider(options = {}, transport = async () => jsonResponse(completed([message('Done')]))) {
  const configuration = new NodeAssistantConfiguration();
  await configuration.update({
    config: {
      ...DEFAULT_ASSISTANT_CONFIGURATION,
      baseUrl: 'https://fixture.invalid/v1',
      authMode: 'none',
      model: 'fixture',
      ...options,
    },
  });
  return { service: new NodeAssistantProvider({ configuration, transport }), configuration };
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
for (const protocol of ['responses', 'chat-completions'])
  test(`${protocol} correlates tools and native images over complete turns`, async () => {
    const requests = [];
    const fixtureImage = await sharp({ create: { width: 2, height: 2, channels: 3, background: '#ff0000' } })
      .png()
      .toBuffer();
    const toolCall = {
      id: 'tool-item',
      type: 'function_call',
      name: 'inspect_image',
      call_id: 'inspect-1',
      arguments: '{"path":"fixture.png"}',
    };
    const { service, configuration } = await provider({ protocol, vision: true }, async (_url, options) => {
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
            artifactId: 'image-1',
            source: 'visible-image',
            mimeType: 'image/png',
            width: 2,
            height: 2,
            dataBase64: fixtureImage.toString('base64'),
          },
        },
      ],
    });
    assert.ok(JSON.stringify(requests[1]).includes(protocol === 'responses' ? 'input_image' : 'image_url'));
    assert.ok(JSON.stringify(requests[1]).includes(protocol === 'responses' ? 'function_call_output' : 'tool_call_id'));
    assert.equal(requests[0].model, 'fixture');
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
