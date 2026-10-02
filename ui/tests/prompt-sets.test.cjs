const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');
const Module = require('node:module');
const ts = require('typescript');
const { NextRequest } = require('next/server');
const YAML = require('yaml');

const routePath = path.resolve(__dirname, '../src/app/api/prompt-sets/route.ts');

function loadRoute(toolkitRoot) {
  const source = fs.readFileSync(routePath, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const routeModule = new Module(routePath, module);
  routeModule.filename = routePath;
  routeModule.paths = Module._nodeModulePaths(path.dirname(routePath));
  const originalRequire = routeModule.require.bind(routeModule);
  routeModule.require = id => id === '@/paths' ? { TOOLKIT_ROOT: toolkitRoot } : originalRequire(id);
  routeModule._compile(compiled, routePath);
  return routeModule.exports;
}

function request(name) {
  const url = new URL('http://localhost/api/prompt-sets');
  if (name !== undefined) url.searchParams.set('name', name);
  return new NextRequest(url);
}

function postRequest(name, promptSet) {
  return new NextRequest('http://localhost/api/prompt-sets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, prompt_set: promptSet }),
  });
}

test('prompt sets save and load ordered mixed prompts and shared prefixes', async () => {
  const toolkitRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'aitk-prompt-sets-test-'));
  try {
    const route = loadRoute(toolkitRoot);
    const directory = path.join(toolkitRoot, 'prompt_sets');
    assert.equal(fs.existsSync(directory), false);
    let response = await route.GET(request());
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { names: [] });
    assert.equal(fs.statSync(directory).isDirectory(), true);

    const promptSet = {
      version: 1,
      prompt_entries: [
        { kind: 'simple', prompt: 'a detailed illustration\nwith line art' },
        {
          kind: 'specific', neutral_prompt: 'a person', positive_prompt: 'a close-up',
          negative_prompt: 'a full-body shot', cfg_negative_prompt: 'blur',
          cfg_negative_prompt_positive: 'distant', cfg_negative_prompt_negative: 'close-up',
        },
      ],
      positive_prefix: 'clear', negative_prefix: 'hazy',
      cfg_negative_prefix: 'muddy', cfg_negative_prefix_positive: 'distant',
      cfg_negative_prefix_negative: 'close-up',
    };

    response = await route.POST(postRequest('zeta set', promptSet));
    assert.equal(response.status, 201);
    const yaml = fs.readFileSync(path.join(directory, 'zeta set.yaml'), 'utf8');
    assert.deepEqual(YAML.parse(yaml), promptSet);

    response = await route.GET(request('zeta set'));
    assert.equal(response.status, 200);
    assert.deepEqual((await response.json()).prompt_set, promptSet);

    response = await route.POST(postRequest('Alpha', promptSet));
    assert.equal(response.status, 201);
    response = await route.GET(request());
    assert.deepEqual((await response.json()).names, ['Alpha', 'zeta set']);

    response = await route.POST(postRequest('zeta set', { ...promptSet, positive_prefix: 'changed' }));
    assert.equal(response.status, 409);
    assert.deepEqual(YAML.parse(fs.readFileSync(path.join(directory, 'zeta set.yaml'), 'utf8')), promptSet);

    response = await route.POST(postRequest('../escape', promptSet));
    assert.equal(response.status, 400);
    response = await route.POST(postRequest('invalid', { ...promptSet, prompt_entries: [{ kind: 'simple', prompt: 42 }] }));
    assert.equal(response.status, 400);

    fs.writeFileSync(path.join(directory, 'broken.yaml'), 'prompt_entries: [\n');
    response = await route.GET(request('broken'));
    assert.equal(response.status, 400);
  } finally {
    fs.rmSync(toolkitRoot, { recursive: true, force: true });
  }
});
