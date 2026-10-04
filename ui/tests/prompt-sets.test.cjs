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

function loadMultipoint() {
  const filename = path.resolve(__dirname, '../src/app/jobs/new/fizgigMultipoint.ts');
  const compiled = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const instance = new Module(filename, module);
  instance._compile(compiled, filename);
  return instance.exports;
}

function loadRoute(toolkitRoot) {
  const source = fs.readFileSync(routePath, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, esModuleInterop: true, target: ts.ScriptTarget.ES2020 },
  }).outputText;
  const routeModule = new Module(routePath, module);
  routeModule.filename = routePath;
  routeModule.paths = Module._nodeModulePaths(path.dirname(routePath));
  const originalRequire = routeModule.require.bind(routeModule);
  routeModule.require = id => id === '@/paths' ? { TOOLKIT_ROOT: toolkitRoot }
    : id === '@/app/jobs/new/fizgigMultipoint' ? loadMultipoint() : originalRequire(id);
  routeModule._compile(compiled, routePath);
  return routeModule.exports;
}

function request(name) {
  const url = new URL('http://localhost/api/prompt-sets');
  if (name !== undefined) url.searchParams.set('name', name);
  return new NextRequest(url);
}

function postRequest(name, promptSet, overwrite = false) {
  return new NextRequest('http://localhost/api/prompt-sets', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, prompt_set: promptSet, overwrite }),
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

    const anchoredSet = { ...promptSet, anchor_prompts: [
      { prompt: 'a medium-sized dog', negative_prompt: 'blur' }, { prompt: 'a landscape' },
    ] };
    response = await route.POST(postRequest('anchors', anchoredSet));
    assert.equal(response.status, 201);
    response = await route.GET(request('anchors'));
    assert.deepEqual((await response.json()).prompt_set, anchoredSet);
    assert.deepEqual(YAML.parse(fs.readFileSync(path.join(directory, 'anchors.yaml'), 'utf8')), anchoredSet);
    for (const anchor_prompts of [null, 'dog', [{ prompt: 1 }], [{ prompt: 'dog', negative_prompt: [] }]]) {
      response = await route.POST(postRequest('bad anchors', { ...promptSet, anchor_prompts }));
      assert.equal(response.status, 400);
    }
    response = await route.DELETE(request('anchors'));
    assert.equal(response.status, 200);

    const multipointSet = { ...anchoredSet, multipoint: true, multipoint_config: {
      points: [
        { id: 'baby', strength: -.5, prefix: 'baby', negative_prefix: 'adult' },
        { id: 'zero', strength: 0, prefix: '', negative_prefix: 'blur' },
        { id: 'mega', strength: 2, prefix: 'mega', negative_prefix: 'baby' },
      ], neutral_negative_prefix: 'noise', prompt_entries: [
        { kind: 'simple', prompt: 'cat\nblue coat' },
        { kind: 'specific', neutral_prompt: 'legacy neutral', neutral_negative_prompt: '', targets: [
          { point_id: 'baby', prompt: 'baby cat', negative_prompt: 'adult' },
          { point_id: 'zero', prompt: 'cat', negative_prompt: 'blur' },
          { point_id: 'mega', prompt: 'mega cat', negative_prompt: '' },
        ] },
      ],
    } };
    response = await route.POST(postRequest('multipoint', multipointSet));
    assert.equal(response.status, 201);
    assert.deepEqual((await (await route.GET(request('multipoint'))).json()).prompt_set, multipointSet);
    assert.deepEqual(YAML.parse(fs.readFileSync(path.join(directory, 'multipoint.yaml'), 'utf8')), multipointSet);
    for (const edit of [
      { ...multipointSet, multipoint: 'yes' },
      { ...multipointSet, multipoint_config: undefined },
      { ...multipointSet, multipoint_config: { ...multipointSet.multipoint_config, points: [multipointSet.multipoint_config.points[1]] } },
      { ...multipointSet, multipoint_config: { ...multipointSet.multipoint_config, points: [multipointSet.multipoint_config.points[0], { ...multipointSet.multipoint_config.points[2], strength: -.5 }] } },
      { ...multipointSet, multipoint_config: { ...multipointSet.multipoint_config, points: [{ ...multipointSet.multipoint_config.points[0], strength: NaN }] } },
      { ...multipointSet, multipoint_config: { ...multipointSet.multipoint_config, prompt_entries: [{ kind: 'specific', neutral_prompt: 'cat', neutral_negative_prompt: '', targets: [] }] } },
    ]) {
      assert.equal((await route.POST(postRequest('invalid multipoint', edit))).status, 400);
    }
    assert.equal((await route.DELETE(request('multipoint'))).status, 200);

    response = await route.POST(postRequest('Alpha', promptSet));
    assert.equal(response.status, 201);
    response = await route.GET(request());
    assert.deepEqual((await response.json()).names, ['Alpha', 'zeta set']);

    response = await route.POST(postRequest('zeta set', { ...promptSet, positive_prefix: 'changed' }));
    assert.equal(response.status, 409);
    assert.deepEqual(YAML.parse(fs.readFileSync(path.join(directory, 'zeta set.yaml'), 'utf8')), promptSet);

    const overwrittenSet = { ...promptSet, positive_prefix: 'changed' };
    response = await route.POST(postRequest('zeta set', overwrittenSet, true));
    assert.equal(response.status, 200);
    assert.deepEqual(YAML.parse(fs.readFileSync(path.join(directory, 'zeta set.yaml'), 'utf8')), overwrittenSet);
    assert.deepEqual((await (await route.GET(request('zeta set'))).json()).prompt_set, overwrittenSet);
    assert.equal(fs.readdirSync(directory).some(name => name.endsWith('.tmp')), false);

    response = await route.POST(postRequest('missing', promptSet, true));
    assert.equal(response.status, 404);

    const protectedFile = path.join(toolkitRoot, 'protected.txt');
    fs.writeFileSync(protectedFile, 'leave unchanged');
    fs.symlinkSync(protectedFile, path.join(directory, 'linked.yaml'));
    response = await route.POST(postRequest('linked', promptSet, true));
    assert.equal(response.status, 400);
    assert.equal(fs.readFileSync(protectedFile, 'utf8'), 'leave unchanged');

    response = await route.DELETE(request('linked'));
    assert.equal(response.status, 400);
    assert.equal(fs.readFileSync(protectedFile, 'utf8'), 'leave unchanged');

    response = await route.DELETE(request('../escape'));
    assert.equal(response.status, 400);
    response = await route.DELETE(request('missing'));
    assert.equal(response.status, 404);

    response = await route.DELETE(request('zeta set'));
    assert.equal(response.status, 200);
    assert.deepEqual(await response.json(), { name: 'zeta set' });
    assert.equal(fs.existsSync(path.join(directory, 'zeta set.yaml')), false);
    assert.deepEqual((await (await route.GET(request())).json()).names, ['Alpha']);
    response = await route.DELETE(request('zeta set'));
    assert.equal(response.status, 404);

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
