const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const test = require('node:test');
const ts = require('typescript');

const projectRoot = path.resolve(__dirname, '../..');

function loadResolver(toolkitRoot) {
  const filename = path.resolve(__dirname, '../cron/pythonPath.ts');
  const instance = new Module(filename, module);
  instance.filename = filename;
  instance.paths = Module._nodeModulePaths(path.dirname(filename));
  const originalRequire = instance.require.bind(instance);
  instance.require = id => id === './paths' ? { TOOLKIT_ROOT: toolkitRoot } : originalRequire(id);
  instance._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    fileName: filename,
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText, filename);
  return instance.exports;
}

function touchExecutable(filePath) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, 'fixture executable');
  fs.chmodSync(filePath, 0o755);
}

test('Python resolver prefers explicit override, then active Conda over repo venv', t => {
  const tempRoot = fs.mkdtempSync(path.join(projectRoot, 'tmp', 'python-path-'));
  t.after(() => fs.rmSync(tempRoot, { recursive: true, force: true }));

  const toolkitRoot = path.join(tempRoot, 'toolkit');
  const condaPrefix = path.join(tempRoot, 'envs', 'ai-toolkit');
  const override = path.join(tempRoot, 'explicit', 'python');
  const repoVenv = path.join(toolkitRoot, '.venv', 'bin', 'python');
  const condaPython = path.join(condaPrefix, 'bin', 'python');
  touchExecutable(override);
  touchExecutable(repoVenv);
  touchExecutable(condaPython);

  const previous = {
    AI_TOOLKIT_PYTHON: process.env.AI_TOOLKIT_PYTHON,
    CONDA_PREFIX: process.env.CONDA_PREFIX,
  };
  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  const resolver = loadResolver(toolkitRoot);
  process.env.AI_TOOLKIT_PYTHON = '';
  process.env.CONDA_PREFIX = condaPrefix;
  assert.equal(resolver.resolvePythonPath(), condaPython,
    'the active Conda interpreter must win over a repository-local venv');

  process.env.AI_TOOLKIT_PYTHON = override;
  assert.equal(resolver.resolvePythonPath(), override, 'an explicit interpreter override has highest priority');
});

test('Python resolver falls back to repo venv when active Conda interpreter is absent', t => {
  const tempRoot = fs.mkdtempSync(path.join(projectRoot, 'tmp', 'python-path-'));
  t.after(() => fs.rmSync(tempRoot, { recursive: true, force: true }));
  const toolkitRoot = path.join(tempRoot, 'toolkit');
  const repoPython = path.join(toolkitRoot, '.venv', 'bin', 'python');
  touchExecutable(repoPython);

  const previous = {
    AI_TOOLKIT_PYTHON: process.env.AI_TOOLKIT_PYTHON,
    CONDA_PREFIX: process.env.CONDA_PREFIX,
  };
  t.after(() => {
    for (const [key, value] of Object.entries(previous)) {
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  });

  process.env.AI_TOOLKIT_PYTHON = '';
  process.env.CONDA_PREFIX = path.join(tempRoot, 'missing-conda');
  assert.equal(loadResolver(toolkitRoot).resolvePythonPath(), repoPython);
});

test('Python resolver rejects an invalid explicit interpreter override', t => {
  const tempRoot = fs.mkdtempSync(path.join(projectRoot, 'tmp', 'python-path-'));
  t.after(() => fs.rmSync(tempRoot, { recursive: true, force: true }));
  const toolkitRoot = path.join(tempRoot, 'toolkit');
  touchExecutable(path.join(toolkitRoot, '.venv', 'bin', 'python'));
  const previous = process.env.AI_TOOLKIT_PYTHON;
  t.after(() => {
    if (previous === undefined) delete process.env.AI_TOOLKIT_PYTHON;
    else process.env.AI_TOOLKIT_PYTHON = previous;
  });

  process.env.AI_TOOLKIT_PYTHON = path.join(tempRoot, 'missing', 'python');
  assert.throws(() => loadResolver(toolkitRoot).resolvePythonPath(), /AI_TOOLKIT_PYTHON/);
});
