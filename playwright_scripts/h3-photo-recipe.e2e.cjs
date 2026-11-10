const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const ui = path.join(root, 'ui');
const ts = require(path.join(ui, 'node_modules/typescript'));
const swc = require(path.join(ui, 'node_modules/next/dist/build/swc'));
const out = path.join(root, 'tmp/h3-photo');
function loadTs(file) {
  const filename = path.join(ui, 'src', file);
  const mod = new Module(filename, module); mod.filename = filename; mod.paths = Module._nodeModulePaths(path.join(ui, 'node_modules'));
  const original = mod.require.bind(mod);
  mod.require = id => {
    if (id === '@/helpers/basic') return { isMac: false };
    if (id === '@/helpers/defaultSamples') return loadTs('helpers/defaultSamples.ts');
    if (id === './trainingCapabilities') return { defaultDiffusionKTOConfig: {} };
    return original(id);
  };
  mod._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText, filename);
  return mod.exports;
}
async function main() {
  const config = structuredClone(loadTs('app/jobs/new/jobConfig.ts').defaultJobConfig);
  const p = config.config.process[0];
  config.config.name = 'h3-fixture';
  p.type = 'diffusion_trainer'; p.model.arch = 'minimax_h3'; p.model.name_or_path = '/fixture/base';
  p.model.model_kwargs = { partition: 'fl2va_pruned', transformer_path: '/fixture/transformer' };
  p.model.low_vram = true; p.model.quantize = true; p.train.gradient_checkpointing = true; p.train.steps = 1234;
  p.datasets[0].folder_path = '/fixture/dataset'; p.datasets[0].num_frames = 49;
  p.sample.samples[0].prompt = 'fixture retained prompt'; p.sample.num_frames = 49; p.sample.guidance_scale = 3;
  const source = path.join(root, 'extensions_built_in/diffusion_models/ui.tsx');
  const transpiled = await swc.transform(fs.readFileSync(source, 'utf8'), { filename: source, jsc: { parser: { syntax: 'typescript', tsx: true }, transform: { react: { runtime: 'automatic', development: false } }, target: 'es2020' }, module: { type: 'commonjs' }, sourceMaps: false, isModule: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1365, height: 1000 } }); page.setDefaultTimeout(60000);
  let payload; const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('dialog', dialog => dialog.dismiss());
  await page.route('**/api/**', route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/model_archs') return route.fulfill({ json: { modules: [{ id: 'extensions_built_in/diffusion_models', file: source, mtimeMs: fs.statSync(source).mtimeMs, code: transpiled.code }], errors: [] } });
    if (url.pathname === '/api/jobs' && route.request().method() === 'GET') return route.fulfill({ json: { id: 'fixture', gpu_ids: '0', job_config: JSON.stringify(config) } });
    if (url.pathname === '/api/jobs' && route.request().method() === 'POST') { payload = route.request().postDataJSON(); return route.fulfill({ status: 400, json: { error: 'Synthetic save captured' } }); }
    if (url.pathname === '/api/settings') return route.fulfill({ json: { TRAINING_FOLDER: '/fixture/output', DATASETS_FOLDER: '/fixture', MODELS_PATH: '/fixture/models' } });
    if (url.pathname === '/api/monitor') return route.fulfill({ contentType: 'text/event-stream', body: `event: init\ndata: ${JSON.stringify({ t: Date.now(), cpu: null, gpu: { gpus: [] }, history: [] })}\n\n` });
    if (url.pathname === '/api/datasets/list') return route.fulfill({ json: ['dataset'] });
    return route.abort();
  });
  const save = async () => { payload = undefined; const response = page.waitForResponse(r => new URL(r.url()).pathname === '/api/jobs' && r.request().method() === 'POST'); await page.getByRole('button', { name: 'Update Job' }).click(); await response; assert.ok(payload); return payload.job_config.config.process[0]; };
  try {
    await page.goto('http://0.0.0.0:8676/jobs/new?id=fixture', { waitUntil: 'domcontentloaded', timeout: 120000 });
    const recipe = page.getByLabel('Training recipe'); await recipe.waitFor();
    let saved = await save();
    assert.equal(saved.model.model_kwargs.h3_photo_training, undefined); assert.equal(saved.sample.num_frames, 49); assert.equal(saved.train.steps, 1234);
    await recipe.focus(); await page.keyboard.press('ArrowDown'); await page.getByRole('option', { name: 'Apply still-photo recipe' }).waitFor(); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
    await page.getByText('Apply still-photo recipe', { exact: true }).waitFor();
    await page.screenshot({ path: path.join(out, 'recipe-desktop.png'), fullPage: false });
    await page.setViewportSize({ width: 390, height: 844 }); await recipe.evaluate(el => el.scrollIntoView({ block: 'center' })); await page.screenshot({ path: path.join(out, 'recipe-narrow.png'), fullPage: false }); await page.setViewportSize({ width: 1365, height: 1000 });
    await page.getByRole('button', { name: 'Show Advanced' }).click();
    await page.locator('.monaco-editor').waitFor();
    const advanced = await page.locator('.monaco-editor').innerText();
    assert.match(advanced, /blocks\.20\./); assert.match(advanced, /token_refiner/);
    await page.getByRole('button', { name: 'Show Simple' }).click();
    saved = await save();
    assert.equal(saved.model.model_kwargs.h3_photo_training, true); assert.equal(saved.train.lr, 1e-6); assert.equal(saved.train.optimizer, 'automagic3');
    assert.equal(saved.sample.num_frames, 1); assert.equal(saved.sample.guidance_scale, 1); assert.equal(saved.sample.comfy.enabled, false);
    assert.equal(saved.train.steps, 1234); assert.equal(saved.datasets[0].folder_path, '/fixture/dataset'); assert.equal(saved.sample.samples[0].prompt, 'fixture retained prompt');
    assert.equal(saved.model.name_or_path, '/fixture/base'); assert.equal(saved.model.model_kwargs.transformer_path, '/fixture/transformer'); assert.equal(saved.model.low_vram, true); assert.equal(saved.model.quantize, true); assert.equal(saved.train.gradient_checkpointing, true);
    await page.getByText('Learning Rate', { exact: true }).locator('..').locator('input').fill('0.000002');
    await page.getByText('Photo mode — customized settings', { exact: true }).waitFor();
    await recipe.click(); await page.getByRole('option', { name: 'Video mode (keep current settings)' }).click();
    saved = await save(); assert.equal(saved.model.model_kwargs.h3_photo_training, false); assert.equal(saved.train.lr, 0.000002); assert.equal(saved.sample.num_frames, 1); assert.equal(saved.sample.samples[0].prompt, 'fixture retained prompt');
    assert.deepEqual(errors, []); console.log('H3 photo recipe browser scenario passed');
  } catch (error) { console.error('Page errors:', errors, 'Body:', (await page.locator('body').innerText()).slice(0, 5000)); await page.screenshot({ path: path.join(out, 'failure.png') }); throw error; }
  finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
