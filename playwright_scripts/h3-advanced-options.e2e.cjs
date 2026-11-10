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
  let p = config.config.process[0];
  config.config.name = 'h3-advanced-fixture';
  p.type = 'diffusion_trainer'; p.model.arch = 'minimax_h3_image'; p.model.name_or_path = 'Comfy-Org/MiniMax-H3-Image';
  p.model.model_kwargs = { partition: 'image', dit_image_path: '/fixture/dit', text_encoder_path: '/fixture/te-legacy', image_vae_path: '/fixture/vae-legacy' };
  p.model.te_name_or_path = '/fixture/te-active'; p.model.text_encoder_path = '/fixture/te-old'; p.model.vae_path = '/fixture/vae-active';
  p.network.network_kwargs.only_if_contains = [];
  p.network.network_kwargs.ignore_if_contains = ['fixture.unknown'];
  if (process.env.H3_PRESENTATION_ONLY === '1') p.network.network_kwargs.ignore_if_contains = [];
  p.model.low_vram = true; p.model.quantize = true; p.train.gradient_checkpointing = false; p.train.steps = 678;
  p.datasets[0].folder_path = '/fixture/dataset'; p.sample.samples[0].prompt = 'fixture photo prompt';
  const source = path.join(root, 'extensions_built_in/diffusion_models/ui.tsx');
  const transpiled = await swc.transform(fs.readFileSync(source, 'utf8'), { filename: source, jsc: { parser: { syntax: 'typescript', tsx: true }, transform: { react: { runtime: 'automatic', development: false } }, target: 'es2020' }, module: { type: 'commonjs' }, sourceMaps: false, isModule: true });
  const browser = await chromium.launch({ headless: true, executablePath: '/usr/bin/google-chrome', args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1365, height: 1000 } }); page.setDefaultTimeout(60000);
  const errors = []; let payload;
  page.on('pageerror', e => errors.push(e.stack || e.message)); page.on('dialog', dialog => dialog.dismiss());
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
    await page.goto('http://0.0.0.0:8686/jobs/new?id=fixture', { waitUntil: 'domcontentloaded', timeout: 120000 });
    await page.getByLabel('Training recipe').waitFor();
    if (process.env.H3_PRESENTATION_ONLY === '1') {
      await page.getByText('Advanced target modules', { exact: true }).click();
      for (const text of ['Add module…', 'Add exclusion…']) {
        const placeholder = page.getByText(text, { exact: true });
        const control = placeholder.locator('xpath=ancestor::div[contains(@class,"aitk-react-select__control")]');
        const help = placeholder.locator('xpath=ancestor::div[contains(@class,"aitk-react-select-container")]/../following-sibling::div');
        const [pbox, cbox, hbox] = await Promise.all([placeholder.boundingBox(), control.boundingBox(), help.boundingBox()]);
        assert.ok(pbox && cbox && hbox, `${text} geometry is visible`);
        assert.ok(pbox.y >= cbox.y && pbox.y + pbox.height <= cbox.y + cbox.height, `${text} fits inside its control`);
        assert.ok(pbox.y + pbox.height <= hbox.y, `${text} does not overlap help`);
      }
      fs.mkdirSync(path.join(root, 'tmp/h3-advanced-options'), { recursive: true });
      await page.getByText('Advanced target modules', { exact: true }).scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(root, 'tmp/h3-advanced-options/browser-desktop-final.png') });
      p.network.network_kwargs.only_if_contains = Array.from({ length: 30 }, (_, index) => `blocks.${index + 20}.`);
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.getByText('Advanced target modules', { exact: true }).click();
      const include = page.getByLabel('Include modules');
      const container = include.locator('xpath=ancestor::div[contains(@class,"aitk-react-select-container")]');
      const control = container.locator('.aitk-react-select__control');
      const help = container.locator('xpath=../following-sibling::div');
      const chips = container.locator('.aitk-react-select__multi-value');
      assert.equal(await chips.count(), 30);
      const labels = container.locator('.aitk-react-select__multi-value__label');
      assert.deepEqual(await labels.allTextContents(), Array.from({ length: 30 }, (_, index) => `#${index + 20}`));
      assert.ok(await labels.evaluateAll(nodes => nodes.every(node => node.scrollWidth <= node.clientWidth)), 'all block labels are readable');
      const [cbox, hbox] = await Promise.all([control.boundingBox(), help.boundingBox()]);
      assert.ok(cbox && hbox);
      for (const chip of await chips.all()) {
        const box = await chip.boundingBox();
        assert.ok(box && box.y >= cbox.y && box.y + box.height <= cbox.y + cbox.height, 'chip fits inside control');
      }
      assert.ok(cbox.y + cbox.height <= hbox.y, 'help appears below all selected chips');
      await container.scrollIntoViewIfNeeded();
      await page.screenshot({ path: path.join(root, 'tmp/h3-advanced-options/browser-desktop-chips.png') });
      await page.setViewportSize({ width: 390, height: 844 });
      await container.scrollIntoViewIfNeeded();
      const [narrowControl, narrowHelp] = await Promise.all([control.boundingBox(), help.boundingBox()]);
      assert.ok(narrowControl && narrowHelp && narrowControl.y + narrowControl.height <= narrowHelp.y);
      for (const chip of await chips.all()) {
        const box = await chip.boundingBox();
        assert.ok(box && box.y >= narrowControl.y && box.y + box.height <= narrowControl.y + narrowControl.height, 'chip fits inside narrow control');
      }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
      await page.screenshot({ path: path.join(root, 'tmp/h3-advanced-options/browser-narrow-chips.png') });
      assert.deepEqual(errors, []); console.log('H3 advanced target desktop presentation passed');
      return;
    }
    let saved = await save();
    assert.deepEqual(saved.model.model_kwargs, p.model.model_kwargs);
    assert.equal(saved.model.te_name_or_path, '/fixture/te-active');
    assert.equal(saved.model.text_encoder_path, '/fixture/te-old');
    assert.equal(saved.model.vae_path, '/fixture/vae-active');
    assert.deepEqual(saved.network.network_kwargs.only_if_contains, []);
    assert.deepEqual(saved.network.network_kwargs.ignore_if_contains, ['fixture.unknown']);
    await page.getByText('Advanced model settings', { exact: true }).click();
    await page.getByText('Advanced target modules', { exact: true }).click();
    assert.equal(await page.getByLabel('Text encoder path (optional)').inputValue(), '/fixture/te-active');
    assert.equal(await page.getByLabel('VAE path (optional)').inputValue(), '/fixture/vae-active');
    assert.equal(await page.getByLabel('DiT path (optional)').inputValue(), '/fixture/dit');
    assert.equal(await page.getByLabel('Caption token limit').inputValue(), '');
    assert.ok(await page.getByText('fixture.unknown', { exact: true }).isVisible());
    saved = await save();
    assert.deepEqual(saved.network.network_kwargs.only_if_contains, []);
    assert.deepEqual(saved.model.model_kwargs, p.model.model_kwargs);
    await page.getByLabel('Text encoder path (optional)').fill('');
    await page.getByLabel('VAE path (optional)').fill('');
    await page.getByLabel('DiT path (optional)').fill('/fixture/new-dit');
    await page.getByLabel('Quantized cache directory (optional)').fill('/fixture/cache');
    await page.getByLabel('Caption token limit').fill('0');
    const include = page.getByLabel('Include modules');
    await include.fill('custom.block'); await include.press('Enter');
    assert.ok(await page.getByText('custom.block', { exact: true }).isVisible());
    const exclude = page.getByLabel('Exclude modules');
    await exclude.focus(); await exclude.press('Backspace');
    assert.equal(await page.getByText('fixture.unknown', { exact: true }).count(), 0);
    await exclude.fill('custom.skip'); await exclude.press('Enter');
    saved = await save();
    assert.equal(saved.model.te_name_or_path, undefined); assert.equal(saved.model.text_encoder_path, undefined);
    assert.equal(saved.model.model_kwargs.text_encoder_path, undefined);
    assert.equal(saved.model.vae_path, undefined); assert.equal(saved.model.model_kwargs.image_vae_path, undefined);
    assert.equal(saved.model.model_kwargs.dit_image_path, '/fixture/new-dit');
    assert.equal(saved.model.quantized_model_cache_dir, '/fixture/cache');
    assert.equal(saved.model.model_kwargs.max_text_length, 0);
    assert.deepEqual(saved.network.network_kwargs.only_if_contains, ['custom.block']);
    assert.deepEqual(saved.network.network_kwargs.ignore_if_contains, ['custom.skip']);
    await include.focus(); await include.press('Backspace');
    await page.getByLabel('Caption token limit').fill('');
    saved = await save();
    assert.equal(saved.network.network_kwargs.only_if_contains, undefined);
    assert.equal(saved.model.model_kwargs.max_text_length, undefined);
    // Legacy process-level filters win when nested filters are empty. Opening
    // the form preserves both; editing must remove the hidden aliases too.
    p.only_if_contains = ['legacy.include']; p.ignore_if_contains = ['legacy.exclude'];
    p.network.network_kwargs.only_if_contains = []; p.network.network_kwargs.ignore_if_contains = [];
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.getByText('Advanced target modules', { exact: true }).click();
    assert.ok(await page.getByText('legacy.include', { exact: true }).isVisible());
    assert.ok(await page.getByText('legacy.exclude', { exact: true }).isVisible());
    saved = await save();
    assert.deepEqual(saved.only_if_contains, ['legacy.include']); assert.deepEqual(saved.ignore_if_contains, ['legacy.exclude']);
    assert.deepEqual(saved.network.network_kwargs.only_if_contains, []); assert.deepEqual(saved.network.network_kwargs.ignore_if_contains, []);
    await page.getByLabel('Include modules').focus(); await page.getByLabel('Include modules').press('Backspace');
    await page.getByLabel('Exclude modules').focus(); await page.getByLabel('Exclude modules').press('Backspace');
    saved = await save();
    assert.equal(saved.only_if_contains, undefined); assert.equal(saved.ignore_if_contains, undefined);
    assert.equal(saved.network.network_kwargs.only_if_contains, undefined);
    await page.getByText('Advanced model settings', { exact: true }).click();
    fs.mkdirSync(path.join(root, 'tmp/h3-advanced-options'), { recursive: true });
    await page.getByText('Advanced model settings', { exact: true }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(root, 'tmp/h3-advanced-options/browser-desktop.png') });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByLabel('Caption token limit').scrollIntoViewIfNeeded();
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
    await page.screenshot({ path: path.join(root, 'tmp/h3-advanced-options/browser-narrow.png') });
    assert.deepEqual(errors, []); console.log('H3 advanced options browser scenario passed');
  } catch (error) { console.error('Page errors:', errors, 'HTML:', (await page.content()).slice(0, 5000)); throw error; }
  finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
