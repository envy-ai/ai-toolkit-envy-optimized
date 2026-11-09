const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const ROOT = path.resolve(__dirname, '..');
const BASE = process.env.TRAINING_MAPS_BASE_URL || 'http://0.0.0.0:8676';
const OUTPUT = path.join(ROOT, 'tmp', 'fizgig-port');
const id = 'training-maps-fixture';
const levels = [0.25, 0.5, 0.75];
const probes = ['a', 'b'].map((key, i) => ({ key, image_path: `/fixture/${key}.svg`, image_url: `/fixture/${key}.svg`, caption: `Fixture ${i + 1}`, width: 640, height: 400 }));
const scales = Object.fromEntries(probes.map(p => [p.key, Object.fromEntries(levels.map(l => [String(l), { improvement: 1, error: 1, effect: 1 }]))]));
const grid = { w: 2, h: 2, v: [0, 0.25, 0.6, 1] };
function frame(step) {
  return { step, maps: Object.fromEntries(probes.map(p => [p.key, Object.fromEntries(levels.map(l => [String(l), { error: grid, effect: grid, ...(step === 10 ? {} : { improvement: grid }) }]))])) };
}
async function main() {
  fs.mkdirSync(OUTPUT, { recursive: true });
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined, args: ['--disable-gpu'] });
  const page = await browser.newPage({ viewport: { width: 1365, height: 1000 } });
  page.setDefaultTimeout(60000);
  let steps = [10, 20];
  let mode = 'ready';
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/jobs' && url.searchParams.get('id') === id) return route.fulfill({ json: { id, name: 'Training maps fixture', job_type: 'train', status: 'stopped', gpu_ids: '', job_config: JSON.stringify({ config: { process: [{}] } }) } });
    if (url.pathname === `/api/jobs/${id}/training-maps`) {
      if (mode === 'empty') return route.fulfill({ json: { available: false } });
      if (mode === 'error') return route.fulfill({ status: 500, json: { error: 'Fixture map failure', stack: 'FixtureStack: diagnostic detail' } });
      const step = Number(url.searchParams.get('step') || steps.at(-1));
      return route.fulfill({ json: { available: true, manifest: { version: 1, status: 'ready', probes, levels, steps, scales }, frame: frame(step) } });
    }
    if (url.pathname.endsWith('/plugin')) return route.fulfill({ json: { exists: false } });
    return route.abort();
  });
  await page.route('**/fixture/*.svg', route => route.fulfill({ contentType: 'image/svg+xml', body: '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400"><rect width="640" height="400" fill="#818b91"/><circle cx="320" cy="200" r="120" fill="#c6b090"/></svg>' }));
  try {
    await page.goto(`${BASE}/jobs/${id}`, { waitUntil: 'domcontentloaded', timeout: 90000 });
    await page.getByRole('button', { name: 'Training Maps' }).click();
    const viewer = page.getByRole('region', { name: 'Training map viewer' });
    await viewer.waitFor();
    await viewer.getByText('Step 20 / 20').waitFor();
    for (const [option, description] of [['Not learned yet', 'Prediction error against'], ['Where the adapter acts', 'Absolute prediction change'], ['Learned since the start', 'Green: lower prediction']]) {
      await page.getByLabel('Map', { exact: true }).click();
      await page.getByRole('option', { name: option }).click();
      await viewer.getByText(description, { exact: false }).waitFor();
    }
    for (const option of ['Low (detail)', 'High (layout)', 'Mid']) {
      await page.getByLabel('Noise level').click();
      await page.getByRole('option', { name: option }).click();
      assert.equal(await viewer.locator('canvas').count(), 1);
    }
    await page.getByRole('button', { name: 'Picture 2' }).click();
    assert.equal(await page.getByRole('button', { name: 'Picture 2' }).getAttribute('aria-pressed'), 'true');
    await viewer.getByRole('button', { name: 'Previous' }).click();
    await viewer.getByText('Step 10 / 20').waitFor();
    await viewer.getByText('This first measurement is the baseline.', { exact: false }).waitFor();
    await viewer.getByRole('button', { name: 'Next' }).click();
    await viewer.getByText('Step 20 / 20').waitFor();
    await viewer.getByLabel('Recorded training step').fill('0');
    await viewer.getByText('Step 10 / 20').waitFor();
    await viewer.focus();
    await page.keyboard.press('ArrowRight');
    await viewer.getByText('Step 20 / 20').waitFor();
    await viewer.getByLabel(/Map opacity/).fill('50');
    assert.equal(await viewer.locator('canvas').evaluate(el => el.style.opacity), '0.5');
    const alignment = await viewer.evaluate(el => { const img = el.querySelector('img[alt^="Fixed training crop"]'); const canvas = el.querySelector('canvas'); const a = img.getBoundingClientRect(), b = canvas.getBoundingClientRect(); return [a.x-b.x, a.y-b.y, a.width-b.width, a.height-b.height]; });
    assert.ok(alignment.every(n => Math.abs(n) < 1), `Overlay must align with image: ${alignment}`);
    await page.screenshot({ path: path.join(OUTPUT, 'training-maps-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(OUTPUT, 'training-maps-narrow.png'), fullPage: true });
    steps = [10, 20, 30];
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await viewer.getByText('Step 30 / 30').waitFor();
    mode = 'error';
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    const alert = page.getByRole('alert').filter({ hasText: 'Fixture map failure' });
    await alert.getByText('Fixture map failure').waitFor();
    await alert.getByText('Error details and stack trace').click();
    await alert.getByText('FixtureStack: diagnostic detail').waitFor();
    assert.equal(await alert.isVisible(), true);
    mode = 'empty';
    await page.getByRole('button', { name: 'Refresh', exact: true }).click();
    await page.getByText('No training maps recorded.', { exact: false }).waitFor();
    assert.deepEqual(errors, []);
    console.log('Training maps browser scenario passed');
  } catch (error) {
    console.error('Page:', await page.locator('body').innerText(), 'Page errors:', errors);
    await page.screenshot({ path: path.join(OUTPUT, 'training-maps-failure.png') });
    throw error;
  } finally { await browser.close(); }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
