const assert = require('node:assert/strict');
const path = require('node:path');

const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const ROOT = path.resolve(__dirname, '..');
const BASE_URL = process.env.LORA_TRIM_BASE_URL || 'http://127.0.0.1:8676';
const JOB_ID = 'trim-e2e';
const JOB_FOLDER = `${ROOT}/tmp/lora-trim/e2e-output/trim_fixture`;
const SOURCE_NAME = 'synthetic_dora.safetensors';
const JOB_BASE = `${JOB_FOLDER}/synthetic_base.safetensors`;
const OVERRIDE_BASE = `${JOB_FOLDER}/synthetic_base_override.safetensors`;

async function main() {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined,
    args: ['--disable-gpu'],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));

  try {
    await page.route('**/api/monitor', route => route.abort());
    await page.goto(`${BASE_URL}/jobs/${JOB_ID}`, { waitUntil: 'domcontentloaded' });
    await page.setViewportSize({ width: 390, height: 844 });

    const openDora = page.getByRole('button', { name: `Reduce ${SOURCE_NAME}` });
    await openDora.waitFor({ state: 'visible' });
    await openDora.click();
    let dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
    let bake = dialog.getByRole('switch', { name: 'Bake DoRA to ordinary LoRA before reducing' });
    assert.equal(await bake.getAttribute('aria-checked'), 'false', 'Opening the dialog must not opt into baking');
    assert.equal(await dialog.getByLabel('DoRA convention').count(), 0, 'Convention stays unselected and hidden until baking is enabled');

    await bake.click();
    const baseInput = dialog.getByLabel('Matching base safetensors checkpoint');
    assert.equal(await baseInput.inputValue(), JOB_BASE, 'The base path should be inherited from this training job');
    assert.match(await dialog.innerText(), /NVFP4, W4A8 and ConvRot W4A4 are supported/);
    assert.match(await dialog.innerText(), /Exported DoHa uses ComfyUI; native DoHa magnitude uses AI Toolkit/);
    await page.screenshot({ path: path.join(ROOT, 'tmp/lora-trim/quantized-bake-mobile.png'), fullPage: true });
    const precision = dialog.getByLabel('Quantized base precision');
    const selectedPrecision = async () =>
      (await dialog.locator('.aitk-react-select__single-value').allInnerTexts()).at(-1);
    assert.equal(await selectedPrecision(), 'BF16', 'Job train.dtype=bf16 should prefill quantized-base precision');

    await baseInput.fill(OVERRIDE_BASE);
    await precision.click();
    await page.getByText('FP16', { exact: true }).click();
    await bake.click();
    assert.equal(await dialog.getByLabel('Matching base safetensors checkpoint').count(), 0, 'Turning baking off hides its fields');
    await bake.click();
    assert.equal(
      await dialog.getByLabel('Matching base safetensors checkpoint').inputValue(),
      OVERRIDE_BASE,
      'Toggling baking must preserve the user-edited path within the open dialog',
    );
    assert.equal(await selectedPrecision(), 'FP16', 'Toggling baking must preserve precision edits within the dialog');
    assert.equal(await dialog.getByLabel('DoRA convention').inputValue(), '', 'Opening bake fields must not select a convention');

    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();
    await dialog.waitFor({ state: 'detached' });
    await openDora.click();
    dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
    bake = dialog.getByRole('switch', { name: 'Bake DoRA to ordinary LoRA before reducing' });
    assert.equal(await bake.getAttribute('aria-checked'), 'false');
    await bake.click();
    assert.equal(
      await dialog.getByLabel('Matching base safetensors checkpoint').inputValue(),
      JOB_BASE,
      'A newly opened dialog should initialize from the current training job again',
    );
    assert.equal(await selectedPrecision(), 'BF16', 'A new dialog should reinitialize precision from the training job');
    assert.equal(await dialog.getByLabel('DoRA convention').inputValue(), '', 'The convention remains an explicit user choice');
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();

    assert.deepEqual(errors, [], `Unexpected browser errors: ${errors.join('; ')}`);
    console.log(JSON.stringify({
      checks: [
        'job base path prefilled only after explicit DoRA bake opt-in',
        'training dtype prefills precision and user precision edits survive toggles and reopen',
        'bake convention remains unselected',
        'manual path override survives off/on toggles within a dialog',
        'close and reopen initializes from current job path',
        'quantized-base format support and DoHa convention guidance are visible in the narrow layout',
      ],
      trainingBasePrefill: JOB_BASE,
      userOverride: OVERRIDE_BASE,
    }));
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error?.stack || error);
  process.exitCode = 1;
});
