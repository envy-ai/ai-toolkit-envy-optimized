const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');
const path = require('node:path');

const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const ROOT = path.resolve(__dirname, '..');
const BASE_URL = process.env.LORA_TRIM_BASE_URL || 'http://127.0.0.1:8676';
const JOB_ID = 'trim-e2e';
const JOB_FOLDER = `${ROOT}/tmp/lora-trim/e2e-output/trim_fixture`;
const SOURCE_NAME = 'synthetic_lora.safetensors';
const SOURCE_PATH = `${JOB_FOLDER}/${SOURCE_NAME}`;
const OUTPUT_NAME = 'e2e_trimmed.safetensors';
const OUTPUT_PATH = `${JOB_FOLDER}/${OUTPUT_NAME}`;
const SCREENSHOT_FOLDER = path.join(ROOT, 'tmp', 'lora-trim', 'screenshots');

const sha256File = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const waitScriptCall = page =>
  page.waitForResponse(response => {
    if (new URL(response.url()).pathname !== '/api/scripts' || response.request().method() !== 'POST') return false;
    try {
      return response.request().postDataJSON()?.script === 'trim_lora.py';
    } catch {
      return false;
    }
  });

async function main() {
  assert.ok(fs.existsSync(SOURCE_PATH), `Missing synthetic source: ${SOURCE_PATH}`);
  fs.rmSync(OUTPUT_PATH, { force: true });
  const sourceHash = sha256File(SOURCE_PATH);
  fs.mkdirSync(SCREENSHOT_FOLDER, { recursive: true });
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined,
    args: ['--disable-gpu'],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));

  try {
    // Keep hardware telemetry out of the smoke; the trimming workflow itself
    // is CPU-only and uses only the synthetic safetensors file.
    await page.route('**/api/monitor', route => route.abort());
    await page.goto(`${BASE_URL}/jobs/${JOB_ID}`, { waitUntil: 'domcontentloaded' });
    const openSource = page.getByRole('button', { name: `Trim ${SOURCE_NAME}` });
    await openSource.waitFor({ state: 'visible' });
    await openSource.click();
    let dialog = page.getByRole('dialog', { name: 'Trim LoRA' });
    await dialog.waitFor({ state: 'visible' });
    await dialog.locator('.aitk-react-select__control').waitFor({ state: 'visible' });

    for (const label of [
      'Criterion',
      'Retained energy (%)',
      'Minimum rank',
      'Maximum rank (optional)',
      'Output filename',
    ]) {
      assert.equal(await dialog.getByLabel(label).count(), 1, `Expected associated accessible label ${label}`);
    }
    assert.match(await dialog.getByLabel('Criterion').getAttribute('id'), /criterion/);

    // Tab navigation must remain in the open dialog, and Escape restores the
    // invoking checkpoint action's focus.
    for (let index = 0; index < 12; index += 1) {
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => Boolean(document.activeElement?.closest('[role="dialog"]'))), true);
    }
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), `Trim ${SOURCE_NAME}`);

    await openSource.click();
    dialog = page.getByRole('dialog', { name: 'Trim LoRA' });
    const previewResponsePromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Analyze' }).click();
    const previewResponse = await previewResponsePromise;
    const previewEnvelope = await previewResponse.json();
    assert.equal(previewEnvelope.ok, true, JSON.stringify(previewEnvelope));
    const preview = previewEnvelope.result;
    assert.equal(preview.written, false);
    assert.equal(preview.layers.length, 1);
    assert.equal(preview.layers[0].old_rank, 8);
    assert.equal(preview.layers[0].new_rank, 3);
    assert.ok(preview.layers[0].relative_error < 0.003);
    assert.ok(preview.estimated_output_bytes < preview.input_bytes);
    await dialog.getByRole('region', { name: 'Analysis preview' }).waitFor({ state: 'visible' });
    await dialog.getByText('8 → 3', { exact: true }).waitFor({ state: 'visible' });
    await page.screenshot({ path: path.join(SCREENSHOT_FOLDER, 'preview-desktop.png'), fullPage: true });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(SCREENSHOT_FOLDER, 'preview-390px.png'), fullPage: true });
    await page.setViewportSize({ width: 1280, height: 900 });

    // Changing a setting invalidates the confirmed preview and disables write.
    const threshold = dialog.getByLabel('Retained energy (%)');
    await threshold.fill('99');
    await dialog.getByRole('region', { name: 'Analysis preview' }).waitFor({ state: 'detached' });
    assert.equal(await dialog.getByRole('button', { name: 'Create trimmed copy' }).isDisabled(), true);
    const secondPreviewPromise = waitScriptCall(page);
    await threshold.press('Enter');
    await dialog.getByRole('status').getByText(/Analyzing on CPU/).waitFor({ state: 'visible' });
    assert.equal(await dialog.getByRole('button', { name: 'Analyzing…', exact: true }).isDisabled(), true);
    assert.equal(await dialog.getByRole('button', { name: 'Cancel' }).isDisabled(), true);
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'visible' });
    const secondPreviewResponse = await secondPreviewPromise;
    const secondPreviewEnvelope = await secondPreviewResponse.json();
    assert.equal(secondPreviewEnvelope.ok, true, JSON.stringify(secondPreviewEnvelope));

    await dialog.getByLabel('Output filename').fill(OUTPUT_NAME);
    const writePromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Create trimmed copy' }).click();
    const writeResponse = await writePromise;
    const writeEnvelope = await writeResponse.json();
    assert.equal(writeEnvelope.ok, true, JSON.stringify(writeEnvelope));
    assert.equal(writeEnvelope.result.written, true);
    assert.equal(writeEnvelope.result.output_path, OUTPUT_PATH);
    await dialog.getByRole('status').getByText(`Created ${OUTPUT_NAME}.`).waitFor({ state: 'visible' });
    assert.ok(fs.existsSync(OUTPUT_PATH), 'Expected the new synthetic output to exist');
    assert.equal(sha256File(SOURCE_PATH), sourceHash, 'The source LoRA must remain byte-identical');
    assert.ok(fs.statSync(OUTPUT_PATH).size < fs.statSync(SOURCE_PATH).size);

    await dialog.getByRole('button', { name: 'Close', exact: true }).click();
    await dialog.waitFor({ state: 'detached' });
    await page.getByRole('button', { name: `Trim ${OUTPUT_NAME}` }).waitFor({ state: 'visible' });

    // A second attempt at the same new filename must fail visibly and leave
    // the first valid output intact.
    await openSource.click();
    dialog = page.getByRole('dialog', { name: 'Trim LoRA' });
    await dialog.getByLabel('Output filename').fill(OUTPUT_NAME);
    const duplicatePreviewPromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Analyze' }).click();
    const duplicatePreview = await duplicatePreviewPromise;
    assert.equal((await duplicatePreview.json()).ok, true);
    const duplicateWritePromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Create trimmed copy' }).click();
    const duplicateWrite = await duplicateWritePromise;
    const duplicateEnvelope = await duplicateWrite.json();
    assert.equal(duplicateEnvelope.ok, false);
    await dialog.getByRole('alert').waitFor({ state: 'visible' });
    assert.match(await dialog.getByRole('alert').innerText(), /FileExistsError|already exists/i);
    await threshold.fill('98');
    await dialog.getByRole('region', { name: 'Analysis preview' }).waitFor({ state: 'detached' });
    assert.equal(await dialog.getByRole('button', { name: 'Create trimmed copy' }).isDisabled(), true);
    await dialog.getByRole('button', { name: 'Cancel' }).click();

    // The rejected DoRA fixture produces a readable error without publishing
    // any output. This path also traverses the real /api/scripts Python bridge.
    await page.getByRole('button', { name: 'Trim invalid_dora.safetensors' }).click();
    dialog = page.getByRole('dialog', { name: 'Trim LoRA' });
    const invalidResponsePromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Analyze' }).click();
    const invalidResponse = await invalidResponsePromise;
    const invalidEnvelope = await invalidResponse.json();
    assert.equal(invalidEnvelope.ok, false);
    await dialog.getByRole('alert').waitFor({ state: 'visible' });
    assert.match(await dialog.getByRole('alert').innerText(), /DoRA.*unsupported|unsupported.*DoRA/i);
    assert.equal(fs.existsSync(`${JOB_FOLDER}/invalid_dora_trimmed.safetensors`), false);
    await dialog.getByRole('button', { name: 'Cancel' }).click();

    assert.deepEqual(errors, [], `Unexpected browser errors: ${errors.join('; ')}`);
    console.log(
      JSON.stringify({
        checks: [
          'associated accessible labels',
          'keyboard focus trap and restoration',
          'preview invalidation on setting change',
          'CPU synthetic preview and rank/error report',
          'no-clobber successful create through /api/scripts',
          'source remains byte-identical and output is smaller',
          'duplicate output failure is visible',
          'unsupported DoRA failure is visible and writes nothing',
        ],
        preview: {
          inputBytes: preview.input_bytes,
          estimatedOutputBytes: preview.estimated_output_bytes,
          options: preview.options,
          layer: {
            oldRank: preview.layers[0].old_rank,
            newRank: preview.layers[0].new_rank,
            relativeError: preview.layers[0].relative_error,
            retainedEnergy: preview.layers[0].retained_energy,
          },
        },
        written: {
          outputBytes: writeEnvelope.result.estimated_output_bytes,
          options: writeEnvelope.result.options,
          layer: {
            oldRank: writeEnvelope.result.layers[0].old_rank,
            newRank: writeEnvelope.result.layers[0].new_rank,
            relativeError: writeEnvelope.result.layers[0].relative_error,
            retainedEnergy: writeEnvelope.result.layers[0].retained_energy,
          },
        },
      }),
    );
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error?.stack || error);
  process.exitCode = 1;
});
