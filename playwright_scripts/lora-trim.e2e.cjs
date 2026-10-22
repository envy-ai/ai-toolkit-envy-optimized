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
const DORA_SOURCE = 'synthetic_dora.safetensors';
const DORA_PATH = `${JOB_FOLDER}/${DORA_SOURCE}`;
const DORA_BASE_PATH = `${JOB_FOLDER}/synthetic_base.safetensors`;
const DORA_OUTPUT_NAME = 'e2e_baked_dora.safetensors';
const DORA_OUTPUT_PATH = `${JOB_FOLDER}/${DORA_OUTPUT_NAME}`;
const LOHA_SOURCE = 'synthetic_loha.safetensors';
const LOHA_PATH = `${JOB_FOLDER}/${LOHA_SOURCE}`;
const UNSUPPORTED_SOURCE = 'unsupported_doha.safetensors';
const UNSUPPORTED_PATH = `${JOB_FOLDER}/${UNSUPPORTED_SOURCE}`;
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
  assert.ok(fs.existsSync(DORA_PATH), `Missing synthetic DoRA source: ${DORA_PATH}`);
  assert.ok(fs.existsSync(DORA_BASE_PATH), `Missing synthetic base: ${DORA_BASE_PATH}`);
  assert.ok(fs.existsSync(LOHA_PATH), `Missing synthetic LoHa source: ${LOHA_PATH}`);
  assert.ok(fs.existsSync(UNSUPPORTED_PATH), `Missing synthetic unsupported source: ${UNSUPPORTED_PATH}`);
  fs.rmSync(OUTPUT_PATH, { force: true });
  fs.rmSync(DORA_OUTPUT_PATH, { force: true });
  const sourceHash = sha256File(SOURCE_PATH);
  const doraHash = sha256File(DORA_PATH);
  const baseHash = sha256File(DORA_BASE_PATH);
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
    const openSource = page.getByRole('button', { name: `Reduce ${SOURCE_NAME}` });
    await openSource.waitFor({ state: 'visible' });
    await openSource.click();
    let dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
    await dialog.waitFor({ state: 'visible' });
    await dialog.locator('.aitk-react-select__control').waitFor({ state: 'visible' });

    for (const label of [
      'Rank policy',
      'Retained energy (%)',
      'Minimum rank',
      'Maximum rank (optional)',
      'Output filename',
    ]) {
      assert.equal(await dialog.getByLabel(label).count(), 1, `Expected associated accessible label ${label}`);
    }
    assert.match(await dialog.getByLabel('Rank policy').getAttribute('id'), /criterion/);

    // Tab navigation must remain in the open dialog, and Escape restores the
    // invoking checkpoint action's focus.
    for (let index = 0; index < 12; index += 1) {
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => Boolean(document.activeElement?.closest('[role="dialog"]'))), true);
    }
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), `Reduce ${SOURCE_NAME}`);

    await openSource.click();
    dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
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
    await page.getByRole('button', { name: `Reduce ${OUTPUT_NAME}` }).waitFor({ state: 'visible' });

    // A second attempt at the same new filename must fail visibly and leave
    // the first valid output intact.
    await openSource.click();
    dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
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
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click();

    // Bake a synthetic DoRA against its matching local base, then verify the
    // preview identifies the final effective update and writes a new copy.
    await page.getByRole('button', { name: `Reduce ${DORA_SOURCE}` }).click();
    dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
    await dialog.getByRole('switch', { name: 'Bake DoRA to ordinary LoRA before reducing' }).click();
    await dialog.getByLabel('Matching base safetensors checkpoint').fill(DORA_BASE_PATH);
    await dialog.getByLabel('DoRA convention').click();
    await page.getByText('AI Toolkit / native', { exact: true }).click();
    await dialog.getByLabel('Rank policy').click();
    await page.getByText('Fixed rank', { exact: true }).click();
    await dialog.getByLabel('Target rank').fill('64');
    const bakePreviewPromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Analyze' }).click();
    const bakePreviewResponse = await bakePreviewPromise;
    const bakePreviewEnvelope = await bakePreviewResponse.json();
    assert.equal(bakePreviewEnvelope.ok, true, JSON.stringify(bakePreviewEnvelope));
    assert.equal(bakePreviewEnvelope.result.layers[0].baked, true);
    assert.equal(bakePreviewEnvelope.result.layers[0].output_format, 'lora');
    assert.equal(bakePreviewEnvelope.result.layers[0].metric_scope, 'effective_update');
    await dialog.getByText('DoRA was baked using the matching base checkpoint', { exact: false }).waitFor({ state: 'visible' });
    await dialog.getByLabel('Output filename').fill(DORA_OUTPUT_NAME);
    const bakeWritePromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Create trimmed copy' }).click();
    const bakeWrite = await bakeWritePromise;
    const bakeWriteEnvelope = await bakeWrite.json();
    assert.equal(bakeWriteEnvelope.ok, true, JSON.stringify(bakeWriteEnvelope));
    assert.ok(fs.existsSync(DORA_OUTPUT_PATH));
    assert.equal(sha256File(DORA_PATH), doraHash);
    assert.equal(sha256File(DORA_BASE_PATH), baseHash);
    await dialog.getByRole('button', { name: 'Close', exact: true }).click();

    // LoHa conversion is opt-in and its preview labels the effective update.
    await page.getByRole('button', { name: `Reduce ${LOHA_SOURCE}` }).click();
    dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
    await dialog.getByRole('switch', { name: 'Convert LoHa / LoKr to ordinary LoRA' }).click();
    await dialog.getByLabel('Rank policy').click();
    await page.getByText('Fixed rank', { exact: true }).click();
    await dialog.getByLabel('Target rank').fill('4');
    const lohaPreviewPromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Analyze' }).click();
    const lohaPreview = await lohaPreviewPromise;
    const lohaEnvelope = await lohaPreview.json();
    assert.equal(lohaEnvelope.ok, true, JSON.stringify(lohaEnvelope));
    assert.equal(lohaEnvelope.result.layers[0].input_format, 'loha');
    assert.equal(lohaEnvelope.result.layers[0].output_format, 'lora');
    assert.equal(lohaEnvelope.result.layers[0].metric_scope, 'effective_update');
    await dialog.getByRole('button', { name: 'Cancel' }).click();

    // Unsupported DoHa remains a readable validation failure with no output.
    await page.getByRole('button', { name: `Reduce ${UNSUPPORTED_SOURCE}` }).click();
    dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
    const invalidResponsePromise = waitScriptCall(page);
    await dialog.getByRole('button', { name: 'Analyze' }).click();
    const invalidResponse = await invalidResponsePromise;
    const invalidEnvelope = await invalidResponse.json();
    assert.equal(invalidEnvelope.ok, false);
    await dialog.getByRole('alert').waitFor({ state: 'visible' });
    assert.match(await dialog.getByRole('alert').innerText(), /unsupported|CP\/Tucker/i);
    assert.equal(fs.existsSync(`${JOB_FOLDER}/unsupported_doha_trimmed.safetensors`), false);
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
          'DoRA bake requires matching base and writes an ordinary LoRA copy',
          'LoHa opt-in conversion preview reports effective-update scope',
          'unsupported DoHa failure is visible and writes nothing',
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
