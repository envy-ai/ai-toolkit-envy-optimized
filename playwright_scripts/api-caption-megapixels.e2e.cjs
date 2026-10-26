// Offline browser regression for API caption image_megapixels default/null/value
// persistence. The target must be an isolated no-queue-worker server with the
// synthetic DB produced by prepare_api_caption_regex_e2e.py.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const playwright = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const baseUrl = (process.env.CAPTION_E2E_URL || 'http://127.0.0.1:8676').replace(/\/$/, '');
const outputDir = path.resolve(process.env.CAPTION_E2E_OUTPUT_DIR || 'tmp/caption-megapixels');
const datasets = [
  process.env.CAPTION_E2E_DATASET_PATH || '/datasets/regex_fixture',
  process.env.CAPTION_E2E_SECOND_DATASET_PATH || '/datasets/regex_fixture_second',
  process.env.CAPTION_E2E_THIRD_DATASET_PATH || '/datasets/regex_fixture_third',
];
const megapixelsLabel = process.env.CAPTION_E2E_MEGAPIXELS_LABEL || 'Normalize image (megapixels)';

async function openSetup(page, datasetPath) {
  const response = await page.goto(`${baseUrl}${datasetPath}`, { waitUntil: 'domcontentloaded' });
  assert.equal(response && response.status(), 200, `failed to load ${datasetPath}`);
  await page.getByRole('button', { name: /Auto Caption/ }).click();
  const dialog = page.getByRole('dialog', { name: 'Caption Dataset' });
  await dialog.waitFor({ state: 'visible', timeout: 10000 });
  return dialog;
}

async function submitAndWait(dialog) {
  const observedResponses = [];
  const page = dialog.page();
  const responseListener = response => {
    if (/\/api\/(?:jobs|queue|caption\/settings)/.test(response.url())) {
      observedResponses.push({ url: new URL(response.url()).pathname, status: response.status() });
    }
  };
  page.on('response', responseListener);
  const jobResponsePromise = dialog.page().waitForResponse(response =>
    response.request().method() === 'POST' && /\/api\/jobs(?:\?|$)/.test(response.url()), { timeout: 15000 });
  const settingsResponsePromise = dialog.page().waitForResponse(response =>
    response.request().method() === 'POST' && /\/api\/caption\/settings(?:\?|$)/.test(response.url()), { timeout: 15000 });
  await dialog.getByRole('button', { name: 'Add to Queue' }).click();
  let jobResponse;
  let settingsResponse;
  try {
    [jobResponse, settingsResponse] = await Promise.all([jobResponsePromise, settingsResponsePromise]);
  } catch (error) {
    const alert = dialog.getByRole('alert');
    const message = await alert.isVisible().catch(() => false) ? await alert.innerText() : '';
    throw new Error(`${error.message}; alert=${message}; observedResponses=${JSON.stringify(observedResponses)}`);
  } finally {
    page.off('response', responseListener);
  }
  assert.equal(jobResponse.status(), 200, `job API returned ${jobResponse.status()}`);
  assert.equal(settingsResponse.status(), 200, `settings API returned ${settingsResponse.status()}`);
  await dialog.waitFor({ state: 'hidden', timeout: 10000 });
  return jobResponse;
}

async function getPreferences(page) {
  const response = await page.request.get(`${baseUrl}/api/caption/settings`);
  assert.equal(response.status(), 200);
  return (await response.json()).preferences;
}

async function main() {
  const browser = await playwright.chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH,
    args: ['--disable-gpu'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1365, height: 1000 } });
    const pageErrors = [];
    const jobs = [];
    let providerRequests = 0;
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('request', request => {
      if (request.url().startsWith('http://127.0.0.1:9/')) providerRequests += 1;
      if (request.method() === 'POST' && /\/api\/jobs(?:\?|$)/.test(request.url())) jobs.push(request.postDataJSON());
    });

    const firstDialog = await openSetup(page, datasets[0]);
    const firstInput = firstDialog.getByLabel(megapixelsLabel);
    const firstModel = firstDialog.getByLabel('Model (optional)');
    assert.equal(await firstInput.inputValue(), '1', 'missing/default preferences should show the 1 MP default');
    await firstModel.fill('');
    assert.equal(await firstModel.inputValue(), '', 'the API model may be left blank');
    fs.mkdirSync(outputDir, { recursive: true });
    await page.screenshot({ path: path.join(outputDir, 'image-megapixels-default.png'), fullPage: true });
    await firstInput.fill('');
    assert.equal(await firstInput.inputValue(), '', 'blank input represents disabled resizing');
    await submitAndWait(firstDialog);

    assert.equal(jobs.length, 1);
    assert.equal(jobs[0].job_config.config.process[0].caption.image_megapixels, null);
    assert.equal(jobs[0].job_config.config.process[0].caption.model_name_or_path, '');
    const firstPreferences = await getPreferences(page);
    assert.equal(firstPreferences.caption.image_megapixels, null);
    assert.equal(firstPreferences.caption.model_name_or_path, '');

    const secondDialog = await openSetup(page, datasets[1]);
    const secondInput = secondDialog.getByLabel(megapixelsLabel);
    const secondModel = secondDialog.getByLabel('Model (optional)');
    assert.equal(await secondInput.inputValue(), '', 'explicit null should restore as a blank field');
    assert.equal(await secondModel.inputValue(), '', 'a saved blank model should restore blank');
    await secondModel.fill(' \t ');
    await secondInput.fill('0.25');
    assert.equal(await secondInput.inputValue(), '0.25');
    await page.screenshot({ path: path.join(outputDir, 'image-megapixels-quarter.png'), fullPage: true });
    await submitAndWait(secondDialog);

    assert.equal(jobs.length, 2);
    assert.equal(jobs[1].job_config.config.process[0].caption.image_megapixels, 0.25);
    assert.equal(jobs[1].job_config.config.process[0].caption.model_name_or_path, '');
    const secondPreferences = await getPreferences(page);
    assert.equal(secondPreferences.caption.image_megapixels, 0.25);
    assert.equal(secondPreferences.caption.model_name_or_path, '', 'whitespace-only model input normalizes to blank');

    const thirdDialog = await openSetup(page, datasets[2]);
    assert.equal(await thirdDialog.getByLabel(megapixelsLabel).inputValue(), '0.25');
    assert.equal(await thirdDialog.getByLabel('Model (optional)').inputValue(), '', 'normalized blank model should restore blank');
    await thirdDialog.getByRole('button', { name: 'Cancel' }).click();

    assert.equal(providerRequests, 0, 'browser test must not contact the configured fake provider endpoint');
    assert.deepEqual(pageErrors, [], `browser runtime errors: ${pageErrors.join('; ')}`);
    process.stdout.write(`${JSON.stringify({
      result: 'passed',
      defaultMegapixels: 1,
      blankSavedAndRestoredAs: null,
      blankModelSavedRestoredAndNormalized: true,
      fractionalSavedAndRestored: 0.25,
      submittedJobs: jobs.length,
      providerRequests,
      screenshots: [
        path.join(outputDir, 'image-megapixels-default.png'),
        path.join(outputDir, 'image-megapixels-quarter.png'),
      ],
    })}\n`);
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
