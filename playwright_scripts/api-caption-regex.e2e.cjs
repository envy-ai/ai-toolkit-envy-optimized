// Offline browser regression for the saved API-caption cleanup settings.
// The target is an isolated, no-queue-worker UI server with a synthetic dataset.
// No caption job is sent to an external provider by this test.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const playwright = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const baseUrl = (process.env.CAPTION_E2E_URL || 'http://127.0.0.1:8676').replace(/\/$/, '');
const datasetPath = process.env.CAPTION_E2E_DATASET_PATH || '/datasets/regex_fixture';
const screenshotPath = path.resolve(
  process.env.CAPTION_E2E_SCREENSHOT || 'tmp/caption-cleanup/modal.png',
);

async function setSwitch(scope, name, checked) {
  const control = scope.getByRole('switch', { name });
  const current = (await control.getAttribute('aria-checked')) === 'true';
  if (current !== checked) await control.click();
}

async function main() {
  const browser = await playwright.chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH,
    args: ['--disable-gpu'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1365, height: 1000 } });
    const errors = [];
    const postedJobConfigs = [];
    const apiResponses = [];
    let providerRequests = 0;
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      if (request.url().startsWith('http://127.0.0.1:9/')) providerRequests += 1;
      if (request.method() === 'POST' && /\/api\/jobs(?:\?|$)/.test(request.url())) {
        postedJobConfigs.push(request.postDataJSON());
      }
    });
    page.on('response', response => {
      if (/\/api\/(?:jobs|queue|caption\/settings)/.test(response.url())) {
        apiResponses.push({ url: new URL(response.url()).pathname, status: response.status() });
      }
    });

    const response = await page.goto(`${baseUrl}${datasetPath}`, { waitUntil: 'domcontentloaded' });
    assert.equal(response && response.status(), 200, 'synthetic dataset page should load');
    const openButton = page.getByRole('button', { name: /Auto Caption/ });
    await openButton.waitFor({ state: 'visible', timeout: 15000 });
    await openButton.click();

    const dialog = page.getByRole('dialog', { name: 'Caption Dataset' });
    await dialog.waitFor({ state: 'visible', timeout: 10000 });
    await dialog.getByText('Caption cleanup · optional, off by default').click();

    await setSwitch(dialog, 'Enable caption cleanup', true);
    const pattern = String.raw`.*<caption>(.*)<\/caption>.*`;
    await dialog.getByLabel('From (regular expression)').fill(pattern);
    await dialog.getByLabel('To (replacement)').fill('$1');
    await setSwitch(dialog, 'Ignore case (i)', true);
    await setSwitch(dialog, 'Multiline: ^/$ per line (m)', true);
    await setSwitch(dialog, 'Dot matches newlines (s)', true);
    await setSwitch(dialog, 'Ignore pattern whitespace/comments (x)', true);
    await setSwitch(dialog, 'Replace all matches', false);

    fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
    await page.screenshot({ path: screenshotPath, fullPage: true });
    const jobResponsePromise = page.waitForResponse(response =>
      response.request().method() === 'POST' && /\/api\/jobs(?:\?|$)/.test(response.url()), { timeout: 15000 });
    const settingsResponsePromise = page.waitForResponse(response =>
      response.request().method() === 'POST' && /\/api\/caption\/settings(?:\?|$)/.test(response.url()), { timeout: 15000 });
    await dialog.getByRole('button', { name: 'Add to Queue' }).click();
    const [jobResponse, settingsPostResponse] = await Promise.all([jobResponsePromise, settingsResponsePromise]);
    assert.equal(jobResponse.status(), 200, `job API returned ${jobResponse.status()}`);
    assert.equal(settingsPostResponse.status(), 200, `settings API returned ${settingsPostResponse.status()}`);
    await page.waitForTimeout(250);
    const alert = dialog.getByRole('alert');
    if (await alert.isVisible().catch(() => false)) {
      throw new Error(`caption modal reported: ${(await alert.innerText()).slice(0, 300)}`);
    }
    try {
      await dialog.waitFor({ state: 'hidden', timeout: 5000 });
    } catch {
      throw new Error(`modal did not close after successful API writes; API responses: ${JSON.stringify(apiResponses)}`);
    }

    assert.equal(postedJobConfigs.length, 1, 'the modal should submit one job config');
    const posted = postedJobConfigs[0];
    const jobConfig = posted.job_config;
    assert.equal(jobConfig.config.process[0].type, 'APICaptioner');
    assert.deepEqual(jobConfig.config.process[0].caption.regex_replace, {
      enabled: true,
      pattern,
      replacement: '$1',
      ignore_case: true,
      multiline: true,
      dotall: true,
      verbose: true,
      global: false,
    });
    assert.doesNotMatch(JSON.stringify(jobConfig), /api_key|fixture-private-key/i);

    const settingsResponse = await page.request.get(`${baseUrl}/api/caption/settings`);
    assert.equal(settingsResponse.status(), 200);
    const savedSettings = await settingsResponse.json();
    assert.deepEqual(savedSettings.preferences.caption.regex_replace, jobConfig.config.process[0].caption.regex_replace);

    const secondDatasetPath = process.env.CAPTION_E2E_SECOND_DATASET_PATH || '/datasets/regex_fixture_second';
    const secondPageResponse = await page.goto(`${baseUrl}${secondDatasetPath}`, { waitUntil: 'domcontentloaded' });
    assert.equal(secondPageResponse && secondPageResponse.status(), 200, 'second synthetic dataset should load');
    await page.getByRole('button', { name: /Auto Caption/ }).click();
    const restoredDialog = page.getByRole('dialog', { name: 'Caption Dataset' });
    await restoredDialog.waitFor({ state: 'visible', timeout: 10000 });
    await restoredDialog.getByText('Caption cleanup · optional, off by default').click();
    assert.equal(await restoredDialog.getByLabel('From (regular expression)').inputValue(), pattern);
    assert.equal(await restoredDialog.getByLabel('To (replacement)').inputValue(), '$1');
    assert.equal(await restoredDialog.getByRole('switch', { name: 'Enable caption cleanup' }).getAttribute('aria-checked'), 'true');
    for (const label of [
      'Ignore case (i)',
      'Multiline: ^/$ per line (m)',
      'Dot matches newlines (s)',
      'Ignore pattern whitespace/comments (x)',
    ]) assert.equal(await restoredDialog.getByRole('switch', { name: label }).getAttribute('aria-checked'), 'true', label);
    assert.equal(await restoredDialog.getByRole('switch', { name: 'Replace all matches' }).getAttribute('aria-checked'), 'false');
    await restoredDialog.getByRole('button', { name: 'Cancel' }).click();
    assert.equal(providerRequests, 0, 'UI test must not contact the fake provider endpoint');
    assert.deepEqual(errors, [], `browser runtime errors: ${errors.join('; ')}`);

    process.stdout.write(`${JSON.stringify({
      result: 'passed',
      pageStatus: response.status(),
      submittedJobs: postedJobConfigs.length,
      regexFlags: ['ignore_case', 'multiline', 'dotall', 'verbose', 'global'],
      settingsRoundTrip: true,
      reopenedOnSecondDataset: true,
      apiResponses,
      providerRequests,
      screenshot: screenshotPath,
    })}\n`);
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
