// Offline browser smoke for assistant image megapixel settings. Assistant
// settings are intercepted so this does not read or write server preferences.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const playwright = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const baseUrl = (process.env.ASSISTANT_E2E_URL || 'http://127.0.0.1:8675').replace(/\/$/, '');
const outputDir = path.resolve(process.env.ASSISTANT_E2E_OUTPUT_DIR || 'tmp/assistant-image-megapixels');

const defaultConfig = {
  baseUrl: 'https://fixture.invalid/v1',
  protocol: 'responses',
  model: 'fixture-model',
  authMode: 'none',
  strictTools: true,
  stream: true,
  vision: false,
  imageMegapixels: 1,
  timeoutMs: 120000,
  maxOutputTokens: 4096,
  maxRunTokens: 64000,
  maxToolRounds: 12,
  finalResponseReserve: 1024,
  maxToolCallsPerRound: 20,
  maxBatchSize: 100,
  maxJobs: 4,
};

async function main() {
  const browser = await playwright.chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH,
    args: ['--disable-gpu'],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1365, height: 1000 } });
    const pageErrors = [];
    const savedPayloads = [];
    let savedConfig = { ...defaultConfig };
    let settingsReads = 0;
    let modelReads = 0;
    let turnRequests = 0;
    page.on('pageerror', error => pageErrors.push(error.message));
    page.on('request', request => {
      if (/\/api\/assistant\/turn(?:\?|$)/.test(request.url())) turnRequests++;
    });
    await page.route('**/api/assistant/**', async route => {
      const pathname = new URL(route.request().url()).pathname;
      if (pathname === '/api/assistant/settings' && route.request().method() === 'GET') {
        settingsReads++;
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ config: savedConfig, revision: settingsReads, credentialPresent: false, credentialPersisted: false, credentialId: null }),
        });
        return;
      }
      if (pathname === '/api/assistant/settings' && route.request().method() === 'POST') {
        const payload = route.request().postDataJSON();
        savedPayloads.push(payload);
        savedConfig = payload.config;
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ config: savedConfig, revision: savedPayloads.length + 1, credentialPresent: false, credentialPersisted: false, credentialId: null }),
        });
        return;
      }
      if (pathname === '/api/assistant/models' && route.request().method() === 'GET') {
        modelReads++;
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ models: ['fixture-model'] }) });
        return;
      }
      if (pathname === '/api/assistant/sessions' && route.request().method() === 'GET') {
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) });
        return;
      }
      await route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ error: `Unexpected assistant route: ${pathname}` }) });
    });

    const response = await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
    assert.equal(response && response.status(), 200, 'UI should load');
    await page.getByRole('button', { name: 'Open AI assistant' }).click();
    await page.getByRole('button', { name: 'Assistant settings' }).click();
    const input = page.getByLabel('Image size (megapixels)');
    await input.waitFor({ state: 'visible' });
    assert.equal(await input.inputValue(), '1', 'new/default settings should show 1 MP');
    fs.mkdirSync(outputDir, { recursive: true });
    await page.screenshot({ path: path.join(outputDir, 'default.png'), fullPage: true });

    await input.fill('0');
    assert.equal(await input.evaluate(element => element.validity.valid), false, 'zero must be invalid');
    await page.getByRole('button', { name: 'Save settings' }).click();
    assert.equal(savedPayloads.length, 0, 'invalid settings must not be submitted');

    await input.fill('0.25');
    assert.equal(await input.evaluate(element => element.validity.valid), true, 'positive fractions must be valid');
    await page.screenshot({ path: path.join(outputDir, 'quarter.png'), fullPage: true });
    await page.getByRole('button', { name: 'Save settings' }).click();
    await page
      .getByRole('dialog', { name: 'AI assistant' })
      .getByRole('button', { name: 'Close AI assistant' })
      .click();
    await page.getByRole('button', { name: 'Open AI assistant' }).click();
    await page.getByRole('button', { name: 'Assistant settings' }).click();
    const reopenedInput = page.getByLabel('Image size (megapixels)');
    assert.equal(await reopenedInput.inputValue(), '0.25', 'saved fraction should restore after reopening');
    assert.equal(savedPayloads.length, 1);
    assert.equal(savedPayloads[0].config.imageMegapixels, 0.25);
    assert.ok(settingsReads >= 2, 'settings should be fetched again after closing and reopening the assistant');
    assert.ok(modelReads >= 1, 'model discovery should use the fixture route');
    assert.equal(turnRequests, 0, 'settings smoke must not create a chat turn');
    await page.setViewportSize({ width: 390, height: 844 });
    await page.screenshot({ path: path.join(outputDir, 'quarter-narrow.png'), fullPage: true });
    assert.deepEqual(pageErrors, [], `browser runtime errors: ${pageErrors.join('; ')}`);

    process.stdout.write(`${JSON.stringify({
      result: 'passed',
      defaultMegapixels: 1,
      rejectedZero: true,
      savedAndRestoredFraction: 0.25,
      settingsReads,
      modelReads,
      turnRequests,
      screenshots: [path.join(outputDir, 'default.png'), path.join(outputDir, 'quarter.png'), path.join(outputDir, 'quarter-narrow.png')],
    })}\n`);
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error.stack || error);
  process.exitCode = 1;
});
