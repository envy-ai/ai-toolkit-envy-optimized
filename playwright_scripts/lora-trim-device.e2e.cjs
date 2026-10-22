const assert = require('node:assert/strict');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');

const ROOT = path.resolve(__dirname, '..');
const BASE_URL = process.env.LORA_TRIM_BASE_URL || 'http://127.0.0.1:8676';
const JOB_ID = 'trim-e2e';
const SOURCE_NAME = 'synthetic_lora.safetensors';
const SOURCE_PATH = path.join(ROOT, 'tmp/lora-trim/e2e-output/trim_fixture', SOURCE_NAME);

async function main() {
  const browser = await chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_EXECUTABLE_PATH || undefined,
    args: ['--disable-gpu'],
  });
  const page = await browser.newPage();
  page.setDefaultTimeout(30000);
  try {
    await page.route('**/api/monitor', route => route.abort());
    await page.goto(`${BASE_URL}/jobs/${JOB_ID}`, { waitUntil: 'domcontentloaded' });
    await page.getByRole('button', { name: `Reduce ${SOURCE_NAME}` }).click();
    const dialog = page.getByRole('dialog', { name: 'Reduce adapter' });
    await dialog.waitFor({ state: 'visible' });

    const responsePromise = page.waitForResponse(response => {
      if (new URL(response.url()).pathname !== '/api/scripts' || response.request().method() !== 'POST') return false;
      try {
        const request = response.request().postDataJSON();
        return request?.script === 'trim_lora.py' && request.args?.input === SOURCE_PATH;
      } catch {
        return false;
      }
    });
    await dialog.getByRole('button', { name: 'Analyze' }).click();
    const response = await responsePromise;
    const envelope = await response.json();
    assert.equal(response.status(), 200);
    assert.equal(envelope.ok, true, JSON.stringify(envelope));
    assert.equal(envelope.result.device_requested, 'auto');
    assert.match(envelope.result.device_resolved, /^cuda:\d+$/);
    assert.match(envelope.result.runtime.python_executable, /ai-toolkit/);
    assert.match(envelope.result.runtime.gpu_name, /NVIDIA/i);

    const summary = dialog.getByText(/layers analyzed on cuda:\d+ \(NVIDIA/i);
    await summary.waitFor({ state: 'visible' });
    const title = await summary.getAttribute('title');
    assert.match(title, /^Python: .*ai-toolkit/);
    console.log(JSON.stringify({
      source: SOURCE_NAME,
      result: {
        device_requested: envelope.result.device_requested,
        device_resolved: envelope.result.device_resolved,
        gpu_name: envelope.result.runtime.gpu_name,
        python_executable: envelope.result.runtime.python_executable,
      },
      ui_summary: await summary.innerText(),
      ui_python_title: title,
    }));
  } finally {
    await browser.close();
  }
}

main().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
