import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import nunjucks from 'nunjucks';

const readStdin = async () => new Promise((resolve, reject) => {
  let input = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', chunk => {
    input += chunk;
  });
  process.stdin.on('end', () => resolve(input));
  process.stdin.on('error', reject);
});

const stripSafetensors = value => {
  const basename = path.basename(String(value || ''));
  if (basename.endsWith('.safetensors')) {
    return basename.slice(0, -'.safetensors'.length);
  }
  return basename.replace(path.extname(basename), '');
};

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..', '..');
const templateArg = process.argv[2];

if (!templateArg) {
  console.error('Usage: node ui/scripts/render_comfy_template.mjs <workflow-template-path>');
  process.exit(2);
}

const templatePath = path.isAbsolute(templateArg)
  ? templateArg
  : path.resolve(repoRoot, templateArg);

if (!fs.existsSync(templatePath)) {
  console.error(`ComfyUI workflow template does not exist: ${templatePath}`);
  process.exit(2);
}

let context = {};
const stdin = await readStdin();
if (stdin.trim() !== '') {
  try {
    context = JSON.parse(stdin);
  } catch (error) {
    console.error(`Template context was not valid JSON: ${error.message}`);
    process.exit(2);
  }
}

const env = new nunjucks.Environment(
  new nunjucks.FileSystemLoader(path.dirname(templatePath), { noCache: true }),
  { autoescape: false },
);

env.addFilter('json', value => JSON.stringify(value ?? ''));
env.addFilter('basename', value => path.basename(String(value || '')));
env.addFilter('strip_safetensors', value => stripSafetensors(value));

try {
  process.stdout.write(env.render(path.basename(templatePath), context));
} catch (error) {
  console.error(error.stack || error.message);
  process.exit(1);
}
