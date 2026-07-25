import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawnSync } from 'child_process';

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const uiRoot = path.resolve(scriptDir, '..');
const isDryRun = process.argv.includes('--dry-run');
const isCheckOnly = process.argv.includes('--check');

const outputPaths = [
  path.join(uiRoot, '.next', 'BUILD_ID'),
  path.join(uiRoot, 'dist', 'cron', 'worker.js'),
  path.join(uiRoot, 'dist', 'cron', 'fileServer.js'),
];

const sourcePaths = [
  path.join(uiRoot, 'src'),
  path.join(uiRoot, 'cron'),
  path.join(uiRoot, 'prisma', 'schema.prisma'),
  path.join(uiRoot, 'package.json'),
  path.join(uiRoot, 'package-lock.json'),
  path.join(uiRoot, 'next.config.ts'),
  path.join(uiRoot, 'tsconfig.json'),
  path.join(uiRoot, 'tsconfig.worker.json'),
  path.join(uiRoot, 'tailwind.config.ts'),
  path.join(uiRoot, 'postcss.config.mjs'),
];

function getMtimeMs(filePath) {
  try {
    return fs.statSync(filePath).mtimeMs;
  } catch {
    return null;
  }
}

function newestSourceMtime(entryPath) {
  const stat = fs.statSync(entryPath);
  if (!stat.isDirectory()) {
    return stat.mtimeMs;
  }

  let newest = stat.mtimeMs;
  for (const entry of fs.readdirSync(entryPath, { withFileTypes: true })) {
    if (entry.name === 'node_modules' || entry.name === '.next' || entry.name === 'dist') {
      continue;
    }

    const childPath = path.join(entryPath, entry.name);
    newest = Math.max(newest, newestSourceMtime(childPath));
  }
  return newest;
}

function oldestOutputMtime() {
  let oldest = Number.POSITIVE_INFINITY;
  for (const outputPath of outputPaths) {
    const outputMtime = getMtimeMs(outputPath);
    if (outputMtime === null) {
      return null;
    }
    oldest = Math.min(oldest, outputMtime);
  }
  return oldest;
}

function shouldBuild() {
  const oldestOutput = oldestOutputMtime();
  if (oldestOutput === null) {
    return { needed: true, reason: 'required build output is missing' };
  }

  let newestSource = 0;
  for (const sourcePath of sourcePaths) {
    if (!fs.existsSync(sourcePath)) {
      continue;
    }
    newestSource = Math.max(newestSource, newestSourceMtime(sourcePath));
  }

  if (newestSource > oldestOutput) {
    return { needed: true, reason: 'source files are newer than build output' };
  }

  return { needed: false, reason: 'build output is current' };
}

const buildState = shouldBuild();

if (!buildState.needed) {
  console.log(`[build-if-needed] Skipping build: ${buildState.reason}.`);
  process.exit(0);
}

console.log(`[build-if-needed] Build needed: ${buildState.reason}.`);

if (isDryRun || isCheckOnly) {
  process.exit(isCheckOnly ? 1 : 0);
}

const result = spawnSync('npm', ['run', 'build'], {
  cwd: uiRoot,
  env: process.env,
  stdio: 'inherit',
});

process.exit(result.status ?? 1);
