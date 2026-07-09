import path from 'path';
import fs from 'fs';
import { TOOLKIT_ROOT } from './paths';

const isWindows = process.platform === 'win32';

// Shared resolver used by both the cron worker and Next.js API routes
// so the Python interpreter is configured in exactly one place.
export const resolvePythonPath = (): string => {
  const candidates: string[] = [];

  if (isWindows) {
    candidates.push(path.join(TOOLKIT_ROOT, '.venv', 'Scripts', 'python.exe'));
    candidates.push(path.join(TOOLKIT_ROOT, 'venv', 'Scripts', 'python.exe'));
    if (process.env.CONDA_PREFIX) {
      candidates.push(path.join(process.env.CONDA_PREFIX, 'python.exe'));
    }
  } else {
    candidates.push(path.join(TOOLKIT_ROOT, '.venv', 'bin', 'python'));
    candidates.push(path.join(TOOLKIT_ROOT, 'venv', 'bin', 'python'));
    if (process.env.CONDA_PREFIX) {
      candidates.push(path.join(process.env.CONDA_PREFIX, 'bin', 'python'));
    }
    candidates.push(path.join(process.env.HOME || '', '.conda', 'envs', 'ai-toolkit', 'bin', 'python'));
  }

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return isWindows ? 'python.exe' : 'python3';
};
