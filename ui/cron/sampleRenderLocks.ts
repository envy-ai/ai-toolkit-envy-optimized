import fs from 'fs';
import path from 'path';
import { randomUUID } from 'crypto';

export const processAlive = (pid: number): boolean => {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error: any) {
    return error.code === 'EPERM';
  }
};

export const gpuKeys = (gpuIds: string): string[] => {
  const ids = [
    ...new Set(
      gpuIds
        .split(',')
        .map(id => id.trim())
        .filter(Boolean),
    ),
  ];
  if (!ids.length) throw new Error('The job has no GPU configured.');
  return ids.map(id => `gpu-${id}`).sort();
};

export class GpuBusyError extends Error {}
export const SAMPLE_RENDER_GATE = 'saved-sample-render';

export function hasSampleRenderLease(root: string, jobId: string): boolean {
  const ownerPath = path.join(root, 'data', 'gpu-launch-locks', SAMPLE_RENDER_GATE, 'owner.json');
  try {
    const owner = JSON.parse(fs.readFileSync(ownerPath, 'utf8'));
    return owner.jobId === jobId && processAlive(owner.pid);
  } catch (error: any) {
    if (error.code === 'ENOENT') return false;
    throw error;
  }
}

/** A shared atomic gate for the queue's launch and a saved sample render.
 * The global sample gate is necessary because the Comfy server's device is
 * independent of CUDA_VISIBLE_DEVICES on the caller. Dead owners
 * are reclaimed; a short grace period covers a crash between mkdir and owner.
 */
export function acquireGpuLocks(root: string, keys: string[], jobId: string) {
  const lockRoot = path.join(root, 'data', 'gpu-launch-locks');
  fs.mkdirSync(lockRoot, { recursive: true });
  const token = randomUUID();
  const acquired: string[] = [];
  const owner = { token, pid: process.pid, jobId };
  const release = () => {
    for (const folder of acquired) {
      const ownerPath = path.join(folder, 'owner.json');
      try {
        if (JSON.parse(fs.readFileSync(ownerPath, 'utf8')).token === token) fs.rmSync(folder, { recursive: true });
      } catch (error: any) {
        if (error.code !== 'ENOENT') throw error;
        // We created this directory but its initial owner publication failed.
        fs.rmSync(folder, { recursive: true, force: true });
      }
    }
  };
  try {
    for (const key of [...new Set(keys)].sort()) {
      const folder = path.join(lockRoot, encodeURIComponent(key));
      if (fs.existsSync(folder)) {
        let live = true;
        try {
          live = processAlive(JSON.parse(fs.readFileSync(path.join(folder, 'owner.json'), 'utf8')).pid);
        } catch (error: any) {
          if (error.code !== 'ENOENT') throw error;
          live = Date.now() - fs.statSync(folder).mtimeMs < 30_000;
        }
        if (!live) fs.rmSync(folder, { recursive: true });
      }
      try {
        fs.mkdirSync(folder);
      } catch (error: any) {
        if (error.code === 'EEXIST')
          throw new GpuBusyError(`GPU or renderer is busy (${key}). Wait for the current operation to finish.`);
        throw error;
      }
      acquired.push(folder);
      const temporary = path.join(folder, `${token}.tmp`);
      fs.writeFileSync(temporary, JSON.stringify(owner));
      fs.renameSync(temporary, path.join(folder, 'owner.json'));
    }
    return {
      token,
      folders: acquired,
      setPid(pid: number) {
        for (const folder of acquired) {
          const temporary = path.join(folder, `${token}.tmp`);
          fs.writeFileSync(temporary, JSON.stringify({ ...owner, pid }));
          fs.renameSync(temporary, path.join(folder, 'owner.json'));
        }
      },
      release,
    };
  } catch (error) {
    release();
    throw error;
  }
}
