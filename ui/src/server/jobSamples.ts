import fs from 'fs';
import path from 'path';
import { createHash, randomUUID } from 'crypto';
import { execFile, spawn } from 'child_process';
import { promisify } from 'util';
import { Job } from '@prisma/client';
import prisma from '@/server/prisma';
import { getTrainingFolder, getHFToken, getModelsPath } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';
import { resolveDetachedPythonPath } from '../../cron/pythonPath';
import { acquireGpuLocks, gpuKeys, processAlive, SAMPLE_RENDER_GATE } from '../../cron/sampleRenderLocks';

const execFileAsync = promisify(execFile);
export interface SampleRenderState {
  status: 'idle' | 'running' | 'completed' | 'error';
  message: string;
  error?: string;
  stack?: string;
  checkpoint?: string;
  step?: number;
  pid?: number;
}

async function renderFolder(job: Job, requireCheckpointFolder = true) {
  const root = path.resolve(await getTrainingFolder());
  const folder = path.resolve(root, job.name);
  if (folder === root || !folder.startsWith(root + path.sep)) throw new Error('Invalid job output folder.');
  if (!fs.existsSync(folder)) {
    if (requireCheckpointFolder)
      throw new Error('The job output folder does not exist. Save a checkpoint before rendering samples.');
  } else if (fs.realpathSync(folder) !== folder)
    throw new Error('The job output folder must not contain symbolic links.');
  const render = path.join(folder, '.sample_render');
  if (fs.existsSync(render) && fs.realpathSync(render) !== render)
    throw new Error('The sample render folder must not be a symbolic link.');
  return render;
}

function writeState(filename: string, state: SampleRenderState) {
  const temporary = `${filename}.${randomUUID()}.tmp`;
  fs.writeFileSync(temporary, JSON.stringify(state, null, 2));
  fs.renameSync(temporary, filename);
}

export async function readSampleRender(job: Job): Promise<SampleRenderState> {
  const filename = path.join(await renderFolder(job, false), 'status.json');
  let state: SampleRenderState;
  try {
    state = JSON.parse(fs.readFileSync(filename, 'utf8'));
  } catch (error: any) {
    if (error.code === 'ENOENT') return { status: 'idle', message: '' };
    throw error;
  }
  if (state.status === 'running' && state.pid && !processAlive(state.pid)) {
    state = {
      ...state,
      status: 'error',
      error: 'The sample renderer exited before reporting completion.',
      message: 'The sample renderer exited before reporting completion.',
      stack: fs.readFileSync(path.join(path.dirname(filename), 'log.txt'), 'utf8'),
    };
    writeState(filename, state);
  }
  return state;
}

export async function startSampleRender(job: Job): Promise<SampleRenderState> {
  if (job.job_type !== 'train' || !['stopped', 'completed', 'error'].includes(job.status)) {
    throw new Error('Render Samples is available for stopped, completed or failed training jobs.');
  }
  const config = JSON.parse(job.job_config);
  const comfyUrl = config.config?.process?.[0]?.sample?.comfy?.api_url || '';
  const keys = gpuKeys(job.gpu_ids);
  const comfyKey = `comfy-${createHash('sha256').update(comfyUrl.replace(/\/+$/, '')).digest('hex')}`;
  const locks = acquireGpuLocks(TOOLKIT_ROOT, [...keys, comfyKey, SAMPLE_RENDER_GATE], job.id);
  let launched = false;
  let statusPath: string | undefined;
  try {
    const jobs = await prisma.job.findMany();
    const conflict = jobs.find(
      other => ['running', 'stopping'].includes(other.status) || (other.pid != null && processAlive(other.pid)),
    );
    if (conflict)
      throw new Error(
        `Job "${conflict.name}" is active. Wait for active jobs to stop before rendering saved samples: ComfyUI's GPU is configured independently of the training job. Queued jobs will wait for the render to finish.`,
      );
    const folder = await renderFolder(job);
    fs.mkdirSync(folder, { recursive: true });
    statusPath = path.join(folder, 'status.json');
    const requestPath = path.join(folder, 'request.json');
    fs.writeFileSync(
      requestPath,
      JSON.stringify({
        config,
        job_folder: path.dirname(folder),
        status_path: statusPath,
        lock_folders: locks.folders,
        lock_token: locks.token,
      }),
    );
    const python = resolveDetachedPythonPath();
    const args = ['-m', 'toolkit.render_job_samples', requestPath];
    let validated: SampleRenderState;
    try {
      const { stdout } = await execFileAsync(python, [...args, '--validate'], {
        cwd: TOOLKIT_ROOT,
        timeout: 30_000,
        maxBuffer: 1024 * 1024,
      });
      validated = JSON.parse(stdout);
    } catch (error: any) {
      throw new Error(error.stderr || error.message);
    }
    const state: SampleRenderState = {
      ...validated,
      status: 'running',
      message: `Starting sample render from checkpoint at step ${validated.step}…`,
      pid: process.pid,
    };
    writeState(statusPath, state);
    const hfToken = await getHFToken();
    const modelsPath = await getModelsPath();
    const logFd = fs.openSync(path.join(folder, 'log.txt'), 'w');
    try {
      const child = spawn(python, args, {
        cwd: TOOLKIT_ROOT,
        detached: true,
        windowsHide: true,
        stdio: ['ignore', logFd, logFd],
        // Workflow rendering needs no local CUDA context. Comfy selects its
        // own device; hiding local GPUs also keeps Accelerate's logging on CPU.
        env: {
          ...process.env,
          AITK_JOB_ID: '',
          CUDA_VISIBLE_DEVICES: '',
          ACCELERATE_USE_CPU: 'true',
          PYTHONUNBUFFERED: '1',
          ...(hfToken ? { HF_TOKEN: hfToken } : {}),
          ...(!process.env.MODELS_PATH && modelsPath ? { MODELS_PATH: modelsPath } : {}),
        },
      });
      await new Promise<void>((resolve, reject) => {
        child.once('spawn', resolve);
        child.once('error', reject);
      });
      if (!child.pid) throw new Error('The sample renderer did not report a process ID.');
      launched = true;
      try {
        locks.setPid(child.pid);
      } catch (error) {
        // A spawned child may already be rendering. Do not release its gate
        // until it is gone if ownership bookkeeping fails.
        const exited = new Promise<void>(resolve => child.once('exit', () => resolve()));
        if (child.kill('SIGKILL')) {
          await exited;
          launched = false;
        }
        throw error;
      }
      const rendererStatusPath = statusPath;
      child.once('exit', (code, signal) => {
        try {
          const current = JSON.parse(fs.readFileSync(rendererStatusPath, 'utf8')) as SampleRenderState;
          if (current.status !== 'running') return;
          const message = `Sample renderer exited without completing (${signal || `exit code ${code}`}).`;
          writeState(rendererStatusPath, {
            ...current,
            status: 'error',
            message,
            error: message,
            stack: fs.readFileSync(path.join(folder, 'log.txt'), 'utf8'),
          });
        } catch (error) {
          console.error('Could not record renderer exit:', error);
        }
      });
      state.pid = child.pid;
      child.unref();
      return state;
    } finally {
      fs.closeSync(logFd);
    }
  } catch (error: any) {
    if (statusPath)
      writeState(statusPath, { status: 'error', message: error.message, error: error.message, stack: error.stack });
    throw error;
  } finally {
    if (!launched) locks.release();
  }
}
