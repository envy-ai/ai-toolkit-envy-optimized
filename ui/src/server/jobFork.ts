import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { TOOLKIT_ROOT } from '@/paths';
import { resolvePythonPath } from '../../cron/pythonPath';
import { Job } from '@prisma/client';

export class JobForkError extends Error {
  constructor(message: string, public status = 400, stack?: string) {
    super(message);
    if (stack) this.stack = stack;
  }
}

const safeName = (name: string) => {
  if (typeof name !== 'string' || !name.trim() || name === '.' || name === '..' || /[\\/\0]/.test(name)) {
    throw new JobForkError('Fork job name must be a nonempty folder name without path separators');
  }
};

export const assertForkSourceSafe = (job: Job, folder: string) => {
  if (job.job_type !== 'train' || !['stopped', 'completed', 'error'].includes(job.status)) {
    throw new JobForkError('Pause the source training job and wait for it to stop before forking', 409);
  }
  const pids = new Set<number>();
  if (job.pid != null) pids.add(job.pid);
  for (const name of ['pid.txt', '.job_pid']) {
    const filename = path.join(folder, name);
    if (!fs.existsSync(filename)) continue;
    const text = fs.readFileSync(filename, 'utf8').trim();
    if (!text || text.startsWith('error:')) continue;
    if (!/^\d+$/.test(text) || Number(text) <= 0) throw new JobForkError(`Invalid source process record: ${filename}`);
    pids.add(Number(text));
  }
  for (const pid of pids) {
    try {
      process.kill(pid, 0);
    } catch (error: any) {
      if (error.code === 'ESRCH') continue;
      throw new JobForkError(`Cannot verify that source process ${pid} has stopped: ${error.message}`, 409);
    }
    throw new JobForkError(`Source process ${pid} is still alive. Wait for it to exit before forking`, 409);
  }
};

export type ForkSnapshot = { step: number; checkpoint: string; layout: string; warnings?: string[] };

const runHelper = (action: 'inspect' | 'copy', payload: Record<string, unknown>): Promise<ForkSnapshot> =>
  new Promise((resolve, reject) => {
    const child = spawn(resolvePythonPath(), [path.join(TOOLKIT_ROOT, 'toolkit', 'job_fork.py'), action], {
      cwd: TOOLKIT_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', data => { stdout += data.toString(); });
    child.stderr.on('data', data => { stderr += data.toString(); });
    child.once('error', reject);
    child.stdin.on('error', reject);
    child.once('close', code => {
      if (code !== 0) {
        try {
          const details = JSON.parse(stderr.trim());
          reject(new JobForkError(details.error, 400, details.stack));
        } catch {
          reject(new JobForkError(`Checkpoint ${action} failed: ${stderr.trim() || `exit ${code}`}`));
        }
        return;
      }
      try { resolve(JSON.parse(stdout.trim())); } catch (error) { reject(error); }
    });
    child.stdin.end(JSON.stringify(payload));
  });

const getSource = async (id: string) => {
  const job = await prisma.job.findUnique({ where: { id } });
  if (!job) throw new JobForkError('Source job not found', 404);
  safeName(job.name);
  const root = await getTrainingFolder();
  const folder = path.join(root, job.name);
  assertForkSourceSafe(job, folder);
  const config = JSON.parse(job.job_config);
  if (config.config.name !== job.name || config.config.process.length !== 1) {
    throw new JobForkError('Source job name/process configuration does not match its training folder');
  }
  const configuredRoot = config.config.process[0].training_folder;
  if (typeof configuredRoot !== 'string' || path.resolve(TOOLKIT_ROOT, configuredRoot) !== root) {
    throw new JobForkError('Source training_folder differs from the configured training root');
  }
  return { job, config, root, folder };
};

const verifySource = async (source: Awaited<ReturnType<typeof getSource>>) => {
  const current = await prisma.job.findUnique({ where: { id: source.job.id } });
  if (!current || current.name !== source.job.name || current.job_config !== source.job.job_config) {
    throw new JobForkError('Source job changed while forking; reload and retry', 409);
  }
  assertForkSourceSafe(current, source.folder);
};

export const previewJobFork = async (id: string) => {
  const source = await getSource(id);
  const snapshot = await runHelper('inspect', {
    source_folder: source.folder, source_name: source.job.name, source_config: source.config,
  });
  await verifySource(source);
  let name = `${source.job.name}_fork`;
  const jobs = await prisma.job.findMany({ select: { name: true } });
  const names = new Set(jobs.map(job => job.name));
  for (let suffix = 2; names.has(name) || fs.existsSync(path.join(source.root, name)); suffix++) {
    name = `${source.job.name}_fork_${suffix}`;
  }
  const config = JSON.parse(source.job.job_config);
  config.config.name = name;
  delete config.config.process[0].name;
  delete config.config.process[0].train.start_step;
  return { ...source.job, name, job_config: JSON.stringify(config), fork_snapshot: snapshot };
};

export const createJobFork = async (
  id: string, name: string, config: any, create: (snapshot: ForkSnapshot) => Promise<Job>,
) => {
  safeName(name);
  const source = await getSource(id);
  if (name === source.job.name || await prisma.job.findUnique({ where: { name } })) {
    throw new JobForkError('Job name already exists. Choose a different name', 409);
  }
  if (config?.config?.name !== name || config?.config?.process?.length !== 1) {
    throw new JobForkError('Fork name must match config.name and have one training process');
  }
  if (path.resolve(TOOLKIT_ROOT, config.config.process[0].training_folder) !== source.root) {
    throw new JobForkError('Fork must use the configured training folder');
  }
  const target = path.join(source.root, name);
  if (fs.existsSync(target)) throw new JobForkError('A training folder already uses this name. Choose a different name', 409);
  const snapshot = await runHelper('copy', {
    source_folder: source.folder, target_folder: target, source_name: source.job.name,
    target_name: name, source_config: source.config, target_config: config,
  });
  try {
    await verifySource(source);
    return await create(snapshot);
  } catch (error) {
    await fs.promises.rm(target, { recursive: true, force: true });
    throw error;
  }
};
