import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { isMac } from '@/helpers/basic';
import fs from 'fs';
import path from 'path';
import { cached } from '@/server/apiCache';
import { mergeSampleOnlyJobConfig } from '@/helpers/sampleOnlyJobConfig';
import { validateTrainingCapabilities } from '@/app/jobs/new/trainingCapabilities';
import { validateSliderSpace } from '@/app/jobs/new/sliderspace';

const TOOLKIT_ROOT = path.resolve('@', '..', '..');
const defaultTrainFolder = path.join(TOOLKIT_ROOT, 'output');
const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']);
const firstDatasetImageCache = new Map<string, { mtimeMs: number; imagePath: string | null }>();

const cloneJson = <T>(value: T): T => JSON.parse(JSON.stringify(value));

const getTrainingFolder = async () => {
  const row = await prisma.settings.findFirst({
    where: {
      key: 'TRAINING_FOLDER',
    },
  });
  if (row?.value && row.value !== '') {
    return row.value as string;
  }
  return defaultTrainFolder;
};

const writeRunningJobConfigSnapshot = async (jobName: string, jobConfig: any) => {
  const trainingRoot = await getTrainingFolder();
  const trainingFolder = path.join(trainingRoot, jobName);
  const configPath = path.join(trainingFolder, '.job_config.json');
  const tempPath = `${configPath}.${process.pid}.${Date.now()}.tmp`;
  const snapshot = cloneJson(jobConfig);

  snapshot.config.process[0].sqlite_db_path = path.join(TOOLKIT_ROOT, 'aitk_db.db');
  fs.mkdirSync(trainingFolder, { recursive: true });
  fs.writeFileSync(tempPath, JSON.stringify(snapshot, null, 2));
  fs.renameSync(tempPath, configPath);
};

const findFirstImageInDirectory = (dir: string): string | null => {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return null;
  }

  entries.sort((a, b) => a.name.localeCompare(b.name));

  for (const entry of entries) {
    if (!entry.isFile() || entry.name.startsWith('.')) continue;
    const ext = path.extname(entry.name).toLowerCase();
    if (IMAGE_EXTENSIONS.has(ext)) {
      return path.join(dir, entry.name);
    }
  }

  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.') || entry.name === '_controls') continue;
    const imagePath = findFirstImageInDirectory(path.join(dir, entry.name));
    if (imagePath) return imagePath;
  }

  return null;
};

const getFirstDatasetImagePath = (jobConfigJson: string): string | null => {
  try {
    const jobConfig = JSON.parse(jobConfigJson);
    const firstDatasetFolder = jobConfig?.config?.process?.[0]?.datasets?.[0]?.folder_path;
    if (!firstDatasetFolder || typeof firstDatasetFolder !== 'string') {
      return null;
    }

    const stat = fs.statSync(firstDatasetFolder);
    const cached = firstDatasetImageCache.get(firstDatasetFolder);
    if (cached && cached.mtimeMs === stat.mtimeMs) {
      return cached.imagePath;
    }

    const imagePath = findFirstImageInDirectory(firstDatasetFolder);
    firstDatasetImageCache.set(firstDatasetFolder, { mtimeMs: stat.mtimeMs, imagePath });
    return imagePath;
  } catch {
    return null;
  }
};

const attachDatasetThumbnail = <T extends { job_config: string }>(job: T) => {
  const firstImagePath = getFirstDatasetImagePath(job.job_config);
  return {
    ...job,
    dataset_thumbnail_path: firstImagePath,
    dataset_thumbnail_url: firstImagePath ? `/api/img/${encodeURIComponent(firstImagePath)}` : null,
  };
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const id = searchParams.get('id');
  const job_ref = searchParams.get('job_ref');
  const job_type = searchParams.get('job_type');
  const only_active = searchParams.get('only_active');
  const name = searchParams.get('name')?.trim() || '';
  const matchesName = (job: { name: string }) => job.name.toLowerCase().includes(name.toLowerCase());

  try {
    if (id) {
      const job = await prisma.job.findUnique({
        where: { id },
      });
      return NextResponse.json(job ? attachDatasetThumbnail(job) : job);
    }
    if (job_ref) {
      const job = await prisma.job.findFirst({
        where: { job_ref },
        orderBy: { updated_at: 'desc' },
      });
      return NextResponse.json(job ? attachDatasetThumbnail(job) : job);
    }

    const where: any = {};
    if (job_type) {
      where.job_type = job_type;
    }
    if (name) {
      where.name = { contains: name };
    }
    if (only_active === 'true') {
      where.status = { in: ['running', 'queued', 'stopping'] };
      const jobs = await cached(
        'jobs-active',
        () =>
          prisma.job.findMany({
            where,
            orderBy: { created_at: 'desc' },
          }),
        5000,
        { job_type, name },
      );
      // SQLite LIKE treats underscores and percent signs as wildcards. Keep
      // name searches literal, since those characters are common in job names.
      return NextResponse.json({ jobs: jobs.filter(matchesName).map(attachDatasetThumbnail) });
    }

    const jobs = await prisma.job.findMany({
      where,
      orderBy: { created_at: 'desc' },
    });
    return NextResponse.json({ jobs: jobs.filter(matchesName).map(attachDatasetThumbnail) });
  } catch (error) {
    console.error(error);
    return NextResponse.json({ error: 'Failed to fetch training data' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { id, name, job_config, sample_only } = body;
    if (!sample_only) {
      const errors = [...validateTrainingCapabilities(job_config), ...validateSliderSpace(job_config)];
      if (errors.length) return NextResponse.json({ error: [...new Set(errors)].join('\n') }, { status: 400 });
    }
    let gpu_ids: string = body.gpu_ids;

    if (isMac()) {
      gpu_ids = 'mps';
    }

    const extra: any = {};
    if ('job_ref' in body) {
      extra['job_ref'] = body.job_ref;
    }

    if ('job_type' in body) {
      extra['job_type'] = body.job_type;
    }

    if (id) {
      if (sample_only) {
        const existingJob = await prisma.job.findUnique({
          where: { id },
        });
        if (!existingJob) {
          return NextResponse.json({ error: 'Job not found' }, { status: 404 });
        }
        if (existingJob.status !== 'running') {
          return NextResponse.json(
            { error: 'Sample-only editing is only available for running jobs' },
            { status: 409 },
          );
        }

        const existingConfig = JSON.parse(existingJob.job_config);
        let mergedConfig;
        try {
          mergedConfig = mergeSampleOnlyJobConfig(existingConfig, job_config);
        } catch (error) {
          return NextResponse.json({ error: error instanceof Error ? error.message : 'Invalid sample settings' }, { status: 400 });
        }
        await writeRunningJobConfigSnapshot(existingJob.name, mergedConfig);

        const training = await prisma.job.update({
          where: { id },
          data: {
            job_config: JSON.stringify(mergedConfig),
          },
        });
        return NextResponse.json(training);
      }

      // Update existing training
      const training = await prisma.job.update({
        where: { id },
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          ...extra,
        },
      });
      return NextResponse.json(training);
    } else {
      // find the highest queue position and add 1000
      const highestQueuePosition = await prisma.job.aggregate({
        _max: {
          queue_position: true,
        },
      });
      const newQueuePosition = (highestQueuePosition._max.queue_position || 0) + 1000;

      // Create new training
      const training = await prisma.job.create({
        data: {
          name,
          gpu_ids,
          job_config: JSON.stringify(job_config),
          queue_position: newQueuePosition,
          ...extra,
        },
      });
      return NextResponse.json(training);
    }
  } catch (error: any) {
    if (error.code === 'P2002') {
      // Handle unique constraint violation, 409=Conflict
      return NextResponse.json({ error: 'Job name already exists' }, { status: 409 });
    }
    console.error(error);
    // Handle other errors
    return NextResponse.json({ error: 'Failed to save training data' }, { status: 500 });
  }
}
