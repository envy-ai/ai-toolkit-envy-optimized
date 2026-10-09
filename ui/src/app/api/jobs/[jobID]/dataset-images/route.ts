import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import fs from 'fs';
import path from 'path';
import { getDataRoot, getDatasetsRoot, getTrainingFolder } from '@/server/settings';

const IMAGE_EXTENSIONS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']);

interface DatasetImage {
  path: string;
  url: string;
  filename: string;
  dataset_path: string;
  dataset_name: string;
  caption: string;
  caption_ext: string;
  caption_path: string;
}

const isUnderRoot = (filepath: string, root: string): boolean => {
  const resolvedPath = path.resolve(filepath);
  const resolvedRoot = path.resolve(root);
  return resolvedPath === resolvedRoot || resolvedPath.startsWith(resolvedRoot + path.sep);
};

const isAllowedPath = (filepath: string, allowedRoots: string[]): boolean =>
  allowedRoots.some(root => isUnderRoot(filepath, root));

const normalizeCaptionExt = (ext: unknown): string => {
  if (typeof ext !== 'string') return 'txt';
  return ext.replace(/^\.+/, '').trim() || 'txt';
};

const readCaptionForImage = (imagePath: string, captionExt: string) => {
  const caption_path = imagePath.replace(/\.[^/.]+$/, '') + '.' + captionExt;
  let caption = '';
  try {
    if (fs.existsSync(caption_path)) {
      caption = fs.readFileSync(caption_path, 'utf-8');
    }
  } catch {
    caption = '';
  }

  return { caption, caption_path };
};

const findDatasetImages = (dir: string): string[] => {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }

  const images: string[] = [];
  entries.sort((a, b) => a.name.localeCompare(b.name));

  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const entryPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      if (entry.name === '_controls') continue;
      images.push(...findDatasetImages(entryPath));
      continue;
    }

    if (!entry.isFile()) continue;
    const ext = path.extname(entry.name).toLowerCase();
    if (IMAGE_EXTENSIONS.has(ext)) {
      images.push(entryPath);
    }
  }

  return images;
};

export async function GET(_request: Request, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;

  try {
    const job = await prisma.job.findUnique({
      where: { id: jobID },
    });

    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const jobConfig = JSON.parse(job.job_config);
    const datasets = jobConfig?.config?.process?.[0]?.datasets;
    if (!Array.isArray(datasets)) {
      return NextResponse.json({ images: [] });
    }

    const allowedRoots = await Promise.all([getDatasetsRoot(), getTrainingFolder(), getDataRoot()]);
    const images: DatasetImage[] = [];

    for (const dataset of datasets) {
      if (!dataset || typeof dataset.folder_path !== 'string') continue;

      const datasetPath = path.resolve(dataset.folder_path);
      if (!isAllowedPath(datasetPath, allowedRoots)) continue;

      const stat = fs.existsSync(datasetPath) ? fs.statSync(datasetPath) : null;
      if (!stat?.isDirectory()) continue;

      const captionExt = normalizeCaptionExt(dataset.caption_ext || 'txt');
      const datasetName = path.basename(datasetPath);
      const datasetImages = findDatasetImages(datasetPath);

      for (const imagePath of datasetImages) {
        if (!isAllowedPath(imagePath, allowedRoots)) continue;
        const { caption, caption_path } = readCaptionForImage(imagePath, captionExt);
        images.push({
          path: imagePath,
          url: `/api/img/${encodeURIComponent(imagePath)}`,
          filename: path.basename(imagePath),
          dataset_path: datasetPath,
          dataset_name: datasetName,
          caption,
          caption_ext: captionExt,
          caption_path,
        });
      }
    }

    return NextResponse.json({ images });
  } catch (error) {
    console.error('Error listing job dataset images:', error);
    return NextResponse.json({ error: 'Failed to list dataset images' }, { status: 500 });
  }
}
