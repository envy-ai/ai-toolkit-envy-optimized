import { Job } from '@prisma/client';
import { getTotalSteps } from '@/utils/jobs';

const DEFAULT_FAVICON_HREF = '/icon.png';
const NORMAL_FAVICON_SELECTOR = 'link[rel="icon"]:not([data-training-progress-favicon="true"]), link[rel="shortcut icon"]';
const PROGRESS_FAVICON_SELECTOR = 'link[data-training-progress-favicon="true"]';
const PLAY_PAUSE_COLOR = '#67e8f9';
const QUEUE_COUNT_COLOR = '#bef264';

export type TrainingQueueFaviconState = {
  queueCount: number;
  isRunning: boolean;
};

export function getRunningTrainingProgress(jobs: Job[]): number | null {
  const runningTrainingJobs = jobs.filter(job => job.job_type === 'train' && job.status === 'running');
  let completedSteps = 0;
  let totalSteps = 0;

  for (const job of runningTrainingJobs) {
    const jobTotalSteps = getTotalSteps(job);
    if (!Number.isFinite(jobTotalSteps) || jobTotalSteps <= 0) continue;

    totalSteps += jobTotalSteps;
    completedSteps += Math.min(jobTotalSteps, Math.max(0, job.step || 0));
  }

  if (totalSteps <= 0) {
    return null;
  }

  return Math.min(1, Math.max(0, completedSteps / totalSteps));
}

export function getTrainingQueueFaviconState(jobs: Job[]): TrainingQueueFaviconState | null {
  const queueCount = jobs.filter(job => job.job_type === 'train' && job.status === 'queued').length;
  if (queueCount <= 0) {
    return null;
  }

  return {
    queueCount,
    isRunning: jobs.some(job => job.job_type === 'train' && job.status === 'running'),
  };
}

export function getDefaultFaviconHref(documentRef: Document = document): string {
  const normalFavicon = documentRef.querySelector<HTMLLinkElement>(NORMAL_FAVICON_SELECTOR);
  return normalFavicon?.href || DEFAULT_FAVICON_HREF;
}

function ensureDefaultFaviconLink(documentRef: Document): HTMLLinkElement {
  const existingLink = documentRef.querySelector<HTMLLinkElement>(NORMAL_FAVICON_SELECTOR);
  if (existingLink) {
    return existingLink;
  }

  const link = documentRef.createElement('link');
  link.rel = 'icon';
  documentRef.head.appendChild(link);
  return link;
}

function ensureProgressFaviconLink(documentRef: Document): HTMLLinkElement {
  const existingLink = documentRef.querySelector<HTMLLinkElement>(PROGRESS_FAVICON_SELECTOR);
  if (existingLink) {
    return existingLink;
  }

  const link = documentRef.createElement('link');
  link.rel = 'icon';
  link.type = 'image/png';
  link.setAttribute('data-training-progress-favicon', 'true');
  documentRef.head.appendChild(link);
  return link;
}

export function restoreDefaultFavicon(documentRef: Document = document): string {
  documentRef.querySelector<HTMLLinkElement>(PROGRESS_FAVICON_SELECTOR)?.remove();
  const normalFavicon = ensureDefaultFaviconLink(documentRef);
  normalFavicon.type = 'image/png';
  normalFavicon.href = DEFAULT_FAVICON_HREF;
  return normalFavicon.href;
}

export function drawTrainingProgressFavicon(
  defaultFaviconHref: string,
  progress: number | null,
  queueState: TrainingQueueFaviconState | null = null,
  documentRef: Document = document,
  shouldApply: () => boolean = () => true,
) {
  const size = 64;
  const canvas = documentRef.createElement('canvas');
  canvas.width = size;
  canvas.height = size;

  const context = canvas.getContext('2d');
  if (!context) {
    restoreDefaultFavicon(documentRef);
    return;
  }

  const drawProgressUnderlay = () => {
    if (progress === null) return;

    const clampedProgress = Math.min(1, Math.max(0, progress));
    const barHeight = Math.round(size * clampedProgress);
    if (barHeight > 0) {
      const y = size - barHeight;
      context.fillStyle = '#facc15';
      context.fillRect(0, y, size, barHeight);
      context.fillStyle = 'rgba(0, 0, 0, 0.35)';
      context.fillRect(0, Math.max(0, y - 1), size, 1);
    }
  };

  const drawOutlinedText = (text: string, x: number, y: number, fillColor: string) => {
    context.lineJoin = 'round';
    context.miterLimit = 2;
    context.strokeStyle = '#000000';
    context.lineWidth = 5;
    context.strokeText(text, x, y);
    context.fillStyle = fillColor;
    context.fillText(text, x, y);
  };

  const drawQueueBadges = () => {
    if (!queueState || queueState.queueCount <= 0) return;

    context.font = '700 24px Arial, sans-serif';
    context.textBaseline = 'bottom';

    context.textAlign = 'left';
    drawOutlinedText(queueState.isRunning ? '\u25b6' : '\u23f8', 2, size - 1, PLAY_PAUSE_COLOR);

    context.textAlign = 'right';
    drawOutlinedText(String(queueState.queueCount), size - 2, size - 1, QUEUE_COUNT_COLOR);
  };

  const publishProgressFavicon = () => {
    ensureProgressFaviconLink(documentRef).href = canvas.toDataURL('image/png');
  };

  const image = new Image();
  image.onload = () => {
    if (!shouldApply()) return;

    context.clearRect(0, 0, size, size);
    drawProgressUnderlay();
    context.drawImage(image, 0, 0, size, size);
    drawQueueBadges();
    publishProgressFavicon();
  };
  image.onerror = () => {
    if (!shouldApply()) return;

    context.clearRect(0, 0, size, size);
    drawProgressUnderlay();
    drawQueueBadges();
    publishProgressFavicon();
  };
  image.src = defaultFaviconHref;
}
