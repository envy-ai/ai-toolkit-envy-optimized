import fs from 'fs/promises';
import path from 'path';
import type { TrainingMapFrame, TrainingMapGrid, TrainingMapManifest } from '../utils/trainingMaps';

function object(value: unknown): value is Record<string, any> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function validStep(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

export function parseTrainingMapStep(raw: string | null): number | null {
  if (raw === null) return null;
  if (!/^\d+$/.test(raw) || !validStep(Number(raw))) throw new Error('Step must be a nonnegative safe integer.');
  return Number(raw);
}

export function validateTrainingMapManifest(value: unknown): TrainingMapManifest {
  if (!object(value) || value.version !== 1 || !['ready', 'error'].includes(value.status)
      || !Array.isArray(value.probes) || !Array.isArray(value.levels) || !Array.isArray(value.steps) || !object(value.scales))
    throw new Error('Invalid training map manifest.');
  if (!value.levels.length || !value.levels.every((v: unknown) => typeof v === 'number' && Number.isFinite(v) && v > 0 && v < 1)
      || !value.steps.every(validStep) || new Set(value.steps).size !== value.steps.length
      || value.steps.some((step: number, i: number) => i > 0 && step <= value.steps[i - 1]))
    throw new Error('Invalid training map levels or steps.');
  if (value.status === 'error' && (!object(value.error) || typeof value.error.message !== 'string' || typeof value.error.stack !== 'string'))
    throw new Error('Training map error details are missing.');
  const keys = new Set<string>();
  for (const probe of value.probes) {
    if (!object(probe) || typeof probe.key !== 'string' || !probe.key || keys.has(probe.key)
        || typeof probe.image_path !== 'string' || !probe.image_path || typeof probe.caption !== 'string'
        || !Number.isSafeInteger(probe.width) || probe.width < 1 || !Number.isSafeInteger(probe.height) || probe.height < 1)
      throw new Error('Invalid training map probe.');
    keys.add(probe.key);
    if (value.steps.length) for (const level of value.levels) {
      const scales = value.scales[probe.key]?.[String(level)];
      if (!object(scales)) throw new Error('Training map scales are missing.');
      for (const kind of ['error', 'effect', ...(value.steps.length > 1 ? ['improvement'] : [])]) {
        if (typeof scales[kind] !== 'number' || !Number.isFinite(scales[kind]) || scales[kind] < 0)
          throw new Error('Invalid training map scale.');
      }
    }
  }
  return value as unknown as TrainingMapManifest;
}

function validateGrid(value: unknown): asserts value is TrainingMapGrid {
  if (!object(value) || !Number.isSafeInteger(value.w) || value.w < 1
      || !Number.isSafeInteger(value.h) || value.h < 1 || !Array.isArray(value.v)
      || value.v.length !== value.w * value.h || !value.v.every((v: unknown) => typeof v === 'number' && Number.isFinite(v)))
    throw new Error('Invalid training map grid.');
}

export function validateTrainingMapFrame(value: unknown, manifest: TrainingMapManifest, step: number): TrainingMapFrame {
  if (!object(value) || value.step !== step || !object(value.maps)) throw new Error('Invalid training map frame.');
  for (const probe of manifest.probes) {
    const levels = value.maps[probe.key];
    if (!object(levels)) throw new Error(`Training map probe ${probe.key} is missing from step ${step}.`);
    for (const level of manifest.levels) {
      const maps = levels[String(level)];
      if (!object(maps) || !maps.error || !maps.effect || (step !== manifest.steps[0] && !maps.improvement))
        throw new Error(`Training map level ${level} is incomplete at step ${step}.`);
      for (const kind of ['error', 'improvement', 'effect']) if (maps[kind] !== undefined) validateGrid(maps[kind]);
    }
  }
  return value as unknown as TrainingMapFrame;
}

/** A missing manifest is a run without maps; a corrupt or incomplete recorded frame is an error. */
export async function readTrainingMaps(directory: string, requestedStep: number | null) {
  let text: string;
  try {
    text = await fs.readFile(path.join(directory, 'manifest.json'), 'utf8');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { available: false as const };
    throw error;
  }
  const manifest = validateTrainingMapManifest(JSON.parse(text));
  const step = requestedStep ?? manifest.steps.at(-1);
  if (step === undefined) return { available: true as const, manifest, frame: null };
  if (!manifest.steps.includes(step)) throw new Error(`Training map step ${step} is not recorded.`);
  const filename = `step-${String(step).padStart(9, '0')}.json`;
  const frame = validateTrainingMapFrame(JSON.parse(await fs.readFile(path.join(directory, filename), 'utf8')), manifest, step);
  return { available: true as const, manifest, frame };
}
