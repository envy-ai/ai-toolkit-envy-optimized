export interface TrainingMapGrid {
  w: number;
  h: number;
  v: number[];
}

export type TrainingMapKind = 'error' | 'improvement' | 'effect';
export interface TrainingMapProbe {
  key: string;
  image_path: string;
  source_path?: string;
  fingerprint?: string;
  image_url?: string;
  caption: string;
  width: number;
  height: number;
  crop?: number[];
}

export interface TrainingMapManifest {
  version: 1;
  status: 'ready' | 'error';
  error?: { message: string; stack: string } | null;
  probes: TrainingMapProbe[];
  levels: number[];
  steps: number[];
  scales: Record<string, Record<string, Record<TrainingMapKind, number>>>;
}

export interface TrainingMapFrame {
  step: number;
  maps: Record<string, Record<string, Partial<Record<TrainingMapKind, TrainingMapGrid>>>>;
}

export interface TrainingMapsResponse {
  available: boolean;
  manifest?: TrainingMapManifest;
  frame?: TrainingMapFrame | null;
}

/** A fixed scale across the frames loaded for a probe/level, so fading means a real change. */
export function trainingMapScale(grids: TrainingMapGrid[]): number {
  let scale = 0;
  for (const grid of grids) for (const value of grid.v) scale = Math.max(scale, Math.abs(value));
  return scale;
}

export function trainingMapColor(value: number, scale: number, signed: boolean): string {
  if (scale === 0 || value === 0) return 'transparent';
  const share = Math.abs(value) / scale;
  if (signed) return value > 0 ? `rgba(34,197,94,${share * 0.85})` : `rgba(232,67,162,${share * 0.85})`;
  const [r, g, b] = share < 0.33 ? [59, 130, 246] : share < 0.66 ? [250, 204, 21] : [239, 68, 68];
  return `rgba(${r},${g},${b},${share * 0.85})`;
}
