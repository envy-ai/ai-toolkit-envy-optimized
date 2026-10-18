import type { DatasetConfig } from '@/types';

/** Preserve source order, including incomplete slots while the form is being edited. */
export function getDatasetControlPaths(dataset: DatasetConfig): string[] {
  const paths = dataset.control_path;
  return Array.isArray(paths) ? paths : paths ? [paths] : [];
}

export function setDatasetControlPath(dataset: DatasetConfig, index: number, value: string): string[] {
  const paths = [...getDatasetControlPaths(dataset)];
  while (paths.length <= index) paths.push('');
  paths[index] = value;
  while (paths.at(-1) === '') paths.pop();
  return paths;
}
