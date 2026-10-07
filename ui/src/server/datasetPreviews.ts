import fs from 'fs/promises';
import path from 'path';

const imageExtensions = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.avif']);

/** Stop at the first image in a sorted traversal; skip cache/control directories. */
export async function findFirstDatasetImage(directory: string): Promise<string | null> {
  const entries = (await fs.readdir(directory, { withFileTypes: true }))
    .filter(entry => !entry.name.startsWith('.') && entry.name !== '_controls')
    .sort((a, b) => a.name.localeCompare(b.name));
  for (const entry of entries) {
    const filename = path.join(directory, entry.name);
    if (entry.isFile() && imageExtensions.has(path.extname(entry.name).toLowerCase())) return filename;
    if (entry.isDirectory()) {
      const nested = await findFirstDatasetImage(filename).catch(() => null);
      if (nested) return nested;
    }
  }
  return null;
}

export interface DatasetPreview {
  name: string;
  first_image: string | null;
}

export async function getDatasetPreviews(root: string, names: string[]): Promise<DatasetPreview[]> {
  const results: DatasetPreview[] = new Array(names.length);
  let cursor = 0;
  // Limit filesystem concurrency when the datasets root contains many folders.
  await Promise.all(
    Array.from({ length: Math.min(8, names.length) }, async () => {
      while (cursor < names.length) {
        const index = cursor++;
        const name = names[index];
        results[index] = { name, first_image: await findFirstDatasetImage(path.join(root, name)).catch(() => null) };
      }
    }),
  );
  return results;
}
