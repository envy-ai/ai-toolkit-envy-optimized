import { lstat, open, rename, unlink } from 'fs/promises';
import { randomUUID } from 'crypto';

export function errorHasCode(error: unknown, code: string) {
  return error instanceof Error && 'code' in error && error.code === code;
}

export async function rejectSymlink(filename: string, label: string) {
  const info = await lstat(filename).catch(error => {
    if (errorHasCode(error, 'ENOENT')) return null;
    throw error;
  });
  if (info?.isSymbolicLink()) throw new Error(`${label} must not be a symbolic link.`);
}

export async function writeFileAtomic(filename: string, bytes: Uint8Array) {
  const temporary = `${filename}.${randomUUID()}.tmp`;
  try {
    const file = await open(temporary, 'wx', 0o600);
    try {
      await file.writeFile(bytes);
      await file.sync();
    } finally {
      await file.close();
    }
    await rename(temporary, filename);
  } finally {
    await unlink(temporary).catch(error => {
      if (!errorHasCode(error, 'ENOENT')) throw error;
    });
  }
}
