import { randomUUID } from 'node:crypto';
import { lstat, mkdir, open, readFile, rename, unlink } from 'node:fs/promises';
import path from 'node:path';
import { apiCaptionSettingsSchema, assertNoCaptionCredentials, captionApiEndpoint } from './captionApiConfig';

export const captionCredentialDirectory = (root: string) =>
  path.resolve(process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR || path.join(root, 'data', 'caption-api', 'credentials'));
const idPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
async function rejectLink(filename: string) {
  const info = await lstat(filename).catch(error => {
    if (error.code === 'ENOENT') return null;
    throw error;
  });
  if (info?.isSymbolicLink()) throw new Error('Private caption credential paths must not be symbolic links.');
}
export async function createCaptionApiCredential(root: string, apiUrl: string, key: string): Promise<string> {
  if (typeof key !== 'string' || !key.trim() || key.length > 8192 || /[\r\n]/.test(key))
    throw new Error('Enter a valid API key.');
  if (process.platform === 'win32')
    throw new Error('Owner-only caption credential persistence is unavailable on Windows.');
  const endpoint = captionApiEndpoint(apiUrl);
  const directory = captionCredentialDirectory(root);
  await mkdir(directory, { recursive: true, mode: 0o700 });
  await rejectLink(directory);
  const id = randomUUID();
  const temporary = path.join(directory, `${id}.tmp`);
  const file = await open(temporary, 'wx', 0o600);
  try {
    await file.writeFile(JSON.stringify({ version: 1, endpoint, key: key.trim() }));
    await file.sync();
  } finally {
    await file.close();
  }
  try {
    await rename(temporary, path.join(directory, `${id}.json`));
  } finally {
    await unlink(temporary).catch(error => {
      if (error.code !== 'ENOENT') throw error;
    });
  }
  return id;
}
export async function readCaptionApiCredential(root: string, id: string, apiUrl: string): Promise<string> {
  if (!idPattern.test(id)) throw new Error('Invalid caption credential reference.');
  const directory = captionCredentialDirectory(root);
  const filename = path.join(directory, `${id}.json`);
  await rejectLink(directory);
  await rejectLink(filename);
  const info = await lstat(filename);
  if (
    !info.isFile() ||
    info.size > 32_768 ||
    process.platform === 'win32' ||
    (info.mode & 0o077) !== 0 ||
    info.uid !== process.getuid?.()
  )
    throw new Error('Caption credential file must belong to the current user with owner-only permissions.');
  let record: any;
  try {
    record = JSON.parse(await readFile(filename, 'utf8'));
  } catch {
    throw new Error('Caption credential file is missing or malformed.');
  }
  if (record.version !== 1 || typeof record.key !== 'string' || !record.key || /[\r\n]/.test(record.key))
    throw new Error('Caption credential file is invalid.');
  if (record.endpoint !== captionApiEndpoint(apiUrl))
    throw new Error('The saved key belongs to a different API URL. Enter the key again or choose no authentication.');
  return record.key;
}

/** The only credential handoff: inherited environment of the detached caption child. */
export async function apiCaptionChildEnvironment(
  root: string,
  process: { caption: unknown },
): Promise<Record<string, string>> {
  assertNoCaptionCredentials(process);
  const caption = apiCaptionSettingsSchema.parse(process.caption);
  return {
    CUDA_VISIBLE_DEVICES: '',
    AITK_API_CAPTION_KEY: caption.api_credential_id
      ? await readCaptionApiCredential(root, caption.api_credential_id, caption.api_url)
      : '',
  };
}
