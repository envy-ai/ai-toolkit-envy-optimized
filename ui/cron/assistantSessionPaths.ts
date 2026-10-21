import path from 'node:path';
import { realpath } from 'node:fs/promises';

const under = (filename: string, directory: string) =>
  filename === directory || filename.startsWith(directory + path.sep);
const canonical = async (filename: string): Promise<string | null> =>
  realpath(filename).catch(error => {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return null;
    throw error;
  });
/** Never expose conversations or caption credentials through public file/media endpoints, including symlink aliases. */
export async function isAssistantSessionPath(filename: string, toolkitRoot: string): Promise<boolean> {
  const directory = path.resolve(
    process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR || path.join(toolkitRoot, 'data', 'assistant'),
    'sessions',
  );
  const credentials = path.resolve(
    process.env.AI_TOOLKIT_CAPTION_CREDENTIALS_DIR || path.join(toolkitRoot, 'data', 'caption-api', 'credentials'),
  );
  const directories = [directory, credentials];
  const resolved = path.resolve(filename);
  if (directories.some(privateRoot => under(resolved, privateRoot))) return true;
  const [actualFile, ...actualDirectories] = await Promise.all([canonical(resolved), ...directories.map(canonical)]);
  return Boolean(
    actualFile &&
      directories.some(
        (privateRoot, index) =>
          under(actualFile, privateRoot) || (actualDirectories[index] && under(actualFile, actualDirectories[index]!)),
      ),
  );
}
