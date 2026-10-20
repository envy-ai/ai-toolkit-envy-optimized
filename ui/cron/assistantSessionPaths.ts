import path from 'node:path';
import { realpath } from 'node:fs/promises';

const under = (filename: string, directory: string) =>
  filename === directory || filename.startsWith(directory + path.sep);
const canonical = async (filename: string): Promise<string | null> =>
  realpath(filename).catch(error => {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') return null;
    throw error;
  });
/** Never expose saved conversations through public file/media endpoints, including symlink aliases. */
export async function isAssistantSessionPath(filename: string, toolkitRoot: string): Promise<boolean> {
  const directory = path.resolve(
    process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR || path.join(toolkitRoot, 'data', 'assistant'),
    'sessions',
  );
  const resolved = path.resolve(filename);
  if (under(resolved, directory)) return true;
  const [actualFile, actualDirectory] = await Promise.all([canonical(resolved), canonical(directory)]);
  return Boolean(
    actualFile && (under(actualFile, directory) || (actualDirectory && under(actualFile, actualDirectory))),
  );
}
