import { chmod, lstat, mkdir, readFile, readdir, unlink } from 'node:fs/promises';
import path from 'node:path';
import { z } from 'zod';
import {
  assistantSessionSchema,
  automaticSessionTitle,
  hasSessionContent,
  sessionContentSchema,
  sessionIdSchema,
  sessionUpdateSchema,
  type AssistantSession,
  type SessionContent,
  type SessionMetadata,
  type SessionUpdate,
} from '@/assistant/sessions';
import { errorHasCode, rejectSymlink, writeFileAtomic } from './storage';

export const MAX_SESSION_BYTES = 8 * 1024 * 1024;
const tombstoneSchema = z
  .object({ version: z.literal(1), id: sessionIdSchema, deletedAt: z.string().datetime() })
  .strict();
export class SessionStoreError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}
const metadata = ({ version: _version, content: _content, ...value }: AssistantSession): SessionMetadata => value;

/** One Next.js writer, with serialized revision checks for concurrent browser tabs. */
export class AssistantSessionStore {
  readonly #index = new Map<string, SessionMetadata>();
  readonly #deleted = new Set<string>();
  #writes = Promise.resolve();
  private constructor(readonly directory: string) {}
  static async create(directory: string) {
    await mkdir(directory, { recursive: true, mode: 0o700 });
    await rejectSymlink(directory, 'Assistant sessions directory');
    await chmod(directory, 0o700);
    const store = new AssistantSessionStore(directory);
    const files = await readdir(directory);
    for (const filename of files.filter(name => name.endsWith('.deleted.json'))) {
      const value = tombstoneSchema.parse(await store.#readJson(path.join(directory, filename)));
      if (filename !== `${value.id}.deleted.json`)
        throw new Error('Assistant session tombstone ID does not match its filename.');
      store.#deleted.add(value.id);
    }
    for (const filename of files.filter(name => name.endsWith('.json') && !name.endsWith('.deleted.json'))) {
      const value = assistantSessionSchema.parse(await store.#readJson(path.join(directory, filename)));
      if (filename !== `${value.id}.json`) throw new Error('Assistant session ID does not match its filename.');
      if (value.entryCount !== value.content.entries.length)
        throw new Error('Assistant session entry count is inconsistent.');
      if (!store.#deleted.has(value.id)) store.#index.set(value.id, metadata(value));
    }
    return store;
  }
  async #readJson(filename: string): Promise<unknown> {
    await rejectSymlink(filename, 'Assistant session');
    const info = await lstat(filename);
    if (!info.isFile() || info.size > MAX_SESSION_BYTES) throw new Error('Invalid assistant session file or size.');
    if (process.platform !== 'win32' && (info.uid !== process.getuid?.() || (info.mode & 0o077) !== 0))
      throw new Error('Assistant session files must belong to the current user with owner-only permissions.');
    return JSON.parse(await readFile(filename, 'utf8'));
  }
  #serialize<T>(operation: () => Promise<T>): Promise<T> {
    const pending = this.#writes.then(operation);
    this.#writes = pending.then(
      () => undefined,
      () => undefined,
    );
    return pending;
  }
  #filename(id: string) {
    return path.join(this.directory, `${sessionIdSchema.parse(id)}.json`);
  }
  async #get(id: string): Promise<AssistantSession> {
    sessionIdSchema.parse(id);
    if (this.#deleted.has(id) || !this.#index.has(id))
      throw new SessionStoreError('Chat was deleted or does not exist.', 404);
    return assistantSessionSchema.parse(await this.#readJson(this.#filename(id)));
  }
  async #write(session: AssistantSession) {
    const bytes = Buffer.from(JSON.stringify(assistantSessionSchema.parse(session)));
    if (bytes.length > MAX_SESSION_BYTES)
      throw new SessionStoreError('Chat exceeds the 8 MiB save limit. Your current text has been retained.', 413);
    await rejectSymlink(this.directory, 'Assistant sessions directory');
    const filename = this.#filename(session.id);
    await rejectSymlink(filename, 'Assistant session');
    await writeFileAtomic(filename, bytes);
    this.#index.set(session.id, metadata(session));
    return session;
  }
  list(search = '') {
    return this.#serialize(async () =>
      [...this.#index.values()]
        .filter(item => item.title.toLowerCase().includes(search.toLowerCase().trim()))
        .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt) || b.id.localeCompare(a.id))
        .map(item => ({ ...item })),
    );
  }
  get(id: string) {
    return this.#serialize(() => this.#get(id));
  }
  create(id: string, value: SessionContent) {
    return this.#serialize(async () => {
      sessionIdSchema.parse(id);
      const content = sessionContentSchema.parse(value);
      if (this.#deleted.has(id)) throw new SessionStoreError('This chat was deleted. Start a new chat.', 410);
      if (this.#index.has(id)) return this.#get(id); // Retry after an acknowledged write was lost.
      if (!hasSessionContent(content)) throw new SessionStoreError('Empty chats are not saved.', 400);
      const now = new Date().toISOString();
      return this.#write({
        version: 1,
        id,
        title: automaticSessionTitle(content),
        titleSource: 'auto',
        createdAt: now,
        updatedAt: now,
        revision: 1,
        entryCount: content.entries.length,
        content,
      });
    });
  }
  update(id: string, value: SessionUpdate) {
    return this.#serialize(async () => {
      const update = sessionUpdateSchema.parse(value);
      const previous = await this.#get(id);
      if (update.revision !== previous.revision)
        throw new SessionStoreError(
          'This chat changed in another tab. Your local text is retained; reload the saved chat to resolve the conflict.',
          409,
        );
      const content = update.content ?? previous.content;
      return this.#write({
        ...previous,
        content,
        entryCount: content.entries.length,
        title: update.title ?? (previous.titleSource === 'auto' ? automaticSessionTitle(content) : previous.title),
        titleSource: update.title !== undefined ? 'manual' : previous.titleSource,
        revision: previous.revision + 1,
        updatedAt: new Date().toISOString(),
      });
    });
  }
  delete(id: string, revision: number) {
    return this.#serialize(async () => {
      sessionIdSchema.parse(id);
      if (this.#deleted.has(id)) return;
      const previous = await this.#get(id);
      if (revision !== previous.revision)
        throw new SessionStoreError('This chat changed in another tab. Refresh history before deleting it.', 409);
      await rejectSymlink(this.directory, 'Assistant sessions directory');
      const tombstone = path.join(this.directory, `${id}.deleted.json`);
      await rejectSymlink(tombstone, 'Assistant session tombstone');
      await writeFileAtomic(
        tombstone,
        Buffer.from(JSON.stringify({ version: 1, id, deletedAt: new Date().toISOString() })),
      );
      this.#deleted.add(id);
      this.#index.delete(id);
      await unlink(this.#filename(id)).catch(error => {
        if (!errorHasCode(error, 'ENOENT')) throw error;
      });
    });
  }
}
