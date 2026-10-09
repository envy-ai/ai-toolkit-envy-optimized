import { randomUUID } from 'node:crypto';
import { lstat, mkdir, readFile, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { z } from 'zod';
import { writeFileAtomic, rejectSymlink, errorHasCode } from './storage';
import {
  assistantConfigurationSchema,
  DEFAULT_ASSISTANT_CONFIGURATION,
  type AssistantConfiguration,
  type AssistantConfigurationState,
  type AssistantConfigurationUpdate,
} from '@/assistant/AssistantProvider';

const updateSchema = z
  .object({
    config: assistantConfigurationSchema,
    apiKey: z
      .string()
      .trim()
      .min(1)
      .max(8192)
      .refine(key => !/[\r\n]/.test(key), 'API key cannot contain newlines.')
      .optional(),
    persistCredential: z.boolean().optional(),
    clearCredential: z.boolean().optional(),
  })
  .strict();
const storedConfigSchema = z.object({ version: z.literal(1), config: assistantConfigurationSchema }).strict();
const credentialSchema = z.object({ version: z.literal(1), key: updateSchema.shape.apiKey.unwrap() }).strict();

/** Credentials never enter editor preferences or project data. */
export class NodeAssistantConfiguration {
  #config: AssistantConfiguration = { ...DEFAULT_ASSISTANT_CONFIGURATION };
  #key: string | undefined;
  #credentialId: string | null = null;
  #persisted = false;
  #revision = 1;
  readonly #directory?: string;
  #writes = Promise.resolve();
  constructor(settingsDirectory?: string) {
    this.#directory = settingsDirectory;
  }
  static async create(settingsDirectory?: string): Promise<NodeAssistantConfiguration> {
    const store = new NodeAssistantConfiguration(settingsDirectory);
    if (settingsDirectory !== undefined) {
      await mkdir(settingsDirectory, { recursive: true, mode: 0o700 });
      await rejectSymlink(settingsDirectory, 'assistantSettings');
      const config = await store.#read('assistant.json');
      if (config !== null) store.#config = storedConfigSchema.parse(config).config;
      const credential = await store.#read('assistant-credential.json', true);
      if (credential !== null) {
        store.#key = credentialSchema.parse(credential).key;
        store.#credentialId = randomUUID();
        store.#persisted = true;
      }
    }
    return store;
  }
  async #read(name: string, secret = false): Promise<unknown | null> {
    const path = join(this.#directory!, name);
    try {
      await rejectSymlink(path, 'assistantSettings');
      const info = await lstat(path);
      if (!info.isFile() || info.size > 64 * 1024) throw new Error(`Invalid assistant settings file: ${name}.`);
      if (secret && process.platform === 'win32')
        throw new Error(
          'Owner-only credential file storage is unavailable on this Windows host; remove the credential file and use process memory.',
        );
      if (secret && process.platform !== 'win32' && ((info.mode & 0o077) !== 0 || info.uid !== process.getuid?.())) {
        throw new Error('Assistant credential file must belong to the current user and have owner-only permissions.');
      }
      const bytes = await readFile(path, 'utf8');
      try {
        return JSON.parse(bytes) as unknown;
      } catch (error) {
        if (!secret) throw error;
        // JSON parser messages can quote malformed secret bytes. Preserve location,
        // but never expose that message or its source snippet to the renderer.
        const sanitized = new Error('Assistant credential file contains malformed JSON.');
        if (error instanceof Error && error.stack)
          sanitized.stack = sanitized.message + '\n' + error.stack.split('\n').slice(1).join('\n');
        throw sanitized;
      }
    } catch (error) {
      if (errorHasCode(error, 'ENOENT')) return null;
      throw error;
    }
  }
  state(): AssistantConfigurationState {
    return {
      config: { ...this.#config },
      revision: this.#revision,
      credentialPresent: this.#key !== undefined,
      credentialPersisted: this.#persisted,
      credentialId: this.#credentialId,
    };
  }
  /** Host-internal only; never expose this class through the renderer bridge. */
  key(): string | undefined {
    return this.#key;
  }
  async update(value: AssistantConfigurationUpdate): Promise<AssistantConfigurationState> {
    const operation = this.#writes.then(async () => {
      const update = updateSchema.parse(value);
      if (update.clearCredential && update.apiKey !== undefined)
        throw new Error('Cannot set and clear the credential together.');
      const key = update.clearCredential ? undefined : (update.apiKey ?? this.#key);
      const persist = update.clearCredential
        ? false
        : (update.persistCredential ?? (update.apiKey !== undefined ? false : this.#persisted));
      if (persist && key === undefined) throw new Error('No API key is available to persist.');
      if (persist && this.#directory === undefined)
        throw new Error('Credential persistence is unavailable without a host settings directory.');
      if (persist && process.platform === 'win32')
        throw new Error(
          'Owner-only credential file storage is not supported by this Windows host; keep the key in process memory.',
        );
      if (this.#directory !== undefined) {
        const credentialPath = join(this.#directory, 'assistant-credential.json');
        await rejectSymlink(credentialPath, 'assistantCredential');
        if (persist) {
          await writeFileAtomic(credentialPath, new TextEncoder().encode(JSON.stringify({ version: 1, key })));
        } else {
          try {
            await unlink(credentialPath);
          } catch (error) {
            if (!errorHasCode(error, 'ENOENT')) throw error;
          }
        }
        const configPath = join(this.#directory, 'assistant.json');
        await rejectSymlink(configPath, 'assistantConfiguration');
        await writeFileAtomic(
          configPath,
          new TextEncoder().encode(JSON.stringify({ version: 1, config: update.config })),
        );
      }
      this.#key = key;
      if (update.apiKey !== undefined || update.clearCredential)
        this.#credentialId = key === undefined ? null : randomUUID();
      this.#persisted = persist && key !== undefined;
      this.#config = update.config;
      this.#revision += 1;
      return this.state();
    });
    this.#writes = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  }
}
