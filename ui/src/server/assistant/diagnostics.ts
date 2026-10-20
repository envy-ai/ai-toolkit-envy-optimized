import { constants } from 'node:fs';
import { open, rename, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { errorHasCode, rejectSymlink } from './storage';

const MAX_LOG_BYTES = 5 * 1024 * 1024;

/** Metadata only: callers must never supply provider inputs, outputs or credentials. */
export type AssistantDiagnosticSink = (event: Record<string, unknown>) => Promise<void>;

/** The service opts in; standalone providers and mocked transports do not write user logs. */
export function createAssistantDiagnosticSink(directory: string): AssistantDiagnosticSink {
  const filename = join(directory, 'assistant-diagnostics.jsonl');
  const archive = filename + '.1';
  let writes = Promise.resolve();
  return event => {
    const operation = writes.then(async () => {
      try {
        await rejectSymlink(directory, 'assistantDiagnosticsDirectory');
        await rejectSymlink(filename, 'assistantDiagnostics');
        const line = JSON.stringify({ ...event, version: 1, timestamp: new Date().toISOString() }) + '\n';
        if (Buffer.byteLength(line) > MAX_LOG_BYTES)
          throw new Error('Assistant diagnostic event exceeds the log size limit.');
        const existing = await open(
          filename,
          constants.O_WRONLY | constants.O_APPEND | constants.O_CREAT | constants.O_NOFOLLOW,
          0o600,
        );
        let rotate = false;
        try {
          const info = await existing.stat();
          if (!info.isFile()) throw new Error('Assistant diagnostics must be a regular file.');
          await existing.chmod(0o600);
          rotate = info.size + Buffer.byteLength(line) > MAX_LOG_BYTES;
          if (!rotate) await existing.writeFile(line);
        } finally {
          await existing.close();
        }
        if (!rotate) return;
        await rejectSymlink(archive, 'assistantDiagnosticsArchive');
        await unlink(archive).catch(error => {
          if (!errorHasCode(error, 'ENOENT')) throw error;
        });
        await rename(filename, archive);
        const file = await open(
          filename,
          constants.O_WRONLY | constants.O_APPEND | constants.O_CREAT | constants.O_NOFOLLOW,
          0o600,
        );
        try {
          if (!(await file.stat()).isFile()) throw new Error('Assistant diagnostics must be a regular file.');
          await file.chmod(0o600);
          await file.writeFile(line);
        } finally {
          await file.close();
        }
      } catch (error) {
        console.error('[assistant-diagnostics] Log write failed.', error);
        throw new Error('Assistant diagnostic log write failed.', { cause: error });
      }
    });
    // Keep subsequent writes available; the caller still receives the original rejection.
    writes = operation.then(
      () => undefined,
      () => undefined,
    );
    return operation;
  };
}
