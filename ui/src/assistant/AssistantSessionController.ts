import { v4 as uuidv4 } from 'uuid';
import {
  automaticSessionTitle,
  emptySessionContent,
  hasSessionContent,
  type AssistantSession,
  type SessionContent,
  type SessionMetadata,
  type SessionUpdate,
} from './sessions';

export type SessionApi = {
  list(): Promise<SessionMetadata[]>;
  get(id: string): Promise<AssistantSession>;
  create(id: string, content: SessionContent): Promise<AssistantSession>;
  update(id: string, update: SessionUpdate): Promise<AssistantSession>;
  delete(id: string, revision: number): Promise<void>;
};
export type SessionRecovery = { id: string; revision: number; content: SessionContent };
export type SessionHost = {
  lastId(): string | null;
  remember(id: string | null): void;
  legacy(): SessionContent | null;
  legacyId(): string | null;
  rememberLegacy(id: string): void;
  completeLegacy(): void;
  recovery(): SessionRecovery | null;
  recover(value: SessionRecovery | null): void;
};
export type SessionFailure = { message: string; stack?: string; conflict: boolean };
export type SessionState = {
  id: string;
  metadata: SessionMetadata | null;
  content: SessionContent;
  sessions: SessionMetadata[];
  ready: boolean;
  loading: boolean;
  saveState: 'idle' | 'unsaved' | 'saving' | 'saved' | 'error';
  error: SessionFailure | null;
  historyError: SessionFailure | null;
  refreshingHistory: boolean;
};
const describe = (cause: any): SessionFailure => ({
  message: cause.response?.data?.error || cause.message || 'Chat storage failed.',
  stack: cause.response?.data?.stack || cause.stack,
  conflict: [404, 409, 410].includes(cause.response?.status ?? cause.status),
});
const summary = ({ version: _version, content: _content, ...metadata }: AssistantSession): SessionMetadata => metadata;

/** Owns local snapshots and serialized saves; provider runs/approval callbacks never enter this state. */
export class AssistantSessionController {
  #state: SessionState;
  #listeners = new Set<() => void>();
  #initializing: Promise<void> | null = null;
  #saving: Promise<boolean> | null = null;
  #changed = 0;
  #saved = 0;
  #legacyPending = false;
  constructor(
    readonly api: SessionApi,
    readonly host: SessionHost,
    readonly newId = uuidv4,
  ) {
    this.#state = {
      id: newId(),
      metadata: null,
      content: emptySessionContent(),
      sessions: [],
      ready: false,
      loading: false,
      saveState: 'idle',
      error: null,
      historyError: null,
      refreshingHistory: false,
    };
  }
  snapshot = () => this.#state;
  subscribe = (listener: () => void) => {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  };
  #publish(update: Partial<SessionState>) {
    this.#state = { ...this.#state, ...update };
    this.#listeners.forEach(listener => listener());
  }
  #history(metadata: SessionMetadata) {
    return [...this.#state.sessions.filter(item => item.id !== metadata.id), metadata].sort(
      (a, b) => b.updatedAt.localeCompare(a.updatedAt) || b.id.localeCompare(a.id),
    );
  }
  #backup() {
    this.host.recover(
      this.#changed > this.#saved
        ? { id: this.#state.id, revision: this.#state.metadata?.revision ?? 0, content: this.#state.content }
        : null,
    );
  }
  #adopt(session: AssistantSession) {
    this.#changed = this.#saved = 0;
    const metadata = summary(session);
    this.host.remember(session.id);
    this.host.recover(null);
    this.#publish({
      id: session.id,
      metadata,
      content: session.content,
      sessions: this.#history(metadata),
      saveState: 'saved',
      error: null,
    });
  }
  initialize = (): Promise<void> => {
    if (this.#state.ready) return Promise.resolve();
    if (this.#initializing) return this.#initializing;
    this.#publish({ loading: true, error: null });
    this.#initializing = (async () => {
      try {
        const legacy = this.host.legacy();
        const recovery = this.host.recovery();
        const local = recovery?.content ?? legacy;
        let localId: string | null = null;
        if (local && hasSessionContent(local)) {
          localId = recovery?.id ?? this.host.legacyId() ?? this.newId();
          this.#changed = 1;
          this.#legacyPending = Boolean(legacy && (!recovery || this.host.legacyId() === localId));
          this.#publish({ id: localId, content: local, ready: true, saveState: 'unsaved' });
          if (!recovery) this.host.rememberLegacy(localId);
        }
        let sessions: SessionMetadata[] = [];
        try {
          sessions = await this.api.list();
          this.#publish({ sessions, historyError: null });
        } catch (cause) {
          this.#publish({ historyError: describe(cause) });
        }
        if (legacy && hasSessionContent(legacy) && !recovery) {
          const id = localId!;
          this.host.rememberLegacy(id); // Stable before first request, including lost responses.
          this.#legacyPending = true;
          this.#changed = 1;
          this.#publish({ id, content: legacy, metadata: null, ready: true, saveState: 'unsaved' });
          await this.flush();
        } else if (recovery) {
          this.#changed = 1;
          this.#legacyPending = Boolean(legacy && this.host.legacyId() === recovery.id);
          const now = new Date().toISOString();
          // Show the only unsaved copy before any network request can fail.
          this.#publish({
            id: recovery.id,
            content: recovery.content,
            ready: true,
            saveState: 'unsaved',
            metadata:
              recovery.revision > 0
                ? {
                    id: recovery.id,
                    title: automaticSessionTitle(recovery.content),
                    titleSource: 'auto',
                    revision: recovery.revision,
                    createdAt: now,
                    updatedAt: now,
                    entryCount: recovery.content.entries.length,
                  }
                : null,
          });
          if (recovery.revision > 0) {
            const saved = await this.api.get(recovery.id);
            if (JSON.stringify(saved.content) === JSON.stringify(recovery.content)) {
              if (this.#legacyPending) {
                this.host.completeLegacy();
                this.#legacyPending = false;
              }
              this.#adopt(saved);
              this.#publish({ ready: true });
              return;
            }
            this.#publish({ metadata: { ...summary(saved), revision: recovery.revision } });
          }
          await this.flush();
        } else {
          const last = this.host.lastId();
          if (last && sessions.some(item => item.id === last)) this.#adopt(await this.api.get(last));
          this.#publish({ ready: true });
        }
      } catch (cause) {
        this.#publish({ error: describe(cause), saveState: 'error' });
      } finally {
        this.#initializing = null;
        this.#publish({ loading: false });
      }
    })();
    return this.#initializing;
  };
  updateContent = (update: (previous: SessionContent) => SessionContent) => {
    if (!this.#state.ready) return;
    this.#changed += 1;
    this.#publish({ content: update(this.#state.content), saveState: this.#state.error ? 'error' : 'unsaved' });
    try {
      this.#backup();
    } catch (cause) {
      this.#publish({ error: describe(cause), saveState: 'error' });
    }
  };
  flush = (): Promise<boolean> => {
    if (this.#saving) return this.#saving;
    if (this.#state.error || !this.#state.ready) return Promise.resolve(false);
    if (!this.#state.metadata && !hasSessionContent(this.#state.content)) {
      this.#saved = this.#changed;
      this.#publish({ saveState: 'idle' });
      return Promise.resolve(true);
    }
    // Defer execution until after assignment, even a clean flush has no await.
    this.#saving = Promise.resolve().then(async () => {
      try {
        while (this.#changed !== this.#saved) {
          const version = this.#changed;
          const { id, metadata, content } = this.#state;
          this.#publish({ saveState: 'saving' });
          const session = metadata
            ? await this.api.update(id, { revision: metadata.revision, content })
            : await this.api.create(id, content);
          if (!metadata && JSON.stringify(session.content) !== JSON.stringify(content))
            throw Object.assign(
              new Error(
                'The saved chat differs from this recovered copy. Save a copy or reload the saved chat to resolve it.',
              ),
              { status: 409 },
            );
          this.#saved = version;
          const nextMetadata = summary(session);
          this.#publish({ metadata: nextMetadata, sessions: this.#history(nextMetadata) });
          this.host.remember(id);
          this.#backup();
        }
        this.#backup();
        if (this.#legacyPending) {
          this.host.completeLegacy();
          this.#legacyPending = false;
        }
        this.#publish({ saveState: 'saved' });
        return true;
      } catch (cause) {
        try {
          this.#backup();
        } catch (backupError) {
          console.error('Chat recovery backup failed:', backupError);
        }
        this.#publish({ error: describe(cause), saveState: 'error' });
        return false;
      } finally {
        this.#saving = null;
      }
    });
    return this.#saving;
  };
  retry = async () => {
    if (!this.#state.ready) return this.initialize();
    this.#publish({ error: null, saveState: 'unsaved' });
    return this.flush(); // Revision is deliberately unchanged on conflict.
  };
  refreshHistory = async () => {
    if (this.#state.refreshingHistory) return;
    this.#publish({ refreshingHistory: true, historyError: null });
    try {
      this.#publish({ sessions: await this.api.list() });
    } catch (cause) {
      this.#publish({ historyError: describe(cause) });
    } finally {
      this.#publish({ refreshingHistory: false });
    }
  };
  async #transition(operation: () => Promise<boolean>, flushFirst = true) {
    if (this.#state.loading) return false;
    this.#publish({ loading: true });
    try {
      if (flushFirst && !(await this.flush())) return false;
      return await operation();
    } catch (cause) {
      this.#publish({ error: describe(cause) });
      return false;
    } finally {
      this.#publish({ loading: false });
    }
  }
  newChat = async () => {
    return this.#transition(async () => {
      this.#changed = this.#saved = 0;
      this.host.remember(null);
      this.host.recover(null);
      this.#publish({
        id: this.newId(),
        metadata: null,
        content: emptySessionContent(),
        saveState: 'idle',
        error: null,
      });
      return true;
    });
  };
  open = async (id: string, discardLocal = false) => {
    return this.#transition(async () => {
      this.#adopt(await this.api.get(id));
      return true;
    }, !discardLocal);
  };
  rename = async (id: string, title: string) => {
    return this.#transition(async () => {
      const metadata = this.#state.sessions.find(item => item.id === id);
      if (!metadata) throw new Error('Chat no longer appears in history.');
      const session = await this.api.update(id, { revision: metadata.revision, title });
      this.#publish({
        sessions: this.#history(summary(session)),
        ...(this.#state.id === id ? { metadata: summary(session) } : {}),
      });
      return true;
    });
  };
  delete = async (id: string) => {
    return this.#transition(async () => {
      const metadata = this.#state.sessions.find(item => item.id === id);
      if (!metadata) throw new Error('Chat no longer appears in history.');
      await this.api.delete(id, metadata.revision);
      this.#publish({ sessions: this.#state.sessions.filter(item => item.id !== id) });
      if (this.#state.id === id) {
        this.#changed = this.#saved = 0;
        this.host.remember(null);
        this.host.recover(null);
        this.#publish({
          id: this.newId(),
          metadata: null,
          content: emptySessionContent(),
          saveState: 'idle',
          error: null,
        });
      }
      return true;
    });
  };
  saveCopy = async () => {
    if (this.#state.loading || this.#saving) return false;
    this.#changed = 1;
    this.#saved = 0;
    this.#publish({ id: this.newId(), metadata: null, error: null, saveState: 'unsaved' });
    return this.flush();
  };
  title = () => this.#state.metadata?.title ?? automaticSessionTitle(this.#state.content);
}
