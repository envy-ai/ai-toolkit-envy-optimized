'use client';
import { useEffect, useRef, useState } from 'react';
import { MdAdd, MdDeleteOutline, MdEdit, MdRefresh } from 'react-icons/md';
import type { SessionFailure } from '@/assistant/AssistantSessionController';
import type { SessionMetadata } from '@/assistant/sessions';

type Props = {
  sessions: SessionMetadata[];
  activeId: string;
  disabled: boolean;
  refreshing: boolean;
  error: SessionFailure | null;
  onOpen: (id: string) => void;
  onNew: () => void;
  onRename: (session: SessionMetadata) => void;
  onDelete: (session: SessionMetadata) => void;
  onRefresh: () => void;
};
const button =
  'rounded p-1.5 hover:bg-gray-700 focus-visible:outline focus-visible:outline-blue-400 disabled:opacity-40';
export default function AssistantHistory({
  sessions,
  activeId,
  disabled,
  refreshing,
  error,
  onOpen,
  onNew,
  onRename,
  onDelete,
  onRefresh,
}: Props) {
  const [query, setQuery] = useState('');
  const search = useRef<HTMLInputElement>(null);
  useEffect(() => {
    search.current?.focus();
  }, []);
  const matching = sessions.filter(session => session.title.toLowerCase().includes(query.toLowerCase().trim()));
  return (
    <section
      id="assistant-history"
      aria-label="Saved chats"
      className="shrink-0 border-b border-gray-700 bg-gray-950/40 p-3"
    >
      <div className="flex items-center justify-between gap-2 text-sm">
        <h3 className="font-medium">Saved chats</h3>
        <div className="flex gap-1">
          <button
            type="button"
            aria-label="Refresh saved chats"
            title="Refresh saved chats"
            disabled={refreshing}
            className={button}
            onClick={onRefresh}
          >
            <MdRefresh className={refreshing ? 'animate-spin' : ''} />
          </button>
          <button type="button" disabled={disabled} className={`${button} flex items-center gap-1`} onClick={onNew}>
            <MdAdd /> New chat
          </button>
        </div>
      </div>
      <label htmlFor="assistant-history-search" className="mb-1 mt-2 block text-xs text-gray-400">
        Search chats
      </label>
      <input
        id="assistant-history-search"
        ref={search}
        type="search"
        value={query}
        onChange={event => setQuery(event.target.value)}
        className="w-full rounded border border-gray-700 bg-gray-950 px-2 py-1.5 text-sm focus:border-blue-400 focus:outline-none"
      />
      {error && (
        <div role="alert" className="mt-2 text-xs text-red-300">
          <p>{error.message}</p>
          <details>
            <summary>Error details</summary>
            <pre className="overflow-auto whitespace-pre-wrap">{error.stack}</pre>
          </details>
          <button type="button" className="underline" onClick={onRefresh}>
            Retry history
          </button>
        </div>
      )}
      <div className="mt-2 max-h-56 overflow-y-auto" aria-label="Chat history">
        {!matching.length && (
          <p className="p-2 text-xs text-gray-500">{query ? 'No chats match your search.' : 'No saved chats yet.'}</p>
        )}
        {matching.map(session => (
          <div
            key={session.id}
            className={`mb-1 flex items-center gap-1 rounded ${session.id === activeId ? 'bg-blue-950' : 'hover:bg-gray-800'}`}
          >
            <button
              type="button"
              disabled={disabled}
              aria-current={session.id === activeId ? 'true' : undefined}
              className="min-w-0 flex-1 rounded px-2 py-2 text-left focus-visible:outline focus-visible:outline-blue-400 disabled:opacity-40"
              onClick={() => onOpen(session.id)}
            >
              <span className="block truncate text-sm" title={session.title}>
                {session.title}
              </span>
              <span className="block text-xs text-gray-500">
                {new Date(session.updatedAt).toLocaleDateString()} · {session.entryCount} messages
              </span>
            </button>
            <button
              type="button"
              disabled={disabled}
              aria-label={`Rename ${session.title}`}
              title="Rename chat"
              className={button}
              onClick={() => onRename(session)}
            >
              <MdEdit />
            </button>
            <button
              type="button"
              disabled={disabled}
              aria-label={`Delete ${session.title}`}
              title="Delete chat"
              className={button}
              onClick={() => onDelete(session)}
            >
              <MdDeleteOutline />
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}
