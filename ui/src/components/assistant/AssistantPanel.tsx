'use client';

import { useEffect, useRef, useState, type CSSProperties, type PointerEvent, type SetStateAction } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import { v4 as uuidv4 } from 'uuid';
import {
  MdClose,
  MdSettings,
  MdSmartToy,
  MdStop,
  MdSend,
  MdAttachFile,
  MdDeleteOutline,
  MdChevronRight,
  MdHistory,
  MdAdd,
  MdEdit,
} from 'react-icons/md';
import { apiClient } from '@/utils/api';
import type {
  AssistantConfigurationState,
  AssistantToolCall,
  AssistantTurnResult,
} from '@/assistant/AssistantProvider';
import { isReadOnlyTool } from '@/assistant/tools';
import AssistantSettings from './AssistantSettings';
import { useAssistant } from './AssistantContext';
import useAssistantSessions from '@/hooks/useAssistantSessions';
import type { AssistantEntry, SessionMetadata } from '@/assistant/sessions';
import AssistantHistory from './AssistantHistory';
import { openConfirm } from '@/components/ConfirmModal';

type Entry = AssistantEntry;
type Review = { call: AssistantToolCall; resolve: (approved: boolean) => void };
const errorText = (error: any) => error.response?.data?.error || error.message || 'Assistant request failed.';
const action =
  'rounded-md p-2 text-gray-300 hover:bg-gray-700 focus-visible:outline focus-visible:outline-blue-400 disabled:opacity-40';
const DEFAULT_PANEL_WIDTH = 560;
const MIN_PANEL_WIDTH = 360;
const PANEL_WIDTH_STORAGE_KEY = 'toolkit:assistant:width';

function MessageText({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\[[^\]\n]+\]\((?:https?:\/\/|\/)[^\s)]+\))/g).map((part, index) => {
        const link = part.match(/^\[([^\]]+)\]\(((?:https?:\/\/|\/)[^\s)]+)\)$/);
        return link ? (
          <a
            key={index}
            href={link[2]}
            target={link[2].startsWith('http') ? '_blank' : undefined}
            rel="noopener noreferrer"
            className="text-blue-300 underline"
          >
            {link[1]}
          </a>
        ) : (
          part
        );
      })}
    </>
  );
}

function EntryContent({ entry }: { entry: Entry }) {
  return (
    <>
      <div className="whitespace-pre-wrap break-words">
        <MessageText text={entry.text} />
      </div>
      {entry.preview && (
        <a href={entry.preview.url} target="_blank" rel="noopener noreferrer" className="mt-2 block">
          <img
            src={entry.preview.url}
            alt={entry.preview.path.split(/[\\/]/).pop() || 'Observed image'}
            loading="lazy"
            className="max-h-64 w-full rounded-md object-contain"
          />
        </a>
      )}
    </>
  );
}

export default function AssistantPanel() {
  const { open, setOpen, busy, setBusy, trigger } = useAssistant();
  const [preferredWidth, setPreferredWidth] = useState(DEFAULT_PANEL_WIDTH);
  const [viewportWidth, setViewportWidth] = useState<number | null>(null);
  const [resizing, setResizing] = useState(false);
  const resizeDrag = useRef<{ pointerId: number; x: number; width: number } | null>(null);
  const maxPanelWidth = Math.max(MIN_PANEL_WIDTH, (viewportWidth ?? DEFAULT_PANEL_WIDTH + 64) - 64);
  const panelWidth = Math.min(preferredWidth, maxPanelWidth);
  const [settings, setSettings] = useState(false);
  const [configuration, setConfiguration] = useState<AssistantConfigurationState | null>(null);
  const sessions = useAssistantSessions();
  const { entries, draft: input, attachments } = sessions.content;
  const setEntries = (update: SetStateAction<Entry[]>) =>
    sessions.controller.updateContent(content => ({
      ...content,
      entries: typeof update === 'function' ? update(content.entries) : update,
    }));
  const setInput = (draft: string) => sessions.controller.updateContent(content => ({ ...content, draft }));
  const setAttachments = (update: SetStateAction<string[]>) =>
    sessions.controller.updateContent(content => ({
      ...content,
      attachments: typeof update === 'function' ? update(content.attachments) : update,
    }));
  const [uploading, setUploading] = useState(false);
  const [historyOpen, setHistoryOpen] = useState(false);
  const historyTrigger = useRef<HTMLButtonElement>(null);
  const [reviewChanges, setReviewChanges] = useState(true);
  const reviewPreference = useRef(true);
  const [review, setReview] = useState<Review | null>(null);
  const reviewRef = useRef<Review | null>(null);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [usage, setUsage] = useState<{ total: number; estimated: boolean } | null>(null);
  const run = useRef<{ id: string; controller: AbortController } | null>(null);
  const end = useRef<HTMLDivElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const router = useRouter();
  const pathname = usePathname();
  const navigationBlocked = busy || uploading || sessions.loading || sessions.saveState === 'saving' || !sessions.ready;
  const editingBlocked = busy || sessions.loading || !sessions.ready;
  const sessionActionBlocked = useRef(false);
  sessionActionBlocked.current = busy || uploading || sessions.loading || !sessions.ready;
  const append = (role: Entry['role'], text: string, preview?: Entry['preview']) =>
    setEntries(current => [...current, { id: uuidv4(), role, text, preview }]);
  useEffect(() => {
    const updateViewport = () => setViewportWidth(window.innerWidth);
    updateViewport();
    try {
      const saved = Number(localStorage.getItem(PANEL_WIDTH_STORAGE_KEY));
      if (Number.isFinite(saved) && saved >= MIN_PANEL_WIDTH) setPreferredWidth(saved);
    } catch {
      /* Storage is optional. */
    }
    window.addEventListener('resize', updateViewport);
    return () => window.removeEventListener('resize', updateViewport);
  }, []);
  function resizePanel(value: number) {
    const width = Math.round(Math.max(MIN_PANEL_WIDTH, Math.min(maxPanelWidth, value)));
    setPreferredWidth(width);
    try {
      localStorage.setItem(PANEL_WIDTH_STORAGE_KEY, String(width));
    } catch {}
  }
  function startResize(event: PointerEvent<HTMLDivElement>) {
    if (event.button !== 0 || !event.isPrimary) return;
    event.preventDefault();
    event.currentTarget.focus();
    event.currentTarget.setPointerCapture(event.pointerId);
    resizeDrag.current = { pointerId: event.pointerId, x: event.clientX, width: panelWidth };
    setResizing(true);
  }
  function moveResize(event: PointerEvent<HTMLDivElement>) {
    const drag = resizeDrag.current;
    if (drag?.pointerId === event.pointerId) resizePanel(drag.width + drag.x - event.clientX);
  }
  function finishResize() {
    resizeDrag.current = null;
    setResizing(false);
  }
  useEffect(() => {
    try {
      setOpen(localStorage.getItem('toolkit:assistant:open') === 'true');
    } catch {
      /* Storage is optional. */
    }
    return () => {
      reviewRef.current?.resolve(false);
      const active = run.current;
      if (active) {
        active.controller.abort();
        void apiClient.post('/api/assistant/cancel', { runId: active.id }).catch(() => {});
      }
    };
  }, []);
  useEffect(() => {
    end.current?.scrollIntoView({ block: 'nearest' });
  }, [entries, review, status]);
  useEffect(() => {
    if (!open) return;
    try {
      localStorage.setItem('toolkit:assistant:open', 'true');
    } catch {}
    apiClient
      .get('/api/assistant/settings')
      .then(response => setConfiguration(response.data))
      .catch(cause => setError(errorText(cause)));
    composer.current?.focus();
  }, [open]);
  function close() {
    setOpen(false);
    try {
      localStorage.setItem('toolkit:assistant:open', 'false');
    } catch {}
    trigger.current?.focus();
  }
  function cancel() {
    reviewRef.current?.resolve(false);
    reviewRef.current = null;
    setReview(null);
    const active = run.current;
    if (active) {
      active.controller.abort();
      void apiClient.post('/api/assistant/cancel', { runId: active.id }).catch(() => {});
    }
  }
  function decide(approved: boolean) {
    reviewRef.current?.resolve(approved);
    reviewRef.current = null;
    setReview(null);
  }
  function clearTransient() {
    setError('');
    setUsage(null);
    setStatus('');
  }
  async function newChat() {
    if (navigationBlocked) return;
    if (await sessions.controller.newChat()) {
      clearTransient();
      setHistoryOpen(false);
      composer.current?.focus();
    }
  }
  async function openChat(id: string) {
    if (navigationBlocked) return;
    if (await sessions.controller.open(id)) {
      clearTransient();
      setHistoryOpen(false);
      composer.current?.focus();
    }
  }
  function renameChat(session: SessionMetadata) {
    if (navigationBlocked) return;
    openConfirm({
      title: 'Rename chat',
      type: 'info',
      layer: 'overlay',
      inputTitle: 'Chat title',
      inputLabel: 'Chat title',
      initialInputValue: session.title,
      inputRequired: true,
      confirmText: 'Save title',
      onConfirm: async value => {
        if (!sessionActionBlocked.current) await sessions.controller.rename(session.id, value ?? '');
      },
    });
  }
  function deleteChat(session: SessionMetadata) {
    if (navigationBlocked) return;
    openConfirm({
      title: 'Delete chat?',
      message: `Delete “${session.title}”? This removes its saved conversation.`,
      type: 'danger',
      layer: 'overlay',
      confirmText: 'Delete chat',
      onConfirm: async () => {
        if (sessionActionBlocked.current) return;
        if (await sessions.controller.delete(session.id)) {
          clearTransient();
          composer.current?.focus();
        }
      },
    });
  }
  function reloadChat() {
    if (navigationBlocked) return;
    openConfirm({
      title: 'Reload saved chat?',
      message:
        'This replaces your local unsaved text with the saved version. Use Save a copy to retain it in another chat.',
      type: 'warning',
      layer: 'overlay',
      confirmText: 'Reload saved chat',
      onConfirm: async () => {
        if (sessionActionBlocked.current) return;
        if (await sessions.controller.open(sessions.id, true)) {
          clearTransient();
          composer.current?.focus();
        }
      },
    });
  }
  async function send() {
    if (!input.trim() || editingBlocked || uploading || sessions.error || !configuration) return;
    if (!configuration.config.model) {
      setError('Choose a model in Assistant Settings first.');
      setSettings(true);
      return;
    }
    const active = { id: uuidv4(), controller: new AbortController() };
    run.current = active;
    setBusy(true);
    if (!(await sessions.controller.flush()) || active.controller.signal.aborted) {
      if (run.current === active) run.current = null;
      setBusy(false);
      return;
    }
    const prompt =
      input.trim() + (attachments.length ? `\nAttached images (local paths):\n${attachments.join('\n')}` : '');
    const messages = entries
      .filter(entry => entry.role !== 'tool')
      .slice(-40)
      .map(({ role, text }) => ({ role, content: text }));
    messages.push({ role: 'user', content: prompt });
    sessions.controller.updateContent(content => ({
      entries: [...content.entries, { id: uuidv4(), role: 'user', text: prompt }],
      draft: '',
      attachments: [],
    }));
    setError('');
    setUsage(null);
    let spent = 0,
      estimated = false;
    let turn: any = {
      runId: active.id,
      requestId: uuidv4(),
      configurationRevision: configuration.revision,
      messages,
      screen: pathname,
    };
    try {
      for (;;) {
        active.controller.signal.throwIfAborted();
        setStatus('Thinking…');
        const response = await apiClient.post<AssistantTurnResult>('/api/assistant/turn', turn, {
          signal: active.controller.signal,
        });
        const result = response.data;
        spent += result.usage.totalTokens;
        estimated ||= result.usage.estimated;
        setUsage({ total: spent, estimated });
        if (result.text) append('assistant', result.text);
        if (!result.toolCalls.length) break;
        const toolResults = [],
          images = [];
        for (const call of result.toolCalls) {
          active.controller.signal.throwIfAborted();
          if (reviewPreference.current && !isReadOnlyTool(call.name, call.arguments)) {
            setStatus('Waiting for your review');
            const approved = await new Promise<boolean>(resolve => {
              const pending = { call, resolve };
              reviewRef.current = pending;
              setReview(pending);
            });
            active.controller.signal.throwIfAborted();
            if (!approved) {
              append('tool', `Declined ${call.name}`);
              toolResults.push({
                callId: call.callId,
                output: JSON.stringify({
                  rejected: true,
                  reason: 'The user declined this action. Do not retry or work around this rejection.',
                }),
              });
              continue;
            }
          }
          setStatus(`Running ${call.name}…`);
          try {
            const executed = await apiClient.post('/api/assistant/tools', call, { signal: active.controller.signal });
            const tool = executed.data;
            append(
              'tool',
              `${call.name}: ${tool.output.length > 2000 ? tool.output.slice(0, 2000) + '…' : tool.output}`,
              tool.preview,
            );
            toolResults.push({ callId: call.callId, output: tool.output });
            if (tool.image) images.push({ callId: call.callId, image: tool.image });
            if (tool.navigate) router.push(tool.navigate);
            if (tool.changed) {
              router.refresh();
              window.dispatchEvent(new Event('toolkit:assistant:changed'));
            }
          } catch (cause) {
            active.controller.signal.throwIfAborted();
            const failure = errorText(cause);
            append('tool', `${call.name} failed: ${failure}`);
            toolResults.push({ callId: call.callId, output: JSON.stringify({ error: failure }) });
          }
        }
        turn = {
          runId: active.id,
          requestId: uuidv4(),
          configurationRevision: configuration.revision,
          toolResults,
          ...(images.length ? { attachments: images } : {}),
        };
      }
    } catch (cause) {
      if (active.controller.signal.aborted) append('assistant', 'Stopped. Completed changes are retained.');
      else {
        setError(errorText(cause));
        // Another browser may have changed the shared provider configuration.
        void apiClient
          .get('/api/assistant/settings')
          .then(response => setConfiguration(response.data))
          .catch(() => {});
      }
    } finally {
      void apiClient.post('/api/assistant/cancel', { runId: active.id }).catch(() => {});
      if (run.current === active) run.current = null;
      reviewRef.current = null;
      setReview(null);
      setBusy(false);
      setStatus('');
      await sessions.controller.flush();
    }
  }
  async function upload(files: FileList | null) {
    if (!files?.length || editingBlocked) return;
    if (files.length + attachments.length > 4 || Array.from(files).some(file => file.size > 32 * 1024 * 1024)) {
      setError('Attach at most four images, each up to 32 MiB.');
      if (fileInput.current) fileInput.current.value = '';
      return;
    }
    setUploading(true);
    setError('');
    try {
      const form = new FormData();
      Array.from(files).forEach(file => form.append('files', file));
      const response = await apiClient.post('/api/img/upload', form);
      setAttachments(current => [...current, ...response.data.files]);
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setUploading(false);
      if (fileInput.current) fileInput.current.value = '';
    }
  }
  return (
    <>
      <aside
        id="toolkit-assistant"
        role="dialog"
        aria-label="AI assistant"
        aria-hidden={!open}
        inert={!open}
        onKeyDown={event => {
          if (event.key === 'Escape') {
            event.stopPropagation();
            if (historyOpen) {
              setHistoryOpen(false);
              historyTrigger.current?.focus();
            } else settings ? setSettings(false) : close();
          }
        }}
        style={{ '--assistant-width': `${preferredWidth}px` } as CSSProperties}
        className={`fixed inset-y-0 right-0 z-50 flex w-full sm:w-[var(--assistant-width)] sm:max-w-[calc(100vw-4rem)] flex-col border-l border-gray-700 bg-gray-900 text-gray-100 shadow-2xl transition-transform duration-200 ${open ? 'translate-x-0' : 'translate-x-full'} ${resizing ? 'select-none' : ''}`}
      >
        <div
          role="separator"
          aria-label="Resize AI assistant"
          aria-orientation="vertical"
          aria-controls="toolkit-assistant"
          aria-valuemin={MIN_PANEL_WIDTH}
          aria-valuemax={maxPanelWidth}
          aria-valuenow={panelWidth}
          tabIndex={0}
          title="Drag to resize. Arrow keys adjust width; double-click resets."
          className="group absolute inset-y-0 left-0 z-10 hidden w-2.5 cursor-col-resize touch-none items-center justify-center hover:bg-blue-500/10 focus-visible:bg-blue-500/10 focus-visible:outline-none sm:flex"
          onPointerDown={startResize}
          onPointerMove={moveResize}
          onPointerUp={finishResize}
          onPointerCancel={finishResize}
          onLostPointerCapture={finishResize}
          onDoubleClick={() => resizePanel(DEFAULT_PANEL_WIDTH)}
          onKeyDown={event => {
            const step = event.shiftKey ? 64 : 16;
            const value =
              event.key === 'ArrowLeft'
                ? panelWidth + step
                : event.key === 'ArrowRight'
                  ? panelWidth - step
                  : event.key === 'Home'
                    ? MIN_PANEL_WIDTH
                    : event.key === 'End'
                      ? maxPanelWidth
                      : null;
            if (value === null) return;
            event.preventDefault();
            event.stopPropagation();
            resizePanel(value);
          }}
        >
          <span
            aria-hidden="true"
            className="h-12 w-1 rounded-full bg-gray-600 group-hover:bg-blue-400 group-focus-visible:bg-blue-400"
          />
        </div>
        <header className="flex shrink-0 items-center gap-2 border-b border-gray-700 px-4 py-3">
          <MdSmartToy className="h-5 w-5 text-blue-300" />
          <h2 className="flex-1 font-semibold">{settings ? 'Assistant settings' : 'AI assistant'}</h2>
          {!settings && (
            <>
              <button
                ref={historyTrigger}
                type="button"
                title="Chat history"
                aria-label="Chat history"
                aria-controls="assistant-history"
                aria-expanded={historyOpen}
                className={action}
                onClick={() => {
                  if (!historyOpen) void sessions.controller.refreshHistory();
                  setHistoryOpen(current => !current);
                }}
              >
                <MdHistory className="h-5 w-5" />
              </button>
              <button
                type="button"
                title="New conversation"
                aria-label="New conversation"
                disabled={navigationBlocked}
                className={action}
                onClick={() => void newChat()}
              >
                <MdAdd className="h-5 w-5" />
              </button>
            </>
          )}
          <button
            type="button"
            title={settings ? 'Back to conversation' : 'Assistant settings'}
            aria-label={settings ? 'Back to conversation' : 'Assistant settings'}
            disabled={busy}
            className={action}
            onClick={() => setSettings(current => !current)}
          >
            <MdSettings className="h-5 w-5" />
          </button>
          <button type="button" aria-label="Close AI assistant" className={action} onClick={close}>
            <MdClose className="h-5 w-5" />
          </button>
        </header>
        {!settings && (
          <>
            <div className="flex shrink-0 items-center gap-2 border-b border-gray-800 px-4 py-2">
              <h3 className="min-w-0 flex-1 truncate text-sm text-gray-300" title={sessions.title}>
                {sessions.title}
              </h3>
              {sessions.metadata && (
                <>
                  <button
                    type="button"
                    aria-label="Rename current chat"
                    title="Rename chat"
                    disabled={navigationBlocked}
                    className={action}
                    onClick={() => renameChat(sessions.metadata!)}
                  >
                    <MdEdit />
                  </button>
                  <button
                    type="button"
                    aria-label="Delete current chat"
                    title="Delete chat"
                    disabled={navigationBlocked}
                    className={action}
                    onClick={() => deleteChat(sessions.metadata!)}
                  >
                    <MdDeleteOutline />
                  </button>
                </>
              )}
            </div>
            {historyOpen && (
              <AssistantHistory
                sessions={sessions.sessions}
                activeId={sessions.id}
                disabled={navigationBlocked}
                refreshing={sessions.refreshingHistory}
                error={sessions.historyError}
                onOpen={id => void openChat(id)}
                onNew={() => void newChat()}
                onRename={renameChat}
                onDelete={deleteChat}
                onRefresh={() => void sessions.controller.refreshHistory()}
              />
            )}
          </>
        )}
        {settings && configuration ? (
          <AssistantSettings state={configuration} onSaved={setConfiguration} onClose={() => setSettings(false)} />
        ) : (
          <>
            <div className="min-h-0 flex-1 overflow-y-auto p-4" aria-live="polite">
              {!entries.length && (
                <div className="space-y-4 pt-6">
                  <h3 className="text-lg font-medium">What would you like to work on?</h3>
                  <p className="text-sm text-gray-400">
                    Review a training run, inspect samples, prepare a dataset, edit captions, or configure a new job.
                  </p>
                  <div className="flex flex-col items-start gap-2">
                    {[
                      'Summarize my active training jobs.',
                      'Show me the latest sample outputs.',
                      'Help me prepare a training dataset.',
                    ].map(prompt => (
                      <button
                        key={prompt}
                        type="button"
                        disabled={editingBlocked}
                        className="rounded-lg border border-gray-700 px-3 py-2 text-left text-sm hover:bg-gray-800"
                        onClick={() => {
                          setInput(prompt);
                          composer.current?.focus();
                        }}
                      >
                        {prompt}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {entries.map(entry =>
                entry.role === 'tool' ? (
                  <details key={entry.id} className="group mb-3 text-xs text-gray-400">
                    <summary className="flex cursor-pointer list-none items-center gap-1 rounded py-1 hover:text-gray-300 focus-visible:outline focus-visible:outline-blue-400 [&::-webkit-details-marker]:hidden">
                      <MdChevronRight
                        aria-hidden="true"
                        className="h-3 w-3 shrink-0 transition-transform group-open:rotate-90"
                      />
                      <span className="truncate">{entry.text.replace(/^Declined /, '').split(/[:\s]/, 1)[0]}</span>
                    </summary>
                    <div className="mt-1 max-h-64 overflow-auto pl-4">
                      <EntryContent entry={entry} />
                    </div>
                  </details>
                ) : (
                  <div
                    key={entry.id}
                    className={`mb-4 rounded-lg p-3 ${entry.role === 'user' ? 'ml-6 bg-blue-950' : 'mr-3 bg-gray-800 text-sm'}`}
                  >
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-gray-400">
                      {entry.role === 'user' ? 'You' : 'Assistant'}
                    </p>
                    <EntryContent entry={entry} />
                  </div>
                ),
              )}
              {review && (
                <section
                  aria-label="Review assistant action"
                  className="mb-4 rounded-lg border border-amber-600 bg-amber-950/30 p-3"
                >
                  <h3 className="text-sm font-semibold">Review change: {review.call.name}</h3>
                  <pre className="my-3 max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs">
                    {JSON.stringify(review.call.arguments, null, 2)}
                  </pre>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      className="rounded-md bg-blue-600 px-3 py-2 text-sm hover:bg-blue-500"
                      onClick={() => decide(true)}
                    >
                      Approve action
                    </button>
                    <button
                      type="button"
                      className="rounded-md border border-gray-500 px-3 py-2 text-sm hover:bg-gray-700"
                      onClick={() => decide(false)}
                    >
                      Decline
                    </button>
                  </div>
                </section>
              )}
              {error && (
                <p role="alert" className="mb-3 rounded-md bg-red-950 p-3 text-sm text-red-200 break-words">
                  {error}
                </p>
              )}
              {status && (
                <p role="status" className="text-sm text-blue-300">
                  {status}
                </p>
              )}
              <div ref={end} />
            </div>
            <footer className="shrink-0 space-y-2 border-t border-gray-700 p-3">
              <div role="status" className="text-xs text-gray-500">
                {!sessions.ready || sessions.loading
                  ? 'Loading chats…'
                  : sessions.saveState === 'saving'
                    ? 'Saving…'
                    : sessions.saveState === 'saved'
                      ? 'Saved'
                      : sessions.saveState === 'unsaved'
                        ? 'Unsaved changes'
                        : sessions.saveState === 'error'
                          ? 'Chat not saved'
                          : 'New chat'}
              </div>
              {sessions.error && (
                <div role="alert" className="max-h-48 overflow-auto rounded-md bg-red-950 p-3 text-xs text-red-200">
                  <p>{sessions.error.message}</p>
                  <details className="mt-1">
                    <summary>Error details</summary>
                    <pre className="overflow-auto whitespace-pre-wrap">{sessions.error.stack}</pre>
                  </details>
                  <div className="mt-2 flex flex-wrap gap-3">
                    <button
                      type="button"
                      disabled={busy || uploading || sessions.loading}
                      className="underline disabled:opacity-40"
                      onClick={() => void sessions.controller.retry()}
                    >
                      Retry save
                    </button>
                    {sessions.error.conflict && (
                      <>
                        <button
                          type="button"
                          disabled={navigationBlocked}
                          className="underline disabled:opacity-40"
                          onClick={() => void sessions.controller.saveCopy()}
                        >
                          Save a copy
                        </button>
                        <button
                          type="button"
                          disabled={navigationBlocked}
                          className="underline disabled:opacity-40"
                          onClick={reloadChat}
                        >
                          Reload saved chat
                        </button>
                      </>
                    )}
                  </div>
                </div>
              )}
              <div className="flex items-center justify-between gap-2 text-xs text-gray-400">
                <label className="flex items-center gap-2">
                  <input
                    type="checkbox"
                    checked={reviewChanges}
                    onChange={event => {
                      setReviewChanges(event.target.checked);
                      reviewPreference.current = event.target.checked;
                    }}
                  />
                  Review changes before execution
                </label>
                {usage && (
                  <span>
                    {usage.estimated ? '≈' : ''}
                    {usage.total.toLocaleString()} tokens
                  </span>
                )}
              </div>
              {attachments.map(filename => (
                <div key={filename} className="flex items-center gap-2 text-xs text-gray-300">
                  <span className="flex-1 truncate">{filename.split(/[\\/]/).pop()}</span>
                  <button
                    type="button"
                    aria-label={`Remove attachment ${filename}`}
                    disabled={editingBlocked || uploading}
                    className={action}
                    onClick={() => setAttachments(current => current.filter(item => item !== filename))}
                  >
                    <MdClose />
                  </button>
                </div>
              ))}
              <form
                className="flex items-end gap-2"
                onSubmit={event => {
                  event.preventDefault();
                  void send();
                }}
              >
                <input
                  ref={fileInput}
                  type="file"
                  multiple
                  accept="image/*"
                  className="hidden"
                  onChange={event => void upload(event.target.files)}
                />
                <button
                  type="button"
                  aria-label="Attach images"
                  title="Attach images"
                  disabled={editingBlocked || uploading}
                  className={action}
                  onClick={() => fileInput.current?.click()}
                >
                  <MdAttachFile className="h-5 w-5" />
                </button>
                <textarea
                  ref={composer}
                  aria-label="Message AI assistant"
                  value={input}
                  disabled={editingBlocked}
                  onChange={event => setInput(event.target.value)}
                  onKeyDown={event => {
                    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      void send();
                    }
                  }}
                  rows={3}
                  placeholder="Ask about training, samples, or datasets…"
                  className="min-w-0 flex-1 resize-none rounded-lg border border-gray-600 bg-gray-950 p-2 text-sm focus:border-blue-400 focus:outline-none disabled:opacity-60"
                />
                {busy ? (
                  <button
                    type="button"
                    aria-label="Stop assistant"
                    title="Stop assistant"
                    className={action}
                    onClick={cancel}
                  >
                    <MdStop className="h-6 w-6 text-red-300" />
                  </button>
                ) : (
                  <button
                    type="submit"
                    aria-label="Send message"
                    title="Send message"
                    disabled={!input.trim() || !configuration || uploading || editingBlocked || Boolean(sessions.error)}
                    className={action}
                  >
                    <MdSend className="h-5 w-5 text-blue-300" />
                  </button>
                )}
              </form>
              <p className="text-xs text-gray-500">
                {configuration?.config.model || 'Choose a model in settings'}
                {configuration?.config.reasoningEffort ? ` · ${configuration.config.reasoningEffort}` : ''} ·{' '}
                {configuration?.config.vision ? 'Vision enabled' : 'Vision off'}
              </p>
            </footer>
          </>
        )}
      </aside>
    </>
  );
}
