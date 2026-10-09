'use client';

import { useEffect, useId, useState } from 'react';
import { apiClient } from '@/utils/api';
import {
  assistantConfigurationSchema,
  AssistantConfiguration,
  AssistantConfigurationState,
} from '@/assistant/AssistantProvider';

const field =
  'w-full rounded-md border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 focus:border-blue-400 focus:outline-none';
const button = 'rounded-md border border-gray-600 px-3 py-2 text-sm hover:bg-gray-700 disabled:opacity-50';
const limits = [
  ['timeoutMs', 'Request timeout (ms)', 1000, 600000],
  ['maxOutputTokens', 'Output tokens per request', 64, 131072],
  ['maxRunTokens', 'Run token budget', 128, 2000000],
  ['finalResponseReserve', 'Final response reserve', 64, 131072],
  ['maxToolRounds', 'Maximum tool rounds', 1, 100],
  ['maxToolCallsPerRound', 'Tool calls per round', 1, 100],
  ['maxBatchSize', 'Maximum batch operations', 1, 500],
  ['maxJobs', 'Concurrent assistant runs', 1, 20],
] as const;

export default function AssistantSettings({
  state,
  onSaved,
  onClose,
}: {
  state: AssistantConfigurationState;
  onSaved: (state: AssistantConfigurationState) => void;
  onClose: () => void;
}) {
  const [config, setConfig] = useState(state.config);
  const [key, setKey] = useState('');
  const [remember, setRemember] = useState(state.credentialPersisted);
  const [clear, setClear] = useState(false);
  const [models, setModels] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const modelList = useId();
  useEffect(() => {
    if (state.config.authMode === 'none' || state.credentialPresent) {
      const controller = new AbortController();
      apiClient
        .get('/api/assistant/models', { signal: controller.signal })
        .then(response => setModels(response.data.models))
        .catch(() => {});
      return () => controller.abort();
    }
  }, [state.credentialPresent, state.config.authMode]);
  const change = (patch: Partial<AssistantConfiguration>) => {
    setConfig(current => ({ ...current, ...patch }));
    setNotice('');
    if ('baseUrl' in patch || 'authMode' in patch || 'protocol' in patch) setModels([]);
  };
  async function save(discover = false) {
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const validated = assistantConfigurationSchema.parse(config);
      const result = await apiClient.post('/api/assistant/settings', {
        config: validated,
        ...(key ? { apiKey: key } : {}),
        persistCredential: remember,
        clearCredential: clear,
      });
      onSaved(result.data);
      setKey('');
      setClear(false);
      if (discover) {
        const response = await apiClient.get('/api/assistant/models');
        setModels(response.data.models);
        setNotice(`Connected. ${response.data.models.length} models available; choose a model or enter its exact ID.`);
      } else onClose();
    } catch (cause: any) {
      setError(cause.response?.data?.error || cause.message || 'Could not save settings.');
    } finally {
      setBusy(false);
    }
  }
  return (
    <form
      className="flex min-h-0 flex-1 flex-col"
      onSubmit={event => {
        event.preventDefault();
        void save();
      }}
    >
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        <p className="text-sm text-gray-400">
          Connect a provider that supports Responses or Chat completions. The configured model handles chat and visual
          observations.
        </p>
        {error && (
          <p role="alert" className="rounded-md bg-red-950 p-3 text-sm text-red-200 break-words">
            {error}
          </p>
        )}
        <fieldset disabled={busy} className="space-y-4">
          <label className="block space-y-1 text-sm">
            API base URL
            <input
              type="url"
              required
              autoFocus
              className={field}
              value={config.baseUrl}
              onChange={event => change({ baseUrl: event.target.value })}
            />
            <span className="block text-xs text-gray-400">
              Include the API prefix, such as /v1. Routes are appended automatically.
            </span>
          </label>
          <label className="block space-y-1 text-sm">
            Protocol
            <select
              className={field}
              value={config.protocol}
              onChange={event => change({ protocol: event.target.value as AssistantConfiguration['protocol'] })}
            >
              <option value="responses">Responses</option>
              <option value="chat-completions">Chat completions</option>
            </select>
          </label>
          <label className="block space-y-1 text-sm">
            Model
            <div className="flex gap-2">
              <input
                className={field}
                list={modelList}
                value={config.model}
                onChange={event => change({ model: event.target.value })}
                placeholder="Choose or enter an exact model ID"
              />
              <button type="button" className={button} aria-label="Refresh models" onClick={() => void save(true)}>
                Refresh
              </button>
            </div>
            <datalist id={modelList}>
              {models.map(model => (
                <option key={model} value={model} />
              ))}
            </datalist>
            <span className="block text-xs text-gray-400">
              Refresh saves the current connection settings and loads models from that URL.
            </span>
          </label>
          <label className="block space-y-1 text-sm">
            Reasoning effort
            <select
              className={field}
              value={config.reasoningEffort || ''}
              onChange={event =>
                change({
                  reasoningEffort: (event.target.value || undefined) as AssistantConfiguration['reasoningEffort'],
                })
              }
            >
              <option value="">Provider default</option>
              {['none', 'minimal', 'low', 'medium', 'high', 'xhigh'].map(effort => (
                <option key={effort} value={effort}>
                  {effort}
                </option>
              ))}
            </select>
          </label>
          <label className="block space-y-1 text-sm">
            Authentication
            <select
              className={field}
              value={config.authMode}
              onChange={event => change({ authMode: event.target.value as AssistantConfiguration['authMode'] })}
            >
              <option value="api-key">API key</option>
              <option value="none">None</option>
            </select>
          </label>
          {config.authMode === 'api-key' && (
            <>
              <label className="block space-y-1 text-sm">
                API key
                <input
                  className={field}
                  type="password"
                  autoComplete="new-password"
                  value={key}
                  placeholder={state.credentialPresent ? 'Key configured; enter a replacement' : 'Enter API key'}
                  onChange={event => {
                    setKey(event.target.value);
                    setModels([]);
                  }}
                />
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input type="checkbox" checked={remember} onChange={event => setRemember(event.target.checked)} />
                Remember key on this device
              </label>
              <p className="text-xs text-gray-400">
                By default, keys stay in server memory. Remembering a key stores it in a private local file; it does not
                use an encrypted keychain.
              </p>
              {state.credentialPresent && (
                <label className="flex items-center gap-2 text-sm">
                  <input type="checkbox" checked={clear} onChange={event => setClear(event.target.checked)} />
                  Remove configured key
                </label>
              )}
            </>
          )}
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={config.vision}
              onChange={event => change({ vision: event.target.checked })}
            />
            Allow visual observations
          </label>
          <p className="text-xs text-gray-400">
            Lets the assistant send dataset images and sample outputs to the configured model.
          </p>
          <details className="rounded-md border border-gray-700 p-3">
            <summary className="cursor-pointer text-sm">Execution limits</summary>
            <div className="mt-3 space-y-3">
              {limits.map(([name, label, min, max]) => (
                <label key={name} className="block space-y-1 text-sm">
                  {label}
                  <input
                    type="number"
                    className={field}
                    min={min}
                    max={max}
                    step={1}
                    required
                    value={config[name]}
                    onChange={event => change({ [name]: Number(event.target.value) })}
                  />
                </label>
              ))}
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.strictTools}
                  onChange={event => change({ strictTools: event.target.checked })}
                />
                Strict schemas for compatible tools
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.stream}
                  onChange={event => change({ stream: event.target.checked })}
                />
                Stream provider response
              </label>
            </div>
          </details>
        </fieldset>
        {notice && (
          <p role="status" className="text-sm text-green-300">
            {notice}
          </p>
        )}
      </div>
      <footer className="flex flex-wrap justify-end gap-2 border-t border-gray-700 p-3">
        <button type="button" disabled={busy} className={button} onClick={() => void save(true)}>
          Save and test connection
        </button>
        <button type="button" disabled={busy} className={button} onClick={onClose}>
          Cancel
        </button>
        <button
          type="submit"
          disabled={busy}
          className="rounded-md bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-500 disabled:opacity-50"
        >
          {busy ? 'Saving…' : 'Save settings'}
        </button>
      </footer>
    </form>
  );
}
