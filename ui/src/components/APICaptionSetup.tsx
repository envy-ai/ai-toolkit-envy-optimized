import { useId } from 'react';
import { CaptionProcessConfig } from '@/types';
import { DEFAULT_CAPTION_REGEX_REPLACE } from '../../cron/captionApiConfig';
import { Checkbox, CreatableSelectInput, NumberInput, SelectInput, TextAreaInput, TextInput } from './formInputs';

type Props = {
  caption: CaptionProcessConfig['caption'];
  setCaption: (value: any, key: string) => void;
  apiKey: string;
  onKeyChange: (key: string) => void;
  clearKey: boolean;
  onClearKeyChange: (clear: boolean) => void;
};
export default function APICaptionSetup({
  caption,
  setCaption,
  apiKey,
  onKeyChange,
  clearKey,
  onClearKeyChange,
}: Props) {
  const set = (key: string, value: unknown) => setCaption(value, `config.process[0].caption.${key}`);
  const cleanupId = useId();
  const cleanup = { ...DEFAULT_CAPTION_REGEX_REPLACE, ...caption.regex_replace };
  const setCleanup = (key: keyof typeof cleanup, value: string | boolean) =>
    set('regex_replace', { ...cleanup, [key]: value });
  return (
    <div className="mt-4 space-y-4">
      <TextInput
        label="API URL"
        value={caption.api_url ?? ''}
        placeholder="http://server:5005/v1"
        required
        onChange={value => {
          set('api_url', value);
          set('api_credential_id', undefined);
          onKeyChange('');
        }}
      />
      <div>
        <TextInput
          label="API key"
          type="password"
          value={apiKey}
          onChange={onKeyChange}
          disabled={clearKey}
          placeholder={
            caption.api_credential_id
              ? 'Saved key for this API URL · enter to replace'
              : 'Optional for servers without authentication'
          }
        />
        <p className="mt-1 text-xs text-gray-500">
          Saved privately on this server; never included in job configuration. Changing the API URL requires entering
          the key again.
        </p>
        <Checkbox
          label="Use no authentication (clear the saved key for this job)"
          checked={clearKey}
          onChange={onClearKeyChange}
        />
      </div>
      <TextInput
        label="Model"
        value={caption.model_name_or_path}
        onChange={value => set('model_name_or_path', value)}
        placeholder="Exact model ID served by the API"
        required
      />
      <TextAreaInput
        label="System prompt"
        value={caption.system_prompt ?? ''}
        onChange={value => set('system_prompt', value)}
        placeholder="Optional system message"
      />
      <TextAreaInput
        label="Caption prompt"
        value={caption.caption_prompt ?? ''}
        onChange={value => set('caption_prompt', value)}
        required
      />
      <details className="rounded-md border border-gray-700 p-3">
        <summary className="cursor-pointer text-sm text-gray-200">Caption cleanup · optional, off by default</summary>
        <div className="mt-3 space-y-3">
          <Checkbox label="Enable caption cleanup" checked={cleanup.enabled}
            onChange={value => setCleanup('enabled', value)} />
          <TextAreaInput id={`${cleanupId}-pattern`} label="From (regular expression)"
            value={cleanup.pattern} onChange={value => setCleanup('pattern', value)}
            placeholder={'.*<caption>(.*)</caption>.*'} rows={3} disabled={!cleanup.enabled} />
          <TextAreaInput id={`${cleanupId}-replacement`} label="To (replacement)"
            value={cleanup.replacement} onChange={value => setCleanup('replacement', value)}
            placeholder="$1" rows={2} disabled={!cleanup.enabled} />
          <p className="text-xs text-gray-500">
            Leave To empty to remove matches. Use $1 for capture group 1, $&amp; for the whole match,
            and $$ for a literal $. / does not need escaping. Replacement backslashes are literal;
            enter an actual newline for a line break.
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {([
              ['ignore_case', 'Ignore case (i)'],
              ['multiline', 'Multiline: ^/$ per line (m)'],
              ['dotall', 'Dot matches newlines (s)'],
              ['verbose', 'Ignore pattern whitespace/comments (x)'],
              ['global', 'Replace all matches'],
            ] as const).map(([key, label]) => (
              <Checkbox key={key} label={label} checked={cleanup[key]}
                onChange={value => setCleanup(key, value)} disabled={!cleanup.enabled} />
            ))}
          </div>
          <p className="text-xs text-gray-500">
            Uses Python regular expressions, validated when the job starts before any API request.
            Dot matches newlines lets . match line breaks; it does not turn them into spaces.
            The x flag ignores pattern whitespace and comments, not whitespace in the caption.
            Applied to the API response before saving; no match leaves it unchanged.
            An empty result stops the job without writing a caption.
          </p>
        </div>
      </details>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <CreatableSelectInput
          label="Caption extension"
          value={caption.caption_extension ?? 'txt'}
          onChange={value => set('caption_extension', value)}
          options={[
            { value: 'txt', label: 'txt' },
            { value: 'caption', label: 'caption' },
            { value: 'json', label: 'json' },
          ]}
        />
        <div className="pt-4">
          <Checkbox
            label="Recaption existing captions"
            checked={caption.recaption}
            onChange={value => set('recaption', value)}
          />
        </div>
      </div>
      <p className="text-xs text-gray-500">
        Still images only. Uses the CPU/API queue alongside training; this toolkit does not load a captioning model onto
        a GPU.
      </p>
      <details className="rounded-md border border-gray-700 p-3">
        <summary className="cursor-pointer text-sm text-gray-200">Advanced options</summary>
        <div className="mt-3 grid grid-cols-1 gap-4 sm:grid-cols-2">
          <NumberInput
            label="Maximum response tokens"
            value={caption.max_new_tokens ?? null}
            onChange={value => set('max_new_tokens', value ?? undefined)}
            min={1}
            step={1}
            allowEmpty
            clampOnBlur={false}
            required
          />
          <NumberInput
            label="Request timeout (seconds)"
            value={caption.api_timeout ?? null}
            onChange={value => set('api_timeout', value ?? undefined)}
            min={0.001}
            max={3600}
            allowEmpty
            clampOnBlur={false}
            required
          />
          {(
            [
              ['temperature', 'Temperature', 0, 2],
              ['top_p', 'Top P', 0, 1],
              ['frequency_penalty', 'Frequency penalty', -2, 2],
              ['presence_penalty', 'Presence penalty', -2, 2],
            ] as const
          ).map(([key, label, min, max]) => (
            <NumberInput
              key={key}
              label={label}
              value={caption[key] ?? null}
              onChange={value => set(key, value ?? undefined)}
              min={min}
              max={max}
              allowEmpty
              clampOnBlur={false}
              placeholder="Provider default"
            />
          ))}
          <NumberInput
            label="Seed"
            value={caption.seed ?? null}
            onChange={value => set('seed', value ?? undefined)}
            step={1}
            allowEmpty
            clampOnBlur={false}
            placeholder="Provider default"
          />
          <SelectInput
            label="Response token field"
            value={caption.api_max_tokens_field ?? 'max_tokens'}
            onChange={value => set('api_max_tokens_field', value)}
            options={[
              { value: 'max_tokens', label: 'max_tokens (local / legacy compatible)' },
              { value: 'max_completion_tokens', label: 'max_completion_tokens (modern)' },
            ]}
          />
        </div>
        <div className="mt-4">
          <TextAreaInput
            label="Stop sequences (one per line, at most four)"
            value={caption.stop?.join('\n') ?? ''}
            onChange={value => set('stop', value === '' ? undefined : value.split('\n'))}
          />
        </div>
        <p className="mt-2 text-xs text-gray-500">
          Blank optional parameters use the provider defaults. The response limit defaults to 4096 and accepts a
          configurable positive integer; the API server enforces its model limits. Invalid values are rejected, never
          silently adjusted. Requests are not retried with a different token field.
        </p>
      </details>
    </div>
  );
}
