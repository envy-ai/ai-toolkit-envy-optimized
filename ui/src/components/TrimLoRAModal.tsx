'use client';

import React, { useEffect, useId, useRef, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Modal } from './Modal';
import { NumberInput, SelectInput, TextInput } from './formInputs';
import { callScript, ScriptResult } from '@/utils/callScript';
import { encodeFilePathForUrl, getFilename } from '@/utils/basic';

interface TrimLoRAModalState {
  path: string;
  onCreated?: () => void;
  returnFocus: HTMLElement | null;
}

export const trimLoRAModalState = createGlobalState<TrimLoRAModalState | null>(null);

export const openTrimLoRAModal = (path: string, onCreated?: () => void) => {
  trimLoRAModalState.set({
    path,
    onCreated,
    returnFocus: document.activeElement instanceof HTMLElement ? document.activeElement : null,
  });
};

interface LayerReport {
  name: string;
  old_rank: number;
  new_rank: number;
  retained_energy: number;
  relative_error: number;
}

interface TrimReport {
  schema_version: number;
  preview_token: string;
  input_bytes: number;
  estimated_output_bytes: number;
  layers: LayerReport[];
  warnings: string[];
  written: boolean;
  output_path: string | null;
}

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const parseReport = (value: unknown): TrimReport => {
  if (
    !isRecord(value) ||
    value.schema_version !== 1 ||
    typeof value.preview_token !== 'string' ||
    typeof value.input_bytes !== 'number' ||
    !Number.isFinite(value.input_bytes) ||
    typeof value.estimated_output_bytes !== 'number' ||
    !Number.isFinite(value.estimated_output_bytes) ||
    typeof value.written !== 'boolean' ||
    !Array.isArray(value.warnings) ||
    !value.warnings.every(warning => typeof warning === 'string') ||
    !Array.isArray(value.layers) ||
    !value.layers.every(
      layer =>
        isRecord(layer) &&
        typeof layer.name === 'string' &&
        ['old_rank', 'new_rank', 'retained_energy', 'relative_error'].every(
          key => typeof layer[key] === 'number' && Number.isFinite(layer[key]),
        ),
    ) ||
    (value.written && typeof value.output_path !== 'string')
  ) {
    throw new Error('The trimming script returned an invalid report. See script output below.');
  }
  return value as unknown as TrimReport;
};

const bytes = (value: number) => {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
};

const outputPathFor = (source: string, filename: string) => {
  const index = Math.max(source.lastIndexOf('/'), source.lastIndexOf('\\'));
  return `${index < 0 ? '' : source.slice(0, index + 1)}${filename}`;
};

const buttonClass =
  'px-4 py-2 text-sm rounded-md disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500';

function TrimLoRAForm({ info }: { info: TrimLoRAModalState }) {
  const id = useId();
  const formRef = useRef<HTMLFormElement>(null);
  const busyRef = useRef(false);
  const [criterion, setCriterion] = useState('energy');
  const [threshold, setThreshold] = useState<number | null>(99.99);
  const [minRank, setMinRank] = useState<number | null>(1);
  const [maxRank, setMaxRank] = useState<number | null>(null);
  const [outputName, setOutputName] = useState(
    () => getFilename(info.path).replace(/\.safetensors$/i, '') + '_trimmed.safetensors',
  );
  const [preview, setPreview] = useState<TrimReport | null>(null);
  const [createdPath, setCreatedPath] = useState<string | null>(null);
  const [busy, setBusy] = useState<'analyze' | 'write' | null>(null);
  const [error, setError] = useState<{ message: string; details: string } | null>(null);

  const close = () => {
    if (!busyRef.current) trimLoRAModalState.set(null);
  };

  useEffect(() => {
    const frame = requestAnimationFrame(() => document.getElementById(`${id}-criterion`)?.focus());
    const trapTab = (event: KeyboardEvent) => {
      if (event.key !== 'Tab') return;
      const dialog = formRef.current?.closest('[role="dialog"]');
      const controls = Array.from(
        dialog?.querySelectorAll<HTMLElement>('button, input, select, textarea, summary, a[href], [tabindex]') ?? [],
      ).filter(
        control => control.tabIndex >= 0 && !control.matches(':disabled') && control.getClientRects().length > 0,
      );
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (!first) {
        event.preventDefault();
        formRef.current?.focus();
        return;
      }
      if (event.shiftKey && (document.activeElement === first || !dialog?.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialog?.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', trapTab);
    return () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('keydown', trapTab);
      if (info.returnFocus?.isConnected) info.returnFocus.focus();
    };
  }, [id, info.returnFocus]);

  const invalidatePreview = () => {
    setPreview(null);
    setCreatedPath(null);
  };
  const validSettings =
    threshold !== null &&
    Number.isFinite(threshold) &&
    (criterion === 'energy' ? threshold > 0 && threshold <= 100 : threshold >= 0 && threshold <= 100) &&
    minRank !== null &&
    Number.isInteger(minRank) &&
    minRank >= 1 &&
    (maxRank === null || (Number.isInteger(maxRank) && maxRank >= minRank));
  const outputError = !outputName.trim()
    ? 'Enter an output filename.'
    : /[\\/\x00-\x1f]/.test(outputName) || outputName.includes('..') || outputName !== outputName.trim()
      ? 'Use a filename without path separators, traversal, or surrounding spaces.'
      : !outputName.endsWith('.safetensors')
        ? 'The output filename must end in .safetensors.'
        : outputName.toLowerCase() === getFilename(info.path).toLowerCase()
          ? 'Choose a different filename from the source.'
          : null;

  const run = async (write: boolean) => {
    if (
      busyRef.current ||
      !validSettings ||
      !formRef.current?.reportValidity() ||
      (write && (!preview || outputError || createdPath))
    )
      return;
    busyRef.current = true;
    setBusy(write ? 'write' : 'analyze');
    setError(null);
    if (!write) invalidatePreview();
    let result: ScriptResult | undefined;
    try {
      result = await callScript('trim_lora.py', {
        args: {
          input: info.path,
          [criterion === 'energy' ? 'retained-energy' : 'relative-error']: threshold! / 100,
          'min-rank': minRank!,
          'max-rank': maxRank,
          output: write ? outputPathFor(info.path, outputName) : undefined,
          'expected-preview': write ? preview!.preview_token : undefined,
          json: true,
        },
      });
      if (!result.ok) {
        const scriptMessage = result.stderr?.trim().split(/\r?\n/).pop();
        throw new Error(
          result.error ||
            scriptMessage ||
            (result.timedOut ? 'Trimming timed out.' : `Trimming failed (exit ${result.exitCode ?? 'unknown'}).`),
        );
      }
      const report = parseReport(result.result);
      if (report.written !== write) throw new Error('The trimming script returned an unexpected write status.');
      setPreview(report);
      if (write) {
        setCreatedPath(report.output_path);
        info.onCreated?.();
      }
    } catch (caught) {
      console.error('LoRA trimming failed:', caught, result);
      const exception = caught instanceof Error ? caught : new Error(String(caught));
      setError({
        message: exception.message,
        details: [exception.stack, result?.stderr, result?.stdout].filter(Boolean).join('\n\n'),
      });
    } finally {
      busyRef.current = false;
      setBusy(null);
    }
  };

  return (
    <Modal isOpen onClose={close} title="Trim LoRA" size="lg" showCloseButton={!busy} closeOnOverlayClick={!busy}>
      <form
        ref={formRef}
        tabIndex={-1}
        aria-busy={busy !== null}
        onSubmit={event => {
          event.preventDefault();
          void run(false);
        }}
        className="space-y-4"
      >
        <div className="text-sm">
          <p className="text-gray-300">Source LoRA</p>
          <p className="font-medium break-all">{getFilename(info.path)}</p>
          <p className="text-xs text-gray-400 break-all mt-1">{info.path}</p>
        </div>

        <fieldset disabled={busy !== null} onInput={invalidatePreview}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <SelectInput
              id={`${id}-criterion`}
              label="Criterion"
              value={criterion}
              disabled={busy !== null}
              options={[
                { value: 'energy', label: 'Retained energy' },
                { value: 'error', label: 'Relative reconstruction error' },
              ]}
              onChange={value => {
                setCriterion(value);
                setThreshold(value === 'energy' ? 99.99 : 1);
                invalidatePreview();
              }}
            />
            <NumberInput
              id={`${id}-threshold`}
              label={criterion === 'energy' ? 'Retained energy (%)' : 'Maximum relative error (%)'}
              value={threshold}
              onChange={value => {
                setThreshold(value);
                invalidatePreview();
              }}
              min={0}
              max={100}
              step="any"
              required
              allowEmpty
              clampOnBlur={false}
            />
          </div>
          <details className="mt-3 text-sm text-gray-300">
            <summary className="cursor-pointer w-fit focus:outline-none focus:ring-2 focus:ring-blue-500 rounded">
              Advanced rank limits
            </summary>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-1">
              <NumberInput
                id={`${id}-min-rank`}
                label="Minimum rank"
                value={minRank}
                onChange={value => {
                  setMinRank(value);
                  invalidatePreview();
                }}
                min={1}
                step={1}
                required
                allowEmpty
                clampOnBlur={false}
              />
              <NumberInput
                id={`${id}-max-rank`}
                label="Maximum rank (optional)"
                value={maxRank}
                onChange={value => {
                  setMaxRank(value);
                  invalidatePreview();
                }}
                min={1}
                step={1}
                allowEmpty
                clampOnBlur={false}
                placeholder="Original rank"
              />
            </div>
          </details>
        </fieldset>
        {!validSettings && (
          <p className="text-sm text-amber-400">
            Enter a valid threshold and positive whole-number rank limits. Maximum rank must be at least minimum rank.
          </p>
        )}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={busy !== null || !validSettings}
            className={`${buttonClass} bg-blue-600 hover:bg-blue-700 text-white`}
          >
            {busy === 'analyze' ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>

        {preview && (
          <section className="border-t border-gray-700 pt-4 space-y-3" aria-label="Analysis preview">
            <h4 className="font-semibold">Analysis preview</h4>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
              <div>
                <p className="text-gray-400 text-xs">Original</p>
                <p className="text-lg font-semibold">{bytes(preview.input_bytes)}</p>
              </div>
              <div>
                <p className="text-gray-400 text-xs">Estimated output</p>
                <p className="text-lg font-semibold">{bytes(preview.estimated_output_bytes)}</p>
              </div>
              <p className="self-center">
                {preview.layers.filter(layer => layer.new_rank < layer.old_rank).length} of {preview.layers.length}{' '}
                layers reduced
              </p>
            </div>
            <div className="overflow-auto max-h-64 border border-gray-700 rounded-md">
              <table className="w-full text-xs text-left whitespace-nowrap">
                <caption className="sr-only">Per-layer ranks and numerical reconstruction measurements</caption>
                <thead className="sticky top-0 bg-gray-800 text-gray-300">
                  <tr>
                    <th scope="col" className="px-3 py-2">
                      Layer
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Rank
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Retained energy
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Relative error
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {preview.layers.map(layer => (
                    <tr key={layer.name} className="border-t border-gray-700">
                      <th scope="row" className="px-3 py-2 font-normal text-gray-200">
                        {layer.name}
                      </th>
                      <td className="px-3 py-2">
                        {layer.old_rank} → {layer.new_rank}
                      </td>
                      <td className="px-3 py-2">{(layer.retained_energy * 100).toFixed(4)}%</td>
                      <td className="px-3 py-2">{(layer.relative_error * 100).toFixed(4)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {preview.warnings.map(warning => (
              <p key={warning} className="text-xs text-gray-400">
                {warning}
              </p>
            ))}
          </section>
        )}

        <p className="rounded-md border border-amber-600/60 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
          Numerical energy retention and reconstruction error do not predict visual quality. Compare generated samples
          before relying on the trimmed copy.
        </p>

        <div>
          <TextInput
            id={`${id}-output`}
            label="Output filename"
            value={outputName}
            disabled={busy !== null}
            onChange={value => {
              setOutputName(value);
              setCreatedPath(null);
            }}
          />
          <p className="text-xs text-gray-400 mt-1">
            Creates a new file in the source folder. Existing files are never overwritten.
          </p>
          {outputError && <p className="text-xs text-amber-400 mt-1">{outputError}</p>}
        </div>

        {error && (
          <div role="alert" className="rounded-md border border-rose-500/60 bg-rose-500/10 p-3 text-sm text-rose-300">
            <p className="font-medium">{error.message}</p>
            <details className="mt-2">
              <summary className="cursor-pointer">Error details</summary>
              <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-all text-xs">{error.details}</pre>
            </details>
          </div>
        )}
        <div role="status" aria-live="polite" className="text-sm">
          {busy && (
            <p className="text-blue-300">
              {busy === 'analyze' ? 'Analyzing on CPU. The source is unchanged.' : 'Creating trimmed copy…'} Please keep
              this dialog open.
            </p>
          )}
          {createdPath && (
            <p className="text-emerald-400 break-all">
              Created {getFilename(createdPath)}.{' '}
              <a className="underline" href={`/api/files/${encodeFilePathForUrl(createdPath)}`} download>
                Download copy
              </a>
            </p>
          )}
        </div>
        <div className="border-t border-gray-700 pt-4 flex flex-wrap items-center justify-between gap-3">
          <span className="text-xs text-gray-400">Esc to close · Enter to analyze</span>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={busy !== null}
              onClick={close}
              className={`${buttonClass} bg-gray-800 hover:bg-gray-700 text-gray-200`}
            >
              {createdPath ? 'Close' : 'Cancel'}
            </button>
            <button
              type="button"
              disabled={busy !== null || !preview || !validSettings || !!outputError || !!createdPath}
              onClick={() => {
                void run(true);
              }}
              className={`${buttonClass} bg-blue-600 hover:bg-blue-700 text-white`}
            >
              {busy === 'write' ? 'Creating…' : 'Create trimmed copy'}
            </button>
          </div>
        </div>
      </form>
    </Modal>
  );
}

export default function TrimLoRAModal() {
  const [info] = trimLoRAModalState.use();
  return info ? <TrimLoRAForm info={info} key={info.path} /> : null;
}
