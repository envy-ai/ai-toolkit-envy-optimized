'use client';

import React, { useEffect, useId, useRef, useState } from 'react';
import { createGlobalState } from 'react-global-hooks';
import { Modal } from './Modal';
import { Checkbox, NumberInput, SelectInput, TextInput } from './formInputs';
import { callScript, ScriptResult } from '@/utils/callScript';
import { encodeFilePathForUrl, getFilename } from '@/utils/basic';

interface TrimLoRAModalState {
  path: string;
  initialBaseModelPath: string;
  initialBaseDequantDtype: string;
  onCreated?: () => void;
  returnFocus: HTMLElement | null;
}

export const trimLoRAModalState = createGlobalState<TrimLoRAModalState | null>(null);

const trainingPrecision = (dtype?: string) => {
  if (dtype === 'bf16' || dtype === 'bfloat16') return 'bfloat16';
  if (dtype === 'fp16' || dtype === 'float16') return 'float16';
  return 'float32';
};

export const openTrimLoRAModal = (
  path: string,
  onCreated?: () => void,
  trainingBaseModelPath?: string,
  trainingDtype?: string,
) => {
  trimLoRAModalState.set({
    path,
    initialBaseModelPath:
      typeof trainingBaseModelPath === 'string' &&
      trainingBaseModelPath.startsWith('/') &&
      trainingBaseModelPath.endsWith('.safetensors')
        ? trainingBaseModelPath
        : '',
    initialBaseDequantDtype: trainingPrecision(trainingDtype),
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
  input_format: string;
  output_format: string;
  metric_scope: 'effective_update' | 'pre_normalization_direction';
  original_stored_rank: number | null;
  expanded_rank: number | null;
  baked?: boolean;
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

type RankPolicy = 'energy' | 'relative_error' | 'fixed' | 'sv_ratio' | 'sv_cumulative' | 'sv_fro';

const rankPolicies: Record<RankPolicy, { label: string; valueLabel: string; defaultValue: number; hint: string }> = {
  energy: {
    label: 'Retained energy',
    valueLabel: 'Retained energy (%)',
    defaultValue: 99.99,
    hint: 'Smallest rank meeting squared singular-value energy retention after storage rounding.',
  },
  relative_error: {
    label: 'Relative reconstruction error',
    valueLabel: 'Maximum relative error (%)',
    defaultValue: 1,
    hint: 'Smallest rank meeting the relative Frobenius error limit after storage rounding.',
  },
  fixed: {
    label: 'Fixed rank',
    valueLabel: 'Target rank',
    defaultValue: 32,
    hint: 'Choose a positive whole-number rank. Layers never increase beyond their original factor rank.',
  },
  sv_ratio: {
    label: 'Singular-value ratio',
    valueLabel: 'Singular-value ratio',
    defaultValue: 1000,
    hint: 'Keep singular values strictly above the largest singular value divided by this ratio (at least 1).',
  },
  sv_cumulative: {
    label: 'Cumulative singular values',
    valueLabel: 'Cumulative singular values (%)',
    defaultValue: 99.99,
    hint: 'Smallest rank retaining this percentage of the sum of singular values.',
  },
  sv_fro: {
    label: 'Frobenius norm retention',
    valueLabel: 'Frobenius norm retention (%)',
    defaultValue: 99.99,
    hint: 'Select by retained Frobenius norm, using the square of this fraction as the energy target.',
  },
};

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
        typeof layer.input_format === 'string' &&
        typeof layer.output_format === 'string' &&
        (layer.baked === undefined || typeof layer.baked === 'boolean') &&
        ['effective_update', 'pre_normalization_direction'].includes(String(layer.metric_scope)) &&
        ['original_stored_rank', 'expanded_rank'].every(
          key => layer[key] === null || (typeof layer[key] === 'number' && Number.isFinite(layer[key])),
        ) &&
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
  const [rankPolicy, setRankPolicy] = useState<RankPolicy>('energy');
  const [threshold, setThreshold] = useState<number | null>(99.99);
  const [minRank, setMinRank] = useState<number | null>(1);
  const [maxRank, setMaxRank] = useState<number | null>(null);
  const [convertToLoRA, setConvertToLoRA] = useState(false);
  const [bakeDoRA, setBakeDoRA] = useState(false);
  const [baseModelPath, setBaseModelPath] = useState(info.initialBaseModelPath);
  const [doraConvention, setDoraConvention] = useState('');
  const [baseDequantDtype, setBaseDequantDtype] = useState(info.initialBaseDequantDtype);
  const [maxWorkingMB, setMaxWorkingMB] = useState<number | null>(512);
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
  const policy = rankPolicies[rankPolicy];
  const percentagePolicy = rankPolicy !== 'fixed' && rankPolicy !== 'sv_ratio';
  const needsWorkingMemory = convertToLoRA || bakeDoRA;
  const effectiveWorkingMB = needsWorkingMemory ? maxWorkingMB : 512;
  const validBakeSettings =
    !bakeDoRA ||
    (baseModelPath.trim().length > 0 &&
      baseModelPath === baseModelPath.trim() &&
      baseModelPath.toLowerCase().endsWith('.safetensors') &&
      ['comfy', 'toolkit'].includes(doraConvention));
  const validThreshold =
    threshold !== null &&
    Number.isFinite(threshold) &&
    (rankPolicy === 'fixed'
      ? Number.isInteger(threshold) && threshold >= 1
      : rankPolicy === 'sv_ratio'
        ? threshold >= 1
        : rankPolicy === 'relative_error'
          ? threshold >= 0 && threshold <= 100
          : threshold > 0 && threshold <= 100);
  const validSettings =
    validThreshold &&
    validBakeSettings &&
    minRank !== null &&
    Number.isInteger(minRank) &&
    minRank >= 1 &&
    (maxRank === null || (Number.isInteger(maxRank) && maxRank >= minRank)) &&
    effectiveWorkingMB !== null &&
    Number.isFinite(effectiveWorkingMB) &&
    effectiveWorkingMB > 0;
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
          'rank-policy': rankPolicy,
          'rank-value': percentagePolicy ? threshold! / 100 : threshold!,
          'min-rank': minRank!,
          'max-rank': maxRank,
          'convert-to-lora': convertToLoRA,
          'bake-dora-to-lora': bakeDoRA,
          'base-model-path': bakeDoRA ? baseModelPath : undefined,
          'dora-convention': bakeDoRA ? doraConvention : undefined,
          'base-dequant-dtype': bakeDoRA ? baseDequantDtype : undefined,
          'max-working-mb': effectiveWorkingMB!,
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
    <Modal isOpen onClose={close} title="Reduce adapter" size="lg" showCloseButton={!busy} closeOnOverlayClick={!busy}>
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
          <p className="text-gray-300">Source adapter (read-only)</p>
          <p className="font-medium break-all">{getFilename(info.path)}</p>
          <p className="text-xs text-gray-400 break-all mt-1">{info.path}</p>
        </div>

        <fieldset disabled={busy !== null} onInput={invalidatePreview}>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <SelectInput
              id={`${id}-criterion`}
              label="Rank policy"
              value={rankPolicy}
              disabled={busy !== null}
              options={Object.entries(rankPolicies).map(([value, configuration]) => ({
                value,
                label: configuration.label,
              }))}
              onChange={value => {
                const nextPolicy = value as RankPolicy;
                setRankPolicy(nextPolicy);
                setThreshold(rankPolicies[nextPolicy].defaultValue);
                invalidatePreview();
              }}
            />
            <NumberInput
              id={`${id}-threshold`}
              label={policy.valueLabel}
              value={threshold}
              onChange={value => {
                setThreshold(value);
                invalidatePreview();
              }}
              min={percentagePolicy ? 0 : 1}
              max={percentagePolicy ? 100 : undefined}
              step={rankPolicy === 'fixed' ? 1 : 'any'}
              required
              allowEmpty
              clampOnBlur={false}
            />
          </div>
          <p className="text-xs text-gray-400 mt-2">{policy.hint}</p>
          {!['energy', 'relative_error'].includes(rankPolicy) && (
            <p className="text-xs text-gray-400 mt-1">
              This policy chooses a rank from the spectrum, with any rank limits applied. The preview reports actual
              retention and error rather than guaranteeing an error bound.
            </p>
          )}
          <div className="mt-4 space-y-2">
            <Checkbox
              label="Bake DoRA to ordinary LoRA before reducing"
              checked={bakeDoRA}
              disabled={busy !== null}
              onChange={value => {
                setBakeDoRA(value);
                invalidatePreview();
              }}
            />
            {bakeDoRA && (
              <div className="space-y-2">
                <p className="text-sm text-amber-300">
                  Bake the final effective DoRA update using matching base weights, then reduce it to ordinary LoRA. The
                  source adapter and base checkpoint stay unchanged.
                </p>
                <TextInput
                  id={`${id}-base-model`}
                  label="Matching base safetensors checkpoint"
                  value={baseModelPath}
                  required
                  disabled={busy !== null}
                  placeholder="/path/to/matching-base.safetensors"
                  onChange={value => {
                    setBaseModelPath(value);
                    invalidatePreview();
                  }}
                />
                <p className="text-xs text-gray-400">
                  Required: a matching local .safetensors checkpoint. Floating weights, Comfy ConvRot8 / int8, FP8,
                  NVFP4, W4A8 and ConvRot W4A4 are supported with their quantization metadata. Other layouts fail with
                  details.
                </p>
                <SelectInput
                  id={`${id}-dora-convention`}
                  label="DoRA convention"
                  value={doraConvention}
                  disabled={busy !== null}
                  options={[
                    { value: '', label: 'Choose a convention…' },
                    { value: 'comfy', label: 'ComfyUI' },
                    { value: 'toolkit', label: 'AI Toolkit / native' },
                  ]}
                  onChange={value => {
                    setDoraConvention(value);
                    invalidatePreview();
                  }}
                />
                <p className="text-xs text-gray-400">
                  Required: choose how you normally load the adapter. A tensor-key suffix alone cannot determine the
                  normalization convention. Exported DoHa uses ComfyUI; native DoHa magnitude uses AI Toolkit.
                </p>
                <SelectInput
                  id={`${id}-base-dequant-dtype`}
                  label="Quantized base precision"
                  value={baseDequantDtype}
                  disabled={busy !== null}
                  options={[
                    { value: 'float32', label: 'FP32' },
                    { value: 'bfloat16', label: 'BF16' },
                    { value: 'float16', label: 'FP16' },
                  ]}
                  onChange={value => {
                    setBaseDequantDtype(value);
                    invalidatePreview();
                  }}
                />
                <p className="text-xs text-gray-400">
                  Used only for quantized weights. Prefilled from the job training dtype; editable. Choose the precision
                  used during training. FP32 is used when no supported training dtype is available.
                </p>
              </div>
            )}
            <Checkbox
              label="Convert LoHa / LoKr to ordinary LoRA"
              checked={convertToLoRA}
              disabled={busy !== null}
              onChange={value => {
                setConvertToLoRA(value);
                invalidatePreview();
              }}
            />
            <p className="text-xs text-gray-400">
              Off by default. Supported 2D LoHa / LoKr layouts require explicit conversion; unsupported layouts fail.
            </p>
            {convertToLoRA && (
              <p className="text-sm text-amber-300">
                Conversion changes the output adapter format to ordinary LoRA. Check the format and measured error in
                the preview before creating a copy.
              </p>
            )}
          </div>
          <details className="mt-3 text-sm text-gray-300">
            <summary className="cursor-pointer w-fit focus:outline-none focus:ring-2 focus:ring-blue-500 rounded">
              Advanced rank limits and memory
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
            {needsWorkingMemory && (
              <div className="space-y-2">
                <NumberInput
                  id={`${id}-working-memory`}
                  label="Conversion / baking working-memory limit (MiB)"
                  value={maxWorkingMB}
                  onChange={value => {
                    setMaxWorkingMB(value);
                    invalidatePreview();
                  }}
                  min={0}
                  step="any"
                  required
                  allowEmpty
                  clampOnBlur={false}
                />
                <p className="text-xs text-gray-400">
                  Conversion factors and baking/decomposition work are checked against this estimate before allocation.
                  Over-budget operations fail.
                </p>
              </div>
            )}
          </details>
        </fieldset>
        {!validSettings && (
          <p className="text-sm text-amber-400">
            Enter a valid policy value, positive whole-number rank limits, and a positive conversion memory limit.
            Maximum rank must be at least minimum rank. DoRA baking also requires a matching safetensors path and an
            explicit convention.
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
              <p className="self-center">{preview.layers.length} layers analyzed</p>
            </div>
            {preview.layers.some(layer => layer.metric_scope === 'pre_normalization_direction') && (
              <p className="rounded-md border border-amber-600/60 bg-amber-500/10 px-3 py-2 text-sm text-amber-300">
                DoRA measurements cover direction factors before normalization only. Magnitude tensors stay unchanged.
                These values do not bound the final normalized weight error or image changes without the base model.
              </p>
            )}
            {preview.layers.some(layer => layer.baked) && (
              <p className="rounded-md border border-blue-500/60 bg-blue-500/10 px-3 py-2 text-sm text-blue-300">
                DoRA was baked using the matching base checkpoint into ordinary LoRA. Measurements compare the final
                effective update before and after reduction; generated-image quality still needs comparison.
              </p>
            )}
            <div className="overflow-auto max-h-64 border border-gray-700 rounded-md">
              <table className="w-full text-xs text-left whitespace-nowrap">
                <caption className="sr-only">Per-layer ranks and numerical reconstruction measurements</caption>
                <thead className="sticky top-0 bg-gray-800 text-gray-300">
                  <tr>
                    <th scope="col" className="px-3 py-2">
                      Layer
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Factor rank
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Retained energy
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Relative error
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Format
                    </th>
                    <th scope="col" className="px-3 py-2">
                      Measurement scope
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {preview.layers.map(layer => (
                    <tr key={layer.name} className="border-t border-gray-700">
                      <th scope="row" className="px-3 py-2 font-normal text-gray-200">
                        {layer.name}
                        {layer.expanded_rank !== null && (
                          <span className="block text-gray-400 mt-1">
                            {layer.original_stored_rank === null
                              ? 'Full stored components'
                              : `Stored rank ${layer.original_stored_rank}`}{' '}
                            · expanded factor rank {layer.expanded_rank}
                          </span>
                        )}
                      </th>
                      <td className="px-3 py-2">
                        {layer.old_rank} → {layer.new_rank}
                      </td>
                      <td className="px-3 py-2">{(layer.retained_energy * 100).toFixed(4)}%</td>
                      <td className="px-3 py-2">{(layer.relative_error * 100).toFixed(4)}%</td>
                      <td className="px-3 py-2">
                        {layer.input_format} → {layer.output_format}
                      </td>
                      <td className="px-3 py-2">
                        {layer.metric_scope === 'pre_normalization_direction'
                          ? 'DoRA direction before normalization'
                          : layer.baked
                            ? 'Baked DoRA effective update'
                            : 'Effective update'}
                      </td>
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
          before relying on the reduced copy.
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
