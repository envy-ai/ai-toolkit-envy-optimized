'use client';

import { useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import Lightbox from 'yet-another-react-lightbox';
import Captions from 'yet-another-react-lightbox/plugins/captions';
import Counter from 'yet-another-react-lightbox/plugins/counter';
import Zoom from 'yet-another-react-lightbox/plugins/zoom';
import { apiClient } from '@/utils/api';

export interface LossReportEntry {
  step: number;
  microbatch: number;
  item_index: number;
  loss: number | null;
  weighted_loss: number | null;
  loss_kind: string;
  loss_weight: number | null;
  timestep: number | null;
  teacher_correction_rms: number | null;
  noise_mean: number | null;
  noise_std: number | null;
  step_loss: number | null;
  baseline: number | null;
  spike_ratio: number | null;
  spike_kind: string;
  relative_path: string;
  path: string;
  image_url: string | null;
  wall_time: number;
  presentation: Record<string, number | boolean | string | null>;
}
interface Report {
  available: boolean;
  total: number;
  recorded: number;
  entries: LossReportEntry[];
}
interface Detail extends LossReportEntry {
  metadata: Record<string, unknown> & { path: string; caption: string };
  rng_state: unknown;
  source_available: boolean;
  learning_rate?: number | null;
  microbatch_size: number;
}
interface Filters {
  mode: string;
  target: string;
  window: string;
  threshold: string;
  top: string;
  start: string;
  end: string;
}
const defaults: Filters = {
  mode: 'threshold',
  target: 'step',
  window: '100',
  threshold: '3',
  top: '20',
  start: '',
  end: '',
};
const pageSize = 50;
const controlClass = 'rounded border border-gray-600 bg-gray-900 px-2 py-1.5 text-sm';
const buttonClass = 'rounded bg-gray-700 px-3 py-1.5 text-sm hover:bg-gray-600 disabled:opacity-40';
export const formatLoss = (value: number | null | undefined) => (value == null ? '—' : value.toPrecision(5));
export const formatSpike = (entry: LossReportEntry) =>
  entry.spike_kind === 'nonfinite'
    ? 'Nonfinite'
    : entry.spike_kind === 'zero_baseline' || entry.spike_ratio === null
      ? '∞×'
      : `${entry.spike_ratio.toFixed(2)}×`;

/** Shared by table and grid: entries are exposures, not a verdict about dataset quality. */
export function LossReportResults({
  entries,
  grid,
  onSelect,
}: {
  entries: LossReportEntry[];
  grid: boolean;
  onSelect: (index: number) => void;
}) {
  if (grid)
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
        {entries.map((entry, index) => (
          <button
            type="button"
            key={`${entry.step}/${entry.microbatch}/${entry.item_index}`}
            onClick={() => onSelect(index)}
            className="rounded border border-gray-700 overflow-hidden bg-gray-900 text-left hover:border-blue-400"
          >
            {entry.image_url ? (
              <img
                loading="lazy"
                src={`${entry.image_url}?thumb=1`}
                alt={entry.relative_path}
                className="w-full h-48 object-contain"
              />
            ) : (
              <div className="h-48 flex items-center justify-center text-gray-400">Image unavailable</div>
            )}
            <div className="p-3 text-sm space-y-1">
              <div className="break-all">{entry.relative_path}</div>
              <div>
                Step {entry.step} · {formatSpike(entry)}
              </div>
              <div className="text-gray-400">
                Image loss {formatLoss(entry.weighted_loss)} · Step loss {formatLoss(entry.step_loss)}
              </div>
              <div className="text-gray-400">
                {String(entry.presentation.loss_attribution ?? 'per_image').replaceAll('_', ' ')}
              </div>
            </div>
          </button>
        ))}
      </div>
    );
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead className="text-gray-400 border-b border-gray-700">
          <tr>
            {[
              'Dataset image path',
              'Step',
              'Batch / item',
              'Raw loss',
              'Weighted loss',
              'Loss attribution',
              'Step loss',
              'Prior mean',
              'Spike',
              'Timestep',
              'Teacher RMS',
              'Crop',
            ].map(label => (
              <th key={label} className="p-2 whitespace-nowrap">
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map((entry, index) => (
            <tr
              key={`${entry.step}/${entry.microbatch}/${entry.item_index}`}
              onClick={() => onSelect(index)}
              className="border-b border-gray-800 hover:bg-gray-800 cursor-pointer"
            >
              <td className="p-2">
                <button
                  type="button"
                  className="text-blue-300 text-left break-all hover:underline"
                  onClick={() => onSelect(index)}
                >
                  {entry.relative_path}
                </button>
              </td>
              <td className="p-2">{entry.step}</td>
              <td className="p-2">
                {entry.microbatch} / {entry.item_index}
              </td>
              {[entry.loss, entry.weighted_loss].map((value, i) => (
                <td key={i} className="p-2">
                  {formatLoss(value)}
                </td>
              ))}
              <td className="p-2 whitespace-nowrap">{String(entry.presentation.loss_attribution ?? 'per_image').replaceAll('_', ' ')}</td>
              {[entry.step_loss, entry.baseline].map((value, i) => <td key={i} className="p-2">{formatLoss(value)}</td>)}
              <td className="p-2 whitespace-nowrap">{formatSpike(entry)}</td>
              <td className="p-2">{formatLoss(entry.timestep)}</td>
              <td className="p-2">{formatLoss(entry.teacher_correction_rms)}</td>
              <td className="p-2 whitespace-nowrap">
                {entry.presentation.crop_width ?? '?'} × {entry.presentation.crop_height ?? '?'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function JobLossReport({ job, onShowStep }: { job: Job; onShowStep?: (step: number) => void }) {
  const [draft, setDraft] = useState<Filters>(defaults);
  const [filters, setFilters] = useState<Filters>(defaults);
  const [page, setPage] = useState(0);
  const [refresh, setRefresh] = useState(0);
  const [grid, setGrid] = useState(false);
  const [restoredJob, setRestoredJob] = useState<string | null>(null);
  const [report, setReport] = useState<Report | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<number | null>(null);
  const [detail, setDetail] = useState<Detail | null>(null);
  const [detailError, setDetailError] = useState('');
  const openAt = useRef<'first' | 'last' | null>(null);

  useEffect(() => {
    let savedFilters = defaults;
    let savedGrid = false;
    try {
      const saved = JSON.parse(localStorage.getItem(`loss-report:${job.id}`) ?? 'null');
      if (saved?.filters && Object.keys(defaults).every(key => typeof saved.filters[key] === 'string'))
        savedFilters = { ...defaults, ...saved.filters };
      savedGrid = saved?.grid === true;
    } catch {
      /* Storage can be unavailable. */
    }
    setDraft(savedFilters);
    setFilters(savedFilters);
    setGrid(savedGrid);
    setPage(0);
    setSelected(null);
    setRestoredJob(job.id);
  }, [job.id]);

  useEffect(() => {
    if (restoredJob !== job.id) return;
    try {
      localStorage.setItem(`loss-report:${job.id}`, JSON.stringify({ filters, grid }));
    } catch {
      /* Filtering also works without storage. */
    }
  }, [job.id, restoredJob, filters, grid]);

  useEffect(() => {
    if (restoredJob !== job.id) return;
    const controller = new AbortController();
    setLoading(true);
    setError('');
    setReport(null);
    const params = new URLSearchParams({ ...filters, offset: String(page * pageSize), limit: String(pageSize) });
    apiClient
      .get(`/api/jobs/${job.id}/loss-report?${params}`, { signal: controller.signal })
      .then(({ data }: { data: Report }) => {
        if (controller.signal.aborted) return;
        if (page > 0 && page * pageSize >= data.total) {
          setPage(Math.max(0, Math.ceil(data.total / pageSize) - 1));
          return;
        }
        setReport(data);
        if (openAt.current && data.entries.length)
          setSelected(openAt.current === 'first' ? 0 : data.entries.length - 1);
        openAt.current = null;
      })
      .catch(err => {
        if (!controller.signal.aborted) setError(err.response?.data?.error ?? 'Could not load the loss report.');
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [job.id, restoredJob, filters, page, refresh]);

  const entry = selected === null ? null : report?.entries[selected];
  useEffect(() => {
    setDetail(null);
    setDetailError('');
    if (!entry) return;
    const controller = new AbortController();
    const params = new URLSearchParams({
      detail: '1',
      step: String(entry.step),
      microbatch: String(entry.microbatch),
      item: String(entry.item_index),
    });
    apiClient
      .get(`/api/jobs/${job.id}/loss-report?${params}`, { signal: controller.signal })
      .then(({ data }) => {
        if (!controller.signal.aborted) setDetail(data);
      })
      .catch(err => {
        if (!controller.signal.aborted) setDetailError(err.response?.data?.error ?? 'Could not load image details.');
      });
    return () => controller.abort();
  }, [job.id, entry]);

  const movePage = (direction: 'first' | 'last') => {
    setSelected(null);
    openAt.current = direction;
    setPage(value => value + (direction === 'first' ? 1 : -1));
  };
  const details = (
    <div className="max-w-5xl mx-auto text-left text-sm text-gray-100 max-h-[40vh] overflow-y-auto p-2">
      {entry && (
        <>
          <p className="break-all">{detail?.metadata.path ?? entry.path}</p>
          <p className="my-2">
            Step {entry.step} · Microbatch {entry.microbatch} · Item {entry.item_index} · {formatSpike(entry)} · Raw
            loss {formatLoss(entry.loss)} · Weighted loss {formatLoss(entry.weighted_loss)} · Step loss{' '}
            {formatLoss(entry.step_loss)} · Prior mean {formatLoss(entry.baseline)}
          </p>
          <div className="flex gap-2 flex-wrap mb-2">
            {onShowStep && (
              <button type="button" className={buttonClass} onClick={() => onShowStep(entry.step)}>
                Show step on graph
              </button>
            )}
            {page > 0 && selected === 0 && (
              <button type="button" className={buttonClass} onClick={() => movePage('last')}>
                Previous report page
              </button>
            )}
            {report && (page + 1) * pageSize < report.total && selected === report.entries.length - 1 && (
              <button type="button" className={buttonClass} onClick={() => movePage('first')}>
                Next report page
              </button>
            )}
          </div>
          {detailError ? (
            <p role="alert">{detailError}</p>
          ) : !detail ? (
            <p>Loading full record…</p>
          ) : (
            <>
              {!detail.source_available && detail.metadata.source_kind !== 'prompt' && (
                <p className="text-amber-300">
                  The original image is missing or has moved. Its training record is still available.
                </p>
              )}
              <p className="text-gray-400">{detail.metadata.source_kind === 'prompt' ? 'This objective uses prompts without a source image.' : 'Preview shows the original training input. The recorded crop and flips appear below.'}</p>
              <p className="mt-2 font-semibold">Training caption</p>
              <p className="whitespace-pre-wrap break-words">{detail.metadata.caption || '(empty caption)'}</p>
              <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 my-3">
                {Object.entries({
                  ...detail.metadata,
                  ...detail.presentation,
                  loss_weight: detail.loss_weight,
                  learning_rate: detail.learning_rate,
                  microbatch_size: detail.microbatch_size,
                  timestep: detail.timestep,
                  teacher_correction_rms: detail.teacher_correction_rms,
                  noise_mean: detail.noise_mean,
                  noise_std: detail.noise_std,
                  loss_kind: detail.loss_kind,
                  recorded_at: new Date(detail.wall_time * 1000).toISOString(),
                  rng_snapshot: detail.rng_state ? 'Recorded' : 'Not recorded',
                })
                  .filter(([key]) => key !== 'caption' && key !== 'path')
                  .map(([key, value]) => (
                    <div key={key}>
                      <dt className="text-gray-400">{key.replaceAll('_', ' ')}</dt>
                      <dd className="whitespace-pre-wrap break-all">
                        {typeof value === 'object' ? JSON.stringify(value) : String(value ?? '—')}
                      </dd>
                    </div>
                  ))}
              </dl>
              <a
                className="text-blue-300 underline"
                download={`loss-step-${entry.step}-batch-${entry.microbatch}-item-${entry.item_index}.json`}
                href={`data:application/json;charset=utf-8,${encodeURIComponent(JSON.stringify({ ...detail, baseline: entry.baseline, step_loss: entry.step_loss, spike_kind: entry.spike_kind, spike_ratio: entry.spike_ratio }, null, 2))}`}
              >
                Download full record JSON{detail.rng_state ? ' (including RNG state)' : ''}
              </a>
            </>
          )}
        </>
      )}
    </div>
  );

  return (
    <div className="space-y-4 text-gray-100">
      <h2 className="text-xl font-semibold">Loss Report</h2>
      <form
        onSubmit={event => {
          event.preventDefault();
          setSelected(null);
          setPage(0);
          setFilters({ ...draft });
        }}
        className="flex flex-wrap items-end gap-3"
      >
        <label className="flex flex-col gap-1 text-sm">
          Find
          <select
            className={controlClass}
            value={draft.mode}
            onChange={e => setDraft({ ...draft, mode: e.target.value })}
          >
            <option value="threshold">Above moving average</option>
            <option value="top">Top X spikes</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm">
          Compare
          <select
            className={controlClass}
            value={draft.target}
            onChange={e => setDraft({ ...draft, target: e.target.value })}
          >
            <option value="step">Training steps (all images)</option>
            <option value="image">Individual weighted image losses</option>
          </select>
        </label>
        {(['window', 'threshold', 'top', 'start', 'end'] as const).map(key => (
          <label key={key} className="flex flex-col gap-1 text-sm">
            {
              {
                window: 'Prior steps',
                threshold: 'Threshold (× mean)',
                top: 'Number of spikes (Top X)',
                start: 'Start step (optional)',
                end: 'End step (optional)',
              }[key]
            }
            <input
              type="number"
              className={`${controlClass} w-36 disabled:opacity-50`}
              value={draft[key]}
              required={key !== 'start' && key !== 'end'}
              disabled={(key === 'top' && draft.mode !== 'top') || (key === 'threshold' && draft.mode !== 'threshold')}
              min={key === 'start' || key === 'end' ? 0 : 1}
              step={key === 'threshold' ? 'any' : 1}
              max={key === 'window' || key === 'top' ? 10000 : key === 'threshold' ? 1000000 : Number.MAX_SAFE_INTEGER}
              placeholder={key === 'start' || key === 'end' ? 'Any step' : ''}
              onChange={e => setDraft({ ...draft, [key]: e.target.value })}
            />
          </label>
        ))}
        <button className={buttonClass} type="submit">
          Apply
        </button>
        <button
          className={buttonClass}
          type="button"
          disabled={loading}
          onClick={() => {
            setSelected(null);
            setRefresh(value => value + 1);
          }}
        >
          Refresh
        </button>
      </form>
      <p className="text-sm text-gray-400">
        Spikes compare against the mean of previous logged training steps, excluding the current step. At least{' '}
        {Math.min(20, Number(filters.window))} prior finite losses are required. Step bounds are inclusive and preserve
        the earlier baseline. Top X ranks increases above the mean by ratio; nonfinite losses and positive losses above
        a zero baseline are shown first. Images can be difficult without being unsuitable.
      </p>
      <div className="flex gap-2 items-center text-sm">
        <button type="button" className={buttonClass} aria-pressed={!grid} onClick={() => setGrid(false)}>
          Table
        </button>
        <button type="button" className={buttonClass} aria-pressed={grid} onClick={() => setGrid(true)}>
          Image grid
        </button>
        {report && (
          <span className="text-gray-400">
            {report.total} matching exposures · {report.recorded} recorded
          </span>
        )}
      </div>
      {loading && <p role="status">Loading report…</p>}
      {error && (
        <p role="alert" className="text-red-300">
          {error}
        </p>
      )}
      {report && (!report.available || report.recorded === 0) && (
        <p className="rounded bg-gray-800 p-4">
          No training-input records yet. Loss reporting is on by default. Check the “Loss reporting” switch in
          Training settings and start or resume training. Existing losses cannot be linked to inputs retroactively.
        </p>
      )}
      {report && report.recorded > 0 && report.entries.length === 0 && (
        <p>No matching spikes. Try a lower threshold, a wider step range, or a shorter history window.</p>
      )}
      {report && <LossReportResults entries={report.entries} grid={grid} onSelect={setSelected} />}
      {report && report.total > 0 && (
        <div className="flex gap-3 items-center text-sm">
          <button
            className={buttonClass}
            type="button"
            disabled={page === 0 || loading}
            onClick={() => {
              setSelected(null);
              setPage(page - 1);
            }}
          >
            Previous
          </button>
          <span>
            Page {page + 1} of {Math.ceil(report.total / pageSize)}
          </span>
          <button
            className={buttonClass}
            type="button"
            disabled={(page + 1) * pageSize >= report.total || loading}
            onClick={() => {
              setSelected(null);
              setPage(page + 1);
            }}
          >
            Next
          </button>
        </div>
      )}
      <Lightbox
        open={selected !== null && !!entry}
        close={() => setSelected(null)}
        index={selected ?? 0}
        slides={(report?.entries ?? []).map((row, index) => ({
          src: row.image_url ?? '',
          title: row.relative_path,
          description: index === selected ? details : undefined,
        }))}
        plugins={[Captions, Counter, Zoom]}
        on={{ view: ({ index }) => setSelected(index) }}
        carousel={{ finite: true, imageFit: 'contain', preload: 1 }}
        captions={{ descriptionTextAlign: 'start', descriptionMaxLines: 100, showToggle: true }}
        zoom={{ maxZoomPixelRatio: 4, scrollToZoom: true }}
        controller={{ closeOnBackdropClick: true }}
      />
    </div>
  );
}
