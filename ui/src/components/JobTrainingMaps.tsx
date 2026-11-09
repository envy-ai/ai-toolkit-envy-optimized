'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import { SelectInput } from '@/components/formInputs';
import { trainingMapColor } from '@/utils/trainingMaps';
import type { TrainingMapFrame, TrainingMapGrid, TrainingMapKind, TrainingMapManifest, TrainingMapsResponse } from '@/utils/trainingMaps';

const buttonClass = 'rounded bg-gray-700 px-3 py-1.5 text-sm hover:bg-gray-600 disabled:opacity-40';
const mapOptions = [
  { value: 'improvement', label: 'Learned since the start' },
  { value: 'error', label: 'Not learned yet' },
  { value: 'effect', label: 'Where the adapter acts' },
];
const descriptions: Record<TrainingMapKind, string> = {
  improvement: 'Green: lower prediction error than the first recorded step. Magenta: higher error. These are fixed noisy-image probes, not a measure of likeness.',
  error: 'Prediction error against the flow target. Blue → yellow → red means more error. Follow an area across steps; fine detail naturally has higher error.',
  effect: 'Absolute prediction change with this adapter enabled versus disabled at the same input. Blue → yellow → red means a larger change.',
};

export function TrainingMapOverlay({ grid, scale, signed, opacity }: {
  grid: TrainingMapGrid; scale: number; signed: boolean; opacity: number;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = canvas.current?.getContext('2d');
    if (!context) return;
    context.clearRect(0, 0, grid.w, grid.h);
    grid.v.forEach((value, index) => {
      context.fillStyle = trainingMapColor(value, scale, signed);
      context.fillRect(index % grid.w, Math.floor(index / grid.w), 1, 1);
    });
  }, [grid, scale, signed]);
  return <canvas ref={canvas} width={grid.w} height={grid.h} aria-hidden="true"
    className="absolute inset-0 w-full h-full pointer-events-none"
    style={{ imageRendering: 'pixelated', opacity: opacity / 100 }} />;
}

export default function JobTrainingMaps({ job }: { job: Job }) {
  const [manifest, setManifest] = useState<TrainingMapManifest>();
  const [frame, setFrame] = useState<TrainingMapFrame | null>(null);
  const [available, setAvailable] = useState<boolean>();
  const [probeKey, setProbeKey] = useState('');
  const [level, setLevel] = useState('0.5');
  const [kind, setKind] = useState<TrainingMapKind>('improvement');
  const [opacity, setOpacity] = useState(100);
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<{ message: string; stack?: string }>();
  const requestId = useRef(0);
  const cache = useRef(new Map<number, TrainingMapFrame>());

  const load = useCallback(async (step?: number, refresh = false) => {
    const request = ++requestId.current;
    if (!refresh && step !== undefined && cache.current.has(step)) {
      setFrame(cache.current.get(step)!);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const response = await apiClient.get<TrainingMapsResponse>(`/api/jobs/${job.id}/training-maps`, {
        params: step === undefined ? {} : { step },
      });
      if (request !== requestId.current) return;
      const data = response.data;
      if (refresh) cache.current.clear();
      setAvailable(data.available);
      setManifest(data.manifest);
      setFrame(data.frame ?? null);
      setProbeKey(current => data.manifest?.probes.some(probe => probe.key === current)
        ? current : data.manifest?.probes[0]?.key ?? '');
      if (data.frame) cache.current.set(data.frame.step, data.frame);
      setError(undefined);
    } catch (caught) {
      if (request !== requestId.current) return;
      const failure = caught as { response?: { data?: { error?: string; stack?: string } }; message?: string; stack?: string };
      setError({ message: failure.response?.data?.error ?? failure.message ?? 'Could not read training maps.',
        stack: failure.response?.data?.stack ?? failure.stack });
      setPlaying(false);
    } finally {
      if (request === requestId.current) setLoading(false);
    }
  }, [job.id]);

  useEffect(() => {
    cache.current.clear();
    setManifest(undefined);
    setFrame(null);
    setAvailable(undefined);
    setPlaying(false);
    void load();
    return () => { requestId.current += 1; };
  }, [load]);

  const steps = manifest?.steps ?? [];
  const index = frame ? steps.indexOf(frame.step) : -1;
  const move = (direction: number) => {
    const next = index + direction;
    if (!loading && next >= 0 && next < steps.length) void load(steps[next]);
  };
  useEffect(() => {
    if (!playing || loading) return;
    if (index < 0 || index >= steps.length - 1) { setPlaying(false); return; }
    const timer = window.setTimeout(() => { void load(steps[index + 1]); }, 700);
    return () => window.clearTimeout(timer);
  }, [playing, loading, index, steps, load]);

  const probe = manifest?.probes.find(item => item.key === probeKey);
  const grid = frame?.maps[probeKey]?.[level]?.[kind];
  const scale = manifest?.scales[probeKey]?.[level]?.[kind] ?? 0;
  const recordedError = manifest?.status === 'error' ? manifest.error : undefined;

  return <div className="space-y-4 px-4 pb-6">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div><h2 className="text-xl font-semibold">Training Maps</h2>
        <p className="text-sm text-gray-400">Fixed training crops and noise. Refresh to load newly saved measurements.</p></div>
      <button type="button" className={buttonClass} disabled={loading}
        onClick={() => { setPlaying(false); void load(undefined, true); }}>{loading ? 'Loading…' : 'Refresh'}</button>
    </div>
    {available === false && <p className="rounded border border-gray-700 p-4 text-gray-300">
      No training maps recorded. Enable Training maps in the training settings before starting or resuming a supported job.
    </p>}
    {manifest && <>
      <div className="grid gap-4 sm:grid-cols-2 max-w-3xl">
        <SelectInput id="training-map-kind" label="Map" value={kind} options={mapOptions}
          onChange={value => setKind(value as TrainingMapKind)} />
        <SelectInput id="training-map-noise" label="Noise level" value={level}
          options={manifest.levels.map(value => ({ value: String(value),
            label: value === 0.25 ? 'Low (detail)' : value === 0.75 ? 'High (layout)' : 'Mid' }))}
          onChange={setLevel} />
      </div>
      <div className="flex flex-wrap gap-3" aria-label="Training map pictures">
        {manifest.probes.map((item, number) => <button key={item.key} type="button"
          className={`${buttonClass} flex items-center gap-3 border ${item.key === probeKey ? 'border-blue-400' : 'border-gray-700'}`}
          aria-pressed={item.key === probeKey} onClick={() => setProbeKey(item.key)}>
          <img src={item.image_url} alt="" className="h-12 w-12 object-contain" />Picture {number + 1}
        </button>)}
      </div>
      {probe && <div className="rounded border border-gray-700 bg-gray-900 p-3 space-y-3"
        tabIndex={0} role="region" aria-label="Training map viewer"
        onKeyDown={event => {
          if (event.target !== event.currentTarget) return;
          if (event.key === 'ArrowLeft') { event.preventDefault(); move(-1); }
          if (event.key === 'ArrowRight') { event.preventDefault(); move(1); }
          if (event.key === ' ') { event.preventDefault(); setPlaying(value => !value); }
        }}>
        <div className="flex flex-wrap items-center gap-3">
          <button type="button" className={buttonClass} disabled={loading || index <= 0} onClick={() => move(-1)}>Previous</button>
          <button type="button" className={buttonClass} disabled={loading || steps.length < 2}
            aria-label={playing ? 'Pause map playback' : 'Play map playback'}
            onClick={() => { if (!playing && index === steps.length - 1) void load(steps[0]); setPlaying(value => !value); }}>
            {playing ? 'Pause' : 'Play'}</button>
          <label htmlFor="training-map-step" className="sr-only">Recorded training step</label>
          <input id="training-map-step" className="flex-1 min-w-24" type="range" min={0}
            max={Math.max(0, steps.length - 1)} value={Math.max(0, index)} disabled={loading || !steps.length}
            onChange={event => { setPlaying(false); void load(steps[Number(event.target.value)]); }} />
          <button type="button" className={buttonClass} disabled={loading || index >= steps.length - 1}
            onClick={() => move(1)}>Next</button>
          <span className="text-sm" aria-live="polite">{frame ? `Step ${frame.step} / ${steps.at(-1)}` : 'No recorded steps'}</span>
        </div>
        <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_18rem]">
          <div className="min-w-0">
            <div className="relative w-fit max-w-full mx-auto">
              <img src={probe.image_url} alt={`Fixed training crop: ${probe.caption}`}
                width={probe.width} height={probe.height} className="block max-w-full max-h-[65vh] w-auto h-auto" />
              {grid && <TrainingMapOverlay grid={grid} scale={scale} signed={kind === 'improvement'} opacity={opacity} />}
            </div>
            <p className="mt-2 text-xs text-gray-400">Training crop · {probe.width} × {probe.height}. Focus this viewer for ← / → steps and Space to play/pause.</p>
          </div>
          <div className="space-y-4 text-sm">
            <h3 className="font-semibold">{mapOptions.find(option => option.value === kind)?.label}</h3>
            <p className="text-gray-300">{descriptions[kind]}</p>
            {!grid && <p className="text-amber-300">{kind === 'improvement'
              ? 'This first measurement is the baseline. Choose a later step to see the change.'
              : 'This map is not available at the selected step.'}</p>}
            <p className="text-gray-400">One scale per picture, map and noise level across the recorded steps.</p>
            <div><label htmlFor="training-map-opacity" className="block mb-2">Map opacity · {opacity}%</label>
              <input id="training-map-opacity" type="range" className="w-full" min={0} max={100}
                value={opacity} onChange={event => setOpacity(Number(event.target.value))} /></div>
          </div>
        </div>
        <details className="border-t border-gray-700 pt-3 text-sm"><summary className="cursor-pointer">Source image and caption</summary>
          <p className="mt-2 break-all text-gray-400">{probe.source_path ?? probe.image_path}</p>
          <p className="mt-2 whitespace-pre-wrap">{probe.caption}</p>
        </details>
      </div>}
    </>}
    {(error || recordedError) && <div role="alert" className="rounded border border-red-700 bg-red-950/40 p-4 text-red-200">
      <p className="font-semibold">{error?.message ?? recordedError?.message}</p>
      <details className="mt-2"><summary className="cursor-pointer">Error details and stack trace</summary>
        <pre className="mt-2 overflow-auto whitespace-pre-wrap text-xs">{error?.stack ?? recordedError?.stack}</pre></details>
    </div>}
  </div>;
}
