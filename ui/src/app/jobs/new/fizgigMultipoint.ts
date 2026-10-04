import { FizgigMultipointConfig, FizgigPromptEntry, FizgigSliderPoint, JobConfig } from '@/types';

export const sortedPoints = (points: FizgigSliderPoint[]) => [...points].sort((a, b) => a.strength - b.strength);
export const strengthLabel = (strength: number) => strength === 0 ? '0 — Neutral reference (base model)' : `${strength > 0 ? '+' : ''}${strength}`;

export function multipointNeutralPrompt(config: FizgigMultipointConfig | undefined): string | undefined {
  const entry = config?.prompt_entries[0];
  if (!entry) return undefined;
  const zero = config!.points.find(point => point.strength === 0);
  if (entry.kind === 'specific') return zero ? entry.targets.find(target => target.point_id === zero.id)?.prompt : entry.neutral_prompt;
  const base = entry.prompt.trim();
  return zero?.prefix.trim() ? `${zero.prefix.trim()}\n\n${base}` : base;
}

export function toggleMultipoint(config: JobConfig, enabled: boolean): JobConfig {
  const next = structuredClone(config);
  const process = next.config.process[0];
  const slider = process.fizgig_slider!;
  slider.multipoint = enabled;
  if (!enabled || slider.multipoint_config) return next;
  const legacy: FizgigPromptEntry[] = slider.prompt_entries
    ?? slider.prompt_triplets?.map(entry => ({ kind: 'specific', ...entry }))
    ?? [{ kind: 'specific', neutral_prompt: slider.neutral_prompt ?? '', positive_prompt: slider.positive_prompt ?? '',
      negative_prompt: slider.negative_prompt ?? '', cfg_negative_prompt: slider.cfg_negative_prompt,
      cfg_negative_prompt_positive: slider.cfg_negative_prompt_positive, cfg_negative_prompt_negative: slider.cfg_negative_prompt_negative }];
  slider.multipoint_config = {
    points: [
      { id: 'minus', strength: -1, prefix: slider.negative_prefix ?? '', negative_prefix: slider.cfg_negative_prefix_negative ?? slider.cfg_negative_prefix ?? '' },
      { id: 'plus', strength: 1, prefix: slider.positive_prefix ?? '', negative_prefix: slider.cfg_negative_prefix_positive ?? slider.cfg_negative_prefix ?? '' },
      { id: 'extra', strength: 2, prefix: '', negative_prefix: '' },
    ],
    neutral_negative_prefix: slider.cfg_negative_prefix ?? '',
    prompt_entries: legacy.map(entry => entry.kind === 'simple' ? { ...entry } : {
      kind: 'specific', neutral_prompt: entry.neutral_prompt, neutral_negative_prompt: entry.cfg_negative_prompt ?? '',
      targets: [
        { point_id: 'minus', prompt: entry.negative_prompt, negative_prompt: entry.cfg_negative_prompt_negative ?? entry.cfg_negative_prompt ?? '' },
        { point_id: 'plus', prompt: entry.positive_prompt, negative_prompt: entry.cfg_negative_prompt_positive ?? entry.cfg_negative_prompt ?? '' },
        { point_id: 'extra', prompt: '', negative_prompt: '' },
      ],
    }),
  };
  process.datasets.forEach(dataset => {
    dataset.multipoint_images ??= [
      { point_id: 'minus', folder_path: dataset.control_path_1 ?? '' },
      { point_id: 'plus', folder_path: dataset.folder_path },
      { point_id: 'extra', folder_path: '' },
    ];
  });
  const prompt = process.type === 'fizgig_prompt_slider'
    ? (legacy[0]?.kind === 'simple' ? legacy[0].prompt : legacy[0]?.neutral_prompt ?? '')
    : process.datasets[0]?.default_caption || 'a detailed illustration';
  process.sample.walk_seed = false;
  process.sample.samples = [-1, 0, 1, 2].map(network_multiplier => ({ prompt, network_multiplier }));
  return next;
}

/** Stable IDs carry all content across numerical edits/reordering. */
export function updateMultipointPoints(config: JobConfig, points: FizgigSliderPoint[]): JobConfig {
  points = sortedPoints(points);
  const next = structuredClone(config);
  const process = next.config.process[0];
  const multipoint = process.fizgig_slider!.multipoint_config!;
  const oldZero = multipoint.points.find(point => point.strength === 0);
  multipoint.prompt_entries = multipoint.prompt_entries.map(entry => {
    if (entry.kind === 'simple') return entry;
    const previousNeutral = oldZero ? entry.targets.find(target => target.point_id === oldZero.id) : undefined;
    const neutral_prompt = previousNeutral?.prompt ?? entry.neutral_prompt;
    const neutral_negative_prompt = previousNeutral?.negative_prompt ?? entry.neutral_negative_prompt;
    return { ...entry, neutral_prompt, neutral_negative_prompt, targets: points.map(point =>
      entry.targets.find(target => target.point_id === point.id) ?? { point_id: point.id,
        prompt: point.strength === 0 ? neutral_prompt : '', negative_prompt: point.strength === 0 ? neutral_negative_prompt : '' }) };
  });
  process.datasets.forEach(dataset => {
    dataset.multipoint_images = points.map(point => dataset.multipoint_images?.find(target => target.point_id === point.id)
      ?? { point_id: point.id, folder_path: '' });
  });
  multipoint.points = sortedPoints(points);
  return next;
}

/** Prompt sets can contain unfinished text, but never ambiguous point mappings. */
export function validateMultipointConfig(value: unknown): FizgigMultipointConfig | null {
  if (!value || typeof value !== 'object') return null;
  const data = value as FizgigMultipointConfig;
  const text = (value: unknown): value is string => typeof value === 'string' && value.length <= 100_000;
  if (!Array.isArray(data.points) || data.points.length < 1 || data.points.length > 128 || !text(data.neutral_negative_prefix)
      || !Array.isArray(data.prompt_entries) || data.prompt_entries.length < 1 || data.prompt_entries.length > 128) return null;
  const ids = new Set<string>(), strengths = new Set<number>();
  const points: FizgigSliderPoint[] = [];
  for (const point of data.points) {
    if (!point || !text(point.id) || !point.id || point.id.length > 128 || ids.has(point.id)
        || typeof point.strength !== 'number' || !Number.isFinite(point.strength) || strengths.has(point.strength)
        || !text(point.prefix) || !text(point.negative_prefix)) return null;
    ids.add(point.id); strengths.add(point.strength);
    points.push({ id: point.id, strength: point.strength, prefix: point.prefix, negative_prefix: point.negative_prefix });
  }
  if (![...strengths].some(strength => strength !== 0)) return null;
  const entries: FizgigMultipointConfig['prompt_entries'] = [];
  for (const entry of data.prompt_entries) {
    if (!entry) return null;
    if (entry.kind === 'simple' && text(entry.prompt)) entries.push({ kind: 'simple', prompt: entry.prompt });
    else if (entry.kind === 'specific' && text(entry.neutral_prompt) && text(entry.neutral_negative_prompt)
        && Array.isArray(entry.targets) && entry.targets.length === points.length) {
      const used = new Set<string>();
      const targets = [];
      for (const target of entry.targets) {
        if (!target || !ids.has(target.point_id) || used.has(target.point_id) || !text(target.prompt) || !text(target.negative_prompt)) return null;
        used.add(target.point_id);
        targets.push({ point_id: target.point_id, prompt: target.prompt, negative_prompt: target.negative_prompt });
      }
      entries.push({ kind: 'specific', neutral_prompt: entry.neutral_prompt, neutral_negative_prompt: entry.neutral_negative_prompt, targets });
    } else return null;
  }
  return { points: sortedPoints(points), neutral_negative_prefix: data.neutral_negative_prefix, prompt_entries: entries };
}
