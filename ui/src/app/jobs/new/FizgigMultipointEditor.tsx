'use client';
import { FizgigMultipointConfig, FizgigMultipointEntry, FizgigSliderPoint } from '@/types';
import { NumberInput, TextAreaInput } from '@/components/formInputs';
import Card from '@/components/Card';
import { sortedPoints, strengthLabel } from './fizgigMultipoint';

const buttonClass = 'rounded bg-gray-700 px-4 py-2 text-white hover:bg-gray-600';

export function MultipointStrengths({ points, onChange }: {
  points: FizgigSliderPoint[]; onChange: (points: FizgigSliderPoint[]) => void;
}) {
  const invalid = points.some(point => !Number.isFinite(point.strength)) || new Set(points.map(point => point.strength)).size !== points.length
    || !points.some(point => point.strength !== 0);
  return <div className="my-4 space-y-3">
    {sortedPoints(points).map(point => <div key={point.id} className="flex flex-wrap items-end gap-3">
      <NumberInput label={`Strength ${strengthLabel(point.strength)}`} value={point.strength}
        onChange={value => { if (value !== null) onChange(points.map(p => p.id === point.id ? { ...p, strength: value } : p)); }} />
      <button type="button" className={buttonClass} disabled={points.length <= 1}
        onClick={() => onChange(points.filter(p => p.id !== point.id))}>Remove Point</button>
    </div>)}
    {invalid && <p className="text-sm text-red-400" role="alert">Use unique finite strengths and at least one nonzero target.</p>}
    <button type="button" className={buttonClass} onClick={() => onChange([...points, {
      id: `point-${Date.now()}-${Math.random().toString(36).slice(2)}`, strength: Math.floor(Math.max(0, ...points.map(p => p.strength))) + 1, prefix: '', negative_prefix: '',
    }])}>Add Point</button>
    <p className="text-sm text-gray-400">Zero is the base-model reference, not a training target or preservation anchor.
      More points take more training time and CPU cache storage, without loading additional models. Preview samples remain editable.</p>
  </div>;
}

export default function FizgigMultipointEditor({ config, onChange, cfgEnabled, onAddPreservationPrompt }: {
  config: FizgigMultipointConfig; onChange: (config: FizgigMultipointConfig) => void; cfgEnabled: boolean;
  onAddPreservationPrompt: () => void;
}) {
  const points = sortedPoints(config.points);
  const zero = points.find(point => point.strength === 0);
  const simple = config.prompt_entries.some(entry => entry.kind === 'simple');
  const setEntry = (index: number, entry: FizgigMultipointEntry) => onChange({ ...config,
    prompt_entries: config.prompt_entries.map((old, i) => i === index ? entry : old) });
  const addEntry = (entry: FizgigMultipointEntry) => onChange({ ...config, prompt_entries: [...config.prompt_entries, entry] });
  return <div className="space-y-4">
    <Card title="Shared Prefixes for Simplified Prompts">
      <p className="mb-4 text-sm text-gray-400">Each prefix is prepended to the base prompt followed by a blank line.
        Nonzero positive prefixes are required. Empty zero prefixes use the base prompt alone; empty CFG negatives remain empty.</p>
      {!zero && <TextAreaInput label="Neutral CFG Negative Prefix (0)" rows={4} value={config.neutral_negative_prefix}
        disabled={!cfgEnabled || !simple} onChange={value => onChange({ ...config, neutral_negative_prefix: value })} />}
      {points.map(point => <div key={point.id} className="mt-4 space-y-3">
        <Card title={`Strength ${strengthLabel(point.strength)}`}>
          <TextAreaInput label="Positive Prefix" rows={5} value={point.prefix} disabled={!simple}
            onChange={value => onChange({ ...config, points: config.points.map(p => p.id === point.id ? { ...p, prefix: value } : p) })} />
          <TextAreaInput label="CFG Negative Prefix (optional)" rows={4} value={point.negative_prefix} disabled={!cfgEnabled || !simple}
            onChange={value => onChange({ ...config, points: config.points.map(p => p.id === point.id ? { ...p, negative_prefix: value } : p) })} />
        </Card>
      </div>)}
    </Card>
    <p className="text-sm text-gray-400">Total practice images are distributed across these entries, not multiplied by target points.
      The student uses neutral conditioning; the frozen teacher uses each point's own prompts and CFG negative.</p>
    {config.prompt_entries.map((entry, index) => <div key={index} className="space-y-4">
      <div className="flex justify-between"><span>Prompt Entry {index + 1}</span>
        <button type="button" className={buttonClass} disabled={config.prompt_entries.length <= 1}
          onClick={() => onChange({ ...config, prompt_entries: config.prompt_entries.filter((_, i) => i !== index) })}>Remove Entry</button></div>
      {entry.kind === 'simple' ? <Card title="Simplified Base Prompt">
        <TextAreaInput label="Base Prompt" rows={8} value={entry.prompt} onChange={value => setEntry(index, { ...entry, prompt: value })} />
      </Card> : <>
        {!zero && <Card title="Neutral reference (base model)">
          <TextAreaInput label="Neutral Prompt" rows={6} value={entry.neutral_prompt}
            onChange={value => setEntry(index, { ...entry, neutral_prompt: value })} />
          <TextAreaInput label="Neutral CFG Negative (optional)" rows={4} value={entry.neutral_negative_prompt} disabled={!cfgEnabled}
            onChange={value => setEntry(index, { ...entry, neutral_negative_prompt: value })} />
        </Card>}
        {points.map(point => {
          const target = entry.targets.find(target => target.point_id === point.id);
          const update = (key: 'prompt' | 'negative_prompt', value: string) => setEntry(index, { ...entry,
            targets: points.map(p => p.id === point.id ? { point_id: p.id, prompt: target?.prompt ?? '', negative_prompt: target?.negative_prompt ?? '', [key]: value }
              : entry.targets.find(t => t.point_id === p.id) ?? { point_id: p.id, prompt: '', negative_prompt: '' }) });
          return <Card key={point.id} title={`Strength ${strengthLabel(point.strength)}`}>
            <TextAreaInput label="Positive Prompt" rows={6} value={target?.prompt ?? ''} onChange={value => update('prompt', value)} />
            <TextAreaInput label="CFG Negative (optional)" rows={4} value={target?.negative_prompt ?? ''} disabled={!cfgEnabled}
              onChange={value => update('negative_prompt', value)} />
          </Card>;
        })}
      </>}
    </div>)}
    <div className="flex flex-wrap gap-3">
      <button type="button" className={buttonClass} onClick={() => addEntry({ kind: 'simple', prompt: '' })}>Add Simplified Prompt</button>
      <button type="button" className={buttonClass} onClick={() => addEntry({ kind: 'specific', neutral_prompt: '', neutral_negative_prompt: '',
        targets: points.map(p => ({ point_id: p.id, prompt: '', negative_prompt: '' })) })}>Add Specific Prompts</button>
      <button type="button" className={buttonClass} onClick={onAddPreservationPrompt}>Add Preservation Prompt</button>
    </div>
  </div>;
}
