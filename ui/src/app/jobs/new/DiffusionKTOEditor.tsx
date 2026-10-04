'use client';

import type { JobConfig } from '@/types';
import Card from '@/components/Card';
import { NumberInput, SelectInput } from '@/components/formInputs';
import { defaultDiffusionKTOConfig } from './trainingCapabilities';

export default function DiffusionKTOEditor({ jobConfig, setJobConfig }: {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key?: string) => void;
}) {
  const settings = { ...defaultDiffusionKTOConfig, ...jobConfig.config.process[0].diffusion_kto };
  return <Card title="Diffusion-KTO (experimental)">
    <p className="mb-4 text-sm text-gray-400">
      Label each image folder Liked or Disliked below. Images use their own captions and buckets;
      filenames, sizes and class counts do not need to match. This flow-matching adaptation uses
      velocity error as a surrogate, not a validated diffusion likelihood. Use a base checkpoint.
    </p>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <NumberInput label="Beta" value={settings.beta} min={0.000001} required
        onChange={value => setJobConfig(value, 'config.process[0].diffusion_kto.beta')} />
      <NumberInput label="Liked Weight" value={settings.liked_weight} min={0.000001} required
        onChange={value => setJobConfig(value, 'config.process[0].diffusion_kto.liked_weight')} />
      <NumberInput label="Disliked Weight" value={settings.disliked_weight} min={0.000001} required
        onChange={value => setJobConfig(value, 'config.process[0].diffusion_kto.disliked_weight')} />
    </div>
    <SelectInput label="Reference-point Estimator" className="pt-4" value={settings.reference_estimator}
      options={[{ value: 'batch_mean', label: 'Batch mean (published default)' },
        { value: 'score_window', label: 'Pooled score window (for small batches)' }]}
      onChange={value => setJobConfig(value, 'config.process[0].diffusion_kto.reference_estimator')} />
    {settings.reference_estimator === 'score_window' ? <>
      <NumberInput label="Score Window (accumulation batches)" className="pt-4"
        value={settings.score_window_size} min={2} required
        onChange={value => setJobConfig(value, 'config.process[0].diffusion_kto.score_window_size')} />
      <p className="mt-2 text-sm text-gray-400">
        Set Gradient Accumulation below to at least {settings.score_window_size}. Each window is scored
        before gradients are replayed at unchanged weights, retaining CPU inputs, not activation graphs.
        Short final windows use their actual image count. Checkpoints occur only after complete optimizer steps.
      </p>
    </> : <p className="mt-2 text-sm text-gray-400">
      The detached reference point is estimated separately within each batch. With batch size 1 this
      is a single-image estimate; choose a pooled score window for a broader estimate. No moving average is used.
    </p>}
    <p className="mt-2 text-sm text-gray-400">
      Utility weights balance feedback strength; dataset repeats change sampling frequency. Single-class
      datasets are allowed but can drift. Text embeddings and latents must be cached; edit references,
      pretrained adapters, dropout and additional training losses are not supported.
    </p>
  </Card>;
}
