'use client';

import Card from '@/components/Card';
import { Checkbox, NumberInput, SelectInput, TextAreaInput, TextInput } from '@/components/formInputs';
import type { JobConfig, SliderSpaceConfig } from '@/types';
import { parseSliderSpaceAutoStrengths, updateSliderSpaceConcepts } from './sliderspace';
import { cfgNegativeTextEnabled, flowTrainingModels } from './trainingCapabilities';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key?: string) => void;
  disabled?: boolean;
  datasetOptions?: { value: string; label: string }[];
};

export default function SliderSpaceEditor({ jobConfig, setJobConfig, disabled = false, datasetOptions = [] }: Props) {
  const value = jobConfig.config.process[0].sliderspace!;
  const model = jobConfig.config.process[0].model;
  const bucketDivisibility = flowTrainingModels[model.arch]?.bucketDivisibility ?? 32;
  const prompts = Array.isArray(value.concept_prompts) && value.concept_prompts.length > 0
    ? value.concept_prompts.map(prompt => typeof prompt === 'string' ? prompt : '') : [''];
  const set = (key: keyof SliderSpaceConfig, next: unknown) => setJobConfig(next, `config.process[0].sliderspace.${key}`);
  const updatePrompts = (next: string[]) => setJobConfig(updateSliderSpaceConcepts(jobConfig, next));
  const mode = value.discovery_mode ?? 'generated';
  const generates = mode !== 'provided';
  const provided = mode !== 'generated';
  const folders = Array.isArray(value.discovery_datasets) ? value.discovery_datasets : [];
  const changeMode = (next: string) => {
    const config = structuredClone(jobConfig);
    const settings = config.config.process[0].sliderspace!;
    settings.discovery_mode = next as SliderSpaceConfig['discovery_mode'];
    if (next !== 'generated' && !settings.discovery_datasets?.length) settings.discovery_datasets = [{ folder_path: '', default_caption: '' }];
    setJobConfig(config);
  };

  return (
    <Card title="SliderSpace">
      <fieldset disabled={disabled} className={disabled ? 'opacity-50' : ''}>
        <p className="mb-4 text-sm text-gray-400">
          Discover visual variations and train one LoRA per direction using your images, generated images, or both.
          This flow-model adaptation is experimental.
        </p>
        <SelectInput label="Discovery images" value={mode} disabled={disabled}
          options={[{ value: 'generated', label: 'Generated images' }, { value: 'provided', label: 'Provided images' }, { value: 'both', label: 'Provided + generated images' }]}
          onChange={changeMode} />
        {provided && <div className="my-4 space-y-3">
          {folders.map((folder, index) => <div key={index} className="rounded-lg border border-gray-700 p-3">
            <div className="flex justify-between items-center">
              <h3 className="text-sm font-medium">Discovery image folder {index + 1}</h3>
              <button type="button" disabled={disabled} aria-label={`Remove discovery image folder ${index + 1}`}
                className="text-xs text-red-400 hover:text-red-300"
                onClick={() => set('discovery_datasets', folders.filter((_, i) => i !== index))}>Remove</button>
            </div>
            <SelectInput label="Dataset folder" value={folder?.folder_path ?? ''} disabled={disabled}
              options={[{ value: '', label: 'Select dataset…' }, ...datasetOptions]}
              onChange={next => set('discovery_datasets', folders.map((item, i) => i === index ? { ...item, folder_path: next } : item))} />
            <TextInput label="Image folder path" value={folder?.folder_path ?? ''} disabled={disabled} required
              placeholder="/path/to/images"
              onChange={next => set('discovery_datasets', folders.map((item, i) => i === index ? { ...item, folder_path: next } : item))} />
            <TextAreaInput label="Default caption (optional)" value={folder?.default_caption ?? ''} disabled={disabled} rows={2}
              onChange={next => set('discovery_datasets', folders.map((item, i) => i === index ? { ...item, default_caption: next } : item))} />
          </div>)}
          <button type="button" disabled={disabled} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm"
            onClick={() => set('discovery_datasets', [...folders, { folder_path: '', default_caption: '' }])}>Add Discovery Image Folder</button>
          <p className="text-xs text-gray-400">All images in these folders and subfolders are included. Overlapping folders are deduplicated.
            Matching .txt captions take priority, then the folder default, then the first concept prompt (otherwise blank conditioning).
            Originals are never changed; copies are Mitchell-resized and center-cropped to fit their buckets.</p>
          <SelectInput label="Provided image sizing" value={value.discovery_buckets ? 'buckets' : 'square'} disabled={disabled}
            options={[{ value: 'buckets', label: 'Aspect-ratio buckets' }, { value: 'square', label: 'Square crop' }]}
            onChange={next => set('discovery_buckets', next === 'buckets')} />
          <p className="text-xs text-gray-400">Buckets keep image proportions with minimal cropping and {bucketDivisibility}-pixel alignment, up to resolution² pixels.
            Small images are not deliberately upscaled. Features use the full image with mean-color padding; differing shapes can still influence discovered directions.
            Generated images remain square.</p>
        </div>}
        <div className="space-y-3">
          {(generates ? prompts : prompts.slice(0, 1)).map((prompt, index) => (
            <div key={index}>
              <TextAreaInput label={index === 0 ? (generates ? 'Concept prompt' : 'Fallback concept prompt (optional)') : `Concept prompt ${index + 1}`}
                value={prompt} disabled={disabled} required={generates} rows={3} placeholder="A spaceship exploring deep space"
                onChange={next => updatePrompts(prompts.map((item, i) => i === index ? next : item))} />
              {generates && !prompt.trim() && <p className="mt-1 text-xs text-amber-400">Enter a concept prompt before creating the job.</p>}
              {generates && prompts.length > 1 && <button type="button" className="mt-1 text-xs text-gray-400 hover:text-gray-200"
                aria-label={`Remove concept prompt ${index + 1}`} onClick={() => updatePrompts(prompts.filter((_, i) => i !== index))}>
                Remove concept prompt {index + 1}
              </button>}
            </div>
          ))}
          {generates && <button type="button" onClick={() => updatePrompts([...prompts, ''])}
            className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-sm">Add Concept Prompt</button>}
          <p className="text-xs text-gray-400">{generates ? 'Use a broad concept with room to vary. All generated prompts contribute to the same discovery space.' : 'The first concept prompt is a fallback for provided images without captions. Descriptive captions are recommended.'}</p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
          <div>
            <NumberInput label="Directions to train" value={value.num_directions} onChange={next => set('num_directions', next)} min={1} max={64} step={1} clampOnBlur={false} required />
            <p className="mt-1 text-xs text-gray-400">Each direction produces its own LoRA.</p>
          </div>
          {generates && <div>
            <NumberInput label="Generated discovery images" value={value.discovery_samples} onChange={next => set('discovery_samples', next)} min={Math.max(mode === 'generated' ? value.num_directions + 1 : 1, prompts.length)} step={1} required />
            <p className="mt-1 text-xs text-gray-400">{provided ? 'Generated images are added to all provided images, not used as a total limit.' : 'A larger bank explores more variations and takes longer before training starts.'}</p>
          </div>}
          <div>
            <NumberInput label="Discovery resolution (pixels)" value={value.resolution} onChange={next => set('resolution', next)} min={128} step={32} required />
            <p className="mt-1 text-xs text-gray-400">{provided && value.discovery_buckets ? 'Bucket area budget: resolution² pixels. ' : 'Square images. '}Multiples of 32; higher resolutions use more memory during training.</p>
          </div>
        </div>
        <p className="my-4 text-sm text-gray-300">{provided ? `All images from ${folders.length} folder(s)${generates ? ` + ${value.discovery_samples} generated images` : ''}` : `${value.discovery_samples} generated discovery images`} → {value.num_directions} direction LoRAs.
          {provided && ` At least ${value.num_directions + 1} total images are required; checked recursively when the job starts.`}</p>
        <details className="rounded-lg border border-gray-700 p-3">
          <summary className="cursor-pointer text-sm">Discovery settings</summary>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
            {generates && <NumberInput label="Generation steps" value={value.discovery_steps} onChange={next => set('discovery_steps', next)} min={1} step={1} required />}
            <div>
              <NumberInput label="Discovery / training CFG" value={value.cfg_scale} onChange={next => set('cfg_scale', next)} min={1} required />
              <p className="mt-1 text-xs text-gray-400">Used for discovery and for both the base and adapted predictions during training. Preview guidance is separate.</p>
            </div>
            <div className="md:col-span-2">
              <TextAreaInput label="Negative prompt (optional)" value={value.negative_prompt} onChange={next => set('negative_prompt', next)} disabled={disabled || value.cfg_scale <= 1 || !cfgNegativeTextEnabled(model)} rows={2} />
              <p className="mt-1 text-xs text-gray-400">Used in discovery and both training predictions; inactive at CFG 1. This is not the negative end of a direction.</p>
            </div>
            <div>
              <NumberInput label="Discovery / training seed" value={value.seed} onChange={next => set('seed', next)} min={0} max={4294967295} step={1} required />
              <p className="mt-1 text-xs text-gray-400">Seeds generated discovery images and training image/noise choices; separate from the preview seed.</p>
            </div>
            <div>
              <TextInput label="Feature model name or path" value={value.feature_encoder} onChange={next => set('feature_encoder', next)} disabled={disabled} required />
              <p className="mt-1 text-xs text-gray-400">The first run may download this model. Feature extraction runs on {String(value.feature_device).toUpperCase()}.</p>
            </div>
            <div>
              <SelectInput label="Feature device" value={value.feature_device} disabled={disabled}
                options={[{ value: 'cpu', label: 'CPU (lower VRAM use)' }, { value: 'cuda', label: 'CUDA' }]}
                onChange={next => set('feature_device', next)} />
              <p className="mt-1 text-xs text-gray-400">CPU keeps the frozen feature model off the training GPU. CUDA uses additional GPU memory.</p>
            </div>
            <div>
              <NumberInput label="Semantic loss weight" value={value.loss_weight} min={0} clampOnBlur={false} required
                onChange={next => set('loss_weight', next)} />
              <p className="mt-1 text-xs text-gray-400">Scales the training objective; must be greater than zero. Start with 1.</p>
            </div>
          </div>
          <p className="mt-3 text-xs text-gray-400">{generates ? 'Generated discovery uses the native renderer, including when previews use ComfyUI. ' : ''}Matching discovery images are reused automatically. Image/caption modification times are part of the provided-image cache identity.</p>
        </details>
      </fieldset>
    </Card>
  );
}

export function SliderSpacePreview({ jobConfig, setJobConfig }: Props) {
  const value = jobConfig.config.process[0].sliderspace!;
  const strengths = parseSliderSpaceAutoStrengths(value.preview_auto_strengths === undefined ? '-1, 1' : value.preview_auto_strengths);
  const count = Number.isInteger(value.num_directions) && value.num_directions > 0 && value.num_directions <= 64 ? value.num_directions : 0;
  return <div className="mb-4 rounded-lg border border-gray-700 p-3">
    <Checkbox label="Auto sampling" checked={value.preview_auto ?? false}
      onChange={enabled => {
        const next = structuredClone(jobConfig);
        next.config.process[0].sliderspace!.preview_auto = enabled;
        if (enabled) next.config.process[0].sample.samples = [next.config.process[0].sample.samples[0] ?? { prompt: '' }];
        setJobConfig(next);
      }} />
    {value.preview_auto ? <>
      <TextInput label="Auto sample strengths" className="mt-3"
        value={value.preview_auto_strengths ?? '-1, 1'}
        onChange={next => setJobConfig(next, 'config.process[0].sliderspace.preview_auto_strengths')}
        placeholder="-1, 1" required />
      {strengths === null && <p className="mt-1 text-sm text-red-400" role="alert">Enter finite numbers separated by commas.</p>}
      <TextAreaInput label="Auto sample prompt" className="mt-3"
        value={jobConfig.config.process[0].sample.samples[0]?.prompt ?? ''}
        onChange={prompt => setJobConfig([{ ...jobConfig.config.process[0].sample.samples[0], prompt }], 'config.process[0].sample.samples')}
        placeholder="One prompt for the base model and every direction" rows={3} required />
      <p className="mt-2 text-xs text-gray-400">Renders one base image, then every direction at the listed strengths in order using the same prompt and seed{strengths !== null ? ` (${1 + strengths.length * count} images per sampling round)` : ''}. Zero and duplicate strengths are rendered once.</p>
    </> : <>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <SelectInput label="Preview direction" value={String(value.preview_direction)}
        options={Array.from({ length: count }, (_, i) => ({ value: String(i + 1), label: `Direction ${i + 1}` }))}
        onChange={next => setJobConfig(Number(next), 'config.process[0].sliderspace.preview_direction')} />
      <NumberInput label="Preview strength" value={value.preview_strength}
        onChange={next => setJobConfig(next, 'config.process[0].sliderspace.preview_strength')} />
    </div>
    <p className="mt-2 text-xs text-gray-400">Only the selected direction is rendered, once per sample prompt. Compare negative, zero (base), and positive strength using the same sample seed. All directions are exported.</p>
    </>}
  </div>;
}
