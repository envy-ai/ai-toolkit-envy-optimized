'use client';
import { useEffect, useMemo, useState } from 'react';
import {
  ModelArch,
  quantizationOptions,
  defaultQtype,
  jobTypeOptions,
  SampleTags,
} from './options';
import { useModelArchs } from '@/extensions/modelArchs';
import { defaultCompileOptions, defaultDatasetConfig, defaultSliderConfig } from './jobConfig';
import { FizgigPromptEntry, FizgigSliderConfig, GroupedSelectOption, JobConfig, SelectOption } from '@/types';
import { objectCopy, tagsToObj, objToTags } from '@/utils/basic';
import {
  TextInput,
  TextAreaInput,
  SelectInput,
  Checkbox,
  FormGroup,
  NumberInput,
  SliderInput,
  CreatableSelectInput,
} from '@/components/formInputs';
import Card from '@/components/Card';
import { X, Copy, Wand2, SquareDashed, Info } from 'lucide-react';
import { openDoc } from '@/components/DocModal';
import { openUpsamplePromptsModal, toAspectRatio } from '@/components/UpsamplePromptsModal';
import { openPromptBoxEditor } from '@/components/PromptBoxEditorModal';
import AddSingleImageModal, { openAddImageModal } from '@/components/AddSingleImageModal';
import SampleControlImage from '@/components/SampleControlImage';
import { FlipHorizontal2, FlipVertical2 } from 'lucide-react';
import { handleModelArchChange } from './utils';
import { IoFlaskSharp } from 'react-icons/io5';
import { isMac } from '@/helpers/basic';
import { apiClient } from '@/utils/api';
import { Modal } from '@/components/Modal';
import FizgigMultipointEditor, { MultipointStrengths } from './FizgigMultipointEditor';
import SliderSpaceEditor, { SliderSpacePreview } from './SliderSpaceEditor';
import DiffusionKTOEditor from './DiffusionKTOEditor';
import { multipointNeutralPrompt, sortedPoints, strengthLabel, toggleMultipoint, updateMultipointPoints } from './fizgigMultipoint';
import { canonicalTrainingMode, cfgNegativeTextEnabled, flowTrainingModels, isSpecializedTrainingMode,
  switchSpecializedArchitecture, trainingModeVisible } from './trainingCapabilities';

type Props = {
  jobConfig: JobConfig;
  setJobConfig: (value: any, key?: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDs: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  sampleOnlyMode?: boolean;
  isLoading?: boolean;
};

const isDev = process.env.NODE_ENV === 'development';

type ComfyOptions = {
  model: string[];
  vae: string[];
  text_encoder: string[];
  inference_lora: string[];
  sampler: string[];
  scheduler: string[];
  output_format: string[];
  output_quality: string[];
};

const emptyComfyOptions: ComfyOptions = {
  model: [],
  vae: [],
  text_encoder: [],
  inference_lora: [],
  sampler: [],
  scheduler: [],
  output_format: [],
  output_quality: [],
};

const toOptions = (values: string[]): SelectOption[] => values.map(value => ({ value, label: value }));

type ModelPathSource = 'standard' | 'comfy_checkpoint';

const modelPathSourceOptions: SelectOption[] = [
  { value: 'standard', label: 'AI Toolkit / Hugging Face' },
  { value: 'comfy_checkpoint', label: 'ComfyUI Checkpoint Path' },
];

const comfyWorkflowOptions: SelectOption[] = [
  { value: 'config/comfy_templates/anima_lora_sample.json.njk', label: 'Anima LoRA image' },
  { value: 'config/comfy_templates/ideogram4_lora_sample.json.njk', label: 'Ideogram 4 LoRA image' },
  {
    value: 'config/comfy_templates/krea2_lora_sample.json.njk',
    label: 'Krea 2 LoRA image',
  },
  {
    value: 'config/comfy_templates/qwen_image_2_lora_sample.json.njk',
    label: 'Qwen Image 2.1 LoRA image',
  },
  {
    value: 'config/comfy_templates/qwen_image_edit_lora_sample.json.njk',
    label: 'Qwen Image Edit LoRA',
  },
  {
    value: 'config/comfy_templates/qwen_image_edit_plus_lora_sample.json.njk',
    label: 'Qwen Image Edit Plus LoRA',
  },
  {
    value: 'config/comfy_templates/minimax_h3_fl2v_lora_sample.json.njk',
    label: 'MiniMax H3 T2V / FL2V LoRA video',
  },
];

const neutralPromptForEntry = (entry: FizgigPromptEntry | undefined): string | undefined =>
  entry?.kind === 'simple' ? entry.prompt : entry?.neutral_prompt;

type SavedPromptSet = Pick<FizgigSliderConfig,
  'positive_prefix' | 'negative_prefix' | 'cfg_negative_prefix' |
  'cfg_negative_prefix_positive' | 'cfg_negative_prefix_negative'> & {
  version: 1;
  prompt_entries: FizgigPromptEntry[];
  anchor_prompts?: FizgigSliderConfig['anchor_prompts'];
  multipoint?: boolean;
  multipoint_config?: FizgigSliderConfig['multipoint_config'];
};

const promptSetErrorMessage = (error: unknown): string =>
  (error as { response?: { data?: { error?: string } } })?.response?.data?.error ??
  'Could not reach the prompt sets service.';

const promptSetAlreadyExists = (error: unknown): boolean =>
  (error as { response?: { status?: number } })?.response?.status === 409;

export default function SimpleJob({
  jobConfig,
  setJobConfig,
  handleSubmit,
  status,
  runId,
  gpuIDs,
  setGpuIDs,
  gpuList,
  datasetOptions,
  sampleOnlyMode = false,
  isLoading,
}: Props) {
  const { archs: modelArchs, groupedModelOptions } = useModelArchs();
  const modelArch = useMemo(() => {
    return modelArchs.find(a => a.name === jobConfig.config.process[0].model.arch) as ModelArch;
  }, [modelArchs, jobConfig.config.process[0].model.arch]);

  const jobType = useMemo(() => {
    return jobTypeOptions.find(j => j.value === jobConfig.config.process[0].type);
  }, [jobConfig.config.process[0].type]);

  const disableSections = useMemo(() => {
    let sections: string[] = [];
    if (modelArch?.disableSections) {
      sections = sections.concat(modelArch.disableSections);
    }
    if (jobType?.disableSections) {
      sections = sections.concat(jobType.disableSections);
    }
    return sections;
  }, [modelArch, jobType]);

  const isVideoModel = !!(modelArch?.group === 'video');
  const isAudioModel = !!(modelArch?.group === 'audio');
  const isPromptSlider = jobConfig.config.process[0].type === 'slider';
  const isFizgigImageSlider = jobConfig.config.process[0].type === 'fizgig_image_slider';
  const isFizgigPromptSlider = jobConfig.config.process[0].type === 'fizgig_prompt_slider';
  const isFizgigSlider = isFizgigImageSlider || isFizgigPromptSlider;
  const isQwenFlowDPO = canonicalTrainingMode(jobConfig.config.process[0].type) === 'flow_dpo';
  const isQwenGuidanceDistillation = canonicalTrainingMode(jobConfig.config.process[0].type) === 'guidance_distillation';
  const isSliderSpace = jobConfig.config.process[0].type === 'sliderspace';
  const isDiffusionKTO = jobConfig.config.process[0].type === 'diffusion_kto';
  const trainingModel = jobConfig.config.process[0].model;
  const specializedMode = isSpecializedTrainingMode(jobConfig.config.process[0].type);
  const textNegativesEnabled = cfgNegativeTextEnabled(trainingModel);
  const editSourcesEnabled = !!flowTrainingModels[trainingModel.arch]?.editReferences
    && (trainingModel.arch !== 'krea2' || !!trainingModel.model_kwargs?.edit);
  const isPairedImageTraining = isFizgigImageSlider || isQwenFlowDPO;
  const fizgigSlider = jobConfig.config.process[0].fizgig_slider;
  const isMultipoint = fizgigSlider?.multipoint ?? false;
  const multipointConfig = fizgigSlider?.multipoint_config;
  const promptEntries: FizgigPromptEntry[] = fizgigSlider?.prompt_entries
    ?? fizgigSlider?.prompt_triplets?.map(triplet => ({ kind: 'specific' as const, ...triplet }))
    ?? [{
      kind: 'specific',
      neutral_prompt: fizgigSlider?.neutral_prompt ?? '',
      positive_prompt: fizgigSlider?.positive_prompt ?? '',
      negative_prompt: fizgigSlider?.negative_prompt ?? '',
      cfg_negative_prompt: fizgigSlider?.cfg_negative_prompt ?? '',
      cfg_negative_prompt_positive: fizgigSlider?.cfg_negative_prompt_positive ?? fizgigSlider?.cfg_negative_prompt ?? '',
      cfg_negative_prompt_negative: fizgigSlider?.cfg_negative_prompt_negative ?? fizgigSlider?.cfg_negative_prompt ?? '',
    }];
  const hasSimplifiedPrompts = promptEntries.some(entry => entry.kind === 'simple');
  const sliderCfgEnabled = (fizgigSlider?.cfg_scale ?? 1) > 1 && textNegativesEnabled;
  const anchorPrompts = fizgigSlider?.anchor_prompts ?? [];
  const addPreservationPrompt = () => setJobConfig(
    [...anchorPrompts, { prompt: '', negative_prompt: '' }],
    'config.process[0].fizgig_slider.anchor_prompts',
  );
  const updatePromptEntries = (nextEntries: FizgigPromptEntry[]) => {
    const updated = objectCopy(jobConfig);
    const process = updated.config.process[0];
    if (!process.fizgig_slider) return;
    process.fizgig_slider.prompt_entries = nextEntries;
    // Migrate both earlier triplet formats when first edited.
    delete process.fizgig_slider.prompt_triplets;
    delete process.fizgig_slider.neutral_prompt;
    delete process.fizgig_slider.positive_prompt;
    delete process.fizgig_slider.negative_prompt;
    delete process.fizgig_slider.cfg_negative_prompt;
    delete process.fizgig_slider.cfg_negative_prompt_positive;
    delete process.fizgig_slider.cfg_negative_prompt_negative;
    const oldNeutral = neutralPromptForEntry(promptEntries[0]);
    const newNeutral = neutralPromptForEntry(nextEntries[0]);
    if (oldNeutral !== newNeutral && newNeutral !== undefined) {
      process.sample.samples = process.sample.samples.map(sample =>
        sample.prompt === oldNeutral ? { ...sample, prompt: newNeutral } : sample,
      );
    }
    process.fizgig_slider.bank_size = Math.max(process.fizgig_slider.bank_size ?? 16, nextEntries.length);
    setJobConfig(updated);
  };
  const selectedOptimizer = jobConfig.config.process[0].train.optimizer ?? '';
  const showProdigyOptimizerParams = selectedOptimizer.toLowerCase().startsWith('prodigy');
  const networkType = jobConfig.config.process[0].network?.type ?? 'lora';
  const frequencyLossType = jobConfig.config.process[0].train.frequency_loss_type ?? 'none';
  const showNetworkConv = !isFizgigSlider && (networkType == 'dora' || !disableSections.includes('network.conv'));
  const comfyConfig = jobConfig.config.process[0].sample.comfy;
  const comfyEnabled = comfyConfig?.enabled || false;
  const [comfyOptions, setComfyOptions] = useState<ComfyOptions>(emptyComfyOptions);
  const [promptSetNames, setPromptSetNames] = useState<string[]>([]);
  const [selectedPromptSet, setSelectedPromptSet] = useState('');
  const [activePromptSetName, setActivePromptSetName] = useState('');
  const [promptSetBusy, setPromptSetBusy] = useState(false);
  const [promptSetError, setPromptSetError] = useState('');
  const [promptSetNotice, setPromptSetNotice] = useState('');
  const [savePromptSetOpen, setSavePromptSetOpen] = useState(false);
  const [newPromptSetName, setNewPromptSetName] = useState('');
  const [modelPathSource, setModelPathSource] = useState<ModelPathSource>(() => {
    const modelPath = jobConfig.config.process[0].model.name_or_path ?? '';
    return modelPath.startsWith('/') && modelPath.endsWith('.safetensors') ? 'comfy_checkpoint' : 'standard';
  });
  const comfyCheckpointPathSelected = modelPathSource === 'comfy_checkpoint';

  useEffect(() => {
    if (!isFizgigPromptSlider) return;
    let cancelled = false;
    apiClient.get<{ names: string[] }>('/api/prompt-sets')
      .then(response => {
        if (!cancelled) setPromptSetNames(response.data.names.sort((a, b) =>
          a.localeCompare(b, undefined, { sensitivity: 'base' }) || a.localeCompare(b)));
      })
      .catch(error => {
        if (!cancelled) setPromptSetError(promptSetErrorMessage(error));
      });
    return () => { cancelled = true; };
  }, [isFizgigPromptSlider]);

  const applyPromptSet = (promptSet: SavedPromptSet) => {
    const updated = objectCopy(jobConfig);
    const process = updated.config.process[0];
    if (!process.fizgig_slider) return;
    const oldNeutral = isMultipoint ? multipointNeutralPrompt(multipointConfig) : neutralPromptForEntry(promptEntries[0]);
    const newNeutral = promptSet.multipoint ? multipointNeutralPrompt(promptSet.multipoint_config) : neutralPromptForEntry(promptSet.prompt_entries[0]);
    Object.assign(process.fizgig_slider, {
      positive_prefix: promptSet.positive_prefix,
      negative_prefix: promptSet.negative_prefix,
      cfg_negative_prefix: promptSet.cfg_negative_prefix,
      cfg_negative_prefix_positive: promptSet.cfg_negative_prefix_positive,
      cfg_negative_prefix_negative: promptSet.cfg_negative_prefix_negative,
      prompt_entries: promptSet.prompt_entries,
      anchor_prompts: promptSet.anchor_prompts ?? [],
      multipoint: promptSet.multipoint ?? false,
      ...(promptSet.multipoint_config ? { multipoint_config: promptSet.multipoint_config } : {}),
    });
    // Remove legacy prompt fields so they cannot override the loaded entries.
    delete process.fizgig_slider.prompt_triplets;
    delete process.fizgig_slider.neutral_prompt;
    delete process.fizgig_slider.positive_prompt;
    delete process.fizgig_slider.negative_prompt;
    delete process.fizgig_slider.cfg_negative_prompt;
    delete process.fizgig_slider.cfg_negative_prompt_positive;
    delete process.fizgig_slider.cfg_negative_prompt_negative;
    if (oldNeutral !== newNeutral && newNeutral !== undefined) {
      process.sample.samples = process.sample.samples.map(sample =>
        sample.prompt === oldNeutral ? { ...sample, prompt: newNeutral } : sample,
      );
    }
    process.fizgig_slider.bank_size = Math.max(process.fizgig_slider.bank_size ?? 16,
      promptSet.multipoint ? promptSet.multipoint_config?.prompt_entries.length ?? 1 : promptSet.prompt_entries.length);
    setJobConfig(updated);
  };

  const loadPromptSet = async () => {
    if (!selectedPromptSet || promptSetBusy) return;
    setPromptSetBusy(true);
    setPromptSetError('');
    setPromptSetNotice('');
    try {
      const response = await apiClient.get<{ prompt_set: SavedPromptSet }>('/api/prompt-sets', {
        params: { name: selectedPromptSet },
      });
      applyPromptSet(response.data.prompt_set);
      setActivePromptSetName(selectedPromptSet);
      setPromptSetNotice(`Loaded “${selectedPromptSet}”.`);
    } catch (error) {
      setPromptSetError(promptSetErrorMessage(error));
    } finally {
      setPromptSetBusy(false);
    }
  };

  const deletePromptSet = async () => {
    const name = selectedPromptSet;
    if (!name || promptSetBusy) return;
    if (!window.confirm(`Delete saved prompt set “${name}”? This cannot be undone. The prompts currently on this form will remain.`)) return;
    setPromptSetBusy(true);
    setPromptSetError('');
    setPromptSetNotice('');
    try {
      await apiClient.delete('/api/prompt-sets', { params: { name } });
      setPromptSetNames(previous => previous.filter(savedName => savedName !== name));
      setSelectedPromptSet('');
      if (activePromptSetName === name) setActivePromptSetName('');
      setPromptSetNotice(`Deleted “${name}”. Current form prompts were not changed.`);
    } catch (error) {
      setPromptSetError(promptSetErrorMessage(error));
    } finally {
      setPromptSetBusy(false);
    }
  };

  const savePromptSet = async (requestedName: string, closeModal: boolean) => {
    const name = requestedName.trim();
    if (!name || promptSetBusy) return;
    const promptSet: SavedPromptSet = {
      version: 1,
      prompt_entries: promptEntries,
      anchor_prompts: anchorPrompts,
      multipoint: isMultipoint,
      ...(multipointConfig ? { multipoint_config: multipointConfig } : {}),
      positive_prefix: fizgigSlider?.positive_prefix ?? '',
      negative_prefix: fizgigSlider?.negative_prefix ?? '',
      cfg_negative_prefix: fizgigSlider?.cfg_negative_prefix ?? '',
      cfg_negative_prefix_positive: fizgigSlider?.cfg_negative_prefix_positive ?? fizgigSlider?.cfg_negative_prefix ?? '',
      cfg_negative_prefix_negative: fizgigSlider?.cfg_negative_prefix_negative ?? fizgigSlider?.cfg_negative_prefix ?? '',
    };
    setPromptSetBusy(true);
    setPromptSetError('');
    setPromptSetNotice('');
    try {
      try {
        await apiClient.post('/api/prompt-sets', { name, prompt_set: promptSet });
      } catch (error) {
        if (!promptSetAlreadyExists(error)) throw error;
        if (!window.confirm(`Overwrite prompt set “${name}”? This will replace its saved prompts and prefixes.`)) return;
        await apiClient.post('/api/prompt-sets', { name, prompt_set: promptSet, overwrite: true });
      }
      setPromptSetNames(previous => [...new Set([...previous, name])].sort((a, b) =>
        a.localeCompare(b, undefined, { sensitivity: 'base' }) || a.localeCompare(b)));
      setSelectedPromptSet(name);
      setActivePromptSetName(name);
      if (closeModal) {
        setSavePromptSetOpen(false);
        setNewPromptSetName('');
      }
      setPromptSetNotice(`Saved “${name}”.`);
    } catch (error) {
      setPromptSetError(promptSetErrorMessage(error));
    } finally {
      setPromptSetBusy(false);
    }
  };

  useEffect(() => {
    const modelPath = jobConfig.config.process[0].model.name_or_path ?? '';
    if (modelPath.startsWith('/') && modelPath.endsWith('.safetensors')) {
      setModelPathSource('comfy_checkpoint');
    } else if (modelPath !== '' && !modelPath.startsWith('/')) {
      setModelPathSource('standard');
    }
  }, [jobConfig.config.process[0].model.name_or_path]);

  useEffect(() => {
    if (networkType != 'dora') {
      return;
    }

    const network = jobConfig.config.process[0].network;
    const convValue = network?.conv ?? 16;

    if (network?.conv === undefined || network?.conv === null) {
      setJobConfig(16, 'config.process[0].network.conv');
    }
    if (network?.conv_alpha === undefined || network?.conv_alpha === null) {
      setJobConfig(convValue, 'config.process[0].network.conv_alpha');
    }
  }, [
    networkType,
    jobConfig.config.process[0].network?.conv,
    jobConfig.config.process[0].network?.conv_alpha,
    setJobConfig,
  ]);

  useEffect(() => {
    if (!comfyEnabled) {
      setComfyOptions(emptyComfyOptions);
      return;
    }

    let isCancelled = false;
    apiClient
      .get('/api/comfy/options', {
        params: {
          url: comfyConfig?.api_url || 'http://127.0.0.1:8188',
        },
      })
      .then(res => {
        if (!isCancelled) {
          setComfyOptions({ ...emptyComfyOptions, ...res.data });
        }
      })
      .catch(() => {
        if (!isCancelled) {
          setComfyOptions(emptyComfyOptions);
        }
      });

    return () => {
      isCancelled = true;
    };
  }, [comfyEnabled, comfyConfig?.api_url]);

  const taggedSampleArr: Record<string, any>[] | null = useMemo(() => {
    if (!modelArch) return null;
    if (!modelArch.sampleTags) return null;
    if (!jobConfig.config.process[0].sample.samples) return null;
    let sampleArr: any[] = [];
    for (let i = 0; i < jobConfig.config.process[0].sample.samples.length; i++) {
      const taggedPrompt = jobConfig.config.process[0].sample.samples[i].prompt;
      const tagsObj = tagsToObj(taggedPrompt);
      sampleArr.push(tagsObj);
    }
    return sampleArr;
  }, [modelArch, jobConfig.config.process[0].sample.samples]);

  const modelArchTagSections: SampleTags[] | null = useMemo(() => {
    if (!modelArch?.sampleTags) return null;
    const maxPerGroup = 5;
    let sections: SampleTags[] = [];
    let subSection: SampleTags = {};
    for (const [tagKey, tag] of Object.entries(modelArch.sampleTags)) {
      if ((tag.full && Object.keys(subSection).length > 0) || Object.keys(subSection).length >= maxPerGroup) {
        // reset the sub section build if the next tag is full or max per group is reached
        sections.push(subSection);
        subSection = {};
      }
      subSection[tagKey] = tag;
      if (tag.full) {
        // if the tag is full, push the section immediately and reset the sub section build
        sections.push(subSection);
        subSection = {};
      }
    }
    if (Object.keys(subSection).length > 0) {
      sections.push(subSection);
    }
    return sections.length > 0 ? sections : null;
  }, [modelArch]);

  const numTopCards = useMemo(() => {
    let count = 4; // job settings, model config, target config, save config
    if (modelArch?.additionalSections?.includes('model.multistage')) {
      count += 1; // add multistage card
    }
    if (!disableSections.includes('model.quantize')) {
      count += 1; // add quantization card
    }
    if (!disableSections.includes('slider')) {
      count += 1; // add slider card
    }
    return count;
  }, [modelArch, disableSections]);

  let topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 xl:grid-cols-4 gap-6';

  if (numTopCards == 5) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6';
  }
  if (numTopCards == 6) {
    topBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-3 2xl:grid-cols-6 gap-6';
  }

  const numTrainingCols = useMemo(() => {
    let count = 4;
    if (!disableSections.includes('train.diff_output_preservation')) {
      count += 1;
    }
    return count;
  }, [disableSections]);

  let trainingBarClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';

  if (numTrainingCols == 5) {
    trainingBarClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }
  if (isSliderSpace) trainingBarClass = 'grid grid-cols-1 md:grid-cols-2 gap-6';

  const transformerQuantizationOptions: GroupedSelectOption[] | SelectOption[] = useMemo(() => {
    const hasARA = modelArch?.accuracyRecoveryAdapters && Object.keys(modelArch.accuracyRecoveryAdapters).length > 0;
    if (!hasARA) {
      return quantizationOptions;
    }
    let newQuantizationOptions = [
      {
        label: 'Standard',
        options: [quantizationOptions[0], quantizationOptions[1]],
      },
    ];

    // add ARAs if they exist for the model
    let ARAs: SelectOption[] = [];
    if (modelArch.accuracyRecoveryAdapters) {
      for (const [label, value] of Object.entries(modelArch.accuracyRecoveryAdapters)) {
        ARAs.push({ value, label });
      }
    }
    if (ARAs.length > 0) {
      newQuantizationOptions.push({
        label: 'Accuracy Recovery Adapters',
        options: ARAs,
      });
    }

    let additionalQuantizationOptions: SelectOption[] = [];
    // add the quantization options if they are not already included
    for (let i = 2; i < quantizationOptions.length; i++) {
      const option = quantizationOptions[i];
      additionalQuantizationOptions.push(option);
    }
    if (additionalQuantizationOptions.length > 0) {
      newQuantizationOptions.push({
        label: 'Additional Quantization Options',
        options: additionalQuantizationOptions,
      });
    }
    return newQuantizationOptions;
  }, [modelArch]);

  const showGPUSelect = !isMac();
  const sampleOnlyLockedClass = sampleOnlyMode ? 'opacity-50 pointer-events-none select-none' : '';

  const validationConfig = jobConfig.config.process[0].train.validation_config;

  let numDatasetCols = 4;
  let numSampleTopCols = 4;
  let datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  let sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6';
  if (isVideoModel) {
    numSampleTopCols += 1;
  }
  if (isAudioModel) {
    numDatasetCols -= 1;
    numSampleTopCols -= 1;
  }
  if (numDatasetCols == 3) {
    datasetStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }
  if (numSampleTopCols == 5) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-6';
  }
  if (numSampleTopCols == 3) {
    sampleTopStyleClass = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6';
  }

  const sliderTargets = jobConfig.config.process[0].slider?.targets ?? [];
  const defaultSliderTarget = () =>
    objectCopy(
      defaultSliderConfig.targets?.[0] ?? {
        target_class: '',
        positive: '',
        negative: '',
        weight: 1.0,
        shuffle: false,
      },
    );
  const sliderTargetsEditor = (
    <Card title="Slider Targets">
      <>
        <div className="mb-4 flex items-center justify-between">
          <label className="block text-xs text-gray-300">Prompt Sets ({sliderTargets.length})</label>
        </div>
        <div className="space-y-4">
          {sliderTargets.map((target, i) => (
            <div key={i} className="p-4 rounded-lg bg-gray-800 relative">
              <div className="absolute top-2 right-2">
                <button
                  type="button"
                  onClick={() =>
                    setJobConfig(
                      sliderTargets.filter((_, index) => index !== i),
                      'config.process[0].slider.targets',
                    )
                  }
                  className="bg-red-600 hover:bg-red-700 text-white rounded-full p-2 text-sm transition-colors"
                  title="Remove Prompt Set"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <h2 className="text-lg font-bold mb-4">Prompt Set {i + 1}</h2>
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 pr-10">
                <TextInput
                  label="Target Class"
                  value={target.target_class ?? ''}
                  onChange={value => setJobConfig(value, `config.process[0].slider.targets[${i}].target_class`)}
                  placeholder="eg. person"
                />
                <NumberInput
                  label="Weight"
                  value={target.weight ?? 1.0}
                  onChange={value => setJobConfig(value, `config.process[0].slider.targets[${i}].weight`)}
                  placeholder="eg. 1.0"
                  min={0}
                />
                <FormGroup label="Options" className="lg:col-span-2">
                  <Checkbox
                    label="Shuffle comma-separated phrases"
                    checked={target.shuffle || false}
                    onChange={value => setJobConfig(value, `config.process[0].slider.targets[${i}].shuffle`)}
                  />
                </FormGroup>
              </div>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
                <TextAreaInput
                  label="Positive Prompt"
                  value={target.positive ?? ''}
                  onChange={value => setJobConfig(value, `config.process[0].slider.targets[${i}].positive`)}
                  placeholder="eg. person who is happy"
                  rows={3}
                  required
                />
                <TextAreaInput
                  label="Negative Prompt"
                  value={target.negative ?? ''}
                  onChange={value => setJobConfig(value, `config.process[0].slider.targets[${i}].negative`)}
                  placeholder="eg. person who is sad"
                  rows={3}
                  required
                />
              </div>
            </div>
          ))}
        </div>
        <button
          type="button"
          onClick={() => setJobConfig([...sliderTargets, defaultSliderTarget()], 'config.process[0].slider.targets')}
          className="w-full px-4 py-2 mt-4 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
        >
          Add Prompt Set
        </button>
      </>
    </Card>
  );

  return (
    <>
      <form
        onSubmit={handleSubmit}
        className={`space-y-8 relative ${isLoading ? 'pointer-events-none opacity-50' : ''}`}
      >
        {isLoading && (
          <div className="absolute inset-0 z-50 flex items-center justify-center">
            <div className="flex flex-col items-center gap-3">
              <div className="h-8 w-8 animate-spin rounded-full border-4 border-gray-400 border-t-blue-500" />
              <span className="text-sm text-gray-400">Loading...</span>
            </div>
          </div>
        )}
        <div className={`${topBarClass} ${sampleOnlyLockedClass}`}>
          <Card title="Job">
            <div className="sm:hidden">
              <SelectInput label="Training mode" value={jobConfig.config.process[0].type}
                disabled={sampleOnlyMode}
                options={jobTypeOptions.filter(option => trainingModeVisible(trainingModel.arch, option.value, jobConfig.config.process[0].type))}
                onChange={value => {
                  if (value === jobConfig.config.process[0].type) return;
                  let next = objectCopy(jobConfig);
                  next = jobTypeOptions.find(option => option.value === next.config.process[0].type)?.onDeactivate?.(next) ?? next;
                  next = jobTypeOptions.find(option => option.value === value)?.onActivate?.(next) ?? next;
                  next.config.process[0].type = value;
                  setJobConfig(next);
                }} />
            </div>
            <TextInput
              label="Training Name"
              value={jobConfig.config.name}
              docKey="config.name"
              onChange={value => setJobConfig(value, 'config.name')}
              placeholder="Enter training name"
              disabled={runId !== null}
              required
            />
            {showGPUSelect && (
              <SelectInput
                label="GPU ID"
                value={`${gpuIDs}`}
                docKey="gpuids"
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            )}
            {disableSections.includes('trigger_word') ? null : (
              <TextInput
                label="Trigger Word"
                value={jobConfig.config.process[0].trigger_word || ''}
                docKey="config.process[0].trigger_word"
                onChange={(value: string | null) => {
                  if (value?.trim() === '') {
                    value = null;
                  }
                  setJobConfig(value, 'config.process[0].trigger_word');
                }}
                placeholder=""
                required
              />
            )}
          </Card>

          {/* Model Configuration Section */}
          <Card title="Model">
            <SelectInput
              label="Model Architecture"
              value={jobConfig.config.process[0].model.arch}
              onChange={value => {
                if (value === trainingModel.arch) return;
                if (specializedMode) {
                  const next = switchSpecializedArchitecture(jobConfig, value);
                  setJobConfig(next.config);
                  alert(next.notice);
                  return;
                }
                handleModelArchChange(
                  modelArchs,
                  jobConfig.config.process[0].model.arch,
                  value,
                  jobConfig,
                  setJobConfig,
                );
              }}
              options={specializedMode ? Object.entries(flowTrainingModels).map(([value, model]) => ({ value, label: model.label })) : groupedModelOptions}
            />
            {specializedMode && <p className="my-3 text-xs text-gray-400">
              Krea 2, Anima and Ideogram 4 ports are experimental until actual-checkpoint validation.
              Training uses the model-native flow profile; practice and previews use its native generation schedule.
            </p>}
            {specializedMode && trainingModel.arch === 'ideogram4' && <details className="my-3 rounded border border-gray-700 p-3">
              <summary className="cursor-pointer text-sm">Ideogram guidance</summary>
              <SelectInput label="CFG reference" value={String(trainingModel.model_kwargs?.ideogram_cfg_reference ?? 'image_only')}
                options={[{ value: 'image_only', label: 'Image-only (native default)' }, { value: 'negative_prompt', label: 'Text negative (experimental)' }]}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.ideogram_cfg_reference')} />
              <p className="mt-2 text-xs text-gray-400">Image-only CFG uses zero text tokens, not a blank caption. Negative text fields are inactive in this mode; their drafts are retained.
                Text-negative guidance must be supported by the selected preview renderer.</p>
            </details>}
            <SelectInput
              label="Base Model Source"
              value={modelPathSource}
              onChange={value => setModelPathSource(value as ModelPathSource)}
              options={modelPathSourceOptions}
            />
            <TextInput
              label={comfyCheckpointPathSelected ? 'ComfyUI Checkpoint Path' : 'Name or Path'}
              value={jobConfig.config.process[0].model.name_or_path}
              docKey="config.process[0].model.name_or_path"
              onChange={(value: string | null) => {
                if (value?.trim() === '') {
                  value = null;
                }
                setJobConfig(value, 'config.process[0].model.name_or_path');
              }}
              placeholder={
                comfyCheckpointPathSelected ? 'Paste the full absolute path to a .safetensors checkpoint' : ''
              }
              required
            />
            {(!isSliderSpace || modelArch?.name === 'qwen_image_2') && modelArch?.additionalSections?.includes('model.assistant_lora_path') && (
              <TextInput
                label="Helper LoRA Path"
                value={jobConfig.config.process[0].model.assistant_lora_path ?? ''}
                docKey="config.process[0].model.assistant_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.assistant_lora_path');
                }}
                placeholder="/absolute/path/to/training_adapter.safetensors"
              />
            )}
            {modelArch?.additionalSections?.includes('model.text_encoder_path') && (
              <TextInput
                label="Text Encoder Safetensors Path"
                value={jobConfig.config.process[0].model.text_encoder_path ?? ''}
                docKey="config.process[0].model.text_encoder_path"
                onChange={(value: string | undefined) => {
                  setJobConfig(value?.trim() || undefined, 'config.process[0].model.text_encoder_path');
                }}
                placeholder="/absolute/path/to/qwen3vl_8b.safetensors"
              />
            )}
            {modelArch?.additionalSections?.includes('model.vae_path') && (
              <TextInput
                label="Image VAE Checkpoint"
                value={jobConfig.config.process[0].model.vae_path ?? ''}
                onChange={(value: string | undefined) => setJobConfig(value?.trim() || undefined, 'config.process[0].model.vae_path')}
                placeholder="/absolute/path/to/hunyuan_image_3_vae_fp16.safetensors"
                required
              />
            )}
            {modelArch?.additionalSections?.includes('model.vision_path') && (
              <TextInput
                label="Edit Vision Checkpoint (required for references)"
                value={jobConfig.config.process[0].model.model_kwargs?.vision_path ?? ''}
                onChange={(value: string | undefined) => setJobConfig(value?.trim() || undefined, 'config.process[0].model.model_kwargs.vision_path')}
                placeholder="/absolute/path/to/hunyuan_image_3_instruct_siglip2_so400m_naflex.safetensors"
              />
            )}
            {!isSliderSpace && modelArch?.additionalSections?.includes('model.unconditional_lora_path') && (
              <TextInput
                label="Unconditional Adapter Path"
                value={jobConfig.config.process[0].model.unconditional_lora_path ?? ''}
                docKey="config.process[0].model.unconditional_lora_path"
                onChange={(value: string | undefined) => {
                  if (value?.trim() === '') {
                    value = undefined;
                  }
                  setJobConfig(value, 'config.process[0].model.unconditional_lora_path');
                }}
                placeholder=""
              />
            )}
            {modelArch?.modelNotes && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    const gateUrl = modelArch.gateUrl as string;
                    openDoc({
                      title: `Notes - ${modelArch.label}`,
                      description: <div className="space-y-3">{modelArch.modelNotes}</div>,
                    });
                  }}
                  className="w-full flex items-center gap-2 rounded-md bg-blue-950/60 border border-blue-800 px-3 py-2 text-sm text-blue-200 hover:bg-blue-900/60 text-left"
                >
                  <Info className="w-4 h-4 shrink-0 text-blue-400" />
                  <span>Model notes</span>
                </button>
              </div>
            )}
            {modelArch?.gateUrl && (
              <div className="pt-2">
                <button
                  type="button"
                  onClick={() => {
                    const gateUrl = modelArch.gateUrl as string;
                    openDoc({
                      title: 'Gated Model',
                      description: (
                        <div className="space-y-3">
                          <p>
                            This model is gated on Huggingface. Before you can use it, you will need to accept the model
                            terms on the model page:
                          </p>
                          <p>
                            <a
                              href={gateUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:text-blue-300 underline"
                            >
                              {gateUrl}
                            </a>
                          </p>
                          <p>
                            You will also need to create a Huggingface{' '}
                            <a
                              href="https://huggingface.co/settings/tokens"
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-400 hover:text-blue-300 underline"
                            >
                              read token
                            </a>{' '}
                            and add it on the{' '}
                            <a href="/settings" className="text-blue-400 hover:text-blue-300 underline">
                              settings page
                            </a>
                            .
                          </p>
                        </div>
                      ),
                    });
                  }}
                  className="w-full flex items-center gap-2 rounded-md bg-yellow-950/60 border border-yellow-800 px-3 py-2 text-sm text-yellow-200 hover:bg-yellow-900/60 text-left"
                >
                  <Info className="w-4 h-4 shrink-0 text-yellow-400" />
                  <span>
                    Gated model. <span className="underline">Learn more.</span>
                  </span>
                </button>
              </div>
            )}
            {modelArch?.additionalSections?.includes('model.low_vram') && (
              <FormGroup label="Options">
                <Checkbox
                  label="Low VRAM"
                  checked={jobConfig.config.process[0].model.low_vram}
                  onChange={value => setJobConfig(value, 'config.process[0].model.low_vram')}
                />
                {modelArch?.additionalSections?.includes('model.low_vram_layer_streaming') &&
                  jobConfig.config.process[0].model.low_vram && (
                    <Checkbox
                      label="Stream H3 Layers Through CPU RAM"
                      checked={jobConfig.config.process[0].model.low_vram_layer_streaming ?? true}
                      onChange={value => setJobConfig(value, 'config.process[0].model.low_vram_layer_streaming')}
                      docKey="model.low_vram_layer_streaming"
                    />
                  )}
              </FormGroup>
            )}
            {modelArch?.additionalSections?.includes('model.model_kwargs.kv_cache') && (
              <Checkbox
                label="KV Cache"
                docKey="model.model_kwargs.kv_cache"
                checked={jobConfig.config.process[0].model.model_kwargs.kv_cache || false}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.kv_cache')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.qie.match_target_res') && (
              <Checkbox
                label="Match Target Res"
                docKey="model.qie.match_target_res"
                checked={jobConfig.config.process[0].model.model_kwargs.match_target_res}
                onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.match_target_res')}
              />
            )}
            {modelArch?.additionalSections?.includes('model.layer_offloading') && !isMac() && (
              <>
                <Checkbox
                  label={
                    <>
                      Layer Offloading <IoFlaskSharp className="inline text-yellow-500" name="Experimental" />{' '}
                    </>
                  }
                  checked={jobConfig.config.process[0].model.layer_offloading || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.layer_offloading')}
                  docKey="model.layer_offloading"
                />
                {jobConfig.config.process[0].model.layer_offloading && (
                  <div className="pt-2">
                    <SliderInput
                      label="Transformer Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_transformer_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_transformer_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                    <SliderInput
                      label="Text Encoder Offload %"
                      value={Math.round(
                        (jobConfig.config.process[0].model.layer_offloading_text_encoder_percent ?? 1) * 100,
                      )}
                      onChange={value =>
                        setJobConfig(value * 0.01, 'config.process[0].model.layer_offloading_text_encoder_percent')
                      }
                      min={0}
                      max={100}
                      step={1}
                    />
                  </div>
                )}
              </>
            )}
          </Card>
          {disableSections.includes('model.quantize') ? null : (
            <Card title="Quantize / Compile">
              <SelectInput
                label="Transformer"
                value={jobConfig.config.process[0].model.quantize ? jobConfig.config.process[0].model.qtype : ''}
                onChange={value => {
                  if (value === '') {
                    setJobConfig(false, 'config.process[0].model.quantize');
                    value = defaultQtype;
                  } else {
                    setJobConfig(true, 'config.process[0].model.quantize');
                  }
                  setJobConfig(value, 'config.process[0].model.qtype');
                }}
                options={transformerQuantizationOptions}
              />
              {!disableSections.includes('model.quantize_te') && (
                <SelectInput
                  label="Text Encoder"
                  value={
                    jobConfig.config.process[0].model.quantize_te ? jobConfig.config.process[0].model.qtype_te : ''
                  }
                  onChange={value => {
                    if (value === '') {
                      setJobConfig(false, 'config.process[0].model.quantize_te');
                      value = defaultQtype;
                    } else {
                      setJobConfig(true, 'config.process[0].model.quantize_te');
                    }
                    setJobConfig(value, 'config.process[0].model.qtype_te');
                  }}
                  options={quantizationOptions}
                />
              )}
              <FormGroup label="Compile Options">
                <></>
              </FormGroup>
              <Checkbox
                label="Compile Model"
                checked={jobConfig.config.process[0].model.compile || false}
                onChange={value => {
                  setJobConfig(value, 'config.process[0].model.compile');
                  if (value) {
                    for (const key in defaultCompileOptions) {
                      setJobConfig((defaultCompileOptions as any)[key], `config.process[0].model.${key}`);
                    }
                  } else {
                    for (const key in defaultCompileOptions) {
                      setJobConfig(undefined, `config.process[0].model.${key}`);
                    }
                  }
                }}
              />
            </Card>
          )}
          {modelArch?.additionalSections?.includes('model.multistage') && (
            <Card title="Multistage">
              <FormGroup label="Stages to Train" docKey={'model.multistage'}>
                <Checkbox
                  label="High Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_high_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_high_noise')}
                />
                <Checkbox
                  label="Low Noise"
                  checked={jobConfig.config.process[0].model.model_kwargs?.train_low_noise || false}
                  onChange={value => setJobConfig(value, 'config.process[0].model.model_kwargs.train_low_noise')}
                />
              </FormGroup>
              <NumberInput
                label="Switch Every"
                value={jobConfig.config.process[0].train.switch_boundary_every}
                onChange={value => setJobConfig(value, 'config.process[0].train.switch_boundary_every')}
                placeholder="eg. 1"
                docKey={'train.switch_boundary_every'}
                min={1}
                required
              />
            </Card>
          )}
          <Card title="Target">
            <SelectInput
              label="Target Type"
              value={networkType}
              onChange={value => setJobConfig(value, 'config.process[0].network.type')}
              options={isQwenFlowDPO || isQwenGuidanceDistillation || isSliderSpace || isDiffusionKTO ? [
                { value: 'lora', label: 'LoRA' },
              ] : isFizgigSlider ? [
                { value: 'lora', label: 'LoRA' },
                { value: 'dora', label: 'DoRA (signed slider)' },
              ] : [
                { value: 'lora', label: 'LoRA' },
                { value: 'dora', label: 'DoRA' },
                { value: 'loha', label: 'LoHa (LyCORIS)' },
                { value: 'lokr', label: 'LoKr' },
              ]}
            />
            {networkType == 'lokr' && (
              <SelectInput
                label="LoKr Factor"
                value={`${jobConfig.config.process[0].network?.lokr_factor ?? -1}`}
                onChange={value => setJobConfig(parseInt(value), 'config.process[0].network.lokr_factor')}
                options={[
                  { value: '-1', label: 'Auto' },
                  { value: '4', label: '4' },
                  { value: '8', label: '8' },
                  { value: '16', label: '16' },
                  { value: '32', label: '32' },
                ]}
              />
            )}
            {networkType == 'loha' && (
              <Checkbox
                label="DoRA Weight Decomposition (DoHa)"
                checked={jobConfig.config.process[0].network?.loha_dora ?? false}
                onChange={value => setJobConfig(value, 'config.process[0].network.loha_dora')}
                docKey="config.process[0].network.loha_dora"
              />
            )}
            {(networkType == 'lora' || networkType == 'dora' || networkType == 'loha') && (
              <>
                <TextInput
                  label="Pretrained LoRA Path"
                  value={jobConfig.config.process[0].network?.pretrained_lora_path ?? ''}
                  docKey="config.process[0].network.pretrained_lora_path"
                  onChange={value => {
                    setJobConfig(
                      value.trim() === '' ? undefined : value,
                      'config.process[0].network.pretrained_lora_path',
                    );
                  }}
                  placeholder="/path/to/existing_lora.safetensors"
                />
                <NumberInput
                  label="Linear Rank"
                  value={jobConfig.config.process[0].network?.linear ?? 32}
                  onChange={value => {
                    const currentRank = jobConfig.config.process[0].network?.linear;
                    const currentAlpha = jobConfig.config.process[0].network?.linear_alpha;
                    setJobConfig(value, 'config.process[0].network.linear');
                    if (
                      currentAlpha === undefined ||
                      currentAlpha === null ||
                      currentAlpha === currentRank
                    ) {
                      setJobConfig(value, 'config.process[0].network.linear_alpha');
                    }
                  }}
                  placeholder="eg. 16"
                  min={0}
                  max={1024}
                  required
                />
                <NumberInput
                  label="Linear Alpha"
                  value={
                    jobConfig.config.process[0].network?.linear_alpha ??
                    jobConfig.config.process[0].network?.linear ??
                    32
                  }
                  docKey="config.process[0].network.linear_alpha"
                  onChange={value => setJobConfig(value, 'config.process[0].network.linear_alpha')}
                  placeholder="eg. 16"
                  min={0}
                  max={1024}
                  required
                />
                {networkType == 'dora' && (
                  <Checkbox
                    label="Save magnitude-less LoRAs"
                    checked={jobConfig.config.process[0].network?.save_magnitude_less_lora || false}
                    onChange={value => setJobConfig(value, 'config.process[0].network.save_magnitude_less_lora')}
                  />
                )}
                {showNetworkConv ? (
                  <>
                    <NumberInput
                      label="Conv Rank"
                      value={jobConfig.config.process[0].network?.conv ?? 16}
                      onChange={value => {
                        const currentConv = jobConfig.config.process[0].network?.conv;
                        const currentConvAlpha = jobConfig.config.process[0].network?.conv_alpha;
                        setJobConfig(value, 'config.process[0].network.conv');
                        if (
                          (networkType == 'lora' || networkType == 'loha') &&
                          (currentConvAlpha === undefined ||
                            currentConvAlpha === null ||
                            currentConvAlpha === currentConv)
                        ) {
                          setJobConfig(value, 'config.process[0].network.conv_alpha');
                        }
                      }}
                      placeholder="eg. 16"
                      min={0}
                      max={1024}
                    />
                    {(networkType == 'lora' || networkType == 'dora' || networkType == 'loha') && (
                      <NumberInput
                        label="Conv Alpha"
                        value={
                          jobConfig.config.process[0].network?.conv_alpha ??
                          jobConfig.config.process[0].network?.conv ??
                          16
                        }
                        onChange={value => setJobConfig(value, 'config.process[0].network.conv_alpha')}
                        placeholder="eg. 16"
                        min={0}
                        max={1024}
                      />
                    )}
                  </>
                ) : null}
              </>
            )}
          </Card>
          {!disableSections.includes('slider') && (
            <Card title="Slider">
              {isPromptSlider ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <NumberInput
                      label="Training Width"
                      value={jobConfig.config.process[0].slider?.resolutions?.[0]?.[0] ?? 1024}
                      onChange={value => {
                        const height = jobConfig.config.process[0].slider?.resolutions?.[0]?.[1] ?? 1024;
                        setJobConfig([[value, height]], 'config.process[0].slider.resolutions');
                      }}
                      placeholder="eg. 1024"
                      min={64}
                      required
                    />
                    <NumberInput
                      label="Training Height"
                      value={jobConfig.config.process[0].slider?.resolutions?.[0]?.[1] ?? 1024}
                      onChange={value => {
                        const width = jobConfig.config.process[0].slider?.resolutions?.[0]?.[0] ?? 1024;
                        setJobConfig([[width, value]], 'config.process[0].slider.resolutions');
                      }}
                      placeholder="eg. 1024"
                      min={64}
                      required
                    />
                  </div>
                  <FormGroup label="Batching" className="pt-4">
                    <Checkbox
                      label="Full Slide Batch"
                      checked={jobConfig.config.process[0].slider?.batch_full_slide || false}
                      onChange={value => setJobConfig(value, 'config.process[0].slider.batch_full_slide')}
                    />
                  </FormGroup>
                </>
              ) : (
                <>
                  <TextInput
                    label="Target Class"
                    className=""
                    value={jobConfig.config.process[0].slider?.target_class ?? ''}
                    onChange={value => setJobConfig(value, 'config.process[0].slider.target_class')}
                    placeholder="eg. person"
                  />
                  <TextInput
                    label="Positive Prompt"
                    className=""
                    value={jobConfig.config.process[0].slider?.positive_prompt ?? ''}
                    onChange={value => setJobConfig(value, 'config.process[0].slider.positive_prompt')}
                    placeholder="eg. person who is happy"
                  />
                  <TextInput
                    label="Negative Prompt"
                    disabled={specializedMode && !textNegativesEnabled}
                    className=""
                    value={jobConfig.config.process[0].slider?.negative_prompt ?? ''}
                    onChange={value => setJobConfig(value, 'config.process[0].slider.negative_prompt')}
                    placeholder="eg. person who is sad"
                  />
                  <TextInput
                    label="Anchor Class"
                    className=""
                    value={jobConfig.config.process[0].slider?.anchor_class ?? ''}
                    onChange={value => setJobConfig(value, 'config.process[0].slider.anchor_class')}
                    placeholder=""
                  />
                </>
              )}
            </Card>
          )}
          <Card title="Save">
            <SelectInput
              label="Data Type"
              value={jobConfig.config.process[0].save.dtype}
              onChange={value => setJobConfig(value, 'config.process[0].save.dtype')}
              options={[
                { value: 'bf16', label: 'BF16' },
                { value: 'fp16', label: 'FP16' },
                { value: 'fp32', label: 'FP32' },
              ]}
            />
            <NumberInput
              label="Save Every"
              value={jobConfig.config.process[0].save.save_every}
              onChange={value => setJobConfig(value, 'config.process[0].save.save_every')}
              placeholder="eg. 250"
              min={1}
              required
            />
            <NumberInput
              label="Max Step Saves to Keep"
              value={jobConfig.config.process[0].save.max_step_saves_to_keep}
              onChange={value => setJobConfig(value, 'config.process[0].save.max_step_saves_to_keep')}
              placeholder="eg. 4"
              min={1}
              required
            />
            {!isSliderSpace && <><NumberInput
              label="Record Low Window (Steps)"
              value={jobConfig.config.process[0].save.record_low_window_size ?? 3000}
              onChange={value => setJobConfig(value, 'config.process[0].save.record_low_window_size')}
              placeholder="eg. 3000"
              min={1}
              required
            />
            <NumberInput
              label="Record Low Start Step"
              value={jobConfig.config.process[0].save.record_low_start_step ?? 50}
              onChange={value => setJobConfig(value, 'config.process[0].save.record_low_start_step')}
              placeholder="eg. 50"
              min={0}
              required
            />
            <NumberInput
              label="Max Record Low Saves to Keep"
              value={jobConfig.config.process[0].save.max_record_low_saves_to_keep ?? 5}
              onChange={value => setJobConfig(value, 'config.process[0].save.max_record_low_saves_to_keep')}
              placeholder="eg. 5"
              min={1}
              required
            />
            </>}
          </Card>
        </div>
        {isSliderSpace && jobConfig.config.process[0].sliderspace && (
          <SliderSpaceEditor jobConfig={jobConfig} setJobConfig={setJobConfig} datasetOptions={datasetOptions} disabled={sampleOnlyMode} />
        )}
        {isDiffusionKTO && <div className={sampleOnlyLockedClass}>
          <DiffusionKTOEditor jobConfig={jobConfig} setJobConfig={setJobConfig} />
        </div>}
        {isQwenGuidanceDistillation && (
          <div className={sampleOnlyLockedClass}>
            <Card title="Guidance Distillation">
              <p className="mb-4 text-sm text-gray-400">
                Train a CFG-1 LoRA to match the frozen base model using the teacher CFG and negative below.
                Dataset images provide latent states; their captions provide the positive prompts. Use varied
                captions covering the subjects and styles you want the LoRA to handle.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <NumberInput
                  label="Teacher CFG"
                  value={jobConfig.config.process[0].guidance_distillation?.teacher_cfg_scale ?? 4}
                  onChange={value => setJobConfig(value, 'config.process[0].guidance_distillation.teacher_cfg_scale')}
                  min={1}
                  docKey="guidance_distillation.teacher_cfg_scale"
                />
                <SelectInput
                  label="Distillation Objective"
                  value={jobConfig.config.process[0].guidance_distillation?.objective ?? 'full_guidance'}
                  onChange={value => setJobConfig(value, 'config.process[0].guidance_distillation.objective')}
                  options={[
                    { value: 'full_guidance', label: 'Full guided teacher result' },
                    { value: 'negative_only', label: 'Negative prompt contribution only' },
                  ]}
                  docKey="guidance_distillation.objective"
                />
              </div>
              <TextAreaInput
                label="Teacher Negative Prompt"
                className="mt-4"
                value={jobConfig.config.process[0].guidance_distillation?.negative_prompt ?? ''}
                disabled={!textNegativesEnabled || (jobConfig.config.process[0].guidance_distillation?.teacher_cfg_scale ?? 4) <= 1}
                onChange={value => setJobConfig(value, 'config.process[0].guidance_distillation.negative_prompt')}
                placeholder="The fixed negative prompt whose effects you want to learn"
                rows={6}
                docKey="guidance_distillation.negative_prompt"
              />
              <p className="mt-3 text-sm text-gray-400">
                Full guidance learns the teacher&apos;s complete CFG result. Negative-only learns the difference
                between your negative and an empty negative at the same CFG; an empty teacher negative makes that
                objective a no-op. Sample and inference CFG should start at 1 with the LoRA at strength 1.
                Higher sample CFG adds guidance on top of the learned behavior. This mode keeps the usual step count.
              </p>
              {trainingModel.arch === 'ideogram4' && <p className="mt-2 text-xs text-gray-400">Native full guidance uses image-only CFG. Negative-only requires text-negative CFG and a nonempty teacher negative; its baseline is the image-only prediction, not an encoded blank caption.</p>}
            </Card>
          </div>
        )}
        {isQwenFlowDPO && (
          <div className={sampleOnlyLockedClass}>
            <Card title="Flow-DPO">
              <p className="text-sm text-gray-400 mb-4">
                Target Dataset contains preferred outputs. Control Dataset 1 contains matching rejected outputs,
                using the same filename stems and image sizes. Rejected images are not model edit references.
                {editSourcesEnabled ? ' For edit training, put source images in Control Datasets 2 and 3.' : ' This model/configuration has no edit-source conditioning.'}
                {' '}Both outputs share the target caption and any edit inputs. Beta scales a mean latent-element error difference.
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <NumberInput
                  label="DPO Beta"
                  value={jobConfig.config.process[0].flow_dpo?.beta ?? 1}
                  onChange={value => setJobConfig(value, 'config.process[0].flow_dpo.beta')}
                  min={0.0001}
                  placeholder="eg. 1.0"
                />
                <NumberInput
                  label="Preferred Image Loss Weight"
                  value={jobConfig.config.process[0].flow_dpo?.sft_weight ?? 0}
                  onChange={value => setJobConfig(value, 'config.process[0].flow_dpo.sft_weight')}
                  min={0}
                  placeholder="0 = DPO only"
                />
              </div>
            </Card>
          </div>
        )}
        {isFizgigSlider && (
          <div className={sampleOnlyLockedClass}>
            <Card title={isFizgigImageSlider ? 'Fizgig Image Slider' : 'Fizgig Prompt Slider'}>
              <Checkbox label="Multi-point" checked={isMultipoint} docKey="fizgig_slider.multipoint"
                onChange={value => setJobConfig(toggleMultipoint(jobConfig, value))} />
              {isMultipoint && multipointConfig && <MultipointStrengths points={multipointConfig.points}
                onChange={points => setJobConfig(updateMultipointPoints(jobConfig, points))} />}
              <p className="text-sm text-gray-400 mb-4">
                {isMultipoint ? 'Each nonzero strength learns its own target using one LoRA or DoRA. The base model defines neutral.'
                  : '+1 and −1 are trained as opposite strengths of one LoRA or DoRA. The three sample prompts use the same seed at strengths −1, 0 and +1.'}
              </p>
              {trainingModel.arch === 'ideogram4' && <p className="mb-3 text-xs text-amber-400">For structured JSON captions, use complete specific prompts or multipoint targets. Prepending a simple text prefix makes JSON no longer a standalone structured caption; strings are not silently rewritten.</p>}
              {isFizgigImageSlider ? (
                <>
                  <p className="text-sm text-gray-400 mb-4">
                    {isMultipoint ? 'Select matched folders for every strength. Captions describe the neutral input and come from zero if present, otherwise +1, otherwise the lowest numbered point. Zero images are references, not preservation anchors.'
                      : 'Select positive images under Target Dataset and their matched negative images under Control Dataset 1.'}
                    {' '}Give each pair the same filename stem, framing and size. Target images are not Qwen edit references.
                  </p>
                  <NumberInput
                    label="Pair Difference Weight"
                    value={jobConfig.config.process[0].fizgig_slider?.diff_weight ?? 1}
                    onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.diff_weight')}
                    min={0}
                    max={1}
                    placeholder="0 = uniform loss; 1 = focus on changed areas"
                  />
                  {anchorPrompts.length > 0 && (
                    <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                      <NumberInput label="Anchor CFG (Practice + Training)" value={fizgigSlider?.cfg_scale ?? 1}
                        onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.cfg_scale')} min={0} />
                      <NumberInput label="Anchor Practice Image Resolution" value={fizgigSlider?.bank_resolution ?? 768}
                        onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.bank_resolution')} min={64} />
                      <NumberInput label="Anchor Practice Render Steps" value={fizgigSlider?.bank_steps ?? 25}
                        onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.bank_steps')} min={1} />
                    </div>
                  )}
                </>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                  {!isMultipoint && <NumberInput
                    label="Guidance / Push Strength"
                    value={jobConfig.config.process[0].fizgig_slider?.guidance ?? 3}
                    onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.guidance')}
                    min={0.01}
                    placeholder="eg. 3"
                  />}
                  <NumberInput
                    label="CFG (Practice + Training)"
                    value={fizgigSlider?.cfg_scale ?? 1}
                    onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.cfg_scale')}
                    min={0}
                    placeholder="1 = off"
                  />
                  <NumberInput
                    label="Total Practice Images"
                    value={jobConfig.config.process[0].fizgig_slider?.bank_size ?? 16}
                    onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.bank_size')}
                    min={isMultipoint ? multipointConfig?.prompt_entries.length ?? 1 : promptEntries.length}
                    placeholder="Distributed across prompt entries"
                  />
                  <NumberInput
                    label="Practice Image Resolution"
                    value={jobConfig.config.process[0].fizgig_slider?.bank_resolution ?? 768}
                    onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.bank_resolution')}
                    min={64}
                    placeholder="Multiple of 32"
                  />
                  <NumberInput
                    label="Practice Render Steps"
                    value={jobConfig.config.process[0].fizgig_slider?.bank_steps ?? 25}
                    onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.bank_steps')}
                    min={1}
                    placeholder="eg. 25"
                  />
                  <p className="text-xs text-gray-400 md:col-span-2 lg:col-span-5">
                    This CFG controls the practice bank and teacher/student training predictions. Sample-image CFG
                    is a separate setting in the Samples section.
                  </p>
                </div>
              )}
            </Card>
          </div>
        )}
        {isFizgigPromptSlider && (
          <div className={`${sampleOnlyLockedClass} space-y-4`}>
            <Card title="Prompt Sets">
              <p className="mb-3 text-sm text-gray-400">
                Save or load simplified/specific prompts, anchor prompts, shared prefixes and multi-point strengths.
                Training weights, image folders and practice-image settings stay with this job.
              </p>
              <div className="flex flex-wrap items-end gap-3">
                <SelectInput
                  label="Saved Prompt Set"
                  className="min-w-64 flex-1"
                  value={selectedPromptSet}
                  onChange={setSelectedPromptSet}
                  options={promptSetNames.map(name => ({ value: name, label: name }))}
                  disabled={promptSetBusy}
                />
                <button
                  type="button"
                  onClick={loadPromptSet}
                  disabled={!selectedPromptSet || promptSetBusy}
                  className="rounded bg-gray-700 px-4 py-2 text-white hover:bg-gray-600 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Load
                </button>
                <button
                  type="button"
                  onClick={() => { void deletePromptSet(); }}
                  disabled={!selectedPromptSet || promptSetBusy}
                  className="rounded bg-red-800 px-4 py-2 text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Delete
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (activePromptSetName) {
                      void savePromptSet(activePromptSetName, false);
                    } else {
                      setPromptSetError('');
                      setNewPromptSetName('');
                      setSavePromptSetOpen(true);
                    }
                  }}
                  disabled={promptSetBusy}
                  className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => { setPromptSetError(''); setNewPromptSetName(''); setSavePromptSetOpen(true); }}
                  disabled={promptSetBusy}
                  className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Save As
                </button>
              </div>
              {activePromptSetName && <p className="mt-2 text-xs text-gray-400">Editing “{activePromptSetName}”.</p>}
              {promptSetError && <p className="mt-2 text-sm text-red-400" role="alert">{promptSetError}</p>}
              {promptSetNotice && <p className="mt-2 text-sm text-green-400" role="status">{promptSetNotice}</p>}
            </Card>
            {isMultipoint && multipointConfig ? <FizgigMultipointEditor config={multipointConfig} cfgEnabled={sliderCfgEnabled}
              onAddPreservationPrompt={addPreservationPrompt}
              onChange={config => {
                const updated = objectCopy(jobConfig);
                const oldNeutral = multipointNeutralPrompt(multipointConfig);
                const newNeutral = multipointNeutralPrompt(config);
                if (newNeutral !== undefined && oldNeutral !== newNeutral) {
                  updated.config.process[0].sample.samples = updated.config.process[0].sample.samples.map(sample =>
                    sample.prompt === oldNeutral ? { ...sample, prompt: newNeutral } : sample);
                }
                updated.config.process[0].fizgig_slider!.multipoint_config = config;
                updated.config.process[0].fizgig_slider!.bank_size = Math.max(fizgigSlider?.bank_size ?? 16, config.prompt_entries.length);
                setJobConfig(updated);
              }} /> : <>
            <Card title="Shared Prefixes for Simplified Prompts">
              <p className="mb-4 text-sm text-gray-400">
                Simplified entries use their base prompt for neutral (0). For each pole, its prefix is placed
                before the base prompt with one blank line between them. CFG negatives are assembled the same
                way when CFG is above 1: neutral (0) conditions practice images and the student, while the
                +1 and −1 negatives separately condition their teacher predictions. These are separate from
                the −1 slider-direction positive prompt. Specific triplets ignore these prefixes.
              </p>
              <div className="space-y-4">
                <TextAreaInput
                  label="Negative Prefix (−1)"
                  value={fizgigSlider?.negative_prefix ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.negative_prefix')}
                  placeholder="Instructions for the −1 direction"
                  rows={3}
                  required={hasSimplifiedPrompts}
                />
                <TextAreaInput
                  label="Positive Prefix (+1)"
                  value={fizgigSlider?.positive_prefix ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.positive_prefix')}
                  placeholder="Instructions for the +1 direction"
                  rows={3}
                  required={hasSimplifiedPrompts}
                />
                <TextAreaInput
                  label="CFG Negative Prefix for Neutral (0)"
                  value={fizgigSlider?.cfg_negative_prefix ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.cfg_negative_prefix')}
                  placeholder="Optional neutral/practice-image negative prefix"
                  rows={3}
                  disabled={!sliderCfgEnabled || !hasSimplifiedPrompts}
                />
                <TextAreaInput
                  label="CFG Negative Prefix for Positive (+1)"
                  value={fizgigSlider?.cfg_negative_prefix_positive ?? fizgigSlider?.cfg_negative_prefix ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.cfg_negative_prefix_positive')}
                  placeholder="What the +1 teacher should avoid, e.g. close-up portrait"
                  rows={3}
                  disabled={!sliderCfgEnabled || !hasSimplifiedPrompts}
                />
                <TextAreaInput
                  label="CFG Negative Prefix for Negative (−1)"
                  value={fizgigSlider?.cfg_negative_prefix_negative ?? fizgigSlider?.cfg_negative_prefix ?? ''}
                  onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.cfg_negative_prefix_negative')}
                  placeholder="What the −1 teacher should avoid, e.g. full-body shot"
                  rows={3}
                  disabled={!sliderCfgEnabled || !hasSimplifiedPrompts}
                />
              </div>
            </Card>
            <p className="text-sm text-gray-400">
              Practice images render only the neutral (0) prompt and are distributed across entries in order.
              The +1 and −1 prompts drive teacher predictions during training, not separate practice renders.
              Keep the subject matched between the two directions of each entry.
            </p>
            {promptEntries.map((entry, index) => (
              <div key={index} className="space-y-4 rounded-lg border border-gray-700 p-4">
                <div className="flex items-center justify-between gap-4">
                  <h3 className="text-lg font-semibold">
                    Prompt {index + 1} — {entry.kind === 'simple' ? 'Simplified' : 'Specific Triplet'}
                  </h3>
                  <button
                    type="button"
                    onClick={() => updatePromptEntries(promptEntries.filter((_, otherIndex) => otherIndex !== index))}
                    disabled={promptEntries.length === 1}
                    className="flex items-center gap-1 rounded bg-red-700 px-3 py-1.5 text-sm text-white disabled:cursor-not-allowed disabled:opacity-40"
                    aria-label={`Remove prompt ${index + 1}`}
                  >
                    <X className="h-4 w-4" /> Remove
                  </button>
                </div>
                {entry.kind === 'simple' ? (
                  <Card title="Base Prompt (0)">
                    <TextAreaInput
                      label="The complete neutral prompt; shared by both prefixed directions"
                      value={entry.prompt}
                      onChange={value => updatePromptEntries(promptEntries.map((item, itemIndex) =>
                        itemIndex === index && item.kind === 'simple' ? { ...item, prompt: value } : item,
                      ))}
                      rows={5}
                      required
                    />
                  </Card>
                ) : (
                  <>
                    <Card title="Neutral Prompt (0)">
                      <TextAreaInput
                        label="What this triplet's practice images depict without either slider direction"
                        value={entry.neutral_prompt}
                        onChange={value => updatePromptEntries(promptEntries.map((item, itemIndex) =>
                          itemIndex === index && item.kind === 'specific' ? { ...item, neutral_prompt: value } : item,
                        ))}
                        rows={5}
                        required
                      />
                    </Card>
                    <Card title="Positive Prompt (+1)">
                      <TextAreaInput
                        label="Full prompt describing the +1 end"
                        value={entry.positive_prompt}
                        onChange={value => updatePromptEntries(promptEntries.map((item, itemIndex) =>
                          itemIndex === index && item.kind === 'specific' ? { ...item, positive_prompt: value } : item,
                        ))}
                        rows={5}
                        required
                      />
                    </Card>
                    <Card title="Negative Prompt (−1)">
                      <TextAreaInput
                        label="Full prompt describing the −1 end"
                        value={entry.negative_prompt}
                        onChange={value => updatePromptEntries(promptEntries.map((item, itemIndex) =>
                          itemIndex === index && item.kind === 'specific' ? { ...item, negative_prompt: value } : item,
                        ))}
                        rows={5}
                        required
                      />
                    </Card>
                    <Card title="CFG Negative Prompt for Neutral (0)">
                      <TextAreaInput
                        label="Optional full CFG negative prompt for practice images and the student"
                        value={entry.cfg_negative_prompt ?? ''}
                        onChange={value => updatePromptEntries(promptEntries.map((item, itemIndex) =>
                          itemIndex === index && item.kind === 'specific'
                            ? { ...item, cfg_negative_prompt: value } : item,
                        ))}
                        rows={5}
                        disabled={!sliderCfgEnabled}
                      />
                    </Card>
                    <Card title="CFG Negative Prompt for Positive (+1)">
                      <TextAreaInput
                        label="Optional full CFG negative prompt for the +1 teacher prediction"
                        value={entry.cfg_negative_prompt_positive ?? entry.cfg_negative_prompt ?? ''}
                        onChange={value => updatePromptEntries(promptEntries.map((item, itemIndex) =>
                          itemIndex === index && item.kind === 'specific'
                            ? { ...item, cfg_negative_prompt_positive: value } : item,
                        ))}
                        rows={5}
                        disabled={!sliderCfgEnabled}
                      />
                    </Card>
                    <Card title="CFG Negative Prompt for Negative (−1)">
                      <TextAreaInput
                        label="Optional full CFG negative prompt for the −1 teacher prediction"
                        value={entry.cfg_negative_prompt_negative ?? entry.cfg_negative_prompt ?? ''}
                        onChange={value => updatePromptEntries(promptEntries.map((item, itemIndex) =>
                          itemIndex === index && item.kind === 'specific'
                            ? { ...item, cfg_negative_prompt_negative: value } : item,
                        ))}
                        rows={5}
                        disabled={!sliderCfgEnabled}
                      />
                    </Card>
                  </>
                )}
              </div>
            ))}
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => updatePromptEntries([...promptEntries, { kind: 'simple', prompt: '' }])}
                className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500"
              >
                Add Simplified Prompt
              </button>
              <button
                type="button"
                onClick={() => updatePromptEntries([...promptEntries, {
                  kind: 'specific', neutral_prompt: '', positive_prompt: '', negative_prompt: '',
                  cfg_negative_prompt: '',
                  cfg_negative_prompt_positive: '', cfg_negative_prompt_negative: '',
                }])}
                className="rounded bg-gray-700 px-4 py-2 text-white hover:bg-gray-600"
              >
                Add Specific Triplet
              </button>
              <button
                type="button"
                onClick={addPreservationPrompt}
                className="rounded bg-gray-700 px-4 py-2 text-white hover:bg-gray-600"
              >
                Add Preservation Prompt
              </button>
            </div>
            </>}
          </div>
        )}
        {isFizgigSlider && (
          <div className={`${sampleOnlyLockedClass} space-y-4`}>
            <Card title="Preservation Anchors">
              <p className="mb-4 text-sm text-gray-400">
                Preservation prompts (anchors) should stay unchanged at every slider strength. Their predictions are
                matched to the model with this slider disabled, not merely to each other. This preserves
                baseline behavior; it does not teach a new dog size. Anchor prompts ignore slider prefixes.
                One extra practice image is generated for each anchor prompt. Anchor images use their own
                captions and are selected under each image dataset.
              </p>
              <NumberInput label="Preservation Weight" value={fizgigSlider?.preservation_weight ?? 1}
                onChange={value => setJobConfig(value, 'config.process[0].fizgig_slider.preservation_weight')}
                min={0} docKey="fizgig_slider.preservation_weight" placeholder="1 = equal loss weight; 0 = disabled" />
              {isFizgigImageSlider && <button type="button"
                onClick={addPreservationPrompt}
                className="mt-4 rounded bg-gray-700 px-4 py-2 text-white hover:bg-gray-600">
                Add Anchor Prompt
              </button>}
            </Card>
            {anchorPrompts.map((anchor, index) => (
              <Card key={index} title={`Preservation Prompt ${index + 1} (Anchor)`}>
                <div className="mb-3 flex justify-end">
                  <button type="button"
                    onClick={() => setJobConfig(anchorPrompts.filter((_, i) => i !== index), 'config.process[0].fizgig_slider.anchor_prompts')}
                    className="rounded bg-red-700 px-3 py-1.5 text-sm text-white" aria-label={`Remove anchor prompt ${index + 1}`}>
                    Remove Anchor
                  </button>
                </div>
                <TextAreaInput label="Anchor Positive Prompt" value={anchor.prompt} rows={5} required
                  placeholder="A medium-sized dog sitting beside a chair"
                  onChange={value => setJobConfig(anchorPrompts.map((item, i) => i === index ? { ...item, prompt: value } : item), 'config.process[0].fizgig_slider.anchor_prompts')} />
                <TextAreaInput label="Anchor Negative Prompt (optional)" value={anchor.negative_prompt ?? ''}
                  rows={4} className="mt-4" disabled={!sliderCfgEnabled}
                  onChange={value => setJobConfig(anchorPrompts.map((item, i) => i === index ? { ...item, negative_prompt: value } : item), 'config.process[0].fizgig_slider.anchor_prompts')} />
              </Card>
            ))}
          </div>
        )}
        <div className={sampleOnlyLockedClass}>
          <Card title="Training">
            <div className="mb-4 space-y-3 text-sm">
              <Checkbox label="Loss reporting" docKey="logging.record_training_examples"
                checked={jobConfig.config.process[0].logging.record_training_examples ?? true}
                onChange={checked => setJobConfig({ ...jobConfig.config.process[0].logging,
                  record_training_examples: checked,
                  record_training_rng: checked && (jobConfig.config.process[0].logging.record_training_rng ?? false),
                }, 'config.process[0].logging')} />
              <p className="text-gray-400">Records training inputs for the Loss Report in all training modes. Shared objectives are labeled; individual losses are recorded when available. Turn off to disable recording.</p>
              <Checkbox label="Record noise RNG state (uses more storage)" docKey="logging.record_training_rng"
                checked={jobConfig.config.process[0].logging.record_training_rng ?? false}
                disabled={jobConfig.config.process[0].logging.record_training_examples === false}
                onChange={checked => setJobConfig(checked, 'config.process[0].logging.record_training_rng')} />
            </div>
            {isSliderSpace && <p className="text-sm text-gray-400">
              {jobConfig.config.process[0].train.steps.toLocaleString()} total training steps · approximately{' '}
              {Math.floor(jobConfig.config.process[0].train.steps / (jobConfig.config.process[0].sliderspace?.num_directions || 1)).toLocaleString()} per direction, after discovery.
              {' '}Batch size and gradient accumulation are fixed at 1. Text embeddings are cached automatically.
            </p>}
            <div className={trainingBarClass}>
              <div>
                {!isSliderSpace && <><NumberInput
                  label="Batch Size"
                  value={jobConfig.config.process[0].train.batch_size}
                  onChange={value => setJobConfig(value, 'config.process[0].train.batch_size')}
                  placeholder="eg. 4"
                  min={1}
                  required
                />
                <NumberInput
                  label="Gradient Accumulation"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.gradient_accumulation}
                  onChange={value => setJobConfig(value, 'config.process[0].train.gradient_accumulation')}
                  placeholder="eg. 1"
                  min={1}
                  required
                />
                </>}
                <NumberInput
                  label={isSliderSpace ? 'Total training steps' : 'Steps'}
                  className="pt-2"
                  value={jobConfig.config.process[0].train.steps}
                  onChange={value => setJobConfig(value, 'config.process[0].train.steps')}
                  placeholder="eg. 2000"
                  min={1}
                  required
                />
              </div>
              <div>
                <SelectInput
                  label="Optimizer"
                  value={jobConfig.config.process[0].train.optimizer}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer')}
                  options={[
                    { value: 'adafactor', label: 'Adafactor' },
                    { value: 'adam', label: 'Adam' },
                    { value: 'adamw', label: 'AdamW' },
                    { value: 'adamw8bit', label: 'AdamW8Bit' },
                    { value: 'automagic', label: 'Automagic' },
                    { value: 'automagic2', label: 'Automagic v2' },
                    { value: 'automagic3', label: 'Automagic v3' },
                    { value: 'prodigy', label: 'Prodigy' },
                    { value: 'prodigy8bit', label: 'Prodigy8Bit' },
                    { value: 'rose', label: 'Rose' },
                  ]}
                />
                <NumberInput
                  label="Learning Rate"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.lr}
                  onChange={value => setJobConfig(value, 'config.process[0].train.lr')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
                <NumberInput
                  label="Weight Decay"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.optimizer_params.weight_decay}
                  onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.weight_decay')}
                  placeholder="eg. 0.0001"
                  min={0}
                  required
                />
                {showProdigyOptimizerParams && (
                  <>
                    <NumberInput
                      label="Initial D Estimate"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.optimizer_params.d0 ?? 0.000001}
                      onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.d0')}
                      placeholder="eg. 0.0001"
                      min={1e-12}
                    />
                    <NumberInput
                      label="D Coefficient"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.optimizer_params.d_coef ?? 1.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.optimizer_params.d_coef')}
                      placeholder="eg. 2.0"
                      min={0}
                    />
                  </>
                )}
              </div>
              {!isSliderSpace && <div>
                {disableSections.includes('train.timestep_type') || isFizgigSlider ? null : (
                  <SelectInput
                    label="Timestep Type"
                    value={jobConfig.config.process[0].train.timestep_type}
                    disabled={disableSections.includes('train.timestep_type') || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.timestep_type')}
                    options={[
                      { value: 'sigmoid', label: 'Sigmoid' },
                      { value: 'linear', label: 'Linear' },
                      { value: 'shift', label: 'Shift' },
                      { value: 'weighted', label: 'Weighted' },
                    ]}
                  />
                )}
                {!isFizgigSlider && !isQwenGuidanceDistillation && !isSliderSpace && !isDiffusionKTO && <>
                <SelectInput
                  label="Content Or Style"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.content_or_style}
                  onChange={value => setJobConfig(value, 'config.process[0].train.content_or_style')}
                  docKey="train.content_or_style"
                  options={[
                    { value: 'balanced', label: 'Balanced' },
                    { value: 'content', label: 'Content' },
                    { value: 'style', label: 'Style' },
                  ]}
                />
                <SelectInput
                  label="Loss Type"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.loss_type}
                  onChange={value => setJobConfig(value, 'config.process[0].train.loss_type')}
                  options={[
                    { value: 'mse', label: 'Mean Squared Error' },
                    { value: 'mae', label: 'Mean Absolute Error' },
                    { value: 'wavelet', label: 'Wavelet' },
                    { value: 'stepped', label: 'Stepped Recovery' },
                  ]}
                />
                <SelectInput
                  label="Pixel Frequency Loss"
                  className="pt-2"
                  value={frequencyLossType}
                  onChange={value => setJobConfig(value, 'config.process[0].train.frequency_loss_type')}
                  docKey="train.frequency_loss_type"
                  options={[
                    { value: 'none', label: 'Disabled' },
                    { value: 'low_pass', label: 'Low Pass (long periods)' },
                    { value: 'high_pass', label: 'High Pass (short periods)' },
                    { value: 'band_pass', label: 'Band Pass' },
                    { value: 'notch', label: 'Notch (exclude band)' },
                  ]}
                />
                {frequencyLossType !== 'none' && (
                  <>
                    <NumberInput
                      label="Frequency Loss Weight"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.frequency_loss_weight ?? 0.1}
                      onChange={value => setJobConfig(value, 'config.process[0].train.frequency_loss_weight')}
                      docKey="train.frequency_loss_weight"
                      placeholder="eg. 0.1"
                      min={0}
                    />
                    {(frequencyLossType === 'low_pass' || frequencyLossType === 'high_pass') && (
                      <NumberInput
                        label="Cutoff Period (pixels)"
                        className="pt-2"
                        value={jobConfig.config.process[0].train.frequency_loss_cutoff ?? 18}
                        onChange={value => setJobConfig(value, 'config.process[0].train.frequency_loss_cutoff')}
                        docKey="train.frequency_loss_cutoff"
                        placeholder="eg. 18"
                        min={0.01}
                      />
                    )}
                    {(frequencyLossType === 'band_pass' || frequencyLossType === 'notch') && (
                      <>
                        <NumberInput
                          label="Shortest Period (pixels)"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.frequency_loss_min_period ?? 14}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.frequency_loss_min_period')
                          }
                          docKey="train.frequency_loss_min_period"
                          placeholder="eg. 14"
                          min={0.01}
                        />
                        <NumberInput
                          label="Longest Period (pixels)"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.frequency_loss_max_period ?? 28}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.frequency_loss_max_period')
                          }
                          docKey="train.frequency_loss_max_period"
                          placeholder="eg. 28"
                          min={0.01}
                        />
                      </>
                    )}
                    <NumberInput
                      label="Transition Width (pixels)"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.frequency_loss_transition ?? 4}
                      onChange={value => setJobConfig(value, 'config.process[0].train.frequency_loss_transition')}
                      docKey="train.frequency_loss_transition"
                      placeholder="eg. 4"
                      min={0}
                    />
                    <NumberInput
                      label="Frequency Patch Size (pixels)"
                      className="pt-2"
                      value={jobConfig.config.process[0].train.frequency_loss_patch_size ?? 384}
                      onChange={value => setJobConfig(value, 'config.process[0].train.frequency_loss_patch_size')}
                      docKey="train.frequency_loss_patch_size"
                      placeholder="eg. 384; 0 uses full image"
                      min={0}
                    />
                    <Checkbox
                      label="Offload Frequency Activations"
                      className="pt-2"
                      checked={jobConfig.config.process[0].train.frequency_loss_activation_offload ?? true}
                      onChange={value =>
                        setJobConfig(value, 'config.process[0].train.frequency_loss_activation_offload')
                      }
                      docKey="train.frequency_loss_activation_offload"
                    />
                  </>
                )}
                <NumberInput
                  label="Min SNR Gamma"
                  className="pt-2"
                  value={jobConfig.config.process[0].train.min_snr_gamma ?? 5.0}
                  onChange={value => setJobConfig(value, 'config.process[0].train.min_snr_gamma')}
                  placeholder="eg. 5.0"
                  docKey={'train.min_snr_gamma'}
                  min={0}
                />
                </>}
                {modelArch?.additionalSections?.includes('train.audio_loss_multiplier') && (
                  <NumberInput
                    label="Audio Loss Multiplier"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.audio_loss_multiplier ?? 1.0}
                    onChange={value => setJobConfig(value, 'config.process[0].train.audio_loss_multiplier')}
                    placeholder="eg. 1.0"
                    docKey={'train.audio_loss_multiplier'}
                    min={0}
                  />
                )}
              </div>}
              {!isSliderSpace && <div>
                <FormGroup label="EMA (Exponential Moving Average)">
                  <Checkbox
                    label="Use EMA"
                    className="pt-1"
                    checked={jobConfig.config.process[0].train.ema_config?.use_ema || false}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.use_ema')}
                    disabled={isFizgigSlider || isQwenFlowDPO || isQwenGuidanceDistillation || isDiffusionKTO}
                  />
                </FormGroup>
                {jobConfig.config.process[0].train.ema_config?.use_ema && (
                  <NumberInput
                    label="EMA Decay"
                    className="pt-2"
                    value={jobConfig.config.process[0].train.ema_config?.ema_decay as number}
                    onChange={value => setJobConfig(value, 'config.process[0].train.ema_config.ema_decay')}
                    placeholder="eg. 0.99"
                    min={0}
                  />
                )}

                <FormGroup label="Text Encoder Optimizations" className="pt-2">
                  {!disableSections.includes('train.unload_text_encoder') && (
                    <Checkbox
                      label="Unload TE"
                      checked={jobConfig.config.process[0].train.unload_text_encoder || false}
                      docKey={'train.unload_text_encoder'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.unload_text_encoder');
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.cache_text_embeddings');
                        }
                      }}
                    />
                  )}
                  <Checkbox
                    label="Cache Text Embeddings"
                    checked={jobConfig.config.process[0].train.cache_text_embeddings || false}
                    docKey={'train.cache_text_embeddings'}
                    disabled={isFizgigPromptSlider || isQwenFlowDPO || isQwenGuidanceDistillation || isDiffusionKTO}
                    onChange={value => {
                      setJobConfig(value, 'config.process[0].train.cache_text_embeddings');
                      if (value) {
                        setJobConfig(false, 'config.process[0].train.unload_text_encoder');
                      }
                    }}
                  />
                </FormGroup>
              </div>}
              {!isSliderSpace && <div>
                {disableSections.includes('train.diff_output_preservation') ||
                disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <FormGroup label="Regularization">
                    <></>
                  </FormGroup>
                )}
                {disableSections.includes('train.diff_output_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Differential Output Preservation"
                      docKey={'train.diff_output_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.diff_output_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.diff_output_preservation');
                        if (value && jobConfig.config.process[0].train.blank_prompt_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.blank_prompt_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.diff_output_preservation && (
                      <>
                        <NumberInput
                          label="DOP Loss Multiplier"
                          className="pt-2"
                          value={jobConfig.config.process[0].train.diff_output_preservation_multiplier as number}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                        <TextInput
                          label="DOP Preservation Class"
                          className="pt-2 pb-4"
                          value={jobConfig.config.process[0].train.diff_output_preservation_class as string}
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.diff_output_preservation_class')
                          }
                          placeholder="eg. woman"
                        />
                      </>
                    )}
                  </>
                )}
                {disableSections.includes('train.blank_prompt_preservation') ? null : (
                  <>
                    <Checkbox
                      label="Blank Prompt Preservation"
                      docKey={'train.blank_prompt_preservation'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.blank_prompt_preservation || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.blank_prompt_preservation');
                        if (value && jobConfig.config.process[0].train.diff_output_preservation) {
                          // only one can be enabled at a time
                          setJobConfig(false, 'config.process[0].train.diff_output_preservation');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.blank_prompt_preservation && (
                      <>
                        <NumberInput
                          label="BPP Loss Multiplier"
                          className="pt-2"
                          value={
                            (jobConfig.config.process[0].train.blank_prompt_preservation_multiplier as number) || 1.0
                          }
                          onChange={value =>
                            setJobConfig(value, 'config.process[0].train.blank_prompt_preservation_multiplier')
                          }
                          placeholder="eg. 1.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                )}
                {!isQwenGuidanceDistillation && !isSliderSpace && !isDiffusionKTO && <FormGroup label="Other" className="pt-2">
                  <>
                    <Checkbox
                      label="Contrastive Guidance Loss"
                      docKey={'train.do_guidance_loss'}
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.do_guidance_loss || false}
                      onChange={value => {
                        if (value) {
                          setJobConfig(true, 'config.process[0].train.do_guidance_loss');
                          if (!jobConfig.config.process[0].train.guidance_loss_target) {
                            setJobConfig(4.0, 'config.process[0].train.guidance_loss_target');
                          }
                        } else {
                          setJobConfig(undefined, 'config.process[0].train.do_guidance_loss');
                          setJobConfig(undefined, 'config.process[0].train.guidance_loss_target');
                        }
                      }}
                    />
                    {jobConfig.config.process[0].train.do_guidance_loss && (
                      <>
                        <NumberInput
                          label="Guidance Loss Target"
                          docKey={'train.guidance_loss_target'}
                          value={(jobConfig.config.process[0].train.guidance_loss_target as number) || 4.0}
                          onChange={value => setJobConfig(value, 'config.process[0].train.guidance_loss_target')}
                          placeholder="eg. 3.0"
                          min={0}
                        />
                      </>
                    )}
                  </>
                </FormGroup>}
              </div>}
            </div>
          </Card>
        </div>
        {!isSliderSpace && <div className={sampleOnlyLockedClass}>
          <Card
            title="Validation"
            toggled={!!validationConfig}
            onToggle={value => {
              if (value) {
                setJobConfig(
                  {
                    validation_items: [{ image_path: '', prompt: '' }],
                    resolution: 1024,
                    validate_every_n_steps: 1,
                    validation_sigmas: [0.5],
                  },
                  'config.process[0].train.validation_config',
                );
              } else {
                setJobConfig(undefined, 'config.process[0].train.validation_config');
              }
            }}
          >
            {validationConfig && (
              <>
                <p className="text-sm text-gray-400 mb-4">
                  Validation runs a stable loss check on a fixed set of images. Each image is encoded once at startup
                  and predicted at the selected sigmas with fixed seeds, so the result is always deterministic and
                  comparable across the run. The average loss is logged as val/loss every time validation runs. The
                  images need to match the concept of your dataset, but{' '}
                  <span className="font-bold text-gray-300">do not include the validation images in the dataset</span>.
                  They must be images containing the concept you want to train, but not an image trained on.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <NumberInput
                    label="Validate Every"
                    value={validationConfig.validate_every_n_steps}
                    onChange={value =>
                      setJobConfig(value, 'config.process[0].train.validation_config.validate_every_n_steps')
                    }
                    placeholder="eg. 10"
                    min={1}
                    required
                  />
                  <NumberInput
                    label="Validation Resolution"
                    value={validationConfig.resolution}
                    onChange={value => setJobConfig(value, 'config.process[0].train.validation_config.resolution')}
                    placeholder="eg. 512"
                    min={64}
                    required
                  />
                  <SelectInput
                    label="Validation Sigmas"
                    value={(validationConfig.validation_sigmas ?? [1.0, 0.75, 0.5, 0.25]).join(', ')}
                    onChange={value =>
                      setJobConfig(
                        value.split(',').map((v: string) => parseFloat(v)),
                        'config.process[0].train.validation_config.validation_sigmas',
                      )
                    }
                    options={[
                      { value: '0.5', label: '0.5' },
                      { value: '1, 0.5', label: '1.0, 0.5' },
                      { value: '1, 0.66, 0.33', label: '1.0, 0.66, 0.33' },
                      { value: '1, 0.75, 0.5, 0.25', label: '1.0, 0.75, 0.5, 0.25' },
                    ]}
                  />
                </div>
                <div className="mt-4">
                  <label className="block text-xs text-gray-300 mb-2">
                    Validation Images ({validationConfig.validation_items.length})
                  </label>
                  {validationConfig.validation_items.map((item, i) => (
                    <div key={i} className="rounded-lg pl-4 pr-1 py-3 mb-4 bg-gray-950">
                      <div className="flex items-center space-x-4">
                        <SampleControlImage
                          instruction="Add Image"
                          src={item.image_path === '' ? null : item.image_path}
                          onNewImageSelected={imagePath => {
                            setJobConfig(
                              imagePath ?? '',
                              `config.process[0].train.validation_config.validation_items[${i}].image_path`,
                            );
                          }}
                        />
                        <div className="flex-1">
                          <TextInput
                            label="Prompt"
                            value={item.prompt}
                            onChange={value =>
                              setJobConfig(
                                value,
                                `config.process[0].train.validation_config.validation_items[${i}].prompt`,
                              )
                            }
                            placeholder="Enter prompt"
                          />
                        </div>
                        <div>
                          <button
                            type="button"
                            onClick={() =>
                              setJobConfig(
                                validationConfig.validation_items.filter((_, index) => index !== i),
                                'config.process[0].train.validation_config.validation_items',
                              )
                            }
                            className="rounded-full p-1 text-sm"
                          >
                            <X />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                  <button
                    type="button"
                    onClick={() =>
                      setJobConfig(
                        [...validationConfig.validation_items, { image_path: '', prompt: '' }],
                        'config.process[0].train.validation_config.validation_items',
                      )
                    }
                    className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                  >
                    Add Validation Image
                  </button>
                </div>
              </>
            )}
          </Card>
        </div>}
        {!isSliderSpace && <div className={sampleOnlyLockedClass}>
          <Card title="Advanced" collapsible>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div>
                <Checkbox
                  label="Do Differential Guidance"
                  disabled={isQwenGuidanceDistillation || isDiffusionKTO}
                  docKey={'train.do_differential_guidance'}
                  className="pt-1"
                  checked={jobConfig.config.process[0].train.do_differential_guidance || false}
                  onChange={value => {
                    let newValue = value == false ? undefined : value;
                    setJobConfig(newValue, 'config.process[0].train.do_differential_guidance');
                    if (!newValue) {
                      setJobConfig(undefined, 'config.process[0].train.differential_guidance_scale');
                    } else if (
                      jobConfig.config.process[0].train.differential_guidance_scale === undefined ||
                      jobConfig.config.process[0].train.differential_guidance_scale === null
                    ) {
                      // set default differential guidance scale to 3.0
                      setJobConfig(3.0, 'config.process[0].train.differential_guidance_scale');
                    }
                  }}
                />
                {jobConfig.config.process[0].train.differential_guidance_scale && (
                  <>
                    <NumberInput
                      label="Differential Guidance Scale"
                      className="pt-2"
                      value={(jobConfig.config.process[0].train.differential_guidance_scale as number) || 3.0}
                      onChange={value => setJobConfig(value, 'config.process[0].train.differential_guidance_scale')}
                      placeholder="eg. 3.0"
                      min={0}
                    />
                  </>
                )}
              </div>
            </div>
          </Card>
        </div>}
        {!disableSections.includes('datasets') && (
          <div className={sampleOnlyLockedClass}>
            <Card title="Datasets">
              <>
                {jobConfig.config.process[0].datasets.map((dataset, i) => (
                  <div key={i} className="p-4 rounded-lg bg-gray-800 relative">
                    <div className="absolute top-2 right-2 flex gap-1">
                      <button
                        type="button"
                        onClick={() => {
                          const duplicated = objectCopy(dataset);
                          const datasets = [...jobConfig.config.process[0].datasets];
                          datasets.splice(i + 1, 0, duplicated);
                          setJobConfig(datasets, 'config.process[0].datasets');
                        }}
                        className="bg-gray-700 hover:bg-gray-600 rounded-full p-2 text-sm transition-colors"
                        title="Duplicate Dataset"
                      >
                        <Copy className="w-4 h-4" />
                      </button>
                      <button
                        type="button"
                        onClick={() =>
                          setJobConfig(
                            jobConfig.config.process[0].datasets.filter((_, index) => index !== i),
                            'config.process[0].datasets',
                          )
                        }
                        className="bg-red-600 hover:bg-red-700 text-white rounded-full p-2 text-sm transition-colors"
                        title="Remove Dataset"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                    <h2 className="text-lg font-bold mb-4">Dataset {i + 1}</h2>
                    <div className={datasetStyleClass}>
                      <div>
                        {isDiffusionKTO && <SelectInput label="Feedback Label" value={dataset.kto_label ?? ''}
                          options={[{ value: '', label: 'Select feedback…' }, { value: 'liked', label: 'Liked' },
                            { value: 'disliked', label: 'Disliked' }]}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].kto_label`)} />}
                        {isDiffusionKTO && <NumberInput label="Dataset Loss Weight" className="pt-2"
                          value={dataset.loss_multiplier ?? 1} min={0} required
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].loss_multiplier`)} />}
                        {isFizgigImageSlider && isMultipoint && multipointConfig ? <>
                          {sortedPoints(multipointConfig.points).map(point => <div key={point.id} className="mb-3">
                            <NumberInput label={`Strength ${strengthLabel(point.strength)}`} value={point.strength}
                              onChange={value => { if (value !== null) setJobConfig(updateMultipointPoints(jobConfig,
                                multipointConfig.points.map(p => p.id === point.id ? { ...p, strength: value } : p))); }} />
                            <SelectInput label="Image Folder" value={dataset.multipoint_images?.find(mapping => mapping.point_id === point.id)?.folder_path ?? ''}
                              options={[{ value: '', label: 'Select dataset…' }, ...datasetOptions]}
                              onChange={value => setJobConfig(multipointConfig.points.map(p => ({ point_id: p.id,
                                folder_path: p.id === point.id ? value : dataset.multipoint_images?.find(mapping => mapping.point_id === p.id)?.folder_path ?? '' })),
                                `config.process[0].datasets[${i}].multipoint_images`)} />
                          </div>)}
                          <SelectInput label="Anchor Images (optional)" docKey="datasets.anchor_path" value={dataset.anchor_path ?? ''}
                            options={[{ value: '', label: 'None' }, ...datasetOptions]}
                            onChange={value => setJobConfig(value === '' ? null : value, `config.process[0].datasets[${i}].anchor_path`)} />
                        </> : <SelectInput
                          label={isDiffusionKTO ? 'Image Folder' : 'Target Dataset'}
                          value={dataset.folder_path}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].folder_path`)}
                          options={datasetOptions}
                        />}
                        {!isDiffusionKTO && !isPairedImageTraining && (!isQwenGuidanceDistillation || editSourcesEnabled) && modelArch?.additionalSections?.includes('datasets.control_path') && (
                          <SelectInput
                            label="Control Dataset"
                            docKey="datasets.control_path"
                            value={dataset.control_path ?? ''}
                            className="pt-2"
                            onChange={value =>
                              setJobConfig(value == '' ? null : value, `config.process[0].datasets[${i}].control_path`)
                            }
                            options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                          />
                        )}
                        {!isDiffusionKTO && (isPairedImageTraining || (modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (!isQwenGuidanceDistillation || editSourcesEnabled))) && !(isFizgigImageSlider && isMultipoint) && (
                          <>
                            <SelectInput
                              label={isFizgigImageSlider ? 'Control Dataset 1 (−1 Images)' : isQwenFlowDPO ? 'Control Dataset 1 (Rejected Images)' : 'Control Dataset 1'}
                              docKey="datasets.multi_control_paths"
                              value={dataset.control_path_1 ?? ''}
                              className="pt-2"
                              onChange={value =>
                                setJobConfig(
                                  value == '' ? null : value,
                                  `config.process[0].datasets[${i}].control_path_1`,
                                )
                              }
                              options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                            />
                            {isFizgigImageSlider && <SelectInput
                              label="Anchor Images (optional)"
                              docKey="datasets.anchor_path"
                              value={dataset.anchor_path ?? ''}
                              className="pt-2"
                              onChange={value => setJobConfig(value === '' ? null : value, `config.process[0].datasets[${i}].anchor_path`)}
                              options={[{ value: '', label: 'None' }, ...datasetOptions]}
                            />}
                            {!isFizgigImageSlider && (!isQwenFlowDPO || editSourcesEnabled) && <SelectInput
                              label={isQwenFlowDPO ? 'Control Dataset 2 (Edit Source 1)' : 'Control Dataset 2'}
                              docKey="datasets.multi_control_paths"
                              value={dataset.control_path_2 ?? ''}
                              className="pt-2"
                              onChange={value =>
                                setJobConfig(
                                  value == '' ? null : value,
                                  `config.process[0].datasets[${i}].control_path_2`,
                                )
                              }
                              options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                            />}
                            {!isFizgigImageSlider && (!isQwenFlowDPO || editSourcesEnabled) && <SelectInput
                              label={isQwenFlowDPO ? 'Control Dataset 3 (Edit Source 2)' : 'Control Dataset 3'}
                              docKey="datasets.multi_control_paths"
                              value={dataset.control_path_3 ?? ''}
                              className="pt-2"
                              onChange={value =>
                                setJobConfig(
                                  value == '' ? null : value,
                                  `config.process[0].datasets[${i}].control_path_3`,
                                )
                              }
                              options={[{ value: '', label: <>&nbsp;</> }, ...datasetOptions]}
                            />}
                          </>
                        )}
                        {!isDiffusionKTO && !isPairedImageTraining && !isQwenGuidanceDistillation && <NumberInput
                          label="LoRA Weight"
                          value={dataset.network_weight}
                          className="pt-2"
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].network_weight`)}
                          placeholder="eg. 1.0"
                        />}
                        <NumberInput
                          label="Num Repeats"
                          value={dataset.num_repeats || 1}
                          className="pt-2"
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_repeats`)}
                          placeholder="eg. 1"
                          min={1}
                          docKey={'dataset.num_repeats'}
                        />
                      </div>
                      <div>
                        <TextInput
                          label="Default Caption"
                          value={dataset.default_caption}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].default_caption`)}
                          placeholder="eg. A photo of a cat"
                        />
                        {!isDiffusionKTO && !isPairedImageTraining && !isQwenGuidanceDistillation && <NumberInput
                          label="Caption Dropout Rate"
                          className="pt-2"
                          value={dataset.caption_dropout_rate}
                          onChange={value =>
                            setJobConfig(value, `config.process[0].datasets[${i}].caption_dropout_rate`)
                          }
                          placeholder="eg. 0.05"
                          min={0}
                          required
                        />}
                        <CreatableSelectInput
                          label="Caption Extension"
                          className="pt-2"
                          value={dataset.caption_ext || 'txt'}
                          onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].caption_ext`)}
                          options={[
                            { value: 'txt', label: 'txt' },
                            { value: 'json', label: 'json' },
                            { value: 'caption', label: 'caption' },
                          ]}
                        />

                        {modelArch?.additionalSections?.includes('datasets.num_frames') &&
                          !dataset.auto_frame_count && (
                            <NumberInput
                              label="Num Frames"
                              className="pt-2"
                              docKey="datasets.num_frames"
                              value={dataset.num_frames}
                              onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].num_frames`)}
                              placeholder="eg. 41"
                              min={1}
                              required
                            />
                          )}
                      </div>
                      <div>
                        <FormGroup label="Settings" className="">
                          <Checkbox
                            label={isFizgigImageSlider ? (isMultipoint ? 'Cache All Point Latents (required)' : 'Cache +1 Latents (required)') : isQwenFlowDPO ? 'Cache Preferred Latents (required)' : isQwenGuidanceDistillation || isDiffusionKTO ? 'Cache Latents (required)' : 'Cache Latents'}
                            checked={dataset.cache_latents_to_disk || false}
                            onChange={value =>
                              setJobConfig(value, `config.process[0].datasets[${i}].cache_latents_to_disk`)
                            }
                            disabled={isPairedImageTraining || isQwenGuidanceDistillation || isDiffusionKTO}
                          />
                          {!isDiffusionKTO && !isPairedImageTraining && !isQwenGuidanceDistillation && <Checkbox
                            label="Is Regularization"
                            checked={dataset.is_reg || false}
                            onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].is_reg`)}
                          />}
                          {modelArch?.additionalSections?.includes('datasets.auto_frame_count') && (
                            <Checkbox
                              label="Auto Frame Count"
                              checked={dataset.auto_frame_count || false}
                              onChange={value =>
                                setJobConfig(value, `config.process[0].datasets[${i}].auto_frame_count`)
                              }
                              docKey="datasets.auto_frame_count"
                            />
                          )}
                          {modelArch?.additionalSections?.includes('datasets.do_i2v') && (
                            <Checkbox
                              label="Do I2V"
                              checked={dataset.do_i2v || false}
                              onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].do_i2v`)}
                              docKey="datasets.do_i2v"
                            />
                          )}
                          {modelArch?.additionalSections?.includes('datasets.do_audio') && (
                            <Checkbox
                              label="Do Audio"
                              checked={dataset.do_audio || false}
                              onChange={value => {
                                if (!value) {
                                  setJobConfig(undefined, `config.process[0].datasets[${i}].do_audio`);
                                } else {
                                  setJobConfig(value, `config.process[0].datasets[${i}].do_audio`);
                                }
                              }}
                              docKey="datasets.do_audio"
                            />
                          )}
                          {modelArch?.additionalSections?.includes('datasets.audio_normalize') && (
                            <Checkbox
                              label="Audio Normalize"
                              checked={dataset.audio_normalize || false}
                              onChange={value => {
                                if (!value) {
                                  setJobConfig(undefined, `config.process[0].datasets[${i}].audio_normalize`);
                                } else {
                                  setJobConfig(value, `config.process[0].datasets[${i}].audio_normalize`);
                                }
                              }}
                              docKey="datasets.audio_normalize"
                            />
                          )}
                          {modelArch?.additionalSections?.includes('datasets.audio_preserve_pitch') && (
                            <Checkbox
                              label="Audio Preserve Pitch"
                              checked={dataset.audio_preserve_pitch || false}
                              onChange={value => {
                                if (!value) {
                                  setJobConfig(undefined, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                                } else {
                                  setJobConfig(value, `config.process[0].datasets[${i}].audio_preserve_pitch`);
                                }
                              }}
                              docKey="datasets.audio_preserve_pitch"
                            />
                          )}
                        </FormGroup>
                        {!isAudioModel && (
                          <FormGroup label="Flipping" docKey={'datasets.flip'} className="mt-2">
                            <Checkbox
                              label={
                                <>
                                  Flip X <FlipHorizontal2 className="inline-block w-4 h-4 ml-1" />
                                </>
                              }
                              checked={dataset.flip_x || false}
                              onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_x`)}
                            />
                            <Checkbox
                              label={
                                <>
                                  Flip Y <FlipVertical2 className="inline-block w-4 h-4 ml-1" />
                                </>
                              }
                              checked={dataset.flip_y || false}
                              onChange={value => setJobConfig(value, `config.process[0].datasets[${i}].flip_y`)}
                            />
                          </FormGroup>
                        )}
                      </div>
                      {!isAudioModel && (
                        <div>
                          <FormGroup label="Resolutions" className="pt-2">
                            <div className="grid grid-cols-2 gap-2">
                              {[
                                [256, 512, 768, 1024],
                                [1280, 1328, 1536, 2048],
                              ].map(resGroup => (
                                <div key={resGroup[0]} className="space-y-2">
                                  {resGroup.map(res => (
                                    <Checkbox
                                      key={res}
                                      label={res.toString()}
                                      checked={dataset.resolution.includes(res)}
                                      onChange={value => {
                                        const resolutions = dataset.resolution.includes(res)
                                          ? dataset.resolution.filter(r => r !== res)
                                          : [...dataset.resolution, res];
                                        setJobConfig(resolutions, `config.process[0].datasets[${i}].resolution`);
                                      }}
                                    />
                                  ))}
                                </div>
                              ))}
                            </div>
                          </FormGroup>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
                <button
                  type="button"
                  onClick={() => {
                    const newDataset = objectCopy(defaultDatasetConfig);
                    // automaticallt add the controls for a new dataset
                    const controls = isPairedImageTraining || isDiffusionKTO ? [] : (modelArch?.controls ?? []);
                    newDataset.controls = controls;
                    if (isPairedImageTraining || isQwenGuidanceDistillation || isDiffusionKTO) {
                      newDataset.cache_latents_to_disk = true;
                      newDataset.caption_dropout_rate = 0;
                      newDataset.network_weight = 1;
                    }
                    if (isDiffusionKTO) newDataset.kto_label = 'liked';
                    if (isFizgigImageSlider && isMultipoint && multipointConfig) {
                      newDataset.multipoint_images = multipointConfig.points.map(point => ({ point_id: point.id, folder_path: '' }));
                    }
                    setJobConfig([...jobConfig.config.process[0].datasets, newDataset], 'config.process[0].datasets');
                  }}
                  className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
                >
                  Add Dataset
                </button>
              </>
            </Card>
          </div>
        )}
        {isPromptSlider && <div className={sampleOnlyLockedClass}>{sliderTargetsEditor}</div>}
        <div>
          <Card title="Sample">
            {isSliderSpace && jobConfig.config.process[0].sliderspace && <SliderSpacePreview jobConfig={jobConfig} setJobConfig={setJobConfig} />}
            {!isSliderSpace && <Checkbox
              label="Sample on Record Low"
              checked={jobConfig.config.process[0].save.sample_on_record_low ?? true}
              onChange={value => setJobConfig(value, 'config.process[0].save.sample_on_record_low')}
              docKey="config.process[0].save.sample_on_record_low"
            />}
            <div className={sampleTopStyleClass}>
              <div>
                <NumberInput
                  label="Sample Every"
                  value={jobConfig.config.process[0].sample.sample_every}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_every')}
                  placeholder="eg. 250"
                  min={1}
                  required
                />
                <NumberInput
                  label="Sample Start Step"
                  value={jobConfig.config.process[0].sample.sample_start_step ?? 0}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_start_step')}
                  placeholder="eg. 0"
                  className="pt-2"
                  min={0}
                  required
                />
                <SelectInput
                  label="Sampler"
                  className="pt-2"
                  value={jobConfig.config.process[0].sample.sampler}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sampler')}
                  options={[
                    { value: 'flowmatch', label: 'FlowMatch' },
                    { value: 'ddpm', label: 'DDPM' },
                  ]}
                />
                <NumberInput
                  label="Guidance Scale"
                  value={jobConfig.config.process[0].sample.guidance_scale}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.guidance_scale')}
                  placeholder="eg. 1.0"
                  className="pt-2"
                  min={0}
                  required
                />
                <NumberInput
                  label="Sample Steps"
                  value={jobConfig.config.process[0].sample.sample_steps}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.sample_steps')}
                  placeholder="eg. 1"
                  className="pt-2"
                  min={1}
                  required
                />
              </div>

              {!isAudioModel && (
                <div>
                  <NumberInput
                    label="Width"
                    value={jobConfig.config.process[0].sample.width}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.width')}
                    placeholder="eg. 1024"
                    min={0}
                    required
                  />
                  <NumberInput
                    label="Height"
                    value={jobConfig.config.process[0].sample.height}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.height')}
                    placeholder="eg. 1024"
                    className="pt-2"
                    min={0}
                    required
                  />
                  {isVideoModel && (
                    <div>
                      <NumberInput
                        label="Num Frames"
                        value={jobConfig.config.process[0].sample.num_frames}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.num_frames')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                      <NumberInput
                        label="FPS"
                        value={jobConfig.config.process[0].sample.fps}
                        onChange={value => setJobConfig(value, 'config.process[0].sample.fps')}
                        placeholder="eg. 0"
                        className="pt-2"
                        min={0}
                        required
                      />
                    </div>
                  )}
                </div>
              )}

              <div>
                <NumberInput
                  label="Seed"
                  value={jobConfig.config.process[0].sample.seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.seed')}
                  placeholder="eg. 0"
                  min={0}
                  required
                />
                <Checkbox
                  label="Walk Seed"
                  className="pt-4 pl-2"
                  checked={jobConfig.config.process[0].sample.walk_seed}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.walk_seed')}
                />
              </div>
              <div>
                <FormGroup label="Advanced Sampling" className="pt-2">
                  <div>
                    <Checkbox
                      label="Skip First Sample"
                      className="pt-4"
                      checked={jobConfig.config.process[0].train.skip_first_sample || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.skip_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Force First Sample"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.force_first_sample || false}
                      docKey={'train.force_first_sample'}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.force_first_sample');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.skip_first_sample');
                        }
                      }}
                    />
                  </div>
                  <div>
                    <Checkbox
                      label="Disable Sampling"
                      className="pt-1"
                      checked={jobConfig.config.process[0].train.disable_sampling || false}
                      onChange={value => {
                        setJobConfig(value, 'config.process[0].train.disable_sampling');
                        // cannot do both, so disable the other
                        if (value) {
                          setJobConfig(false, 'config.process[0].train.force_first_sample');
                        }
                      }}
                    />
                  </div>
                </FormGroup>
              </div>
            </div>
            {!isSliderSpace && <TextInput
              label="Inference LoRA Path"
              value={jobConfig.config.process[0].model.inference_lora_path ?? ''}
              docKey="config.process[0].model.inference_lora_path"
              onChange={value => {
                if (value.trim() === '') {
                  setJobConfig(undefined, 'config.process[0].model.inference_lora_path');
                } else {
                  setJobConfig(value, 'config.process[0].model.inference_lora_path');
                }
              }}
              placeholder="output/krea2_raw_to_turbo_r256.safetensors"
              className="pt-2"
            />}
            <div className="pt-4">
              <Checkbox
                label="Use ComfyUI Renderer"
                checked={comfyEnabled}
                onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.enabled')}
              />
            </div>
            {comfyEnabled && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 pt-2">
                <TextInput
                  label="ComfyUI URL"
                  value={comfyConfig?.api_url || 'http://127.0.0.1:8188'}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.api_url')}
                  placeholder="http://127.0.0.1:8188"
                />
                {modelArch?.name !== 'minimax_h3' && (
                  <TextAreaInput
                    label="Negative Prompt"
                    value={comfyConfig?.negative_prompt || ''}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.negative_prompt')}
                    placeholder="Optional negative prompt for ComfyUI samples"
                    rows={2}
                  />
                )}
                <CreatableSelectInput
                  label="Sample Renderer Workflow"
                  value={comfyConfig?.workflow_path || 'config/comfy_templates/krea2_lora_sample.json.njk'}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.workflow_path')}
                  options={comfyWorkflowOptions}
                />
                <Checkbox
                  label="Send Prompts as Batch"
                  className="pt-6"
                  checked={comfyConfig?.send_prompts_as_batch || false}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.send_prompts_as_batch')}
                />
                <Checkbox
                  label="Run ComfyUI in Background"
                  className="pt-6"
                  checked={comfyConfig?.run_in_background || false}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.run_in_background')}
                />
                <TextInput
                  label="LoRA Path Replace From"
                  value={comfyConfig?.training_lora_path_replace_from || ''}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].sample.comfy.training_lora_path_replace_from')
                  }
                  placeholder="/mnt/training"
                />
                <TextInput
                  label="LoRA Path Replace To"
                  value={comfyConfig?.training_lora_path_replace_to || ''}
                  onChange={value =>
                    setJobConfig(value, 'config.process[0].sample.comfy.training_lora_path_replace_to')
                  }
                  placeholder="R:/training"
                />
                <CreatableSelectInput
                  label="Comfy Model"
                  value={comfyConfig?.model || ''}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.model')}
                  options={toOptions(comfyOptions.model)}
                />
                <CreatableSelectInput
                  label="Comfy VAE"
                  value={comfyConfig?.vae || ''}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.vae')}
                  options={toOptions(comfyOptions.vae)}
                />
                {modelArch?.name === 'minimax_h3' && (
                  <CreatableSelectInput
                    label="Comfy Audio VAE"
                    value={comfyConfig?.audio_vae || ''}
                    onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.audio_vae')}
                    options={toOptions(comfyOptions.vae)}
                  />
                )}
                <CreatableSelectInput
                  label="Comfy Text Encoder"
                  value={comfyConfig?.text_encoder || ''}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.text_encoder')}
                  options={toOptions(comfyOptions.text_encoder)}
                />
                <CreatableSelectInput
                  label="Comfy Sampler"
                  value={comfyConfig?.sampler || 'euler'}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.sampler')}
                  options={toOptions(comfyOptions.sampler)}
                />
                <CreatableSelectInput
                  label="Comfy Scheduler"
                  value={comfyConfig?.scheduler || 'simple'}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.scheduler')}
                  options={toOptions(comfyOptions.scheduler)}
                />
                <CreatableSelectInput
                  label="Comfy Inference LoRA"
                  value={comfyConfig?.inference_lora || ''}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.inference_lora')}
                  options={toOptions(comfyOptions.inference_lora)}
                />
                <NumberInput
                  label="Inference LoRA Strength"
                  value={comfyConfig?.inference_lora_strength ?? 1.0}
                  onChange={value =>
                    setJobConfig(value ?? 1.0, 'config.process[0].sample.comfy.inference_lora_strength')
                  }
                  placeholder="eg. 1.0"
                  min={0}
                  required
                />
                <CreatableSelectInput
                  label="Output Format"
                  value={comfyConfig?.output_format || 'webp_with_json'}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.output_format')}
                  options={toOptions(comfyOptions.output_format)}
                />
                <CreatableSelectInput
                  label="Output Quality"
                  value={comfyConfig?.output_quality || 'high'}
                  onChange={value => setJobConfig(value, 'config.process[0].sample.comfy.output_quality')}
                  options={toOptions(comfyOptions.output_quality)}
                />
              </div>
            )}
            {!(isSliderSpace && jobConfig.config.process[0].sliderspace?.preview_auto) && <>
            <div className="pt-2 mb-2 flex items-center justify-between">
              <label className="block text-xs text-gray-300">
                Sample Prompts ({jobConfig.config.process[0].sample.samples.length})
              </label>
              {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                <button
                  type="button"
                  disabled={jobConfig.config.process[0].sample.samples.length === 0}
                  onClick={() => {
                    const sampleCfg = jobConfig.config.process[0].sample;
                    const items = sampleCfg.samples
                      .map((s, i) => ({
                        index: i,
                        prompt: s.prompt || '',
                        aspectRatio: toAspectRatio(s.width || sampleCfg.width, s.height || sampleCfg.height),
                      }))
                      .filter(it => it.prompt.trim() !== '');
                    if (items.length === 0) return;
                    openUpsamplePromptsModal(items, (index, newPrompt) => {
                      setJobConfig(newPrompt, `config.process[0].sample.samples[${index}].prompt`);
                    });
                  }}
                  className="px-3 py-1.5 text-sm bg-purple-600 hover:bg-purple-700 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-md inline-flex items-center gap-2"
                >
                  <Wand2 className="w-4 h-4" />
                  Upsample Prompts
                </button>
              )}
            </div>
            {jobConfig.config.process[0].sample.samples.map((sample, i) => (
              <div key={i} className="rounded-lg pl-4 pr-1 mb-4 bg-gray-950">
                <div className="flex items-center space-x-2">
                  <div className="flex-1">
                    <div className="flex">
                      <div className="flex-1">
                        {modelArch?.sampleTags && taggedSampleArr && modelArchTagSections ? (
                          <>
                            {modelArchTagSections.map((sampleTagSection, sti) => (
                              <div key={sti} className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                                {Object.entries(sampleTagSection).map(([tagKey, tag]) => (
                                  <div key={tagKey} className="mb-2">
                                    {tag.type === 'text' && (
                                      <TextInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'multiline' && (
                                      <TextAreaInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                    {tag.type === 'number' && (
                                      <NumberInput
                                        label={tag.title}
                                        value={taggedSampleArr[i][tagKey] ?? ''}
                                        onChange={value => {
                                          let taggedSample = { ...taggedSampleArr[i] };
                                          taggedSample[tagKey] = value;
                                          setJobConfig(
                                            objToTags(taggedSample),
                                            `config.process[0].sample.samples[${i}].prompt`,
                                          );
                                        }}
                                        placeholder={`Enter ${tag.title.toLowerCase()}`}
                                      />
                                    )}
                                  </div>
                                ))}
                              </div>
                            ))}
                          </>
                        ) : (
                          <>
                            {modelArch?.hasMultiLinePrompts ? (
                              <TextAreaInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder="Enter prompt"
                                required
                              />
                            ) : (
                              <TextInput
                                label={`Prompt`}
                                value={sample.prompt}
                                onChange={value => setJobConfig(value, `config.process[0].sample.samples[${i}].prompt`)}
                                placeholder="Enter prompt"
                                required
                              />
                            )}
                          </>
                        )}

                        {modelArch?.additionalSections?.includes('ideogram_4_prompt') && (
                          <div className="mt-2">
                            <button
                              type="button"
                              onClick={() => {
                                const sampleCfg = jobConfig.config.process[0].sample;
                                openPromptBoxEditor({
                                  prompt: sample.prompt || '',
                                  aspectRatio: toAspectRatio(
                                    sample.width || sampleCfg.width,
                                    sample.height || sampleCfg.height,
                                  ),
                                  title: `Prompt #${i + 1}`,
                                  onApply: newPrompt =>
                                    setJobConfig(newPrompt, `config.process[0].sample.samples[${i}].prompt`),
                                });
                              }}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs rounded-md border border-gray-600 text-gray-300 hover:bg-gray-800 transition-colors"
                            >
                              <SquareDashed className="w-3.5 h-3.5" />
                              Edit caption &amp; boxes
                            </button>
                          </div>
                        )}

                        <div className="grid w-full lg:grid-flow-col lg:auto-cols-fr gap-4 mt-2">
                          {!isAudioModel && (
                            <TextInput
                              label={`Width`}
                              value={sample.width ? `${sample.width}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].width;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].width`);
                                  } else {
                                    console.warn('Invalid width value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.width} (default)`}
                            />
                          )}
                          {!isAudioModel && (
                            <TextInput
                              label={`Height`}
                              value={sample.height ? `${sample.height}` : ''}
                              onChange={value => {
                                // remove any non-numeric characters
                                value = value.replace(/\D/g, '');
                                if (value === '') {
                                  // remove the key from the config if empty
                                  let newConfig = objectCopy(jobConfig);
                                  if (newConfig.config.process[0].sample.samples[i]) {
                                    delete newConfig.config.process[0].sample.samples[i].height;
                                    setJobConfig(
                                      newConfig.config.process[0].sample.samples,
                                      'config.process[0].sample.samples',
                                    );
                                  }
                                } else {
                                  const intValue = parseInt(value);
                                  if (!isNaN(intValue)) {
                                    setJobConfig(intValue, `config.process[0].sample.samples[${i}].height`);
                                  } else {
                                    console.warn('Invalid height value:', value);
                                  }
                                }
                              }}
                              placeholder={`${jobConfig.config.process[0].sample.height} (default)`}
                            />
                          )}
                          <TextInput
                            label={`Seed`}
                            value={sample.seed ? `${sample.seed}` : ''}
                            onChange={value => {
                              // remove any non-numeric characters
                              value = value.replace(/\D/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].seed;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                const intValue = parseInt(value);
                                if (!isNaN(intValue)) {
                                  setJobConfig(intValue, `config.process[0].sample.samples[${i}].seed`);
                                } else {
                                  console.warn('Invalid seed value:', value);
                                }
                              }
                            }}
                            placeholder={`${jobConfig.config.process[0].sample.walk_seed ? jobConfig.config.process[0].sample.seed + i : jobConfig.config.process[0].sample.seed} (default)`}
                          />
                          {!isSliderSpace && <TextInput
                            label={`LoRA Scale`}
                            value={sample.network_multiplier ? `${sample.network_multiplier}` : ''}
                            onChange={value => {
                              // remove any non-numeric, - or . characters
                              value = value.replace(/[^0-9.-]/g, '');
                              if (value === '') {
                                // remove the key from the config if empty
                                let newConfig = objectCopy(jobConfig);
                                if (newConfig.config.process[0].sample.samples[i]) {
                                  delete newConfig.config.process[0].sample.samples[i].network_multiplier;
                                  setJobConfig(
                                    newConfig.config.process[0].sample.samples,
                                    'config.process[0].sample.samples',
                                  );
                                }
                              } else {
                                // set it as a string
                                setJobConfig(value, `config.process[0].sample.samples[${i}].network_multiplier`);
                                return;
                              }
                            }}
                            placeholder={`1.0 (default)`}
                          />}
                        </div>
                      </div>
                      {modelArch?.additionalSections?.includes('datasets.multi_control_paths') && (
                        <FormGroup label="Control Images" className="pt-2 ml-4">
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-2 mt-2">
                            {(isQwenFlowDPO ? ['ctrl_img_1', 'ctrl_img_2'] : ['ctrl_img_1', 'ctrl_img_2', 'ctrl_img_3']).map((ctrlKey, ctrl_idx) => (
                              <SampleControlImage
                                key={ctrlKey}
                                instruction={isQwenFlowDPO ? `Add Edit Source ${ctrl_idx + 1}` : `Add Control Image ${ctrl_idx + 1}`}
                                className=""
                                src={sample[ctrlKey as keyof typeof sample] as string}
                                onNewImageSelected={imagePath => {
                                  if (!imagePath) {
                                    let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                                    delete newSamples[i][ctrlKey as keyof typeof sample];
                                    setJobConfig(newSamples, 'config.process[0].sample.samples');
                                  } else {
                                    setJobConfig(imagePath, `config.process[0].sample.samples[${i}].${ctrlKey}`);
                                  }
                                }}
                              />
                            ))}
                          </div>
                        </FormGroup>
                      )}
                      {modelArch?.additionalSections?.includes('sample.ctrl_img') && (
                        <SampleControlImage
                          className="mt-6 ml-4"
                          src={sample.ctrl_img}
                          onNewImageSelected={imagePath => {
                            if (!imagePath) {
                              let newSamples = objectCopy(jobConfig.config.process[0].sample.samples);
                              delete newSamples[i].ctrl_img;
                              setJobConfig(newSamples, 'config.process[0].sample.samples');
                            } else {
                              setJobConfig(imagePath, `config.process[0].sample.samples[${i}].ctrl_img`);
                            }
                          }}
                        />
                      )}
                    </div>
                    <div className="pb-4"></div>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={() =>
                        setJobConfig(
                          jobConfig.config.process[0].sample.samples.filter((_, index) => index !== i),
                          'config.process[0].sample.samples',
                        )
                      }
                      className="rounded-full p-1 text-sm"
                    >
                      <X />
                    </button>
                  </div>
                </div>
              </div>
            ))}
            <button
              type="button"
              onClick={() =>
                setJobConfig(
                  [...jobConfig.config.process[0].sample.samples, { prompt: '' }],
                  'config.process[0].sample.samples',
                )
              }
              className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors"
            >
              Add Prompt
            </button>
            </>}
          </Card>
        </div>

        {status === 'success' && <p className="text-green-500 text-center">Training saved successfully!</p>}
        {status === 'error' && <p className="text-red-500 text-center">Error saving training. Please try again.</p>}
      </form>
      <Modal isOpen={savePromptSetOpen} onClose={() => setSavePromptSetOpen(false)} title="Save Prompt Set As" size="sm">
        <form onSubmit={event => { event.preventDefault(); void savePromptSet(newPromptSetName, true); }} className="space-y-4 p-4">
          <TextInput
            label="Prompt set name"
            value={newPromptSetName}
            onChange={setNewPromptSetName}
            placeholder="e.g. close-up vs full-body"
            required
          />
          <p className="text-xs text-gray-400">
            Names may contain letters, numbers, spaces, dots, underscores and hyphens. You will be asked to confirm an overwrite.
          </p>
          {promptSetError && <p className="text-sm text-red-400" role="alert">{promptSetError}</p>}
          <div className="flex justify-end gap-2">
            <button type="button" onClick={() => setSavePromptSetOpen(false)}
              className="rounded bg-gray-700 px-4 py-2 text-white hover:bg-gray-600">Cancel</button>
            <button type="submit" disabled={promptSetBusy || !newPromptSetName.trim()}
              className="rounded bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40">
              {promptSetBusy ? 'Saving…' : 'Save As'}
            </button>
          </div>
        </form>
      </Modal>
      <AddSingleImageModal />
    </>
  );
}
