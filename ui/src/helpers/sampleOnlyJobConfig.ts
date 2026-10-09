import { mergeSliderSpacePreview } from '@/app/jobs/new/sliderspace';

const cloneJson = <T>(value: T): T => JSON.parse(JSON.stringify(value));

/** Running jobs may change previews only; always merge into their stored training configuration. */
export function mergeSampleOnlyJobConfig(existingConfig: any, incomingConfig: any) {
  const mergedConfig = cloneJson(existingConfig);
  const existingProcess = mergedConfig.config.process[0];
  const incomingProcess = incomingConfig.config.process[0];
  mergeSliderSpacePreview(mergedConfig, incomingConfig);
  existingProcess.sample = cloneJson(incomingProcess.sample);

  if (existingProcess.type !== 'sliderspace') {
    if (typeof incomingProcess.save?.sample_on_record_low === 'boolean') {
      existingProcess.save = { ...existingProcess.save, sample_on_record_low: incomingProcess.save.sample_on_record_low };
    }
  }
  if (incomingProcess.model && existingProcess.model) {
    if (incomingProcess.model.inference_lora_path === undefined) delete existingProcess.model.inference_lora_path;
    else existingProcess.model.inference_lora_path = incomingProcess.model.inference_lora_path;
  }

  if (incomingProcess.train && existingProcess.train) {
    for (const key of ['skip_first_sample', 'force_first_sample', 'disable_sampling']) {
      if (key in incomingProcess.train) existingProcess.train[key] = incomingProcess.train[key];
    }
  }
  return mergedConfig;
}
