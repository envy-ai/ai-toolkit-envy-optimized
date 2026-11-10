import type { CustomModelOption } from './options';
import type { JobConfig } from '@/types';
import { Checkbox, NumberInput, SelectInput, TextInput } from '@/components/formInputs';

type Props = {
  options?: CustomModelOption[];
  jobConfig: JobConfig;
  setJobConfig: (value: any, key: string) => void;
  idPrefix: string;
  summary?: string;
};

/** One adapter for extension controls in both ordinary and advanced sections. */
export default function CustomModelOptions({ options, jobConfig, setJobConfig, idPrefix, summary }: Props) {
  if (!options?.length) return null;
  const controls = options.map((option, index) => {
    const id = `${idPrefix}-${index}`;
    const shared = { id, label: option.label, doc: option.doc };
    let control;
    if (option.type === 'checkbox') {
      control = <Checkbox label={option.label} doc={option.doc} checked={option.getValue(jobConfig)}
        onChange={value => option.onChange(value, jobConfig, setJobConfig)} />;
    } else if (option.type === 'number') {
      control = <NumberInput {...shared} value={option.getValue(jobConfig)} min={option.min} step={option.step}
        allowEmpty={option.allowEmpty} placeholder={option.placeholder} clampOnBlur={false}
        onChange={value => option.onChange(value, jobConfig, setJobConfig)} />;
    } else if (option.type === 'text') {
      control = <TextInput {...shared} value={option.getValue(jobConfig) ?? ''} placeholder={option.placeholder}
        onChange={value => option.onChange(value, jobConfig, setJobConfig)} />;
    } else if (option.type === 'multiselect') {
      control = <SelectInput {...shared} multiple value={option.getValue(jobConfig)} options={option.options}
        allowCustomValues={option.allowCustomValues} placeholder={option.placeholder}
        onChange={value => option.onChange(value, jobConfig, setJobConfig)} />;
    } else {
      control = <SelectInput {...shared} value={option.getValue(jobConfig) ?? ''} options={option.options}
        onChange={value => option.onChange(value, jobConfig, setJobConfig)} />;
    }
    return <div key={option.label}>{control}
      {option.help && <div className="mt-1 text-xs text-gray-400">{option.help}</div>}
    </div>;
  });
  if (!summary) return <>{controls}</>;
  return <details className="mt-4 border-t border-gray-700 pt-3">
    <summary className="cursor-pointer text-sm font-medium text-gray-200">{summary}</summary>
    <div className="mt-2 space-y-3">{controls}</div>
  </details>;
}
