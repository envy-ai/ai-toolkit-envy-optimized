import { NextRequest, NextResponse } from 'next/server';

type OptionsResponse = {
  model: string[];
  vae: string[];
  text_encoder: string[];
  inference_lora: string[];
  sampler: string[];
  scheduler: string[];
  output_format: string[];
  output_quality: string[];
};

const emptyOptions = (): OptionsResponse => ({
  model: [],
  vae: [],
  text_encoder: [],
  inference_lora: [],
  sampler: [],
  scheduler: [],
  output_format: [],
  output_quality: [],
});

const normalizeBaseUrl = (url: string | null): string => {
  const value = (url || 'http://127.0.0.1:8188').trim() || 'http://127.0.0.1:8188';
  return value.replace(/\/+$/, '');
};

const fetchJson = async (baseUrl: string, path: string): Promise<any> => {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 2500);
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      cache: 'no-store',
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`ComfyUI ${path} returned ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
};

const normalizeStringArray = (value: unknown): string[] => {
  if (!Array.isArray(value)) return [];
  return value.filter(item => typeof item === 'string').map(item => item as string);
};

const extractInputOptions = (objectInfo: any, nodeClass: string, inputName: string): string[] => {
  const nodeInfo = objectInfo?.[nodeClass] ?? (objectInfo && Object.keys(objectInfo).length === 1
    ? objectInfo[Object.keys(objectInfo)[0]]
    : null);
  const input = nodeInfo?.input ?? {};
  for (const sectionName of ['required', 'optional']) {
    const spec = input?.[sectionName]?.[inputName];
    if (!Array.isArray(spec) || spec.length === 0) continue;
    if (Array.isArray(spec[0])) return normalizeStringArray(spec[0]);
    if (spec[0] === 'COMBO' && spec[1]?.options) return normalizeStringArray(spec[1].options);
  }
  return [];
};

export async function GET(request: NextRequest) {
  const baseUrl = normalizeBaseUrl(request.nextUrl.searchParams.get('url'));
  const options = emptyOptions();

  try {
    const [
      diffusionModels,
      vaes,
      textEncoders,
      loras,
      kSamplerInfo,
      saveImageInfo,
    ] = await Promise.all([
      fetchJson(baseUrl, '/api/models/diffusion_models').catch(() => []),
      fetchJson(baseUrl, '/api/models/vae').catch(() => []),
      fetchJson(baseUrl, '/api/models/text_encoders').catch(() => []),
      fetchJson(baseUrl, '/api/models/loras').catch(() => []),
      fetchJson(baseUrl, '/api/object_info/KSampler').catch(() => ({})),
      fetchJson(baseUrl, '/api/object_info/SaveImageWithMetaData').catch(() => ({})),
    ]);

    options.model = normalizeStringArray(diffusionModels);
    options.vae = normalizeStringArray(vaes);
    options.text_encoder = normalizeStringArray(textEncoders);
    options.inference_lora = normalizeStringArray(loras);
    options.sampler = extractInputOptions(kSamplerInfo, 'KSampler', 'sampler_name');
    options.scheduler = extractInputOptions(kSamplerInfo, 'KSampler', 'scheduler');
    options.output_format = extractInputOptions(saveImageInfo, 'SaveImageWithMetaData', 'output_format');
    options.output_quality = extractInputOptions(saveImageInfo, 'SaveImageWithMetaData', 'quality');
  } catch (error) {
    return NextResponse.json(emptyOptions());
  }

  return NextResponse.json(options);
}
