import { z } from 'zod';

export const API_CAPTIONER = 'APICaptioner';
export const API_CAPTION_QUEUE = 'cpu';
export const API_CAPTION_PROMPT =
  'Write a descriptive caption for this image. Describe the main subjects, their appearance and actions, the setting, composition, lighting, colors, and visual style. Describe only what is visible, without speculation. Return only the caption, with no introduction, headings, or quotation marks.';
export const DEFAULT_CAPTION_REGEX_REPLACE = {
  enabled: false, pattern: '', replacement: '', ignore_case: false,
  multiline: false, dotall: false, verbose: false, global: true,
};
const captionRegexReplaceSchema = z.object({
  enabled: z.boolean().default(false),
  pattern: z.string().max(64_000).default(''),
  replacement: z.string().max(64_000).default(''),
  ignore_case: z.boolean().default(false),
  multiline: z.boolean().default(false),
  dotall: z.boolean().default(false),
  verbose: z.boolean().default(false),
  global: z.boolean().default(true),
}).strict().refine(value => !value.enabled || value.pattern.length > 0,
  'Caption cleanup From pattern is required when enabled.');
export function captionApiEndpoint(value: string): string {
  const url = new URL(value);
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password || url.search || url.hash)
    throw new Error('API URL must use HTTP(S) without credentials, query, or fragment.');
  url.pathname = url.pathname.replace(/\/+$/, '');
  if (!url.pathname.endsWith('/chat/completions')) url.pathname += '/chat/completions';
  return url.toString();
}
const apiUrl = z
  .string()
  .min(1)
  .max(8192)
  .refine(value => {
    try {
      captionApiEndpoint(value);
      return true;
    } catch {
      return false;
    }
  }, 'API URL must be an HTTP(S) base URL or chat/completions URL without credentials, query, or fragment.');
export const apiCaptionSettingsSchema = z
  .object({
    api_url: apiUrl,
    model_name_or_path: z.string().trim().min(1).max(500),
    system_prompt: z.string().max(64_000).optional(),
    caption_prompt: z.string().min(1).max(64_000),
    max_new_tokens: z.number().int().positive().safe(),
    api_timeout: z.number().finite().positive().max(3600),
    api_max_tokens_field: z.enum(['max_tokens', 'max_completion_tokens']),
    temperature: z.number().finite().min(0).max(2).optional(),
    top_p: z.number().finite().min(0).max(1).optional(),
    frequency_penalty: z.number().finite().min(-2).max(2).optional(),
    presence_penalty: z.number().finite().min(-2).max(2).optional(),
    seed: z.number().int().safe().optional(),
    stop: z.array(z.string().min(1).max(4096)).max(4).optional(),
    regex_replace: captionRegexReplaceSchema.optional(),
    api_credential_id: z.string().uuid().optional(),
    path_to_caption: z.string().min(1),
    caption_extension: z.string().regex(/^[A-Za-z0-9_-]+$/),
    extensions: z.array(z.enum(['png', 'jpg', 'jpeg', 'webp', 'bmp', 'gif', 'avif'])).min(1),
  })
  .passthrough();
export function assertNoCaptionCredentials(value: unknown): void {
  if (Array.isArray(value)) {
    value.forEach(assertNoCaptionCredentials);
    return;
  }
  if (value && typeof value === 'object')
    for (const [key, item] of Object.entries(value)) {
      if (/^(api_key|apiKey|authorization|key|token|secret)$/i.test(key))
        throw new Error('API credentials must be entered in the private API key field, never in job configuration.');
      assertNoCaptionCredentials(item);
    }
}
export function isApiCaptionJob(job: { job_type: string; gpu_ids: string; job_config: string }): boolean {
  if (job.job_type !== 'caption' || job.gpu_ids !== API_CAPTION_QUEUE) return false;
  try {
    const processes = JSON.parse(job.job_config)?.config?.process;
    if (
      !Array.isArray(processes) ||
      processes.length !== 1 ||
      processes[0].type !== API_CAPTIONER ||
      processes[0].device !== 'cpu'
    )
      return false;
    assertNoCaptionCredentials(processes[0]);
    return apiCaptionSettingsSchema.safeParse(processes[0].caption).success;
  } catch {
    return false;
  }
}
