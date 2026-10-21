import { z } from 'zod';
import prisma from '@/server/prisma';
import { TOOLKIT_ROOT } from '@/paths';
import { captionerTypes } from '@/helpers/captionOptions';
import { captionPreferenceFields, type CaptionJobPreferences } from '@/helpers/captionJobConfig';
import { API_CAPTIONER, apiCaptionSettingsSchema, assertNoCaptionCredentials } from '../../cron/captionApiConfig';
import { readCaptionApiCredential } from '../../cron/captionApiCredentials';

export const CAPTION_SETTINGS_KEY = 'CAPTION_DEFAULTS';
export class CaptionSettingsPersistenceError extends Error {
  code?: string;
  constructor(cause: unknown) {
    super('Could not persist the remembered caption setup.');
    const code = (cause as { code?: unknown })?.code;
    if (typeof code === 'string' && /^(?:E[A-Z0-9]+|P\d{4})$/.test(code)) this.code = code;
    if (cause instanceof Error && cause.stack) this.stack = this.message + '\n' + cause.stack.split('\n').slice(1).join('\n');
  }
}
const preferencesSchema = z.object({
  type: z.string().refine(value => captionerTypes.some(option => option.name === value), 'Unknown captioner type.'),
  caption: z.record(z.string(), z.unknown()),
  gpu_ids: z.string().max(1000).nullable(),
}).strict();

export async function validateCaptionJobPreferences(input: unknown, verifyCredential = true): Promise<CaptionJobPreferences> {
  assertNoCaptionCredentials(input);
  const preferences = preferencesSchema.parse(input);
  if (Object.keys(preferences.caption).some(key => !captionPreferenceFields.includes(key as typeof captionPreferenceFields[number]))) {
    throw new Error('Unsupported caption setting in remembered setup.');
  }
  if (preferences.type === API_CAPTIONER) {
    const caption = apiCaptionSettingsSchema.parse({ ...preferences.caption, path_to_caption: 'current-dataset' });
    if (verifyCredential && caption.api_credential_id) await readCaptionApiCredential(TOOLKIT_ROOT, caption.api_credential_id, caption.api_url);
  } else if (preferences.caption.api_credential_id !== undefined) {
    throw new Error('Only API caption setups may reference a private API credential.');
  }
  return preferences as unknown as CaptionJobPreferences;
}

export async function getCaptionJobPreferences(): Promise<CaptionJobPreferences | null> {
  const row = await prisma.settings.findUnique({ where: { key: CAPTION_SETTINGS_KEY } });
  // A missing private file must not prevent reopening to replace/clear its reference.
  return row ? validateCaptionJobPreferences(JSON.parse(row.value), false) : null;
}

export async function saveCaptionJobPreferences(input: unknown): Promise<CaptionJobPreferences> {
  const preferences = await validateCaptionJobPreferences(input);
  try {
    await prisma.settings.upsert({ where: { key: CAPTION_SETTINGS_KEY },
      create: { key: CAPTION_SETTINGS_KEY, value: JSON.stringify(preferences) },
      update: { value: JSON.stringify(preferences) } });
  } catch (cause) {
    throw new CaptionSettingsPersistenceError(cause);
  }
  return preferences;
}
