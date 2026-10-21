import { NextResponse } from 'next/server';
import { CaptionSettingsPersistenceError, getCaptionJobPreferences, saveCaptionJobPreferences } from '@/server/captionSettings';

export const runtime = 'nodejs';
const headers = { 'Cache-Control': 'no-store' };

function settingsError(cause: unknown, operation: 'load' | 'save', stage: string = operation) {
  // Validation/IO errors may include caller-owned content. Keep safe context and stack frames only.
  const causeCode = (cause as { code?: unknown })?.code;
  const code = typeof causeCode === 'string' && /^(?:E[A-Z0-9]+|P\d{4})$/.test(causeCode) ? ` (${causeCode})` : '';
  const error = new Error(`Could not ${operation} the remembered caption setup${code}. Check its settings and private credential reference.`);
  if (cause instanceof Error && cause.stack) error.stack = error.message + '\n' + cause.stack.split('\n').slice(1).join('\n');
  console.error(`Caption settings ${operation} failed at ${stage}:`, { message: error.message, stack: error.stack });
  return NextResponse.json({ error: error.message, stack: error.stack }, { status: operation === 'load' || stage === 'persist' ? 500 : 400, headers });
}

export async function GET() {
  try {
    return NextResponse.json({ preferences: await getCaptionJobPreferences() }, { headers });
  } catch (cause) {
    return settingsError(cause, 'load');
  }
}

export async function POST(request: Request) {
  let stage = 'parse';
  try {
    const input = await request.json();
    stage = 'validate';
    return NextResponse.json({ preferences: await saveCaptionJobPreferences(input) }, { headers });
  } catch (cause) {
    if (cause instanceof CaptionSettingsPersistenceError) stage = 'persist';
    return settingsError(cause, 'save', stage);
  }
}
