import { NextRequest, NextResponse } from 'next/server';
import { TOOLKIT_ROOT } from '@/paths';
import { createCaptionApiCredential } from '../../../../../cron/captionApiCredentials';
import { captionApiEndpoint } from '../../../../../cron/captionApiConfig';
export const runtime = 'nodejs';
export async function POST(request: NextRequest) {
  let stage: 'parse' | 'validate' | 'persist' = 'parse';
  try {
    const body = await request.json();
    stage = 'validate';
    if (typeof body.apiUrl !== 'string' || typeof body.apiKey !== 'string')
      throw new Error('API URL and key are required.');
    captionApiEndpoint(body.apiUrl);
    if (!body.apiKey.trim() || body.apiKey.length > 8192 || /[\r\n]/.test(body.apiKey))
      throw new Error('Invalid API key.');
    stage = 'persist';
    const id = await createCaptionApiCredential(TOOLKIT_ROOT, body.apiUrl, body.apiKey);
    return NextResponse.json({ id }, { headers: { 'Cache-Control': 'no-store' } });
  } catch (cause: any) {
    // Parse errors and URL errors can contain secret input properties. Keep only
    // safe operation context, an errno and the original stack frames.
    const code = typeof cause?.code === 'string' && /^E[A-Z0-9]+$/.test(cause.code) ? ` (${cause.code})` : '';
    const error = new Error(
      stage === 'parse'
        ? 'Invalid JSON in caption credential setup.'
        : stage === 'validate'
          ? 'Enter a valid HTTP(S) API URL and key; URLs cannot contain credentials, a query, or a fragment.'
          : `Could not persist the private caption key${code}. Check owner-only storage permissions.`,
    );
    if (cause instanceof Error && cause.stack)
      error.stack = error.message + '\n' + cause.stack.split('\n').slice(1).join('\n');
    console.error(`Caption credential setup failed at ${stage}:`, { message: error.message, stack: error.stack });
    return NextResponse.json({ error: error.message, stack: error.stack }, { status: stage === 'persist' ? 500 : 400 });
  }
}
