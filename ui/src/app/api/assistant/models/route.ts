import { NextResponse } from 'next/server';
import { getAssistantService } from '@/server/assistant/service';
export const runtime = 'nodejs';
export async function GET() {
  try {
    return NextResponse.json({ models: await (await getAssistantService()).discoverModels() });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Model discovery failed' },
      { status: 400 },
    );
  }
}
