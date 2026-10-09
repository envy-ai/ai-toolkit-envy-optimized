import { NextRequest, NextResponse } from 'next/server';
import { getAssistantService } from '@/server/assistant/service';

export const runtime = 'nodejs';
export async function GET() {
  try {
    return NextResponse.json(await (await getAssistantService()).getConfiguration());
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Settings unavailable' },
      { status: 400 },
    );
  }
}
export async function POST(request: NextRequest) {
  try {
    return NextResponse.json(await (await getAssistantService()).setConfiguration(await request.json()));
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Invalid settings' }, { status: 400 });
  }
}
