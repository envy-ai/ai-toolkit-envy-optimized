import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { getAssistantService } from '@/server/assistant/service';
export async function POST(request: NextRequest) {
  try {
    const { runId } = z
      .object({ runId: z.string().uuid() })
      .strict()
      .parse(await request.json());
    await (await getAssistantService()).cancelRun(runId);
    return NextResponse.json({ cancelled: true });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Cancellation failed' },
      { status: 400 },
    );
  }
}
