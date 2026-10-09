import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { getAssistantService } from '@/server/assistant/service';
import { executeAssistantTool } from '@/server/assistant/toolExecutor';
export const runtime = 'nodejs';
export async function POST(request: NextRequest) {
  try {
    const call = z
      .object({ callId: z.string().min(1).max(200), name: z.string().min(1).max(64), arguments: z.unknown() })
      .strict()
      .parse(await request.json());
    return NextResponse.json(
      await executeAssistantTool(call, {
        origin: request.nextUrl.origin,
        authorization: request.headers.get('Authorization'),
        signal: request.signal,
        provider: await getAssistantService(),
      }),
    );
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : 'Tool failed' }, { status: 400 });
  }
}
