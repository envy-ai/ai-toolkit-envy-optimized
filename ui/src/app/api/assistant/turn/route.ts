import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { getAssistantService } from '@/server/assistant/service';
import { ASSISTANT_SYSTEM_PROMPT, ASSISTANT_TOOLS } from '@/assistant/tools';
import type { AssistantTurnRequest } from '@/assistant/AssistantProvider';
export const runtime = 'nodejs';
const startMessages = z
  .array(z.object({ role: z.enum(['user', 'assistant']), content: z.string().max(64000) }).strict())
  .min(1)
  .max(80);
export async function POST(request: NextRequest) {
  const service = await getAssistantService();
  let runId: string | undefined;
  const cancelled = () => {
    if (runId) void service.cancelRun(runId);
  };
  try {
    const body = await request.json();
    runId = z.string().uuid().parse(body.runId);
    request.signal.addEventListener('abort', cancelled, { once: true });
    request.signal.throwIfAborted();
    const turn: AssistantTurnRequest = {
      runId,
      requestId: body.requestId,
      configurationRevision: body.configurationRevision,
    };
    if (body.messages) {
      const screen = z
        .string()
        .max(2048)
        .parse(body.screen || '/dashboard');
      turn.messages = [
        { role: 'system', content: ASSISTANT_SYSTEM_PROMPT + `\nCurrent screen: ${screen}` },
        ...startMessages.parse(body.messages),
      ];
      turn.tools = ASSISTANT_TOOLS;
    } else {
      turn.toolResults = body.toolResults;
      turn.attachments = body.attachments;
    }
    return NextResponse.json(await service.createTurn(turn));
  } catch (error) {
    if (runId) await service.cancelRun(runId);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Assistant request failed' },
      { status: 400 },
    );
  } finally {
    request.signal.removeEventListener('abort', cancelled);
  }
}
