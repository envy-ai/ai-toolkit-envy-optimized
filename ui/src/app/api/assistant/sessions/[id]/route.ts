import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { sessionIdSchema, sessionUpdateSchema } from '@/assistant/sessions';
import { getAssistantSessionStore } from '@/server/assistant/sessionService';
import { sessionErrorResponse } from '@/server/assistant/sessionResponse';
export const runtime = 'nodejs';
type Context = { params: Promise<{ id: string }> };
const headers = { 'Cache-Control': 'no-store' };
export async function GET(_request: NextRequest, { params }: Context) {
  try {
    return NextResponse.json(await (await getAssistantSessionStore()).get(sessionIdSchema.parse((await params).id)), {
      headers,
    });
  } catch (error) {
    return sessionErrorResponse(error);
  }
}
export async function PATCH(request: NextRequest, { params }: Context) {
  try {
    return NextResponse.json(
      await (
        await getAssistantSessionStore()
      ).update(sessionIdSchema.parse((await params).id), sessionUpdateSchema.parse(await request.json())),
      { headers },
    );
  } catch (error) {
    return sessionErrorResponse(error);
  }
}
export async function DELETE(request: NextRequest, { params }: Context) {
  try {
    const { revision } = z
      .object({ revision: z.number().int().positive() })
      .strict()
      .parse(await request.json());
    await (await getAssistantSessionStore()).delete(sessionIdSchema.parse((await params).id), revision);
    return NextResponse.json({ success: true }, { headers });
  } catch (error) {
    return sessionErrorResponse(error);
  }
}
