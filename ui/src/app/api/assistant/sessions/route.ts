import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { sessionContentSchema, sessionIdSchema } from '@/assistant/sessions';
import { getAssistantSessionStore } from '@/server/assistant/sessionService';
import { sessionErrorResponse } from '@/server/assistant/sessionResponse';
export const runtime = 'nodejs';
const headers = { 'Cache-Control': 'no-store' };
export async function GET(request: NextRequest) {
  try {
    return NextResponse.json(
      { sessions: await (await getAssistantSessionStore()).list(request.nextUrl.searchParams.get('search') ?? '') },
      { headers },
    );
  } catch (error) {
    return sessionErrorResponse(error);
  }
}
export async function POST(request: NextRequest) {
  try {
    const body = z
      .object({ id: sessionIdSchema, content: sessionContentSchema })
      .strict()
      .parse(await request.json());
    return NextResponse.json(await (await getAssistantSessionStore()).create(body.id, body.content), { headers });
  } catch (error) {
    return sessionErrorResponse(error);
  }
}
