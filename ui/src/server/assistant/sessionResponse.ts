import { NextResponse } from 'next/server';
import { z } from 'zod';
import { SessionStoreError } from './AssistantSessionStore';
export function sessionErrorResponse(error: unknown) {
  console.error('Assistant session operation failed:', error);
  return NextResponse.json(
    {
      error: error instanceof Error ? error.message : 'Assistant session operation failed.',
      stack: error instanceof Error ? error.stack : undefined,
    },
    {
      status: error instanceof SessionStoreError ? error.status : error instanceof z.ZodError ? 400 : 500,
      headers: { 'Cache-Control': 'no-store' },
    },
  );
}
