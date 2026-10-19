import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { readSampleRender, startSampleRender } from '@/server/jobSamples';

async function handle(params: Promise<{ jobID: string }>, start: boolean) {
  try {
    const { jobID } = await params;
    const job = await prisma.job.findUnique({ where: { id: jobID } });
    if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    return NextResponse.json(start ? await startSampleRender(job) : await readSampleRender(job), {
      status: start ? 202 : 200,
    });
  } catch (error: any) {
    console.error('Saved sample rendering failed:', error);
    return NextResponse.json(
      { status: 'error', message: error.message, error: error.message, stack: error.stack },
      { status: 409 },
    );
  }
}

export async function GET(_request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  return handle(params, false);
}

export async function POST(_request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  return handle(params, true);
}
