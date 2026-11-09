import { NextRequest, NextResponse } from 'next/server';
import path from 'path';
import prisma from '@/server/prisma';
import { getTrainingFolder } from '@/server/settings';
import { parseTrainingMapStep, readTrainingMaps } from '@/server/trainingMaps';

export const runtime = 'nodejs';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  let step: number | null;
  try {
    step = parseTrainingMapStep(request.nextUrl.searchParams.get('step'));
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 400 });
  }
  try {
    const root = path.join(await getTrainingFolder(), job.name, 'training_maps');
    const result = await readTrainingMaps(root, step);
    if (result.manifest) result.manifest.probes = result.manifest.probes.map(probe => {
      const image = path.resolve(root, probe.image_path);
      if (!image.startsWith(root + path.sep)) throw new Error('Training map probe image must be inside the map folder.');
      return { ...probe, image_url: `/api/img/${encodeURIComponent(image)}?v=${encodeURIComponent(probe.fingerprint ?? '')}` };
    });
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error reading training maps:', error);
    return NextResponse.json({ error: (error as Error).message, stack: (error as Error).stack }, { status: 500 });
  }
}
