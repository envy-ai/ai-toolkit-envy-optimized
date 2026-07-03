import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export async function PATCH(request: NextRequest, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;

  try {
    const body = await request.json();
    const orderedJobIds = body?.orderedJobIds;

    if (!Array.isArray(orderedJobIds) || !orderedJobIds.every(id => typeof id === 'string')) {
      return NextResponse.json({ error: 'orderedJobIds must be an array of job IDs' }, { status: 400 });
    }

    const uniqueJobIds = new Set(orderedJobIds);
    if (uniqueJobIds.size !== orderedJobIds.length) {
      return NextResponse.json({ error: 'orderedJobIds contains duplicate job IDs' }, { status: 400 });
    }

    const submittedJobs = await prisma.job.findMany({
      where: {
        id: { in: orderedJobIds },
        status: 'queued',
        gpu_ids: queueID,
      },
      select: {
        id: true,
      },
    });

    if (submittedJobs.length !== orderedJobIds.length) {
      return NextResponse.json(
        { error: 'All reordered jobs must be queued jobs on the selected queue' },
        { status: 400 },
      );
    }

    await prisma.$transaction(
      orderedJobIds.map((jobID, index) =>
        prisma.job.update({
          where: { id: jobID },
          data: {
            queue_position: (index + 1) * 1000,
          },
        }),
      ),
    );

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error reordering queue:', error);
    return NextResponse.json({ error: 'Failed to reorder queue' }, { status: 500 });
  }
}
