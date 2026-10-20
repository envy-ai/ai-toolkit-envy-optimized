import { NextRequest, NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { invalidateCache } from '@/server/apiCache';

class QueueReorderError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

export async function PATCH(request: NextRequest, { params }: { params: Promise<{ queueID: string }> }) {
  const { queueID } = await params;

  try {
    const body = await request.json();
    const orderedJobIds = body?.orderedJobIds;

    if (!Array.isArray(orderedJobIds) || !orderedJobIds.every(id => typeof id === 'string' && id.length > 0)) {
      return NextResponse.json({ error: 'orderedJobIds must be an array of job IDs' }, { status: 400 });
    }

    const uniqueJobIds = new Set(orderedJobIds);
    if (uniqueJobIds.size !== orderedJobIds.length) {
      return NextResponse.json({ error: 'orderedJobIds contains duplicate job IDs' }, { status: 400 });
    }

    const positions = await prisma.$transaction(async transaction => {
      // Read IDs/ranks only: the visible page or search may omit queued jobs.
      // Running/stopping jobs never participate in this permutation.
      const queuedJobs = await transaction.job.findMany({
        where: { status: 'queued', gpu_ids: queueID },
        select: { id: true, queue_position: true },
        orderBy: [{ queue_position: 'asc' }, { id: 'asc' }],
      });
      const submittedJobs = queuedJobs.filter(job => uniqueJobIds.has(job.id));
      if (submittedJobs.length !== orderedJobIds.length)
        throw new QueueReorderError('All reordered jobs must be queued jobs on the selected queue', 400);
      if (!orderedJobIds.length) return [];

      const previousPosition = new Map(queuedJobs.map(job => [job.id, job.queue_position]));
      let nextVisible = 0;
      const mergedIds = queuedJobs.map(job => (uniqueJobIds.has(job.id) ? orderedJobIds[nextVisible++] : job.id));
      const updates = mergedIds.map((id, index) => ({ id, queue_position: (index + 1) * 1000 }));
      for (const position of updates) {
        // The worker claims with status:'queued' too. A claim, move to another
        // GPU, or concurrent reorder must not modify a now-running job's rank.
        const changed = await transaction.job.updateMany({
          where: {
            id: position.id,
            status: 'queued',
            gpu_ids: queueID,
            queue_position: previousPosition.get(position.id),
          },
          data: { queue_position: position.queue_position },
        });
        if (changed.count !== 1)
          throw new QueueReorderError('Queue changed while reordering. Refresh and try again.', 409);
      }
      return updates;
    });
    invalidateCache('jobs-active');
    return NextResponse.json({ success: true, positions });
  } catch (error) {
    console.error('Error reordering queue:', error);
    const conflict = error instanceof Error && 'code' in error && error.code === 'P2034';
    const status = error instanceof QueueReorderError ? error.status : conflict ? 409 : 500;
    return NextResponse.json(
      {
        error:
          error instanceof QueueReorderError
            ? error.message
            : conflict
              ? 'Queue changed while reordering. Refresh and try again.'
              : 'Failed to reorder queue',
        stack: error instanceof Error ? error.stack : undefined,
      },
      { status },
    );
  }
}
