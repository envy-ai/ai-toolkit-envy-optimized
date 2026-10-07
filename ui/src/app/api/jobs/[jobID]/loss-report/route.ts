import { NextRequest, NextResponse } from 'next/server';
import sqlite3 from 'sqlite3';
import path from 'path';
import fs from 'fs/promises';
import prisma from '@/server/prisma';
import { getTrainingFolder, getDatasetsRoot, getDataRoot } from '@/server/settings';
import { all, parseLossReportOptions, queryLossReport, queryLossReportDetail } from '@/server/lossReport';

export const runtime = 'nodejs';

export async function GET(request: NextRequest, { params }: { params: Promise<{ jobID: string }> }) {
  const { jobID } = await params;
  const job = await prisma.job.findUnique({ where: { id: jobID } });
  if (!job) return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  const search = request.nextUrl.searchParams;
  let options;
  let detail: number[] | null = null;
  try {
    options = parseLossReportOptions(search);
    if (search.get('detail') === '1')
      detail = ['step', 'microbatch', 'item'].map(key => {
        const raw = search.get(key);
        const value = raw === null || raw.trim() === '' ? NaN : Number(raw);
        if (!Number.isSafeInteger(value) || value < 0) throw new Error(`Invalid ${key}.`);
        return value;
      });
  } catch (error) {
    return NextResponse.json({ error: (error as Error).message }, { status: 400 });
  }
  const trainingRoot = await getTrainingFolder();
  const datasetsRoot = await getDatasetsRoot();
  const roots = [trainingRoot, datasetsRoot, await getDataRoot()];
  const filename = path.join(trainingRoot, job.name, 'loss_log.db');
  if (!(await fs.stat(filename).catch(() => null)))
    return NextResponse.json({ available: false, total: 0, entries: [], recorded: 0 });
  let db: sqlite3.Database | null = null;
  const imageInfo = (imagePath: string, datasetPath: string) => {
    if (imagePath.startsWith('prompt://')) return { relative_path: 'Prompt training input', image_url: null };
    const image = path.resolve(imagePath);
    const under = (root: string) => image === root || image.startsWith(root + path.sep);
    const relative = under(datasetsRoot)
      ? path.relative(datasetsRoot, image)
      : `${path.basename(datasetPath)}/${path.relative(datasetPath, image)}`;
    return { relative_path: relative, image_url: roots.some(under) ? `/api/img/${encodeURIComponent(image)}` : null };
  };
  try {
    db = await new Promise<sqlite3.Database>((resolve, reject) => {
      const instance = new sqlite3.Database(filename, sqlite3.OPEN_READONLY, err =>
        err ? reject(err) : resolve(instance),
      );
    });
    db.configure('busyTimeout', 5000);
    // Keep entries, count and recorded total consistent while training/deleting.
    await all(db, 'BEGIN;');
    if (detail) {
      const row = await queryLossReportDetail(db, detail[0], detail[1], detail[2]);
      if (!row) return NextResponse.json({ error: 'Training example not found' }, { status: 404 });
      const source = await fs.stat(row.metadata.path).catch(() => null);
      return NextResponse.json({
        ...row,
        ...imageInfo(row.metadata.path, row.metadata.dataset_path),
        source_available: !!source,
      });
    }
    const result = await queryLossReport(db, options);
    return NextResponse.json({
      ...result,
      options,
      entries: result.entries.map(row => ({ ...row, ...imageInfo(row.path, row.dataset_path) })),
    });
  } catch (error) {
    console.error('Error reading per-image loss report:', error);
    return NextResponse.json({ error: 'Could not read the loss report.' }, { status: 500 });
  } finally {
    if (db) {
      await all(db, 'ROLLBACK;').catch(() => {});
      const connection = db;
      await new Promise<void>(resolve => connection.close(() => resolve()));
    }
  }
}
