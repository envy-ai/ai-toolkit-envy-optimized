import { NextResponse } from 'next/server';
import fs from 'fs';
import { getDatasetsRoot } from '@/server/settings';
import { getDatasetPreviews } from '@/server/datasetPreviews';

export async function GET(request: Request) {
  try {
    let datasetsPath = await getDatasetsRoot();

    // if folder doesnt exist, create it
    try {
      await fs.promises.access(datasetsPath);
    } catch {
      await fs.promises.mkdir(datasetsPath);
    }

    // find all the folders in the datasets folder
    let folders = (await fs.promises.readdir(datasetsPath, { withFileTypes: true }))
      .filter(dirent => dirent.isDirectory())
      .filter(dirent => !dirent.name.startsWith('.'))
      .map(dirent => dirent.name);

    if (new URL(request.url).searchParams.get('previews') === '1') {
      return NextResponse.json(await getDatasetPreviews(datasetsPath, folders));
    }
    return NextResponse.json(folders);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch datasets' }, { status: 500 });
  }
}
