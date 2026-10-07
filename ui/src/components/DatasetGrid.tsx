'use client';

import { useState } from 'react';
import Link from 'next/link';
import { FaRegTrashAlt } from 'react-icons/fa';
import { MdImageNotSupported } from 'react-icons/md';
import { encodeFilePathForUrl } from '@/utils/basic';

export type DatasetGridSize = 'small' | 'medium' | 'large';
const cardWidths: Record<DatasetGridSize, number> = { small: 160, medium: 240, large: 340 };

function DatasetCard({
  name,
  image,
  onDelete,
}: {
  name: string;
  image: string | null;
  onDelete: (name: string) => void;
}) {
  const [failed, setFailed] = useState(false);
  return (
    <article className="relative rounded-lg border border-gray-700 bg-gray-900 overflow-hidden hover:border-gray-500">
      <Link
        href={`/datasets/${encodeURIComponent(name)}`}
        className="block focus-visible:outline focus-visible:outline-2 focus-visible:outline-blue-400"
      >
        <div className="aspect-square bg-gray-950 flex items-center justify-center">
          {image && !failed ? (
            <img
              src={`/api/img/${encodeFilePathForUrl(image)}?thumb=1`}
              alt={`First image in ${name}`}
              loading="lazy"
              className="w-full h-full object-contain"
              onError={() => setFailed(true)}
            />
          ) : (
            <div className="flex flex-col items-center gap-2 text-gray-500 p-4 text-center text-sm">
              <MdImageNotSupported className="w-8 h-8" aria-hidden="true" />
              <span>{failed ? 'Preview unavailable' : 'No image preview'}</span>
            </div>
          )}
        </div>
        <div className="p-3 pr-12 min-h-16 text-sm text-gray-200 break-all line-clamp-2" title={name}>
          {name}
        </div>
      </Link>
      <button
        type="button"
        aria-label={`Delete dataset ${name}`}
        title={`Delete dataset ${name}`}
        className="absolute bottom-2 right-2 text-gray-300 hover:bg-red-600 hover:text-white p-2 rounded-full transition-colors"
        onClick={() => onDelete(name)}
      >
        <FaRegTrashAlt aria-hidden="true" />
      </button>
    </article>
  );
}

export default function DatasetGrid({
  datasets,
  firstImages,
  size,
  onDelete,
}: {
  datasets: string[];
  firstImages: Record<string, string | null>;
  size: DatasetGridSize;
  onDelete: (name: string) => void;
}) {
  return (
    <div
      className="grid gap-4 pb-4"
      style={{ gridTemplateColumns: `repeat(auto-fill, minmax(min(100%, ${cardWidths[size]}px), 1fr))` }}
    >
      {datasets.map(name => (
        <DatasetCard
          key={`${name}/${firstImages[name] ?? ''}`}
          name={name}
          image={firstImages[name] ?? null}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
}
