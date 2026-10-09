'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient } from '@/utils/api';
import type { DatasetPreview } from '@/server/datasetPreviews';

export default function useDatasetList(includePreviews = false) {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [firstImages, setFirstImages] = useState<Record<string, string | null>>({});
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestRef = useRef<AbortController | null>(null);

  const refreshDatasets = useCallback(() => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    setStatus('loading');
    apiClient
      .get(includePreviews ? '/api/datasets/list?previews=1' : '/api/datasets/list', { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        if (controller.signal.aborted) return;
        const entries: DatasetPreview[] = includePreviews
          ? data
          : data.map((name: string) => ({ name, first_image: null }));
        entries.sort((a, b) => a.name.localeCompare(b.name));
        setDatasets(entries.map(entry => entry.name));
        setFirstImages(Object.fromEntries(entries.map(entry => [entry.name, entry.first_image])));
        setStatus('success');
      })
      .catch(error => {
        if (controller.signal.aborted) return;
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  }, [includePreviews]);
  useEffect(() => {
    refreshDatasets();
    window.addEventListener('toolkit:assistant:changed', refreshDatasets);
    return () => {
      requestRef.current?.abort();
      window.removeEventListener('toolkit:assistant:changed', refreshDatasets);
    };
  }, [refreshDatasets]);

  return { datasets, setDatasets, firstImages, status, refreshDatasets };
}
