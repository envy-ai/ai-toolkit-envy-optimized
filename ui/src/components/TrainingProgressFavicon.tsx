'use client';

import { useEffect, useRef } from 'react';
import useJobsList from '@/hooks/useJobsList';
import {
  drawTrainingProgressFavicon,
  getRunningTrainingProgress,
  getTrainingQueueFaviconState,
  restoreDefaultFavicon,
} from '@/utils/faviconProgress';

export default function TrainingProgressFavicon() {
  const { jobs } = useJobsList({ onlyActive: true, reloadInterval: 5000, job_type: 'train' });
  const defaultFaviconHref = useRef<string | null>(null);
  const updateId = useRef(0);

  useEffect(() => {
    defaultFaviconHref.current = restoreDefaultFavicon();

    return () => {
      updateId.current += 1;
      restoreDefaultFavicon();
    };
  }, []);

  useEffect(() => {
    if (!defaultFaviconHref.current) return;

    const currentUpdateId = updateId.current + 1;
    updateId.current = currentUpdateId;

    const progress = getRunningTrainingProgress(jobs);
    const queueState = getTrainingQueueFaviconState(jobs);
    if (queueState === null && progress === null) {
      restoreDefaultFavicon();
      return;
    }

    drawTrainingProgressFavicon(defaultFaviconHref.current, progress, queueState, document, () => updateId.current === currentUpdateId);
  }, [jobs]);

  return null;
}
