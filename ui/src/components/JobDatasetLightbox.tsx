'use client';

import { useEffect, useMemo, useState } from 'react';
import Lightbox from 'yet-another-react-lightbox';
import Captions from 'yet-another-react-lightbox/plugins/captions';
import Counter from 'yet-another-react-lightbox/plugins/counter';
import Zoom from 'yet-another-react-lightbox/plugins/zoom';
import { CgSpinner } from 'react-icons/cg';
import { apiClient } from '@/utils/api';

interface JobDatasetImage {
  path: string;
  url: string;
  filename: string;
  dataset_path: string;
  dataset_name: string;
  caption: string;
  caption_ext: string;
  caption_path: string;
}

interface JobDatasetLightboxProps {
  jobId: string | null;
  initialImagePath?: string | null;
  onClose: () => void;
}

const fallbackMessage = 'No caption found.';

export default function JobDatasetLightbox({ jobId, initialImagePath = null, onClose }: JobDatasetLightboxProps) {
  const [images, setImages] = useState<JobDatasetImage[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [initialIndex, setInitialIndex] = useState(0);

  useEffect(() => {
    if (!jobId) {
      setImages([]);
      setStatus('idle');
      setInitialIndex(0);
      return;
    }

    const controller = new AbortController();
    setStatus('loading');
    setImages([]);
    setInitialIndex(0);

    apiClient
      .get(`/api/jobs/${jobId}/dataset-images`, { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        const nextImages = Array.isArray(data?.images) ? data.images : [];
        const clickedIndex = initialImagePath ? nextImages.findIndex((image: JobDatasetImage) => image.path === initialImagePath) : -1;
        setImages(nextImages);
        setInitialIndex(clickedIndex >= 0 ? clickedIndex : 0);
        setStatus('success');
      })
      .catch(error => {
        if (controller.signal.aborted) return;
        console.error('Error fetching job dataset images:', error);
        setStatus('error');
      });

    return () => controller.abort();
  }, [jobId, initialImagePath]);

  const slides = useMemo(
    () =>
      images.map(image => ({
        src: image.url,
        title: image.filename,
        description: (
          <div className="max-w-5xl text-left text-sm leading-relaxed text-gray-100">
            <div className="mb-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-gray-300">
              <span>{image.dataset_name}</span>
              <span>{image.filename}</span>
              <span>{image.caption_ext}</span>
            </div>
            <div className="whitespace-pre-wrap break-words">{image.caption.trim() || fallbackMessage}</div>
          </div>
        ),
      })),
    [images],
  );

  const handleClose = () => {
    setStatus('idle');
    setImages([]);
    setInitialIndex(0);
    onClose();
  };

  if (!jobId) return null;

  if (status === 'loading') {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/80">
        <div className="flex items-center gap-3 rounded bg-gray-950 px-4 py-3 text-sm text-gray-200 border border-gray-700 shadow-xl">
          <CgSpinner className="h-5 w-5 animate-spin text-blue-400" />
          <span>Loading dataset images...</span>
        </div>
      </div>
    );
  }

  if (status === 'error' || (status === 'success' && slides.length === 0)) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/80 p-4">
        <div className="w-full max-w-md rounded bg-gray-950 border border-gray-700 shadow-xl">
          <div className="px-4 py-3 border-b border-gray-800 text-sm font-semibold text-gray-100">
            Dataset images unavailable
          </div>
          <div className="p-4 text-sm text-gray-300">
            {status === 'error' ? 'The dataset image list could not be loaded.' : 'No images were found in this job dataset.'}
          </div>
          <div className="px-4 py-3 flex justify-end">
            <button
              type="button"
              onClick={handleClose}
              className="rounded bg-gray-800 px-3 py-1.5 text-sm text-gray-100 hover:bg-gray-700"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <Lightbox
      open={status === 'success'}
      close={handleClose}
      slides={slides}
      index={initialIndex}
      plugins={[Captions, Counter, Zoom]}
      captions={{
        descriptionTextAlign: 'start',
        descriptionMaxLines: 12,
        showToggle: true,
      }}
      zoom={{
        maxZoomPixelRatio: 4,
        scrollToZoom: true,
      }}
      carousel={{
        imageFit: 'contain',
        preload: 2,
      }}
      controller={{
        closeOnBackdropClick: true,
      }}
    />
  );
}
