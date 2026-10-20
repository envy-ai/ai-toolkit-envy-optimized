'use client';

import { useEffect, useRef, useState } from 'react';
import { Job } from '@prisma/client';
import { apiClient } from '@/utils/api';
import usePollLoop from '@/hooks/usePollLoop';
import { searchTerms } from '@/utils/search';

type UseJobsListProps = {
  onlyActive?: boolean;
  reloadInterval?: number | null;
  job_type?: string | null;
  /** Opt in: existing active-job consumers still request their complete list. */
  pageSize?: number;
  search?: string;
};

export default function useJobsList({
  onlyActive = false,
  reloadInterval = null,
  job_type = null,
  pageSize,
  search = '',
}: UseJobsListProps = {}) {
  const query = searchTerms(search).join(' ');
  const queryKey = JSON.stringify([onlyActive, job_type, pageSize ?? null, query]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loadedQuery, setLoadedQuery] = useState(queryKey);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [hasMore, setHasMore] = useState(false);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [error, setError] = useState('');
  const [loadMoreError, setLoadMoreError] = useState('');
  const activeRequest = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const depth = useRef(pageSize ?? 0);
  const cursor = useRef<string | null>(null);

  useEffect(() => {
    generation.current += 1;
    activeRequest.current?.abort();
    activeRequest.current = null;
    depth.current = pageSize ?? 0;
    cursor.current = null;
    setJobs([]);
    setLoadedQuery(queryKey);
    setHasMore(false);
    setIsLoadingMore(false);
    setError('');
    setLoadMoreError('');
    setStatus('loading');
    return () => {
      generation.current += 1;
      activeRequest.current?.abort();
      activeRequest.current = null;
    };
  }, [queryKey]);

  const fetchJobs = async (append: boolean) => {
    if (activeRequest.current || (append && (!pageSize || !cursor.current))) return false;
    const controller = new AbortController();
    const requestGeneration = generation.current;
    activeRequest.current = controller;
    if (append) {
      setIsLoadingMore(true);
      setLoadMoreError('');
    } else {
      setStatus('loading');
      setError('');
    }
    const params: Record<string, string> = {};
    if (job_type) params.job_type = job_type;
    if (onlyActive) params.only_active = 'true';
    if (query) params.search = query;
    if (pageSize) {
      params.limit = String(append ? pageSize : depth.current);
      if (append) params.cursor = cursor.current!;
    }
    try {
      const { data } = await apiClient.get('/api/jobs', { params, signal: controller.signal });
      if (controller.signal.aborted || requestGeneration !== generation.current) return;
      if (data.error) throw new Error(data.error);
      if (!Array.isArray(data.jobs)) throw new Error('Jobs response did not contain a job list.');
      if (pageSize && (typeof data.hasMore !== 'boolean' || (data.hasMore && typeof data.nextCursor !== 'string')))
        throw new Error('Jobs response did not contain valid pagination metadata.');
      setJobs(previous =>
        append
          ? [...new Map<string, Job>([...previous, ...data.jobs].map((job: Job) => [job.id, job])).values()]
          : data.jobs,
      );
      if (append) depth.current += pageSize!;
      cursor.current = data.nextCursor ?? null;
      setHasMore(Boolean(data.hasMore));
      setLoadedQuery(queryKey);
      setStatus('success');
    } catch (cause: any) {
      if (controller.signal.aborted || requestGeneration !== generation.current) return;
      const message = cause.response?.data?.error || cause.message || 'Could not fetch jobs.';
      console.error('Error fetching jobs:', cause);
      if (append) setLoadMoreError(message);
      else {
        setError(message);
        setStatus('error');
      }
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setIsLoadingMore(false);
      }
    }
    return true;
  };
  const refreshJobs = () => fetchJobs(false);
  const loadMore = () => fetchJobs(true);
  usePollLoop(refreshJobs, reloadInterval, [queryKey]);

  return {
    jobs: loadedQuery === queryKey ? jobs : [],
    setJobs,
    status: loadedQuery === queryKey ? status : ('loading' as const),
    refreshJobs,
    hasMore: loadedQuery === queryKey && hasMore,
    loadMore,
    isLoadingMore,
    error: loadedQuery === queryKey ? error : '',
    loadMoreError: loadedQuery === queryKey ? loadMoreError : '',
  };
}
