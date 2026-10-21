import { useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import { matchesSearch } from '@/utils/search';
import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { GpuInfo } from '@/types';
import JobActionBar from './JobActionBar';
import { Job, Queue } from '@prisma/client';
import useQueueList from '@/hooks/useQueueList';
import classNames from 'classnames';
import { reorderQueueJobs, startQueue, stopQueue } from '@/utils/queue';
import { CgSpinner } from 'react-icons/cg';
import useGPUInfo from '@/hooks/useGPUInfo';
import { openConfirm } from '@/components/ConfirmModal';
import { deleteJob, getTotalSteps, stopJob } from '@/utils/jobs';
import { GripVertical, Trash2 } from 'lucide-react';
import JobDatasetLightbox from '@/components/JobDatasetLightbox';
import {
  closestCenter,
  DndContext,
  DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';

interface JobsTableProps {
  search?: string;
  autoStartQueue?: boolean;
  onlyActive?: boolean;
  job_type?: string | null;
  pageSize?: number;
  scrollRoot?: RefObject<HTMLDivElement | null>;
}

interface QueueTableProps {
  columns: TableColumn[];
  rows: Job[];
  queueID: string;
  isLoading: boolean;
  onRefresh: () => void;
  onReorder: (queueID: string, orderedJobIds: string[]) => void;
  theadClassName?: string;
  reorderDisabledReason?: string;
}

function renderTableCells(row: Job, columns: TableColumn[]) {
  return columns.map(column => (
    <td key={column.key} className={classNames('px-3 py-2', column.className)}>
      {column.render ? column.render(row) : (row as any)[column.key]}
    </td>
  ));
}

function StaticQueueRow({ row, columns, rowClass }: { row: Job; columns: TableColumn[]; rowClass: string }) {
  return (
    <tr className={`${rowClass} border-b border-gray-700 hover:bg-gray-700`}>
      <td className="px-3 py-2 w-8">
        <GripVertical className="h-4 w-4 text-gray-600" aria-hidden="true" />
      </td>
      {renderTableCells(row, columns)}
    </tr>
  );
}

function SortableQueueRow({
  row,
  columns,
  rowClass,
  disabledReason,
}: {
  row: Job;
  columns: TableColumn[];
  rowClass: string;
  disabledReason?: string;
}) {
  const { attributes, listeners, setActivatorNodeRef, setNodeRef, transform, transition, isDragging } = useSortable({
    id: row.id,
    disabled: Boolean(disabledReason),
  });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
  };

  return (
    <tr
      ref={setNodeRef}
      style={style}
      className={classNames(`${rowClass} border-b border-gray-700 hover:bg-gray-700`, {
        'relative z-10 opacity-80': isDragging,
      })}
    >
      <td className="px-3 py-2 w-8">
        <button
          ref={setActivatorNodeRef}
          type="button"
          disabled={Boolean(disabledReason)}
          title={disabledReason}
          className="inline-flex h-7 w-7 items-center justify-center rounded text-gray-400 hover:bg-gray-700 hover:text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-grab active:cursor-grabbing disabled:cursor-not-allowed disabled:opacity-40"
          aria-label={disabledReason ? `Reorder ${row.name}: ${disabledReason}` : `Reorder ${row.name}`}
          {...attributes}
          {...listeners}
        >
          <GripVertical className="h-4 w-4" aria-hidden="true" />
        </button>
      </td>
      {renderTableCells(row, columns)}
    </tr>
  );
}

function SortableQueueTable({
  columns,
  rows,
  queueID,
  isLoading,
  onRefresh,
  onReorder,
  theadClassName = 'text-gray-400',
  reorderDisabledReason,
}: QueueTableProps) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  const runningRows = rows.filter(job => job.status === 'running' || job.status === 'stopping');
  const queuedRows = rows.filter(job => job.status === 'queued');
  const otherRows = rows.filter(job => !['running', 'stopping', 'queued'].includes(job.status));
  const queuedRowIds = queuedRows.map(job => job.id);

  const handleDragEnd = (event: DragEndEvent) => {
    if (reorderDisabledReason) return;
    const { active, over } = event;
    if (!over || active.id === over.id) return;

    const oldIndex = queuedRowIds.indexOf(String(active.id));
    const newIndex = queuedRowIds.indexOf(String(over.id));
    if (oldIndex === -1 || newIndex === -1) return;

    const reorderedRows = arrayMove(queuedRows, oldIndex, newIndex);
    onReorder(
      queueID,
      reorderedRows.map(job => job.id),
    );
  };

  if (isLoading) {
    return <UniversalTable columns={columns} rows={rows} isLoading={isLoading} onRefresh={onRefresh} />;
  }

  if (rows.length === 0) {
    return <UniversalTable columns={columns} rows={rows} isLoading={isLoading} onRefresh={onRefresh} />;
  }

  return (
    <div className="w-full bg-gray-900 rounded-md shadow-md">
      <div className="overflow-x-auto">
        <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
          <table className="w-full text-sm text-left text-gray-300">
            <thead className={classNames('text-xs uppercase bg-gray-800', theadClassName)}>
              <tr>
                <th className="px-3 py-2 w-8"></th>
                {columns.map(column => (
                  <th key={column.key} className="px-3 py-2">
                    {column.title}
                  </th>
                ))}
              </tr>
            </thead>
            <SortableContext items={queuedRowIds} strategy={verticalListSortingStrategy}>
              <tbody>
                {runningRows.map((row, index) => (
                  <StaticQueueRow
                    key={row.id}
                    row={row}
                    columns={columns}
                    rowClass={index % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800'}
                  />
                ))}
                {queuedRows.map((row, index) => (
                  <SortableQueueRow
                    key={row.id}
                    row={row}
                    columns={columns}
                    rowClass={(runningRows.length + index) % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800'}
                    disabledReason={reorderDisabledReason}
                  />
                ))}
                {otherRows.map((row, index) => (
                  <StaticQueueRow
                    key={row.id}
                    row={row}
                    columns={columns}
                    rowClass={
                      (runningRows.length + queuedRows.length + index) % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800'
                    }
                  />
                ))}
              </tbody>
            </SortableContext>
          </table>
        </DndContext>
      </div>
    </div>
  );
}

export default function JobsTable({
  onlyActive = false,
  job_type = null,
  search = '',
  pageSize,
  scrollRoot,
}: JobsTableProps) {
  const { jobs, setJobs, status, refreshJobs, hasMore, loadMore, isLoadingMore, error, loadMoreError } = useJobsList({
    onlyActive,
    reloadInterval: 5000,
    job_type,
    pageSize,
    search: pageSize ? search : '',
  });
  const latestRefreshJobs = useRef(refreshJobs);
  latestRefreshJobs.current = refreshJobs;
  const { queues, status: queueStatus, refreshQueues } = useQueueList();
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleteProgress, setDeleteProgress] = useState<{ done: number; total: number } | null>(null);
  const [datasetLightboxJob, setDatasetLightboxJob] = useState<{ jobId: string; imagePath: string | null } | null>(
    null,
  );
  const sentinel = useRef<HTMLDivElement>(null);
  const [scrollRequested, setScrollRequested] = useState(false);
  const reorderInFlight = useRef(false);
  const [reorderSaving, setReorderSaving] = useState(false);
  const [reorderError, setReorderError] = useState<{ message: string; stack?: string } | null>(null);
  const reorderDisabledReason = reorderSaving ? 'Saving queue order…' : undefined;

  useEffect(() => {
    const root = scrollRoot?.current;
    if (!pageSize || !root) return;
    root.scrollTop = 0;
    setScrollRequested(false);
    let previousTop = root.scrollTop;
    const onScroll = () => {
      if (root.scrollTop > previousTop) setScrollRequested(true);
      previousTop = root.scrollTop;
    };
    root.addEventListener('scroll', onScroll, { passive: true });
    return () => root.removeEventListener('scroll', onScroll);
  }, [pageSize, search, scrollRoot]);

  useEffect(() => {
    const root = scrollRoot?.current;
    const target = sentinel.current;
    if (
      !pageSize ||
      !root ||
      !target ||
      !hasMore ||
      !scrollRequested ||
      status !== 'success' ||
      !isGPUInfoLoaded ||
      isLoadingMore ||
      loadMoreError ||
      error
    )
      return;
    const observer = new IntersectionObserver(
      entries => {
        if (entries.some(entry => entry.isIntersecting)) {
          void loadMore().then(started => {
            if (started) setScrollRequested(false);
          });
        }
      },
      { root, threshold: 0.01 },
    );
    observer.observe(target);
    return () => observer.disconnect();
  }, [
    pageSize,
    scrollRoot,
    hasMore,
    scrollRequested,
    status,
    isGPUInfoLoaded,
    isLoadingMore,
    loadMoreError,
    error,
    loadMore,
  ]);

  const refresh = () => {
    refreshJobs();
    refreshQueues();
  };

  const isDeleting = deleteProgress !== null;
  const visibleJobs = useMemo(() => jobs.filter(job => matchesSearch(job.name, search)), [jobs, search]);
  const selectedJobs = visibleJobs.filter(job => selectedIds.has(job.id));
  const allSelected = visibleJobs.length > 0 && visibleJobs.every(job => selectedIds.has(job.id));

  const toggleRow = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleAll = () => {
    setSelectedIds(allSelected ? new Set() : new Set(visibleJobs.map(job => job.id)));
  };

  const openDatasetLightbox = (jobId: string, imagePath?: string | null) => {
    setDatasetLightboxJob({ jobId, imagePath: imagePath ?? null });
  };

  const onReorderQueue = async (queueID: string, orderedJobIds: string[]) => {
    if (reorderInFlight.current) return;
    reorderInFlight.current = true;
    setReorderSaving(true);
    setReorderError(null);
    try {
      // Submit exactly the visible queued IDs. Only the server owns the full
      // queue ordering, including jobs omitted by pagination or search.
      const positions = await reorderQueueJobs(queueID, orderedJobIds);
      const positionById = new Map(positions.map(position => [position.id, position.queue_position]));
      setJobs(previous =>
        previous.map(job =>
          job.status === 'queued' && job.gpu_ids === queueID && positionById.has(job.id)
            ? { ...job, queue_position: positionById.get(job.id)! }
            : job,
        ),
      );
    } catch (cause: any) {
      console.error('Could not save queue order:', cause);
      setReorderError({
        message: cause.response?.data?.error || cause.message || 'Could not save queue order.',
        stack: cause.response?.data?.stack || cause.stack,
      });
    } finally {
      // Cancel an older poll so its pre-save ranks cannot overwrite this save.
      await latestRefreshJobs.current({ cancelPending: true });
      reorderInFlight.current = false;
      setReorderSaving(false);
    }
  };

  const onMassDelete = () => {
    const jobsToDelete = selectedJobs;
    if (jobsToDelete.length === 0) return;
    const runningCount = jobsToDelete.filter(job => job.status === 'running').length;
    let message = `Are you sure you want to delete ${jobsToDelete.length} job${
      jobsToDelete.length === 1 ? '' : 's'
    }? This will also permanently remove them from your disk.`;
    if (runningCount > 0) {
      message += ` WARNING: ${runningCount} of them ${
        runningCount === 1 ? 'is' : 'are'
      } currently running and will be stopped first.`;
    }
    openConfirm({
      title: 'Delete Jobs',
      message: message,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        setDeleteProgress({ done: 0, total: jobsToDelete.length });
        for (let i = 0; i < jobsToDelete.length; i++) {
          const job = jobsToDelete[i];
          try {
            if (job.status === 'running') {
              try {
                await stopJob(job.id);
              } catch (e) {
                console.error('Error stopping job before deleting:', e);
              }
            }
            await deleteJob(job.id);
            setSelectedIds(prev => {
              const next = new Set(prev);
              next.delete(job.id);
              return next;
            });
          } catch (e) {
            console.error('Error deleting job:', job.name, e);
          }
          setDeleteProgress({ done: i + 1, total: jobsToDelete.length });
          refreshJobs();
        }
        setDeleteProgress(null);
        refresh();
      },
    });
  };

  const columns: TableColumn[] = [
    {
      title: (
        <input
          type="checkbox"
          checked={allSelected}
          onChange={toggleAll}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
      key: 'select',
      className: 'w-8',
      render: row => (
        <input
          type="checkbox"
          checked={selectedIds.has(row.id)}
          onChange={() => toggleRow(row.id)}
          disabled={isDeleting}
          className="cursor-pointer accent-blue-500"
        />
      ),
    },
    {
      title: '',
      key: 'dataset_thumbnail_url',
      className: 'w-[112px]',
      render: row =>
        row.dataset_thumbnail_url ? (
          <button
            type="button"
            onClick={() => openDatasetLightbox(row.id, row.dataset_thumbnail_path)}
            className="block max-w-[100px] max-h-[100px] rounded border border-gray-700 bg-gray-950 hover:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label={`Browse dataset images for ${row.name}`}
          >
            <img
              src={row.dataset_thumbnail_url}
              alt=""
              loading="lazy"
              className="max-w-[100px] max-h-[100px] object-contain"
            />
          </button>
        ) : null,
    },
    {
      title: 'Name',
      key: 'name',
      className: 'max-w-xs whitespace-normal break-all',
      render: row => {
        let title = row.name;
        let href = `/jobs/${row.id}`;
        // if (row.job_type === 'train') title = `Train: ${title}`;
        if (row.job_type === 'inference') {
          href = `/generate?job=${row.id}`;
          title = (
            <>
              <small className="opacity-50">INFERENCE ENGINE: </small> {row.name}
            </>
          );
        }
        if (row.job_type === 'caption') {
          let splits = row.job_ref.split(/[/\\]/);
          const datasetPath = `${splits[splits.length - 1]}`;
          href = `/datasets/${datasetPath}`;
          title = (
            <>
              <small className="opacity-50">CAPTION: </small> {datasetPath}
            </>
          );
        }
        return (
          <Link href={href} className="font-medium whitespace-normal break-all">
            {['running', 'stopping'].includes(row.status) ? (
              <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
            ) : null}
            {title}
          </Link>
        );
      },
    },
    {
      title: 'Steps',
      key: 'steps',
      render: row => {
        const totalSteps = getTotalSteps(row);
        if (!totalSteps) {
          return <></>;
        }

        return (
          <div>
            <div className="text-xs text-gray-400">
              {row.step} / {totalSteps}
            </div>
            <div className="bg-gray-700 rounded-full h-1.5">
              <div
                className="bg-blue-500 h-1.5 rounded-full"
                style={{ width: `${(row.step / totalSteps) * 100}%` }}
              ></div>
            </div>
          </div>
        );
      },
    },
    {
      title: 'GPU',
      key: 'gpu_ids',
    },
    {
      title: 'Status',
      key: 'status',
      render: row => {
        let statusClass = 'text-gray-400';
        if (row.status === 'completed') statusClass = 'text-green-400';
        if (row.status === 'failed') statusClass = 'text-red-400';
        if (row.status === 'running') statusClass = 'text-blue-400';

        return <span className={statusClass}>{row.status}</span>;
      },
    },
    {
      title: 'Info',
      key: 'info',
      className: 'max-w-xs whitespace-normal break-words [overflow-wrap:anywhere]',
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: row => {
        return <JobActionBar job={row} onRefresh={refreshJobs} autoStartQueue={false} />;
      },
    },
  ];

  const jobsDict = useMemo(() => {
    if (!isGPUInfoLoaded) return {};
    if (jobs.length === 0) return {};
    let jd: { [key: string]: { name: string; jobs: Job[] } } = {};
    gpuList.forEach(gpu => {
      jd[`${gpu.index}`] = { name: `${gpu.name}`, jobs: [] };
    });
    if (jobs.some(job => job.gpu_ids === 'cpu')) jd['cpu'] = { name: 'CPU / API', jobs: [] };
    jd['Idle'] = { name: 'Idle', jobs: [] };
    jobs.forEach(job => {
      const gpu = gpuList.find(gpu => job.gpu_ids?.split(',').includes(gpu.index.toString())) as GpuInfo;
      const key = job.gpu_ids === 'cpu' ? 'cpu' : `${gpu?.index || '0'}`;
      if (['queued', 'running', 'stopping'].includes(job.status) && key in jd) {
        jd[key].jobs.push(job);
      } else {
        jd['Idle'].jobs.push(job);
      }
    });
    // sort the queued/running jobs by queue position
    Object.keys(jd).forEach(key => {
      if (key === 'Idle') {
        jd[key].jobs.sort((a, b) => {
          // sort by updated_at, newest first
          return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
        });
      } else {
        jd[key].jobs.sort((a, b) => {
          if (a.queue_position === null) return 1;
          if (b.queue_position === null) return -1;
          return a.queue_position - b.queue_position;
        });
      }
    });
    return jd;
  }, [jobs, queues, isGPUInfoLoaded]);

  let isLoading = status === 'loading' || queueStatus === 'loading' || !isGPUInfoLoaded;

  // if job dict is populated, we are always loaded
  if (Object.keys(jobsDict).length > 0) isLoading = false;

  return (
    <div>
      <JobDatasetLightbox
        jobId={datasetLightboxJob?.jobId ?? null}
        initialImagePath={datasetLightboxJob?.imagePath ?? null}
        onClose={() => setDatasetLightboxJob(null)}
      />
      {(selectedJobs.length > 0 || isDeleting) && (
        <div className="fixed top-16 left-1/2 -translate-x-1/2 z-50 flex items-center gap-3 px-4 py-2 bg-gray-800 rounded-lg border border-gray-700 shadow-lg">
          {isDeleting ? (
            <>
              <CgSpinner className="inline animate-spin text-red-400" />
              <span className="text-sm text-gray-300">
                Deleting {deleteProgress.done} / {deleteProgress.total}...
              </span>
            </>
          ) : (
            <>
              <span className="text-sm text-gray-300 flex-1">
                {selectedJobs.length} job{selectedJobs.length === 1 ? '' : 's'} selected
              </span>
              <button
                onClick={() => setSelectedIds(new Set())}
                className="text-xs text-gray-300 bg-gray-700 hover:bg-gray-600 px-2 py-1 rounded"
              >
                Clear
              </button>
              <button
                onClick={onMassDelete}
                className="text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded flex items-center gap-1"
              >
                <Trash2 className="w-3.5 h-3.5" />
                Delete Selected
              </button>
            </>
          )}
        </div>
      )}
      {error && (
        <div role="alert" className="mb-4 rounded-md border border-red-700 bg-red-950 p-3 text-sm text-red-200">
          Could not refresh jobs: {error}{' '}
          <button
            type="button"
            onClick={() => void refreshJobs()}
            className="underline focus-visible:outline focus-visible:outline-blue-400"
          >
            Retry
          </button>
        </div>
      )}
      {reorderSaving && (
        <p role="status" className="mb-3 text-sm text-gray-300">
          Saving queue order…
        </p>
      )}
      {reorderError && (
        <div role="alert" className="mb-4 rounded-md border border-red-700 bg-red-950 p-3 text-sm text-red-200">
          <p>Could not save queue order: {reorderError.message}</p>
          {reorderError.stack && (
            <details className="mt-2">
              <summary>Error details</summary>
              <pre className="mt-2 overflow-auto whitespace-pre-wrap">{reorderError.stack}</pre>
            </details>
          )}
        </div>
      )}
      {!isLoading && !error && visibleJobs.length === 0 && search.trim() && (
        <p role="status" className="p-6 text-center text-sm text-gray-400">
          No jobs match your search.
        </p>
      )}
      {Object.keys(jobsDict)
        .sort()
        .filter(key => key !== 'Idle')
        .map(gpuKey => {
          const queue = queues.find(q => `${q.gpu_ids}` === gpuKey) as Queue;
          return (
            <div key={gpuKey} className="mb-6">
              <div
                className={classNames(
                  'text-md flex flex-wrap gap-y-1 px-2 sm:px-4 py-1 rounded-t-lg',
                  { 'bg-green-600 dark:bg-green-900': queue?.is_running },
                  { 'bg-red-600 dark:bg-red-900': !queue?.is_running },
                )}
              >
                <div className="flex items-center space-x-2 flex-1 min-w-0 py-2">
                  <h2 className="font-semibold text-white truncate">{jobsDict[gpuKey].name}</h2>
                  <span className="px-2 py-0.5 bg-gray-700 rounded-full text-xs text-gray-300 flex-shrink-0">
                    # {queue?.gpu_ids}
                  </span>
                </div>
                <div className="text-sm text-gray-300 italic flex items-center flex-shrink-0">
                  {queue?.is_running ? (
                    <>
                      <span className="text-green-100 dark:text-green-400 mr-2">Queue Running</span>
                      <button
                        onClick={async () => {
                          await stopQueue(queue.gpu_ids as string);
                          refresh();
                        }}
                        className="ml-2 sm:ml-4 text-xs text-white bg-red-600 hover:bg-red-700 px-2 py-1 rounded"
                      >
                        STOP
                      </button>
                    </>
                  ) : (
                    <>
                      <span className="text-red-100 dark:text-red-400 mr-2">Queue Stopped</span>
                      <button
                        onClick={async () => {
                          await startQueue(gpuKey);
                          refresh();
                        }}
                        className="ml-2 sm:ml-4 text-xs text-white bg-green-600 hover:bg-green-700 px-2 py-1 rounded"
                      >
                        START
                      </button>
                    </>
                  )}
                </div>
              </div>
              <SortableQueueTable
                columns={columns}
                rows={jobsDict[gpuKey].jobs.filter(job => matchesSearch(job.name, search))}
                queueID={gpuKey}
                isLoading={isLoading}
                onRefresh={refresh}
                onReorder={onReorderQueue}
                reorderDisabledReason={reorderDisabledReason}
                theadClassName={
                  queue?.is_running
                    ? 'bg-green-700 dark:bg-green-950 text-white dark:text-gray-400'
                    : 'bg-red-700 dark:bg-red-950 text-white dark:text-gray-400'
                }
              />
            </div>
          );
        })}
      {!onlyActive && Object.keys(jobsDict).includes('Idle') && (
        <div className="mb-6 opacity-50">
          <div className="text-md flex px-4 py-1 rounded-t-lg bg-slate-600">
            <div className="flex items-center space-x-2 flex-1 py-2">
              <h2 className="font-semibold text-gray-100">Idle</h2>
            </div>
          </div>
          <UniversalTable
            columns={columns}
            rows={jobsDict['Idle'].jobs.filter(job => matchesSearch(job.name, search))}
            isLoading={isLoading}
            onRefresh={refresh}
          />
        </div>
      )}
      {pageSize && (
        <div
          ref={sentinel}
          className="mb-6 flex min-h-16 flex-col items-center justify-center gap-2 text-sm text-gray-400"
        >
          <p role="status">
            {jobs.length} matching job{jobs.length === 1 ? '' : 's'} loaded
            {!hasMore && status === 'success' ? ' · All results loaded' : ''}
          </p>
          {loadMoreError && (
            <p role="alert" className="text-red-300">
              Could not load more jobs: {loadMoreError}
            </p>
          )}
          {hasMore && (
            <button
              type="button"
              disabled={isLoadingMore || status === 'loading'}
              onClick={() => {
                setScrollRequested(false);
                void loadMore();
              }}
              className="rounded-md border border-gray-600 px-4 py-2 text-gray-200 hover:bg-gray-800 disabled:opacity-50 focus-visible:outline focus-visible:outline-blue-400"
            >
              {isLoadingMore ? 'Loading more…' : loadMoreError ? 'Retry loading more' : 'Load more'}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
