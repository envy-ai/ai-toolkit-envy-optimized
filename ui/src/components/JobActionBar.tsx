import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';
import { Eye, Trash2, Pen, Play, Pause, Cog, X, Copy, Database, Save, OctagonX, Image, GitFork } from 'lucide-react';
import { LuLoader } from 'react-icons/lu';
import { Button } from '@headlessui/react';
import { openConfirm } from '@/components/ConfirmModal';
import { Job } from '@prisma/client';
import {
  startJob,
  stopJob,
  deleteJob,
  getAvaliableJobActions,
  markJobAsStopped,
  saveJobNow,
  sampleJobNow,
} from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import { Menu, MenuButton, MenuItem, MenuItems } from '@headlessui/react';
import { redirect } from 'next/navigation';
import { openCaptionDatasetModal } from '@/components/CaptionDatasetModal';
import { Modal } from '@/components/Modal';
import { apiClient } from '@/utils/api';
import type { SampleRenderState } from '@/server/jobSamples';

interface JobActionBarProps {
  job: Job;
  onRefresh?: () => void;
  afterDelete?: () => void;
  hideView?: boolean;
  className?: string;
  autoStartQueue?: boolean;
  // Where the gear menu opens; defaults to below. Use 'top end' when the bar is
  // pinned to the bottom of the screen so the menu opens upward into view.
  menuAnchor?: 'bottom' | 'top end';
}

type PendingAction = 'start' | 'remove' | 'stop' | 'delete' | 'save' | 'sample' | 'markStopped';

// If the job never reports a state change (e.g. the request silently failed
// server-side), unlock the bar after this long so it can't stay stuck.
const PENDING_TIMEOUT_MS = 30_000;

export default function JobActionBar({
  job,
  onRefresh,
  afterDelete,
  className,
  hideView,
  autoStartQueue = false,
  menuAnchor = 'bottom',
}: JobActionBarProps) {
  const { canStart, canStop, canDelete, canEdit, canEditSamples, canRemoveFromQueue } = getAvaliableJobActions(job);

  if (!afterDelete) afterDelete = onRefresh;

  // The action currently in flight. While set, every other action is disabled
  // and the pending one shows a spinner. It is cleared once the job comes back
  // in a different state (status / stop flag), on error, or after a timeout.
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [sampleRender, setSampleRender] = useState<SampleRenderState>({ status: 'idle', message: '' });
  const [samplePollError, setSamplePollError] = useState<{ message: string; stack?: string } | null>(null);
  const [showSampleRender, setShowSampleRender] = useState(false);
  const [hasOpenedMenu, setHasOpenedMenu] = useState(false);
  const sampleRequestVersion = useRef(0);
  const canRenderSaved = job.job_type === 'train' && ['stopped', 'completed', 'error'].includes(job.status);
  const renderingSamples = sampleRender.status === 'running';

  useEffect(() => {
    if ((!canRenderSaved || (!hideView && !hasOpenedMenu)) && !renderingSamples) return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      const version = sampleRequestVersion.current;
      try {
        const { data } = await apiClient.get<SampleRenderState>(`/api/jobs/${job.id}/render_samples`);
        if (disposed || version !== sampleRequestVersion.current) return;
        setSamplePollError(null);
        setSampleRender(data);
        if (data.status === 'error' && renderingSamples) setShowSampleRender(true);
        if (data.status === 'running') timer = setTimeout(poll, 2000);
        else if (renderingSamples) onRefresh?.();
      } catch (error: any) {
        if (!disposed && version === sampleRequestVersion.current) {
          setSamplePollError({
            message: error.response?.data?.error || error.message,
            stack: error.response?.data?.stack || error.stack,
          });
          timer = setTimeout(poll, 2000);
        }
      }
    };
    void poll();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
    // Refresh once on mount/job change, then poll only an active render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job.id, canRenderSaved, renderingSamples, hideView, hasOpenedMenu]);
  // Job state key captured when the pending action was fired. null means the
  // action does not change job state and clears as soon as the request resolves.
  const pendingStateKeyRef = useRef<string | null>(null);
  const jobStateKey = `${job.status}|${job.stop ? 1 : 0}`;

  const clearPending = () => {
    pendingStateKeyRef.current = null;
    setPending(null);
  };

  useEffect(() => {
    if (pending && pendingStateKeyRef.current !== null && pendingStateKeyRef.current !== jobStateKey) {
      clearPending();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobStateKey, pending]);

  useEffect(() => {
    if (!pending) return;
    const timer = setTimeout(clearPending, PENDING_TIMEOUT_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pending]);

  const runAction = async (action: PendingAction, fn: () => Promise<void>, waitForJobUpdate = true) => {
    if (pending) return;
    pendingStateKeyRef.current = waitForJobUpdate ? jobStateKey : null;
    setPending(action);
    try {
      await fn();
      if (!waitForJobUpdate) clearPending();
    } catch (e) {
      console.error(`Error running job action "${action}":`, e);
      clearPending();
    }
  };

  const isBusy = pending !== null || renderingSamples || samplePollError !== null;
  const iconSizeClass = 'w-5 h-5 sm:w-6 sm:h-6';
  const trainEditHref = canEditSamples ? `/jobs/new?id=${job.id}&sampleOnly=1` : `/jobs/new?id=${job.id}`;
  const actionButtonClass = 'ml-1 sm:ml-2 opacity-100 disabled:opacity-40 disabled:cursor-not-allowed';
  const menuItemClass = 'px-4 py-1 rounded flex items-center gap-2';
  const menuItemEnabledClass = `${menuItemClass} cursor-pointer hover:bg-gray-800`;
  const menuItemDisabledClass = `${menuItemClass} opacity-40 cursor-not-allowed`;
  const spinner = <LuLoader className={`${iconSizeClass} animate-spin`} />;
  const menuSpinner = <LuLoader className="w-4 h-4 animate-spin" />;
  const menuPending = pending === 'save' || pending === 'sample' || pending === 'markStopped';
  return (
    <div className={`flex items-center flex-shrink-0 ${className ?? ''}`}>
      {canStart && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canStart) return;
            runAction('start', async () => {
              await startJob(job.id);
              // start the queue as well
              if (autoStartQueue) {
                await startQueue(job.gpu_ids);
              }
              if (onRefresh) onRefresh();
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'start' ? spinner : <Play className={iconSizeClass} />}
        </Button>
      )}
      {canRemoveFromQueue && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canRemoveFromQueue) return;
            runAction('remove', async () => {
              await markJobAsStopped(job.id);
              if (onRefresh) onRefresh();
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'remove' ? spinner : <X className={iconSizeClass} />}
        </Button>
      )}
      {canStop && (
        <Button
          disabled={isBusy}
          onClick={() => {
            if (!canStop || isBusy) return;
            openConfirm({
              title: 'Pause Job',
              message:
                job.job_type === 'train'
                  ? `Pause "${job.name}"? The job will save a resumable checkpoint at the next completed training step, then stop.`
                  : `Stop "${job.name}"?`,
              type: 'info',
              confirmText: 'Pause',
              onConfirm: () =>
                runAction('stop', async () => {
                  await stopJob(job.id, job.job_type === 'train');
                  if (onRefresh) onRefresh();
                }),
            });
          }}
          className={actionButtonClass}
        >
          {pending === 'stop' ? spinner : <Pause className={iconSizeClass} />}
        </Button>
      )}
      {!hideView && (
        <Link href={`/jobs/${job.id}`} className="ml-1 sm:ml-2 text-gray-200 hover:text-gray-100 inline-block">
          <Eye className={iconSizeClass} />
        </Link>
      )}
      {job.job_type === 'caption' && canEdit && (
        <div
          className="ml-1 sm:ml-2 hover:text-gray-100 inline-block cursor-pointer"
          onClick={() =>
            openCaptionDatasetModal(
              job.job_ref || '',
              () => {
                if (onRefresh) onRefresh();
              },
              { jobId: job.id },
            )
          }
        >
          <Pen className={iconSizeClass} />
        </div>
      )}
      {job.job_type === 'train' && (canEdit || canEditSamples) && (
        <Link href={trainEditHref} className="ml-1 sm:ml-2 hover:text-gray-100 inline-block">
          <Pen className={iconSizeClass} />
        </Link>
      )}
      <Button
        disabled={isBusy}
        onClick={() => {
          if (isBusy) return;
          let message = `Are you sure you want to delete the job "${job.name}"? This will also permanently remove it from your disk.`;
          if (job.status === 'running') {
            message += ' WARNING: The job is currently running. You should stop it first if you can.';
          }
          openConfirm({
            title: 'Delete Job',
            message: message,
            type: 'warning',
            confirmText: 'Delete',
            onConfirm: () =>
              runAction('delete', async () => {
                if (job.status === 'running') {
                  try {
                    await stopJob(job.id);
                  } catch (e) {
                    console.error('Error stopping job before deleting:', e);
                  }
                }
                await deleteJob(job.id);
                if (afterDelete) afterDelete();
              }),
          });
        }}
        className={actionButtonClass}
      >
        {pending === 'delete' ? spinner : <Trash2 className={iconSizeClass} />}
      </Button>
      <div className="border-r border-1 border-gray-700 ml-1 sm:ml-2 inline"></div>
      <Menu>
        <MenuButton aria-label="Job actions" onClick={() => setHasOpenedMenu(true)} className={'ml-1 sm:ml-2'}>
          {menuPending ? spinner : <Cog className={iconSizeClass} />}
        </MenuButton>
        <MenuItems
          anchor={{ to: menuAnchor, gap: 16 }}
          className="bg-gray-900 border border-gray-700 rounded shadow-lg w-60 px-2 py-2 z-50"
        >
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?forkId=${job.id}`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              >
                <GitFork className="w-4 h-4" />
                Fork this job
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?cloneId=${job.id}`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              >
                <Copy className="w-4 h-4" />
                Clone Job
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'train' && (
            <MenuItem>
              <Link
                href={`/jobs/new?cloneId=${job.id}&withEmbeddings=1`}
                className="cursor-pointer px-4 py-1 hover:bg-gray-800 rounded flex items-center gap-2"
              >
                <Database className="w-4 h-4" />
                Clone Job w/ Embeddings
              </Link>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem disabled={isBusy}>
              <div
                className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
                onClick={() => {
                  if (isBusy) return;
                  // does not change job state, so release as soon as the request resolves
                  runAction(
                    'save',
                    async () => {
                      await saveJobNow(job.id);
                      if (onRefresh) onRefresh();
                    },
                    false,
                  );
                }}
              >
                {pending === 'save' ? menuSpinner : <Save className="w-4 h-4" />}
                Save Next Step
              </div>
            </MenuItem>
          )}
          {canRenderSaved && (
            <MenuItem disabled={isBusy}>
              <button
                type="button"
                className={`${isBusy ? menuItemDisabledClass : menuItemEnabledClass} w-full text-left data-[focus]:bg-gray-800`}
                disabled={isBusy}
                onClick={async () => {
                  if (isBusy) return;
                  sampleRequestVersion.current++;
                  setShowSampleRender(true);
                  setPending('sample');
                  try {
                    const { data } = await apiClient.post<SampleRenderState>(`/api/jobs/${job.id}/render_samples`);
                    setSampleRender(data);
                  } catch (error: any) {
                    console.error('Saved sample rendering failed:', error);
                    setSampleRender({
                      status: 'error',
                      message: error.response?.data?.error || error.message,
                      stack: error.response?.data?.stack || error.stack,
                    });
                  } finally {
                    clearPending();
                  }
                }}
              >
                {renderingSamples || pending === 'sample' ? menuSpinner : <Image className="w-4 h-4" />}
                {renderingSamples ? 'Rendering samples…' : 'Render Samples'}
              </button>
            </MenuItem>
          )}
          {job.job_type === 'train' && canStop && (
            <MenuItem disabled={isBusy}>
              <div
                className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
                onClick={() => {
                  if (isBusy) return;
                  // does not change job state, so release as soon as the request resolves
                  runAction(
                    'sample',
                    async () => {
                      await sampleJobNow(job.id);
                      if (onRefresh) onRefresh();
                    },
                    false,
                  );
                }}
              >
                {pending === 'sample' ? menuSpinner : <Image className="w-4 h-4" />}
                Sample Next Step
              </div>
            </MenuItem>
          )}
          <MenuItem disabled={isBusy}>
            <div
              className={isBusy ? menuItemDisabledClass : menuItemEnabledClass}
              onClick={() => {
                if (isBusy) return;
                let message = `Are you sure you want to mark this job as stopped? This will set the job status to 'stopped' if the status is hung. Only do this if you are 100% sure the job is stopped. This will NOT stop the job.`;
                openConfirm({
                  title: 'Mark Job as Stopped',
                  message: message,
                  type: 'warning',
                  confirmText: 'Mark as Stopped',
                  onConfirm: () =>
                    runAction('markStopped', async () => {
                      await markJobAsStopped(job.id);
                      onRefresh && onRefresh();
                    }),
                });
              }}
            >
              {pending === 'markStopped' ? menuSpinner : <OctagonX className="w-4 h-4" />}
              Mark as Stopped
            </div>
          </MenuItem>
        </MenuItems>
      </Menu>
      {(sampleRender.status !== 'idle' || samplePollError) && (
        <button
          type="button"
          className={`ml-2 ${sampleRender.status === 'error' || samplePollError ? 'text-red-400' : 'text-gray-300'}`}
          onClick={() => setShowSampleRender(true)}
          aria-label={`Sample render details: ${samplePollError?.message || sampleRender.message}`}
          title={samplePollError?.message || sampleRender.message}
        >
          {renderingSamples ? spinner : <Image className={iconSizeClass} />}
          <span className="sr-only" role="status" aria-live="polite">
            {samplePollError?.message || sampleRender.message}
          </span>
        </button>
      )}
      <Modal isOpen={showSampleRender} onClose={() => setShowSampleRender(false)} title="Render Samples" size="lg">
        <p
          role={sampleRender.status === 'error' ? 'alert' : 'status'}
          aria-live="polite"
          className="whitespace-pre-wrap"
        >
          {pending === 'sample' ? 'Starting sample render…' : sampleRender.message}
        </p>
        {sampleRender.checkpoint && (
          <p className="mt-3 text-sm text-gray-400 break-all">Checkpoint: {sampleRender.checkpoint}</p>
        )}
        {samplePollError && (
          <p role="alert" className="mt-3 text-red-300">
            Cannot refresh sample render status: {samplePollError.message}. Retrying…
          </p>
        )}
        {samplePollError?.stack && (
          <pre className="mt-3 p-3 bg-gray-950 text-red-300 text-xs whitespace-pre-wrap overflow-auto max-h-80">
            {samplePollError.stack}
          </pre>
        )}
        {sampleRender.stack && (
          <pre className="mt-3 p-3 bg-gray-950 text-red-300 text-xs whitespace-pre-wrap overflow-auto max-h-80">
            {sampleRender.stack}
          </pre>
        )}
      </Modal>
    </div>
  );
}
