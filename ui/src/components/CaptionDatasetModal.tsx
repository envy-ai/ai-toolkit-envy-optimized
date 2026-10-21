'use client';
import React, { useState, useEffect, useRef } from 'react';
import { Modal } from '@/components/Modal';
import { createGlobalState } from 'react-global-hooks';
import { CaptionJobConfig } from '@/types';
import { createCaptionJobPreferences, defaultCaptionJobConfig, restoreCaptionJobPreferences } from '@/helpers/captionJobConfig';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import { isMac } from '@/helpers/basic';
import useGPUInfo from '@/hooks/useGPUInfo';
import { apiClient } from '@/utils/api';
import { v4 as uuidv4 } from 'uuid';
import { startJob } from '@/utils/jobs';
import { startQueue } from '@/utils/queue';
import CaptionSimpleJob from '@/components/CaptionSimpleJob';
import AdvancedConfigEditor from '@/components/AdvancedConfigEditor';
import { SelectInput } from '@/components/formInputs';
import { Loader2 } from 'lucide-react';
import { API_CAPTIONER, apiCaptionSettingsSchema, assertNoCaptionCredentials } from '../../cron/captionApiConfig';

export interface CaptionDatasetModalState {
  datasetPath: string;
  jobId?: string | null;
  cloneId?: string | null;
  defaultCaptionExt?: string;
  onClose?: () => void;
}

export const captionDatasetModalState = createGlobalState<CaptionDatasetModalState | null>(null);

export const openCaptionDatasetModal = (
  datasetPath: string,
  onClose?: () => void,
  options?: { jobId?: string | null; cloneId?: string | null; defaultCaptionExt?: string },
) => {
  captionDatasetModalState.set({
    datasetPath,
    onClose,
    jobId: options?.jobId ?? null,
    cloneId: options?.cloneId ?? null,
    defaultCaptionExt: options?.defaultCaptionExt,
  });
};

export const CaptionDatasetModal: React.FC = () => {
  const [modalInfo, setModalInfo] = captionDatasetModalState.use();
  const [jobConfig, setJobConfig] = useNestedState<CaptionJobConfig>(objectCopy(defaultCaptionJobConfig));
  const [gpuIDs, setGpuIDs] = useState<string | null>(null);
  const [existingJobName, setExistingJobName] = useState<string | null>(null);
  const [isConfigLoaded, setIsConfigLoaded] = useState(false);
  const [isLoadingConfig, setIsLoadingConfig] = useState(false);
  const { gpuList, isGPUInfoLoaded } = useGPUInfo();
  const [activeTab, setActiveTab] = useState<'simple' | 'advanced'>('simple');
  const open = modalInfo !== null;
  const isSavingRef = useRef(false);
  const savedJobRef = useRef<{ id: string; name: string } | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [clearApiKey, setClearApiKey] = useState(false);
  const [saveError, setSaveError] = useState<{ message: string; stack?: string } | null>(null);
  const isApi = jobConfig.config.process[0].type === API_CAPTIONER;
  const showGPUSelect = !isMac();
  const showLoadingOverlay = isLoadingConfig || isSaving;

  useEffect(() => {
    if (!modalInfo) return;
    let cancelled = false;
    setIsConfigLoaded(false);
    savedJobRef.current = null;
    setIsLoadingConfig(true);
    setJobConfig(objectCopy(defaultCaptionJobConfig));
    setGpuIDs(null);
    setActiveTab('simple');
    setExistingJobName(null);
    setApiKey('');
    setClearApiKey(false);
    setSaveError(null);
    const load = async () => {
      try {
        const existingId = modalInfo.jobId || modalInfo.cloneId;
        if (existingId) {
          const { data } = await apiClient.get(`/api/jobs?id=${existingId}`);
          if (cancelled) return;
          const config = JSON.parse(data.job_config);
          assertNoCaptionCredentials(config);
          if (modalInfo.cloneId) config.config.name = `${config.config.name}_copy`;
          else setExistingJobName(data.name);
          setGpuIDs(data.gpu_ids);
          setJobConfig(config);
        } else {
          const { data } = await apiClient.get('/api/caption/settings');
          if (cancelled) return;
          setJobConfig(restoreCaptionJobPreferences(data.preferences, modalInfo.datasetPath, modalInfo.defaultCaptionExt));
          setGpuIDs(data.preferences?.gpu_ids ?? null);
        }
        setIsConfigLoaded(true);
      } catch (error: any) {
        if (!cancelled) setSaveError({
          message: error.response?.data?.error || error.message || 'Failed to load caption setup.',
          stack: error.response?.data?.stack || error.stack,
        });
      } finally {
        if (!cancelled) setIsLoadingConfig(false);
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [modalInfo, setJobConfig]);

  useEffect(() => {
    if (isGPUInfoLoaded) {
      if ((gpuIDs === null || (!isApi && gpuIDs === 'cpu')) && gpuList.length > 0) {
        setGpuIDs(`${gpuList[0].index}`);
      }
    }
  }, [gpuList, isGPUInfoLoaded, gpuIDs, isApi]);

  const handleClose = () => {
    if (isSavingRef.current) return;
    if (modalInfo?.onClose) {
      modalInfo.onClose();
    }
    setIsConfigLoaded(false);
    savedJobRef.current = null;
    setModalInfo(null);
    setApiKey('');
    setClearApiKey(false);
    setSaveError(null);
  };

  const saveJob = async () => {
    if (isSavingRef.current || !isConfigLoaded) return;
    if (!modalInfo?.datasetPath) {
      alert('Dataset path is missing. Please try again.');
      return;
    }
    isSavingRef.current = true;
    setIsSaving(true);
    setSaveError(null);

    const isEdit = !!modalInfo.jobId;

    try {
      const snapshot = objectCopy(jobConfig);
      if (isApi) {
        assertNoCaptionCredentials(snapshot);
        const caption = apiCaptionSettingsSchema.parse(snapshot.config.process[0].caption);
        if (clearApiKey) delete caption.api_credential_id;
        else if (apiKey.trim()) {
          caption.api_credential_id = (
            await apiClient.post('/api/caption/credentials', { apiUrl: caption.api_url, apiKey })
          ).data.id;
        }
        snapshot.config.process[0].caption = { ...snapshot.config.process[0].caption, ...caption };
        if (clearApiKey) delete snapshot.config.process[0].caption.api_credential_id;
        snapshot.config.process[0].device = 'cpu';
      }
      const name = isEdit && existingJobName ? existingJobName : savedJobRef.current?.name ?? uuidv4();
      const response = await apiClient.post('/api/jobs', {
        id: isEdit ? modalInfo.jobId : savedJobRef.current?.id ?? null,
        name,
        gpu_ids: gpuIDs,
        job_config: snapshot,
        job_type: 'caption',
        job_ref: modalInfo.datasetPath,
      });
      const jobId = response.data.id;
      savedJobRef.current = { id: jobId, name };
      setJobConfig(snapshot);
      setApiKey('');
      setClearApiKey(false);
      await apiClient.post('/api/caption/settings', createCaptionJobPreferences(snapshot, gpuIDs));
      await startJob(jobId);
      // start the queue as well
      await startQueue(response.data.gpu_ids);
      isSavingRef.current = false;
      setIsSaving(false);
      handleClose();
    } catch (error: any) {
      const details = {
        message: error.response?.data?.error || error.message || 'Failed to save caption job.',
        stack: error.response?.data?.stack || error.stack,
      };
      setSaveError(details);
      // Axios errors include request bodies; never log the raw error/key form.
      console.error('Failed to save caption job:', details);
      isSavingRef.current = false;
      setIsSaving(false);
    }
  };

  const tabButtonClass = (tab: 'simple' | 'advanced') =>
    `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
      activeTab === tab
        ? 'border-blue-500 text-blue-400'
        : 'border-transparent text-gray-400 hover:text-gray-200 hover:border-gray-600'
    }`;

  return (
    <Modal isOpen={open} onClose={handleClose} title="Caption Dataset" size={activeTab === 'advanced' ? 'xl' : 'lg'}>
      <div className="relative space-y-4 text-gray-200">
        {showLoadingOverlay && (
          <div className="absolute -left-6 -right-6 -top-4 -bottom-4 z-10 flex items-center justify-center backdrop-blur-sm bg-gray-900/40">
            <Loader2 className="w-10 h-10 text-blue-400 animate-spin" />
          </div>
        )}
        <div className="flex items-center border-b border-gray-700 -mt-2">
          <button type="button" className={tabButtonClass('simple')} onClick={() => setActiveTab('simple')}>
            Simple
          </button>
          <button type="button" className={tabButtonClass('advanced')} onClick={() => setActiveTab('advanced')}>
            Advanced
          </button>
          <div className="flex-1" />
          {activeTab === 'advanced' && showGPUSelect && !isApi && (
            <div className="pb-2">
              <SelectInput
                value={`${gpuIDs}`}
                onChange={value => setGpuIDs(value)}
                options={gpuList.map((gpu: any) => ({ value: `${gpu.index}`, label: `GPU #${gpu.index}` }))}
              />
            </div>
          )}
        </div>
        <form
          noValidate={isApi}
          onSubmit={e => {
            e.preventDefault();
            saveJob();
          }}
        >
          {activeTab === 'simple' ? (
            <CaptionSimpleJob
              jobConfig={jobConfig}
              setJobConfig={setJobConfig}
              gpuIDs={gpuIDs}
              setGpuIDs={setGpuIDs}
              gpuList={gpuList}
              showGPUSelect={showGPUSelect}
              apiKey={apiKey}
              onApiKeyChange={setApiKey}
              clearApiKey={clearApiKey}
              onClearApiKeyChange={setClearApiKey}
            />
          ) : (
            <div className="h-[60vh] mt-2">
              <AdvancedConfigEditor config={jobConfig} setConfig={setJobConfig} />
            </div>
          )}
          {saveError && (
            <div role="alert" className="mt-4 rounded-md border border-red-700 bg-red-950 p-3 text-sm text-red-200">
              <p>{saveError.message}</p>
              <details className="mt-2">
                <summary>Error details</summary>
                <pre className="overflow-auto whitespace-pre-wrap text-xs">{saveError.stack}</pre>
              </details>
            </div>
          )}

          <p className="mt-4 text-xs text-gray-400">Settings are remembered when you add the job to the queue. Cancel does not save changes.</p>

          <div className="mt-6 flex justify-end space-x-3">
            <button
              type="button"
              className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
              onClick={handleClose}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!isConfigLoaded || isSaving}
              className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              Add to Queue
            </button>
          </div>
        </form>
      </div>
    </Modal>
  );
};
