'use client';

import { useEffect, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import DatasetGrid, { DatasetGridSize } from '@/components/DatasetGrid';
import Loading from '@/components/Loading';

type DatasetView = 'list' | DatasetGridSize;

export default function Datasets() {
  const router = useRouter();
  const [view, setView] = useState<DatasetView>('list');
  const { datasets, firstImages, status, refreshDatasets } = useDatasetList(view !== 'list');
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);

  useEffect(() => {
    try {
      const saved = localStorage.getItem('datasets:view');
      if (saved === 'list' || saved === 'small' || saved === 'medium' || saved === 'large') setView(saved);
    } catch {
      /* The view still works when storage is unavailable. */
    }
  }, []);

  const changeView = (next: DatasetView) => {
    setView(next);
    try {
      localStorage.setItem('datasets:view', next);
    } catch {
      /* Storage is optional. */
    }
  };

  // Transform datasets array into rows with objects
  const tableRows = datasets.map(dataset => ({
    name: dataset,
    actions: dataset, // Pass full dataset name for actions
  }));

  const columns: TableColumn[] = [
    {
      title: 'Dataset Name',
      key: 'name',
      render: row => (
        <Link href={`/datasets/${encodeURIComponent(row.name)}`} className="text-gray-200 hover:text-gray-100">
          {row.name}
        </Link>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-20 text-right',
      render: row => (
        <button
          aria-label={`Delete dataset ${row.name}`}
          className="text-gray-200 hover:bg-red-600 p-2 rounded-full transition-colors"
          onClick={() => handleDeleteDataset(row.name)}
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  const handleDeleteDataset = (datasetName: string) => {
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: () => {
        apiClient
          .post('/api/datasets/delete', { name: datasetName })
          .then(() => {
            console.log('Dataset deleted:', datasetName);
            refreshDatasets();
          })
          .catch(error => {
            console.error('Error deleting dataset:', error);
          });
      },
    });
  };

  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const data = await apiClient.post('/api/datasets/create', { name: newDatasetName }).then(res => res.data);
      console.log('New dataset created:', data);
      refreshDatasets();
      setNewDatasetName('');
      setIsNewDatasetModalOpen(false);
    } catch (error) {
      console.error('Error creating new dataset:', error);
    }
  };

  const openNewDatasetModal = () => {
    openConfirm({
      title: 'New Dataset',
      message: 'Enter the name of the new dataset:',
      type: 'info',
      confirmText: 'Create',
      inputTitle: 'Dataset Name',
      onConfirm: async (name?: string) => {
        if (!name) {
          console.error('Dataset name is required.');
          return;
        }
        try {
          const data = await apiClient.post('/api/datasets/create', { name }).then(res => res.data);
          console.log('New dataset created:', data);
          if (data.name) {
            router.push(`/datasets/${data.name}`);
          } else {
            refreshDatasets();
          }
        } catch (error) {
          console.error('Error creating new dataset:', error);
        }
      },
    });
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-base sm:text-lg">Datasets</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-white bg-slate-600 px-2 sm:px-3 py-1 rounded-md hover:bg-slate-500 transition-colors text-sm sm:text-base whitespace-nowrap"
            onClick={() => openNewDatasetModal()}
          >
            <span className="sm:hidden">+ New</span>
            <span className="hidden sm:inline">New Dataset</span>
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <div className="flex items-center justify-between gap-3 mb-4">
          <label className="flex items-center gap-2 text-sm text-gray-300">
            View
            <select
              value={view}
              onChange={event => changeView(event.target.value as DatasetView)}
              className="rounded-md bg-gray-800 border border-gray-600 text-gray-200 px-3 py-2"
            >
              <option value="list">List</option>
              <option value="small">Small grid</option>
              <option value="medium">Medium grid</option>
              <option value="large">Large grid</option>
            </select>
          </label>
          <button
            type="button"
            onClick={refreshDatasets}
            disabled={status === 'loading'}
            className="rounded-md bg-gray-800 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700 disabled:opacity-50"
          >
            Refresh
          </button>
        </div>
        {status === 'error' && (
          <p role="alert" className="mb-4 text-red-300">
            Could not load datasets. Try refreshing.
          </p>
        )}
        {view === 'list' ? (
          <UniversalTable
            columns={columns}
            rows={tableRows}
            isLoading={status === 'loading'}
            onRefresh={refreshDatasets}
          />
        ) : status === 'loading' ? (
          <div className="p-4 flex justify-center">
            <Loading />
          </div>
        ) : datasets.length === 0 ? (
          <p className="p-6 text-center text-sm text-gray-400">No datasets yet. Create a new dataset to get started.</p>
        ) : (
          <DatasetGrid datasets={datasets} firstImages={firstImages} size={view} onDelete={handleDeleteDataset} />
        )}
      </MainContent>

      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="New Dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="text-sm text-gray-400">
              This will create a new folder with the name below in your dataset folder.
            </div>
            <div className="mt-4">
              <TextInput label="Dataset Name" value={newDatasetName} onChange={value => setNewDatasetName(value)} />
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-700 px-4 py-2 text-gray-200 hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-gray-500"
                onClick={() => setIsNewDatasetModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                Confirm
              </button>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}
