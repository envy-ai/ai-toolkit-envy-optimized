import { apiClient } from '@/utils/api';

export const startQueue = (queueID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/queue/${queueID}/start`)
      .then(res => res.data)
      .then(data => {
        console.log('Queue started:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error starting queue:', error);
        reject(error);
      });
  });
};
export const stopQueue = (queueID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .get(`/api/queue/${queueID}/stop`)
      .then(res => res.data)
      .then(data => {
        console.log('Queue stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping queue:', error);
        reject(error);
      });
  });
};

export type QueueJobPosition = { id: string; queue_position: number };

/** The IDs may be a visible subset; the server preserves omitted queue slots. */
export const reorderQueueJobs = (queueID: string, orderedJobIds: string[]) => {
  return new Promise<QueueJobPosition[]>((resolve, reject) => {
    apiClient
      .patch(`/api/queue/${queueID}/reorder`, { orderedJobIds })
      .then(res => res.data)
      .then(data => {
        if (
          data.success !== true ||
          !Array.isArray(data.positions) ||
          data.positions.some(
            (position: any) => typeof position.id !== 'string' || !Number.isSafeInteger(position.queue_position),
          )
        )
          throw new Error('Queue reorder response did not contain valid positions.');
        resolve(data.positions);
      })
      .catch(error => {
        console.error('Error reordering queue:', error);
        reject(error);
      });
  });
};
