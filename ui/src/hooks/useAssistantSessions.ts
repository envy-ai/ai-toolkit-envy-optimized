'use client';
import { useEffect, useRef, useSyncExternalStore } from 'react';
import { z } from 'zod';
import { apiClient } from '@/utils/api';
import { AssistantSessionController, type SessionApi, type SessionHost } from '@/assistant/AssistantSessionController';
import {
  assistantEntrySchema,
  assistantSessionSchema,
  emptySessionContent,
  sessionContentSchema,
  sessionIdSchema,
  sessionMetadataSchema,
} from '@/assistant/sessions';

const LAST = 'toolkit:assistant:lastSession';
const LEGACY = 'toolkit:assistant:messages';
const IMPORT = 'toolkit:assistant:legacyImportId';
const RECOVERY = 'toolkit:assistant:recovery';
export const browserSessionHost: SessionHost = {
  lastId: () => localStorage.getItem(LAST),
  remember: id => {
    id ? localStorage.setItem(LAST, id) : localStorage.removeItem(LAST);
  },
  legacy: () => {
    const raw = sessionStorage.getItem(LEGACY);
    return raw ? { ...emptySessionContent(), entries: z.array(assistantEntrySchema).parse(JSON.parse(raw)) } : null;
  },
  legacyId: () => sessionStorage.getItem(IMPORT),
  rememberLegacy: id => sessionStorage.setItem(IMPORT, id),
  completeLegacy: () => {
    sessionStorage.removeItem(LEGACY);
    sessionStorage.removeItem(IMPORT);
  },
  recovery: () => {
    const raw = sessionStorage.getItem(RECOVERY);
    return raw
      ? z
          .object({ id: sessionIdSchema, revision: z.number().int().nonnegative(), content: sessionContentSchema })
          .strict()
          .parse(JSON.parse(raw))
      : null;
  },
  recover: value => {
    value ? sessionStorage.setItem(RECOVERY, JSON.stringify(value)) : sessionStorage.removeItem(RECOVERY);
  },
};
export const assistantSessionApi: SessionApi = {
  list: async () =>
    z.array(sessionMetadataSchema).parse((await apiClient.get('/api/assistant/sessions')).data.sessions),
  get: async id => assistantSessionSchema.parse((await apiClient.get(`/api/assistant/sessions/${id}`)).data),
  create: async (id, content) =>
    assistantSessionSchema.parse((await apiClient.post('/api/assistant/sessions', { id, content })).data),
  update: async (id, update) =>
    assistantSessionSchema.parse((await apiClient.patch(`/api/assistant/sessions/${id}`, update)).data),
  delete: async (id, revision) => {
    await apiClient.delete(`/api/assistant/sessions/${id}`, { data: { revision } });
  },
};
export default function useAssistantSessions() {
  const reference = useRef<AssistantSessionController | null>(null);
  if (!reference.current) reference.current = new AssistantSessionController(assistantSessionApi, browserSessionHost);
  const controller = reference.current;
  const state = useSyncExternalStore(controller.subscribe, controller.snapshot, controller.snapshot);
  useEffect(() => {
    void controller.initialize();
  }, [controller]);
  useEffect(() => {
    if (!state.ready || state.loading || state.error || state.saveState !== 'unsaved') return;
    const timer = setTimeout(() => void controller.flush(), 500);
    return () => clearTimeout(timer);
  }, [controller, state]);
  return { ...state, controller, title: controller.title() };
}
