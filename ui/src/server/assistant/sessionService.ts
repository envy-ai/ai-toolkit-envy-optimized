import path from 'node:path';
import { TOOLKIT_ROOT } from '@/paths';
import { AssistantSessionStore } from './AssistantSessionStore';

const shared = globalThis as typeof globalThis & { toolkitAssistantSessions?: Promise<AssistantSessionStore> };
export function getAssistantSessionStore() {
  shared.toolkitAssistantSessions ??= AssistantSessionStore.create(
    path.join(
      process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR || path.join(TOOLKIT_ROOT, 'data', 'assistant'),
      'sessions',
    ),
  ).catch(error => {
    shared.toolkitAssistantSessions = undefined;
    throw error;
  });
  return shared.toolkitAssistantSessions;
}
