import path from 'path';
import { NodeAssistantConfiguration } from './NodeAssistantConfiguration';
import { NodeAssistantProvider } from './NodeAssistantProvider';
import { assistantConfigurationSchema, DEFAULT_ASSISTANT_CONFIGURATION } from '@/assistant/AssistantProvider';
import { TOOLKIT_ROOT } from '@/paths';

const shared = globalThis as typeof globalThis & { toolkitAssistant?: Promise<NodeAssistantProvider> };

export function getAssistantService() {
  shared.toolkitAssistant ??= (async () => {
    // Deliberately separate from general settings, dataset roots, and browser storage.
    const configuration = await NodeAssistantConfiguration.create(
      process.env.AI_TOOLKIT_ASSISTANT_SETTINGS_DIR || path.join(TOOLKIT_ROOT, 'data', 'assistant'),
    );
    if (process.env.AI_TOOLKIT_ASSISTANT_BASE_URL) {
      await configuration.update({
        config: assistantConfigurationSchema.parse({
          ...DEFAULT_ASSISTANT_CONFIGURATION,
          baseUrl: process.env.AI_TOOLKIT_ASSISTANT_BASE_URL,
          authMode: process.env.AI_TOOLKIT_ASSISTANT_AUTH_MODE || 'none',
          model: process.env.AI_TOOLKIT_ASSISTANT_MODEL || '',
          reasoningEffort: process.env.AI_TOOLKIT_ASSISTANT_REASONING_EFFORT || undefined,
          vision: process.env.AI_TOOLKIT_ASSISTANT_VISION === 'true',
        }),
      });
    }
    return new NodeAssistantProvider({ configuration });
  })().catch(error => {
    shared.toolkitAssistant = undefined;
    throw error;
  });
  return shared.toolkitAssistant;
}
