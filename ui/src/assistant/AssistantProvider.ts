import { z } from 'zod';

export const assistantConfigurationSchema = z
  .object({
    baseUrl: z
      .string()
      .url()
      .refine(value => {
        const url = new URL(value);
        return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password && !url.hash && !url.search;
      }, 'API base URL must use HTTP/HTTPS without credentials, a query, or a fragment.'),
    protocol: z.enum(['responses', 'chat-completions']),
    model: z.string().trim().max(200),
    authMode: z.enum(['api-key', 'none']),
    reasoningEffort: z.enum(['none', 'minimal', 'low', 'medium', 'high', 'xhigh']).optional(),
    strictTools: z.boolean(),
    stream: z.boolean(),
    vision: z.boolean().default(false),
    timeoutMs: z.number().int().min(1000).max(600_000),
    maxOutputTokens: z.number().int().min(64).max(131_072),
    maxRunTokens: z.number().int().min(128).max(2_000_000),
    maxToolRounds: z.number().int().min(1).max(100),
    finalResponseReserve: z.number().int().min(64).max(131_072),
    maxToolCallsPerRound: z.number().int().min(1).max(100),
    maxBatchSize: z.number().int().min(1).max(500),
    maxJobs: z.number().int().min(1).max(20),
  })
  .strict()
  .superRefine((value, context) => {
    if (
      value.finalResponseReserve > value.maxOutputTokens ||
      value.maxOutputTokens + value.finalResponseReserve >= value.maxRunTokens
    ) {
      context.addIssue({
        code: 'custom',
        message: 'Run budget must exceed output limit plus final response reserve; reserve must fit the output limit.',
      });
    }
  });
export type AssistantConfiguration = z.infer<typeof assistantConfigurationSchema>;
/** Unconfigured model is intentionally empty; settings must select an exact provider model ID. */
export const DEFAULT_ASSISTANT_CONFIGURATION: AssistantConfiguration = {
  baseUrl: 'https://api.openai.com/v1',
  protocol: 'responses',
  model: '',
  authMode: 'api-key',
  strictTools: true,
  stream: true,
  vision: false,
  timeoutMs: 120_000,
  maxOutputTokens: 4096,
  maxRunTokens: 64_000,
  maxToolRounds: 12,
  finalResponseReserve: 1024,
  maxToolCallsPerRound: 20,
  maxBatchSize: 100,
  maxJobs: 4,
};
export type AssistantConfigurationState = {
  config: AssistantConfiguration;
  revision: number;
  credentialPresent: boolean;
  credentialPersisted: boolean;
  /** Opaque host-owned credential identifier; never the API key. */
  credentialId: string | null;
};
export type AssistantConfigurationUpdate = {
  config: AssistantConfiguration;
  apiKey?: string;
  persistCredential?: boolean;
  clearCredential?: boolean;
};
export type AssistantToolDefinition = {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /** false opts out for schemas with arbitrary arguments/optional fields; global false disables all strictness. */
  strict?: boolean;
};
export type AssistantImageInput = {
  artifactId: string;
  source:
    | 'runtime-playfield'
    | 'editor-map'
    | 'asset-preview'
    | 'animation-preview'
    | 'editor-window'
    | 'visible-canvas'
    | 'visible-image';
  mimeType: 'image/png' | 'image/jpeg' | 'image/webp';
  width: number;
  height: number;
  /** Separate image transport field, never serialized into text/tool results or logs. */
  dataBase64: string;
};
export type AssistantToolAttachment = { callId: string; image: AssistantImageInput };
export type AssistantMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
  images?: AssistantImageInput[];
};
export type AssistantToolCall = { callId: string; name: string; arguments: unknown };
export type AssistantToolResult = { callId: string; output: string };
export type AssistantTurnRequest = {
  runId: string;
  requestId: string;
  configurationRevision: number;
  /** First turn only; subsequent turns provide correlated toolResults. */
  messages?: AssistantMessage[];
  tools?: AssistantToolDefinition[];
  toolResults?: AssistantToolResult[];
  /** Follow-up image observations correlated to this turn's tool result IDs. */
  attachments?: AssistantToolAttachment[];
};
export type AssistantUsage = { inputTokens: number; outputTokens: number; totalTokens: number; estimated: boolean };
export type AssistantTurnResult = {
  text: string;
  toolCalls: AssistantToolCall[];
  usage: AssistantUsage;
  round: number;
  remainingTokens: number;
  /** Effective wire mode; generic tools may explicitly opt out of provider strict schemas. */
  toolSchemaModes?: Array<{ name: string; strict: boolean }>;
};
/** Node-owned transport. Tools execute only through the renderer's shared command executor. */
export type AssistantProviderService = {
  getConfiguration(): Promise<AssistantConfigurationState>;
  setConfiguration(update: AssistantConfigurationUpdate): Promise<AssistantConfigurationState>;
  createTurn(request: AssistantTurnRequest): Promise<AssistantTurnResult>;
  cancelRun(runId: string): Promise<void>;
  endRun(runId: string): Promise<void>;
  /** Explicit settings connection test; contacts only configured baseUrl + /models. */
  discoverModels(): Promise<string[]>;
};

export function assistantEndpoint(
  config: AssistantConfiguration,
  route: 'responses' | 'chat/completions' | 'models',
): URL {
  assistantConfigurationSchema.parse(config);
  const url = new URL(config.baseUrl);
  url.pathname = `${url.pathname.replace(/\/+$/, '')}/${route}`;
  return url;
}
