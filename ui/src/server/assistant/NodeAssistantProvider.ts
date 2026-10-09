import { inspectImageBytes as inspectProjectMediaBytes } from './imageBytes';
import { request as httpRequest } from 'node:http';
import { request as httpsRequest } from 'node:https';
import { z } from 'zod';
import {
  assistantEndpoint,
  type AssistantConfiguration,
  type AssistantConfigurationUpdate,
  type AssistantImageInput,
  type AssistantMessage,
  type AssistantProviderService,
  type AssistantToolCall,
  type AssistantToolDefinition,
  type AssistantTurnRequest,
  type AssistantTurnResult,
  type AssistantUsage,
} from '@/assistant/AssistantProvider';
import { NodeAssistantConfiguration } from './NodeAssistantConfiguration';

const MAX_RESPONSE_BYTES = 16 * 1024 * 1024;
const MAX_CONTEXT_CHARACTERS = 512_000;
const MAX_ACTIVE_RUNS = 20;
const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const MAX_RUN_IMAGES = 4;
const ESTIMATED_IMAGE_TOKENS = 2048;
const RUN_IDLE_MS = 15 * 60_000;
type RecordValue = Record<string, unknown>;
export type AssistantHttpResponse = { status: number; contentType: string; body: AsyncIterable<Uint8Array> };
export type AssistantHttpTransport = (
  url: URL,
  options: {
    method: 'GET' | 'POST';
    headers: Record<string, string>;
    body?: string;
    signal: AbortSignal;
  },
) => Promise<AssistantHttpResponse>;

export const nodeAssistantHttpTransport: AssistantHttpTransport = (url, options) =>
  new Promise((resolve, reject) => {
    // Node verifies TLS using its trust store. No redirects or insecure TLS override.
    const request = (url.protocol === 'https:' ? httpsRequest : httpRequest)(
      url,
      {
        method: options.method,
        headers: options.headers,
        signal: options.signal,
      },
      response =>
        resolve({
          status: response.statusCode ?? 0,
          contentType: response.headers['content-type'] ?? '',
          body: response,
        }),
    );
    request.once('error', reject);
    request.end(options.body);
  });

function record(value: unknown, label: string): RecordValue {
  if (value === null || typeof value !== 'object' || Array.isArray(value))
    throw new Error(`${label} must be an object.`);
  return value as RecordValue;
}
function string(value: unknown, label: string): string {
  if (typeof value !== 'string') throw new Error(`${label} must be a string.`);
  return value;
}
function nonempty(value: unknown, label: string): string {
  const text = string(value, label);
  if (!text) throw new Error(`${label} must not be empty.`);
  return text;
}
function array(value: unknown, label: string): unknown[] {
  if (!Array.isArray(value)) throw new Error(`${label} must be an array.`);
  return value;
}
function usage(
  value: unknown,
  protocol: AssistantConfiguration['protocol'],
  request: string,
  output: unknown,
): AssistantUsage {
  if (value === undefined || value === null) {
    const inputTokens = Math.ceil(budgetText(request).length / 4);
    const outputTokens = Math.ceil(JSON.stringify(output).length / 4);
    return { inputTokens, outputTokens, totalTokens: inputTokens + outputTokens, estimated: true };
  }
  const raw = record(value, 'Provider usage');
  const inputTokens = raw[protocol === 'responses' ? 'input_tokens' : 'prompt_tokens'];
  const outputTokens = raw[protocol === 'responses' ? 'output_tokens' : 'completion_tokens'];
  const totalTokens = raw.total_tokens;
  for (const number of [inputTokens, outputTokens, totalTokens]) {
    if (!Number.isSafeInteger(number) || (number as number) < 0)
      throw new Error('Provider returned invalid token usage.');
  }
  if ((totalTokens as number) < (inputTokens as number) + (outputTokens as number))
    throw new Error('Provider total token usage is inconsistent.');
  return {
    inputTokens: inputTokens as number,
    outputTokens: outputTokens as number,
    totalTokens: totalTokens as number,
    estimated: false,
  };
}
export type ParsedAssistantResponse = {
  text: string;
  toolCalls: AssistantToolCall[];
  continuation: RecordValue[];
  rawUsage: unknown;
};
function parseCall(callId: unknown, name: unknown, argumentsText: unknown): AssistantToolCall {
  const argumentsJson = string(argumentsText, 'Function arguments');
  let argumentsValue: unknown;
  try {
    argumentsValue = JSON.parse(argumentsJson) as unknown;
  } catch (error) {
    throw new Error('Provider returned malformed function argument JSON.', { cause: error });
  }
  return {
    callId: id.parse(nonempty(callId, 'Function call ID')),
    name: nonempty(name, 'Function name'),
    arguments: argumentsValue,
  };
}
export function parseAssistantResponse(
  value: unknown,
  protocol: AssistantConfiguration['protocol'],
): ParsedAssistantResponse {
  const response = record(value, 'Provider response');
  if (response.error) throw new Error(`Provider error: ${JSON.stringify(response.error)}`);
  if (protocol === 'responses') {
    if (response.status !== 'completed')
      throw new Error(
        `Provider response did not complete: ${String(response.status)} ${JSON.stringify(response.incomplete_details ?? {})}`,
      );
    const continuation = array(response.output, 'Response output').map(item => record(item, 'Response output item'));
    const toolCalls: AssistantToolCall[] = [];
    let text = '';
    for (const item of continuation) {
      if (item.type === 'function_call') toolCalls.push(parseCall(item.call_id, item.name, item.arguments));
      else if (item.type === 'message') {
        for (const raw of array(item.content, 'Response message content')) {
          const content = record(raw, 'Response content');
          if (content.type === 'refusal') throw new Error(`Provider refusal: ${string(content.refusal, 'Refusal')}`);
          if (content.type !== 'output_text') throw new Error(`Unsupported response content: ${String(content.type)}.`);
          text += string(content.text, 'Response text');
        }
      } else if (item.type !== 'reasoning') throw new Error(`Unsupported provider output item: ${String(item.type)}.`);
    }
    if (!text && !toolCalls.length) throw new Error('Provider completed without text or function calls.');
    return { text, toolCalls, continuation, rawUsage: response.usage };
  }
  const choices = array(response.choices, 'Chat choices');
  if (choices.length !== 1) throw new Error('Chat provider must return exactly one choice.');
  const choice = record(choices[0], 'Chat choice');
  if (!['stop', 'tool_calls'].includes(String(choice.finish_reason)))
    throw new Error(`Chat response did not complete: ${String(choice.finish_reason)}.`);
  const message = record(choice.message, 'Chat message');
  if (message.refusal) throw new Error(`Provider refusal: ${String(message.refusal)}`);
  if (message.role !== 'assistant') throw new Error('Chat provider returned a non-assistant message.');
  const text = message.content == null ? '' : string(message.content, 'Chat text');
  const toolCalls = (message.tool_calls === undefined ? [] : array(message.tool_calls, 'Chat tool calls')).map(raw => {
    const call = record(raw, 'Chat tool call');
    if (call.type !== 'function') throw new Error('Only function tool calls are supported.');
    const fn = record(call.function, 'Chat function');
    return parseCall(call.id, fn.name, fn.arguments);
  });
  if (!text && !toolCalls.length) throw new Error('Chat provider completed without text or function calls.');
  if ((choice.finish_reason === 'tool_calls') !== toolCalls.length > 0)
    throw new Error('Chat finish reason does not match function calls.');
  return { text, toolCalls, continuation: [message], rawUsage: response.usage };
}

/** Frame SSE across arbitrary byte boundaries, including UTF-8 and CRLF boundaries. */
export async function readAssistantSse(
  body: AsyncIterable<Uint8Array>,
  protocol: AssistantConfiguration['protocol'],
): Promise<unknown> {
  const decoder = new TextDecoder('utf-8', { fatal: true });
  let buffer = '',
    received = 0,
    completed: unknown;
  let done = false;
  const chatMessage: RecordValue = { role: 'assistant', content: '', tool_calls: [] };
  const chatCalls = new Map<number, { id: string; type: string; function: { name: string; arguments: string } }>();
  let finishReason: unknown = null,
    chatUsage: unknown;
  const responseArguments = new Map<number, string>();
  function frame(frameText: string): void {
    const data = frameText
      .split(/\r?\n/)
      .filter(line => line.startsWith('data:'))
      .map(line => line.slice(5).replace(/^ /, ''))
      .join('\n');
    if (!data) return;
    if (data === '[DONE]') {
      done = true;
      return;
    }
    if (done) throw new Error('Provider sent data after stream completion.');
    const event = record(JSON.parse(data) as unknown, 'Stream event');
    if (
      event.error ||
      event.type === 'error' ||
      event.type === 'response.failed' ||
      event.type === 'response.incomplete'
    ) {
      throw new Error(
        `Provider stream failure: ${String(event.type ?? 'error')}: ${JSON.stringify(event.error ?? record(event.response ?? {}, 'Failed response').error ?? {})}`,
      );
    }
    if (protocol === 'responses') {
      if (event.type === 'response.function_call_arguments.delta') {
        if (!Number.isSafeInteger(event.output_index)) throw new Error('Stream function output index is invalid.');
        const index = event.output_index as number;
        responseArguments.set(
          index,
          (responseArguments.get(index) ?? '') + string(event.delta, 'Function argument delta'),
        );
      }
      if (event.type === 'response.completed') {
        if (completed !== undefined) throw new Error('Duplicate response completion event.');
        completed = event.response;
      }
    } else {
      if (event.usage != null) chatUsage = event.usage;
      for (const rawChoice of array(event.choices, 'Chat stream choices')) {
        const choice = record(rawChoice, 'Chat stream choice');
        if (choice.index !== 0) throw new Error('Multiple streamed chat choices are unsupported.');
        if (choice.finish_reason != null) {
          if (finishReason !== null) throw new Error('Duplicate chat completion event.');
          finishReason = choice.finish_reason;
        }
        const delta = record(choice.delta, 'Chat stream delta');
        if (delta.refusal) throw new Error(`Provider refusal: ${String(delta.refusal)}`);
        if (delta.content != null)
          chatMessage.content = String(chatMessage.content) + string(delta.content, 'Chat text delta');
        for (const rawCall of delta.tool_calls === undefined ? [] : array(delta.tool_calls, 'Chat call deltas')) {
          const call = record(rawCall, 'Chat call delta');
          if (!Number.isSafeInteger(call.index) || (call.index as number) < 0)
            throw new Error('Chat function index is invalid.');
          const index = call.index as number;
          const prior = chatCalls.get(index) ?? { id: '', type: 'function', function: { name: '', arguments: '' } };
          if (call.id != null) prior.id += string(call.id, 'Chat call ID delta');
          if (call.type != null && call.type !== 'function') throw new Error('Only function tools are supported.');
          if (call.function != null) {
            const fn = record(call.function, 'Chat function delta');
            if (fn.name != null) prior.function.name += string(fn.name, 'Function name delta');
            if (fn.arguments != null) prior.function.arguments += string(fn.arguments, 'Function argument delta');
          }
          chatCalls.set(index, prior);
        }
      }
    }
  }
  for await (const bytes of body) {
    received += bytes.byteLength;
    if (received > MAX_RESPONSE_BYTES) throw new Error('Assistant stream exceeds the response byte limit.');
    buffer += decoder.decode(bytes, { stream: true });
    let match: RegExpExecArray | null;
    while ((match = /\r?\n\r?\n/.exec(buffer))) {
      frame(buffer.slice(0, match.index));
      buffer = buffer.slice(match.index + match[0].length);
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) frame(buffer);
  if (protocol === 'responses') {
    if (completed === undefined) throw new Error('Responses stream ended without response.completed.');
    const result = record(completed, 'Completed response');
    const output = array(result.output, 'Completed output');
    for (const [index, argumentsText] of responseArguments) {
      if (record(output[index], 'Completed function call').arguments !== argumentsText)
        throw new Error('Streamed function arguments differ from completed arguments.');
    }
    return completed;
  }
  if (!done || finishReason === null) throw new Error('Chat stream ended without a completed choice and [DONE].');
  chatMessage.tool_calls = [...chatCalls].sort(([a], [b]) => a - b).map(([, call]) => call);
  return { choices: [{ message: chatMessage, finish_reason: finishReason }], usage: chatUsage };
}

const id = z
  .string()
  .min(1)
  .max(200)
  .regex(/^[A-Za-z0-9_.:-]+$/);
const toolSchema = z
  .object({
    name: z
      .string()
      .min(1)
      .max(64)
      .regex(/^[A-Za-z0-9_-]+$/),
    description: z.string().max(4000),
    parameters: z.record(z.string(), z.unknown()),
    strict: z.boolean().optional(),
  })
  .strict();
const imageSchema = z
  .object({
    artifactId: id,
    source: z.enum([
      'runtime-playfield',
      'editor-map',
      'asset-preview',
      'animation-preview',
      'editor-window',
      'visible-canvas',
      'visible-image',
    ]),
    mimeType: z.enum(['image/png', 'image/jpeg', 'image/webp']),
    width: z.number().int().positive().max(4096),
    height: z.number().int().positive().max(4096),
    dataBase64: z
      .string()
      .min(4)
      .max(Math.ceil(MAX_IMAGE_BYTES / 3) * 4)
      .regex(/^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$/),
  })
  .strict();
const turnSchema = z
  .object({
    runId: id,
    requestId: id,
    configurationRevision: z.number().int().positive(),
    messages: z
      .array(
        z
          .object({
            role: z.enum(['system', 'user', 'assistant']),
            content: z.string().max(MAX_CONTEXT_CHARACTERS),
            images: z.array(imageSchema).max(MAX_RUN_IMAGES).optional(),
          })
          .strict(),
      )
      .min(1)
      .max(100)
      .optional(),
    tools: z.array(toolSchema).max(100).optional(),
    toolResults: z
      .array(z.object({ callId: id, output: z.string().max(MAX_CONTEXT_CHARACTERS) }).strict())
      .max(100)
      .optional(),
    attachments: z
      .array(z.object({ callId: id, image: imageSchema }).strict())
      .max(MAX_RUN_IMAGES)
      .optional(),
  })
  .strict();
/** Encoded images are native content parts, never part of a text/token estimate. */
function budgetText(value: string | RecordValue[]): string {
  const parsed = typeof value === 'string' ? (JSON.parse(value) as unknown) : value;
  return JSON.stringify(parsed, (key, item: unknown) => (key === 'image_url' ? '[image]' : item));
}
function validateImage(image: AssistantImageInput): number {
  const bytes = Buffer.from(image.dataBase64, 'base64');
  if (bytes.byteLength > MAX_IMAGE_BYTES || bytes.toString('base64') !== image.dataBase64)
    throw new Error('Assistant image encoding or byte size is invalid.');
  const extension = image.mimeType === 'image/jpeg' ? 'jpg' : image.mimeType.split('/')[1];
  const dimensions = inspectProjectMediaBytes(bytes, image.mimeType, `observation.${extension}`);
  if (dimensions.width !== image.width || dimensions.height !== image.height || image.width * image.height > 16_777_216)
    throw new Error('Assistant image dimensions do not match the encoded file or exceed the pixel limit.');
  return bytes.byteLength;
}
function imageContent(image: AssistantImageInput, protocol: AssistantConfiguration['protocol']): RecordValue {
  const url = `data:${image.mimeType};base64,${image.dataBase64}`;
  return protocol === 'responses'
    ? { type: 'input_image', image_url: url, detail: 'auto' }
    : { type: 'image_url', image_url: { url, detail: 'auto' } };
}
function initialMessage(message: AssistantMessage, protocol: AssistantConfiguration['protocol']): RecordValue {
  if (!message.images?.length) return { role: message.role, content: message.content };
  if (message.role !== 'user') throw new Error('Image observations require a user message.');
  return {
    role: message.role,
    content: [
      { type: protocol === 'responses' ? 'input_text' : 'text', text: message.content },
      ...message.images.map(image => imageContent(image, protocol)),
    ],
  };
}

type Run = {
  input: RecordValue[];
  tools: AssistantToolDefinition[];
  pending: AssistantToolCall[];
  imageCount: number;
  imageBytes: number;
  seenCalls: Set<string>;
  round: number;
  spentTokens: number;
  lastUsed: number;
  controller?: AbortController;
};

/** Redact credentials while preserving each original error's message, stack and cause chain. */
function safeError(error: unknown, key: string | undefined, seen = new Set<unknown>()): Error {
  if (seen.has(error)) return new Error('Cyclic error cause omitted.');
  seen.add(error);
  const original = error instanceof Error ? error : new Error(String(error));
  const redact = (value: string) => {
    let output = value.replace(/data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+/g, '[IMAGE DATA REDACTED]');
    if (key) output = output.split(key).join('[REDACTED]').split(JSON.stringify(key).slice(1, -1)).join('[REDACTED]');
    return output;
  };
  const cause = original.cause === undefined ? undefined : safeError(original.cause, key, seen);
  const sanitized = new Error(redact(original.message), { cause });
  sanitized.name = original.name;
  sanitized.stack = original.stack === undefined ? undefined : redact(original.stack);
  return sanitized;
}
function strictToolMode(config: AssistantConfiguration, tool: AssistantToolDefinition): boolean {
  const strict = config.strictTools && tool.strict !== false;
  if (!strict) return false;
  const visit = (value: unknown): void => {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return;
    const schema = value as RecordValue;
    if (schema.type === 'object') {
      const properties = record(schema.properties ?? {}, 'Strict schema properties');
      if (
        schema.additionalProperties !== false ||
        !Array.isArray(schema.required) ||
        Object.keys(properties).some(property => !(schema.required as unknown[]).includes(property))
      ) {
        throw new Error(
          `Tool ${tool.name} does not have a strict-compatible object schema; declare strict:false and retain local validation.`,
        );
      }
    }
    for (const item of Object.values(schema)) {
      if (Array.isArray(item)) item.forEach(visit);
      else if (item && typeof item === 'object') visit(item);
    }
  };
  if (tool.parameters.type !== 'object')
    throw new Error(`Tool ${tool.name} requires an object root for provider strict mode.`);
  visit(tool.parameters);
  return true;
}

export class NodeAssistantProvider implements AssistantProviderService {
  readonly #configuration: NodeAssistantConfiguration;
  readonly #transport: AssistantHttpTransport;
  readonly #runs = new Map<string, Run>();
  readonly #retired = new Set<string>();
  readonly #requests = new Set<string>();
  readonly #discovery = new Set<AbortController>();
  #configurationChanges = 0;
  constructor(options: { configuration?: NodeAssistantConfiguration; transport?: AssistantHttpTransport } = {}) {
    this.#configuration = options.configuration ?? new NodeAssistantConfiguration();
    this.#transport = options.transport ?? nodeAssistantHttpTransport;
  }
  async getConfiguration() {
    return this.#configuration.state();
  }
  async setConfiguration(update: AssistantConfigurationUpdate) {
    // Invalidate before any asynchronous settings writes can race a completion.
    this.#configurationChanges += 1;
    try {
      for (const runId of [...this.#runs.keys()]) await this.cancelRun(runId);
      for (const controller of this.#discovery) controller.abort(new Error('Assistant configuration changed.'));
      return await this.#configuration.update(update);
    } finally {
      this.#configurationChanges -= 1;
    }
  }
  async cancelRun(runId: string) {
    id.parse(runId);
    const run = this.#runs.get(runId);
    run?.controller?.abort(new Error('Assistant run cancelled.'));
    this.#runs.delete(runId);
    this.#retired.add(runId);
    if (this.#retired.size > 4096) this.#retired.delete(this.#retired.values().next().value!);
  }
  async endRun(runId: string) {
    await this.cancelRun(runId);
  }
  async dispose() {
    for (const runId of [...this.#runs.keys()]) await this.cancelRun(runId);
    for (const controller of this.#discovery) controller.abort(new Error('Assistant host stopped.'));
  }
  #headers(): Record<string, string> {
    const { config } = this.#configuration.state();
    const headers: Record<string, string> = {
      Accept: config.stream ? 'text/event-stream, application/json' : 'application/json',
    };
    if (config.authMode === 'api-key') {
      const key = this.#configuration.key();
      if (!key)
        throw new Error('Assistant API key is missing. Configure a key or explicitly select no authentication.');
      headers.Authorization = `Bearer ${key}`;
    }
    return headers;
  }
  async #request(
    route: 'responses' | 'chat/completions' | 'models',
    controller: AbortController,
    body?: string,
  ): Promise<unknown> {
    const { config } = this.#configuration.state();
    const timer = setTimeout(
      () => controller.abort(new Error(`Assistant request timed out after ${config.timeoutMs} ms.`)),
      config.timeoutMs,
    );
    try {
      const response = await this.#transport(assistantEndpoint(config, route), {
        method: body === undefined ? 'GET' : 'POST',
        signal: controller.signal,
        headers: { ...this.#headers(), ...(body === undefined ? {} : { 'Content-Type': 'application/json' }) },
        body,
      });
      if (response.status >= 200 && response.status < 300 && response.contentType.includes('text/event-stream')) {
        return await readAssistantSse(response.body, config.protocol);
      }
      const chunks: Uint8Array[] = [];
      let length = 0;
      for await (const bytes of response.body) {
        length += bytes.byteLength;
        if (length > MAX_RESPONSE_BYTES) throw new Error('Assistant response exceeds the byte limit.');
        chunks.push(bytes);
      }
      const raw = Buffer.concat(chunks, length).toString('utf8');
      if (response.status < 200 || response.status >= 300)
        throw new Error(
          `Assistant provider HTTP ${response.status}: ${raw.replace(/data:image\/(?:png|jpeg|webp);base64,[A-Za-z0-9+/=]+/g, '[IMAGE DATA REDACTED]')}`,
        );
      return JSON.parse(raw) as unknown;
    } catch (error) {
      // A transport may convert the abort reason to a generic AbortError.
      if (controller.signal.aborted)
        throw new Error(
          controller.signal.reason instanceof Error ? controller.signal.reason.message : 'Assistant request aborted.',
          { cause: error },
        );
      throw error;
    } finally {
      clearTimeout(timer);
    }
  }
  async discoverModels(): Promise<string[]> {
    if (this.#configurationChanges) throw new Error('Assistant configuration update is in progress.');
    const controller = new AbortController();
    this.#discovery.add(controller);
    const key = this.#configuration.key();
    try {
      const response = record(await this.#request('models', controller), 'Models response');
      const serialized = JSON.stringify(response);
      if (key && (serialized.includes(key) || serialized.includes(JSON.stringify(key).slice(1, -1))))
        throw new Error('Provider model discovery contained the API credential and was withheld.');
      return array(response.data, 'Model list').map(model => nonempty(record(model, 'Model').id, 'Model ID'));
    } catch (error) {
      throw safeError(error, key);
    } finally {
      this.#discovery.delete(controller);
    }
  }
  async createTurn(value: AssistantTurnRequest): Promise<AssistantTurnResult> {
    const request = turnSchema.parse(value);
    const state = this.#configuration.state();
    const key = this.#configuration.key();
    let run: Run | undefined;
    let ownsRun = false,
      registeredRequest = false;
    try {
      if (this.#configurationChanges) throw new Error('Assistant configuration update is in progress.');
      if (!state.config.model) throw new Error('Select an exact provider model ID before starting an assistant run.');
      if (request.configurationRevision !== state.revision)
        throw new Error('Assistant configuration changed; start a fresh run.');
      for (const [runId, prior] of this.#runs) {
        if (!prior.controller && Date.now() - prior.lastUsed > RUN_IDLE_MS) await this.cancelRun(runId);
      }
      const images = [
        ...(request.messages?.flatMap(message => message.images ?? []) ?? []),
        ...(request.attachments?.map(attachment => attachment.image) ?? []),
      ];
      if (images.length && !state.config.vision)
        throw new Error('Image input is disabled; enable vision explicitly before sending captured observations.');
      if (images.length > MAX_RUN_IMAGES) throw new Error('Assistant image count exceeds the run limit.');
      const imageBytes = images.reduce((bytes, image) => bytes + validateImage(image), 0);
      if (imageBytes > MAX_IMAGE_BYTES) throw new Error('Assistant image input exceeds the combined 8 MiB limit.');
      if (this.#retired.has(request.runId)) throw new Error('Assistant run has ended; use a fresh run ID.');
      if (this.#requests.has(request.requestId)) throw new Error('Assistant request ID is already in flight.');
      run = this.#runs.get(request.runId);
      if (run === undefined) {
        if (this.#runs.size >= Math.min(MAX_ACTIVE_RUNS, state.config.maxJobs))
          throw new Error('Assistant active run limit reached.');
        if (
          !request.messages ||
          !request.tools ||
          request.toolResults !== undefined ||
          request.attachments !== undefined
        )
          throw new Error('First assistant turn requires messages and tools, without tool results.');
        if (new Set(request.tools.map(tool => tool.name)).size !== request.tools.length)
          throw new Error('Assistant tool names must be unique.');
        run = {
          input: request.messages.map(message => initialMessage(message, state.config.protocol)),
          tools: request.tools,
          pending: [],
          imageCount: images.length,
          imageBytes,
          seenCalls: new Set(),
          round: 0,
          spentTokens: 0,
          lastUsed: Date.now(),
        };
        this.#runs.set(request.runId, run);
        ownsRun = true;
      } else {
        if (run.controller) throw new Error('Assistant run already has an active request.');
        ownsRun = true;
        if (request.messages !== undefined || request.tools !== undefined || request.toolResults === undefined)
          throw new Error('Subsequent turns require only correlated tool results.');
        if (
          request.toolResults.length !== run.pending.length ||
          new Set(request.toolResults.map(result => result.callId)).size !== request.toolResults.length
        )
          throw new Error('Every function call requires exactly one correlated result.');
        const results = new Map(request.toolResults.map(result => [result.callId, result]));
        if (run.imageCount + images.length > MAX_RUN_IMAGES || run.imageBytes + imageBytes > MAX_IMAGE_BYTES)
          throw new Error('Assistant retained image limit reached; start a fresh run.');
        if (request.attachments?.some(attachment => !results.has(attachment.callId)))
          throw new Error('Assistant image attachment is not correlated to a current tool result.');
        for (const call of run.pending) {
          const result = results.get(call.callId);
          if (!result) throw new Error(`Missing function result for ${call.callId}.`);
          run.input.push(
            state.config.protocol === 'responses'
              ? { type: 'function_call_output', call_id: result.callId, output: result.output }
              : { role: 'tool', tool_call_id: result.callId, content: result.output },
          );
        }
        for (const attachment of request.attachments ?? []) {
          run.input.push({
            role: 'user',
            content: [
              {
                type: state.config.protocol === 'responses' ? 'input_text' : 'text',
                text: `Captured ${attachment.image.source} observation for function call ${attachment.callId}, artifact ${attachment.image.artifactId}.`,
              },
              imageContent(attachment.image, state.config.protocol),
            ],
          });
        }
        run.imageCount += images.length;
        run.imageBytes += imageBytes;
        run.pending = [];
      }
      const config = state.config;
      const contextEstimate =
        run.imageCount * ESTIMATED_IMAGE_TOKENS +
        Math.ceil((budgetText(run.input).length + JSON.stringify(run.tools).length) / 4);
      const finalOnly =
        run.round >= config.maxToolRounds ||
        run.spentTokens + contextEstimate + config.maxOutputTokens + config.finalResponseReserve > config.maxRunTokens;
      const tools = finalOnly ? [] : run.tools;
      const outputLimit = finalOnly ? config.finalResponseReserve : config.maxOutputTokens;
      const inputLength = budgetText(run.input).length;
      if (inputLength > MAX_CONTEXT_CHARACTERS)
        throw new Error('Assistant context limit reached; start a fresh run with a concise summary.');
      const bodyObject: RecordValue =
        config.protocol === 'responses'
          ? {
              model: config.model,
              input: run.input,
              store: false,
              stream: config.stream,
              max_output_tokens: outputLimit,
              tools: tools.map(tool => ({
                type: 'function',
                name: tool.name,
                description: tool.description,
                parameters: tool.parameters,
                strict: strictToolMode(config, tool),
              })),
              ...(finalOnly ? { tool_choice: 'none' } : {}),
              ...(config.reasoningEffort === undefined ? {} : { reasoning: { effort: config.reasoningEffort } }),
              // Preserve reasoning items locally across the application-owned tool loop.
              include: ['reasoning.encrypted_content'],
            }
          : {
              model: config.model,
              messages: run.input,
              stream: config.stream,
              ...(config.stream ? { stream_options: { include_usage: true } } : {}),
              max_completion_tokens: outputLimit,
              ...(tools.length
                ? {
                    tools: tools.map(tool => ({
                      type: 'function',
                      function: {
                        name: tool.name,
                        description: tool.description,
                        parameters: tool.parameters,
                        strict: strictToolMode(config, tool),
                      },
                    })),
                  }
                : {}),
              ...(finalOnly ? { tool_choice: 'none' } : {}),
              ...(config.reasoningEffort === undefined ? {} : { reasoning_effort: config.reasoningEffort }),
            };
      const body = JSON.stringify(bodyObject);
      const estimatedInput = run.imageCount * ESTIMATED_IMAGE_TOKENS + Math.ceil(budgetText(body).length / 4);
      if (
        run.spentTokens + estimatedInput + outputLimit + (finalOnly ? 0 : config.finalResponseReserve) >
        config.maxRunTokens
      )
        throw new Error('Assistant token budget is exhausted; no request was sent.');
      run.controller = new AbortController();
      this.#requests.add(request.requestId);
      registeredRequest = true;
      const parsed = parseAssistantResponse(
        await this.#request(config.protocol === 'responses' ? 'responses' : 'chat/completions', run.controller, body),
        config.protocol,
      );
      if (
        run.controller.signal.aborted ||
        this.#runs.get(request.runId) !== run ||
        this.#configuration.state().revision !== state.revision
      )
        throw new Error('Assistant completion arrived after the run was invalidated.');
      if (JSON.stringify(parsed).includes('data:image/'))
        throw new Error('Provider echoed encoded image input; response was withheld.');
      if (
        key &&
        (JSON.stringify(parsed).includes(key) || JSON.stringify(parsed).includes(JSON.stringify(key).slice(1, -1)))
      )
        throw new Error('Provider response contained the API credential and was withheld.');
      if (parsed.toolCalls.length > config.maxToolCallsPerRound)
        throw new Error('Provider exceeded the per-round function call limit.');
      if (finalOnly && parsed.toolCalls.length)
        throw new Error('Provider called a function after the tool-round limit.');
      const allowedNames = new Set(tools.map(tool => tool.name));
      const callIds = new Set<string>();
      for (const call of parsed.toolCalls) {
        if (!allowedNames.has(call.name)) throw new Error(`Provider requested an unknown tool: ${call.name}.`);
        if (callIds.has(call.callId) || run.seenCalls.has(call.callId))
          throw new Error(`Provider returned duplicate call ID: ${call.callId}.`);
        callIds.add(call.callId);
      }
      const turnUsage = usage(parsed.rawUsage, config.protocol, body, parsed.continuation);
      if (turnUsage.estimated) {
        turnUsage.inputTokens += run.imageCount * ESTIMATED_IMAGE_TOKENS;
        turnUsage.totalTokens += run.imageCount * ESTIMATED_IMAGE_TOKENS;
      }
      run.spentTokens += turnUsage.totalTokens;
      if (run.spentTokens > config.maxRunTokens)
        throw new Error('Provider response exceeded the run token budget; no function calls were accepted.');
      run.input.push(...parsed.continuation);
      run.pending = parsed.toolCalls;
      for (const call of parsed.toolCalls) run.seenCalls.add(call.callId);
      run.round += 1;
      run.lastUsed = Date.now();
      const result: AssistantTurnResult = {
        text: parsed.text,
        toolCalls: parsed.toolCalls,
        usage: turnUsage,
        round: run.round,
        remainingTokens: config.maxRunTokens - run.spentTokens,
        toolSchemaModes: tools.map(tool => ({ name: tool.name, strict: strictToolMode(config, tool) })),
      };
      if (!parsed.toolCalls.length) await this.endRun(request.runId);
      return result;
    } catch (error) {
      if (ownsRun && run && this.#runs.get(request.runId) === run) await this.cancelRun(request.runId);
      throw safeError(error, key);
    } finally {
      if (registeredRequest) this.#requests.delete(request.requestId);
      if (ownsRun && run) run.controller = undefined;
    }
  }
}
