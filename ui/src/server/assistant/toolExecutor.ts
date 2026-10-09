import fs from 'fs/promises';
import path from 'path';
import { randomUUID } from 'crypto';
import sharp from 'sharp';
import { z } from 'zod';
import { getDataRoot, getDatasetsRoot, getTrainingFolder, getHFToken, getModelsPath } from '@/server/settings';
import type { AssistantImageInput, AssistantToolCall } from '@/assistant/AssistantProvider';
import { API_CATALOG, ASSISTANT_TOOLS, isReadOnlyTool } from '@/assistant/tools';
import { defaultJobConfig } from '@/app/jobs/new/jobConfig';
import { NodeAssistantProvider } from './NodeAssistantProvider';

const text = z.string().min(1).max(4096);
const dimensions = z.number().int().min(1).max(4096);
const schemas = {
  toolkit_help: z.object({}).strict(),
  toolkit_api: z
    .object({
      endpoint: text,
      method: z.enum(['GET', 'POST', 'PATCH', 'DELETE']),
      query: z.record(z.string(), z.union([z.string(), z.number(), z.boolean()])).optional(),
      body: z.record(z.string(), z.unknown()).optional(),
    })
    .strict(),
  inspect_image: z.object({ path: text }).strict(),
  dataset_file: z
    .object({
      operation: z.enum(['read', 'write', 'copy', 'rename', 'delete']),
      path: text,
      destination: text.optional(),
      text: z
        .string()
        .max(1024 * 1024)
        .optional(),
      overwrite: z.boolean().optional(),
    })
    .strict(),
  edit_image: z
    .object({
      source: text.optional(),
      destination: text,
      svg: z
        .string()
        .max(1024 * 1024)
        .optional(),
      width: dimensions.optional(),
      height: dimensions.optional(),
      background: z.string().max(100).optional(),
      crop: z
        .object({
          left: z.number().int().nonnegative(),
          top: z.number().int().nonnegative(),
          width: dimensions,
          height: dimensions,
        })
        .strict()
        .optional(),
      rotate: z.number().finite().min(-360).max(360).optional(),
      flipX: z.boolean().optional(),
      flipY: z.boolean().optional(),
      grayscale: z.boolean().optional(),
      overlay: z
        .object({ path: text, left: z.number().int().nonnegative(), top: z.number().int().nonnegative() })
        .strict()
        .optional(),
      overwrite: z.boolean().optional(),
    })
    .strict(),
  navigate: z.object({ path: text }).strict(),
  upload_file: z
    .object({ source: text, target: z.enum(['dataset', 'reference', 'lora']), datasetName: text.optional() })
    .strict(),
};

// Exact existing endpoints; never an arbitrary URL, shell or provider request.
const allowed = [
  ['GET', '^/api/jobs$'],
  ['POST', '^/api/jobs$'],
  ['GET', '^/api/jobs/[^/]+/(samples|files|log|loss|loss-report|notes|dataset-images|plugin)$'],
  ['GET', '^/api/jobs/[^/]+/(start|stop|kill|delete|mark_stopped|save_now|sample_now)$'],
  ['POST', '^/api/jobs/[^/]+/notes$'],
  ['DELETE', '^/api/jobs/[^/]+/loss$'],
  ['GET', '^/api/queue$'],
  ['GET', '^/api/queue/[^/]+/(start|stop)$'],
  ['PATCH', '^/api/queue/[^/]+/reorder$'],
  ['GET', '^/api/datasets/list$'],
  ['POST', '^/api/datasets/(create|delete|listImages)$'],
  ['POST', '^/api/img/(caption|delete)$'],
  ['POST', '^/api/caption/(get|getBatch)$'],
  ['GET', '^/api/(settings|model_archs|gpu|cpu|monitor|scripts|loras|prompt-sets|comfy/options)$'],
  ['POST', '^/api/(settings|prompt-sets|files/delete)$'],
  ['DELETE', '^/api/prompt-sets$'],
  ['GET', '^/api/inference/[a-zA-Z0-9_/-]+$'],
  ['POST', '^/api/inference/[a-zA-Z0-9_/-]+$'],
  ['DELETE', '^/api/inference/[a-zA-Z0-9_/-]+$'],
];

export function validateApiOperation(value: unknown) {
  const args = schemas.toolkit_api.parse(value);
  if (
    args.endpoint.includes('%') ||
    args.endpoint.includes('..') ||
    !allowed.some(([method, pattern]) => method === args.method && new RegExp(pattern).test(args.endpoint))
  ) {
    throw new Error('This endpoint/method is not available to the assistant.');
  }
  if (
    args.endpoint === '/api/settings' &&
    args.body &&
    ['HF_TOKEN', 'OPENAI_API_KEY', 'AI_TOOLKIT_AUTH'].some(key => key in args.body!)
  ) {
    throw new Error('Manage credentials in the human settings panel.');
  }
  return args;
}

function under(filename: string, root: string) {
  return filename === root || filename.startsWith(root + path.sep);
}

/** Verify real paths, including existing parents of new files; reject symlink escapes. */
export async function scopedPath(filename: string, roots: string[], write = false) {
  const resolved = path.resolve(roots[0], filename);
  const lexical = roots.find(root => under(resolved, path.resolve(root)));
  if (!lexical) throw new Error('Path is outside the permitted folders.');
  let ancestor = resolved;
  while (!(await fs.lstat(ancestor).catch(() => null))) {
    const parent = path.dirname(ancestor);
    if (parent === ancestor) throw new Error('Cannot resolve file parent.');
    ancestor = parent;
  }
  const actual = path.join(await fs.realpath(ancestor), path.relative(ancestor, resolved));
  const realRoot = await fs.realpath(lexical);
  if (!under(actual, realRoot) || (write && actual === realRoot))
    throw new Error('Path escapes its folder through a symbolic link.');
  // Return the canonical path, so edits never accidentally follow a leaf symlink.
  return actual;
}

async function roots() {
  return [await getDatasetsRoot(), await getTrainingFolder(), await getDataRoot()];
}
async function readBounded(filename: string) {
  const info = await fs.stat(filename);
  if (!info.isFile() || info.size > 1024 * 1024) throw new Error('Text file must be at most 1 MiB.');
  return fs.readFile(filename, 'utf8');
}
function sidecar(filename: string) {
  return filename.replace(/\.[^/.]+$/, '') + '.txt';
}
function textFile(filename: string) {
  return /\.(txt|json|ya?ml|md)$/i.test(filename);
}
async function writableDestination(filename: string, overwrite = false) {
  if (!overwrite && (await fs.lstat(filename).catch(() => null)))
    throw new Error('Destination exists; explicit overwrite:true is required.');
  await fs.mkdir(path.dirname(filename), { recursive: true });
}
function mediaUrl(filename: string) {
  return `/api/img/${encodeURIComponent(filename)}`;
}
const privateField = /^(token|api_key|apiKey|authorization|HF_TOKEN|password|secret)$/i;
function redactResult(key: string, value: unknown): unknown {
  if (privateField.test(key)) return '[REDACTED]';
  if (key === 'job_config' && typeof value === 'string') {
    try {
      return JSON.stringify(JSON.parse(value), redactResult);
    } catch {
      return value;
    }
  }
  return value;
}
async function boundedResponseText(response: Response) {
  if (!response.body) return '';
  const reader = response.body.getReader(),
    decoder = new TextDecoder();
  let text = '',
    length = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > 512_000) {
        await reader.cancel();
        throw new Error('API result is too large. Narrow the query or page the results.');
      }
      text += decoder.decode(value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}

export type ToolExecution = {
  output: string;
  image?: AssistantImageInput;
  preview?: { path: string; url: string };
  navigate?: string;
  changed?: boolean;
};
export type ToolContext = {
  origin: string;
  authorization: string | null;
  signal?: AbortSignal;
  provider: NodeAssistantProvider;
  fetcher?: typeof fetch;
};

export async function executeAssistantTool(call: AssistantToolCall, context: ToolContext): Promise<ToolExecution> {
  if (!Object.hasOwn(schemas, call.name)) throw new Error(`Unknown tool: ${call.name}`);
  const args: any = schemas[call.name as keyof typeof schemas].parse(call.arguments);
  context.signal?.throwIfAborted();
  if (call.name === 'toolkit_help')
    return {
      output: JSON.stringify({
        api: API_CATALOG,
        tools: ASSISTANT_TOOLS,
        defaultJobConfig,
        notes:
          'Running jobs: update with sample_only:true. Read the job before editing. Images are observed with inspect_image. File operations use configured roots; do not mutate a running dataset unless requested.',
      }),
    };
  if (call.name === 'navigate') {
    if (
      !/^\/(dashboard|datasets|jobs|generate|settings)(\/[^?#]*)?(\?[^#]*)?$/.test(args.path) ||
      args.path.includes('..') ||
      args.path.includes('//')
    )
      throw new Error('Invalid toolkit screen.');
    return { output: JSON.stringify({ opened: args.path }), navigate: args.path };
  }
  if (call.name === 'toolkit_api') {
    const operation = validateApiOperation(args);
    const config = (await context.provider.getConfiguration()).config;
    const checkBatch = (value: unknown): void => {
      if (Array.isArray(value)) {
        if (value.length > config.maxBatchSize) throw new Error('Operation exceeds the configured batch limit.');
        value.forEach(checkBatch);
      } else if (value && typeof value === 'object') Object.values(value).forEach(checkBatch);
    };
    checkBatch(operation.body);
    if (
      [
        '/api/img/caption',
        '/api/caption/get',
        '/api/caption/getBatch',
        '/api/img/delete',
        '/api/files/delete',
      ].includes(operation.endpoint)
    ) {
      const allowedRoots = await roots();
      const scope =
        operation.endpoint === '/api/files/delete'
          ? [allowedRoots[1]]
          : operation.endpoint === '/api/img/delete'
            ? allowedRoots
            : [allowedRoots[0]];
      if (operation.body?.imgPath)
        operation.body.imgPath = await scopedPath(String(operation.body.imgPath), scope, true);
      if (operation.body?.filePath)
        operation.body.filePath = await scopedPath(String(operation.body.filePath), scope, true);
      if (Array.isArray(operation.body?.imgPaths))
        operation.body!.imgPaths = await Promise.all(
          operation.body!.imgPaths.map(filename => scopedPath(String(filename), scope, true)),
        );
      if (operation.body?.ext && !/^[a-zA-Z0-9]+$/.test(String(operation.body.ext)))
        throw new Error('Caption extension must contain only letters and numbers.');
    }
    const url = new URL(operation.endpoint, context.origin);
    for (const [key, value] of Object.entries(operation.query || {})) url.searchParams.set(key, String(value));
    let body = operation.body;
    if (operation.endpoint === '/api/settings' && operation.method === 'POST')
      body = { ...body, HF_TOKEN: await getHFToken() };
    if (operation.endpoint === '/api/jobs' && operation.method === 'POST' && body?.id && !body.sample_only) {
      const current = await (context.fetcher || fetch)(
        new URL(`/api/jobs?id=${encodeURIComponent(String(body.id))}`, context.origin),
        { headers: context.authorization ? { Authorization: context.authorization } : {}, signal: context.signal },
      );
      const job = await current.json();
      if (['running', 'stopping', 'queued'].includes(job?.status))
        throw new Error(
          'Active jobs support sample-only edits; stop the job explicitly before editing its training configuration.',
        );
    }
    const response = await (context.fetcher || fetch)(url, {
      method: operation.method,
      headers: {
        'Content-Type': 'application/json',
        ...(context.authorization ? { Authorization: context.authorization } : {}),
      },
      body: operation.method === 'GET' ? undefined : JSON.stringify(body || {}),
      signal: context.signal,
      redirect: 'error',
    });
    const bytes = await boundedResponseText(response);
    let result: any;
    try {
      result = JSON.parse(bytes);
    } catch {
      result = { text: bytes };
    }
    if (!response.ok) throw new Error(`Toolkit API HTTP ${response.status}: ${JSON.stringify(result, redactResult)}`);
    if (operation.endpoint === '/api/settings') {
      delete result.HF_TOKEN;
      delete result.OPENAI_API_KEY;
      delete result.AI_TOOLKIT_AUTH;
    }
    // Inference discovery may return internal endpoint credentials.
    const output = JSON.stringify(result, redactResult);
    return { output, changed: !isReadOnlyTool(call.name, operation) };
  }
  const allowedRoots = await roots();
  if (call.name === 'upload_file') {
    const source = await scopedPath(args.source, [...allowedRoots, await getModelsPath()]);
    const info = await fs.stat(source);
    if (!info.isFile()) throw new Error('Upload source must be a regular file.');
    const fetcher = context.fetcher || fetch;
    const headers: Record<string, string> = context.authorization ? { Authorization: context.authorization } : {};
    const uploadRequest = async (route: string, body: BodyInit, json = false) => {
      const response = await fetcher(new URL(route, context.origin), {
        method: 'POST',
        headers: { ...headers, ...(json ? { 'Content-Type': 'application/json' } : {}) },
        body,
        signal: context.signal,
        redirect: 'error',
      });
      const result = await response.json();
      if (!response.ok) throw new Error(`Upload HTTP ${response.status}: ${JSON.stringify(result)}`);
      return result;
    };
    if (args.target === 'lora') {
      if (!/\.safetensors$/i.test(source)) throw new Error('LoRA uploads require a safetensors checkpoint.');
      const fileName = path.basename(source),
        size = info.size;
      const query = new URLSearchParams({ fileName, size: String(size) });
      const started = await uploadRequest(`/api/loras/upload?action=start&${query}`, JSON.stringify({}), true);
      const handle = await fs.open(source, 'r');
      try {
        const bytes = Buffer.alloc(5 * 1024 * 1024);
        for (let offset = 0; offset < size; ) {
          context.signal?.throwIfAborted();
          const { bytesRead } = await handle.read(bytes, 0, Math.min(bytes.length, size - offset), offset);
          if (!bytesRead) throw new Error('Checkpoint changed during upload.');
          await uploadRequest(
            `/api/loras/upload?action=chunk&uploadId=${encodeURIComponent(started.uploadId)}&offset=${offset}`,
            new Uint8Array(bytes.subarray(0, bytesRead)),
          );
          offset += bytesRead;
        }
        const result = await uploadRequest(
          `/api/loras/upload?action=finish&uploadId=${encodeURIComponent(started.uploadId)}&${query}`,
          JSON.stringify({}),
          true,
        );
        return { output: JSON.stringify(result), changed: true };
      } catch (error) {
        // Cancellation cleanup uses a fresh bounded request so a canceled signal cannot leave a partial upload behind.
        await fetcher(
          new URL(`/api/loras/upload?action=cancel&uploadId=${encodeURIComponent(started.uploadId)}`, context.origin),
          { method: 'POST', headers, signal: AbortSignal.timeout(5000) },
        ).catch(() => {});
        throw error;
      } finally {
        await handle.close();
      }
    }
    if (info.size > 32 * 1024 * 1024)
      throw new Error('Image/sidecar upload limit is 32 MiB; use dataset_file copy for larger local files.');
    const form = new FormData();
    form.append('files', new Blob([new Uint8Array(await fs.readFile(source))]), path.basename(source));
    if (args.target === 'dataset') {
      if (!args.datasetName || path.basename(args.datasetName) !== args.datasetName || args.datasetName.startsWith('.'))
        throw new Error('A valid datasetName is required.');
      form.append('datasetName', args.datasetName);
    }
    const result = await uploadRequest(args.target === 'dataset' ? '/api/datasets/upload' : '/api/img/upload', form);
    return { output: JSON.stringify(result), changed: true };
  }
  if (call.name === 'inspect_image') {
    const filename = await scopedPath(args.path, allowedRoots);
    const config = await context.provider.getConfiguration();
    if (!config.config.vision)
      throw new Error('Enable visual observations in Assistant Settings before inspecting images.');
    const pipeline = sharp(filename, { limitInputPixels: 64 * 1024 * 1024 });
    const metadata = await pipeline.metadata();
    const { data, info } = await pipeline
      .resize({ width: 1536, height: 1536, fit: 'inside', withoutEnlargement: true })
      .png()
      .toBuffer({ resolveWithObject: true });
    if (data.length > 8 * 1024 * 1024) throw new Error('Normalized image exceeds observation size limit.');
    const caption = await readBounded(sidecar(filename)).catch(() => null);
    return {
      output: JSON.stringify({
        path: filename,
        width: metadata.width,
        height: metadata.height,
        observedWidth: info.width,
        observedHeight: info.height,
        caption,
        url: mediaUrl(filename),
      }),
      image: {
        artifactId: randomUUID(),
        source: 'visible-image',
        mimeType: 'image/png',
        width: info.width,
        height: info.height,
        dataBase64: data.toString('base64'),
      },
      preview: { path: filename, url: mediaUrl(filename) },
    };
  }
  if (call.name === 'dataset_file') {
    const source = await scopedPath(
      args.path,
      args.operation === 'copy'
        ? allowedRoots
        : args.operation === 'read'
          ? allowedRoots.slice(0, 2)
          : [allowedRoots[0]],
      args.operation !== 'read' && args.operation !== 'copy',
    );
    if (args.operation === 'read') {
      if (!textFile(source)) throw new Error('Use inspect_image for image files.');
      const outputRoot = await fs.realpath(allowedRoots[1]).catch(() => path.resolve(allowedRoots[1]));
      if (
        path.basename(source).startsWith('.') ||
        (under(source, outputRoot) && path.basename(source) === 'engine.json')
      )
        throw new Error('Private runtime metadata is unavailable to assistant file tools.');
      return { output: JSON.stringify({ path: source, text: await readBounded(source) }) };
    }
    if (args.operation === 'write') {
      if (!textFile(source) || args.text === undefined) throw new Error('Write requires a text sidecar path and text.');
      await writableDestination(source, args.overwrite);
      await fs.writeFile(source, args.text, { flag: args.overwrite ? 'w' : 'wx' });
    } else if (args.operation === 'delete') {
      await fs.unlink(source);
      if (!textFile(source))
        await fs.unlink(sidecar(source)).catch(error => {
          if (error.code !== 'ENOENT') throw error;
        });
    } else {
      if (!args.destination) throw new Error('Destination is required.');
      const destination = await scopedPath(args.destination, [allowedRoots[0]], true);
      if (destination === source) throw new Error('Source and destination must differ.');
      await writableDestination(destination, args.overwrite);
      const captionExists = !textFile(source) && (await fs.stat(sidecar(source)).catch(() => null));
      if (captionExists) await writableDestination(sidecar(destination), args.overwrite);
      // copy then unlink also handles datasets stored on different drives.
      await fs.copyFile(source, destination, args.overwrite ? 0 : 1);
      if (captionExists) await fs.copyFile(sidecar(source), sidecar(destination), args.overwrite ? 0 : 1);
      if (args.operation === 'rename') {
        await fs.unlink(source);
        if (captionExists) await fs.unlink(sidecar(source));
      }
      return {
        output: JSON.stringify({ operation: args.operation, path: source, destination }),
        changed: true,
        ...(!textFile(destination) ? { preview: { path: destination, url: mediaUrl(destination) } } : {}),
      };
    }
    return { output: JSON.stringify({ operation: args.operation, path: source }), changed: true };
  }
  const destination = await scopedPath(args.destination, [allowedRoots[0]], true);
  if (!/\.(png|jpe?g|webp)$/i.test(destination)) throw new Error('Image destination must be PNG, JPEG or WebP.');
  let pipeline;
  if (args.svg) {
    if (args.source || /(?:href\s*=|url\s*\(|<!DOCTYPE|<!ENTITY|<script|<foreignObject)/i.test(args.svg))
      throw new Error('SVG must be self-contained and cannot reference external assets.');
    pipeline = sharp(Buffer.from(args.svg), { limitInputPixels: 16 * 1024 * 1024 });
  } else if (args.source)
    pipeline = sharp(await scopedPath(args.source, allowedRoots), { limitInputPixels: 64 * 1024 * 1024 });
  else {
    if (!args.width || !args.height) throw new Error('Creating an image requires width and height.');
    pipeline = sharp({
      create: { width: args.width, height: args.height, channels: 4, background: args.background || '#ffffff' },
    });
  }
  if (args.crop) pipeline = pipeline.extract(args.crop);
  if (args.width || args.height) pipeline = pipeline.resize(args.width, args.height, { fit: 'fill' });
  if (args.rotate) pipeline = pipeline.rotate(args.rotate);
  if (args.flipX) pipeline = pipeline.flop();
  if (args.flipY) pipeline = pipeline.flip();
  if (args.grayscale) pipeline = pipeline.grayscale();
  if (args.overlay)
    pipeline = pipeline.composite([
      { input: await scopedPath(args.overlay.path, allowedRoots), left: args.overlay.left, top: args.overlay.top },
    ]);
  pipeline = /\.webp$/i.test(destination)
    ? pipeline.webp()
    : /\.jpe?g$/i.test(destination)
      ? pipeline.jpeg()
      : pipeline.png();
  await writableDestination(destination, args.overwrite);
  const buffer = await pipeline.toBuffer();
  context.signal?.throwIfAborted();
  await fs.writeFile(destination, buffer, { flag: args.overwrite ? 'w' : 'wx' });
  return {
    output: JSON.stringify({ saved: destination, url: mediaUrl(destination) }),
    preview: { path: destination, url: mediaUrl(destination) },
    changed: true,
  };
}
