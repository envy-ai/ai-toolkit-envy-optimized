import { NextRequest, NextResponse } from 'next/server';
import fs from 'fs/promises';
import path from 'path';
import YAML from 'yaml';
import { TOOLKIT_ROOT } from '@/paths';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const PROMPT_SETS_DIR = path.join(TOOLKIT_ROOT, 'prompt_sets');
const NAME_RE = /^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$/;
const MAX_PROMPT_LENGTH = 100_000;
const MAX_FILE_SIZE = 2_000_000;

type PromptEntry =
  | { kind: 'simple'; prompt: string }
  | {
      kind: 'specific';
      neutral_prompt: string;
      positive_prompt: string;
      negative_prompt: string;
      cfg_negative_prompt?: string;
      cfg_negative_prompt_positive?: string;
      cfg_negative_prompt_negative?: string;
    };

type PromptSet = {
  version: 1;
  prompt_entries: PromptEntry[];
  positive_prefix: string;
  negative_prefix: string;
  cfg_negative_prefix: string;
  cfg_negative_prefix_positive: string;
  cfg_negative_prefix_negative: string;
};

const errorResponse = (message: string, status: number) => NextResponse.json({ error: message }, { status });
const validText = (value: unknown): value is string =>
  typeof value === 'string' && value.length <= MAX_PROMPT_LENGTH;

function validatePromptSet(value: unknown): PromptSet | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const data = value as Record<string, unknown>;
  if (data.version !== 1 || !Array.isArray(data.prompt_entries) ||
      data.prompt_entries.length < 1 || data.prompt_entries.length > 128) return null;
  const prefixes = [
    'positive_prefix', 'negative_prefix', 'cfg_negative_prefix',
    'cfg_negative_prefix_positive', 'cfg_negative_prefix_negative',
  ] as const;
  if (prefixes.some(key => !validText(data[key]))) return null;

  const entries: PromptEntry[] = [];
  for (const raw of data.prompt_entries) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return null;
    const entry = raw as Record<string, unknown>;
    if (entry.kind === 'simple' && validText(entry.prompt)) {
      entries.push({ kind: 'simple', prompt: entry.prompt });
    } else if (entry.kind === 'specific' &&
      validText(entry.neutral_prompt) && validText(entry.positive_prompt) &&
      validText(entry.negative_prompt) &&
      ['cfg_negative_prompt', 'cfg_negative_prompt_positive', 'cfg_negative_prompt_negative']
        .every(key => entry[key] === undefined || validText(entry[key]))) {
      entries.push({
        kind: 'specific',
        neutral_prompt: entry.neutral_prompt,
        positive_prompt: entry.positive_prompt,
        negative_prompt: entry.negative_prompt,
        cfg_negative_prompt: entry.cfg_negative_prompt as string | undefined,
        cfg_negative_prompt_positive: entry.cfg_negative_prompt_positive as string | undefined,
        cfg_negative_prompt_negative: entry.cfg_negative_prompt_negative as string | undefined,
      });
    } else {
      return null;
    }
  }
  return {
    version: 1,
    prompt_entries: entries,
    positive_prefix: data.positive_prefix as string,
    negative_prefix: data.negative_prefix as string,
    cfg_negative_prefix: data.cfg_negative_prefix as string,
    cfg_negative_prefix_positive: data.cfg_negative_prefix_positive as string,
    cfg_negative_prefix_negative: data.cfg_negative_prefix_negative as string,
  };
}

function filePathForName(name: unknown): string | null {
  if (typeof name !== 'string' || !NAME_RE.test(name) || name !== name.trim() || name.endsWith('.')) return null;
  return path.join(PROMPT_SETS_DIR, `${name}.yaml`);
}

export async function GET(request: NextRequest) {
  try {
    await fs.mkdir(PROMPT_SETS_DIR, { recursive: true });
    const name = request.nextUrl.searchParams.get('name');
    if (name !== null) {
      const filePath = filePathForName(name);
      if (!filePath) return errorResponse('Invalid prompt set name.', 400);
      let stat;
      try {
        stat = await fs.lstat(filePath);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') return errorResponse('Prompt set not found.', 404);
        throw error;
      }
      if (!stat.isFile() || stat.size > MAX_FILE_SIZE) return errorResponse('Invalid prompt set file.', 400);
      let parsed: unknown;
      try {
        parsed = YAML.parse(await fs.readFile(filePath, 'utf8'));
      } catch {
        return errorResponse('Prompt set YAML could not be parsed.', 400);
      }
      const promptSet = validatePromptSet(parsed);
      if (!promptSet) return errorResponse('Prompt set has an invalid format.', 400);
      return NextResponse.json({ name, prompt_set: promptSet });
    }

    const entries = await fs.readdir(PROMPT_SETS_DIR, { withFileTypes: true });
    const names = entries
      .filter(entry => entry.isFile() && entry.name.endsWith('.yaml'))
      .map(entry => entry.name.slice(0, -'.yaml'.length))
      .filter(name => filePathForName(name) !== null)
      .sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' }) || a.localeCompare(b));
    return NextResponse.json({ names });
  } catch (error) {
    console.error('Failed to read prompt sets:', error);
    return errorResponse('Could not read prompt sets.', 500);
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const filePath = filePathForName(body?.name);
    if (!filePath) return errorResponse('Use a name of up to 80 letters, numbers, spaces, dots, underscores or hyphens.', 400);
    const promptSet = validatePromptSet(body?.prompt_set);
    if (!promptSet) return errorResponse('Prompt set has an invalid format.', 400);
    const yaml = YAML.stringify(promptSet, { lineWidth: 0 });
    if (Buffer.byteLength(yaml, 'utf8') > MAX_FILE_SIZE) return errorResponse('Prompt set is too large.', 400);
    await fs.mkdir(PROMPT_SETS_DIR, { recursive: true });
    try {
      await fs.writeFile(filePath, yaml, { encoding: 'utf8', flag: 'wx' });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EEXIST') return errorResponse('A prompt set with that name already exists.', 409);
      throw error;
    }
    return NextResponse.json({ name: body.name }, { status: 201 });
  } catch (error) {
    if (error instanceof SyntaxError) return errorResponse('Invalid JSON request.', 400);
    console.error('Failed to save prompt set:', error);
    return errorResponse('Could not save prompt set.', 500);
  }
}
