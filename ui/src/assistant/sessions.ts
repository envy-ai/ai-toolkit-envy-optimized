import { z } from 'zod';

export const sessionIdSchema = z.string().uuid();
export const assistantEntrySchema = z
  .object({
    id: z.string().min(1).max(200),
    role: z.enum(['user', 'assistant', 'tool']),
    text: z.string().max(512_000),
    preview: z
      .object({ path: z.string().max(8192), url: z.string().max(16384).startsWith('/api/img/') })
      .strict()
      .optional(),
  })
  .strict();
export const sessionContentSchema = z
  .object({
    entries: z.array(assistantEntrySchema).max(10_000),
    draft: z.string().max(64_000),
    attachments: z.array(z.string().min(1).max(8192)).max(4),
  })
  .strict();
export const sessionTitleSchema = z.string().trim().min(1).max(120);
export const sessionUpdateSchema = z
  .object({
    revision: z.number().int().nonnegative(),
    content: sessionContentSchema.optional(),
    title: sessionTitleSchema.optional(),
  })
  .strict()
  .refine(value => value.content !== undefined || value.title !== undefined, 'A session update is required.');
export const assistantSessionSchema = z
  .object({
    version: z.literal(1),
    id: sessionIdSchema,
    title: sessionTitleSchema,
    titleSource: z.enum(['auto', 'manual']),
    createdAt: z.string().datetime(),
    updatedAt: z.string().datetime(),
    revision: z.number().int().positive(),
    entryCount: z.number().int().nonnegative(),
    content: sessionContentSchema,
  })
  .strict();

export type AssistantEntry = z.infer<typeof assistantEntrySchema>;
export type SessionContent = z.infer<typeof sessionContentSchema>;
export type AssistantSession = z.infer<typeof assistantSessionSchema>;
export const sessionMetadataSchema = assistantSessionSchema.omit({ version: true, content: true });
export type SessionMetadata = Omit<AssistantSession, 'version' | 'content'>;
export type SessionUpdate = z.infer<typeof sessionUpdateSchema>;
export const emptySessionContent = (): SessionContent => ({ entries: [], draft: '', attachments: [] });
export const hasSessionContent = (content: SessionContent) =>
  Boolean(content.entries.length || content.draft.trim() || content.attachments.length);

/** Local title only; attachment paths and tool output never become a title. */
export function automaticSessionTitle(content: SessionContent): string {
  const first = content.entries.find(entry => entry.role === 'user')?.text ?? content.draft;
  const plain = first
    .split(/\nAttached images \(local paths[^\n]*\):?/)[0]
    .replace(/\s+/g, ' ')
    .trim();
  let title = Array.from(plain).slice(0, 60).join('');
  const shortened = title.length < plain.length;
  if (shortened && title.length >= 120) title = Array.from(title).slice(0, -1).join('');
  return title ? title + (shortened ? '…' : '') : 'New chat';
}
