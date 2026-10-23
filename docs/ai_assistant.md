# AI assistant

Open **AI assistant** with the robot icon at the top right of any toolkit screen,
after the page's action buttons. The drawer stays
available across navigation. Chat supports image attachments, image previews,
activity records, token usage, cancellation, and a new-conversation button.
Function-call activity appears as small grey text showing only the function
name and a chevron, collapsed by default. Click the row, or focus it with Tab
and press Enter/Space, to expand its details and any image preview.
Drag the handle on the drawer's left edge to resize it; the width is remembered
in this browser. Arrow keys also adjust width, and double-click resets it.
On small screens the drawer uses the full screen width.
Chats autosave on the toolkit server. **Chat history** opens a compact saved-chat
list above the conversation, with title search, newest activity first, and
rename/delete controls. **New conversation** starts an empty chat without
discarding the old one. Empty chats do not create history entries until they
contain a message, draft, or attachment. A title is derived locally from the
first message (or draft until then), excluding the appended attachment paths;
there is no model request for titles. Titles can repeat because each chat has
its own UUID. Manual renames survive later autosaves.

Drafts, attachment paths, complete displayed messages, and tool preview
references are saved; tool entries retain their existing bounded activity
detail. Reopening a chat restores its text, not active provider runs, approvals,
encoded images, credentials, or tool execution. The existing last-40-non-tool
message context policy and per-run token budgets are unchanged. A preview can
be unavailable if its source file was moved or deleted.
Each sent question starts a fresh run with its tool-round counter and token
usage at zero; saved chat history does not consume tool rounds.

New/load/rename/delete controls wait while a reply, upload, or session operation
is active; use **Stop assistant** before switching a running conversation.
History selection and new-chat actions flush pending saves first. **Saving…**,
**Saved**, and persistent error details show storage state. Failed saves retain
the current content and offer **Retry save**. Revision conflicts never silently
overwrite another tab's changes: **Save a copy** keeps the local version under
a new UUID, or **Reload saved chat** asks before replacing local text. Open
history or use its refresh control to see changes from another browser tab.

History buttons support Tab and Enter/Space. Opening history focuses its search;
Escape closes history and returns focus to its toggle. Rename uses the shared
prefilled title dialog; delete names the selected chat and asks for confirmation.
The same compact history layout works on narrow drawers.

Saved records live in `data/assistant/sessions/<UUID>.json`, respecting
`AI_TOOLKIT_ASSISTANT_SETTINGS_DIR`. Versioned files use atomic writes and POSIX
owner-only permissions, with an 8 MiB per-chat limit reported as an error rather
than silently truncating messages. A metadata-only in-memory index serves the
history list. Writes and revision checks are serialized by the current single
Next.js writer process. Deleted UUIDs retain small `.deleted.json` tombstones,
preventing delayed saves or create retries from resurrecting them. No training
database schema change is required. Session APIs share the toolkit's existing
authentication and history is shared by authorized users of this toolkit
instance. Public file/media endpoints and assistant file tools block session
files, including canonical symlink aliases.

The last active UUID is remembered in this browser. This tab also keeps an
unsaved recovery snapshot, so storage/network failures do not discard its local
draft on reload. The former single `toolkit:assistant:messages` transcript is
imported once with a stable migration UUID and retained until its actual content
is confirmed saved. Recovery content takes priority over an older legacy copy.
Browser storage contains conversation data and references, never provider
credentials. Session files contain ordinary unencrypted chat text; provider
credentials remain in their separate configuration store.

The settings button opens the connection panel. It follows the RPG Creator
assistant's configuration and reuses its provider transport: API base URL,
Responses or Chat completions, exact model ID, reasoning effort, authentication,
optional credential persistence, visual observations, provider streaming, and
execution limits. **Refresh** saves the current connection draft, calls its
`/models` endpoint, and updates model suggestions. Manual model IDs are supported.
Models are also loaded when opening settings for a configured connection. The
initial model is empty; the application never guesses an available model.

The assistant can use existing toolkit APIs to inspect and manage jobs and GPU
queues, edit configurations and notes, request saves or samples, inspect losses
and reports, manage datasets and prompt sets, configure paths, auto-caption data,
and use an existing inference engine. It discovers the supported routes and
default job configuration with `toolkit_help`. Credentials are omitted from
tool results. Full edits to active jobs are rejected; sample-only edits use the
same merge and running-config snapshot as the human UI.

Search training jobs by name through `toolkit_api` with `method: "GET"`,
`endpoint: "/api/jobs"`, and `query: {name: "lineart", job_type: "train"}`.
The name filter matches literal substrings without case sensitivity; combine it
with `only_active: true` to restrict results to active jobs. Results retain job
IDs, configurations, status, and progress. A blank name keeps the normal list.

`GET /api/jobs` also accepts an optional positive integer `limit` and an opaque
`cursor` returned by the previous page. Paginated responses contain `jobs`,
`hasMore`, and `nextCursor`; they order jobs by descending creation time and ID.
`search` matches all whitespace-separated, case-insensitive literal substrings
in the name, while `name` keeps its single literal substring semantics. Both
filters apply before pagination across the full database. Searching reads only
ID/name/creation-time metadata for candidates and fetches full configurations
for the selected page. Omitting `limit` preserves the existing complete-list
API behavior; the Queue UI explicitly requests 100 results per page.

`PATCH /api/queue/{queueID}/reorder` accepts
`{orderedJobIds: [visible queued job IDs in desired order]}`. The server merges
that subset into its existing slots in the complete queued order using only
ID/position metadata. Hidden queued jobs retain their relative order and slots;
numeric positions are normalized to 1000-step ranks. Running/stopping jobs are
excluded. Validation and conditional queued/GPU/previous-rank updates share one
transaction, so a conflicting worker claim or reorder rolls back rather than
partially changing the queue. Invalid IDs return 400; detected concurrent
changes return 409. Successful responses include `success: true` and authoritative
`positions: [{id, queue_position}]` for the queued jobs. Complete queue-order
submissions remain supported.

Additional tools inspect dataset and sample images, read and write text sidecars
and text sample outputs, copy/rename/delete dataset files, import local files,
and create or edit training images with CPU operations. Image edits support
self-contained SVGs or solid backgrounds, resizing, cropping, rotation, flips,
grayscale, and compositing. Generated model images use the existing inference
API. Imports reuse the human upload APIs; LoRA files upload in 5 MiB chunks.
Image copies and renames carry their matching `.txt` caption. Existing targets
require explicit `overwrite: true`.

**Review changes before execution** defaults on. The drawer shows each action's
arguments and waits for **Approve action** or **Decline**. Read operations run
immediately. Job and queue controls that use GET are still treated as changes.
Turning review off permits automatic actions requested by the user. Canceling
stops further tools and aborts provider requests; completed edits are retained.
Data and sample views refresh after successful changes. The polling loop
coalesces refreshes while an existing request is in progress.

Visual observation must be enabled to send images to the configured provider.
`inspect_image` is advertised in model tools and `toolkit_help` only while visual
observation is enabled; the server also checks the setting when the tool runs.
Images are normalized to PNG on CPU, capped at 1536 pixels per axis, and sent as
native image parts rather than base64 text in tool output. Each run retains at
most four observations and 8 MiB of encoded image bytes. Missing/unsupported
media produces a tool error; the assistant does not claim it viewed that file.
Base64 validation uses a stack-safe character check for large observations,
while retaining strict padding, canonical encoding, byte-size, and image checks.
Video and audio samples can be listed and opened in their existing UI viewers;
the visual observation tool supports still images. Start a fresh request for
additional observations. Image attachment uploads are capped at four files,
32 MiB each. Model generation and auto-captioning can use GPU resources; ordinary
chat, image inspection, and these deterministic image edits do not load a
training model or consume GPU memory.

Settings live under `data/assistant` in the toolkit root, separately from the
general settings API. Keys default to server-process memory. Opt-in POSIX
persistence uses an owner-only file; it is not an encrypted keychain. The same
`AI_TOOLKIT_AUTH` middleware protects assistant routes. Filesystem tools check
both configured folder boundaries and real paths to reject symlink escapes.
Provider requests retain Node TLS verification and do not follow redirects.
Install an appropriate CA or use `NODE_EXTRA_CA_CERTS` for a private proxy.

The server writes assistant budget diagnostics to
`data/assistant/assistant-diagnostics.jsonl`, or
`$AI_TOOLKIT_ASSISTANT_SETTINGS_DIR/assistant-diagnostics.jsonl` when that override
is set. Each line is a versioned JSON event with a UTC timestamp, run/request IDs,
one-based round, initial/continuation phase, model, protocol, configuration
revision, and the reasoning/output/run/reserve/tool-round settings. The active
log rotates before exceeding 5 MiB, retaining one `.jsonl.1` archive. Files use
owner-only permissions. Directly constructed providers do not log unless given
an explicit diagnostic sink, so isolated provider tests do not write user logs.

`request_budget` records admission, spent/remaining tokens, context and request
estimates, normal/final output allowances, predicted/reserved totals, final-only
decision and its causes, item/tool counts, incoming tool-result IDs and lengths,
and retained image counts/bytes. Incoming tool results identify the tool and
call ID with its result length; accounted usage also lists returned tool names,
call IDs, and argument lengths. Its `characters` object separates serialized
input, tool definitions, actual wire schemas, protocol overhead, message text,
tool-result text, function arguments, and opaque encrypted reasoning lengths.
`contextTokensWithoutEncryptedEstimate` shows the same rough context estimate
after subtracting ciphertext characters; it is diagnostic only.
`request_rejected` is written before context, pre-request budget, and post-response
budget errors. `provider_usage` first records selected numeric provider counts
at `stage: "received"`, including cached input and reasoning tokens when supplied,
even if response parsing subsequently rejects an incomplete response. At
`stage: "accounted"`, it adds the application's actual or estimated usage and
updated spent/remaining totals.

Diagnostics contain no prompts, message/tool output text, function argument
contents, credentials, image/base64 data, or reasoning ciphertext. Size estimates
still use the existing characters/4 policy, including ciphertext in the actual
budget calculation; logging does not correct that estimate or change budget
admission. Reported run usage sums input and output across all provider requests,
including replayed context and cached input; it is neither unique context size
nor monetary cost. Missing provider usage details remain absent, rather than
being inferred. Log write failures appear in the server console and fail the
affected operation; there is no silent logging fallback. Restart the UI server
after building to activate this instrumentation.

Optional server environment variables:

| Variable                                | Purpose                                                     |
| --------------------------------------- | ----------------------------------------------------------- |
| `AI_TOOLKIT_ASSISTANT_SETTINGS_DIR`     | Assistant settings directory; useful for isolated tests.    |
| `AI_TOOLKIT_ASSISTANT_BASE_URL`         | Explicit connection override at server startup.             |
| `AI_TOOLKIT_ASSISTANT_AUTH_MODE`        | `none` or `api-key` with the connection override.           |
| `AI_TOOLKIT_ASSISTANT_MODEL`            | Exact model ID with the connection override.                |
| `AI_TOOLKIT_ASSISTANT_REASONING_EFFORT` | Optional reasoning setting with the connection override.    |
| `AI_TOOLKIT_ASSISTANT_VISION`           | `true` enables observations with the connection override.   |
| `AI_TOOLKIT_DATABASE_URL`               | Explicit SQLite datasource URL for an isolated UI instance. |

Without an explicit connection override, saved settings are loaded normally.
Restart/rebuild the UI to load new code; a trainer restart is unnecessary.

Verification uses `node --test tests/assistant.test.cjs` from `ui`, source
TypeScript checking, and desktop/mobile browser checks. Live Shodan testing used
Responses, `gpt-6.1-sol`, medium reasoning and vision, against a separate SQLite
database and dataset/output folders. It verified sample inspection, dataset
creation, copying, horizontal flipping and caption writing while leaving the
fixture running job unchanged. A live job clone preserved the original model,
dataset and memory settings, changed only the requested step/sample intervals,
and remained stopped. On the synthetic two-color fixture, the model
identified the color order correctly but invented a cream background in a
caption; generated captions therefore remain drafts for human review. This is
a measured model-output limitation, not a claim of perfect caption accuracy.

All 22 assistant checks and 35 related UI regression checks passed. Browser tests
verified approval, rejection, cancellation, and refreshing the dataset view.
The production build passed, followed by production-mode image inspection and
live proxy model discovery. The normal UI/worker and temporary test servers were
stopped afterward. Building the main checkout required a fresh webpack cache and
`NODE_OPTIONS=--max-old-space-size=8192`; this override was used only for the build.

Saved-chat verification passed 42 assistant checks and two thumbnail/file-access
checks, plus source and worker TypeScript checks. The production build passed
with a temporary `NODE_OPTIONS=--max-old-space-size=16384`; two attempts at
8192 MiB exhausted Node's heap, including one with a fresh webpack cache. This
override was used only for the build. The UI alone was restarted, the session
history API returned HTTP 200, and generic file/media access to session storage
returned HTTP 403. The training worker remained running. Browser interaction
checks were unavailable because no browser was connected; no live model calls
were needed for saved-chat verification.

Protocol details follow the [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
