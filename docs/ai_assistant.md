# AI assistant

Open **AI assistant** with the robot icon at the top right of any toolkit screen,
after the page's action buttons. The drawer stays
available across navigation. Chat supports image attachments, image previews,
activity records, token usage, cancellation, and a new-conversation button.
Drag the handle on the drawer's left edge to resize it; the width is remembered
in this browser. Arrow keys also adjust width, and double-click resets it.
On small screens the drawer uses the full screen width.
Conversation text is kept in this browser tab's session storage, bounded to the
last 100 entries. Provider credentials never enter that storage.

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
Images are normalized to PNG on CPU, capped at 1536 pixels per axis, and sent as
native image parts rather than base64 text in tool output. Each run retains at
most four observations and 8 MiB of encoded image bytes. Missing/unsupported
media produces a tool error; the assistant does not claim it viewed that file.
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

Protocol details follow the [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
