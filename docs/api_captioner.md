# API dataset captioning

Open a dataset's **AutoCaption** dialog and choose **API (OpenAI-compatible)**.
The main form contains API URL, optional API key, optional model ID, optional
system prompt, and caption prompt. Enter the exact model ID if your server
requires it. For local servers such as Strata that choose their own model,
leave **Model (optional)** blank: a blank, whitespace-only, or omitted
`model_name_or_path` omits the `model` field from the request. A supplied
nonblank model ID is sent to the server; the toolkit does not discover models
or substitute a default. Non-string model values, including null, are rejected.
The endpoint must support Chat Completions messages containing an `image_url`
data URL; a text-only model cannot caption images.

The API URL accepts either a base such as `http://server:5005/v1` or the full
`http://server:5005/v1/chat/completions` endpoint. Only HTTP and HTTPS are
accepted; embedded credentials, query parameters, and fragments are rejected.
Local endpoints are allowed. HTTPS uses normal certificate validation, and
redirects are rejected so credentials cannot follow a redirect to another host.

**Advanced options** starts collapsed. Click its summary, or focus it with Tab
and press Enter/Space. Maximum response tokens defaults to **4096** and accepts
any positive safe integer; there is no fixed 4096 ceiling. The API server
enforces its model's actual context/output limits. Choose `max_tokens` for
compatible local/legacy servers or `max_completion_tokens` for servers that
require the modern field. The configured value is sent under exactly that
field, with no retry using a different field. A truncated response fails the
job and is not written as a caption. The request format follows the
[Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Optional temperature, Top P, frequency penalty, presence penalty, seed, and up
to four stop sequences are omitted when blank, leaving provider defaults.
Timeout defaults to 300 seconds. Invalid values produce a persistent save
error with details; values are never silently clamped. No provider requests
are made just to save or validate the setup.

## Image normalization

**Normalize image (megapixels)** defaults to **1**. It resizes the image sent to
the provider to approximately that decimal pixel area: 1 megapixel means
1,000,000 pixels. Both larger and smaller images are resized, including
upscaling. Clear the field to keep the original dimensions. A blank field is
stored as `image_megapixels: null`; an omitted setting uses the default 1,
including existing API caption jobs that do not specify it.

```yaml
caption:
  image_megapixels: 1       # normalize to approximately 1 MP
  # image_megapixels: null  # disable normalization
```

EXIF orientation is applied first. Width and height are scaled by the same
square-root area ratio, then rounded to the nearest integer pixel, so aspect
ratio and requested area are preserved up to pixel rounding. Resizing uses
LANCZOS and the in-memory PNG preserves alpha. An unchanged target size skips
resizing. Source files are never rewritten; normalization affects only the
image data sent to the API, and the setting is not a provider parameter.

Values must be positive finite numbers; zero, negatives, booleans and nonnumeric
values are rejected before opening an image or making a provider request.
Unrepresentable pixel areas or targets that round an axis to zero also fail
explicitly rather than being clamped. There is no fixed megapixel cap.
The value, including blank/null, is remembered and restored when reopening,
editing or cloning the caption setup. Click or Tab to the shared numeric field
to edit it; blank disables resizing without a separate toggle.

## Optional caption cleanup

Expand **Caption cleanup** below Caption prompt and enable it to apply a regex
find/replace to the API response before the caption file is written. It is off
by default. Settings are remembered with the rest of the setup and retained
when editing or cloning a job. Disabling cleanup keeps its draft fields while
preserving the existing caption behavior.

For the requested tag extraction, enter:

```text
From: .*<caption>(.*)<\/caption>.*
To:   $1
```

Enable **Dot matches newlines (s)** if the response can contain line breaks.
`/` does not need escaping in Python patterns, so `</caption>` also works.
The enabled form is equivalent to:

```yaml
caption:
  regex_replace:
    enabled: true
    pattern: '.*<caption>(.*)<\/caption>.*'
    replacement: '$1'
    ignore_case: false
    multiline: false
    dotall: true
    verbose: false
    global: true
```

Patterns use **Python `re`**, not JavaScript regex syntax or `/pattern/flags`
delimiters. **Ignore case (i)** ignores letter case. **Multiline (m)** makes
`^` and `$` anchor each line; it does not make `.` match newlines. **Dot matches
newlines (s)** changes `.` matching only and does not turn line breaks into
spaces. **Ignore pattern whitespace/comments (x)** ignores unescaped whitespace
and `#` comments in the pattern, outside character classes; it does not remove
caption whitespace. Replace all matches defaults on; disable it to replace only
the first match. Other flags default off.

Replacement text supports `$1`, `$2`, etc. for existing capture groups, `$&`
for the whole match, `$$` for a literal dollar sign, and `$<name>` for a Python
named group such as `(?P<name>...)`. Numeric references consume the whole number:
`$12` requires group 12 rather than falling back to group 1 plus `2`. `$0` is
invalid; use `$&`. Unmatched optional groups insert empty text. Other dollar
sequences remain literal. Replacement backslashes are always literal, including
`\1` and `\n`; enter an actual newline in To for a line break. Leave To empty
to remove matches. Replacement text is never evaluated as code.

The provider response keeps its existing leading/trailing whitespace trimming
before cleanup. Replacement whitespace is preserved. No match leaves the
caption unchanged. A whitespace-only/empty cleaned result fails without writing
or replacing that image's caption. Credential-echo protection is checked before
and after cleanup. Cleanup settings are not sent to the provider.

The UI/server validate the setting structure when saving. Python compiles the
pattern and validates capture references when the job starts, before any API
request or caption write; invalid syntax therefore stops the job with details.
Disabled drafts are not compiled. Expand the section with a mouse click or
Enter/Space, then use Tab through the labeled From/To fields and switches.

## Remembered setup

Adding a job to the queue remembers the selected captioner and its settings on
the server for the next AutoCaption dialog, including API URL, model, prompts,
response limit, advanced sampling parameters, and the private credential
reference. The key stays private; the password field shows a saved-key hint
instead of returning its value. Canceling the dialog does not save changes.

Remembered settings apply to new jobs, including after a page reload. Editing
or cloning a job loads that job's own configuration. The current dataset path
and an explicitly selected dataset-header caption extension take precedence
over the remembered setup. Job identity and dataset paths are not saved as
defaults. Defaults are saved before the job starts, so a subsequent launch or
provider failure does not discard the setup.

Preferences use the existing server settings table under `CAPTION_DEFAULTS`,
accessed through `/api/caption/settings`; they do not depend on browser storage.
If a saved private key file is missing, the dialog still opens so its reference
can be replaced or cleared. Credential validation remains required when saving
or launching a job.

## Private API keys

A blank key on a new setup means no authentication, useful for local servers.
Entering a key creates an immutable private credential file. A saved job keeps
only its UUID reference; reopening the form never returns the key to the
browser. Enter another key to replace the reference. Select **Use no
authentication** to remove that job's reference. Changing the API URL clears
the reference and typed key, requiring explicit entry for the new endpoint.
Advanced JSON edits that retain a key reference for a different endpoint are
rejected. Keys are bound to the normalized complete endpoint, including path.

Credential files live in `data/caption-api/credentials/<uuid>.json` by default.
Set `AI_TOOLKIT_CAPTION_CREDENTIALS_DIR` consistently for the UI and queue
worker to override this directory. New directories use mode 0700, files use
0600, and reads require current-user ownership and owner-only file permissions.
Persistent keyed setups require a POSIX host; unauthenticated setups do not
need credential files. Protect backups and the host account: these files hold
plaintext keys. Removing/replacing a job reference does not delete immutable
files that other jobs may still reference.

Keys are excluded from job JSON/YAML, list/export responses, browser storage,
logs, and process arguments. The queue worker resolves the reference only for
launch and passes the key in the detached caption child's environment as
`AITK_API_CAPTION_KEY`. Generic file/image routes and assistant file tools block
the private directory, including canonical symlink aliases. A reference copied
to another toolkit instance needs the corresponding private file or a newly
entered key.

## Images, outputs, and queue behavior

API captioning reuses the existing captioner's recursive image discovery,
caption extension, recaption policy, progress tracking, and stop requests.
Supported still-image extensions are PNG, JPG/JPEG, WebP, BMP, GIF, and AVIF,
subject to the installed Pillow decoders. For multi-frame files only the first
frame is sent. The source image is EXIF-oriented and encoded as a PNG data URL;
the source file is unchanged. Captions are written beside each image using the
selected extension. Existing nonempty captions are skipped unless Recaption is enabled.
Existing prompt references such as `<caption.txt>` remain supported.

These jobs appear in the **CPU/API** queue. They can run alongside GPU training
or saved-job sample rendering without taking a GPU lock. The Python process
loads no captioning weights or CUDA context. A local API server may itself use
a GPU; its resource usage is outside this queue's control. Other captioners and
GPU queue behavior remain unchanged.

API-only jobs resolve the built-in API captioner directly during Python startup.
They do not discover unrelated GPU model extensions or load other captioner
implementations. When `run.py` receives only API caption configs, it also skips
PyTorch, Accelerate, Diffusers, training imports, and CUDA device checks. This
applies with `--log`, `--name`, `--recover`, `SEED`, and `DEBUG_TOOLKIT` set. A
later GPU job in the same command initializes the usual torch runtime before
loading its models, including unified-memory patches, debug mode, seeding, and
Accelerate. Mixed-process and other extension jobs retain their existing
registration behavior.

In a local empty-dataset CLI check through `conda run`, API startup fell from
7.33 seconds and about 921 MiB peak resident memory to 0.96 seconds and about
50 MiB. This measures Python startup and job setup with no images or provider
request; actual caption time also depends on image encoding and API latency.

Authentication, connection, server, malformed-response, refusal, and truncation
errors stop the job and remain visible in job status/logs. Provider response
bodies are excluded from errors because they may echo secrets. Successfully
written earlier captions remain on disk. Stop is checked before and after each
request; an in-flight HTTP request is bounded by the configured timeout rather
than being resumed or retried in the background.

Verification passed 11 Node API-caption tests and all 54 Python transport/startup
tests, including megapixel defaults, fractional up/down resizing, blank/null
persistence, EXIF orientation, alpha preservation, unchanged source bytes,
invalid inputs, cleanup and credential guards. Coverage includes a configured
16,384-token request. The CPU-startup regression runs the real
`run.py` entry point with `CUDA_VISIBLE_DEVICES` empty, a synthetic image, and
a local fake provider. It checks the HTTP request and the cleaned caption written
to disk, plus preservation of an existing sidecar when cleanup produces empty text,
alongside the boundary between API-only and ordinary extension resolution.
Tests used fixture credentials and no real provider requests.

UI source and worker TypeScript checks and the production build passed. The
build used a temporary 12 GiB Node heap allowance inside a 16 GiB/no-swap
transient systemd unit, peaking at 14.2 GiB. Source checking excludes existing
generated Next route-parameter errors; no new type suppression was added.
The existing optional macOS temperature-sensor warning remains on Linux.

The browser regression uses an isolated synthetic SQLite database/datasets and
software-only Chromium. It saves From/To plus all modifier settings through the
real local job/settings endpoints, then reopens setup on another dataset and
checks that every setting is restored. No caption queue worker or provider ran
for this browser check. Python, Node and browser tests ran in the prescribed
16 GiB/no-swap systemd wrappers. The mockup and screenshot are under
`tmp/caption-cleanup`; reusable scripts are
`playwright_scripts/prepare_api_caption_regex_e2e.py` and
`playwright_scripts/api-caption-regex.e2e.cjs`.

The image-normalization browser regression saves blank as null, reopens it as
blank on a second synthetic dataset, saves 0.25 MP, and restores that value on
a third dataset. It passed on a diagnostic retry after an initial 15-second
API-response timeout whose endpoint was not captured; no production change or
fixture reset was needed. It queued two synthetic jobs and made no provider
requests. The script is `playwright_scripts/api-caption-megapixels.e2e.cjs`,
with screenshots and build logs under `tmp/caption-megapixels`.

Only the UI branch was reloaded to activate these controls. The existing queue
worker remained running with the same PID; the public datasets page returned
HTTP 200. The isolated browser-test server was stopped after verification.
