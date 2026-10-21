# API dataset captioning

Open a dataset's **AutoCaption** dialog and choose **API (OpenAI-compatible)**.
The main form contains API URL, optional API key, model ID, optional system
prompt, and caption prompt. Use the exact model ID exposed by your server.
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
implementations. Mixed-process and other extension jobs retain their existing
registration behavior.

Authentication, connection, server, malformed-response, refusal, and truncation
errors stop the job and remain visible in job status/logs. Provider response
bodies are excluded from errors because they may echo secrets. Successfully
written earlier captions remain on disk. Stop is checked before and after each
request; an in-flight HTTP request is bounded by the configured timeout rather
than being resumed or retried in the background.

Verification passed 27 Python caption tests and 10 Node API-caption tests,
including a configured 16,384-token request, saved-settings round trips, and
credential-reference recovery. The CPU-startup regression runs the real
`run.py` entry point with `CUDA_VISIBLE_DEVICES` empty, a synthetic image, and
a local fake provider. It checks the HTTP request and written caption,
alongside the boundary between API-only and ordinary extension resolution.
Tests used fixture credentials and no real provider requests.

UI source and worker TypeScript checks and the production build passed. The
build used a temporary 16 GiB Node heap allowance. Only the UI was restarted;
the queue worker stayed running. The jobs page and remembered-settings endpoint
passed live HTTP checks after activation. Browser interaction checks were
unavailable because no browser was connected.
