"""Small, non-streaming OpenAI-compatible image-caption transport. No model loads."""
import base64
import io
import json
import math
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "gif", "avif"}
DEFAULT_PROMPT = (
    "Write a descriptive caption for this image. Describe the main subjects, "
    "their appearance and actions, the setting, composition, lighting, colors, "
    "and visual style. Describe only what is visible, without speculation. "
    "Return only the caption, with no introduction, headings, or quotation marks."
)


def validate_regex_replace(settings):
    """Compile optional Python regex and strict dollar-style replacement tokens.

    Replacement backslashes are literal: never pass user text to re's replacement
    parser or evaluate it. Disabled settings retain drafts without compilation.
    """
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise ValueError("Caption regex_replace must be an object")
    flags = {'ignore_case': re.IGNORECASE, 'multiline': re.MULTILINE,
             'dotall': re.DOTALL, 'verbose': re.VERBOSE}
    if set(settings) - {'enabled', 'pattern', 'replacement', 'global', *flags}:
        raise ValueError("Unsupported caption regex replacement setting")
    for name in ('enabled', 'global', *flags):
        if type(settings.get(name, name == 'global')) is not bool:
            raise ValueError(f"Caption regex {name} must be a boolean")
    for name in ('pattern', 'replacement'):
        if not isinstance(settings.get(name, ''), str):
            raise ValueError(f"Caption regex {name} must be text")
    if not settings.get('enabled', False):
        return None
    if not settings.get('pattern'):
        raise ValueError("Caption regex From pattern is required when cleanup is enabled")
    mask = 0
    for name, flag in flags.items():
        if settings.get(name, False):
            mask |= flag
    try:
        pattern = re.compile(settings['pattern'], mask)
    except (re.error, ValueError) as error:
        raise ValueError(f"Invalid caption regex From pattern: {error}") from error
    replacement, parts, literal, index = settings.get('replacement', ''), [], [], 0
    while index < len(replacement):
        if replacement[index] != '$' or index + 1 == len(replacement):
            literal.append(replacement[index])
            index += 1
            continue
        following = replacement[index + 1]
        if following == '$':
            literal.append('$')
            index += 2
            continue
        group = None
        if following == '&':
            group, end = 0, index + 2
        elif following in '0123456789':
            end = index + 2
            while end < len(replacement) and replacement[end] in '0123456789':
                end += 1
            group = int(replacement[index + 1:end])
            if following == '0' or not 1 <= group <= pattern.groups:
                raise ValueError(f"Invalid caption regex replacement capture ${replacement[index + 1:end]}; use an existing group or $& for the whole match")
        elif following == '<':
            end = replacement.find('>', index + 2)
            if end < 0:
                raise ValueError("Invalid caption regex replacement: named capture must end with >")
            group, end = replacement[index + 2:end], end + 1
            if group not in pattern.groupindex:
                raise ValueError(f"Invalid caption regex replacement named capture $<{group}>")
        else:
            literal.append('$')
            index += 1
            continue
        if literal:
            parts.append(('literal', ''.join(literal)))
            literal = []
        parts.append(('group', group))
        index = end
    if literal:
        parts.append(('literal', ''.join(literal)))
    return pattern, parts


def apply_regex_replace(caption, settings):
    compiled = validate_regex_replace(settings)
    if compiled is None:
        return caption
    pattern, parts = compiled
    result = pattern.sub(lambda match: ''.join(value if kind == 'literal' else match.group(value) or ''
                                               for kind, value in parts), caption,
                         count=0 if settings.get('global', True) else 1)
    if not result.strip():
        raise ValueError("Caption regex cleanup produced an empty caption; caption not saved")
    return result


def validate_api_settings(config):
    url = config.get("api_url", "")
    parsed = urlsplit(url)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("API URL must be an HTTP(S) base URL without credentials, query, or fragment")
    # Accessing port also validates malformed/out-of-range ports.
    parsed.port
    if not isinstance(config.get("model_name_or_path"), str) or not config["model_name_or_path"].strip():
        raise ValueError("API model ID is required")
    limit = config.get("max_new_tokens", 4096)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 9_007_199_254_740_991:
        raise ValueError("Maximum response tokens must be a positive safe integer")
    field = config.get("api_max_tokens_field", "max_tokens")
    if field not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("Unsupported response token limit field")
    for name, low, high in (("temperature", 0, 2), ("top_p", 0, 1),
                            ("frequency_penalty", -2, 2), ("presence_penalty", -2, 2)):
        value = config.get(name)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not low <= value <= high):
            raise ValueError(f"{name} must be a finite number from {low} to {high}")
    timeout = config.get("api_timeout", 300)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Request timeout must be a positive finite number of seconds")
    seed = config.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("Seed must be an integer")
    stops = config.get("stop")
    if stops is not None and (not isinstance(stops, list) or len(stops) > 4
                             or any(not isinstance(item, str) or not item for item in stops)):
        raise ValueError("Stop sequences must contain at most four nonempty strings")
    for name in ("system_prompt", "caption_prompt"):
        if config.get(name) is not None and not isinstance(config[name], str):
            raise ValueError(f"{name} must be text")
    extension = config.get("caption_extension", "txt")
    if not isinstance(extension, str) or not extension or any(not (part.isalnum() or part in "_-") for part in extension):
        raise ValueError("Caption extension must contain only letters, numbers, underscores, or hyphens")
    extensions = config.get("extensions", sorted(IMAGE_EXTENSIONS))
    if not isinstance(extensions, list) or not extensions or any(ext not in IMAGE_EXTENSIONS for ext in extensions):
        raise ValueError("API captioning supports still-image extensions only")
    validate_regex_replace(config.get('regex_replace'))
    api_path = parsed.path.rstrip("/")
    if not api_path.endswith("/chat/completions"):
        api_path += "/chat/completions"
    return urlunsplit((parsed.scheme, parsed.netloc, api_path, "", ""))


class NoCaptionRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Caption API redirects are disabled; configure the final API URL")


def image_data_url(filename):
    with Image.open(filename) as image:
        image.seek(0)
        normalized = ImageOps.exif_transpose(image).convert("RGBA")
        output = io.BytesIO()
        normalized.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def generate_api_caption(filename, config, api_key="", opener=None):
    endpoint = validate_api_settings(config)
    messages = []
    if (config.get("system_prompt") or "").strip():
        messages.append({"role": "system", "content": config["system_prompt"]})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": config.get("caption_prompt") or DEFAULT_PROMPT},
        {"type": "image_url", "image_url": {"url": image_data_url(filename)}},
    ]})
    payload = {"model": config["model_name_or_path"], "messages": messages, "stream": False,
               config.get("api_max_tokens_field", "max_tokens"): config.get("max_new_tokens", 4096)}
    for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "seed", "stop"):
        if config.get(name) is not None:
            payload[name] = config[name]
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        if "\n" in api_key or "\r" in api_key:
            raise ValueError("API credential contains invalid characters")
        headers["Authorization"] = "Bearer " + api_key
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with (opener or build_opener(NoCaptionRedirects())).open(request, timeout=config.get("api_timeout", 300)) as response:
            data = response.read(16 * 1024 * 1024 + 1)
            if len(data) > 16 * 1024 * 1024:
                raise RuntimeError("Caption API response exceeded the 16 MiB limit")
            result = json.loads(data)
    except HTTPError as error:
        # Never copy provider-controlled response bodies or echoed headers into logs.
        raise RuntimeError(f"Caption API HTTP {error.code}; check the API URL, key, model, and server logs") from None
    except URLError:
        raise RuntimeError("Cannot reach the caption API; check its URL, certificate, and availability") from None
    except (ValueError, TimeoutError) as error:
        raise RuntimeError("Caption API returned invalid JSON or the request timed out") from None
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError("Caption API returned an error response")
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("Caption API must return exactly one assistant choice")
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Caption API response was truncated at the configured token limit; caption not saved")
    if choice.get("finish_reason") != "stop":
        raise RuntimeError("Caption API did not complete normally; caption not saved")
    message = choice.get("message")
    caption = message.get("content") if isinstance(message, dict) else None
    if not isinstance(caption, str) or not caption.strip() or message.get("refusal"):
        raise RuntimeError("Caption API returned an empty or refused caption")
    if api_key and api_key in caption:
        raise RuntimeError("Caption API echoed its credential; caption withheld")
    caption = apply_regex_replace(caption.strip(), config.get('regex_replace'))
    if api_key and api_key in caption:
        raise RuntimeError("Caption cleanup contains the API credential; caption withheld")
    return caption
