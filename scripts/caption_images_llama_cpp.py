#!/usr/bin/env python3
"""Caption images with a local, vision-capable llama.cpp server.

Usage:
    conda run -n ai-toolkit python scripts/caption_images_llama_cpp.py /path/to/images
    conda run -n ai-toolkit python scripts/caption_images_llama_cpp.py /path/to/images --recursive --overwrite

Uses Python's standard library for HTTP and Pillow to convert WebP to PNG in
memory. The original images are unchanged. Captions are saved beside each image
as UTF-8 text: image.jpg -> image.txt. Existing captions are skipped by default.
The server must have a vision-capable model and its multimodal projector loaded.
"""

import argparse
import base64
import io
import json
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps


IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
DEFAULT_PROMPT = (
    "Write a descriptive caption for this image. Describe the main subjects, "
    "their appearance and actions, the setting, composition, lighting, colors, "
    "and visual style. Describe only what is visible, without speculation. "
    "Return only the caption, with no introduction, headings, or quotation marks."
)


def image_data_url(image_path: Path) -> str:
    """Encode an image, converting WebP for llama.cpp's image decoder."""
    mime_type = IMAGE_MIME_TYPES[image_path.suffix.lower()]
    if image_path.suffix.lower() == ".webp":
        with Image.open(image_path) as image:
            buffer = io.BytesIO()
            ImageOps.exif_transpose(image).convert("RGBA").save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
        mime_type = "image/png"
    else:
        image_bytes = image_path.read_bytes()
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def generate_caption(image_path: Path, args: argparse.Namespace) -> str:
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url(image_path)},
                    },
                ],
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    request = Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {args.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Server returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach {args.base_url}: {exc.reason}") from exc

    try:
        choice = result["choices"][0]
        caption = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Server response has no assistant caption.") from exc
    if choice.get("finish_reason") == "length":
        raise RuntimeError("Caption was truncated; increase --max-tokens and retry.")
    if not isinstance(caption, str) or not caption.strip():
        raise RuntimeError("Server returned an empty caption.")
    return caption.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path, help="Folder containing PNG/JPG/JPEG/WebP images")
    parser.add_argument(
        "--base-url", default="http://localhost:5005/v1",
        help="Server API base URL (default: http://localhost:5005/v1)",
    )
    parser.add_argument("--api-key", default="dummy", help="API key (default: dummy)")
    parser.add_argument("--model", default="dummy-model", help="Model name (default: dummy-model)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Custom caption prompt")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Output token limit (default: 1024)")
    parser.add_argument("--timeout", type=float, default=300, help="Request timeout in seconds")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing captions")
    args = parser.parse_args()

    if not args.folder.is_dir():
        parser.error(f"Not a directory: {args.folder}")
    if args.max_tokens <= 0 or args.timeout <= 0:
        parser.error("--max-tokens and --timeout must be greater than zero")

    paths = args.folder.rglob("*") if args.recursive else args.folder.iterdir()
    images = sorted(
        path for path in paths
        if path.is_file() and path.suffix.lower() in IMAGE_MIME_TYPES
    )
    # Different extensions with the same stem cannot have separate .txt captions.
    destinations = {}
    for image_path in images:
        text_path = image_path.with_suffix(".txt")
        if text_path in destinations:
            parser.error(
                f"{destinations[text_path]} and {image_path} both map to {text_path}; "
                "rename one image before captioning."
            )
        destinations[text_path] = image_path

    if not images:
        print(f"No supported images found in {args.folder}.")
        return 0

    saved = skipped = failed = 0
    for index, image_path in enumerate(images, start=1):
        text_path = image_path.with_suffix(".txt")
        prefix = f"[{index}/{len(images)}]"
        if text_path.exists() and not args.overwrite:
            print(f"{prefix} Skipping {image_path}: caption already exists.", flush=True)
            skipped += 1
            continue
        print(f"{prefix} Captioning {image_path}...", flush=True)
        try:
            caption = generate_caption(image_path, args)
            text_path.write_text(caption + "\n", encoding="utf-8")
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"{prefix} Failed {image_path}: {exc}", file=sys.stderr, flush=True)
            failed += 1
            continue
        print(f"{prefix} Saved {text_path}", flush=True)
        saved += 1

    print(f"Done: {saved} saved, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
