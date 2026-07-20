import json
import re
import ast
import itertools
from math import gcd
from collections import OrderedDict
from typing import Optional

from PIL import Image

from .Qwen3VLCaptioner import Qwen3VLCaptioner
from .prompts.ideogram4_caption_prompt import ideogram4_caption_prompt
from toolkit.ideogram_caption import normalize_caption_dict, swap_bbox_xy_in_text
import transformers
import logging
import warnings

# transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# The deconstruction JSON is long. The base 128-token default truncates it badly;
# 1,536 tokens preserves several complete unique elements, while the partial-JSON
# salvage path prevents repeated-element runaways from consuming the whole batch.
MIN_NEW_TOKENS = 1536

# Largest denominator allowed when snapping a real image's aspect ratio to a
# clean W:H. Keeps captions in the same small-denominator ratio distribution the
# generator was trained on, instead of ugly fractions like 1023:768.
MAX_AR_DENOMINATOR = 16


class Ideogram4Captioner(Qwen3VLCaptioner):
    def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
        super(Ideogram4Captioner, self).__init__(process_id, job, config, **kwargs)
        if self.caption_config.max_new_tokens < MIN_NEW_TOKENS:
            print(
                f"[Ideogram4Captioner] Raising max_new_tokens "
                f"{self.caption_config.max_new_tokens} -> {MIN_NEW_TOKENS} "
                f"(the deconstruction JSON is long)."
            )
            self.caption_config.max_new_tokens = MIN_NEW_TOKENS

    def compute_aspect_ratio(self, width: int, height: int) -> str:
        """Return a clean 'W:H' string for the image, snapped to a small
        denominator so it matches the generator's ratio distribution."""
        if width <= 0 or height <= 0:
            return "1:1"
        g = gcd(width, height)
        rw, rh = width // g, height // g
        # Already clean enough.
        if rw <= MAX_AR_DENOMINATOR and rh <= MAX_AR_DENOMINATOR:
            return f"{rw}:{rh}"
        # Otherwise find the closest p:q (q <= MAX_AR_DENOMINATOR) to the true ratio.
        target = width / height
        best = None
        for q in range(1, MAX_AR_DENOMINATOR + 1):
            p = max(1, round(target * q))
            err = abs(p / q - target)
            if best is None or err < best[0]:
                best = (err, p, q)
        return f"{best[1]}:{best[2]}"

    def build_prompt(self, aspect_ratio: str) -> str:
        # caption_prompt is the user-editable ADDITIONAL INSTRUCTIONS block,
        # injected into the fixed system prompt (not the whole prompt).
        user_instructions = (self.caption_config.caption_prompt or "").strip()
        if not user_instructions:
            user_instructions = "None."
        prompt = ideogram4_caption_prompt.replace("{{aspect_ratio}}", aspect_ratio)
        prompt = prompt.replace("{{user_instructions}}", user_instructions)
        return prompt

    def _extract_json(self, raw: str) -> Optional[dict]:
        """Pull the JSON object out of the model output, tolerating fences and
        stray preamble. Returns the parsed dict or None."""
        text = raw.strip()
        # Strip ```json ... ``` fences if present.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # Fall back to the outermost {...} span.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
        candidate = re.sub(
            r'("high_level_description"\s*:\s*")\'(.*?)\',\s*[\'\"]style_description[\'\"]\s*:',
            lambda match: match.group(1) + match.group(2).replace('"', '\\"') + '","style_description":',
            candidate,
            count=1,
            flags=re.DOTALL,
        )
        candidate = re.sub(r'(?m)^(\s*)=\s*("#[0-9A-Fa-f]{6}")', r'\1\2', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(candidate)
                return value if isinstance(value, dict) else None
            except (ValueError, SyntaxError):
                for length in range(1, 4):
                    for suffix in itertools.product(("}", "]"), repeat=length):
                        repaired = candidate + "".join(suffix)
                        for parser in (json.loads, ast.literal_eval):
                            try:
                                value = parser(repaired)
                                if isinstance(value, dict):
                                    return value
                            except (ValueError, SyntaxError, json.JSONDecodeError):
                                pass
                return None

    def _extract_partial_json(self, raw: str) -> Optional[dict]:
        """Salvage complete leading fields/elements from a truncated generation.

        Qwen occasionally repeats one element until max_new_tokens and therefore
        omits the final array/object delimiters. The complete prefix is still
        useful. Parse only independently complete JSON values, deduplicate exact
        repeated elements, and fail closed if the required scene fields are not
        all recoverable.
        """
        text = raw.strip()
        fence = re.search(r"```(?:json)?\s*(.*)", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        decoder = json.JSONDecoder()

        def value_after(key: str):
            match = re.search(rf'"{re.escape(key)}"\s*:\s*', text)
            if not match:
                raise ValueError(key)
            value, _ = decoder.raw_decode(text, match.end())
            return value

        try:
            high_level = value_after("high_level_description")
            style = value_after("style_description")
            background = value_after("background")
            marker = re.search(r'"elements"\s*:\s*\[', text)
            if not marker:
                return None
            position = marker.end()
            elements = []
            seen = set()
            while position < len(text) and len(elements) < 24:
                while position < len(text) and (text[position].isspace() or text[position] == ","):
                    position += 1
                if position >= len(text) or text[position] == "]":
                    break
                try:
                    element, end = decoder.raw_decode(text, position)
                except json.JSONDecodeError:
                    break
                position = end
                if not isinstance(element, dict):
                    continue
                fingerprint = json.dumps(element, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                elements.append(element)
            if not isinstance(style, dict) or not isinstance(background, str) or not elements:
                return None
            return {
                "high_level_description": high_level,
                "style_description": style,
                "compositional_deconstruction": {
                    "background": background,
                    "elements": elements,
                },
            }
        except (ValueError, json.JSONDecodeError):
            return None

    def _convert_bbox(self, bbox):
        """Qwen3-VL emits NORMALIZED 0-1000 boxes in [x1,y1,x2,y2] order (verified
        empirically: coords are stable across input resolution). Our stored
        format is also 0-1000 but in [y1,x1,y2,x2] order, so this only reorders
        and clamps -- no pixel scaling. Returns the box or None to drop it."""
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        x1, x2 = sorted((max(0, min(1000, round(x1))), max(0, min(1000, round(x2)))))
        y1, y2 = sorted((max(0, min(1000, round(y1))), max(0, min(1000, round(y2)))))
        if y2 <= y1 or x2 <= x1:
            return None
        # stored order is [y1, x1, y2, x2]
        return [y1, x1, y2, x2]

    def _normalize_caption(self, data: dict) -> dict:
        """Cleanup the parsed caption before storage. The model emits bboxes in
        [x1,y1,x2,y2]; convert each to our stored [y1,x1,y2,x2] order, then hand off
        to the shared normalizer for the rest: drop aspect_ratio, enforce the
        photo/art_style branch and key order, canonicalize medium, and cap/uppercase
        color palettes (16 per image, 5 per element)."""
        style = data.get("style_description")
        if not isinstance(style, dict):
            style = {}
            data["style_description"] = style
        medium = str(style.get("medium") or "illustration")
        style.setdefault("aesthetics", "coherent observed detail and clear visual hierarchy")
        style.setdefault("lighting", "observed ambient and directional illumination")
        style["medium"] = medium
        if medium == "photograph":
            style.setdefault("photo", "observed documentary or editorial photography")
            style.pop("art_style", None)
        else:
            style.setdefault("art_style", "observed rendered visual style")
            style.pop("photo", None)

        decon = data.get("compositional_deconstruction", {})
        elements = decon.get("elements", []) if isinstance(decon, dict) else []
        if isinstance(elements, list):
            unique = []
            seen = set()
            for el in elements:
                if not isinstance(el, dict):
                    continue
                aliases = {
                    "typetype": "type", "typ": "type",
                    "box": "bbox", "boundingbox": "bbox", "bounding_box": "bbox",
                    "asc": "desc", "description": "desc",
                    "palette": "color_palette", "colors": "color_palette",
                }
                cleaned = {}
                for key, value in el.items():
                    normalized_key = re.sub(r"^[^A-Za-z]+", "", re.sub(r"\s+", "", str(key).strip()))
                    cleaned[aliases.get(normalized_key.lower(), normalized_key)] = value
                el = cleaned
                kind = str(el.get("type") or "").strip().lower()
                if kind != "text":
                    el["type"] = "text" if isinstance(el.get("text"), str) and el["text"].strip() else "obj"
                if el.get("type") == "text" and isinstance(el.get("text"), str):
                    el["text"] = " | ".join(part.strip() for part in re.split(r"[\r\n]+", el["text"]) if part.strip())
                    if not el["text"]:
                        continue
                    el.setdefault("desc", f"Visible text reading {el['text']!r}")
                elif el.get("type") == "obj":
                    el.setdefault("desc", "Observed foreground object")
                if "bbox" in el:
                    cleaned = self._convert_bbox(el["bbox"])
                    if cleaned is None:
                        el.pop("bbox", None)
                    else:
                        el["bbox"] = cleaned
                fingerprint = json.dumps(el, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if fingerprint in seen:
                    continue
                seen.add(fingerprint)
                unique.append(el)
            decon["elements"] = unique
        normalized = normalize_caption_dict(data)
        # The compact prompt contract forbids overlapping text boxes. Keep the
        # transcription, but drop the less reliable later box when Qwen assigns
        # two text regions to the same pixels.
        text_boxes = []
        for element in normalized.get("compositional_deconstruction", {}).get("elements", []):
            if element.get("type") != "text" or not element.get("bbox"):
                continue
            box = element["bbox"]
            overlaps = any(
                max(box[0], previous[0]) < min(box[2], previous[2])
                and max(box[1], previous[1]) < min(box[3], previous[3])
                for previous in text_boxes
            )
            if overlaps:
                element.pop("bbox", None)
            else:
                text_boxes.append(box)
        return normalized

    def get_caption_for_file(self, file_path: str) -> Optional[str]:
        try:
            # Read true dimensions before any resize so the aspect ratio is exact.
            with Image.open(file_path) as probe:
                width, height = probe.size
            aspect_ratio = self.compute_aspect_ratio(width, height)

            img = self.load_pil_image(file_path, max_res=self.caption_config.max_res)
            prompt = self.build_prompt(aspect_ratio)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device_torch)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.caption_config.max_new_tokens,
                repetition_penalty=1.08,
                no_repeat_ngram_size=6,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            data = self._extract_json(output_text)
            if data is None:
                data = self._extract_partial_json(output_text)
                if data is None:
                    print(
                        f"[IdeogramCaptioner] Could not parse or salvage JSON for {file_path}; "
                        f"saving raw output with regex-adapted bboxes."
                    )
                    # JSON is malformed so we can't swap bboxes per-element. Adapt them
                    # directly in the raw text instead, so the boxes still render right.
                    return swap_bbox_xy_in_text(output_text)
                print(f"[IdeogramCaptioner] Salvaged truncated JSON for {file_path}.")

            data = self._normalize_caption(data)
            # Store pretty JSON for QC/editing; the dataloader minifies at load.
            return json.dumps(data, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
