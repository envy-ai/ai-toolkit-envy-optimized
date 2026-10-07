# Adapted from Tencent HunyuanImage-3.0 and PedroMarinhoDev/ComfyUI-HunyuanImage3.
# Tencent source revision: 2ec2c78bee7d4b94157341fba86c4c2c7b1858b2.
# Pedro source revision: 84ad3a3e2a54472e69e195269729f774b242d120.
# See NOTICE and LICENSE files in this directory.
"""Input sequence construction for HunyuanImage-3.0: prompt + image_size -> token ids.

Ported from the reference's `HunyuanImage3TokenizerFast.apply_chat_template`, `encode_general`,
`encode_sequence`, `ImageInfo` and `ResolutionGroup`. Only the path this port needs is implemented —
text-to-image with an explicit `image_size`, i.e. the reference's `mode="gen_image"`,
`bot_task="image"`, `sequence_template="instruct"`, with this checkpoint's `cfg_distilled` and
`use_meanflow` flags (which is what adds the `<guidance>` and `<timestep_r>` meta tokens). The
reference's section machinery covers many other task shapes (CoT, inpainting, joint VAE+ViT
conditioning, CFG duplication); none of them are reachable from this port, so they are not ported.

Text is encoded with `tokenizers.Tokenizer` (the repo's sanctioned path for a repo-shipped BPE
tokenizer, see `comfy/text_encoders/ace.py`) instead of the reference's transformers tokenizer, and
`add_special_tokens=False` matters: the tokenizer's post-processor would otherwise insert its own
bos/eos, and the reference explicitly disables that. The single leading `<bos>` is ours.

The layout, verified against the reference implementation's own output (5348 ids for a fixed prompt
at 1024x1024, `step6a_port_vs_reference.py`):

    <bos> <system prompt> "\\n\\n" "User: " <prompt> "\\n\\n" "Assistant: " <answer>
    <boi> <img_size_1024> <img_ratio_16> <timestep> <guidance> <timestep_r> <img> x 4096 <eoi>
    </answer> "\\n\\n"

There is no trailing eos: the reference's t2i path passes `add_eos=False`.
"""
import re

import torch

# Meta tokens, as the reference names them (tokenization_hunyuan_image_3.py:567-600). Ids are resolved
# from the tokenizer itself, never hard-coded and never taken from the config's legacy fields.
META_TOKENS = {
    "bos": "<|startoftext|>",
    "boi": "<boi>",
    "eoi": "<eoi>",
    "img": "<img>",
    "joint_img_sep": "<joint_img_sep>",
    "timestep": "<timestep>",
    "timestep_r": "<timestep_r>",
    "guidance": "<guidance>",
    "answer": "<answer>",
    "end_of_answer": "</answer>",
    # the CoT stage's opening and closing tags: the text stage is told which one to start by, and stops
    # at the matching close (spec §49)
    "think": "<think>",
    "end_of_think": "</think>",
    "recaption": "<recaption>",
    "end_of_recaption": "</recaption>",
    # the unconditional rendering's stand-in for each token of user and CoT text
    "cfg": "<cfg>",
}

# image_processor.py:147-155
EXTRA_RESOLUTIONS = ((1024, 768), (1280, 720), (768, 1024), (720, 1280))

VAE_SPATIAL_FACTOR = 16          # vae_downsample_factor * patch_size
SEPARATOR = "\n\n"
USER_PREFIX = "User: "
ASSISTANT_PREFIX = "Assistant: "


class Affixes:
    """The role scaffolding a `sequence_template` puts around each section.

    `apply_general_template` (tokenization:1367-1383) sets all five affixes to the empty string when
    `sequence_template == "pretrain"`, and promotes `<answer>`/`</answer>` only for the `"instruct"`
    rendering. The base checkpoint's `generation_config.json` says `pretrain`; the Instruct's and the
    Instruct-Distil's say `instruct`. Rendering one under the other's template is silent — the sequence is
    well-formed either way and only the model notices — so the template has to travel with the
    checkpoint rather than being assumed.
    """

    __slots__ = ("system_suffix", "user_prefix", "user_suffix", "bot_prefix", "bot_suffix", "answer")

    def __init__(self, sequence_template):
        if sequence_template not in ("instruct", "pretrain"):
            raise ValueError(f"unknown sequence_template {sequence_template!r}; the checkpoints use "
                             "'instruct' (Instruct, Instruct-Distil) or 'pretrain' (base)")
        instruct = sequence_template == "instruct"
        self.system_suffix = SEPARATOR if instruct else ""
        self.user_prefix = USER_PREFIX if instruct else ""
        self.user_suffix = SEPARATOR if instruct else ""
        self.bot_prefix = ASSISTANT_PREFIX if instruct else ""
        self.bot_suffix = SEPARATOR if instruct else ""
        self.answer = instruct


# the checkpoint's `max_position_embeddings`; the node passes the model's own value, this is the
# reference's for callers that do not have a model to hand
DEFAULT_MAX_POSITION_EMBEDDINGS = 22800


class Resolution:
    def __init__(self, height, width):
        self.height = self.h = height
        self.width = self.w = width
        self.ratio = height / width

    def __repr__(self):
        return f"{self.height}x{self.width}"


class ResolutionGroup:
    """The reference's aspect-ratio table: base_size x base_size plus a diagonal staircase of
    resolutions between base/2 and base*2, sorted by ratio, plus the four extra resolutions.

    For base_size 1024 / align 16 / step 64 that is 37 entries, which is why the ratio tokens run up
    to `<img_ratio_36>`.
    """

    def __init__(self, base_size, align=16, extra_resolutions=EXTRA_RESOLUTIONS):
        assert base_size % align == 0, f"base_size {base_size} is not divisible by align {align}"
        self.base_size = base_size
        self.align = align
        self.step = base_size // 16
        # the staircase is sorted by ratio and the extra resolutions are appended afterwards, without
        # re-sorting: that is what the reference does, and it is why 1024x1024 is row 16 of 37 rather
        # than the middle of a fully sorted table
        data = sorted(self._calc_by_step(), key=lambda resolution: resolution.ratio)
        for height, width in extra_resolutions:
            if all(resolution.ratio != height / width for resolution in data):
                data.append(Resolution(height, width))
        self.data = data
        self.ratios = [resolution.ratio for resolution in self.data]

    def _calc_by_step(self):
        low, high = self.base_size // 2, self.base_size * 2
        resolutions = [Resolution(self.base_size, self.base_size)]
        height, width = self.base_size, self.base_size
        while not (height >= high and width <= low):
            height = min(height + self.step, high)
            width = max(width - self.step, low)
            resolutions.append(Resolution(height // self.align * self.align, width // self.align * self.align))
        height, width = self.base_size, self.base_size
        while not (height <= low and width >= high):
            height = max(height - self.step, low)
            width = min(width + self.step, high)
            resolutions.append(Resolution(height // self.align * self.align, width // self.align * self.align))
        return resolutions

    def nearest_index(self, width, height):
        ratio = height / width
        return min(range(len(self.ratios)), key=lambda index: abs(self.ratios[index] - ratio))

    def get_target_size(self, width, height):
        resolution = self.data[self.nearest_index(width, height)]
        return resolution.width, resolution.height

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def parse_image_size(image_size):
    """`"WxH"`, `"768:1024"` (also width:height, as in the reference), `"<img_ratio_16>"`, or a
    `(height, width)` pair -> `(height, width)`.

    The `x` form is width-first because that is how every label in this port reads — the node's preset
    list, the reference's own docs, and what a user types. The `(height, width)` pair is the reverse
    because that is the order the rest of this file and the reference's `Resolution` use. Sizes are not
    snapped here; `ImageGeometry` decides the grid.
    """
    if isinstance(image_size, str):
        if image_size.startswith("<img_ratio_"):
            return None, int(image_size.split("_")[-1].rstrip(">"))
        if "x" in image_size:
            width, height = (int(part) for part in image_size.split("x"))
            return height, width
        if ":" in image_size:
            width, height = (int(part) for part in image_size.split(":"))
            return height, width
        raise ValueError(f"image_size should be 'WxH', 'W:H' or '<img_ratio_i>', got {image_size!r}")
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return int(image_size[0]), int(image_size[1])
    raise ValueError(f"image_size should be a string or a (height, width) pair, got {image_size!r}")


class ImageGeometry:
    """The token grid, and the ratio token that hints at it — resolved independently.

    That separation is the point of this class. RoPE is built from the *token grid*, and the grid comes
    from the requested size rounded to the VAE's spatial factor, so 1152x896 generates at 1152x896
    rather than being snapped onto a nearby table row. `<img_ratio_N>` is a semantic hint: the nearest
    row of the reference's table by aspect ratio. Nothing derives geometry from it (verified in Step 6a,
    and relied on by Addendum 29), which is why the two are allowed to disagree.

    `extra_rows=False` is the base checkpoint's table: its processor (`tokenizer_wrapper.ResolutionGroup`)
    has only the 33-row staircase, and its vocabulary stops at `<img_ratio_32>` — the four extra rows and
    their tokens arrived with the Instruct release. The staircase rows come first in both tables, so they
    share indices.
    """

    def __init__(self, image_size, base_size=None, extra_rows=True):
        height, width = parse_image_size(image_size)
        if base_size is None:
            base_size = base_size_for(height, width)
        self.group = ResolutionGroup(base_size, extra_resolutions=EXTRA_RESOLUTIONS if extra_rows else ())
        if height is None:                       # "<img_ratio_i>" names a row of the table directly
            ratio_index = width                  # capture the index before it is shadowed below
            resolution = self.group[ratio_index]
            height, width = resolution.height, resolution.width
        else:
            ratio_index = self.group.nearest_index(width, height)

        self.requested = (height, width)
        self.height = height // VAE_SPATIAL_FACTOR * VAE_SPATIAL_FACTOR
        self.width = width // VAE_SPATIAL_FACTOR * VAE_SPATIAL_FACTOR
        if self.height < VAE_SPATIAL_FACTOR or self.width < VAE_SPATIAL_FACTOR:
            raise ValueError(f"resolution {width}x{height} leaves no token grid: both sides must be at "
                             f"least {VAE_SPATIAL_FACTOR}")
        self.token_height = self.height // VAE_SPATIAL_FACTOR
        self.token_width = self.width // VAE_SPATIAL_FACTOR
        self.image_token_length = self.token_height * self.token_width
        self.base_size, self.ratio_index = self.group.base_size, ratio_index


PRESET_CUSTOM = "custom"

# the vocabulary's `<img_size_N>` ladder: every table's rows span roughly 0.79x..1.05x of N^2
BASE_SIZES = (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 8192)


def base_size_for(height, width):
    """Which `<img_size_N>` table a requested size belongs to.

    Every table's rows sit in a band around N^2 — 1024's spans 0.786 to 1.049 Mpx — so a request above
    those 1.05 Mpx is a request for a larger table, and the vocabulary has the ladder: `<img_size_256>`
    through `<img_size_8192>`. Emitting `<img_size_1024>` for a 1536x1536 request (2.36 Mpx) described a
    9216-token grid as living in the 1024 position range, which is why sizes beyond the band came back
    scrambled rather than merely different.

    Only ever raises the base: the bands overlap at the small end, and every table additionally carries
    the same absolute `EXTRA_RESOLUTIONS` rows (~1.05 Mpx), so a nearest-row rule is ambiguous below the
    1024 band and would pull 768x1024 down to 256. Nothing already inside the 1024 band moves, which is
    what keeps the presets and the verified off-table sizes exactly as they were.
    """
    if height is None:                       # a `<img_ratio_N>` token names a row of the default table
        return 1024
    area = height * width
    for candidate in BASE_SIZES:
        if candidate < 1024:
            continue
        # the table's largest row, with a little tolerance: 1200x880 is 1.056 Mpx against a 1.049 Mpx
        # largest row and was verified working at 1024, so a request a few percent past the row still
        # describes the same table. 1280x1280 (1.638 Mpx, 56% past) does not, and neither does 1536x1536.
        if area <= max(entry.height * entry.width for entry in ResolutionGroup(candidate).data) * 1.05:
            return candidate
    return BASE_SIZES[-1]


def preset_options(base_size=1024, with_custom=False):
    """The reference's resolution table as a pick-list.

    `with_custom` adds the `custom` entry for nodes that also take free numbers; the resolutions node
    leaves it out, since anything off the table belongs on the width/height inputs rather than being
    named twice.
    """
    group = ResolutionGroup(base_size)
    options = [f"{group[index].width}x{group[index].height}" for index in range(len(group))]
    return options + [PRESET_CUSTOM] if with_custom else options


def resolve_size(resolution, width, height):
    """The size a request generates at, from whichever of the preset and the numbers applies.

    Both nodes call this, which is what keeps the latent's grid and the sequence's grid the same by
    construction rather than by the user keeping two widgets in step. Sizes are rounded down to the
    VAE's spatial factor here and again in `ImageGeometry` (idempotent), because the grid is the only
    thing either of them is for.
    """
    if resolution != PRESET_CUSTOM:
        width, height = (int(part) for part in resolution.split("x"))
    return int(width) // VAE_SPATIAL_FACTOR * VAE_SPATIAL_FACTOR, int(height) // VAE_SPATIAL_FACTOR * VAE_SPATIAL_FACTOR


# config.vit_processor.max_num_patches: the checkpoint caps the vision tower's patches per image, and
# the reference's processor takes its `max_token_length` from that field (spec §43/§44).
COND_PATCH_LIMIT = 1024

# The reference's own limit on conditioning images per request (its README: "up to 3 inputs"). Three at
# ~1 Mpx each is also what fits: every image adds its VAE run (~4k tokens) and its tower run (1024).
MAX_COND_IMAGES = 3

COT_TAGS = ("think", "end_of_think", "recaption", "end_of_recaption")


def _token_ids(tokenizer):
    ids = {name: tokenizer.token_to_id(token) for name, token in META_TOKENS.items()}
    for name, value in ids.items():
        if value is None:
            raise ValueError(f"tokenizer has no {name} token ({META_TOKENS[name]!r})")
    return ids


class _Writer:
    """The token list a builder appends to, plus the two encodings every section needs.

    `uncond` is the reference's unconditional rendering, what CFG's negative pass sees: each token of
    the user's text and of the CoT text is replaced by one `<cfg>` token (`encode_general` with
    `uncond_p=1.0`), while the system prompt, the role scaffolding, the tags and the images stay. The
    sequence keeps its length and its geometry, so the two passes share one grid by construction.
    """

    def __init__(self, tokenizer, ids, uncond=False, extra_rows=True):
        self.tokenizer, self.ids, self.uncond, self.extra_rows = tokenizer, ids, uncond, extra_rows
        self.tokens = []

    def encode(self, text):
        # add_special_tokens=False mirrors the reference's encode_text; an empty affix contributes
        # nothing, so the pretrain rendering drops out rather than adding stray tokens
        return self.tokenizer.encode(text, add_special_tokens=False).ids if text else []

    def text(self, text):
        self.tokens += self.encode(text)

    def user_text(self, text):
        encoded = self.encode(text)
        self.tokens += [self.ids["cfg"]] * len(encoded) if self.uncond else encoded

    def cot(self, cot_text):
        """The CoT text: its tags verbatim, the text between them as user text (`get_cot_sections`)."""
        tags = {META_TOKENS[name]: self.ids[name] for name in COT_TAGS}
        for piece in re.split("(" + "|".join(re.escape(tag) for tag in tags) + ")", cot_text):
            if piece in tags:
                self.tokens.append(tags[piece])
            else:
                self.user_text(piece)

    def image_meta(self, geometry, with_guidance=False, with_timestep_r=False):
        """`<boi> <img_size_N> <img_ratio_i> <timestep> [<guidance>] [<timestep_r>]`; the positions of
        the embedder slots come back so the builder can report them."""
        size_token = self.tokenizer.token_to_id(f"<img_size_{geometry.base_size}>")
        if size_token is None:
            raise ValueError(f"tokenizer has no <img_size_{geometry.base_size}> token")
        self.tokens += [self.ids["boi"], size_token,
                        self.tokenizer.token_to_id(f"<img_ratio_{geometry.ratio_index}>")]
        positions = {"timestep": len(self.tokens), "guidance": None, "timestep_r": None}
        self.tokens.append(self.ids["timestep"])
        if with_guidance:
            positions["guidance"] = len(self.tokens)
            self.tokens.append(self.ids["guidance"])
        if with_timestep_r:
            positions["timestep_r"] = len(self.tokens)
            self.tokens.append(self.ids["timestep_r"])
        return positions

    def cond_image(self, image_size, patch_grid, base_size, reference_vit_padding=False):
        """One conditioning image: its VAE run and its tower run, joined by `<joint_img_sep>` inside one
        bidirectional block, with the image meta tokens but no `<guidance>`/`<timestep_r>` (spec §44).

        With `reference_vit_padding` the tower run is padded with `<img>` to the checkpoint's 1024
        tower slots, which is what the reference's processor emits (`image_processor.py:139,276`); the
        run's true grid stays in the block dict for the tower path."""
        geometry = ImageGeometry(image_size, base_size=base_size, extra_rows=self.extra_rows)
        patch_height, patch_width = patch_grid
        if patch_height * patch_width > COND_PATCH_LIMIT:
            raise ValueError(
                f"a conditioning image produces {patch_height}x{patch_width} tower patches but the "
                f"checkpoint's vit_processor caps them at {COND_PATCH_LIMIT}. Resize the image.")
        positions = self.image_meta(geometry)
        vae_start = len(self.tokens)
        self.tokens += [self.ids["img"]] * geometry.image_token_length
        vae_end = len(self.tokens)
        self.tokens.append(self.ids["joint_img_sep"])
        vit_start = len(self.tokens)
        self.tokens += [self.ids["img"]] * (patch_height * patch_width)
        if reference_vit_padding:
            self.tokens += [self.ids["img"]] * (COND_PATCH_LIMIT - patch_height * patch_width)
        vit_end = len(self.tokens)
        self.tokens.append(self.ids["eoi"])
        return {
            "vae_slice": slice(vae_start, vae_end),
            "vit_slice": slice(vit_start, vit_end),
            "joint_slice": slice(vae_start, vit_end),
            "timestep_position": positions["timestep"],
            "token_height": geometry.token_height,
            "token_width": geometry.token_width,
            "patch_height": patch_height,
            "patch_width": patch_width,
            "image_height": geometry.height,
            "image_width": geometry.width,
        }

    def user_section(self, affixes, system_prompt, prompt, cond_images, base_size,
                     reference_vit_padding=False):
        """`<bos>`, the system prompt, then the user turn: the conditioning images first and the prompt
        after them, which is the reference's order. The caller strips the system
        prompt; an empty prompt, the Base default, contributes nothing."""
        if len(cond_images) > MAX_COND_IMAGES:
            raise ValueError(f"{len(cond_images)} conditioning images; the model takes at most {MAX_COND_IMAGES}")
        self.tokens.append(self.ids["bos"])
        self.text((system_prompt or "").strip())
        self.text(affixes.system_suffix)
        self.text(affixes.user_prefix)
        blocks = [self.cond_image(size, grid, base_size, reference_vit_padding)
                  for size, grid in cond_images]
        self.user_text(prompt)
        self.text(affixes.user_suffix)
        self.text(affixes.bot_prefix)
        return blocks


def build_sequence(tokenizer, prompt, image_size, system_prompt, cot_text=None, base_size=1024,
                   max_position_embeddings=DEFAULT_MAX_POSITION_EMBEDDINGS, *, cfg_distilled,
                   use_meanflow, sequence_template="instruct", cond_images=(), uncond=False, extra_rows=True,
                   reference_vit_padding=False):
    """Encode one image request — text to image, or conditioned on up to three images — into ids.

    `cond_images` is a list of `((height, width), (patch_height, patch_width))`, one per conditioning
    image: its size in pixels (a 16-multiple) and its tower patch grid. Each becomes one block in the
    user turn ahead of the prompt, which is the reference's layout (captured in
    `dev/multi_image_reference_sequences.pt`). `uncond=True` renders the reference's CFG negative for
    the same request (see `_Writer`). `extra_rows=False` selects the base checkpoint's ratio table
    (`ImageGeometry`).

    Returns the ids plus where things are: the generated `<img>` run, its embedder slots, its grid, and
    per conditioning image the runs, grid and `<timestep>` slot the model fills.
    """
    ids = _token_ids(tokenizer)
    geometry = ImageGeometry(image_size, base_size=base_size, extra_rows=extra_rows)
    affixes = Affixes(sequence_template)
    writer = _Writer(tokenizer, ids, uncond, extra_rows)
    blocks = writer.user_section(affixes, system_prompt, prompt, cond_images, base_size,
                                 reference_vit_padding)
    if cot_text:
        # the CoT text is the assistant's text section and takes no `<answer>`; the token belongs to the
        # generated-image section that follows it (`answer == "auto"` promotes it in the instruct
        # rendering), which is also where it sits when there is no CoT text
        writer.cot(cot_text)
    if affixes.answer:
        writer.tokens.append(ids["answer"])

    positions = writer.image_meta(geometry, with_guidance=cfg_distilled, with_timestep_r=use_meanflow)
    image_start = len(writer.tokens)
    if image_start + geometry.image_token_length > max_position_embeddings:
        raise ValueError(
            f"{geometry.width}x{geometry.height} needs {image_start + geometry.image_token_length} "
            f"positions ({image_start} of prompt, conditioning and meta tokens plus "
            f"{geometry.image_token_length} of image) but this checkpoint handles "
            f"{max_position_embeddings}. Reduce the resolution, shorten the prompt, or use fewer "
            f"conditioning images."
        )
    writer.tokens += [ids["img"]] * geometry.image_token_length
    image_end = len(writer.tokens)
    writer.tokens.append(ids["eoi"])
    if affixes.answer:
        writer.text(META_TOKENS["end_of_answer"])
    writer.text(affixes.bot_suffix)

    return {
        "ids": torch.tensor(writer.tokens, dtype=torch.long),
        "image_slice": slice(image_start, image_end),
        "timestep_position": positions["timestep"],
        "guidance_position": positions["guidance"],
        "timestep_r_position": positions["timestep_r"],
        "token_height": geometry.token_height,
        "token_width": geometry.token_width,
        "image_height": geometry.height,
        "image_width": geometry.width,
        "base_size": geometry.base_size,
        "ratio_index": geometry.ratio_index,
        "cond_blocks": blocks,
        "system_prompt": system_prompt,
    }


def build_text_sequence(tokenizer, prompt, system_prompt, bot_task="think", base_size=1024,
                        max_position_embeddings=DEFAULT_MAX_POSITION_EMBEDDINGS, *,
                        sequence_template="instruct", cond_images=()):
    """Encode the prompt-rewriting stage's prompt: the user turn, then the task's opening tag.

    `<think>` asks for the analysis and `<recaption>` for the rewritten prompt; both are the same
    generation stopped at a different close tag. With conditioning images the user turn carries them
    exactly as the image stage does, so the rewrite is written looking at the images (`mode="gen_text"`
    in the reference). `mode="gen_text"` also makes `add_assistant_prefix` true, so the assistant
    scaffolding is part of the prompt and the task tag follows it; under `pretrain` those affixes are
    empty and the prompt runs straight into the tag.
    """
    if bot_task not in ("think", "recaption"):
        raise ValueError(f"unsupported text-stage bot_task {bot_task!r}")
    ids = _token_ids(tokenizer)
    writer = _Writer(tokenizer, ids)
    blocks = writer.user_section(Affixes(sequence_template), system_prompt, prompt, cond_images, base_size)
    writer.tokens.append(ids[bot_task])
    if len(writer.tokens) > max_position_embeddings:
        raise ValueError(
            f"the rewriting stage needs {len(writer.tokens)} positions but this checkpoint handles "
            f"{max_position_embeddings}. Shorten the prompt or use fewer conditioning images."
        )
    return {"ids": torch.tensor(writer.tokens, dtype=torch.long), "bot_task": bot_task, "cond_blocks": blocks}
