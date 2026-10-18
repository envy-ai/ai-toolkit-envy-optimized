"""Krea 2 — instruction image-editing (in-context) extension.

Additive subclass of Krea2Model that turns the T2I model into an editor with
**dual conditioning** (the Qwen-Image-Edit recipe), reusing Krea's components:

  * semantic path  — the source image is fed into the Qwen3-VL text encoder so the
    instruction is image-grounded (``encode_control_in_text_embeddings``);
  * appearance path — the source latent is patchified and concatenated as a block
    of **clean** tokens before the (noisy) target tokens in the single-stream
    sequence, distinguished only by RoPE **frame axis = 1** (h,w stay aligned).

The fork's built-in Krea 2 loaders remain shared. arch: ``krea2_edit``.
"""
import os
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING, List, Optional
from einops import rearrange, repeat

from toolkit.config_modules import ModelConfig
from toolkit.advanced_prompt_embeds import AdvancedPromptEmbeds

from extensions_built_in.diffusion_models.krea2.krea2 import Krea2Model
from extensions_built_in.diffusion_models.krea2.src.mmdit import SingleStreamDiT
from extensions_built_in.diffusion_models.krea2.src.pipeline import pad_text_features, predict_velocity
from extensions_built_in.diffusion_models.krea2.src.text_encoder import (
    SELECT_LAYERS,
    PROMPT_TEMPLATE_ENCODE_PREFIX,
    PROMPT_TEMPLATE_ENCODE_SUFFIX,
    PROMPT_TEMPLATE_ENCODE_START_IDX,
)

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO


# The comfyui-krea2edit inference nodes expose exactly two reference inputs
# (source_image + source_image_b). A LoRA trained with more references cannot be
# reproduced at inference, so training refuses to produce one.
MAX_REFS = 2

# Grounding (semantic path) resolution policy. These defaults MATCH the inference
# node: comfyui-krea2edit's Krea2EditTextEncode defaults `grounding_px` to 768, and
# the released krea2-identity-edit LoRAs trained with per-step jitter down to 384.
# Env vars still override at launch (0 disables the cap = native-resolution
# grounding, which is NOT what the released weights were trained with).
GROUNDING_MAX_PX_DEFAULT = "768"
GROUNDING_JITTER_MIN_DEFAULT = "384"


def grounding_settings():
    """(max_px, jitter_min) for the Qwen3-VL grounding image; env overrides defaults."""
    return (
        int(os.environ.get("GROUNDING_MAX_PX", GROUNDING_MAX_PX_DEFAULT)),
        int(os.environ.get("GROUNDING_JITTER_MIN", GROUNDING_JITTER_MIN_DEFAULT)),
    )


def _img_tokens_and_pos(latent: torch.Tensor, patch: int, frame: int):
    """Patchify a (B,C,h,w) latent → (tokens (B,L,C*p*p), pos (B,L,3), mask (B,L)).

    pos axes are (frame, h, w); ``frame`` distinguishes source (1) from target (0)
    while h,w stay aligned for preservation.
    """
    b, _, h, w = latent.shape
    h_, w_ = h // patch, w // patch
    ids = torch.zeros((h_, w_, 3), device=latent.device)
    ids[..., 0] = frame
    ids[..., 1] = torch.arange(h_, device=latent.device)[:, None]
    ids[..., 2] = torch.arange(w_, device=latent.device)[None, :]
    pos = repeat(ids, "h w t -> b (h w) t", b=b, t=3)
    mask = torch.ones(b, h_ * w_, device=latent.device, dtype=torch.bool)
    tok = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    return tok, pos, mask


def predict_velocity_edit(
    model: SingleStreamDiT,
    target_latents: torch.Tensor,   # (B, C, h, w) noisy
    source_latents,                 # (B, C, h, w) clean reference, OR list of them (multi-ref)
    t: torch.Tensor,
    context: torch.Tensor,          # (B, Lt, n*d)
    text_mask: torch.Tensor,        # (B, Lt)
    fit_offsets: bool = False,      # fit protocol: refs resampled to target density get
                                    # centered fractional position offsets in the target grid
) -> torch.Tensor:
    """Sequence: [text | ref_1(frame=1) | ref_2(frame=2) | ... | target(frame=0)].

    Multi-ref (OpenSubject-style scene+subject conditioning): each reference gets its own
    RoPE frame index (1..N) with its own h,w grid — ref_1 (the scene) stays h/w-aligned with
    the target for preservation; extra refs (subject photos) are spatially independent
    identity/appearance references, distinguished purely by their frame index.
    """
    patch = model.config.patch
    b, c, h, w = target_latents.shape

    n = model.config.txtlayers
    context = context.reshape(context.shape[0], context.shape[1], n, context.shape[-1] // n)

    if not isinstance(source_latents, (list, tuple)):
        source_latents = [source_latents]
    if len(source_latents) > MAX_REFS:
        raise ValueError(
            f"krea2_edit: {len(source_latents)} reference latents in the token sequence, "
            f"but the comfyui-krea2edit nodes accept at most {MAX_REFS} references "
            "(source_image + source_image_b). Training with more would produce a LoRA "
            "the inference nodes cannot reproduce."
        )

    tgt_tok, tgt_pos, tgt_mask = _img_tokens_and_pos(target_latents, patch, frame=0)
    src_toks, src_poss, src_masks, src_len = [], [], [], 0
    for i, sl in enumerate(source_latents):
        tok, pos_i, mask_i = _img_tokens_and_pos(sl, patch, frame=i + 1)
        if fit_offsets:
            gh, gw = sl.shape[-2] // patch, sl.shape[-1] // patch
            # fractional center (2026-07-28): integer floor placed odd-gap refs
            # half a token (8px) off their true center; RoPE positions are
            # continuous floats, so the exact center is representable.
            pos_i[..., 1] += max(0.0, (h // patch - gh) / 2)
            pos_i[..., 2] += max(0.0, (w // patch - gw) / 2)
        src_toks.append(tok); src_poss.append(pos_i); src_masks.append(mask_i)
        src_len += tok.shape[1]

    img_tokens = torch.cat(src_toks + [tgt_tok], dim=1)
    txtlen = context.shape[1]
    txtpos = torch.zeros(b, txtlen, 3, device=target_latents.device)
    pos = torch.cat([txtpos] + src_poss + [tgt_pos], dim=1)
    mask = torch.cat([text_mask.to(target_latents.device).bool()] + src_masks + [tgt_mask], dim=1)

    out = model(img=img_tokens, context=context, t=t, pos=pos, mask=mask)  # [refs... ; tgt_out]
    target_out = out[:, src_len:]  # keep only the target tokens
    velocity = rearrange(
        target_out, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch, pw=patch, h=h // patch, w=w // patch,
    )
    return velocity


class Krea2EditModel(Krea2Model):
    arch = "krea2_edit"

    def __init__(self, device, model_config: ModelConfig, dtype="bf16",
                 custom_pipeline=None, noise_scheduler=None, **kwargs):
        if model_config.model_kwargs.get("edit", False):
            raise ValueError("krea2_edit uses its own edit contract; remove model_kwargs.edit.")
        if model_config.model_kwargs.get("kv_cache", False):
            raise ValueError("krea2_edit does not support the built-in Ostris edit kv_cache contract.")
        super().__init__(device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs)
        # route the control image to get_prompt_embeds (semantic grounding)
        self.encode_control_in_text_embeddings = True
        # Cached and live grounding share the raw reference presentation,
        # including the configured alpha background and grounded caption dropout.
        from toolkit.control_image import KREA_EDIT_CONTROL_PRESENTATION
        self._specialized_control_presentation = KREA_EDIT_CONTROL_PRESENTATION
        self.cache_processed_control_text_embeddings = True
        self.caption_dropout_keeps_control_images = True
        self._control_latents = None  # Tensor (single-ref) or List[Tensor] (multi-ref)
        # FIT protocol (s1e, 2026-07-14): refs resampled to target grid density
        # (AR-preserving fit-inside, no crop, no pad tokens), placed at centered
        # fractional offsets — matches the node's `fit` inference mode, which is
        # also the node's DEFAULT. Default ON here so a stock clone reproduces the
        # released recipe; `model_kwargs: {fit_refs: false}` is an explicit opt-in
        # to the legacy v1/v1.1 crop geometry.
        self.fit_refs = bool((model_config.model_kwargs or {}).get("fit_refs", True))
        self.use_raw_control_images = self.fit_refs
        if self.fit_refs:
            print("[krea2_edit] FIT refs ENABLED (target-density resample, centered offsets) "
                  "— matches node fit_mode: 'fit' (the node default)")
        else:
            bar = "!" * 78
            print(bar)
            print("[krea2_edit] WARNING: fit_refs=false -> LEGACY v1/v1.1 CROP reference geometry.")
            print("[krea2_edit]   References are center-cropped to the target aspect ratio before")
            print("[krea2_edit]   resize. A LoRA trained this way ONLY reproduces at inference with")
            print("[krea2_edit]   fit_mode: 'crop (legacy)' in the comfyui-krea2edit node — the node")
            print("[krea2_edit]   defaults to 'fit' and will misregister this LoRA otherwise.")
            print("[krea2_edit]   The released krea2-identity-edit LoRAs used fit_refs: true.")
            print(bar)
        # Grounding policy, printed once so every run is self-documenting.
        _g_max, _g_jit = grounding_settings()
        print(f"[krea2_edit] grounding: GROUNDING_MAX_PX={_g_max} GROUNDING_JITTER_MIN={_g_jit} "
              f"(recipe defaults 768/384; 0 = off/native resolution)")
        if _g_max <= 0:
            print("[krea2_edit] WARNING: grounding cap disabled -> the VLM sees native-resolution "
                  "references, which is NOT the released recipe (node default grounding_px=768).")
        # multi_ref: true (model_kwargs) -> OpenSubject-style [scene, subject] dual conditioning.
        # Dataloader delivers stacked controls (B, N, C, H, W); each ref becomes its own
        # frame-indexed token block + its own vision block in the Qwen3-VL grounding.
        self.multi_ref = bool(self.model_config.model_kwargs.get("multi_ref", False))
        if self.multi_ref:
            self.has_multiple_control_images = True
            print("[krea2_edit] multi_ref ENABLED: N-control conditioning (frames 1..N)")
        checkpoint_every_n = self.model_config.model_kwargs.get("checkpoint_every_n", 1)
        if type(checkpoint_every_n) is not int or checkpoint_every_n < 1:
            raise ValueError("krea2_edit checkpoint_every_n must be a positive integer.")
        self._validate_raw_control_training_config()

    @property
    def text_embedding_space_version(self):
        # Separate the source-graph contract from built-in Ostris edit embeddings,
        # and invalidate grounding caches whenever its resolution policy changes.
        cap, jitter = grounding_settings()
        return (f"{self.arch}_grounded_v1_fit{int(self.fit_refs)}_multi{int(self.multi_ref)}"
                f"_grounding{cap}_{jitter}_te_len{self.max_text_length}")

    @classmethod
    def validate_sample_config(cls, sample_config):
        from toolkit.comfy_sample import DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH
        comfy = getattr(sample_config, "comfy", None)
        if not (comfy is not None and comfy.enabled
                and comfy.workflow_path == DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH
                and comfy.provider == "ordinary"):
            raise ValueError(
                "krea2_edit previews require ordinary ComfyUI rendering with "
                f"sample.comfy.enabled: true and workflow_path: {DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH}. "
                "Native/shared-model previews use a different conditioning contract. "
                "Alternatively set train.disable_sampling: true."
            )

    # ------------------------------------------------------------------
    # Guards: refuse configurations the inference nodes cannot reproduce
    # ------------------------------------------------------------------
    @staticmethod
    def _find_owning_train_process():
        """Best-effort handle on the ai-toolkit train process constructing this model.

        ai-toolkit hands a model class only its ModelConfig — there is no official
        view of train/dataset config from here — but the process instance is a local
        of the calling frame (``BaseSDTrainProcess.load_model`` -> ``ModelClass(...)``),
        so the guards below can read train_config and the dataset configs. Returns
        None when we are not being constructed by a train process (plain import,
        third-party harness); the guards then no-op and the README restrictions
        apply on the honor system.
        """
        import inspect
        frame = inspect.currentframe()
        try:
            while frame is not None:
                obj = frame.f_locals.get("self", None)
                if (obj is not None and hasattr(obj, "train_config")
                        and hasattr(obj, "dataset_configs")):
                    return obj
                frame = frame.f_back
        except Exception:
            return None
        finally:
            del frame
        return None

    @staticmethod
    def _control_paths_of(ds) -> List[str]:
        """Normalized list of control_path entries declared by a DatasetConfig."""
        cp = getattr(ds, "control_path", None)
        if cp is None or cp == "":
            return []
        return list(cp) if isinstance(cp, (list, tuple)) else [cp]

    @classmethod
    def _audit_pairs(cls, ds):
        """Filename-only audit of one dataset's target/source stem pairing.

        Mirrors ai-toolkit's own resolution exactly (``AiToolkitDataset.__init__``
        builds the target list with ``os.walk`` + ``image_extensions``, skipping
        dotfiles and ``_controls/``; ``ControlFileItemDTOMixin.__init__`` then looks
        for ``<control_dir>/<target stem><ext>`` for ext in ``img_ext_list``), so the
        counts here are what the dataloader will actually resolve.

        The dataset objects themselves do not exist yet at model-construction time —
        ``BaseSDTrainProcess.run()`` builds the model (``self.sd = ModelClass(...)``)
        long before ``get_dataloader_from_datasets`` — so we resolve from the dataset
        *config* instead. Only directory entries are read; no image or caption file is
        ever opened.

        Returns ``None`` when the audit does not apply (json-manifest dataset,
        ``control_from_same_folder``, missing folder), else a dict with the counts.
        """
        ctrl_dirs = cls._control_paths_of(ds)
        if not ctrl_dirs or getattr(ds, "control_from_same_folder", False):
            return None
        root = getattr(ds, "dataset_path", None) or getattr(ds, "folder_path", None)
        if not root or not os.path.isdir(root):
            return None  # json manifest or a bad path — upstream reports that itself
        exts = (".jpg", ".jpeg", ".png", ".webp")
        targets = []
        for dirpath, _dirs, files in os.walk(root):
            if os.path.basename(dirpath) == "_controls":
                continue
            for fn in files:
                if fn.lower().endswith(exts) and not fn.startswith("."):
                    targets.append(fn)
        missing_all, partial, per_dir = [], [], {d: 0 for d in ctrl_dirs}
        for fn in targets:
            stem = os.path.splitext(fn)[0]
            hits = 0
            for d in ctrl_dirs:
                if any(os.path.exists(os.path.join(d, stem + e)) for e in exts):
                    hits += 1
                else:
                    per_dir[d] += 1
            if hits == 0:
                missing_all.append(fn)
            elif hits < len(ctrl_dirs):
                partial.append(fn)
        return {
            "root": root, "ctrl_dirs": ctrl_dirs, "total": len(targets),
            "missing_all": missing_all, "partial": partial, "per_dir": per_dir,
        }

    def _validate_raw_control_training_config(self):
        """Hard-fail configs that silently train something the nodes cannot reproduce.

        Runs once, from ``__init__`` — i.e. before the base weights, the text encoder
        and the latent/text caches are loaded, so a broken config costs seconds rather
        than a full model load (or, worse, a whole run that only looks healthy).
        """
        proc = self._find_owning_train_process()
        if proc is None:
            print("[krea2_edit] NOTE: train/dataset config not visible from the model "
                  "class; the startup guards (batch_size, flip augmentation, "
                  "source-image pairing, reference count, text-embedding caching, "
                  "unload_text_encoder) were skipped. See README, 'Not supported'.")
            return
        datasets = list(getattr(proc, "dataset_configs", None) or [])
        if not getattr(proc.train_config, "disable_sampling", False):
            for name in ("sample_config", "first_sample_config"):
                sample_config = getattr(proc, name, None)
                if sample_config is not None and getattr(sample_config, "samples", None):
                    self.validate_sample_config(sample_config)

        # --- guards specific to the fit protocol (raw control images) --------------
        if self.use_raw_control_images:
            batch_size = int(getattr(proc.train_config, "batch_size", 1) or 1)
            if batch_size > 1:
                raise ValueError(
                    f"krea2_edit: batch_size={batch_size} is not supported with the fit "
                    "reference protocol (fit_refs: true, the default).\n"
                    "  ai-toolkit collates raw control images with torch.cat, which requires "
                    "every source image in a batch to have identical pixel dimensions — a "
                    "mixed-size dataset crashes mid-run.\n"
                    "  Use train.batch_size: 1 (raise gradient_accumulation instead), or set "
                    "model_kwargs.fit_refs: false to fall back to the legacy crop geometry "
                    "(which then requires fit_mode: 'crop (legacy)' at inference)."
                )
            flipped = [
                getattr(d, "folder_path", "<dataset>")
                for d in datasets
                if getattr(d, "flip_x", False) or getattr(d, "flip_y", False)
            ]
            if flipped:
                raise ValueError(
                    "krea2_edit: flip augmentation (flip_x / flip_y) is not supported with the "
                    "fit reference protocol (fit_refs: true, the default).\n"
                    "  ai-toolkit flips the TARGET image but not the raw control (reference) "
                    "images, so every flipped pair is silently desynced — the LoRA is trained "
                    "on mirrored supervision.\n"
                    f"  Offending dataset(s): {flipped}\n"
                    "  Set flip_x: false / flip_y: false, or pre-flip your pairs offline."
                )

        # --- guards that apply to both reference protocols --------------------------
        # The whole arch grounds the instruction on the reference images inside the
        # text encoder. Unloading the TE makes ai-toolkit fall back to a cached BLANK
        # embedding for every step (SDTrainer: `if unload_text_encoder or
        # is_caching_text_embeddings` -> `cached_blank_embeds` when the batch has no
        # cached embeds), i.e. the model trains with no instruction and no grounding.
        if getattr(proc.train_config, "unload_text_encoder", False):
            raise ValueError(
                "krea2_edit: train.unload_text_encoder: true is not supported.\n"
                "  This architecture encodes the reference image(s) INSIDE the text "
                "encoder (the instruction is image-grounded), so the text encoder must "
                "stay resident. With it unloaded and no cached embeddings, ai-toolkit "
                "feeds a blank embedding on every step — the run looks healthy and "
                "trains on no conditioning at all.\n"
                "  Set train.unload_text_encoder: false. To save the TE's VRAM instead, "
                "use dataset cache_text_embeddings: true (which freezes one grounding "
                "scale — see the README VRAM section for the tradeoff)."
            )

        for ds in datasets:
            ctrl_dirs = self._control_paths_of(ds)
            folder = getattr(ds, "folder_path", None) or getattr(ds, "dataset_path", "<dataset>")
            if not ctrl_dirs:
                continue  # targets-only (plain T2I regularization) — legitimate
            if self.fit_refs and not getattr(ds, "full_size_control_images", True):
                raise ValueError(
                    "krea2_edit fit geometry requires full_size_control_images: true "
                    "so the loader retains native source aspect ratio and dimensions."
                )

            # More than two references can never be reproduced at inference. Visible
            # here from the config; _check_ref_count stays as a runtime backstop for
            # datasets that assemble references some other way.
            if len(ctrl_dirs) > MAX_REFS:
                raise ValueError(
                    f"krea2_edit: dataset '{folder}' declares {len(ctrl_dirs)} control_path "
                    f"entries, but the comfyui-krea2edit nodes accept at most {MAX_REFS} "
                    "references (source_image + source_image_b).\n"
                    f"  control_path: {ctrl_dirs}\n"
                    "  A LoRA trained with more reference token blocks cannot be reproduced "
                    "at inference. Use at most two control_path entries per dataset."
                )

            # Cached text embeddings ground only ctrl_img_list[0] unless the model
            # advertises has_multiple_control_images — which only multi_ref sets. So a
            # two-reference dataset with caching on silently drops reference #2 from
            # the semantic path (it still gets appearance tokens: half-conditioned).
            if len(ctrl_dirs) > 1 and getattr(ds, "cache_text_embeddings", False) \
                    and not self.multi_ref:
                raise ValueError(
                    f"krea2_edit: dataset '{folder}' has {len(ctrl_dirs)} control_path entries "
                    "and cache_text_embeddings: true, but model_kwargs.multi_ref is not set.\n"
                    "  When ai-toolkit caches text embeddings it grounds only the FIRST "
                    "control image unless the model reports has_multiple_control_images, "
                    "which only multi_ref: true sets. Reference #2 would be silently dropped "
                    "from the semantic (grounded text) path while still contributing "
                    "appearance tokens — the LoRA learns a conditioning the nodes never "
                    "reproduce.\n"
                    "  Fix either way:\n"
                    "    - model.model_kwargs.multi_ref: true   (grounds every reference), or\n"
                    "    - dataset cache_text_embeddings: false (the recommended setting: it "
                    "also keeps the per-step grounding jitter)."
                )

            # A target with no stem-matched source trains as plain T2I, silently.
            report = self._audit_pairs(ds)
            if report is None:
                continue
            if report["missing_all"]:
                n, m = len(report["missing_all"]), report["total"]
                ex = ", ".join(sorted(report["missing_all"])[:5])
                more = "" if n <= 5 else f" (+{n - 5} more)"
                raise ValueError(
                    f"krea2_edit: {n} of {m} target images in dataset '{report['root']}' have "
                    "no stem-matched source image.\n"
                    "  ai-toolkit resolves a control image by filename stem: for target "
                    "<stem>.<ext> it looks for <control_dir>/<stem>.{jpg,jpeg,png,webp}. When "
                    "nothing matches, the item loses its control image and trains as PLAIN "
                    "TEXT-TO-IMAGE — no reference tokens, no image-grounded text encode. The "
                    "run looks healthy and the LoRA quietly learns the wrong thing.\n"
                    f"  control_path searched: {report['ctrl_dirs']}\n"
                    f"  Example unpaired target filenames: {ex}{more}\n"
                    "  Fix: add the missing source images (stems must match exactly, "
                    "extensions need not), or move the unpaired targets into a separate "
                    "targets-only dataset entry with no control_path if you really do want "
                    "them as T2I regularization."
                )
            if report["partial"]:
                n, m = len(report["partial"]), report["total"]
                ex = ", ".join(sorted(report["partial"])[:5])
                more = "" if n <= 5 else f" (+{n - 5} more)"
                per_dir = ", ".join(f"{d}: {c} missing" for d, c in report["per_dir"].items())
                bar = "!" * 78
                print(bar)
                print(f"[krea2_edit] WARNING: {n} of {m} targets in '{report['root']}' match "
                      "SOME but not all control_path entries.")
                print(f"[krea2_edit]   per control_path -> {per_dir}")
                print(f"[krea2_edit]   examples: {ex}{more}")
                print("[krea2_edit]   Those items train with fewer references, and reference "
                      "ORDER shifts: a target")
                print("[krea2_edit]   matched only by the SECOND control_path gets it as "
                      "reference #1 (frame 1).")
                print("[krea2_edit]   Complete the pairs, or split them into their own "
                      "single-reference dataset entry.")
                print(bar)

    @staticmethod
    def _check_ref_count(n_refs: int):
        """The nodes take at most two references — refuse to train an unusable LoRA.

        Runtime backstop: the config-level control_path count is already checked at
        construction by _validate_raw_control_training_config, so this only fires for
        references assembled outside a plain control_path list.
        """
        if n_refs > MAX_REFS:
            raise ValueError(
                f"krea2_edit: {n_refs} reference images resolved for a sample, but the "
                f"comfyui-krea2edit nodes accept at most {MAX_REFS} references "
                "(source_image + source_image_b).\n"
                "  A LoRA trained with more reference token blocks cannot be reproduced "
                "at inference.\n"
                "  Use at most two control_path entries per dataset."
            )

    def load_model(self):
        super().load_model()
        ckpt_n = self.model_config.model_kwargs.get("checkpoint_every_n", 1)
        if ckpt_n > 1:
            self.transformer.checkpoint_every_n = ckpt_n
            self.print_and_status_update(
                f"Selective checkpointing: every {ckpt_n}th block "
                f"({len(self.transformer.blocks) // ckpt_n} of {len(self.transformer.blocks)} checkpointed)"
            )

    # ------------------------------------------------------------------
    # Native previews: use the source-graph Comfy renderer instead
    # ------------------------------------------------------------------
    _NO_SAMPLING_MSG = (
        "krea2_edit: native in-training sample previews are not supported for this architecture.\n"
        "  ai-toolkit's sampling path renders the inherited plain-T2I pipeline: no "
        "reference token blocks (frame>=1) and no image-grounded Qwen3-VL encode, so the "
        "previews show what the LoRA does WITHOUT the edit conditioning — misleading at "
        "best.\n"
        "  Enable ordinary ComfyUI samples with the krea2_edit_lora_sample.json.njk "
        "workflow, or set train.disable_sampling: true and evaluate checkpoints in "
        "ComfyUI with the comfyui-krea2edit nodes."
    )

    def get_generation_pipeline(self):
        raise NotImplementedError(self._NO_SAMPLING_MSG)

    def generate_single_image(self, *args, **kwargs):
        raise NotImplementedError(self._NO_SAMPLING_MSG)

    def _load_text_encoder(self):
        """Reuse the fork's convrot-aware loader and retain the vision tower.

        Only the loader uses the built-in edit flag here. The upstream grounding
        and token-geometry methods below remain authoritative for this arch.
        """
        previous = self.is_edit
        self.is_edit = True
        try:
            result = super()._load_text_encoder()
        finally:
            self.is_edit = previous
        self._mm_processor = result[2]
        return result

    # --- appearance path: stash the source latent for the prediction step ---
    @staticmethod
    def _fit_ref(x, th, tw):
        """LEGACY (fit_refs: false) crop geometry: center-crop to the target aspect
        ratio, then resize. Mirrors the node's fit_mode: 'crop (legacy)'.

        Direct interpolate (pre-2026-07-07) STRETCHED mixed-AR refs and taught an
        inverse-squash prior on faces (v1 post-release finding); that known-bad path
        is gone — center crop is the only legacy behavior the nodes reproduce."""
        sh, sw = x.shape[2], x.shape[3]
        if sh == th and sw == tw:
            return x
        tar, sar = tw / th, sw / sh
        if abs(sar - tar) > 1e-3:
            if sar > tar:  # ref too wide -> crop width
                nw = max(1, int(round(sh * tar)))
                x0 = (sw - nw) // 2
                x = x[:, :, :, x0:x0 + nw]
            else:          # ref too tall -> crop height
                nh = max(1, int(round(sw / tar)))
                y0 = (sh - nh) // 2
                x = x[:, :, y0:y0 + nh, :]
        return F.interpolate(x, size=(th, tw), mode="bilinear")

    def _fit_prep(self, c, th_px, tw_px):
        """Fit protocol prep: AR-preserving resize to fit INSIDE the target, floor
        /16 snap capped at the target dims (a grid larger than the target would need
        s<1 position scaling -> rounding collisions; see the node-side ghosting bug).
        NEAR-MATCHED AR (2026-07-15, mirrors the inference node's CROP_TOL branch):
        within 8% of the target AR, minimally center-crop to the exact AR and fill
        the target grid completely — otherwise bucket-AR quantization leaves 1-token
        margins on ~20% of same-size pairs that inference never reproduces, and the
        /16 floor squashes the fitted axis anisotropically."""
        _, _, ih, iw = c.shape
        sc = min(th_px / ih, tw_px / iw)
        if ih * sc >= th_px * 0.92 and iw * sc >= tw_px * 0.92:
            s = max(th_px / ih, tw_px / iw)
            ch, cw = min(ih, int(round(th_px / s))), min(iw, int(round(tw_px / s)))
            y0, x0 = (ih - ch) // 2, (iw - cw) // 2
            c = c[:, :, y0:y0 + ch, x0:x0 + cw]
            nh, nw = th_px, tw_px
        else:
            nh = min(max(16, int(ih * sc) // 16 * 16), max(16, th_px // 16 * 16))
            nw = min(max(16, int(iw * sc) // 16 * 16), max(16, tw_px // 16 * 16))
            # CROP-TO-GRID (2026-07-28 seam-doubling RCA, mirrors the node fix):
            # resizing ih*sc -> floor16 squashes ref content up to 15px; the error
            # peaks at the ref band edges = the outpaint seam, and ~43% of the
            # v1.2-era vertical outpaint items ALSO hit an odd token gap (8px
            # floored-center offset). Together these TRAINED IN the seam-doubling
            # band. Center-crop the ref so the fitted axis lands on the /16 grid
            # at scale sc exactly — zero squash; content matches RoPE stride-1.
            ch2, cw2 = min(ih, max(1, int(round(nh / sc)))), min(iw, max(1, int(round(nw / sc))))
            y0, x0 = (ih - ch2) // 2, (iw - cw2) // 2
            c = c[:, :, y0:y0 + ch2, x0:x0 + cw2]
        if (nh, nw) != tuple(c.shape[-2:]):
            c = F.interpolate(c, size=(nh, nw), mode="bilinear", antialias=True)
        return c

    def condition_noisy_latents(self, latents: torch.Tensor, batch: "DataLoaderBatchDTO"):
        with torch.no_grad():
            ctrl = getattr(batch, "control_tensor", None)
            ctrl_list = getattr(batch, "control_tensor_list", None)
            if ctrl is None and ctrl_list:
                # raw-control path: ragged per-item ref list (batch collates B=1)
                th_px, tw_px = (batch.tensor.shape[2], batch.tensor.shape[3]) if batch.tensor is not None \
                    else (batch.file_items[0].crop_height, batch.file_items[0].crop_width)
                refs = ctrl_list[0] if isinstance(ctrl_list[0], (list, tuple)) else ctrl_list
                self._check_ref_count(len(refs))
                lats = []
                for c in refs:
                    if c.dim() == 3:
                        c = c.unsqueeze(0)
                    c = (c * 2 - 1).to(self.vae_device_torch, dtype=self.torch_dtype)
                    c = self._fit_prep(c, th_px, tw_px) if self.fit_refs \
                        else self._fit_ref(c, th_px, tw_px)
                    lats.append(self.encode_images(c).to(latents.device, latents.dtype))
                self._control_latents = lats if len(lats) > 1 else lats[0]
                if not getattr(self, "_edit_path_logged", False):
                    print(f"[krea2_edit] {'FIT' if self.fit_refs else 'CROP (legacy)'} refs: "
                          f"{[tuple(l.shape[-2:]) for l in lats]} "
                          f"for target {tuple(latents.shape[-2:])}", flush=True)
                    self._edit_path_logged = True
                return latents.detach()
            if ctrl is not None:
                th, tw = (batch.tensor.shape[2], batch.tensor.shape[3]) if batch.tensor is not None \
                    else (batch.file_items[0].crop_height, batch.file_items[0].crop_width)
                # ROUTING FIX (2026-07-15, verifier showstopper): fit_refs previously
                # applied ONLY to the raw control_tensor_list path (>=2 controls) —
                # single-ref pairs (~80% of the pack, incl. the outpaint/sheet blocks
                # whose whole point is AR-mismatch supervision) silently trained under
                # the old crop protocol. Route every control through the active
                # protocol prep; _fit_ref (crop) remains the non-fit fallback.
                def _prep(c_):
                    if self.fit_refs:
                        c_ = self._fit_prep(c_, th, tw)
                        if not getattr(self, "_fit_single_logged", False):
                            print(f"[krea2_edit] FIT single-ref: {tuple(c_.shape[-2:])} "
                                  f"for target {(th, tw)}", flush=True)
                            self._fit_single_logged = True
                        return c_
                    return self._fit_ref(c_, th, tw)
                if ctrl.dim() == 5:  # multi-ref: (B, N, C, H, W) -> list of per-ref latents
                    self._check_ref_count(ctrl.shape[1])
                    lats = []
                    for i in range(ctrl.shape[1]):
                        c_i = (ctrl[:, i] * 2 - 1).to(self.vae_device_torch, dtype=self.torch_dtype)
                        c_i = _prep(c_i)
                        lats.append(self.encode_images(c_i).to(latents.device, latents.dtype))
                    self._control_latents = lats
                else:
                    c = (ctrl * 2 - 1).to(self.vae_device_torch, dtype=self.torch_dtype)
                    c = _prep(c)
                    self._control_latents = self.encode_images(c).to(latents.device, latents.dtype)
            else:
                self._control_latents = None
        if not getattr(self, "_edit_path_logged", False):
            nref = (len(self._control_latents) if isinstance(self._control_latents, list)
                    else (1 if self._control_latents is not None else 0))
            print(f"[krea2_edit] control_tensor present: {ctrl is not None} (refs={nref}) "
                  f"-> edit path {'ACTIVE' if self._control_latents is not None else 'INACTIVE (plain T2I!)'}")
            self._edit_path_logged = True
        return latents.detach()

    def get_noise_prediction(self, latent_model_input, timestep, text_embeddings, **kwargs):
        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)
        t = timestep.to(self.device_torch, dtype=torch.float32) / 1000.0
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != latent_model_input.shape[0]:
            t = t.expand(latent_model_input.shape[0])
        context, text_mask = pad_text_features(
            text_embeddings.text_embeds, self.device_torch, self.torch_dtype
        )
        li = latent_model_input.to(self.device_torch, self.torch_dtype)
        m = self.transformer
        if os.environ.get("KREA2_COMPILE") == "1" and not getattr(self, "_compiled", False):
            mode = os.environ.get("KREA2_COMPILE_MODE", "default")
            for i, block in enumerate(self.transformer.blocks):
                self.transformer.blocks[i] = torch.compile(block, mode=mode)
            print(f"[krea2_edit] block-compiled {len(self.transformer.blocks)} transformer blocks", flush=True)
            self._compiled = True
        if self._control_latents is None:
            return predict_velocity(m, li, t, context, text_mask)
        def _fit(s):
            s = s.to(self.device_torch, self.torch_dtype)
            if s.shape[0] != li.shape[0]:
                s = s.expand(li.shape[0], *s.shape[1:])
            return s
        if isinstance(self._control_latents, list):
            src = [_fit(s) for s in self._control_latents]
        else:
            src = _fit(self._control_latents)
        return predict_velocity_edit(m, li, src, t, context, text_mask,
                                     fit_offsets=self.fit_refs)

    # --- semantic path: image-grounded Qwen3-VL encode ---
    def get_prompt_embeds(self, prompt, control_images=None) -> "AdvancedPromptEmbeds":
        if control_images is None:
            return super().get_prompt_embeds(prompt)
        if isinstance(prompt, str):
            prompt = [prompt]
        self._offload_transformer_for_low_vram_prompt_encoding()
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        # TE_BATCHED=1: ONE right-padded Qwen3-VL forward for the whole batch instead of the
        # serial per-item loop below (~0.65s/item on H100 -> dominates on-the-fly step time).
        # Numerics validated vs single-item (relL2<0.05, Spark 2026-07-01 + local re-validation).
        # Single-ref 4D only; multi-ref always takes the serial path.
        if (os.environ.get("TE_BATCHED", "0") == "1" and torch.is_tensor(control_images)
                and control_images.dim() == 4 and control_images.shape[0] == len(prompt)):
            imgs = [control_images[i] for i in range(control_images.shape[0])]
            feats = self._encode_image_prompt_batch(
                [p if isinstance(p, str) else "" for p in prompt], imgs)
            feats = [f.reshape(f.shape[0], -1).to(self.torch_dtype) for f in feats]
            return AdvancedPromptEmbeds(text_embeds=feats)

        # Normalize every input shape into per-item image LISTS (multi-ref grounding = one
        # vision block per reference image in the same user turn):
        #   4D (B,C,H,W)   -> [[img]] per item          (single-ref training batch)
        #   5D (B,N,C,H,W) -> [[img_1..img_N]] per item (multi-ref training batch)
        #   list           -> one item, list of refs    (caching mixin, has_multiple_control_images)
        if torch.is_tensor(control_images):
            if control_images.dim() == 5:
                per_item = [[control_images[i, j] for j in range(control_images.shape[1])]
                            for i in range(control_images.shape[0])]
            elif control_images.dim() == 4 and control_images.shape[0] == len(prompt):
                per_item = [[control_images[i]] for i in range(control_images.shape[0])]
            else:
                per_item = [[control_images.squeeze(0) if control_images.dim() == 4 else control_images]] * len(prompt)
        elif isinstance(control_images, (list, tuple)):
            if control_images and isinstance(control_images[0], (list, tuple)):
                # raw-control path: list of per-item ref LISTS (ragged sizes)
                per_item = [[(im.squeeze(0) if torch.is_tensor(im) and im.dim() == 4 else im)
                             for im in item] for item in control_images]
                if len(per_item) == 1 and len(prompt) > 1:
                    per_item = per_item * len(prompt)
            else:
                refs = [(im.squeeze(0) if torch.is_tensor(im) and im.dim() == 4 else im) for im in control_images]
                per_item = [refs] * len(prompt)
        else:
            per_item = [[control_images]] * len(prompt)

        feats = []
        for i, p in enumerate(prompt):
            # ai-toolkit passes neg=False for an unset negative prompt -> treat any
            # non-string (False/None) as an empty instruction (null text + source image).
            p = p if isinstance(p, str) else ""
            f = self._encode_image_prompt(p, per_item[i])
            feats.append(f.reshape(f.shape[0], -1).to(self.torch_dtype))
        return AdvancedPromptEmbeds(text_embeds=feats)

    def _grounding_pil(self, image: torch.Tensor):
        """Control tensor (C,H,W) 0..1 -> PIL for the VLM, with env-gated downscale.

        The grounding branch is SEMANTIC — fine detail reaches the DiT through the VAE
        source-latent tokens, not through Qwen3-VL — so capping the grounding image cuts
        vision tokens ~(native/cap)^2 with little semantic loss, shrinking the DiT's text
        stream (faster steps) and the TE forward (faster on-the-fly encoding).

        Defaults are the released recipe (max 768 / jitter min 384) and match the
        inference node's grounding_px default; the env vars only override them.

        GROUNDING_MAX_PX     (int, default 768, 0=off): cap the longest side fed to Qwen3-VL.
        GROUNDING_JITTER_MIN (int, default 384, 0=off): with MAX_PX set, sample the cap per
            call from uniform[MIN, MAX_PX] -> the model learns granularity/length
            robustness, so inference may freely use other grounding scales without a
            train/infer mismatch.
        """
        from PIL import Image
        import numpy as np
        import random
        arr = (image.detach().float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil = Image.fromarray(arr.transpose(1, 2, 0))
        cap, jmin = grounding_settings()
        if cap > 0:
            tgt = random.randint(jmin, cap) if 0 < jmin < cap else cap
            if max(pil.size) > tgt:
                s = tgt / max(pil.size)
                pil = pil.resize(
                    (max(1, round(pil.size[0] * s)), max(1, round(pil.size[1] * s))),
                    Image.LANCZOS)
        return pil

    @torch.no_grad()
    def _encode_image_prompt_batch(self, prompts: List[str], images: List[torch.Tensor]) -> List[torch.Tensor]:
        """Batched twin of _encode_image_prompt: one padded forward, per-item valid slice.

        Right padding keeps the fixed START_IDX system prefix aligned at the front of every
        row (causal attention: real tokens never attend to the pads that follow them), and
        the attention_mask valid-length slice drops the pads from the tapped hidden states.
        Returns a list of (L_i, n_layers, d) tensors identical to the single-item path.
        """
        dev = self.text_encoder.device
        if getattr(self, "_mm_processor", None) is None:
            from transformers import AutoProcessor
            te_path = self.model_config.model_kwargs.get(
                "text_encoder_path", "Qwen/Qwen3-VL-4B-Instruct")
            self._mm_processor = AutoProcessor.from_pretrained(te_path)
        pils = [self._grounding_pil(im) for im in images]
        vis = "<|vision_start|><|image_pad|><|vision_end|>"
        texts = [PROMPT_TEMPLATE_ENCODE_PREFIX + vis + p + PROMPT_TEMPLATE_ENCODE_SUFFIX
                 for p in prompts]
        tok = self._mm_processor.tokenizer
        old_side = tok.padding_side
        tok.padding_side = "right"
        try:
            inputs = self._mm_processor(
                text=texts, images=pils, padding=True, return_tensors="pt",
            ).to(dev, non_blocking=True)
        finally:
            tok.padding_side = old_side
        states = self.text_encoder(**inputs, output_hidden_states=True)
        hiddens = torch.stack([states.hidden_states[i] for i in SELECT_LAYERS], dim=2)  # (B,S,n,d)
        mask = inputs["attention_mask"]
        feats = []
        for b in range(len(prompts)):
            valid = int(mask[b].sum().item())
            feats.append(hiddens[b, PROMPT_TEMPLATE_ENCODE_START_IDX:valid])
        return feats

    @torch.no_grad()
    def _encode_image_prompt(self, prompt: str, image) -> torch.Tensor:
        """Feed (reference image(s) + instruction) through Qwen3-VL; tap SELECT_LAYERS.

        `image` may be a single (C,H,W) tensor or a LIST of them (multi-ref): each
        reference contributes its own <|vision_start|><|image_pad|><|vision_end|> block
        in the same user turn, in order (scene first, subject refs after) — mirroring
        the frame-index order of the token blocks in predict_velocity_edit.
        """
        dev = self.text_encoder.device
        # krea2's self.processor is a TEXT tokenizer; load a real multimodal
        # processor (with image_processor) lazily for the vision path.
        if getattr(self, "_mm_processor", None) is None:
            from transformers import AutoProcessor
            te_path = self.model_config.model_kwargs.get(
                "text_encoder_path", "Qwen/Qwen3-VL-4B-Instruct"
            )
            self._mm_processor = AutoProcessor.from_pretrained(te_path)
        images = image if isinstance(image, (list, tuple)) else [image]
        # tensors (C,H,W) 0..1 -> PILs (env-gated grounding downscale/jitter inside)
        pils = [self._grounding_pil(im) for im in images]
        vis = "<|vision_start|><|image_pad|><|vision_end|>" * len(pils)
        prefix = PROMPT_TEMPLATE_ENCODE_PREFIX + vis
        inputs = self._mm_processor(
            text=[prefix + prompt + PROMPT_TEMPLATE_ENCODE_SUFFIX],
            images=pils, return_tensors="pt",
        ).to(dev, non_blocking=True)
        states = self.text_encoder(**inputs, output_hidden_states=True)
        hiddens = torch.stack([states.hidden_states[i] for i in SELECT_LAYERS], dim=2)
        return hiddens[0, PROMPT_TEMPLATE_ENCODE_START_IDX:]
