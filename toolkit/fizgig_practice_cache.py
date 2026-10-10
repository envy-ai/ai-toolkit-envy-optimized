"""Persistent, versioned latents for Fizgig's generated practice images."""

import hashlib
import inspect
import json
import logging
import os

import torch
from safetensors import SafetensorError, safe_open

import toolkit.flow_training as flow_training
from toolkit.flow_cache_identity import (
    local_component_identity, resolved_flow_components, specialized_text_cache_identity,
)
from toolkit.safetensors_cache import atomic_save_file


_LOG = logging.getLogger(__name__)
_VERSION = 1


class FizgigPracticeCache:
    def __init__(self, trainer):
        self.trainer = trainer
        self.directory = None
        self.common = None
        self.source_identities = {}
        model = getattr(trainer, "sd", None)
        config = getattr(trainer, "model_config", None) or getattr(model, "model_config", None)
        root = getattr(trainer, "save_root", None)
        if not root or model is None or config is None:
            return

        paths = {
            key: getattr(config, key, None)
            for key in ("name_or_path", "extras_name_or_path", "vae_path", "text_encoder_path",
                        "assistant_lora_path", "te_name_or_path")
        }
        kwargs = getattr(config, "model_kwargs", {}) or {}
        code_paths = {}
        for key, value in (
            ("trainer", type(trainer)),
            ("model", type(model)),
            ("pipeline", type(getattr(model, "pipeline", None))),
            ("bank_renderer", flow_training),
        ):
            try:
                source = inspect.getsourcefile(value)
            except (TypeError, OSError):
                source = None
            code_paths[key] = local_component_identity(source) if source else None

        self.directory = os.path.join(root, "fizgig_practice_cache")
        self.common = {
            "version": _VERSION,
            "components": resolved_flow_components(model),
            "qwen_text_identity": (
                specialized_text_cache_identity(model, include_qwen=True)
                if getattr(model, "arch", None) == "qwen_image_2" else None
            ),
            "paths": paths,
            "local_paths": {key: local_component_identity(value) for key, value in paths.items()},
            "model_kwargs": kwargs,
            "precision": {key: getattr(config, key, None) for key in
                          ("quantize", "qtype", "quantize_te", "qtype_te")},
            "code": code_paths,
            "resolution": trainer.slider_bank_resolution,
            "steps": trainer.slider_bank_steps,
            "cfg": trainer.slider_cfg,
            "cfg_reference": getattr(trainer, "cfg_reference", None),
        }

    def _entry(self, kind, index, positive, negative, seed):
        if self.directory is None:
            return None, None
        if kind not in ("prompt", "anchor"):
            raise ValueError(f"Unknown Fizgig practice cache kind: {kind}")
        payload = {**self.common, "kind": kind, "index": index,
                   "positive": positive, "negative": negative, "seed": seed}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        return os.path.join(self.directory, f"{kind}_{index:06d}.safetensors"), digest

    def load(self, kind, index, positive, negative, seed):
        path, digest = self._entry(kind, index, positive, negative, seed)
        if path is None:
            return None
        try:
            with safe_open(path, framework="pt", device="cpu") as cache:
                metadata = cache.metadata() or {}
                if metadata.get("fizgig_practice_key") != digest or "latent" not in cache.keys():
                    return None
                latent = cache.get_tensor("latent")
            if latent.ndim != 4 or latent.shape[0] != 1 or not latent.is_floating_point():
                raise ValueError("invalid practice latent shape or dtype")
            if not torch.isfinite(latent).all():
                raise ValueError("nonfinite practice latent")
            self.source_identities[(kind, index)] = metadata.get("practice_image_identity")
            return latent
        except FileNotFoundError:
            return None
        except (SafetensorError, OSError, ValueError, RuntimeError) as error:
            _LOG.warning("Regenerating invalid Fizgig practice cache %s: %s", path, error)
            return None

    def save(self, kind, index, positive, negative, seed, latent):
        path, digest = self._entry(kind, index, positive, negative, seed)
        if path is None:
            return
        try:
            from toolkit.training_examples import recording_enabled
            source = os.path.join(os.path.dirname(self.directory), "loss_report_practice",
                                  f"{'practice' if kind == 'prompt' else 'anchor'}_{index:06d}.png")
            source_identity = (local_component_identity(source)
                               if recording_enabled(self.trainer) and os.path.isfile(source) else None)
            metadata = {"fizgig_practice_key": digest}
            if source_identity is not None:
                metadata["practice_image_identity"] = json.dumps(source_identity)
            atomic_save_file({"latent": latent.detach().cpu().contiguous()}, path,
                             metadata=metadata)
            self.source_identities[(kind, index)] = metadata.get("practice_image_identity")
        except OSError as error:
            # A full/unwritable cache disk should not discard a training step.
            _LOG.warning("Could not save Fizgig practice cache %s: %s", path, error)


def remember_cached_practice_source(trainer, kind, index, latent, caption, expected_identity):
    """Keep the per-image training report linked to the existing practice PNG."""
    from toolkit.training_examples import recording_enabled, source_item
    if not recording_enabled(trainer) or expected_identity is None:
        return
    name = "practice" if kind == "prompt" else "anchor"
    directory = os.path.join(trainer.save_root, "loss_report_practice")
    path = os.path.join(directory, f"{name}_{index:06d}.png")
    if os.path.isfile(path) and json.dumps(local_component_identity(path)) == expected_identity:
        if not hasattr(trainer, "_loss_report_bank_sources"):
            trainer._loss_report_bank_sources = {}
        trainer._loss_report_bank_sources[id(latent)] = source_item(
            path, caption, directory, role="practice_image")
