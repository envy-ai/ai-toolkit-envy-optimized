"""Focused Krea2 edit unit tests; import the model without loading weights."""
import importlib.util
import pathlib
import sys
import types
import unittest
from unittest.mock import patch

import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_edit_module():
    """Load Krea2Model and its edit subclass with heavyweight imports stubbed."""
    from testing.test_krea2_diff_lora import _install_krea_import_stubs

    with patch.dict(sys.modules):
        _install_krea_import_stubs([])

        def module(name, **attrs):
            result = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(result, key, value)
            sys.modules[name] = result
            return result

        module(
            "toolkit.control_image",
            KREA_EDIT_CONTROL_PRESENTATION="rgb",
            load_control_rgb=lambda *args, **kwargs: None,
        )
        module("huggingface_hub.utils", __path__=[])
        module("huggingface_hub.utils.tqdm", is_tqdm_disabled=lambda *args, **kwargs: False)
        module("toolkit.sample_progress", SampleProgressMixin=type("SampleProgressMixin", (), {}))
        module("safetensors", __path__=[])
        module("safetensors.torch", load_file=lambda *args, **kwargs: {},
               save_file=lambda *args, **kwargs: None)
        torchvision = module("torchvision", __path__=[])
        transforms = module("torchvision.transforms", __path__=[])
        functional = module("torchvision.transforms.functional", to_tensor=lambda image: image)
        torchvision.transforms = transforms
        transforms.functional = functional

        # Synthetic package parents prevent diffusion_models/__init__.py from
        # eagerly importing every built-in architecture while the focused Krea
        # base module is loaded.
        module("extensions_built_in", __path__=[str(REPO_ROOT / "extensions_built_in")])
        module("extensions_built_in.diffusion_models", __path__=[
            str(REPO_ROOT / "extensions_built_in/diffusion_models")
        ])
        module("extensions_built_in.diffusion_models.krea2", __path__=[
            str(REPO_ROOT / "extensions_built_in/diffusion_models/krea2")
        ])

        extension_root = "extensions.krea2_edit"
        extensions = module("extensions", __path__=[])
        package = module(extension_root, __path__=[str(REPO_ROOT / "extensions/krea2_edit")])
        package.__package__ = extension_root
        extensions.krea2_edit = package

        base_path = REPO_ROOT / "extensions_built_in/diffusion_models/krea2/krea2.py"
        base_name = "extensions_built_in.diffusion_models.krea2.krea2"
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        base = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = base
        spec.loader.exec_module(base)

        edit_path = REPO_ROOT / "extensions/krea2_edit/krea2_edit.py"
        edit_name = f"{extension_root}.krea2_edit"
        spec = importlib.util.spec_from_file_location(edit_name, edit_path)
        edit = importlib.util.module_from_spec(spec)
        sys.modules[edit_name] = edit
        spec.loader.exec_module(edit)
        return edit


EDIT = _load_edit_module()


class FakeDiT:
    config = types.SimpleNamespace(patch=2, txtlayers=2)

    def __init__(self):
        self.args = None

    def __call__(self, **kwargs):
        self.args = kwargs
        # Returning the image input lets the test verify that only target tokens
        # are unpacked from the final reference-plus-target output sequence.
        return kwargs["img"]


class Krea2EditGeometryTests(unittest.TestCase):
    def setUp(self):
        self.model = object.__new__(EDIT.Krea2EditModel)

    def test_fit_prep_near_matched_ar_fills_the_complete_target_grid(self):
        source = torch.arange(3 * 512 * 720, dtype=torch.float32).reshape(1, 3, 512, 720)

        fitted = self.model._fit_prep(source, 512, 768)

        self.assertEqual(tuple(fitted.shape), (1, 3, 512, 768))
        self.assertTrue(torch.isfinite(fitted).all())

    def test_fit_prep_strongly_mismatched_source_fits_inside_without_stretching(self):
        source = torch.ones(1, 3, 128, 1024)

        fitted = self.model._fit_prep(source, 512, 512)

        self.assertEqual(tuple(fitted.shape), (1, 3, 64, 512))
        self.assertTrue(torch.equal(fitted, torch.ones_like(fitted)))

    def test_predict_velocity_edit_orders_refs_and_returns_target_only(self):
        model = FakeDiT()
        context = torch.zeros(1, 2, 6)
        text_mask = torch.ones(1, 2, dtype=torch.bool)
        ref1 = torch.full((1, 1, 2, 4), 11.0)  # 1x2 tokens
        ref2 = torch.full((1, 1, 2, 2), 22.0)  # 1x1 token
        target = torch.full((1, 1, 4, 4), 33.0)  # 2x2 tokens

        velocity = EDIT.predict_velocity_edit(
            model, target, [ref1, ref2], torch.tensor([0.5]), context, text_mask,
            fit_offsets=True,
        )

        self.assertEqual(tuple(velocity.shape), tuple(target.shape))
        self.assertTrue(torch.equal(velocity, target))
        args = model.args
        # `img` holds ref1 (2), ref2 (1), then target (4) patch tokens.
        self.assertEqual(args["img"].shape[1], 7)
        pos = args["pos"]
        # Text prefix first, then reference frames 1 and 2, then target frame 0.
        self.assertTrue(torch.equal(pos[0, :2], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(pos[0, 2:4, 0], torch.ones(2)))
        self.assertTrue(torch.equal(pos[0, 4:5, 0], torch.full((1,), 2.0)))
        self.assertTrue(torch.equal(pos[0, 5:, 0], torch.zeros(4)))
        # Ref 1 is one token row above target center; ref 2 has half-token
        # offsets on both axes because the center is fractional.
        self.assertTrue(torch.equal(pos[0, 2:4, 1:], torch.tensor([[0.5, 0.0], [0.5, 1.0]])))
        self.assertTrue(torch.equal(pos[0, 4, 1:], torch.tensor([0.5, 0.5])))


class Krea2EditGuardTests(unittest.TestCase):
    def test_embedding_cache_identity_separates_builtin_and_grounding_settings(self):
        model = object.__new__(EDIT.Krea2EditModel)
        model.max_text_length = 256
        model.fit_refs = True
        model.multi_ref = False
        with patch.dict("os.environ", {"GROUNDING_MAX_PX": "768", "GROUNDING_JITTER_MIN": "384"}):
            default_identity = model.text_embedding_space_version
        with patch.dict("os.environ", {"GROUNDING_MAX_PX": "900", "GROUNDING_JITTER_MIN": "500"}):
            changed_grounding_identity = model.text_embedding_space_version

        builtin = object.__new__(EDIT.Krea2Model)
        builtin.max_text_length = 256
        self.assertNotEqual(default_identity, builtin.text_embedding_space_version)
        self.assertNotEqual(default_identity, changed_grounding_identity)

    def test_sample_config_requires_enabled_ordinary_comfy_and_exact_workflow(self):
        from toolkit.comfy_sample import DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH

        valid = types.SimpleNamespace(comfy=types.SimpleNamespace(
            enabled=True, workflow_path=DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH,
            provider="ordinary",
        ))
        EDIT.Krea2EditModel.validate_sample_config(valid)
        cases = (
            (types.SimpleNamespace(enabled=False, workflow_path=DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH,
                                   provider="ordinary"), "sample.comfy.enabled"),
            (types.SimpleNamespace(enabled=True, workflow_path="config/comfy_templates/krea2_lora_sample.json.njk",
                                   provider="ordinary"), "workflow_path"),
            (types.SimpleNamespace(enabled=True, workflow_path=DEFAULT_COMFY_KREA2_EDIT_WORKFLOW_PATH,
                                   provider="native"), "ordinary ComfyUI rendering"),
        )
        for comfy, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                EDIT.Krea2EditModel.validate_sample_config(types.SimpleNamespace(comfy=comfy))

    def test_native_training_sampling_is_rejected(self):
        model = object.__new__(EDIT.Krea2EditModel)
        for method in (model.get_generation_pipeline, model.generate_single_image):
            with self.assertRaisesRegex(NotImplementedError, "previews are not supported"):
                method()

    def test_text_encoder_reuses_builtin_loader_under_edit_flag_and_restores_state(self):
        model = object.__new__(EDIT.Krea2EditModel)
        model.is_edit = False
        model._mm_processor = None
        expected = ("tokenizer", "processor", "vision-processor", "encoder")

        def load_builtin(instance):
            self.assertTrue(instance.is_edit)
            return expected

        with patch.object(EDIT.Krea2Model, "_load_text_encoder", load_builtin, create=True):
            result = model._load_text_encoder()

        self.assertEqual(result, expected)
        self.assertIs(model._mm_processor, expected[2])
        self.assertFalse(model.is_edit)

        def fail_builtin(instance):
            self.assertTrue(instance.is_edit)
            raise RuntimeError("loader failed")

        model._mm_processor = "preserved"
        with patch.object(EDIT.Krea2Model, "_load_text_encoder", fail_builtin, create=True):
            with self.assertRaisesRegex(RuntimeError, "loader failed"):
                model._load_text_encoder()
        self.assertIs(model._mm_processor, "preserved")
        self.assertFalse(model.is_edit)

    def test_training_guards_reject_batch_flips_and_more_than_two_references(self):
        cases = (
            (2, [types.SimpleNamespace(control_path=["ref"], flip_x=False, flip_y=False)], "batch_size=2"),
            (1, [types.SimpleNamespace(control_path=["ref"], flip_x=True, flip_y=False)], "flip augmentation"),
            (1, [types.SimpleNamespace(control_path=["a", "b", "c"], flip_x=False, flip_y=False)], "at most 2 references"),
        )
        for batch_size, datasets, message in cases:
            with self.subTest(message=message):
                model = object.__new__(EDIT.Krea2EditModel)
                model.fit_refs = True
                model.use_raw_control_images = True
                model.multi_ref = False
                process = types.SimpleNamespace(
                    train_config=types.SimpleNamespace(batch_size=batch_size, unload_text_encoder=False),
                    dataset_configs=datasets,
                )
                model._find_owning_train_process = lambda: process
                with self.assertRaisesRegex(ValueError, message):
                    model._validate_raw_control_training_config()

    def test_pairing_audit_rejects_unpaired_target(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as root, TemporaryDirectory() as refs:
            pathlib.Path(root, "target.png").touch()
            model = object.__new__(EDIT.Krea2EditModel)
            model.fit_refs = True
            model.use_raw_control_images = True
            model.multi_ref = False
            dataset = types.SimpleNamespace(
                folder_path=root, dataset_path=root, control_path=[refs],
                flip_x=False, flip_y=False, cache_text_embeddings=False,
            )
            process = types.SimpleNamespace(
                train_config=types.SimpleNamespace(batch_size=1, unload_text_encoder=False),
                dataset_configs=[dataset],
            )
            model._find_owning_train_process = lambda: process
            with self.assertRaisesRegex(ValueError, "no stem-matched source image"):
                model._validate_raw_control_training_config()


if __name__ == "__main__":
    unittest.main()
