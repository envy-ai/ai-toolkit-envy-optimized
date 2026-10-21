"""CPU checkpoint-layout contracts for the native FIBO2 transformer."""

import unittest

import torch

from extensions_built_in.diffusion_models.fibo2.src.transformer import (
    Fibo2Params,
    Fibo2Transformer,
    convert_checkpoint,
)


def tiny_params():
    return Fibo2Params(
        in_channels=4,
        dim=16,
        n_heads=2,
        n_layers=2,
        n_refiner_layers=1,
        injection_layers=(1,),
        cap_feat_dim=12,
        perceiver_layers=1,
        min_gist_tokens=1,
        max_gist_tokens=2,
        gist_step=1,
        gist_min_text_len=2,
        gist_max_text_len=8,
        ffn_dim=24,
        axes_dims=(4, 2, 2),
    )


def native_split_state(canonical):
    """Build the Apache split-projection layout from a tiny canonical state."""
    native = {}
    for key, value in canonical.items():
        if key.endswith(".attention.qkv.weight"):
            root = key[: -len("qkv.weight")]
            for name, tensor in zip(("to_q", "to_k", "to_v"), value.chunk(3, dim=0)):
                native[root + name + ".weight"] = tensor.clone()
        elif key.endswith(".attention.out.weight"):
            native[key[: -len("out.weight")] + "to_out.0.weight"] = value.clone()
        elif key.endswith(".attention.q_norm.weight"):
            native[key[: -len("q_norm.weight")] + "norm_q.weight"] = value.clone()
        elif key.endswith(".attention.k_norm.weight"):
            native[key[: -len("k_norm.weight")] + "norm_k.weight"] = value.clone()
        elif key.endswith(".to_kv.weight"):
            root = key[: -len("to_kv.weight")]
            for name, tensor in zip(("to_k", "to_v"), value.chunk(2, dim=0)):
                native[root + name + ".weight"] = tensor.clone()
        elif key.endswith(".to_out.weight"):
            native[key[: -len("to_out.weight")] + "to_out.0.weight"] = value.clone()
        elif key.endswith(".w13.weight"):
            root = key[: -len("w13.weight")]
            for name, tensor in zip(("w1", "w3"), value.chunk(2, dim=0)):
                native[root + name + ".weight"] = tensor.clone()
        else:
            native[key] = value.clone()
    return native


class Fibo2CheckpointMappingTests(unittest.TestCase):
    def test_apache_split_weights_convert_to_complete_canonical_state(self):
        model = Fibo2Transformer(tiny_params())
        expected = model.state_dict()
        native = native_split_state(expected)

        converted = convert_checkpoint(native)

        self.assertEqual(set(converted), set(expected))
        for key, value in expected.items():
            with self.subTest(key=key):
                torch.testing.assert_close(converted[key], value, rtol=0, atol=0)

    def test_config_inference_preserves_tiny_architecture_cardinality(self):
        model = Fibo2Transformer(tiny_params())

        inferred = Fibo2Transformer.aitk_config_from_state_dict(model.state_dict())

        self.assertEqual(inferred.in_channels, 4)
        self.assertEqual(inferred.dim, 16)
        self.assertEqual(inferred.n_heads, 2)
        self.assertEqual(inferred.n_layers, 2)
        self.assertEqual(inferred.n_refiner_layers, 1)
        self.assertEqual(inferred.injection_layers, (1,))
        self.assertEqual(inferred.cap_feat_dim, 12)
        self.assertEqual(inferred.perceiver_layers, 1)
        self.assertEqual(inferred.max_gist_tokens, 2)
        self.assertEqual(inferred.ffn_dim, 24)

    def test_split_quantized_checkpoint_is_rejected_instead_of_repacked_as_float(self):
        state = {
            "layers.0.attention.to_q.weight": torch.zeros(2, 2),
            "layers.0.attention.to_q.comfy_quant": torch.tensor(1),
        }

        with self.assertRaisesRegex(ValueError, "repacked"):
            convert_checkpoint(state)

    def test_unknown_comfy_attention_metadata_is_rejected(self):
        state = {
            "attention.comfy_attention.config": torch.tensor(list(b'{"attention":"unknown"}'), dtype=torch.uint8),
        }

        with self.assertRaisesRegex(ValueError, "Unsupported FIBO attention metadata"):
            convert_checkpoint(state)

if __name__ == "__main__":
    unittest.main()
