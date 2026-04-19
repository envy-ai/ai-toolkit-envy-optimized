import unittest

from toolkit.lora_key_format import peft_key_to_internal_key, internal_key_to_peft_key


class TestLoraKeyFormat(unittest.TestCase):
    def test_dora_magnitude_round_trips_for_transformer_peft_keys(self):
        internal_key = (
            "transformer$$transformer_blocks$$0$$attn$$to_q.magnitude"
        )

        peft_key = internal_key_to_peft_key(internal_key)

        self.assertEqual(
            peft_key,
            "transformer.transformer_blocks.0.attn.to_q.magnitude",
        )
        self.assertEqual(peft_key_to_internal_key(peft_key), internal_key)

    def test_lora_ab_weights_round_trip_for_transformer_peft_keys(self):
        internal_key = (
            "transformer$$transformer_blocks$$0$$attn$$to_q.lora_down.weight"
        )

        peft_key = internal_key_to_peft_key(internal_key)

        self.assertEqual(
            peft_key,
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
        )
        self.assertEqual(peft_key_to_internal_key(peft_key), internal_key)

    def test_lokr_alpha_keeps_parameter_separator_when_loading_peft_key(self):
        self.assertEqual(
            peft_key_to_internal_key(
                "transformer.transformer_blocks.0.attn.to_q.alpha",
                network_type="lokr",
            ),
            "transformer$$transformer_blocks$$0$$attn$$to_q.alpha",
        )


if __name__ == "__main__":
    unittest.main()
