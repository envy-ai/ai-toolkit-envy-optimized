import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import NetworkConfig
from jobs.process.BaseSDTrainProcess import _normalize_network_config


class NetworkConfigTests(unittest.TestCase):
    def test_network_kwargs_null_defaults_to_empty_dict(self):
        config = NetworkConfig(network_kwargs=None)

        self.assertEqual(config.network_kwargs, {})

    def test_process_level_network_filters_are_moved_to_network_kwargs(self):
        raw_network_config = {"type": "lora", "network_kwargs": None}
        process_config = {
            "network": raw_network_config,
            "only_if_contains": ["transformer_blocks.16."],
            "ignore_if_contains": [".img_mod."],
        }

        normalized = _normalize_network_config(raw_network_config, process_config)

        self.assertEqual(
            normalized["network_kwargs"]["only_if_contains"],
            ["transformer_blocks.16."],
        )
        self.assertEqual(
            normalized["network_kwargs"]["ignore_if_contains"],
            [".img_mod."],
        )


if __name__ == "__main__":
    unittest.main()
