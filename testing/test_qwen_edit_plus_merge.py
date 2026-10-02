import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus import QwenImageEditPlusModel


class QwenEditPlusMergeTests(unittest.TestCase):
    def test_batched_control_images_keep_one_reference_per_prompt_and_skip_lm_head(self):
        model = QwenImageEditPlusModel.__new__(QwenImageEditPlusModel)
        model.device_torch = torch.device('cpu')
        model.pipeline = SimpleNamespace(text_encoder=SimpleNamespace(device=torch.device('cpu')))
        model._encode_qwen_prompt_without_lm_head = Mock(
            return_value=(torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.long))
        )
        controls = torch.zeros(2, 3, 32, 32)

        with patch(
            'extensions_built_in.diffusion_models.qwen_image.qwen_image_edit_plus.F.interpolate',
            side_effect=lambda image, **kwargs: image,
        ):
            embeds = model.get_prompt_embeds(['first', 'second'], controls)

        self.assertEqual(embeds.text_embeds.shape, (2, 2, 3))
        self.assertEqual(model._encode_qwen_prompt_without_lm_head.call_count, 2)
        for call in model._encode_qwen_prompt_without_lm_head.call_args_list:
            self.assertEqual(len(call.kwargs['image']), 1)
            self.assertEqual(call.kwargs['image'][0].shape, (1, 3, 32, 32))


if __name__ == '__main__':
    unittest.main()
