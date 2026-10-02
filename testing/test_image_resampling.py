import unittest

import numpy as np
from PIL import Image

from toolkit.image_resampling import resize_mitchell


class MitchellResizeTest(unittest.TestCase):
    def test_preserves_constant_rgb_image_when_downscaling(self):
        source = np.full((31, 47, 3), (23, 101, 219), dtype=np.uint8)

        resized = resize_mitchell(Image.fromarray(source, "RGB"), (13, 9))

        self.assertEqual(resized.size, (13, 9))
        np.testing.assert_array_equal(
            np.asarray(resized),
            np.full((9, 13, 3), (23, 101, 219), dtype=np.uint8),
        )

    def test_supports_bucket_image_modes(self):
        for mode, shape in (("L", (17, 23)), ("LA", (17, 23, 2)), ("RGBA", (17, 23, 4))):
            with self.subTest(mode=mode):
                source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
                resized = resize_mitchell(Image.fromarray(source, mode), (11, 7))
                self.assertEqual(resized.mode, mode)
                self.assertEqual(resized.size, (11, 7))

    def test_identity_resize_returns_equal_copy(self):
        source = Image.fromarray(np.arange(12 * 8, dtype=np.uint8).reshape(8, 12), "L")

        resized = resize_mitchell(source, source.size)

        self.assertIsNot(resized, source)
        np.testing.assert_array_equal(np.asarray(resized), np.asarray(source))

    def test_rejects_non_positive_dimensions(self):
        source = Image.new("RGB", (4, 4))
        with self.assertRaises(ValueError):
            resize_mitchell(source, (0, 4))


if __name__ == "__main__":
    unittest.main()
