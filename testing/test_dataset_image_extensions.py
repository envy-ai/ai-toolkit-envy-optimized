import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.data_loader import image_extensions
from toolkit.dataloader_mixins import img_ext_list


class DatasetImageExtensionTests(unittest.TestCase):
    def test_dataset_scanner_accepts_jxl_images(self):
        self.assertIn(".jxl", image_extensions)
        self.assertTrue("frame.JXL".lower().endswith(tuple(image_extensions)))

    def test_same_folder_image_pool_accepts_jxl_images(self):
        self.assertIn(".jxl", img_ext_list)


if __name__ == "__main__":
    unittest.main()
