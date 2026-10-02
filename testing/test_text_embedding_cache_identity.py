import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

from toolkit.dataloader_mixins import TextEmbeddingFileItemDTOMixin


class FakeTextEmbeddingItem(TextEmbeddingFileItemDTOMixin):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.caption = "a drawing"
        self.text_embedding_space_version = "test-space"
        self.encode_control_in_text_embeddings = False
        self.control_path = None
        self.control_video_paths = []
        self.encode_first_frame_in_text_embeddings = False
        self.cache_processed_control_text_embeddings = False
        self.dataset_config = SimpleNamespace(do_i2v=False)
        self.is_video = False


def bump_mtime(path: pathlib.Path):
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))


class TextEmbeddingCacheFileIdentityTests(unittest.TestCase):
    def test_plain_text_embedding_does_not_depend_on_source_image_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = pathlib.Path(tmp_dir) / "image.png"
            image_path.write_bytes(b"image")
            item = FakeTextEmbeddingItem(str(image_path))

            before = item._build_text_embedding_path()
            bump_mtime(image_path)
            after = item._build_text_embedding_path()

            self.assertEqual(before, after)

    def test_control_image_mtime_changes_embedding_cache_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = pathlib.Path(tmp_dir) / "image.png"
            control_path = pathlib.Path(tmp_dir) / "control.png"
            image_path.write_bytes(b"image")
            control_path.write_bytes(b"control")
            item = FakeTextEmbeddingItem(str(image_path))
            item.encode_control_in_text_embeddings = True
            item.control_path = str(control_path)

            before = item._build_text_embedding_path()
            bump_mtime(control_path)
            after = item._build_text_embedding_path()

            self.assertNotEqual(before, after)

    def test_control_video_size_changes_embedding_cache_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_path = pathlib.Path(tmp_dir) / "image.png"
            video_path = pathlib.Path(tmp_dir) / "reference.mp4"
            image_path.write_bytes(b"image")
            video_path.write_bytes(b"video")
            item = FakeTextEmbeddingItem(str(image_path))
            item.encode_control_in_text_embeddings = True
            item.control_video_paths = [str(video_path)]

            before = item._build_text_embedding_path()
            video_path.write_bytes(b"larger video")
            after = item._build_text_embedding_path()

            self.assertNotEqual(before, after)

    def test_self_reference_and_first_frame_follow_source_media_mtime(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = pathlib.Path(tmp_dir) / "video.mp4"
            video_path.write_bytes(b"video")
            item = FakeTextEmbeddingItem(str(video_path))
            item.encode_first_frame_in_text_embeddings = True
            item.dataset_config.do_i2v = True
            item.is_video = True

            first_frame_before = item._build_text_embedding_path()
            self_ref_before = item._build_text_embedding_path(dopsd_self_ref=True)
            bump_mtime(video_path)
            first_frame_after = item._build_text_embedding_path()
            self_ref_after = item._build_text_embedding_path(dopsd_self_ref=True)

            self.assertNotEqual(first_frame_before, first_frame_after)
            self.assertNotEqual(self_ref_before, self_ref_after)


if __name__ == "__main__":
    unittest.main()
