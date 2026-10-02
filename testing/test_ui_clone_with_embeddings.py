import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class CloneJobWithEmbeddingsTests(unittest.TestCase):
    def test_training_gear_menu_offers_embedding_cache_clone(self):
        source = (REPO_ROOT / "ui/src/components/JobActionBar.tsx").read_text()

        self.assertIn("Clone Job w/ Embeddings", source)
        self.assertIn("&withEmbeddings=1", source)

    def test_embedding_clone_enables_cache_in_the_editable_form(self):
        source = (REPO_ROOT / "ui/src/app/jobs/new/page.tsx").read_text()

        self.assertIn("searchParams.get('withEmbeddings') === '1'", source)
        self.assertIn("if (cloneWithEmbeddings)", source)
        self.assertIn("train.cache_text_embeddings = true", source)
        self.assertIn("train.unload_text_encoder = false", source)


if __name__ == "__main__":
    unittest.main()
