import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class JobsTableDatasetThumbnailTests(unittest.TestCase):
    def test_jobs_api_attaches_first_dataset_thumbnail_url(self):
        route_source = (REPO_ROOT / "ui/src/app/api/jobs/route.ts").read_text()

        self.assertIn("dataset_thumbnail_url", route_source)
        self.assertIn("getFirstDatasetImagePath", route_source)
        self.assertIn("findFirstImageInDirectory", route_source)
        self.assertIn("encodeURIComponent(firstImagePath)", route_source)
        self.assertIn("IMAGE_EXTENSIONS", route_source)

    def test_jobs_table_renders_unlabeled_thumbnail_column_before_name(self):
        table_source = (REPO_ROOT / "ui/src/components/JobsTable.tsx").read_text()

        thumbnail_column_index = table_source.index("key: 'dataset_thumbnail_url'")
        name_column_index = table_source.index("key: 'name'")

        self.assertLess(thumbnail_column_index, name_column_index)
        self.assertIn("title: ''", table_source)
        self.assertIn("row.dataset_thumbnail_url", table_source)
        self.assertIn("max-w-[100px]", table_source)
        self.assertIn("max-h-[100px]", table_source)


if __name__ == "__main__":
    unittest.main()
