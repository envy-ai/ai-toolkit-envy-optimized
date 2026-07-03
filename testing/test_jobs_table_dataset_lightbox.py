import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class JobsTableDatasetLightboxTests(unittest.TestCase):
    def test_job_dataset_images_api_returns_all_dataset_images_with_captions(self):
        route_source = (REPO_ROOT / "ui/src/app/api/jobs/[jobID]/dataset-images/route.ts").read_text()

        self.assertIn("readCaptionForImage", route_source)
        self.assertIn("findDatasetImages", route_source)
        self.assertIn("jobConfig?.config?.process?.[0]?.datasets", route_source)
        self.assertIn("dataset.caption_ext || 'txt'", route_source)
        self.assertIn("caption_path", route_source)
        self.assertIn("dataset_name", route_source)
        self.assertIn("url: `/api/img/${encodeURIComponent(imagePath)}`", route_source)

    def test_jobs_table_thumbnail_opens_job_dataset_lightbox(self):
        table_source = (REPO_ROOT / "ui/src/components/JobsTable.tsx").read_text()

        self.assertIn("JobDatasetLightbox", table_source)
        self.assertIn("datasetLightboxJob", table_source)
        self.assertIn("openDatasetLightbox", table_source)
        self.assertIn("row.dataset_thumbnail_path", table_source)
        self.assertIn("Browse dataset images", table_source)

    def test_lightbox_uses_npm_lightbox_with_captions(self):
        component_source = (REPO_ROOT / "ui/src/components/JobDatasetLightbox.tsx").read_text()
        layout_source = (REPO_ROOT / "ui/src/app/layout.tsx").read_text()

        self.assertIn("yet-another-react-lightbox", component_source)
        self.assertIn("plugins/captions", component_source)
        self.assertIn("plugins/zoom", component_source)
        self.assertIn("description:", component_source)
        self.assertIn("caption", component_source)
        self.assertIn("dataset_name", component_source)
        self.assertIn("yet-another-react-lightbox/styles.css", layout_source)
        self.assertIn("yet-another-react-lightbox/plugins/captions.css", layout_source)


if __name__ == "__main__":
    unittest.main()
