import contextlib
import io
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from toolkit.dataset_validation import DatasetValidationError
from extensions_built_in.sd_trainer.FizgigSliderTrainer import validate_image_slider_pairs
from extensions_built_in.sd_trainer.fizgig_multipoint import validate_multipoint_images


class DatasetValidationReportTests(unittest.TestCase):
    def image(self, folder, name, size=(64, 64)):
        folder.mkdir(exist_ok=True)
        Image.new('RGB', size).save(folder / name)

    def test_two_folder_reports_both_missing_directions_and_all_sizes_across_datasets(self):
        with tempfile.TemporaryDirectory() as root:
            datasets = []
            for group in ('one', 'two'):
                pos, neg = Path(root) / (group + '-pos'), Path(root) / (group + '-neg')
                self.image(pos, 'only-positive.png')
                self.image(neg, 'only-negative.webp')
                for name in ('bad1.png', 'bad2.png'):
                    self.image(pos, name)
                    self.image(neg, name, (32, 48))
                datasets.append(dict(folder_path=str(pos), control_path_1=str(neg), cache_latents_to_disk=True))
            with self.assertRaises(DatasetValidationError) as caught:
                validate_image_slider_pairs(datasets)
            report = str(caught.exception)
            for dataset in datasets:
                for key in ('folder_path', 'control_path_1'):
                    self.assertIn(dataset[key], report)
            for name in ('only-positive.png', 'only-negative.webp', 'bad1.png', 'bad2.png', '64 x 64', '32 x 48'):
                self.assertIn(name, report)
            self.assertEqual(report.count('Resolution mismatch'), 4)

    def test_multipoint_reports_missing_names_even_with_empty_folder_and_size_errors(self):
        with tempfile.TemporaryDirectory() as root:
            folders = [Path(root) / str(i) for i in range(3)]
            for folder in folders:
                folder.mkdir()
            self.image(folders[0], 'a.png')
            self.image(folders[1], 'b.png')
            self.image(folders[0], 'bad.png')
            self.image(folders[1], 'bad.png', (48, 32))
            points = [dict(id=str(i), strength=i) for i in range(3)]
            dataset = dict(cache_latents_to_disk=True, multipoint_images=[
                dict(point_id=str(i), folder_path=str(folder)) for i, folder in enumerate(folders)])
            with self.assertRaises(DatasetValidationError) as caught:
                validate_multipoint_images([dataset], points)
            report = str(caught.exception)
            for value in ('a.png', 'b.png', 'bad.png', '48 x 32', 'no usable images', *map(str, folders)):
                self.assertIn(value, report)

    def test_edit_references_need_names_but_may_have_different_resolution(self):
        with tempfile.TemporaryDirectory() as root:
            pos, neg, ref = [Path(root) / name for name in ('pos', 'neg', 'ref')]
            self.image(pos, 'a.png')
            self.image(neg, 'a.png')
            self.image(ref, 'a.png', (128, 128))
            dataset = dict(folder_path=str(pos), control_path_1=str(neg), control_path_2=str(ref), cache_latents_to_disk=True)
            validate_image_slider_pairs([dataset], allow_edit_controls=True)

    def test_startup_exits_without_traceback_and_marks_ui_job_stopped(self):
        import run
        with tempfile.TemporaryDirectory() as root:
            db = str(Path(root) / 'jobs.db')
            with sqlite3.connect(db) as conn:
                conn.execute('CREATE TABLE Job (id TEXT, status TEXT, info TEXT)')
                conn.execute("INSERT INTO Job VALUES ('test', 'running', '')")
            error = DatasetValidationError('Missing names: a.png\nResolution mismatch: b.png')
            error.sqlite_db_path = db
            output = io.StringIO()
            with patch.dict(os.environ, AITK_JOB_ID='test'), patch('sys.argv', ['run.py', 'test.yaml']), \
                    patch.object(run, 'get_job', side_effect=error), contextlib.redirect_stdout(output):
                with self.assertRaises(SystemExit) as caught:
                    run.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn('a.png', output.getvalue())
            self.assertIn('b.png', output.getvalue())
            self.assertNotIn('Traceback', output.getvalue())
            with sqlite3.connect(db) as conn:
                self.assertEqual(conn.execute('SELECT status, info FROM Job').fetchone(), ('stopped', str(error)))
