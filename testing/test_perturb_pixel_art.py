import random
import subprocess
import sys
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.perturb_pixel_art import copy_shift, perturb_image, random_operation, run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CopyShiftTests(unittest.TestCase):
    def test_shift_copies_without_cutting_and_uses_snapshot_for_overlap(self):
        image = Image.new("L", (5, 1))
        image.putdata([1, 2, 3, 4, 5])

        copy_shift(image, (0, 0, 4, 1), 1, 0)

        self.assertEqual(list(image.getdata()), [1, 1, 2, 3, 4])

    def test_shift_pastes_alpha_without_using_it_as_a_mask(self):
        image = Image.new("RGBA", (3, 1), (20, 30, 40, 255))
        image.putpixel((0, 0), (200, 100, 50, 0))

        copy_shift(image, (0, 0, 1, 1), 1, 0)

        self.assertEqual(image.getpixel((1, 0)), (200, 100, 50, 0))


class PerturbImageTests(unittest.TestCase):
    def test_action_count_stays_within_configured_default_bounds(self):
        source = Image.new("RGB", (40, 40), (12, 34, 56))

        for seed in (2, 17, 2026):
            with self.subTest(seed=seed), patch(
                "scripts.perturb_pixel_art.random_operation"
            ) as operation:
                perturb_image(source, random.Random(seed))
                self.assertGreaterEqual(operation.call_count, 2)
                self.assertLessEqual(operation.call_count, 6)

    def test_seeded_operations_are_repeatable_and_have_valid_geometry(self):
        source = Image.new("L", (48, 40))
        source.putdata([(x + y) % 256 for y in range(40) for x in range(48)])
        seen_kinds = set()

        for seed in range(64):
            image = source.copy()
            kind, box, dx, dy = random_operation(image, random.Random(seed))
            left, top, right, bottom = box

            self.assertIn(
                kind,
                {"vertical_strip", "horizontal_strip", "cardinal_box", "two_axis_box"},
            )
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(right, source.width)
            self.assertLessEqual(bottom, source.height)
            self.assertTrue(2 <= abs(dx) <= 16 or dx == 0)
            self.assertTrue(2 <= abs(dy) <= 16 or dy == 0)
            if kind == "vertical_strip":
                self.assertEqual((top, bottom), (0, source.height))
                self.assertEqual(dy, 0)
                self.assertNotEqual(dx, 0)
            elif kind == "horizontal_strip":
                self.assertEqual((left, right), (0, source.width))
                self.assertEqual(dx, 0)
                self.assertNotEqual(dy, 0)
            elif kind == "cardinal_box":
                self.assertTrue((dx == 0) != (dy == 0))
            else:
                self.assertNotEqual(dx, 0)
                self.assertNotEqual(dy, 0)

            expected = source.copy()
            copy_shift(expected, box, dx, dy)
            self.assertEqual(list(image.getdata()), list(expected.getdata()))
            seen_kinds.add(kind)

            duplicate = source.copy()
            duplicate_result = random_operation(duplicate, random.Random(seed))
            self.assertEqual(duplicate_result, (kind, box, dx, dy))
            self.assertEqual(list(duplicate.getdata()), list(image.getdata()))
            if len(seen_kinds) == 4:
                break

        self.assertEqual(
            seen_kinds,
            {"vertical_strip", "horizontal_strip", "cardinal_box", "two_axis_box"},
        )

    def test_seeded_perturbation_is_repeatable_and_preserves_palette_mode_and_size(self):
        source = Image.new("P", (40, 40))
        palette = [0, 0, 0, 255, 0, 0, 0, 255, 0] + [0] * (768 - 9)
        source.putpalette(palette)
        source.putdata([(x + y) % 3 for y in range(40) for x in range(40)])
        before = source.copy()

        first = perturb_image(source, random.Random(73))
        second = perturb_image(source, random.Random(73))

        self.assertIsNot(first, source)
        self.assertEqual(first.mode, "P")
        self.assertEqual(first.size, source.size)
        self.assertEqual(first.getpalette(), source.getpalette())
        self.assertEqual(list(first.getdata()), list(second.getdata()))
        self.assertEqual(list(source.getdata()), list(before.getdata()))


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tmp")
        self.root = Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    @staticmethod
    def _write_sources(folder):
        folder.mkdir(parents=True, exist_ok=True)
        for filename, offset in (("a-first.png", 11), ("B-second.PNG", 97)):
            image = Image.new("RGB", (40, 40))
            image.putdata(
                [((x * 5 + offset) % 256, (y * 7 + offset) % 256,
                  ((x + y) * 3 + offset) % 256)
                 for y in range(40) for x in range(40)]
            )
            image.save(folder / filename)
        Image.new("RGB", (40, 40), (80, 100, 120)).save(folder / "ignored.jpg")
        nested = folder / "nested"
        nested.mkdir()
        Image.new("RGB", (40, 40), (220, 170, 20)).save(nested / "nested.png")

    def test_cli_selects_top_level_pngs_and_seed_repeats_identical_outputs(self):
        first_input = self.root / "input_one"
        second_input = self.root / "input_two"
        self._write_sources(first_input)
        self._write_sources(second_input)

        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "perturb_pixel_art.py"),
        ]
        cli_options = ["--output-subdir", "artifacts", "--seed", "504"]
        subprocess.run(command + [str(first_input)] + cli_options, check=True)
        subprocess.run(command + [str(second_input)] + cli_options, check=True)

        first_outputs = first_input / "artifacts"
        second_outputs = second_input / "artifacts"
        expected = ["a-first.png", "B-second.PNG"]
        first_names = sorted(
            (path.name for path in first_outputs.iterdir()), key=str.casefold
        )
        second_names = sorted(
            (path.name for path in second_outputs.iterdir()), key=str.casefold
        )
        self.assertEqual(first_names, expected)
        self.assertEqual(second_names, expected)
        self.assertEqual(
            {name: (first_outputs / name).read_bytes() for name in expected},
            {name: (second_outputs / name).read_bytes() for name in expected},
        )
        rng = random.Random(504)
        for name in expected:
            with Image.open(first_input / name) as original:
                expected_pixels = perturb_image(original, rng)
            with Image.open(first_outputs / name) as output:
                self.assertEqual(list(output.getdata()), list(expected_pixels.getdata()))
        self.assertFalse((first_outputs / "nested.png").exists())
        self.assertFalse((first_outputs / "ignored.jpg").exists())
        self.assertTrue((first_input / "a-first.png").is_file())
        self.assertTrue((first_input / "B-second.PNG").is_file())

    def test_output_pngs_reopen_with_original_pixel_modes_and_palette(self):
        source = self.root / "modes"
        source.mkdir()
        Image.new("RGB", (40, 40), (20, 40, 60)).save(source / "rgb.png")
        Image.new("RGBA", (40, 40), (20, 40, 60, 0)).save(source / "rgba.png")
        Image.new("L", (40, 40), 91).save(source / "gray.png")

        palette_image = Image.new("P", (40, 40))
        palette = [0, 0, 0, 255, 0, 0] + [0] * (768 - 6)
        palette_image.putpalette(palette)
        palette_image.putdata([index % 2 for index in range(40 * 40)])
        palette_image.info["transparency"] = 0
        palette_image.save(source / "palette.png", transparency=0)

        words = array("H", ((index * 97) % 65536 for index in range(40 * 40)))
        Image.frombytes("I;16", (40, 40), words.tobytes()).save(source / "gray16.png")

        run(source, seed=31)

        expected_modes = {
            "rgb.png": "RGB",
            "rgba.png": "RGBA",
            "gray.png": "L",
            "palette.png": "P",
            "gray16.png": "I;16",
        }
        for filename, mode in expected_modes.items():
            with self.subTest(filename=filename), Image.open(
                source / "shifted" / filename
            ) as output:
                self.assertEqual(output.mode, mode)
                self.assertEqual(output.size, (40, 40))
        with Image.open(source / "shifted" / "palette.png") as output:
            self.assertEqual(output.getpalette(), palette_image.getpalette())
            self.assertEqual(output.info.get("transparency"), 0)

    def test_one_pixel_image_keeps_dimensions_pixels_and_source_bytes(self):
        source = self.root / "tiny"
        source.mkdir()
        original = Image.new("RGBA", (1, 1), (123, 45, 67, 89))
        original.save(source / "one.png")
        original_bytes = (source / "one.png").read_bytes()

        altered = perturb_image(original, random.Random(8))
        self.assertEqual(altered.size, (1, 1))
        self.assertEqual(altered.getpixel((0, 0)), (123, 45, 67, 89))
        self.assertEqual(run(source, seed=8), 1)

        self.assertEqual((source / "one.png").read_bytes(), original_bytes)
        with Image.open(source / "shifted" / "one.png") as output:
            self.assertEqual(output.size, (1, 1))
            self.assertEqual(output.mode, "RGBA")
            self.assertEqual(output.getpixel((0, 0)), (123, 45, 67, 89))

    def test_run_preserves_source_bytes_and_requires_explicit_overwrite(self):
        source = self.root / "source"
        self._write_sources(source)
        originals = {
            path.name: path.read_bytes()
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        }

        self.assertEqual(run(source, seed=19), 2)
        self.assertEqual(
            {name: (source / name).read_bytes() for name in originals}, originals
        )
        with self.assertRaises(FileExistsError):
            run(source, seed=19)
        self.assertEqual(run(source, seed=19, overwrite=True), 2)
        self.assertTrue((source / "shifted" / "a-first.png").is_file())

    def test_run_rejects_invalid_subdirectory_and_unreadable_png(self):
        source = self.root / "source"
        source.mkdir()
        Image.new("RGB", (40, 40)).save(source / "good.png")

        with self.assertRaises((ValueError, OSError)):
            run(source, output_subdir="../escape", seed=1)
        with self.assertRaises(ValueError):
            run(source, output_subdir=".", seed=1)

        target = self.root / "target"
        target.mkdir()
        (source / "linked").symlink_to(target, target_is_directory=True)
        with self.assertRaises(ValueError):
            run(source, output_subdir="linked", seed=1)

        (source / "broken.png").write_bytes(b"this is not a png")
        with self.assertRaises((OSError, ValueError)):
            run(source, seed=1)
        self.assertFalse((source / "shifted").exists())


if __name__ == "__main__":
    unittest.main()
