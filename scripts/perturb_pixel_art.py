"""Copy randomly displaced strips and boxes over PNGs without resampling."""

import argparse
import random
import shutil
import struct
import tempfile
from pathlib import Path

from PIL import Image, PngImagePlugin


OPERATION_KINDS = (
    "vertical_strip",
    "horizontal_strip",
    "cardinal_box",
)


def copy_shift(image, box, dx, dy):
    """Snapshot the current crop, then replace destination pixels verbatim."""
    crop = image.crop(box)
    # No cut or mask: keep the source and copy alpha/palette indices exactly.
    image.paste(crop, (box[0] + dx, box[1] + dy))


def _signed_shift(rng, min_shift, max_shift):
    return rng.choice((-1, 1)) * rng.randint(min_shift, max_shift)


def _random_interval(rng, extent, start_grid=1):
    cells = extent // start_grid
    size = rng.randint(1, cells)
    start = rng.randint(0, cells - size)
    return start * start_grid, (start + size) * start_grid


def random_operation(image, rng, min_shift=2, max_shift=16, start_grid=1):
    """Apply one uniformly chosen operation and return its kind, box and offset."""
    kind = rng.choice(OPERATION_KINDS)
    width, height = image.size
    _validate_grid(image, start_grid)
    dx = dy = 0
    if kind == "vertical_strip":
        left, right = _random_interval(rng, width, start_grid)
        top, bottom = 0, height
        dx = _signed_shift(rng, min_shift, max_shift)
    elif kind == "horizontal_strip":
        left, right = 0, width
        top, bottom = _random_interval(rng, height, start_grid)
        dy = _signed_shift(rng, min_shift, max_shift)
    else:
        left, right = _random_interval(rng, width, start_grid)
        top, bottom = _random_interval(rng, height, start_grid)
        # Exactly one axis changes, never a diagonal displacement.
        if rng.choice(("horizontal", "vertical")) == "horizontal":
            dx = _signed_shift(rng, min_shift, max_shift)
        else:
            dy = _signed_shift(rng, min_shift, max_shift)
    box = (left, top, right, bottom)
    copy_shift(image, box, dx, dy)
    return kind, box, dx, dy


def _validate_ranges(min_actions, max_actions, min_shift, max_shift):
    if not 1 <= min_actions <= max_actions:
        raise ValueError("Action counts must satisfy 1 <= min_actions <= max_actions")
    if not 1 <= min_shift <= max_shift:
        raise ValueError("Shift distances must satisfy 1 <= min_shift <= max_shift")


def _validate_grid(image, grid):
    if grid < 1 or image.width % grid or image.height % grid:
        raise ValueError(f"Image dimensions {image.size} must be multiples of positive grid {grid}")


def perturb_image(image, rng, min_actions=2, max_actions=6,
                  min_shift=2, max_shift=16, start_grid=1):
    """Return an altered copy; each action uses the result of previous actions."""
    _validate_ranges(min_actions, max_actions, min_shift, max_shift)
    if start_grid < 1:
        raise ValueError("start_grid must be positive")
    if image.width < 1 or image.height < 1:
        raise ValueError("Image dimensions must be nonempty")
    _validate_grid(image, start_grid)
    result = image.copy()
    for _ in range(rng.randint(min_actions, max_actions)):
        random_operation(result, rng, min_shift, max_shift, start_grid)
    return result


def _open_png(path):
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError("File is not a PNG image")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Animated PNGs are unsupported; supply single-frame PNGs")
            image.load()
            return image.copy()
    except Exception as error:
        raise ValueError(f"Cannot read PNG {path}: {error}") from error


def _png_save_options(image):
    # Preserve useful color/alpha metadata without converting pixel modes.
    info = image.info
    options = {key: info[key] for key in
               ("icc_profile", "transparency", "dpi", "exif") if key in info}
    chunks = PngImagePlugin.PngInfo()
    if "gamma" in info:
        chunks.add(b"gAMA", struct.pack(">I", round(info["gamma"] * 100000)))
    if "srgb" in info:
        chunks.add(b"sRGB", bytes((info["srgb"],)))
    if "chromaticity" in info:
        chunks.add(b"cHRM", struct.pack(">8I", *(
            round(value * 100000) for value in info["chromaticity"])))
    options["pnginfo"] = chunks
    return options


def _save_png(image, target, overwrite):
    # Stage the PNG so an interrupted encode cannot leave a truncated output.
    staged_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".png",
                                         delete=False) as staged:
            staged_path = Path(staged.name)
            image.save(staged, format="PNG", **_png_save_options(image))
        if overwrite:
            staged_path.replace(target)
        else:
            # Exclusive creation catches new collisions on mounted filesystems too.
            created = False
            try:
                with target.open("xb") as destination:
                    created = True
                    with staged_path.open("rb") as source:
                        shutil.copyfileobj(source, destination)
            except Exception:
                if created:
                    target.unlink()
                raise
    finally:
        if staged_path is not None:
            staged_path.unlink(missing_ok=True)


def run(input_folder, output_subdir="shifted", seed=None, overwrite=False,
        min_actions=2, max_actions=6, min_shift=2, max_shift=16, start_grid=1):
    """Process every top-level PNG; validate paths and inputs before writing."""
    _validate_ranges(min_actions, max_actions, min_shift, max_shift)
    if start_grid < 1:
        raise ValueError("start_grid must be positive")
    folder = Path(input_folder).expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise ValueError(f"Input folder is not a directory: {folder}")
    child = Path(output_subdir)
    if (not output_subdir or child.is_absolute() or len(child.parts) != 1
            or child.name in ("", ".", "..")):
        raise ValueError("--output-subdir must name a single child directory")
    output = folder / child
    if output.is_symlink():
        raise ValueError(f"Output directory must not be a symlink: {output}")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output path is not a directory: {output}")
    sources = sorted((path for path in folder.iterdir()
                      if path.is_file() and path.suffix.lower() == ".png"),
                     key=lambda path: (path.name.casefold(), path.name))
    if not sources:
        raise ValueError(f"No PNG files found in {folder}")
    original_paths = {source.resolve() for source in sources}
    for source in sources:
        target = output / source.name
        if target.resolve() in original_paths:
            raise ValueError(f"Output aliases an input file: {target}")
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise ValueError(f"Output must be a regular file: {target}")
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {target}; use --overwrite")
    # Decode all inputs first so a corrupt file cannot silently skip a PNG.
    for source in sources:
        original = _open_png(source)
        try:
            _validate_grid(original, start_grid)
        finally:
            original.close()
    output.mkdir(exist_ok=True)
    rng = random.Random(seed)
    for source in sources:
        original = _open_png(source)
        try:
            altered = perturb_image(original, rng, min_actions, max_actions,
                                    min_shift, max_shift, start_grid)
            try:
                _save_png(altered, output / source.name, overwrite)
            finally:
                altered.close()
        finally:
            original.close()
    return len(sources)


def main():
    parser = argparse.ArgumentParser(description=(
        "Alter every PNG in a folder with 2–6 copied strip/box shifts of "
        "2–16 actual image pixels; keep originals and canvas sizes intact."))
    parser.add_argument("input_folder", help="Folder containing PNG images (nonrecursive)")
    parser.add_argument("--output-subdir", default="shifted",
                        help="Single child directory for outputs (default: shifted)")
    parser.add_argument("--seed", type=int, help="Seed for reproducible random operations")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing output files")
    parser.add_argument("--min-actions", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=6)
    parser.add_argument("--box-grid", "--start-grid", dest="start_grid", type=int, default=1,
                        help="Align all crop boundaries to this pixel grid")
    args = parser.parse_args()
    try:
        count = run(args.input_folder, args.output_subdir, args.seed, args.overwrite,
                    min_actions=args.min_actions, max_actions=args.max_actions,
                    start_grid=args.start_grid)
    except Exception as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Wrote {count} PNGs to {Path(args.input_folder).expanduser() / args.output_subdir}")


if __name__ == "__main__":
    main()
