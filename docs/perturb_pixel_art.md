# Pixel-art perturbation

Copy randomly shifted strips and rectangular patches over every PNG directly
inside a folder:

```bash
conda run -n ai-toolkit python scripts/perturb_pixel_art.py /path/to/images
```

Outputs keep the original filenames in `/path/to/images/shifted`. Originals are
unchanged. PNG extensions are matched without regard to case; nested folders
are excluded. There is no fixed image count.

Each image receives 2–6 operations, with each of these three kinds equally likely:

- A full-height vertical strip copied left or right.
- A full-width horizontal strip copied up or down.
- A rectangular box copied in one cardinal direction.

Each operation moves along exactly one axis by 2–16 **actual image pixels**, chosen independently of
the enlarged pixel-art grid. Random strip widths, strip heights and boxes may
span any size from one pixel to the whole image. Each crop snapshots the current
image after previous operations, then replaces destination pixels directly.
There is no cutting, gap filling, interpolation or alpha blending. Pixels in
the source stay where they were except where the destination overlaps them.
Canvas size, pixel mode, alpha and palette are retained, along with useful PNG
color metadata. Displaced pixels outside the canvas are clipped; tiny images
can receive fully off-canvas operations with no visible change.

Use a seed to reproduce a run with the same input filenames, images and Pillow
version. Choose another output child directory or explicitly replace outputs:

```bash
conda run -n ai-toolkit python scripts/perturb_pixel_art.py /path/to/images \
  --output-subdir shifted-v2 --seed 42
conda run -n ai-toolkit python scripts/perturb_pixel_art.py /path/to/images \
  --seed 42 --overwrite
```

`--output-subdir` accepts one child-directory name, never the input directory,
a nested path or an output symlink. Existing filename collisions fail unless
`--overwrite` is present. Unreadable files, non-PNG files with a `.png` suffix,
animated PNGs and empty input folders produce a nonzero exit. Input decoding
and output-path checks happen before writing any images.

For 1–3 operations with all crop corners on a 32-pixel grid:

```bash
conda run -n ai-toolkit python scripts/perturb_pixel_art.py /path/to/images \
  --min-actions 1 --max-actions 3 --box-grid 32
```

All source crop boundaries (left, top, right and bottom) are multiples of the
grid, including strips. Image dimensions must also be multiples of the grid;
invalid inputs fail before output is written. Displacement remains 2–16 pixels,
so destination corners intentionally move off-grid. `--start-grid` is an alias
for `--box-grid`. Defaults retain 2–6 operations with a one-pixel grid.
