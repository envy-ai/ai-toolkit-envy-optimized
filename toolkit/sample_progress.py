"""Per-image denoising progress shared by internal preview samplers."""

from tqdm.auto import tqdm


def configure_sample_progress(pipeline, image_index, image_count, phase=None):
    """Configure a fresh step bar, including for pipelines supplied by callers."""
    description = f"Sample {image_index + 1}/{image_count}"
    if phase:
        description += f" ({phase})"
    config = dict(disable=False, desc=description, leave=True, position=0,
                  unit="step", dynamic_ncols=True, miniters=1)
    def configure(target):
        setter = getattr(target, "set_progress_bar_config", None)
        if setter is not None:
            # Diffusers replaces the entire config; preserve caller options
            # such as the output stream when enabling and labeling this bar.
            options = dict(getattr(target, "_progress_bar_config", None) or {})
            options.update(config)
            setter(**options)

    configure(pipeline)
    # Diffusers modular pipelines only configure their immediate children.
    # Anima's denoising loop is nested deeper, so configure those blocks too.
    def configure_blocks(blocks):
        for block in getattr(blocks, "sub_blocks", {}).values():
            configure(block)
            configure_blocks(block)

    configure_blocks(getattr(pipeline, "_blocks", None))


class SampleProgressMixin:
    """Diffusers-style progress API for custom sampling loops."""

    def set_progress_bar_config(self, **kwargs):
        # Keep configuration on the instance; partial updates retain disable,
        # stream and other options set by standalone callers.
        config = dict(getattr(self, "_progress_bar_config", {}))
        config.update(kwargs)
        self._progress_bar_config = config

    def progress_bar(self, iterable=None, total=None):
        config = dict(desc="Sampling", leave=True, position=0, unit="step",
                      dynamic_ncols=True, miniters=1)
        config.update(getattr(self, "_progress_bar_config", {}))
        return tqdm(iterable=iterable, total=total, **config)
