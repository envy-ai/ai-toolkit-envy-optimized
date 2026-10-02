import torch


FREQUENCY_FILTER_TYPES = {"low_pass", "high_pass", "band_pass", "notch"}


def _smooth_period_step(periods, edge, transition):
    """0 below ``edge``, 1 above it, with a smooth pixel-period transition."""
    if transition <= 0:
        return (periods >= edge).to(periods.dtype)
    half_transition = transition / 2.0
    position = ((periods - (edge - half_transition)) / transition).clamp(0.0, 1.0)
    return position.square() * (3.0 - 2.0 * position)


def build_pixel_frequency_mask(
    height,
    width,
    filter_type,
    cutoff_period=18.0,
    min_period=14.0,
    max_period=28.0,
    transition=4.0,
    device=None,
    dtype=torch.float32,
):
    """Build an rFFT mask whose thresholds are expressed in output pixels.

    Spatial frequency is radial, in cycles per pixel. Low-pass therefore keeps
    long periods, high-pass keeps short periods, band-pass keeps the selected
    period interval, and notch keeps everything outside that interval.
    """
    if filter_type not in FREQUENCY_FILTER_TYPES:
        raise ValueError(
            f"Unknown frequency filter type {filter_type!r}; expected one of "
            f"{sorted(FREQUENCY_FILTER_TYPES)}"
        )
    if height <= 0 or width <= 0:
        raise ValueError("Frequency-filter dimensions must be positive")
    if cutoff_period <= 0 or min_period <= 0 or max_period <= 0:
        raise ValueError("Frequency-filter periods must be greater than zero")
    if transition < 0:
        raise ValueError("Frequency-filter transition must be zero or greater")
    if filter_type in {"band_pass", "notch"} and min_period >= max_period:
        raise ValueError("frequency_loss_min_period must be less than frequency_loss_max_period")

    fy = torch.fft.fftfreq(height, d=1.0, device=device)
    fx = torch.fft.rfftfreq(width, d=1.0, device=device)
    radial_frequency = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    periods = torch.where(
        radial_frequency > 0,
        radial_frequency.reciprocal(),
        torch.full_like(radial_frequency, float("inf")),
    )

    if filter_type in {"low_pass", "high_pass"}:
        low_pass = _smooth_period_step(periods, float(cutoff_period), float(transition))
        mask = low_pass if filter_type == "low_pass" else 1.0 - low_pass
    else:
        above_min = _smooth_period_step(periods, float(min_period), float(transition))
        above_max = _smooth_period_step(periods, float(max_period), float(transition))
        band_pass = above_min * (1.0 - above_max)
        mask = band_pass if filter_type == "band_pass" else 1.0 - band_pass

    return mask.to(dtype=dtype)


def frequency_filtered_pixel_loss(
    prediction,
    target,
    filter_type,
    cutoff_period=18.0,
    min_period=14.0,
    max_period=28.0,
    transition=4.0,
    reduction="mean",
):
    """Phase-sensitive pixel loss restricted by a spatial-frequency filter.

    The last two tensor dimensions are treated as image-space height and width.
    Other dimensions (channels, and video time when present) are independent.
    With ``reduction='none'``, returns one value per batch item.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Frequency-loss prediction and target shapes differ: "
            f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if prediction.ndim < 3:
        raise ValueError("Frequency loss expects a batch with spatial dimensions")

    error = prediction.float() - target.float()
    height, width = error.shape[-2:]
    mask = build_pixel_frequency_mask(
        height=height,
        width=width,
        filter_type=filter_type,
        cutoff_period=cutoff_period,
        min_period=min_period,
        max_period=max_period,
        transition=transition,
        device=error.device,
        dtype=error.dtype,
    )
    spectrum = torch.fft.rfft2(error, dim=(-2, -1), norm="ortho")
    filtered_error = torch.fft.irfft2(
        spectrum * mask,
        s=(height, width),
        dim=(-2, -1),
        norm="ortho",
    )

    # Normalize for the fraction of the spectrum selected so the configured
    # weight remains useful for narrow bands as well as broad pass filters.
    active_fraction = mask.square().mean().clamp_min(1.0 / (height * width))
    per_element = filtered_error.square() / active_fraction
    per_sample = per_element.reshape(per_element.shape[0], -1).mean(dim=1)

    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unsupported frequency-loss reduction: {reduction!r}")


def _haar_dwt2(tensor):
    """Return one-level orthonormal Haar subbands for the last two axes.

    This is the transform that was previously supplied by
    ``pytorch_wavelets.DWTForward(J=1, mode="zero", wave="haar")``. Keeping the
    small transform here avoids importing pytorch_wavelets, whose package
    initializer relies on the removed ``pkg_resources`` module. Odd dimensions
    are zero-padded on the bottom and right, matching the former zero mode.
    """
    if tensor.ndim < 2:
        raise ValueError("Haar DWT expects at least two spatial dimensions")

    pad_height = tensor.shape[-2] % 2
    pad_width = tensor.shape[-1] % 2
    if pad_height or pad_width:
        tensor = torch.nn.functional.pad(tensor, (0, pad_width, 0, pad_height))

    top_left = tensor[..., 0::2, 0::2]
    top_right = tensor[..., 0::2, 1::2]
    bottom_left = tensor[..., 1::2, 0::2]
    bottom_right = tensor[..., 1::2, 1::2]

    low_low = (top_left + top_right + bottom_left + bottom_right) * 0.5
    low_high = (top_left + top_right - bottom_left - bottom_right) * 0.5
    high_low = (top_left - top_right + bottom_left - bottom_right) * 0.5
    high_high = (top_left - top_right - bottom_left + bottom_right) * 0.5
    return low_low, low_high, high_low, high_high


def wavelet_loss(model_pred, latents, noise):
    model_pred = model_pred.float()
    latents = latents.float()
    noise = noise.float()
    with torch.no_grad():
        model_input = torch.cat(_haar_dwt2(latents), dim=1)

    # reverse the noise to get the model prediction of the pure latents
    model_pred = noise - model_pred

    model_pred = torch.cat(_haar_dwt2(model_pred), dim=1)

    return torch.nn.functional.mse_loss(model_pred, model_input, reduction="none")


def stepped_loss(model_pred, latents, noise, noisy_latents, timesteps, scheduler):
    # this steps the on a 20 step timescale from the current step (50 idx steps ahead)
    # and then reconstructs the original image at that timestep. This should lessen the error
    # possible in high noise timesteps and make the flow smoother.
    bs = model_pred.shape[0]

    noise_pred_chunks = torch.chunk(model_pred, bs)
    timestep_chunks = torch.chunk(timesteps, bs)
    noisy_latent_chunks = torch.chunk(noisy_latents, bs)
    noise_chunks = torch.chunk(noise, bs)

    x0_pred_chunks = []

    for idx in range(bs):
        model_output = noise_pred_chunks[idx]  # predicted noise (same shape as latent)
        timestep = timestep_chunks[idx]  # scalar tensor per sample (e.g., [t])
        sample = noisy_latent_chunks[idx].to(torch.float32)
        noise_i = noise_chunks[idx].to(sample.dtype).to(sample.device)

        # Initialize scheduler step index for this sample
        scheduler._step_index = None
        scheduler._init_step_index(timestep)

        # ---- Step +50 indices (or to the end) in sigma-space ----
        sigma = scheduler.sigmas[scheduler.step_index]
        target_idx = min(scheduler.step_index + 50, len(scheduler.sigmas) - 1)
        sigma_next = scheduler.sigmas[target_idx]

        # One-step update along the model-predicted direction
        stepped = sample + (sigma_next - sigma) * model_output

        # ---- Inverse-Gaussian recovery at the target timestep ----
        t_01 = (
            (scheduler.sigmas[target_idx]).to(stepped.device).to(stepped.dtype)
        )
        original_samples = (stepped - t_01 * noise_i) / (1.0 - t_01)
        x0_pred_chunks.append(original_samples)

    predicted_images = torch.cat(x0_pred_chunks, dim=0)

    return torch.nn.functional.mse_loss(
        predicted_images.float(),
        latents.float().to(device=predicted_images.device),
        reduction="none",
    )
