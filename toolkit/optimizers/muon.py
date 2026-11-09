"""Batched Muon for adapter and model parameters.

The Newton–Schulz coefficients, momentum rule, and learning-rate adjustments
follow the Fizgig Muon implementation (itself adapted from PyTorch Muon).
Each matrix is normalized independently before batched orthogonalization.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch


_DEFAULT_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
_MAX_BATCH_BYTES = 16 * 1024 * 1024
_MAX_BATCH_ITEMS = 64


def _adjust_lr(lr: float, mode: str | None, shape: tuple[int, int]) -> float:
    rows, columns = shape
    if mode is None or mode == "original":
        ratio = math.sqrt(max(1.0, rows / columns))
    elif mode == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    elif mode == "spectral_unclamped":
        ratio = math.sqrt(rows / columns)
    else:
        raise ValueError(f"Unsupported Muon adjust_lr_fn: {mode}")
    return lr * ratio


def _zeropower_batched(
    matrices: torch.Tensor,
    coefficients: tuple[float, float, float],
    steps: int,
    eps: float,
) -> torch.Tensor:
    """Orthogonalize a stack of equally shaped matrices, with a norm per item."""
    a, b, c = coefficients
    x = matrices.bfloat16()
    tall = x.size(-2) > x.size(-1)
    if tall:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, update, x, beta=a)
    return x.transpose(-2, -1) if tall else x


class BatchedMuon(torch.optim.Optimizer):
    """Muon with bounded batches of same-shape matrices and explicit vector updates.

    Tensors with more than two dimensions are reshaped to ``(size(0), -1)``.
    Vectors and scalars use Fizgig Muon's RMS-normalized momentum update.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_coefficients: tuple[float, float, float] = _DEFAULT_COEFFICIENTS,
        eps: float = 1e-7,
        ns_steps: int = 5,
        adjust_lr_fn: str | None = "match_rms_adamw",
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=ns_coefficients,
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_group(group: dict) -> None:
        for name in ("lr", "weight_decay", "momentum"):
            value = float(group[name])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Muon {name} must be finite and nonnegative")
        if not math.isfinite(float(group["eps"])) or group["eps"] <= 0:
            raise ValueError("Muon eps must be finite and positive")
        if not isinstance(group["ns_steps"], int) or not 0 <= group["ns_steps"] < 100:
            raise ValueError("Muon ns_steps must be between 0 and 99")
        if len(group["ns_coefficients"]) != 3 or not all(
            math.isfinite(float(value)) for value in group["ns_coefficients"]
        ):
            raise ValueError("Muon ns_coefficients must contain three finite values")
        if group["adjust_lr_fn"] not in (None, "original", "match_rms_adamw", "spectral_unclamped"):
            raise ValueError(f"Unsupported Muon adjust_lr_fn: {group['adjust_lr_fn']}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reject unsupported gradients before changing any parameters.
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None and param.grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

        for group in self.param_groups:
            lr = float(group["lr"])
            decay = float(group["weight_decay"])
            momentum = float(group["momentum"])
            buckets = defaultdict(list)

            for param in group["params"]:
                grad = param.grad
                if grad is None or param.numel() == 0:
                    continue
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad, memory_format=torch.preserve_format)
                buffer = state["momentum_buffer"]
                buffer.lerp_(grad, 1.0 - momentum)

                if param.ndim < 2:
                    update = grad.lerp(buffer, momentum) if group["nesterov"] else buffer
                    rms = update.float().pow(2).mean().sqrt().clamp(min=float(group["eps"]))
                    vector_update = (update.float() / rms).to(param.dtype) * 0.2
                    if decay:
                        param.mul_(1.0 - lr * decay)
                    param.add_(vector_update, alpha=-lr)
                    continue

                shape = (param.size(0), param.numel() // param.size(0))
                buckets[(shape, param.device, grad.dtype)].append((param, grad, buffer))

            for (shape, _device, _dtype), items in buckets.items():
                bytes_per_matrix = shape[0] * shape[1] * 2  # Newton–Schulz uses bfloat16.
                batch_size = max(1, min(_MAX_BATCH_ITEMS, _MAX_BATCH_BYTES // bytes_per_matrix))
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], shape)
                for start in range(0, len(items), batch_size):
                    chunk = items[start:start + batch_size]
                    matrices = torch.stack([
                        (grad.lerp(buffer, momentum) if group["nesterov"] else buffer).reshape(shape)
                        for _, grad, buffer in chunk
                    ])
                    ortho = _zeropower_batched(
                        matrices,
                        tuple(group["ns_coefficients"]),
                        int(group["ns_steps"]),
                        float(group["eps"]),
                    )
                    for (param, _, _), matrix in zip(chunk, ortho.unbind(0)):
                        if decay:
                            param.mul_(1.0 - lr * decay)
                        param.add_(matrix.reshape(param.shape).to(param.dtype), alpha=-adjusted_lr)

        return loss
