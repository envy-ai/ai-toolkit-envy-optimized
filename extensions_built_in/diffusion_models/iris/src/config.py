"""Iris architecture defaults, adapted from speridlabs/iris-3b (Apache-2.0)."""
from dataclasses import dataclass, field

@dataclass
class PixelStageConfig:
    """Pixel-level output refinement pathway (PiT blocks)."""

    enabled: bool = True
    depth: int = 4
    hidden_size: int = 16  # PiT per-pixel channel width
    attn_hidden_size: int = 1280  # PiT per-patch compacted-attention width
    num_heads: int = 10
    mlp_ratio: float = 4.0
    # "pre": DiT-style input modulation + gates (6 params/pixel).
    # "post": affine on branch outputs, no gates (4 params/pixel).
    modulation: str = "post"
    abs_pos_embed: bool = True  # fixed full-resolution 2D sincos on PiT pixel tokens


@dataclass
class ModelConfig:
    # patch-level stage
    block: str = "single_stream"  # mmdit (dual-stream) | single_stream
    # Hybrid trunk: the first ``dual_depth`` blocks are dual-stream MM-DiT, the
    # rest are ``block``. A dual block costs exactly 2x the parameters of a
    # single-stream one at the same FLOPs, so this allocates parameters at fixed
    # compute. 0 keeps the homogeneous ``block`` stack.
    dual_depth: int = 8
    # The last block's text output tokens are discarded downstream. "drop" stops
    # building that block's text output path (projection, MLP, their norms and
    # gates), so no parameter is left gradient-free and DDP no longer needs
    # find_unused_parameters. Text still enters that block's attention as keys
    # and values, so the image tokens still read the caption.
    final_block_text: str = "keep"  # keep | drop
    hidden_size: int = 2560
    depth: int = 24
    num_heads: int = 20
    # grouped-query attention; ``None`` keeps full MHA with a fused QKV projection
    num_kv_heads: int | None = 5
    gated_attention: bool = True  # token-wise sigmoid gate before the attention output projection
    sandwich_norm: bool = True  # RMSNorm on each branch output before its residual add
    patch_size: int = 16
    in_channels: int = 3
    mlp_ratio: float = 4.0  # SwiGLU applies the 2/3 width rule on top
    qkv_bias: bool = False
    qk_norm: bool = True
    norm_eps: float = 1e-6
    attn_backend: str = "sdpa"  # sdpa | torch_flash | torch_cudnn | fa3 | fa4
    # patch-block adaLN parameterization: "per_block" = one Linear(D, 6D) per
    # block per stream; "shared_lowrank" = one shared core per stream plus a
    # per-block rank-r conditional residual U_i V_i; "shared_bias" = the same
    # shared core plus a per-block learned bias, so blocks share one timestep
    # response and differ only by a constant offset.
    modulation: str = "shared_bias"
    modulation_rank: int = 64  # shared_lowrank only
    # timestep conditioning
    timestep_max_period: float = 10.0  # flow time lives in [0, 1000]
    # True zero-initializes every adaLN projection (adaLN-zero), so the patch
    # blocks start as identity maps; False keeps PyTorch's default init
    adaln_zero_init: bool = True
    # positional encoding
    rope_theta: float = 10000.0
    rope_scale: float = 16.0  # image RoPE coords span [0, scale] at any resolution
    # "square": both axes normalized to [0, scale] independently, so the
    # encoding is aspect-blind and its angular step is anisotropic on
    # non-square grids. "isotropic": one shared step, aspect ratio preserved.
    rope_aspect: str = "isotropic"
    # reserve the N slowest (x, y) frequency pairs for a frame axis, so a second
    # image can be tagged in-context after pretraining (see nn/rope.py). Those
    # pairs are numerically inert under rope_theta=1e4 + rope_scale=16, and a
    # constant frame index cancels in attention, so 0 vs N is behaviorally
    # equivalent for single-image training.
    rope_frame_pairs: int = 0
    rope_frame_theta: float = 10.0
    text_rope: bool = True
    text_rope_theta: float = 10000.0
    text_abs_pos_embed: bool = True  # learned N(0,1) positional table on text tokens
    # text stream
    text_dim: int = 2560
    text_len: int = 300
    # text adapter capacity: "linear" = Linear + RMSNorm; "blocks2" adds two
    # token-axis transformer blocks; "lap_blocks2" first aggregates selected
    # frozen-encoder layers, then applies the same token-axis refiner.
    text_adapter: str = "lap_blocks2"
    text_lap_num_layers: int = 12  # one per text_encoder.hidden_layers entry
    text_lap_num_heads: int = 32
    text_lap_mlp_ratio: float = 1.3
    # REPA feature capture (1-based block index; 0 disables)
    repa_layer: int = 10
    pixel: PixelStageConfig = field(default_factory=PixelStageConfig)

    def validate(self) -> None:
        if not 0 <= self.dual_depth <= self.depth:
            raise ValueError(
                f"model.dual_depth must be in [0, model.depth={self.depth}], got {self.dual_depth}; "
                "it is the number of leading dual-stream blocks, not a depth increment"
            )
        if self.final_block_text not in ("keep", "drop"):
            raise ValueError(f"unknown model.final_block_text '{self.final_block_text}' (keep | drop)")
