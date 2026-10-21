# Upstream integration at 04733873

The FIBO 2 integration starts from fork revision `1dcaaf91`. Official
`ostris/ai-toolkit` was fetched on 2026-10-20 at
`04733873cf1da8060fce01316ec45daedbd4a918`. Its tree contains no FIBO 2
implementation. The previous upstream integration was `31f4d92f`.

The two new upstream commits add Docker's flash-linear-attention dependency,
make Qwen 3.8's FLA imports lazy, and batch MOSS text-only prompt encoding.
The Docker dependency, lazy imports, and version update are integrated.

MOSS retains the fork's existing unpadded, per-prompt text encoding. Upstream's
right-padded batch increases peak encoder activation memory with the number
and maximum length of prompts; individual output slices can also retain the
whole padded output allocation. Keeping per-prompt forwards preserves the
documented low-memory behavior and all existing reference-image handling.
This is an intentional merge resolution, rather than an encoder correctness
claim about the upstream implementation.

No global Diffusers or Transformers dependency is replaced for FIBO 2.
The preservation requirements and earlier qualification evidence in
`upstream_merge_31f4d92f.md` continue to apply.

Focused merge verification is recorded with the FIBO 2 integration results.
