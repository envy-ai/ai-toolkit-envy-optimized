"""Sequence-length hints for known transformer blocks compiled during training.

The pre-hook runs eagerly, before Dynamo traces a block. Only axes that carry
tokens in the block's forward contract are marked; equal-sized feature, head,
or modulation axes must stay static.
"""

import inspect

import torch


_SEQUENCE_INPUTS = {
    ("moss_image.src.transformer", "_Block"): (
        ("x", 1), ("rope", 1), ("mask", "attention_mask"),
    ),
    ("krea2.src.mmdit", "SingleStreamBlock"): (
        ("x", 1), ("freqs", 1), ("mask", "attention_mask"),
    ),
    ("qwen_image_2.src.transformer", "QwenImage21TransformerBlock"): (
        ("hidden_states", 1), ("rotary_emb", 0),
        ("target_token_mask", 0), ("key_valid", 1),
        ("attention_mask", "attention_mask"),
    ),
}


def sequence_length_hook(block):
    """Return a forward pre-hook for a recognized block, or ``None``.

    Return ``None`` for other models rather than guessing which tensor axes are
    token lengths. The hook accepts both positional and keyword forward inputs.
    """
    module_name = type(block).__module__
    block_name = type(block).__name__
    spec = next((value for (suffix, name), value in _SEQUENCE_INPUTS.items()
                 if name == block_name and module_name.endswith(suffix)), None)
    if spec is None:
        return None

    parameters = tuple(inspect.signature(block.forward).parameters)
    positions = {name: parameters.index(name) for name, _ in spec}
    sequence_name = spec[0][0]

    def get_input(name, args, kwargs):
        index = positions[name]
        return args[index] if index < len(args) else kwargs.get(name)

    def pre_hook(_compiled_block, args, kwargs):
        sequence = get_input(sequence_name, args, kwargs)
        if not (torch.is_tensor(sequence) and sequence.ndim == 3):
            return
        length = sequence.shape[1]
        for name, axis in spec:
            value = get_input(name, args, kwargs)
            if not torch.is_tensor(value):
                continue
            if axis == "attention_mask":
                # SDPA masks may include a cached-reference key tail. Query and
                # key dimensions are distinct contracts, even when both equal L.
                if value.ndim == 4:
                    for mask_axis in (2, 3):
                        if value.shape[mask_axis] == length:
                            torch._dynamo.maybe_mark_dynamic(value, mask_axis)
                continue
            if value.ndim > axis and value.shape[axis] == length:
                torch._dynamo.maybe_mark_dynamic(value, axis)

    return pre_hook
