"""Strict factorized Hunyuan adapter mapping, independent of Comfy imports."""
from aitk_shared_models.protocol import SharedModelError

ATTENTION = ('.self_attn.qkv_proj', '.self_attn.o_proj')
SHARED = ('.mlp.shared_mlp.gate_and_up_proj', '.mlp.shared_mlp.down_proj')


def canonical_lora_map(model, factors, metadata):
    allowed = ATTENTION + (SHARED if metadata['preset'] == 'attention_shared_mlp' else ())
    projections = {name[:-len('.weight')]: (name, tuple(value.shape))
                   for name, value in model.named_parameters() if name.endswith('.weight')}
    result, groups = {}, {}
    for key, value in factors.items():
        suffix = next((suffix for suffix in ('.lora_A.weight', '.lora_B.weight', '.lora_down.weight', '.lora_up.weight', '.alpha') if key.endswith(suffix)), None)
        if suffix is None:
            raise SharedModelError(f'Unsupported snapshot tensor {key}')
        prefix = key[:-len(suffix)]
        if prefix not in projections or not prefix.endswith(allowed):
            raise SharedModelError(f'Snapshot projection does not match its target preset: {prefix}')
        groups.setdefault(prefix, {})[suffix] = value
        result[prefix] = projections[prefix][0]
    for prefix, group in groups.items():
        down = group.get('.lora_A.weight', group.get('.lora_down.weight'))
        up = group.get('.lora_B.weight', group.get('.lora_up.weight'))
        shape = projections[prefix][1]
        rank = metadata['rank']
        if down is None or up is None or len(shape) != 2 or tuple(down.shape) != (rank, shape[1]) or tuple(up.shape) != (shape[0], rank):
            raise SharedModelError(f'Incomplete/mismatched factorized snapshot {prefix}')
        alpha = group.get('.alpha')
        if alpha is not None and (alpha.numel() != 1 or float(alpha.item()) != float(metadata['alpha'])):
            raise SharedModelError(f'Snapshot alpha mismatch for {prefix}')
    return result
