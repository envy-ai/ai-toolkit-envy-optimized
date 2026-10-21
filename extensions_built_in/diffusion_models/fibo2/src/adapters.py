"""Exact LoRA split/fused import, including independent ranks and per-layer scales."""
import torch
from torch.nn import functional as F


def canonical_adapter(state_dict):
    state = {key.replace('diffusion_model.', 'transformer.', 1) if key.startswith('diffusion_model.') else key: value
             for key, value in state_dict.items()}
    state = {key.replace('.lora_down.weight', '.lora_A.weight').replace('.lora_up.weight', '.lora_B.weight'): value
             for key, value in state.items()}
    for key in list(state):
        if key not in state or not key.endswith('.lora_A.weight'):
            continue
        source = key[:-len('.lora_A.weight')]
        sources, target = None, None
        if source.endswith('.attention.to_q'):
            root = source[:-len('to_q')]
            sources, target = [root+part for part in ('to_q', 'to_k', 'to_v')], root+'qkv'
        elif source.endswith('.to_k'):
            root = source[:-len('to_k')]
            sources, target = [root+'to_k', root+'to_v'], root+'to_kv'
        elif source.endswith('.w1'):
            root = source[:-len('w1')]
            sources, target = [root+'w1', root+'w3'], root+'w13'
        if sources is None:
            continue
        if any(key.startswith(target+'.') for key in state):
            raise ValueError(f'FIBO adapter contains both split and fused weights: {target}')
        downs, ups, scales = [], [], []
        for layer in sources:
            down = state.pop(layer+'.lora_A.weight', None)
            up = state.pop(layer+'.lora_B.weight', None)
            if down is None or up is None or down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise ValueError(f'Incomplete FIBO split LoRA projection: {layer}')
            rank = down.shape[0]
            alpha = state.pop(layer+'.alpha', down.new_tensor(rank))
            downs.append(down); ups.append(up); scales.append(alpha/rank)
        if target.endswith('.w13') and ups[0].shape[0] == 9536 and downs[0].shape[1] == 3584:
            ups = [F.pad(up, (0, 0, 0, 192)) for up in ups]
        # A block diagonal factorization preserves unequal projection ranks and
        # scales exactly. No dense model-sized delta is reconstructed.
        rank = sum(down.shape[0] for down in downs)
        state[target+'.lora_A.weight'] = torch.cat(downs)
        state[target+'.lora_B.weight'] = torch.block_diag(*[up*scale.to(up) for up, scale in zip(ups, scales)])
        state[target+'.alpha'] = downs[0].new_tensor(rank)
    out = {}
    for key, value in state.items():
        if '.attention.to_out.0.' in key:
            key = key.replace('.attention.to_out.0.', '.attention.out.')
        else:
            key = key.replace('.to_out.0.', '.to_out.')
        if key.endswith('.w2.lora_A.weight') and value.shape[1] == 9536:
            value = F.pad(value, (0, 192))
        out[key] = value
    return out
