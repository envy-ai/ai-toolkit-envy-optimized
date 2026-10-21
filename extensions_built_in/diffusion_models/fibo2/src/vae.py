"""FLUX.2 VAE with bounded inference decoding; differentiable decode stays intact."""
import torch
from toolkit.models.v2.vae.flux2_kl import AutoEncoder


class Fibo2AutoEncoder(AutoEncoder):
    @torch.no_grad()
    def decode_tiled(self, latents, tile=64, stride=48):
        # One tile's activations at a time, overlapped linear blending on CPU.
        if type(tile) is not int or type(stride) is not int or not 0 < stride < tile:
            raise ValueError('VAE tile and stride must be integers with 0 < stride < tile')
        b, _, height, width = latents.shape
        if height <= tile and width <= tile:
            return super().decode(latents)
        def starts(length):
            return [0] if length <= tile else list(range(0, length-tile, stride)) + [length-tile]
        output = torch.zeros(b, 3, height*16, width*16, dtype=torch.float32, device='cpu')
        weights = torch.zeros(1, 1, height*16, width*16, dtype=torch.float32, device='cpu')
        for y in starts(height):
            for x in starts(width):
                end_y, end_x = min(y+tile, height), min(x+tile, width)
                value = super().decode(latents[:, :, y:end_y, x:end_x]).float().cpu()
                h, w = value.shape[-2:]
                wy, wx = torch.ones(h), torch.ones(w)
                overlap = (tile-stride)*16
                if y:
                    n = min(overlap, h); wy[:n] *= torch.linspace(0, 1, n)
                if end_y < height:
                    n = min(overlap, h); wy[-n:] *= torch.linspace(1, 0, n)
                if x:
                    n = min(overlap, w); wx[:n] *= torch.linspace(0, 1, n)
                if end_x < width:
                    n = min(overlap, w); wx[-n:] *= torch.linspace(1, 0, n)
                weight = (wy[:, None]*wx[None])[None, None]
                output[:, :, y*16:end_y*16, x*16:end_x*16] += value*weight
                weights[:, :, y*16:end_y*16, x*16:end_x*16] += weight
        if not torch.all(weights > 0):
            raise RuntimeError('FIBO2 tiled VAE decode left uncovered output pixels')
        return (output/weights).to(latents.device, latents.dtype)
