"""Lightweight tokenizer/processor lifecycle component; never owns the backbone."""
from pathlib import Path

import torch
from torch import nn

from .._mixin import OstrisModelMixin
from extensions_built_in.diffusion_models.hunyuan_image_3.src.config import pinned_config, validate_variant


class HunyuanImage3Conditioning(nn.Module, OstrisModelMixin):
    def __init__(self, tokenizer, processor=None, vision=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.processor = processor
        self.vision_model = vision
        self.register_buffer('_placement', torch.zeros((), dtype=torch.bfloat16), persistent=False)

    @property
    def device(self):
        return self._placement.device

    @property
    def dtype(self):
        return self._placement.dtype

    @classmethod
    def create(cls, variant, assets, vision_path=None, revision=None):
        from tokenizers import Tokenizer
        if Path(assets).is_dir():
            tokenizer_path = str(Path(assets) / 'tokenizer.json')
        else:
            from huggingface_hub import hf_hub_download
            tokenizer_path = hf_hub_download(assets, 'tokenizer.json', revision=revision)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        processor = vision = None
        if vision_path:
            validate_variant(variant, vision_path)
            from transformers import Siglip2ImageProcessor
            from extensions_built_in.diffusion_models.hunyuan_image_3.src.siglip2 import Siglip2VisionTransformer, Siglip2SdpaAttention
            config = pinned_config(variant)
            # Official tower contains a pooling head unused by image conditioning.
            with torch.device('meta'):
                vision = Siglip2VisionTransformer(dict(config['vit'], vision_use_head=False))
            from ..diffusion_models.hunyuan_image_3 import CheckpointReader
            reader = CheckpointReader(vision_path)
            try:
                state = {key.removeprefix('vision_model.'): reader.tensor(key)
                         for key in reader.keys if key.startswith('vision_model.') and not key.startswith('vision_model.head.')}
                vision.load_state_dict(state, assign=True)
            finally:
                reader.close()
            # SDPA avoids materializing head-wise probabilities during preprocessing.
            for block in vision.encoder.layers:
                block.self_attn.__class__ = Siglip2SdpaAttention
            vision.requires_grad_(False).eval()
            processor = Siglip2ImageProcessor(**config['vit_processor'])
        return cls(tokenizer, processor, vision)

    @torch.no_grad()
    def encode_reference(self, image, device, dtype):
        if self.vision_model is None:
            raise ValueError('Edit conditioning needs model.model_kwargs.vision_path')
        self.vision_model.to(device=device, dtype=dtype)
        encoded = self.processor(images=image, return_tensors='pt')
        pixels = encoded['pixel_values'].to(device=device, dtype=dtype)
        mask = encoded['pixel_attention_mask'].to(device)
        grid = encoded['spatial_shapes'].to(device)
        output = self.vision_model(pixels, mask, grid, output_attentions=False)
        # Padded 1024-slot outputs are retained exactly as the reference tokenizer expects.
        return output.last_hidden_state.detach().cpu(), grid.detach().cpu()
