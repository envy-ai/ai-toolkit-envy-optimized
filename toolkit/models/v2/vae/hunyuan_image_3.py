"""Exact Tencent Hunyuan image VAE, including its spatial tiling/checkpoints."""
from extensions_built_in.diffusion_models.hunyuan_image_3.src.autoencoder_kl_3d import AutoencoderKLConv3D
from extensions_built_in.diffusion_models.hunyuan_image_3.src.config import pinned_config
from .._mixin import OstrisModelMixin


class HunyuanImage3VAE(AutoencoderKLConv3D, OstrisModelMixin):
    aitk_subfolder = None

    @classmethod
    def aitk_load_config(cls, path, subfolder=None):
        return pinned_config()['vae']

    @classmethod
    def aitk_from_pretrained(cls, path, subfolder=None, dtype=None, **kwargs):
        from ..diffusion_models.hunyuan_image_3 import CheckpointReader
        reader = CheckpointReader(path)
        try:
            state = {key.removeprefix('vae.'): reader.tensor(key)
                     for key in reader.keys if key.startswith('vae.')}
            return cls.load_from_state_dict(state, dtype, config=pinned_config()['vae'])
        finally:
            reader.close()
