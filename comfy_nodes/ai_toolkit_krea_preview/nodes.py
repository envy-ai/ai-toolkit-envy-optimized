import json
import torch

from comfy_api.latest import io
import node_helpers

from .reference import composite_reference, prepare_reference


def _krea_template():
    # Schema discovery/registration should not import encoder/model-management
    # code or discover CUDA. Resolve Comfy's own template only during execution.
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
    return KREA2_TEMPLATE


class AIToolkitKreaEditConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='AIToolkitKreaEditConditioning', display_name='AI Toolkit Krea Edit Conditioning',
            category='ai-toolkit/krea2', description='Matched native reference preprocessing; encode reference latents once for both CFG branches.',
            inputs=[io.Clip.Input('clip'), io.Vae.Input('vae'),
                io.String.Input('prompt', multiline=True), io.String.Input('negative_prompt', multiline=True),
                io.Float.Input('cfg', default=1., min=0., max=100.),
                io.Int.Input('width', default=1024, min=16, max=16384, step=16),
                io.Int.Input('height', default=1024, min=16, max=16384, step=16),
                io.Int.Input('vlm_max_pixels', default=384 ** 2, min=1),
                io.Int.Input('control_image_max_pixels', default=1024 ** 2, min=1),
                io.Boolean.Input('match_target_res', default=False),
                io.Combo.Input('dtype', options=['float32', 'bfloat16', 'float16'], default='bfloat16'),
                io.String.Input('background', default='[0, 0, 0]'),
                *[input for i in range(1, 4) for input in (io.Image.Input(f'image{i}', optional=True), io.Mask.Input(f'mask{i}', optional=True))]],
            outputs=[io.Conditioning.Output('positive'), io.Conditioning.Output('negative')])

    @classmethod
    @torch.no_grad()
    def execute(cls, clip, vae, prompt, negative_prompt, cfg, width, height, vlm_max_pixels,
            control_image_max_pixels, match_target_res, dtype, background,
            image1=None, mask1=None, image2=None, mask2=None, image3=None, mask3=None):
        background = json.loads(background)
        if not isinstance(background, list) or len(background) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in background):
            raise ValueError('Reference background must be an RGB list of three integers from 0 to 255')
        if image2 is not None and image1 is None or image3 is not None and image2 is None:
            raise ValueError('Krea reference images must occupy consecutive slots')
        images, refs = [], []
        for image, mask in ((image1, mask1), (image2, mask2), (image3, mask3)):
            if image is None:
                continue
            image = composite_reference(image, mask, background)
            vlm, pixels = prepare_reference(image, dtype=dtype, vlm_max_pixels=vlm_max_pixels,
                control_image_max_pixels=control_image_max_pixels, match_target_res=match_target_res, width=width, height=height)
            images.append(vlm)
            refs.append(vae.encode(pixels))
        prefix = ''.join(f'Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>' for i in range(len(images)))
        template = _krea_template()
        def encode(text):
            tokens = clip.tokenize(prefix + text, images=images, llama_template=template)
            result = clip.encode_from_tokens_scheduled(tokens)
            return node_helpers.conditioning_set_values(result, {'reference_latents': refs}, append=True) if refs else result
        positive = encode(prompt)
        negative = encode(negative_prompt) if cfg > 1 else positive
        return io.NodeOutput(positive, negative)
