"""Optional ComfyUI extension; importing preprocessing alone loads no models."""


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension
    from .nodes import AIToolkitKreaEditConditioning

    class KreaPreviewExtension(ComfyExtension):
        async def get_node_list(self):
            return [AIToolkitKreaEditConditioning]

    return KreaPreviewExtension()
