"""Opt-in custom node extension; disabled by default, no live deployment side effects."""
from comfy_api.latest import ComfyExtension
from .nodes import AITKSharedHunyuanImage3Loader, AITKSharedLoRASnapshot, AITKSharedModelStatus
from .execution_lease import capabilities, enable_from_environment


class AITKSharedModelsExtension(ComfyExtension):
    async def get_node_list(self):
        return [AITKSharedHunyuanImage3Loader, AITKSharedLoRASnapshot, AITKSharedModelStatus]


async def comfy_entrypoint():
    enable_from_environment()
    # Only register a CPU metadata endpoint. The loader refuses use until its
    # whole-prompt guard is enabled; observers never trigger GPU work.
    from aiohttp import web
    from server import PromptServer
    routes = PromptServer.instance.routes
    if not getattr(PromptServer.instance, '_aitk_shared_models_route', False):
        @routes.get('/aitk-shared-models/capabilities')
        async def shared_capabilities(request):
            return web.json_response(capabilities())
        PromptServer.instance._aitk_shared_models_route = True
    return AITKSharedModelsExtension()
