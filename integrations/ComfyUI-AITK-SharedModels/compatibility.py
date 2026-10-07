"""Fail closed on unsupported Comfy/Pedro source and execution signatures."""
import hashlib
import inspect
import sys
from pathlib import Path

from aitk_shared_models.protocol import SharedModelError

SOURCE_HASHES = {
    'execution': 'c9ef8ea11b8cb7aa80d05f670ca457211869d27623991b66d0014fb9b33fb246',
    'comfy.ops': 'e13f918c6a64f1dc2b9ce836868734396849db3eeb80313579afbfa90dd6ae89',
    'comfy.model_patcher': '982afb7e9762ecc330806299485e6f87550ff500b6d465eaa2f60248e95791e8',
    'pedro.loader': '49590d1ff92dff9b5be44f6d00f8b07f364b38b1a38bc153b39279dd5d48cb08',
    'pedro.model': '4324cb1720e69b8e24edd7b3ec0197b6bfa593789c5f068daaa2c40e36e48e63',
    'pedro.model_base': 'a2e8a9df27ade45bbd62ed4da4842289200aa3ff05c33a48725271c7effbfed9',
}


def check_source(module, key):
    path = Path(module.__file__)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != SOURCE_HASHES[key]:
        raise SharedModelError(f'Unsupported {key} source hash {actual}; review/qualify adapter before changing allowlist')
    return actual


def pedro_modules():
    candidates = [module for module in list(sys.modules.values()) if module is not None
                  and str(getattr(module, '__file__', '')).replace('\\', '/').endswith('/ComfyUI-HunyuanImage3/hunyuan_image_3/loader.py')]
    if len(candidates) != 1:
        raise SharedModelError('Load exactly one installed ComfyUI-HunyuanImage3 node pack first')
    loader = candidates[0]
    import importlib
    package = loader.__package__
    model = importlib.import_module(package + '.model')
    model_base = importlib.import_module(package + '.model_base')
    for module, key in ((loader, 'pedro.loader'), (model, 'pedro.model'), (model_base, 'pedro.model_base')):
        check_source(module, key)
    return loader, model, model_base


def check_execution(executor_class, execution_module=None):
    function = executor_class.execute_async
    if getattr(function, '_aitk_shared_lease', False):
        return function
    if not inspect.iscoroutinefunction(function) or list(inspect.signature(function).parameters) != ['self', 'prompt', 'prompt_id', 'extra_data', 'execute_outputs']:
        raise SharedModelError('PromptExecutor.execute_async signature is unsupported')
    if execution_module is not None:
        check_source(execution_module, 'execution')
        if function.__module__ != 'execution':
            raise SharedModelError('Another extension wrapped prompt execution; shared exclusion cannot be assumed')
    return function
