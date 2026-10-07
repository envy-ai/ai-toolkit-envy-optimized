"""V3 standard MODEL loaders, immutable adapter generation and CPU status."""
import json

from comfy_api.latest import io
from aitk_shared_models.client import Client
from aitk_shared_models.snapshot import verify_snapshot
from aitk_shared_models.protocol import SharedModelError

from .execution_lease import current_prompt


class AITKSharedHunyuanImage3Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='AITKSharedHunyuanImage3Loader', display_name='AITK Shared HunyuanImage 3',
            category='loaders/aitk shared', inputs=[io.String.Input('store_identity'), io.String.Input('socket', default='')],
            outputs=[io.Model.Output()])

    @classmethod
    def execute(cls, store_identity, socket=''):
        from .hunyuan_adapter import build_hunyuan
        guard = current_prompt()['guard']
        if socket and socket != guard.client.socket.getpeername():
            raise SharedModelError('Loader socket differs from prompt lease broker')
        arena = guard.client.open_store(store_identity, {'variant': 'instruct', 'quantization': 'int8_tensorwise'})
        return io.NodeOutput(build_hunyuan(arena, guard.client, store_identity))


class AITKSharedLoRASnapshot(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='AITKSharedLoRASnapshot', display_name='AITK Shared LoRA Snapshot',
            category='loaders/aitk shared', inputs=[io.Model.Input('model'),
                io.String.Input('snapshot_id', default='latest'), io.Float.Input('strength', default=1., min=-4., max=4.)],
            outputs=[io.Model.Output(), io.String.Output(display_name='snapshot_report')])

    @classmethod
    def fingerprint_inputs(cls, model=None, snapshot_id='latest', strength=1.):
        # Comfy's fingerprint prepass intentionally supplies constants only;
        # linked MODEL inputs are unavailable. Request fresh execution rather
        # than silently caching a previously resolved `latest` generation.
        # execute() still validates the actual MODEL and pins its snapshot.
        if model is None:
            return float('nan')
        metadata = cls.resolve(model, snapshot_id)
        return metadata['snapshot_id'], metadata['content_hash'], float(strength)

    @classmethod
    def resolve(cls, model, snapshot_id):
        if not getattr(model, 'aitk_store', None):
            raise SharedModelError('Snapshot requires a shared Hunyuan MODEL')
        transaction = current_prompt()
        key = (model.aitk_store['identity'], snapshot_id)
        snapshots = transaction['snapshots']
        if key not in snapshots:
            snapshots[key] = transaction['guard'].client.open_adapter(snapshot_id, key[0])
        return snapshots[key]

    @classmethod
    def execute(cls, model, snapshot_id='latest', strength=1.):
        import comfy.lora
        from safetensors.torch import load_file
        metadata = cls.resolve(model, snapshot_id)
        verify_snapshot(metadata, model.aitk_store['arena'].manifest)
        cloned = model.clone()
        if strength != 0:
            factors = load_file(metadata['path'], device='cpu')
            if any('magnitude' in name or 'lokr' in name or 'loha' in name for name in factors):
                raise SharedModelError('Shared release v1 accepts ordinary factorized LoRA only')
            from .lora_mapping import canonical_lora_map
            key_map = canonical_lora_map(model.model, factors, metadata)
            patches = comfy.lora.load_lora(factors, key_map)
            if not patches:
                raise SharedModelError('Snapshot has no matching canonical Hunyuan adapter keys')
            accepted = cloned.add_patches(patches, strength_patch=float(strength))
            if set(accepted) != set(patches):
                raise SharedModelError('Not all snapshot adapter projections matched the shared model')
        cloned.aitk_snapshot = metadata
        return io.NodeOutput(cloned, json.dumps(metadata, sort_keys=True))


class AITKSharedModelStatus(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='AITKSharedModelStatus', display_name='AITK Shared Model Status',
            category='diagnostics/aitk shared', inputs=[io.String.Input('socket', default='')],
            outputs=[io.String.Output()], not_idempotent=True)

    @classmethod
    def execute(cls, socket=''):
        with Client(socket or None, role='observer') as client:
            return io.NodeOutput(json.dumps(client.status(), sort_keys=True))
