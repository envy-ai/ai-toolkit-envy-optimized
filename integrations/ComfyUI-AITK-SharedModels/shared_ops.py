"""Package-owned staging ops bypass Comfy host packing/pinning and AIMDO."""
import contextlib
import copy
import dataclasses

import torch
import torch.nn.functional as F

from aitk_shared_models.protocol import SharedModelError


def staged_weight(module, input, patches=True):
    weight = module.weight.to(input.device)
    bias = None if module.bias is None else module.bias.to(input.device)
    functions = getattr(module, 'weight_function', ())
    if functions or getattr(module, '_full_precision_mm', False):
        if hasattr(weight, 'dequantize') and hasattr(weight, '_qdata'):
            # GPU-only dense temporary, bounded to one supported projection.
            if weight.numel() * input.element_size() > 512 * 1024**2:
                raise SharedModelError('Patched dense GPU temporary exceeds shared-mode bound')
            weight = weight.dequantize()
    if weight.dtype != input.dtype and not hasattr(weight, '_qdata'):
        if input.device.type == 'cpu':
            from aitk_shared_models.tensors import bounded_cpu_cast
            weight = bounded_cpu_cast(weight, input.dtype)
        else:
            weight = weight.to(input.dtype)
    if bias is not None:
        if input.device.type == 'cpu' and bias.dtype != input.dtype:
            from aitk_shared_models.tensors import bounded_cpu_cast
            bias = bounded_cpu_cast(bias, input.dtype)
        else:
            bias = bias.to(input.dtype)
    if patches:
        for function in functions:
            weight = function(weight)
        for function in getattr(module, 'bias_function', ()):
            bias = function(bias)
    return weight, bias


def shared_operations(dtype, quant_config):
    import comfy.ops
    parent = comfy.ops.mixed_precision_ops(quant_config, compute_dtype=dtype)

    class Linear(parent.Linear):
        def forward_comfy_cast_weights(self, input, compute_dtype=None, want_requant=False, weight_only_quant=False):
            weight, bias = staged_weight(self, input)
            return F.linear(input, weight, bias)

    class MoEExperts(parent.MoEExperts):
        @contextlib.contextmanager
        def bank_resident(self, input):
            self._resident_bank = staged_weight(self, input)
            try:
                yield self
            finally:
                self._resident_bank = None

        def expert_linear(self, input, index):
            if self._resident_bank is not None:
                weight, bias = self._resident_bank
            else:
                weight, bias = staged_weight(self, input)
            # Keep Comfy's activation quantization and fast bank matmul contract.
            return self._expert_linear_impl(input, weight, bias, index)

        def _expert_qt_from(self, weight, index):
            # Preserve every rotation/layout flag; the installed generic slicing
            # helper does not carry convrot=True into INT8 expert Params.
            from comfy.quant_ops import QuantizedTensor
            params = dataclasses.replace(weight._params,
                orig_shape=(self.out_features, self.in_features),
                scale=weight._params.scale[index] if weight._params.scale.dim() else weight._params.scale)
            return QuantizedTensor(weight._qdata[index], weight._layout_cls, params)

    class Operations(parent):
        pass
    Operations.Linear = Linear
    Operations.MoEExperts = MoEExperts
    return Operations


@torch.inference_mode(False)
def shared_quantized_tensor(qdata, layout, params):
    """Comfy Kitchen detach() normally clones scales, even for Parameter wrapping."""
    from comfy.quant_ops import QuantizedTensor
    from aitk_shared_models.tensors import owner_of
    class SharedQuantizedTensor(QuantizedTensor):
        def _copy_with(self, qdata=None, params=None, clone_params=True):
            data = self._qdata if qdata is None else qdata
            options = self._params if params is None else params
            klass = SharedQuantizedTensor if owner_of(data) is not None else QuantizedTensor
            # Cached CPU masters are detached both inside node inference mode
            # and outside it during lease cleanup. A mixed inference wrapper
            # cannot be rewrapped as Parameter on the next prompt in Torch 2.11.
            if klass is SharedQuantizedTensor:
                with torch.inference_mode(False):
                    return klass(data, self._layout_cls, options)
            return klass(data, self._layout_cls, options)

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func == torch.ops.aten.detach.default:
                value = args[0]
                return value._copy_with(qdata=value._qdata.detach(), clone_params=False)
            if func == torch.ops.aten.clone.default:
                raise SharedModelError('Shared quantized weights cannot be cloned on CPU')
            if func == torch.ops.aten._to_copy.default:
                value = args[0]
                device = torch.device(kwargs.get('device', value.device))
                if device.type == 'cpu':
                    raise SharedModelError('Shared quantized CPU dtype/copy operations are unsupported')
                moved = value._qdata.to(device=device, non_blocking=kwargs.get('non_blocking', False))
                options = value._params.to_device(device)
                if kwargs.get('dtype') is not None:
                    options = dataclasses.replace(options, orig_dtype=kwargs['dtype'])
                return QuantizedTensor(moved, value._layout_cls, options)
            return super().__torch_dispatch__(func, types, args, kwargs)
    return SharedQuantizedTensor(qdata, layout, params)
