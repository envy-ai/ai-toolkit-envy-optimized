from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

spec = importlib.util.spec_from_file_location(
    "moss_attention_test_transformer",
    Path(__file__).resolve().parents[1] / "extensions_built_in/diffusion_models/moss_image/src/transformer.py",
)
moss = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = moss
spec.loader.exec_module(moss)


def inputs():
    torch.manual_seed(3)
    return [torch.randn(2, 4, 8, requires_grad=True) for _ in range(3)]


@pytest.mark.parametrize('isolate', [None, 2])
def test_attention_mask_and_isolation_outputs_and_gradients(isolate):
    q, k, v = inputs()
    rope = torch.cat((torch.zeros(1, 4, 2), torch.ones(1, 4, 2)), dim=-1)
    mask = torch.tensor([[[[True, True, False, True]]], [[[True, False, True, True]]]]) if isolate is None else None
    out = moss._attention(q, k, v, rope, 2, mask, isolate_split=isolate)
    qh, kh, vh = [x.reshape(2, 4, 2, 4).transpose(1, 2) for x in (q, k, v)]
    if isolate is None:
        expected = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    else:
        expected = torch.cat((F.scaled_dot_product_attention(qh[:, :, :isolate], kh, vh),
                              F.scaled_dot_product_attention(qh[:, :, isolate:], kh[:, :, isolate:], vh[:, :, isolate:])), dim=2)
    expected = expected.transpose(1, 2).flatten(2)
    torch.testing.assert_close(out, expected)
    actual_grads = torch.autograd.grad(out.sum(), (q, k, v), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (q, k, v))
    for actual, reference in zip(actual_grads, expected_grads): torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize('masked,cuda,enabled,isolate,expected', [
    (True, True, True, None, True), (False, True, True, None, False),
    (True, False, True, None, False), (True, True, False, None, False),
    (True, True, True, 2, False),
])
def test_cudnn_dispatch_only_masked_cuda(masked, cuda, enabled, isolate, expected):
    q, k, v = inputs()
    rope = torch.cat((torch.zeros(1, 4, 2), torch.ones(1, 4, 2)), dim=-1)
    mask = torch.ones(2, 1, 1, 4, dtype=torch.bool) if masked else None
    with patch.object(torch.Tensor, 'is_cuda', new_callable=lambda: property(lambda self: cuda)), \
         patch.object(torch.backends.cuda, 'cudnn_sdp_enabled', return_value=enabled), \
         patch.object(moss, 'sdpa_kernel', return_value=nullcontext()) as kernel:
        moss._attention(q, k, v, rope, 2, mask, isolate_split=isolate)
    assert kernel.called == expected
    if expected:
        assert kernel.call_args.args[0][0] == moss.SDPBackend.CUDNN_ATTENTION
