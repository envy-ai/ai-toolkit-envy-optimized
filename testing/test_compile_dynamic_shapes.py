from unittest.mock import patch

import pytest
import torch

from toolkit.util.compile_dynamic_shapes import sequence_length_hook


def make_block(module, name, forward):
    return type(name, (torch.nn.Module,), {'__module__': module, 'forward': forward})()


@pytest.mark.parametrize('module,name,forward,args,expected', [
    ('moss_image.src.transformer', '_Block', lambda self, x, rope, mask, modulation: x,
     (torch.randn(2, 4, 4), torch.randn(1, 4, 4), torch.ones(2, 1, 4, 4), torch.randn(2, 4)), [(0, 1), (1, 1), (2, 2), (2, 3)]),
    ('krea2.src.mmdit', 'SingleStreamBlock', lambda self, x, freqs, mask, modulation: x,
     (torch.randn(2, 4, 4), torch.randn(1, 4, 4), torch.ones(2, 1, 4, 4), torch.randn(2, 4)), [(0, 1), (1, 1), (2, 2), (2, 3)]),
    ('qwen_image_2.src.transformer', 'QwenImage21TransformerBlock',
     lambda self, hidden_states, rotary_emb, target_token_mask, key_valid, attention_mask, modulation: hidden_states,
     (torch.randn(2, 4, 4), torch.randn(4, 4), torch.ones(4), torch.ones(2, 4), torch.ones(2, 1, 4, 4), torch.randn(2, 4)),
     [(0, 1), (1, 0), (2, 0), (3, 1), (4, 2), (4, 3)]),
])
def test_contract_axes_positional_and_keyword(module, name, forward, args, expected):
    block = make_block(module, name, forward)
    hook = sequence_length_hook(block)
    for positional, keyword in [(args, {}), ((), dict(zip(forward.__code__.co_varnames[1:], args)))]:
        with patch('torch._dynamo.maybe_mark_dynamic') as mark:
            hook(block, positional, keyword)
        actual = [(next(i for i, value in enumerate(args) if value is call.args[0]), call.args[1]) for call in mark.call_args_list]
        assert actual == expected


def test_unknown_block_is_untouched():
    assert sequence_length_hook(torch.nn.Linear(4, 4)) is None


def test_compile_variable_length_output_and_gradient():
    def forward(self, x, rope, mask): return (x * rope).sin()
    block = make_block('moss_image.src.transformer', '_Block', forward)
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(block, backend=backend, dynamic=True)
    compiled.register_forward_pre_hook(sequence_length_hook(block), with_kwargs=True)
    for length in (4, 7):
        x = torch.randn(2, length, 3, requires_grad=True)
        rope = torch.randn(1, length, 3)
        out = compiled(x, rope, None)
        expected = (x * rope).sin()
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(torch.autograd.grad(out.sum(), x, retain_graph=True)[0], torch.autograd.grad(expected.sum(), x)[0])
    assert len(graphs) == 1
