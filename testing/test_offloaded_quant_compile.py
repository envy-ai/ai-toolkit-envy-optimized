import unittest
from types import SimpleNamespace

import torch
from torch import nn

from toolkit.memory_management.manager_modules import OstrisLinearLayerMemoryManager


class _CheckedLinear(nn.Linear):
    def forward(self, x):
        if torch.compiler.is_compiling():
            raise AssertionError("the offloaded quantized forward was traced")
        return super().forward(x)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = _CheckedLinear(4, 4)
        OstrisLinearLayerMemoryManager(
            self.linear, SimpleNamespace(process_device=torch.device("cpu"))
        )

    def forward(self, x):
        return self.linear(x * 2) + 1


class OffloadedQuantCompileTest(unittest.TestCase):
    def test_compiled_block_keeps_offloaded_forward_eager(self):
        block = _Block()
        x = torch.randn(2, 4, requires_grad=True)
        expected = block(x)

        graphs = []

        def capture_graph(graph, inputs):
            graphs.append(graph)
            return graph.forward

        compiled = torch.compile(block, backend=capture_graph, fullgraph=False)
        actual = compiled(x)
        torch.testing.assert_close(actual, expected)
        self.assertGreaterEqual(len(graphs), 2)
        actual.sum().backward()
        self.assertIsNotNone(block.linear.weight.grad)
        self.assertIsNotNone(x.grad)


if __name__ == "__main__":
    unittest.main()
