import pytest
import torch

from toolkit.util.comfy_quant_import import ConvRotInt8Embedding, Int8Embedding


@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('per_row', [False, True])
def test_embedding_scalar_and_per_row_scales_survive_dtype_staging(rotated, per_row):
    codes = torch.tensor([[1, 2, 3, 4], [5, -6, 7, -8], [9, 10, -11, 12], [-13, 14, 15, -16]], dtype=torch.int8)
    scales = torch.tensor([0.25, 0.5, 0.75, 1.0]) if per_row else torch.tensor(0.5)
    embedding = (ConvRotInt8Embedding(codes, scales, 4, torch.float32) if rotated
                 else Int8Embedding(codes, scales, torch.float32))
    embedding.to(dtype=torch.bfloat16)
    assert embedding.scales.dtype == torch.uint8
    ids = torch.tensor([[3, 1, 3], [0, 2, 1]])
    full = codes.float() * (scales[:, None] if per_row else scales)
    if rotated:
        matrix = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32) / 2
        full = full @ matrix
    torch.testing.assert_close(embedding(ids), full[ids])
    torch.testing.assert_close(embedding.weight, full)


@pytest.mark.parametrize('rotated', [False, True])
def test_embedding_rejects_invalid_scale_count(rotated):
    codes = torch.ones((4, 4), dtype=torch.int8)
    with pytest.raises(ValueError, match='scale'):
        if rotated:
            ConvRotInt8Embedding(codes, torch.ones(2), 4, torch.float32)
        else:
            Int8Embedding(codes, torch.ones(2), torch.float32)
