"""Tiny real CUDA streaming regression for the official image PDD head."""
import torch
import pytest
from toolkit.memory_management import MemoryManager
from testing.test_h3_official_image import tiny_image_model


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_streamed_image_head_six_steps_and_checkpointed_training():
    torch.manual_seed(42)
    model = tiny_image_model(6).requires_grad_(False)
    model.gradient_checkpointing = True
    MemoryManager.attach(model, torch.device('cuda:0'), offload_percent=1.0)
    try:
        assert model.final_layer.video_out.weight.device.type == 'cpu'
        hidden = torch.randn(1, 2, 8, device='cuda', requires_grad=True)
        kwargs = dict(hidden_states=hidden, encoder_hidden_states=torch.randn(1, 2, 8, device='cuda'),
            row_timesteps=torch.full((1, 4), .4, device='cuda'), token_tags=torch.tensor([[1, 1, 0, 0]], device='cuda'),
            position_ids=torch.zeros(1, 4, 3, device='cuda'), video_indices=torch.tensor([2, 3], device='cuda'),
            text_indices=torch.tensor([0, 1], device='cuda'))
        for step in range(6):
            output = model(**kwargs, pdd_step=step)
            assert output.device.type == 'cuda' and torch.isfinite(output).all()
        train = model(**kwargs)
        train.square().mean().backward()
        assert hidden.grad is not None and torch.isfinite(hidden.grad).all() and hidden.grad.abs().sum() > 0
        torch.cuda.synchronize()
        print('GPU_PEAK_ALLOCATED', torch.cuda.max_memory_allocated(), 'GPU_PEAK_RESERVED', torch.cuda.max_memory_reserved())
    finally:
        MemoryManager.detach(model)
