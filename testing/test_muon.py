import copy
import math

import pytest
import torch

from toolkit.optimizers.muon import BatchedMuon


def reference_step(p, grad, buffer, *, lr, weight_decay, momentum, nesterov, adjust_lr_fn):
    """Single-parameter Fizgig Muon equations, independent of batched implementation."""
    buffer.lerp_(grad, 1 - momentum)
    update = grad.lerp(buffer, momentum) if nesterov else buffer
    if p.ndim >= 2:
        rows, cols = p.shape[0], p.numel() // p.shape[0]
        x = update.reshape(rows, cols).bfloat16()
        if rows > cols:
            x = x.T
        x = x / x.norm().clamp(min=1e-7)
        for _ in range(5):
            gram = x @ x.T
            x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315), x, beta=3.4445)
        if rows > cols:
            x = x.T
        update = x.reshape(p.shape).to(p.dtype)
        ratio = (math.sqrt(max(1, rows / cols)) if adjust_lr_fn in (None, 'original') else
                 0.2 * math.sqrt(max(rows, cols)) if adjust_lr_fn == 'match_rms_adamw' else
                 math.sqrt(rows / cols))
    else:
        update = (update.float() / update.float().square().mean().sqrt().clamp(min=1e-7)).to(p.dtype) * .2
        ratio = 1
    p.mul_(1 - lr * weight_decay).add_(update, alpha=-lr * ratio)


@pytest.mark.parametrize('mode', [None, 'original', 'match_rms_adamw', 'spectral_unclamped'])
@pytest.mark.parametrize('nesterov', [True, False])
def test_matches_independent_reference(mode, nesterov):
    torch.manual_seed(7)
    shapes = [(2, 3), (2, 3), (3, 2), (2, 2, 2), (3,), ()]
    params = [torch.nn.Parameter(torch.randn(shape, dtype=torch.float64 if i == 1 else torch.float32)) for i, shape in enumerate(shapes)]
    expected = [p.detach().clone() for p in params]
    buffers = [torch.zeros_like(p) for p in params]
    groups = [{'params': params[:3], 'lr': .01}, {'params': params[3:], 'lr': .02}]
    optimizer = BatchedMuon(groups, weight_decay=.07, momentum=.8, nesterov=nesterov, adjust_lr_fn=mode)
    for step in range(3):
        for i, p in enumerate(params):
            p.grad = torch.randn_like(p) * (10 if i == 0 else .03 if i == 1 else 1)
            reference_step(expected[i], p.grad, buffers[i], lr=.01 if i < 3 else .02,
                           weight_decay=.07, momentum=.8, nesterov=nesterov, adjust_lr_fn=mode)
        optimizer.step()
    for p, target in zip(params, expected):
        torch.testing.assert_close(p, target, rtol=1e-5, atol=1e-6)


def test_chunk_boundary_resume_closure_and_missing_gradient():
    torch.manual_seed(4)
    params = [torch.nn.Parameter(torch.randn(2, 3)) for _ in range(65)]
    opt = BatchedMuon(params, lr=.01)
    for p in params: p.grad = torch.randn_like(p)
    opt.step()
    state = copy.deepcopy(opt.state_dict())
    clones = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = BatchedMuon(clones, lr=.01)
    resumed.load_state_dict(state)
    for p, q in zip(params, clones):
        p.grad = torch.randn_like(p); q.grad = p.grad.clone()
    params[-1].grad = clones[-1].grad = None
    before = params[-1].detach().clone()
    loss = opt.step(lambda: (params[0] ** 2).sum())
    resumed.step()
    assert loss.requires_grad
    for p, q in zip(params, clones): torch.testing.assert_close(p, q)
    torch.testing.assert_close(params[-1], before)


def test_sparse_rejected_before_any_mutation_and_zero_finite():
    a, b = torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2, 2))
    opt = BatchedMuon([a, b])
    a.grad = torch.ones_like(a)
    b.grad = torch.sparse_coo_tensor([[0], [0]], [1.], (2, 2))
    with pytest.raises(RuntimeError, match='sparse'): opt.step()
    torch.testing.assert_close(a, torch.ones_like(a))
    a.grad = torch.zeros_like(a); b.grad = torch.zeros_like(b)
    opt.step()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
