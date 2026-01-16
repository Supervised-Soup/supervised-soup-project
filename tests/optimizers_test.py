"""
Unit tests for optimizer builder.
"""

import torch
import torch.nn as nn

from supervised_soup.optimizers import build_optimizer


def test_optimizers_update_parameters():
    """Each supported optimizer should update parameters after backward + step."""
    optimizer_names = ["sgd", "adam", "adamw", "rmsprop", "adagrad"]

    for name in optimizer_names:
        torch.manual_seed(0)

        model = nn.Linear(4, 3)
        optimizer = build_optimizer(name, model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))

        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()

        assert any(p.grad is not None for p in model.parameters()), f"No grads for optimizer '{name}'"

        before = [p.detach().clone() for p in model.parameters()]

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert any(
            not torch.equal(b, a.detach())
            for b, a in zip(before, model.parameters())
        ), f"Optimizer '{name}' did not update parameters"


def test_build_optimizer_filters_frozen_parameters():
    """Optimizer should not include frozen parameters (requires_grad=False)."""
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 3),
    )

    for p in model[0].parameters():
        p.requires_grad = False

    opt = build_optimizer("sgd", model.parameters(), lr=0.01)
    opt_params = [p for group in opt.param_groups for p in group["params"]]

    assert len(opt_params) == sum(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in opt_params), "Optimizer contains frozen parameters"
