"""
Tests for run_training() with different optimizers and schedulers.

Run from project root:
    python tests/train_test.py
"""

import torch
import torch.nn as nn
from pathlib import Path
from supervised_soup.train import run_training
from supervised_soup.optimizers import build_optimizer
import supervised_soup.config as config

def test_run_training_smoke():
    """Quick test run for one epoch.
    if you run this on CPU it'll take like 20 minutes or so"""

    # Create a temporary folder for checkpoints
    test_checkpoint_path = Path("test_results")
    test_checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Temporarily override the CHECKPOINTS_PATH
    config.CHECKPOINTS_PATH = test_checkpoint_path
    model, history = run_training(
        epochs=1,            
        with_augmentation=False, 
        lr=0.01,
        optimizer_name="sgd",
        scheduler_name="cosine",
    )

    assert len(history["train_loss"]) == 1, "History not recorded correctly"
    assert len(history["val_acc"]) == 1, "Validation metrics missing"
    print("Default config test passed!")

    return history

def test_optimizers_update_parameters():
    """
    Unit test: each supported optimizer should update model parameters after backward + step.
    """
    optimizer_names = ["sgd", "adam", "adamw", "rmsprop", "adagrad"]

    for name in optimizer_names:
        torch.manual_seed(0)

        model = nn.Linear(4, 3)
        optimizer = build_optimizer(name, model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        x = torch.randn(8, 4)
        y = torch.randint(0, 3, (8,))

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        # make sure we actually have gradients
        assert any(p.grad is not None for p in model.parameters()), f"No grads for optimizer '{name}'"

        before = [p.detach().clone() for p in model.parameters()]

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert any(
            not torch.allclose(b, a.detach(), rtol=0.0, atol=0.0)
            for b, a in zip(before, model.parameters())
        ), f"Optimizer '{name}' did not update parameters"


def test_build_optimizer_filters_frozen_parameters():
    """
    Unit test: optimizer should not include parameters with requires_grad=False.
    """
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 3),
    )

    # Freeze the first layer
    for p in model[0].parameters():
        p.requires_grad = False

    opt = build_optimizer("sgd", model.parameters(), lr=0.01)

    opt_params = [p for group in opt.param_groups for p in group["params"]]

    assert len(opt_params) == sum(p.requires_grad for p in model.parameters())
    assert all(p.requires_grad for p in opt_params), "Optimizer contains frozen parameters"


if __name__ == "__main__":
    test_optimizers_update_parameters()
    test_build_optimizer_filters_frozen_parameters()
    # I'll comment this out to avoid long test time
    # test_run_training_smoke()

    print("All training tests passed!")



