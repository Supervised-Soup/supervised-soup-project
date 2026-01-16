"""
Integration smoke test for run_training().

Run (pytest):
    pytest -k train_smoke

Run (python):
    python tests/train_smoke_test.py
"""

from pathlib import Path

import torch

import supervised_soup.config as config
from supervised_soup.train import run_training


def test_run_training_smoke():
    """End-to-end smoke test. Can be slow on CPU."""
    torch.manual_seed(0)

    test_checkpoint_path = Path("test_results/smoke")
    test_checkpoint_path.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINTS_PATH = test_checkpoint_path

    model, history = run_training(
        epochs=1,
        with_augmentation=False,
        lr=0.01,
        optimizer_name="sgd",
        scheduler_name="none",  # keep smoke test minimal/fast
    )

    assert len(history["train_loss"]) == 1, "History not recorded correctly"
    assert len(history["val_acc"]) == 1, "Validation metrics missing"


if __name__ == "__main__":
    test_run_training_smoke()
    print("train_smoke passed!")