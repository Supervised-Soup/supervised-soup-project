"""
Evaluates best checkpoints on the final test set.

Usage:
    python -m supervised_soup.eval_runs RUN_ID

Example:
    python -m supervised_soup.eval_runs wfpyvep6
"""

# currently resnet101 hardcoded

import os
import sys
import glob
import torch
import wandb

from supervised_soup.evaluation import evaluate_model
from supervised_soup.config import DEVICE
from supervised_soup.models.model import build_model

ENTITY = "neural-spi-university"
PROJECT = "x-AI-Proj-ImageClassification"


def get_best_model_artifact(run_id: str) -> str:
    api = wandb.Api()
    artifact_name = f"{ENTITY}/{PROJECT}/best-model-{run_id}:latest"
    try:
        artifact = api.artifact(artifact_name, type="model")
    except wandb.errors.CommError:
        raise RuntimeError(f"No best-model artifact found for run '{run_id}'")

    print(f"Using artifact {artifact.name}:{artifact.version}")

    artifact_dir = artifact.download()
    checkpoint_files = glob.glob(os.path.join(artifact_dir, "*.pt"))
    if not checkpoint_files:
        raise RuntimeError(f"No .pt checkpoint found in artifact '{artifact.name}'")
    return checkpoint_files[0]


def evaluate_runs(run_ids: list[str]):
    for run_id in run_ids:
        print(f"\n=== Evaluating run {run_id} ===\n")

        wandb.init(
            project=PROJECT,
            entity=ENTITY,
            name=f"eval_test_{run_id}",
            job_type="evaluation",
            tags=["test", "final"],
        )

        checkpoint_path = get_best_model_artifact(run_id)
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

        model = build_model(
            model_name="resnet101",
            num_classes=10,
            pretrained=False,
        ).to(DEVICE)

        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        evaluate_model(
            model,
            run_name=run_id,
            log_to_wandb=True,
        )

        wandb.finish()


if __name__ == "__main__":
    run_ids = sys.argv[1:]
    if not run_ids:
        raise RuntimeError("No run ID provided Example usage: python -m supervised_soup.eval_runs wfpyvep6")

    evaluate_runs(run_ids)
