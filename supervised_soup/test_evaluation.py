# Run with:
# python -m test_evaluation --run-name <wanb run name>

import argparse
import wandb
from torchvision import models
from supervised_soup.evaluation import evaluate_model
from supervised_soup.config import DEVICE


parser = argparse.ArgumentParser(description="Final evaluation of a run")
parser.add_argument(
    "--run-name", 
    type=str, 
    required=True, 
    help="wandb run name that should be evaluated on the final test set"
)
args = parser.parse_args()
TRAIN_RUN_NAME = args.run_name


wandb.init(
    project="x-AI-Proj-ImageClassification",
    entity="neural-spi-university",
    name=f"eval_test_{TRAIN_RUN_NAME}",
)

model = models.resnet101(num_classes=10)
model.to(DEVICE)

metrics = evaluate_model(
    model,
    run_name=TRAIN_RUN_NAME,
    log_to_wandb=True,
)

print("Metrics dict:")
for k, v in metrics.items():
    print(f"{k}: {v}")

wandb.finish()
