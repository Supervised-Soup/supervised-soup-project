# Run with:
# python -m tests.evaluation_test

import wandb
from torchvision import models
from supervised_soup.evaluation import evaluate_model
from supervised_soup.config import DEVICE


TRAIN_RUN_NAME = "resnet18_seed42_pretrained_frozen_noAug_cleaned"

wandb.init(
    project="x-AI-Proj-ImageClassification",
    entity="neural-spi-university",
    name=f"eval_test_{TRAIN_RUN_NAME}",
)

model = models.resnet18(num_classes=10)
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
