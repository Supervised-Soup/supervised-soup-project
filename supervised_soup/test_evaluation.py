# Run with:
# python -m test_evaluation

import wandb
from torchvision import models
from supervised_soup.evaluation import evaluate_model
from supervised_soup.config import DEVICE


TRAIN_RUN_NAME = "resnet101_partial_layer2_aug1_presetautoaugment_seed42_dscleaned_PhaseX"

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
