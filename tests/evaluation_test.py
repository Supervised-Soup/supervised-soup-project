
# Run with: 
# python -m tests.evaluation_test

import torch
from torchvision import models

from supervised_soup.evaluation import evaluate_model
from supervised_soup.config import DEVICE


# Dummy model (ResNet-18) for 10 classes
model = models.resnet18(num_classes=10)
model.to(DEVICE)

# Run evaluation without logging to W&B
metrics = evaluate_model(model, log_to_wandb=False)

# Print metrics
print("Metrics dict:")
for k, v in metrics.items():
    print(f"{k}: {v}")
