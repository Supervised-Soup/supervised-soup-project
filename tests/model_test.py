"""
Simple tests for ResNet-18 mdoel

Run in command line from project root with: python tests/model_test.py

"""

import torch
from supervised_soup.models.resnet18 import build_model
from supervised_soup import config
from supervised_soup.dataloader import get_dataloaders


def simple_test_with_random_input_forward():
    """
    Tests forward pass with random inuts.
    - Exected output: torch.Size([2, 10])
    """

    model = build_model(num_classes=10, pretrained=False, freeze_layers=False)
    model.to(config.DEVICE)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print(y.shape)


def simple_test_with_random_input_backward():
    """
    Tests backward pass with random inputs.
    """
    model.train()
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))

    out = model(x)
    loss = torch.nn.CrossEntropyLoss()(out, labels)
    loss.backward()

    print("Backward pass OK")


# test which parameters are trainable
model = build_model(num_classes=10, pretrained=True, freeze_layers=True)
for name, p in model.named_parameters():
    print(name, p.requires_grad)


def test_model_with_dataloader():
    """
    Runs a batch from the dataloader through the model.

    - Input batch shape: [batch_size, 3, 224, 224]
    - Model output shape: [batch_size, num_classes]

    Example output for batch_size=64, num_classes=10:
    - Output shape: torch.Size([64, 10])
    """
    train_loader, _ = get_dataloaders(with_augmentation=False)
    model = build_model(num_classes=10, pretrained=False, freeze_layers=False)

    images, labels = next(iter(train_loader))
    outputs = model(images)

    print("Output shape:", outputs.shape)


simple_test_with_random_input_forward()
simple_test_with_random_input_backward()
test_model_with_dataloader()