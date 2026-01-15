"""
Optimizer functions.
"""

from __future__ import annotations

import torch
import torch.optim as optim

from collections.abc import Iterable


def build_optimizer(
    optimizer_name: str,
    params: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
) -> optim.Optimizer:
    """
    Builds an optimizer based on optimizer_name.

    Supported:
        - "sgd"
        - "adam"
        - "adamw"
        - "adagrad"
        - "rmsprop"
    """
    name = optimizer_name.lower()
    
    # Avoid iterating over frozen parameters
    trainable_params = [p for p in params if p.requires_grad]

    if len(trainable_params) == 0:
        raise ValueError(
            "No trainable parameters found (all params have requires_grad=False). "
            "Did you freeze the entire model by accident?"
        )

    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    if name == "adagrad":
        return optim.Adagrad(params, lr=lr, weight_decay=weight_decay)

    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. "
        f"Choose from: sgd, adam, adamw, adagrad, rmsprop."
    )
