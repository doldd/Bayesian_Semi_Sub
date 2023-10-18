"""Utility functions."""

import json
from typing import Any, Dict

import torch
import torch.nn as nn


def load_json(file_path: str) -> Dict:
    """Load a json file as dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_as_json(dictionary: Any, target: str) -> None:
    """Save a python object as JSON file."""
    with open(target, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


def init_weights(layer: nn.Module) -> None:
    """Create checkpoint with network(s) to be loaded in learning."""
    if isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight)
    if getattr(layer, 'bias', None) is not None:
        nn.init.zeros_(layer.bias)


def get_weight_vector(model: nn.Module, device: str) -> torch.Tensor:
    """Get and stack model weights."""
    weights = torch.empty(0, device=device)
    for child in model.children():
        for param in child.parameters():
            weights = torch.cat((weights, torch.flatten(param.to(device))))
    return weights


def custom_loss_fun(
    preds: torch.Tensor,
    targets: torch.Tensor,
    sigma: torch.Tensor,
    nn_weights: torch.Tensor,
) -> torch.Tensor:
    """Compute custom loss."""
    squared_loss = torch.square(preds - targets).sum()
    regularization = torch.dot(nn_weights, nn_weights)
    loss = 0.5 / torch.square(sigma) * squared_loss + preds.shape[0] * torch.log(sigma)
    return loss + 0.5 * regularization
