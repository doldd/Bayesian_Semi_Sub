"""Models for experiments."""

from typing import List

import torch


class MLP(torch.nn.Module):
    """Simple MLP network."""

    def __init__(self, input_size: int, hidden_sizes: List[int]) -> None:
        """Instantiate MLP."""
        super().__init__()
        hidden_id = '_'.join([str(x) for x in hidden_sizes])
        self.model_id = f'MLP_{input_size}_{hidden_id}_1'
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.net = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]))
        for i, o in zip(hidden_sizes, hidden_sizes[1:] + [1]):
            self.net.append(torch.nn.Tanh())
            self.net.append(torch.nn.Linear(i, o))
        self.sigma = torch.nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define forward pass."""
        return self.net(x)
