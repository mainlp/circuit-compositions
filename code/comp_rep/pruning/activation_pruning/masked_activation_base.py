"""
Base classes for masked activation layers
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from comp_rep.pruning.masked_base import MaskedLayer


class MaskedActivationLayer(MaskedLayer):
    """
    An abstract base class for a masked activation layer.
    """

    def __init__(self, layer: nn.Module, hidden_size: int, ticket: bool = False):
        """
        Initializes the masked activation layer.

        Args:
            layer (nn.Module): The layer to mask its output.
            hidden_size (int): The dimension of the layer's output.
            ticket (bool, optional): Whether to configure the mask in binary mode. Default: False.
        """
        super(MaskedActivationLayer, self).__init__(ticket=ticket)
        self.layer = layer
        self.hidden_size = hidden_size


class ContinuousMaskedActivationLayer(MaskedActivationLayer):
    """
    A base class for a continuous masked activation layer.
    """

    def __init__(
        self,
        layer: nn.Module,
        hidden_size: int,
        ablation_values: Tensor = torch.zeros(1),
        ticket: bool = False,
        mask_initial_value: float = 0.0,
        initial_temp: float = 1.0,
        temperature_increase: float = 1.0,
    ):
        """
        Initializes the masked activation layer.

        Args:
            layer (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
        """
        super(ContinuousMaskedActivationLayer, self).__init__(
            layer=layer, hidden_size=hidden_size, ticket=ticket
        )
        self.mask_initial_value = mask_initial_value
        self.temperature_increase = temperature_increase
        self.temp = initial_temp
        self.ablation_values = ablation_values.to(next(self.layer.parameters()).device)

        self.s_matrix = self.init_s_matrix()
        self.register_parameter("s_matrix", self.s_matrix)
        self.register_buffer("b_matrix", torch.zeros_like(self.s_matrix))
        self.compute_mask()

    def init_s_matrix(self) -> nn.Parameter:
        """
        Initializes the s_matrix with constant values.

        Returns:
            nn.Parameter: The s_matrix.
        """
        s_matrix = nn.Parameter(
            nn.init.constant_(
                torch.Tensor(self.hidden_size),
                self.mask_initial_value,
            )
        )
        return s_matrix

    def compute_mask(self) -> None:
        """
        Compute and sets the mask.
        """
        if self.ticket:
            self.b_matrix = (self.s_matrix > 0).float()
        else:
            self.b_matrix = F.sigmoid(self.temp * self.s_matrix)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Applies the layer transformation to the input and then masks corresponding output activations.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The masekd output activation tensor.
        """
        layer_activations = self.layer(*args, **kwargs)
        masked_activations = (
            layer_activations * self.b_matrix
            + (1 - self.b_matrix) * self.ablation_values
        )

        return masked_activations

    def update_temperature(self):
        """
        Updates the temperature.
        """
        self.temp = self.temp * self.temperature_increase

    def compute_l1_norm(self):
        """
        Computes the L1 norm of the s_matrix

        Returns:
            Tensor: The L1 norm
        """
        return torch.norm(self.b_matrix, p=1)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, s_matrix={self.s_matrix.shape}, b_matrix={self.b_matrix.shape}"
