"""
Base classes for masked weights layers
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from comp_rep.pruning.masked_base import MaskedLayer


class MaskedWeightsLayer(MaskedLayer):
    """
    An abstract base class for a masked weight layer.
    """

    def __init__(
        self, weight: Tensor, bias: Optional[Tensor] = None, ticket: bool = False
    ):
        """
        Initializes the masked weight layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
        """
        super(MaskedWeightsLayer, self).__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.ticket = ticket


class SampledMaskedWeightsLayer(MaskedWeightsLayer):
    """
    An abstract base class for a sampled masked layer.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        ticket: bool = False,
        tau: float = 1.0,
        num_masks: int = 1,
    ):
        """
        Initializes the masked layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
        """
        super(SampledMaskedWeightsLayer, self).__init__(
            weight=weight, bias=bias, ticket=ticket
        )
        self.tau = tau
        self.num_masks = num_masks

        self.logits = nn.Parameter(
            torch.ones_like(weight) * torch.log(torch.tensor(9.0)), requires_grad=True
        )
        self.register_parameter("logits", self.logits)

        self.register_buffer("s_matrix", self.init_s_matrix())
        self.register_buffer("b_matrix", torch.zeros_like(self.s_matrix))
        self.compute_mask()

    def sample_s_matrix(self, eps: float = 1e-10) -> Tensor:
        """
        Samples the s_matrix using the specified formula:
        s_i = sigma((l_i - log(log(U_1) / log(U_2))) / tau) with U_1, U_2 ~ U(0, 1).

        Returns:
            Tensor: The initialized s_matrix tensor.
        """
        min_sampled_value = torch.ones_like(self.weight) * eps
        U1 = torch.maximum(min_sampled_value, torch.rand_like(self.weight))
        U2 = torch.maximum(min_sampled_value, torch.rand_like(self.weight))

        log_ratio = torch.log(torch.log(U1) / torch.log(U2))
        s_matrix = torch.sigmoid((self.logits - log_ratio) / self.tau)

        return s_matrix

    def init_s_matrix(self) -> Tensor:
        """
        Initializes multiple s_matrices.

        Returns:
            Tensor: Stacked s_matrix tensors (stacked in the 0th dimension).
        """
        s_matrices = [self.sample_s_matrix(eps=1e-10) for _ in range(self.num_masks)]
        return torch.stack(s_matrices, dim=0)

    def compute_mask(self) -> None:
        """
        Computes and sets the mask to be applied to the weights using the straight-through estimator.
        """
        if self.ticket:
            self.b_matrix = (torch.sigmoid(self.logits) > 0.5).float()
        else:
            self.s_matrix = self.init_s_matrix()
            with torch.no_grad():
                b_i = (self.s_matrix > 0.5).type_as(self.s_matrix)
            self.b_matrix = (b_i - self.s_matrix).detach() + self.s_matrix

    def compute_l1_norm(self):
        """
        Computes the L1 norm of the s_matrix

        Returns:
            Tensor: The L1 norm
        """
        return torch.sum(self.logits)


class ContinuousMaskedWeightsLayer(MaskedWeightsLayer):
    """
    An abstract base class for a continuous masked layer.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        ticket: bool = False,
        mask_initial_value: float = 0.0,
        initial_temp: float = 1.0,
        temperature_increase: float = 1.0,
    ):
        """
        Initializes the masked layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
        """
        super(ContinuousMaskedWeightsLayer, self).__init__(
            weight=weight, bias=bias, ticket=ticket
        )
        self.mask_initial_value = mask_initial_value
        self.temperature_increase = temperature_increase
        self.temp = initial_temp

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
                torch.Tensor(self.weight.shape).to(self.weight.device),
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
