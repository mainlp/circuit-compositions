"""
Base classes for masked layers
"""

import abc
from typing import Any

import torch.nn as nn
from torch import Tensor


class MaskedLayer(nn.Module, abc.ABC):
    """
    An abstract base class for a masked layer.
    """

    def __init__(self, ticket: bool = False):
        """
        Initializes the masked layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
        """
        super(MaskedLayer, self).__init__()
        self.ticket = ticket

    @abc.abstractmethod
    def init_s_matrix(self) -> Tensor:
        """
        Initializes and returns the variable introduced to compute the binary mask matrix.

        Returns:
            Tensor: The additional variable.
        """
        pass

    @abc.abstractmethod
    def compute_mask(self) -> None:
        """
        Computes and sets the mask to be applied to the weights.

        """
        pass

    @abc.abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Applies the linear transformation to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        pass

    @abc.abstractmethod
    def compute_l1_norm(self):
        """
        Computes the L1 norm of the s_matrix

        Returns:
            Tensor: The L1 norm
        """
        pass

    def compute_remaining_mask(self, fraction: bool = True) -> float:
        """
        Computes and returns the percentage of remaining non-zero mask values

        Returns:
            Tensor: The percentage of remaining non-zero mask values
        """
        above_zero = float((self.b_matrix > 0).sum())

        if not fraction:
            return above_zero

        original = self.s_matrix.numel()
        return above_zero / original

    @abc.abstractmethod
    def extra_repr(self) -> str:
        """
        The module representation string.
        """
        pass
