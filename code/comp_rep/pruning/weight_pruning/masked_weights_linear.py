"""
Masked linear layers for model weight pruning.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from comp_rep.pruning.weight_pruning.batch_ops import batch_bias_add, batch_matmul
from comp_rep.pruning.weight_pruning.masked_weights_base import (
    ContinuousMaskedWeightsLayer,
    SampledMaskedWeightsLayer,
)


class SampledMaskedWeightsLinear(SampledMaskedWeightsLayer):
    """
    A masked linear layer based on sampling. Masks are binarized to only keep or remove individual weights.
    This is achieved using a Gumbel-Sigmoid with a straight-through estimator.
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
        Initializes the SampledMaskLinear layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
            tau (float): The tau parameter for the s_i computation. Default: 1.0.
            num_masks (int): The number of mask samples. Default: 1.
        """
        super(SampledMaskedWeightsLinear, self).__init__(
            weight=weight, bias=bias, ticket=ticket, tau=tau, num_masks=num_masks
        )
        self.out_features, self.in_features = weight.shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the linear transformation to the input data using masked weights.

        Args:
            x (Tensor): The input tensor [B, L, in_features].

        Returns:
            Tensor: The output tensor.
        """
        batch_seqlength = list(x.shape[:-1])
        x = x.flatten(end_dim=-2)  # shape: [B*L, in_features]

        masked_weight = self.weight.unsqueeze(0) * self.b_matrix

        output = batch_matmul(
            x, masked_weight.transpose(-1, -2)
        )  # shape: [B, L*out_features]

        if self.bias is not None:
            output = batch_bias_add(output, self.bias)

        reshaped_output = output.view(
            *batch_seqlength, output.shape[-1]
        )  # shape: [B, L, out_features]
        return reshaped_output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, s_matrix={self.s_matrix.shape}, b_matrix={self.b_matrix.shape}"


class ContinuousMaskedWeightsLinear(ContinuousMaskedWeightsLayer):
    """
    A masked linear layer based on continuous sparsification.
    Masks are binarized to only keep or remove individual weights.
    """

    def __init__(
        self,
        weight: Tensor,
        bias: Optional[Tensor] = None,
        mask_initial_value: float = 0.0,
        temperature_increase: float = 1.0,
        ticket: bool = False,
    ):
        super(ContinuousMaskedWeightsLinear, self).__init__(
            weight=weight,
            bias=bias,
            ticket=ticket,
            mask_initial_value=mask_initial_value,
            temperature_increase=temperature_increase,
        )
        self.out_features, self.in_features = weight.shape

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the linear transformation to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        masked_weight = self.weight * self.b_matrix
        return F.linear(x, masked_weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, s_matrix={self.s_matrix.shape}, b_matrix={self.b_matrix.shape}"


if __name__ == "__main__":
    batch_size = 18
    num_mask = 2
    tau = 1.0
    in_features = 3
    out_features = 5

    # create a dummy input tensor
    input_tensor = torch.randn(batch_size, in_features)

    # layers
    linear_layer = nn.Linear(in_features, out_features, bias=True)
    print(f"Linear layer: \n{linear_layer}")

    sampled_mask_linear = SampledMaskedWeightsLinear(
        linear_layer.weight,
        linear_layer.bias,
        ticket=False,
        tau=tau,
        num_masks=num_mask,
    )
    print(isinstance(sampled_mask_linear, SampledMaskedWeightsLayer))
    print(f"Sampled masked layer: \n{sampled_mask_linear}")

    linear_layer = nn.Linear(in_features, out_features, bias=True)
    cont_mask_linear = ContinuousMaskedWeightsLinear(
        linear_layer.weight, linear_layer.bias
    )
    print(isinstance(cont_mask_linear, ContinuousMaskedWeightsLinear))
    print(f"Continuous  masked layer: \n{cont_mask_linear}")

    output_tensor_sample = sampled_mask_linear(input_tensor)
    output_tensor_cont = cont_mask_linear(input_tensor)
    print(f"Sampled out tensor: \n{output_tensor_sample}")
    print(f"Continuous out tensor: \n{output_tensor_cont}")
