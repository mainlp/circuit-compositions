"""
Masked layer norm layers for model weight pruning.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from comp_rep.pruning.weight_pruning.batch_ops import batch_bias_add, batch_const_mul
from comp_rep.pruning.weight_pruning.masked_weights_base import (
    ContinuousMaskedWeightsLayer,
    SampledMaskedWeightsLayer,
)


class SampledMaskedWeightsLayerNorm(SampledMaskedWeightsLayer):
    """
    A masked LayerNorm module based on sampling. Masks are binarized to only keep or remove individual weights.
    This is achieved using a Gumbel-Sigmoid with a straight-through estimator.
    """

    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        weight: Tensor,
        bias: Optional[Tensor] = None,
        eps: float = 1e-5,
        tau: float = 1.0,
        num_masks: int = 1,
        ticket: bool = False,
    ):
        """
        Initializes the SampledMaskLinear layer.

        Args:
            weight (Tensor): The weight matrix of the linear layer.
            bias (Tensor, optional): The bias vector of the linear layer. Default: None.
            tau (float): The tau parameter for the s_i computation. Default: 1.0.
            num_masks (int): The number of mask samples. Default: 1.
        """
        super(SampledMaskedWeightsLayerNorm, self).__init__(
            weight=weight, bias=bias, ticket=ticket, tau=tau, num_masks=num_masks
        )
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies the layer norm to the input data using masked weights.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        masked_weight = self.weight.unsqueeze(0) * self.b_matrix

        mu = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        frac = batch_const_mul((x - mu) / (std + self.eps), masked_weight)

        if self.bias is not None:
            return batch_bias_add(frac, self.bias)

        return frac

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, s_matrix={self.s_matrix.shape}, b_matrix={self.b_matrix.shape}"


class ContinuousMaskedWeightsLayerNorm(ContinuousMaskedWeightsLayer):
    def __init__(
        self,
        normalized_shape: Tuple[int, ...],
        weight: Tensor,
        bias: Optional[Tensor],
        eps: float = 1e-5,
        mask_initial_value: float = 0.0,
        temperature_increase: float = 1.0,
        ticket: bool = False,
    ):
        super(ContinuousMaskedWeightsLayerNorm, self).__init__(
            weight=weight,
            bias=bias,
            ticket=ticket,
            mask_initial_value=mask_initial_value,
            temperature_increase=temperature_increase,
        )
        self.normalized_shape = normalized_shape
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass of the layer.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying layer normalization with a masked weight.
        """
        masked_weight = self.weight * self.b_matrix
        return F.layer_norm(
            x, self.normalized_shape, masked_weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, s_matrix={self.s_matrix.shape}, b_matrix={self.b_matrix.shape}"


if __name__ == "__main__":
    batch_size = 18
    in_features = 3

    # create a dummy input tensor
    input_tensor = torch.randn(batch_size, in_features)

    # ContinuousMaskLayerNorm
    layer_norm = nn.LayerNorm(in_features)
    print(f"Layer norm: \n{layer_norm}")

    cont_mask_layernorm = ContinuousMaskedWeightsLayerNorm(
        layer_norm.normalized_shape, layer_norm.weight, layer_norm.bias, layer_norm.eps
    )
    output_tensor_cont = cont_mask_layernorm(input_tensor)
    output_tensor = layer_norm(input_tensor)

    print(f"Continuous layer norm output: \n{output_tensor_cont}")
    print(f"L1 norm: \n{cont_mask_layernorm.compute_l1_norm()}")  # should be 0
    print(f"Normal layer norm output: \n{output_tensor}")

    # SampledMaskLayerNorm
    new_layer_norm = nn.LayerNorm(in_features)
    print(f"Layer norm: \n{new_layer_norm}")

    sampled_mask_layernorm = SampledMaskedWeightsLayerNorm(
        new_layer_norm.normalized_shape,
        new_layer_norm.weight,
        new_layer_norm.bias,
        new_layer_norm.eps,
    )
    print(f"Sampled layer norm: \n{sampled_mask_layernorm}")

    output_tensor_sampled = sampled_mask_layernorm(input_tensor)
    output_tensor = new_layer_norm(input_tensor)

    print(f"Normal layer norm output: \n{output_tensor.shape}")
    print(f"Sampled layer norm output: \n{output_tensor_sampled.shape}")
    print(f"L1 norm: \n{sampled_mask_layernorm.compute_l1_norm()}")  # should be 0
