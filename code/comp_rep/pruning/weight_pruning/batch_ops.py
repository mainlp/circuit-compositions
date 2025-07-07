"""
Code given by *Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks* (Róbert Csordás et al. 2021)
<https://github.com/RobertCsordas/modules/blob/master/layers/batch_ops.py>

Copyright 2020 Robert Csordas
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F


def batch_matmul(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """
    Batch matrix multiply different batch elements of the input with different weights.

    If weights are 3D, Input batch is divided in N groups, and each group element is matrix multiplied by a different
    filter. In case B == N, each input element will be multipied by a different filter.

    :param input: input tensor, shape [B, Ci]
    :param weight: weight tensor, either [Ci, Co] or [N, Ci, Co]. In the latter case B must be divisible by N
    :return: tensor of [B, Co]
    """
    assert input.ndim == 2, f"Input dims: {input.shape}. Weight dim: {weight.shape}"

    if weight.ndim == 3:
        weight = weight.squeeze(0)

    if weight.ndim == 2:
        return torch.mm(input, weight)

    assert weight.ndim == 3
    assert input.shape[0] % weight.shape[0] == 0

    res = torch.bmm(input.view(weight.shape[0], -1, input.shape[-1]), weight)
    return res.view(input.shape[0], -1)


def batch_elementwise(
    input: torch.Tensor,
    param: torch.Tensor,
    op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    input_batch_dim: int = 0,
    pndim: int = 1,
) -> torch.Tensor:
    """
    Do elementwise operation in groups.

    :param input: input, any shape, [..., Ci, Cj, ...]
    :param param: the parameter, shape [N, Ci, Cj....], in which case B % N == 0, or [Ci, Cj....]
    :param input_batch_dim: which dimension is the batch in the input
    :param op: the operation to perform
    :param pndim: number of parameter dimensions without batch
    :return: input with the op performed, the same shape as input
    """

    if param.ndim == pndim + 1:
        # If the param is batched, check if it the batch size is 1
        param = param.squeeze(0)

    if param.ndim == pndim:
        # If the param has no batch, do the normal op
        return op(input, param)

    assert param.ndim == pndim + 1
    assert input.shape[input_batch_dim] % param.shape[0] == 0

    input_r = input.view(
        *input.shape[:input_batch_dim],
        param.shape[0],
        -1,
        *input.shape[input_batch_dim + 1 :],
    )

    param_r = param.view(
        *([1] * input_batch_dim),
        param.shape[0],
        *([1] * (input_r.ndim - input_batch_dim - param.ndim)),
        *param.shape[1:],
    )

    return op(input_r, param_r).view_as(input)


def batch_bias_add(*args, **kwargs) -> torch.Tensor:
    """
    Batch add bias to the inputs.

    For more details, see batch_elementwise
    """

    return batch_elementwise(*args, op=lambda a, b: a + b, **kwargs)  # type: ignore


def batch_const_mul(*args, **kwargs) -> torch.Tensor:
    """
    Batch multiplies bias to the inputs.

    For more details, see batch_elementwise
    """

    return batch_elementwise(*args, op=lambda a, b: a * b, **kwargs)  # type: ignore


def batch_const_div(*args, **kwargs) -> torch.Tensor:
    """
    Batch multiplies bias to the inputs.

    For more details, see batch_elementwise
    """

    return batch_elementwise(*args, op=lambda a, b: a / b, **kwargs)  # type: ignore


def batch_conv2d(
    input: torch.Tensor,
    filter: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Convolve some elements of the batch with different filters.

    (M,) is optional

    :param input: input image, [N, C, H, W]
    :param filter: [(M,) out channels, in channels, h, w]
    :param bias: [(M,) out channels]
    :return:Image convolved by the filter
    """

    assert input.ndim == 4

    if filter.ndim == 5:
        assert input.shape[0] % filter.shape[0] == 0, (
            f"Number of batches {input.shape[0]} must be divisible "
            f"by number of filters {filter.shape[0]}"
        )

        res = F.conv2d(
            input.view(filter.shape[0], -1, *input.shape[1:])
            .transpose(0, 1)
            .flatten(1, 2),
            filter.view(-1, *filter.shape[2:]),
            None if bias is None or bias.ndim != 1 else bias,
            **kwargs,
            groups=filter.shape[0],
        )

        res = (
            res.view(res.shape[0], -1, filter.shape[1], *res.shape[2:])
            .transpose(0, 1)
            .flatten(0, 1)
        )

        if bias is not None and bias.ndim > 1:
            assert bias.ndim == 2
            res = res.view(bias.shape[0], -1, *res.shape[1:]) + bias.unsqueeze(
                1
            ).unsqueeze(-1).unsqueeze(-1)
            res = res.flatten(end_dim=1)
    else:
        return F.conv2d(input, filter, bias, **kwargs, groups=1)

    return res
