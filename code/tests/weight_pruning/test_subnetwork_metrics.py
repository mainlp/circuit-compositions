"""
Tests for subnetwork set operations
"""

import copy

import pytest
import torch
from torch import nn

from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.subnetwork_mask_metrics import (
    intersection_over_minimum,
    intersection_over_union,
    intersection_remaining_mask_by_layer_and_module,
    iom_by_layer_and_module,
    iom_models,
    iou_by_layer_and_module,
    iou_models,
    union_remaining_mask_by_layer_and_module,
)
from comp_rep.pruning.subnetwork_set_operations import complement_
from comp_rep.pruning.weight_pruning.masked_weights_layernorm import (
    ContinuousMaskedWeightsLayerNorm,
)
from comp_rep.pruning.weight_pruning.masked_weights_linear import (
    ContinuousMaskedWeightsLinear,
)


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, norm_shape):
        super(Transformer, self).__init__()
        linear_weight_1 = nn.Parameter(torch.randn(output_dim, input_dim))
        linear_weight_2 = nn.Parameter(torch.randn(output_dim, output_dim))
        norm_layer_weights = nn.Parameter(torch.randn(norm_shape))

        linear_layer1 = ContinuousMaskedWeightsLinear(
            weight=linear_weight_1, bias=None, ticket=True
        )
        norm_layer1 = ContinuousMaskedWeightsLayerNorm(
            normalized_shape=norm_shape,
            weight=norm_layer_weights,
            bias=None,
            ticket=True,
        )

        linear_layer2 = ContinuousMaskedWeightsLinear(
            weight=linear_weight_2, bias=None, ticket=True
        )
        norm_layer2 = ContinuousMaskedWeightsLayerNorm(
            normalized_shape=norm_shape,
            weight=norm_layer_weights,
            bias=None,
            ticket=True,
        )

        self.layers = nn.ModuleList(
            [linear_layer1, norm_layer1, linear_layer2, norm_layer2]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture
def modelA():
    input_dim = 4
    output_dim = 2
    norm_shape = 2
    return Transformer(input_dim, output_dim, norm_shape)


@pytest.fixture
def modelB():
    input_dim = 4
    output_dim = 2
    norm_shape = 2
    return Transformer(input_dim, output_dim, norm_shape)


@pytest.fixture
def continuous_linear() -> ContinuousMaskedWeightsLinear:
    """
    Fixture to create a ContinuousMaskedWeightsLinear layer.

    Returns:
        ContinuousMaskedWeightsLinear: The ContinuousMaskedWeightsLinear layer.
    """
    input_dim = 4
    output_dim = 2
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))

    return ContinuousMaskedWeightsLinear(weight=linear_weight, bias=None, ticket=True)


@pytest.fixture
def continuous_layernorm() -> ContinuousMaskedWeightsLayerNorm:
    """
    Fixture to create a ContinuousMaskedWeightsLayerNorm layer.

    Returns:
        ContinuousMaskedWeightsLayerNorm: The ContinuousMaskedWeightsLayerNorm layer.
    """
    norm_shape = (2,)
    norm_layer_weights = nn.Parameter(torch.randn(norm_shape))

    return ContinuousMaskedWeightsLayerNorm(
        normalized_shape=norm_shape, weight=norm_layer_weights, bias=None, ticket=True
    )


def test_intersection_over_union(continuous_linear):
    # set s_matrix and b_matrix
    model_linear_s_matrix = torch.Tensor(
        [
            [0.2, -0.2, -1.0, 0.6],
            [0.5, 0.3, 1.0, 0.0],
        ]
    )

    with torch.no_grad():
        continuous_linear.s_matrix.copy_(model_linear_s_matrix)
    continuous_linear.compute_mask()

    # test identity
    iou_value = intersection_over_union([continuous_linear])
    assert iou_value == 1

    # test equivalence
    iou_value = intersection_over_union([continuous_linear, continuous_linear])
    assert iou_value == 1

    # test complementary iou
    continuous_linear_complement = copy.deepcopy(continuous_linear)
    complement_(continuous_linear_complement)
    iou_value = intersection_over_union(
        [continuous_linear, continuous_linear_complement]
    )
    assert iou_value == 0

    # test fraction
    model_b_linear_s_matrix = torch.Tensor(
        [
            [0.6, -0.2, 1.0, 0.6],
            [0.5, 0.3, 0.0, 0.0],
        ]
    )
    continuous_linear_copy = copy.deepcopy(continuous_linear)
    with torch.no_grad():
        continuous_linear_copy.s_matrix.copy_(model_b_linear_s_matrix)
    continuous_linear_copy.compute_mask()
    iou_value = intersection_over_union([continuous_linear, continuous_linear_copy])
    assert iou_value == 2 / 3

    # test symmetry
    iou_value = intersection_over_union([continuous_linear_copy, continuous_linear])
    assert iou_value == 2 / 3

    # test 3 entries
    iou_value = intersection_over_union(
        [continuous_linear, continuous_linear_complement, continuous_linear_copy]
    )
    assert iou_value == 0


def test_intersection_over_minimum(continuous_linear):
    # set s_matrix and b_matrix
    model_linear_s_matrix = torch.Tensor(
        [
            [0.2, -0.2, -1.0, 0.6],
            [0.5, 0.3, 1.0, 0.0],
        ]
    )

    with torch.no_grad():
        continuous_linear.s_matrix.copy_(model_linear_s_matrix)
    continuous_linear.compute_mask()

    # test identity
    iou_value = intersection_over_minimum([continuous_linear])
    assert iou_value == 1

    # test equivalence
    iou_value = intersection_over_minimum([continuous_linear, continuous_linear])
    assert iou_value == 1

    # test complementary iou
    continuous_linear_complement = copy.deepcopy(continuous_linear)
    complement_(continuous_linear_complement)
    iou_value = intersection_over_minimum(
        [continuous_linear, continuous_linear_complement]
    )
    assert iou_value == 0

    # test fraction
    model_b_linear_s_matrix = torch.Tensor(
        [
            [0.6, -0.2, 1.0, 0.6],
            [0.5, 0.3, 0.0, 0.0],
        ]
    )
    continuous_linear_copy = copy.deepcopy(continuous_linear)
    with torch.no_grad():
        continuous_linear_copy.s_matrix.copy_(model_b_linear_s_matrix)
    continuous_linear_copy.compute_mask()
    iou_value = intersection_over_minimum([continuous_linear, continuous_linear_copy])
    assert iou_value == 4 / 5

    # test symmetry
    iou_value = intersection_over_minimum([continuous_linear_copy, continuous_linear])
    assert iou_value == 4 / 5

    # test 3 entries
    iou_value = intersection_over_minimum(
        [continuous_linear, continuous_linear_complement, continuous_linear_copy]
    )
    assert iou_value == 0


def test_intersection_remaining_weights_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix1 = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix1 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix1
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix1

    model_a_linear_b_matrix2 = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    model_a_layernorm_b_matrix2 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[2].b_matrix = model_a_linear_b_matrix2
    modelA.layers[3].b_matrix = model_a_layernorm_b_matrix2

    # set modelB
    model_b_linear_b_matrix1 = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix1 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix1
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix1

    model_b_linear_b_matrix2 = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ]
    )
    model_b_layernorm_b_matrix2 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[2].b_matrix = model_b_linear_b_matrix2
    modelB.layers[3].b_matrix = model_b_layernorm_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2, 3]
    module_types = [MaskedLayer]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
        fraction=True,
    )
    assert iou == (4 / 8 + 1 / 2 + 2 / 4 + 1 / 2) / 4

    # check identity for some types
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (4 / 8 + 2 / 4) / 2

    # check fraction
    layer_idx = [0]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == 3 / 8

    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (3 / 8 + 1 / 4) / 2

    # check sum
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=False
    )
    assert iou == (3 + 1) / 2


def test_union_remaining_weights_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix1 = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix1 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix1
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix1

    model_a_linear_b_matrix2 = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    model_a_layernorm_b_matrix2 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[2].b_matrix = model_a_linear_b_matrix2
    modelA.layers[3].b_matrix = model_a_layernorm_b_matrix2

    # set modelB
    model_b_linear_b_matrix1 = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix1 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix1
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix1

    model_b_linear_b_matrix2 = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ]
    )
    model_b_layernorm_b_matrix2 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[2].b_matrix = model_b_linear_b_matrix2
    modelB.layers[3].b_matrix = model_b_layernorm_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2, 3]
    module_types = [MaskedLayer]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
        fraction=True,
    )
    assert iou == (4 / 8 + 1 / 2 + 2 / 4 + 1 / 2) / 4

    # check identity for some types
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (4 / 8 + 2 / 4) / 2

    # check fraction
    layer_idx = [0]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == 4 / 8

    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (4 / 8 + 2 / 4) / 2

    # check sum
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=False
    )
    assert iou == (4 + 2) / 2


def test_iou_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix1 = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix1 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix1
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix1

    model_a_linear_b_matrix2 = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    model_a_layernorm_b_matrix2 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[2].b_matrix = model_a_linear_b_matrix2
    modelA.layers[3].b_matrix = model_a_layernorm_b_matrix2

    # set modelB
    model_b_linear_b_matrix1 = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix1 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix1
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix1

    model_b_linear_b_matrix2 = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ]
    )
    model_b_layernorm_b_matrix2 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[2].b_matrix = model_b_linear_b_matrix2
    modelB.layers[3].b_matrix = model_b_layernorm_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2, 3]
    module_types = [MaskedLayer]
    iou = iou_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
    )
    assert iou == 1

    # check identity for some types
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iou_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
    )
    assert iou == 1

    # check fraction
    layer_idx = [0]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iou_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == 3 / 4

    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iou_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (3 / 8 + 1 / 4) / (4 / 8 + 2 / 4)

    # check sum
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iou_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=False
    )
    assert iou == (3 + 1) / (4 + 2)


def test_iom_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix1 = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix1 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix1
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix1

    model_a_linear_b_matrix2 = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    model_a_layernorm_b_matrix2 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[2].b_matrix = model_a_linear_b_matrix2
    modelA.layers[3].b_matrix = model_a_layernorm_b_matrix2

    # set modelB
    model_b_linear_b_matrix1 = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix1 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix1
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix1

    model_b_linear_b_matrix2 = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ]
    )
    model_b_layernorm_b_matrix2 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[2].b_matrix = model_b_linear_b_matrix2
    modelB.layers[3].b_matrix = model_b_layernorm_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2, 3]
    module_types = [MaskedLayer]
    iou = iom_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
    )
    assert iou == 1

    # check identity for some types
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iom_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
    )
    assert iou == 1

    # check fraction
    layer_idx = [0]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iom_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == 1

    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iom_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (3 / 8 + 1 / 4) / (3 / 8 + 1 / 4)

    # check sum version
    layer_idx = [0, 1, 2, 3]
    module_types = [ContinuousMaskedWeightsLinear]
    iou = iom_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=False
    )
    assert iou == (3 + 1) / (3 + 1)


def test_iou_by_models(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix1 = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix1 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix1
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix1

    model_a_linear_b_matrix2 = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    model_a_layernorm_b_matrix2 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[2].b_matrix = model_a_linear_b_matrix2
    modelA.layers[3].b_matrix = model_a_layernorm_b_matrix2

    # set modelB
    model_b_linear_b_matrix1 = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix1 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix1
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix1

    model_b_linear_b_matrix2 = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ]
    )
    model_b_layernorm_b_matrix2 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[2].b_matrix = model_b_linear_b_matrix2
    modelB.layers[3].b_matrix = model_b_layernorm_b_matrix2

    # check identity for all layers
    iou = iou_models(
        [modelA],
    )
    assert iou == 1

    # check fraction
    iou = iou_models([modelA], fraction=True)
    assert iou == 1

    iou = iou_models([modelA, modelB], fraction=True)
    assert iou == (3 / 8 + 1 / 4) / (4 / 8 + 2 / 2 + 2 / 4 + 2 / 2)

    # check sum version
    iou = iou_models([modelA, modelB], fraction=False)
    assert iou == (3 + 1) / (4 + 2 + 2 + 2)


def test_iom_by_models(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix1 = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix1 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix1
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix1

    model_a_linear_b_matrix2 = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ]
    )
    model_a_layernorm_b_matrix2 = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[2].b_matrix = model_a_linear_b_matrix2
    modelA.layers[3].b_matrix = model_a_layernorm_b_matrix2

    # set modelB
    model_b_linear_b_matrix1 = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix1 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix1
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix1

    model_b_linear_b_matrix2 = torch.tensor(
        [
            [0, 0],
            [1, 0],
        ]
    )
    model_b_layernorm_b_matrix2 = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[2].b_matrix = model_b_linear_b_matrix2
    modelB.layers[3].b_matrix = model_b_layernorm_b_matrix2

    # check identity for all layers
    iou = iom_models(
        [modelA],
    )
    assert iou == 1

    # check fraction
    iou = iom_models([modelA], fraction=True)
    assert iou == 1

    iou = iom_models([modelA, modelB], fraction=True)
    assert iou == (3 / 8 + 1 / 4) / (3 / 8 + 1 / 2 + 1 / 4 + 1 / 2)

    # check sum version
    iou = iom_models([modelA, modelB], fraction=False)
    assert iou == (3 + 1) / (3 + 1 + 1 + 1)


def test_intersection_over_union_layers(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix

    # Manual IoU
    linear_intersect = (model_a_linear_b_matrix & model_b_linear_b_matrix).sum()
    norm_intersect = (model_a_layernorm_b_matrix & model_b_layernorm_b_matrix).sum()

    linear_union = (model_a_linear_b_matrix | model_b_linear_b_matrix).sum()
    norm_union = (model_a_layernorm_b_matrix | model_b_layernorm_b_matrix).sum()

    manuel_result = (linear_intersect + norm_intersect) / (linear_union + norm_union)

    # test intersection_over_union
    result = iou_by_layer_and_module([modelA, modelB], layer_idx=[0, 1], fraction=False)
    assert manuel_result == result


def test_intersection_over_minimum_layers(modelA, modelB):
    # set modelA
    model_a_linear_b_matrix = torch.tensor(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_a_layernorm_b_matrix = torch.tensor(
        [
            [0, 1],
        ]
    )
    modelA.layers[0].b_matrix = model_a_linear_b_matrix
    modelA.layers[1].b_matrix = model_a_layernorm_b_matrix

    # set modelB
    model_b_linear_b_matrix = torch.tensor(
        [
            [0, 0, 0, 1],
            [1, 0, 1, 0],
        ]
    )
    model_b_layernorm_b_matrix = torch.tensor(
        [
            [1, 0],
        ]
    )
    modelB.layers[0].b_matrix = model_b_linear_b_matrix
    modelB.layers[1].b_matrix = model_b_layernorm_b_matrix

    # Manual IoU
    linear_intersect = (model_a_linear_b_matrix & model_b_linear_b_matrix).sum()
    norm_intersect = (model_a_layernorm_b_matrix & model_b_layernorm_b_matrix).sum()

    minimum = min(
        (model_a_linear_b_matrix.sum() + model_a_layernorm_b_matrix.sum()),
        (model_b_linear_b_matrix.sum() + model_b_layernorm_b_matrix.sum()),
    )

    manuel_result = (linear_intersect + norm_intersect) / minimum

    # test intersection_over_union
    result = iom_by_layer_and_module([modelA, modelB], layer_idx=[0, 1], fraction=False)
    assert manuel_result == result
