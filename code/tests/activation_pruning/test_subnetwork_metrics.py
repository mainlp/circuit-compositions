"""
Tests for subnetwork set operations
"""

import copy

import pytest
import torch
from torch import nn

from comp_rep.models.model import FeedForward, MultiHeadAttention
from comp_rep.pruning.activation_pruning.masked_activation_base import (
    ContinuousMaskedActivationLayer,
    MaskedActivationLayer,
)
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


class Transformer(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super(Transformer, self).__init__()
        mlp1 = FeedForward(input_dim)
        attn1 = MultiHeadAttention(input_dim, num_heads=num_heads, dropout=dropout)
        mlp2 = FeedForward(input_dim)
        mlp1_pruned = ContinuousMaskedActivationLayer(
            layer=mlp1, hidden_size=mlp1.hidden_size, ticket=True
        )
        attn1_pruned = ContinuousMaskedActivationLayer(
            layer=attn1, hidden_size=mlp1.hidden_size, ticket=True
        )
        mlp2_pruned = ContinuousMaskedActivationLayer(
            layer=mlp2, hidden_size=mlp1.hidden_size, ticket=True
        )

        self.layers = nn.ModuleList([mlp1_pruned, attn1_pruned, mlp2_pruned])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.layers[0](x)
        x = x + self.layers[1](x, x, x)
        x = x + self.layers[2](x)
        return x


@pytest.fixture
def modelA():
    input_dim = 4
    num_heads = 2
    dropout = 0.1
    return Transformer(input_dim, num_heads, dropout)


@pytest.fixture
def modelB():
    input_dim = 4
    num_heads = 2
    dropout = 0.1
    return Transformer(input_dim, num_heads, dropout)


@pytest.fixture
def continuous_feedforward() -> ContinuousMaskedActivationLayer:
    """
    Fixture to create a ContinuousMaskedActivationLayer.

    Returns:
        ContinuousMaskedActivationLayer: The MaskedActivationLayer.
    """
    input_dim = 4
    mlp1 = FeedForward(input_dim)

    return ContinuousMaskedActivationLayer(
        layer=mlp1, hidden_size=input_dim, ticket=True
    )


@pytest.fixture
def continuous_attention() -> ContinuousMaskedActivationLayer:
    """
    Fixture to create a ContinuousMaskedActivationLayer.

    Returns:
        ContinuousMaskedActivationLayer: The ContinuousMaskedActivationLayer.
    """
    input_dim = 4
    num_heads = 2
    dropout = 0.1
    attn1 = MultiHeadAttention(input_dim, num_heads=num_heads, dropout=dropout)

    return ContinuousMaskedActivationLayer(
        layer=attn1, hidden_size=input_dim, ticket=True
    )


def test_intersection_over_union(continuous_feedforward):
    # set s_matrix and b_matrix
    model_a_layer_s_matrix = torch.Tensor([0.2, -0.2, -1.0, 0.6])

    with torch.no_grad():
        continuous_feedforward.s_matrix.copy_(model_a_layer_s_matrix)
    continuous_feedforward.compute_mask()

    # test identity
    iou_value = intersection_over_union([continuous_feedforward])
    assert iou_value == 1

    # test equivalence
    iou_value = intersection_over_union(
        [continuous_feedforward, continuous_feedforward]
    )
    assert iou_value == 1

    # test complementary iou
    continuous_layer_complement = copy.deepcopy(continuous_feedforward)
    complement_(continuous_layer_complement)
    iou_value = intersection_over_union(
        [continuous_feedforward, continuous_layer_complement]
    )
    assert iou_value == 0

    # test fraction
    model_b_layer_s_matrix = torch.Tensor([0.6, -0.2, 1.0, 0.6])
    continuous_layer_copy = copy.deepcopy(continuous_feedforward)
    with torch.no_grad():
        continuous_layer_copy.s_matrix.copy_(model_b_layer_s_matrix)
    continuous_layer_copy.compute_mask()
    iou_value = intersection_over_union([continuous_feedforward, continuous_layer_copy])
    assert iou_value == 2 / 3

    # test symmetry
    iou_value = intersection_over_union([continuous_layer_copy, continuous_feedforward])
    assert iou_value == 2 / 3

    # test 3 entries
    iou_value = intersection_over_union(
        [continuous_feedforward, continuous_layer_complement, continuous_layer_copy]
    )
    assert iou_value == 0


def test_intersection_over_minimum(continuous_feedforward):
    # set s_matrix and b_matrix
    model_a_layer_s_matrix = torch.Tensor([0.2, -0.2, -1.0, 0.6])

    with torch.no_grad():
        continuous_feedforward.s_matrix.copy_(model_a_layer_s_matrix)
    continuous_feedforward.compute_mask()

    # test identity
    iou_value = intersection_over_minimum([continuous_feedforward])
    assert iou_value == 1

    # test equivalence
    iou_value = intersection_over_minimum(
        [continuous_feedforward, continuous_feedforward]
    )
    assert iou_value == 1

    # test complementary iou
    continuous_layer_complement = copy.deepcopy(continuous_feedforward)
    complement_(continuous_layer_complement)
    iou_value = intersection_over_minimum(
        [continuous_feedforward, continuous_layer_complement]
    )
    assert iou_value == 0

    # test fraction
    model_b_layer_s_matrix = torch.Tensor([0.6, -0.2, 1.0, 0.6])
    continuous_layer_copy = copy.deepcopy(continuous_feedforward)
    with torch.no_grad():
        continuous_layer_copy.s_matrix.copy_(model_b_layer_s_matrix)
    continuous_layer_copy.compute_mask()
    iou_value = intersection_over_minimum(
        [continuous_feedforward, continuous_layer_copy]
    )
    assert iou_value == 2 / 2

    # test symmetry
    iou_value = intersection_over_minimum(
        [continuous_layer_copy, continuous_feedforward]
    )
    assert iou_value == 2 / 2

    # test 3 entries
    iou_value = intersection_over_minimum(
        [continuous_feedforward, continuous_layer_complement, continuous_layer_copy]
    )
    assert iou_value == 0


def test_intersection_remaining_mask_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_mlp1layer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    model_a_attnlayer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    modelA.layers[0].b_matrix = model_a_mlp1layer_b_matrix1
    modelA.layers[1].b_matrix = model_a_attnlayer_b_matrix1

    model_a_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelA.layers[2].b_matrix = model_a_mlp2layer_b_matrix2

    # set modelB
    model_b_mlp1layer_b_matrix1 = torch.tensor([1, 0, 0, 1])
    model_b_attnlayer_b_matrix1 = torch.tensor([0, 0, 0, 1])
    modelB.layers[0].b_matrix = model_b_mlp1layer_b_matrix1
    modelB.layers[1].b_matrix = model_b_attnlayer_b_matrix1

    model_b_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelB.layers[2].b_matrix = model_b_mlp2layer_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
        fraction=True,
    )
    assert iou == (2 / 4 + 2 / 4 + 2 / 4) / 3

    # check fraction
    layer_idx = [0]
    module_types = [ContinuousMaskedActivationLayer]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == 1 / 4

    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (1 / 4 + 1 / 4 + 2 / 4) / 3

    # check sum
    layer_idx = [0, 1, 2]
    module_types = [MaskedActivationLayer]
    iou = intersection_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=False
    )
    assert iou == (1 + 1 + 2) / 3


def test_union_remaining_weights_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_mlp1layer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    model_a_attnlayer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    modelA.layers[0].b_matrix = model_a_mlp1layer_b_matrix1
    modelA.layers[1].b_matrix = model_a_attnlayer_b_matrix1

    model_a_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelA.layers[2].b_matrix = model_a_mlp2layer_b_matrix2

    # set modelB
    model_b_mlp1layer_b_matrix1 = torch.tensor([1, 0, 0, 1])
    model_b_attnlayer_b_matrix1 = torch.tensor([0, 0, 0, 1])
    modelB.layers[0].b_matrix = model_b_mlp1layer_b_matrix1
    modelB.layers[1].b_matrix = model_b_attnlayer_b_matrix1

    model_b_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelB.layers[2].b_matrix = model_b_mlp2layer_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
        fraction=True,
    )
    assert iou == (2 / 4 + 2 / 4 + 2 / 4) / 3

    # check fraction
    layer_idx = [0]
    module_types = [ContinuousMaskedActivationLayer]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == 3 / 4

    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=True
    )
    assert iou == (3 / 4 + 2 / 4 + 2 / 4) / 3

    # check sum
    layer_idx = [0, 1, 2]
    module_types = [MaskedActivationLayer]
    iou = union_remaining_mask_by_layer_and_module(
        [modelA, modelB], layer_idx=layer_idx, module_types=module_types, fraction=False
    )
    assert iou == (3 + 2 + 2) / 3


def test_iou_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_mlp1layer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    model_a_attnlayer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    modelA.layers[0].b_matrix = model_a_mlp1layer_b_matrix1
    modelA.layers[1].b_matrix = model_a_attnlayer_b_matrix1

    model_a_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelA.layers[2].b_matrix = model_a_mlp2layer_b_matrix2

    # set modelB
    model_b_mlp1layer_b_matrix1 = torch.tensor([1, 0, 0, 1])
    model_b_attnlayer_b_matrix1 = torch.tensor([0, 0, 0, 1])
    modelB.layers[0].b_matrix = model_b_mlp1layer_b_matrix1
    modelB.layers[1].b_matrix = model_b_attnlayer_b_matrix1

    model_b_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelB.layers[2].b_matrix = model_b_mlp2layer_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = iou_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
        fraction=True,
    )
    assert iou == (2 + 2 + 2) / (2 + 2 + 2)

    # check fraction
    layer_idx = [2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = iou_by_layer_and_module(
        [modelA, modelB],
        layer_idx=layer_idx,
        module_types=module_types,
        average=False,
        fraction=True,
    )
    assert iou == 2 / 2

    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = iou_by_layer_and_module(
        [modelA, modelB],
        layer_idx=layer_idx,
        module_types=module_types,
        average=False,
        fraction=True,
    )

    assert iou == (1 / 4 + 1 / 4 + 2 / 4) / (3 / 4 + 2 / 4 + 2 / 4)

    # check sum
    layer_idx = [0, 1, 2]
    module_types = [MaskedActivationLayer]
    iou = iou_by_layer_and_module(
        [modelA, modelB],
        layer_idx=layer_idx,
        module_types=module_types,
        average=False,
        fraction=False,
    )
    assert iou == (1 + 1 + 2) / (3 + 2 + 2)

    # check average
    layer_idx = [0, 1, 2]
    module_types = [MaskedActivationLayer]
    iou = iou_by_layer_and_module(
        [modelA, modelB],
        layer_idx=layer_idx,
        module_types=module_types,
        average=True,
        fraction=False,
    )
    assert iou == (1 / 3 + 1 / 2 + 2 / 2) / 3


def test_iom_by_layer_and_module(modelA, modelB):
    # set modelA
    model_a_mlp1layer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    model_a_attnlayer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    modelA.layers[0].b_matrix = model_a_mlp1layer_b_matrix1
    modelA.layers[1].b_matrix = model_a_attnlayer_b_matrix1

    model_a_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelA.layers[2].b_matrix = model_a_mlp2layer_b_matrix2

    # set modelB
    model_b_mlp1layer_b_matrix1 = torch.tensor([1, 0, 0, 1])
    model_b_attnlayer_b_matrix1 = torch.tensor([0, 0, 0, 1])
    modelB.layers[0].b_matrix = model_b_mlp1layer_b_matrix1
    modelB.layers[1].b_matrix = model_b_attnlayer_b_matrix1

    model_b_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelB.layers[2].b_matrix = model_b_mlp2layer_b_matrix2

    # check identity for all layers
    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = iom_by_layer_and_module(
        [modelA],
        layer_idx=layer_idx,
        module_types=module_types,
        fraction=True,
        average=False,
    )
    assert iou == (2 / 4 + 2 / 4 + 2 / 4) / (2 / 4 + 2 / 4 + 2 / 4)

    # check fraction
    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = iom_by_layer_and_module(
        [modelA, modelB],
        layer_idx=layer_idx,
        module_types=module_types,
        average=True,
        fraction=True,
    )
    assert iou == ((1 / 4) / (2 / 4) + (1 / 4) / (1 / 4) + (2 / 4) / (2 / 4)) / 3

    layer_idx = [0, 1, 2]
    module_types = [ContinuousMaskedActivationLayer]
    iou = iom_by_layer_and_module(
        [modelA, modelB],
        layer_idx=layer_idx,
        module_types=module_types,
        average=False,
        fraction=True,
    )
    assert iou == (1 / 4 + 1 / 4 + 2 / 4) / (2 / 4 + 1 / 4 + 2 / 4)

    # check sum
    layer_idx = [0, 1, 2]
    module_types = [MaskedActivationLayer]
    iou = iom_by_layer_and_module(
        [modelA, modelB],
        layer_idx=layer_idx,
        module_types=module_types,
        average=True,
        fraction=False,
    )
    assert iou == (1 / 2 + 1 / 1 + 2 / 2) / 3


def test_iou_by_models(modelA, modelB):
    # set modelA
    model_a_mlp1layer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    model_a_attnlayer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    modelA.layers[0].b_matrix = model_a_mlp1layer_b_matrix1
    modelA.layers[1].b_matrix = model_a_attnlayer_b_matrix1

    model_a_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelA.layers[2].b_matrix = model_a_mlp2layer_b_matrix2

    # set modelB
    model_b_mlp1layer_b_matrix1 = torch.tensor([1, 0, 0, 1])
    model_b_attnlayer_b_matrix1 = torch.tensor([0, 0, 0, 1])
    modelB.layers[0].b_matrix = model_b_mlp1layer_b_matrix1
    modelB.layers[1].b_matrix = model_b_attnlayer_b_matrix1

    model_b_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelB.layers[2].b_matrix = model_b_mlp2layer_b_matrix2

    # check identity for all layers
    iou = iou_models(
        [modelA],
    )
    assert iou == 1

    # check average
    iou = iou_models([modelA], average=False, fraction=True)
    assert iou == 1

    iou = iou_models([modelA], average=False, fraction=True)
    assert iou == 1

    # check fraction
    iou = iou_models([modelA, modelB], average=False, fraction=True)
    assert iou == (1 / 4 + 1 / 4 + 2 / 4) / (3 / 4 + 2 / 4 + 2 / 4)

    # check sum version
    iou = iou_models([modelA, modelB], average=False, fraction=False)
    assert iou == (1 + 1 + 2) / (3 + 2 + 2)


def test_iom_by_models(modelA, modelB):
    # set modelA
    model_a_mlp1layer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    model_a_attnlayer_b_matrix1 = torch.tensor([0, 1, 0, 1])
    modelA.layers[0].b_matrix = model_a_mlp1layer_b_matrix1
    modelA.layers[1].b_matrix = model_a_attnlayer_b_matrix1

    model_a_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelA.layers[2].b_matrix = model_a_mlp2layer_b_matrix2

    # set modelB
    model_b_mlp1layer_b_matrix1 = torch.tensor([1, 0, 0, 1])
    model_b_attnlayer_b_matrix1 = torch.tensor([0, 0, 0, 1])
    modelB.layers[0].b_matrix = model_b_mlp1layer_b_matrix1
    modelB.layers[1].b_matrix = model_b_attnlayer_b_matrix1

    model_b_mlp2layer_b_matrix2 = torch.tensor([0, 1, 1, 0])

    modelB.layers[2].b_matrix = model_b_mlp2layer_b_matrix2
    # check identity for all layers
    iou = iom_models(
        [modelA],
    )
    assert iou == 1

    # check fraction
    iou = iom_models([modelA], average=True, fraction=True)
    assert iou == 1

    iou = iom_models([modelA], average=False, fraction=True)
    assert iou == 1

    iou = iom_models([modelA, modelB], average=True, fraction=True)
    assert iou == ((1 / 4) / (2 / 4) + (1 / 4) / (1 / 4) + (2 / 4) / (2 / 4)) / 3

    # check sum version
    iou = iom_models([modelA, modelB], average=False, fraction=False)
    assert iou == (1 + 1 + 2) / (2 + 1 + 2)
