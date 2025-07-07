"""
Tests for subnetwork set  fingerained operations
"""

import pytest
import torch

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.subnetwork_set_operations import (
    difference_by_layer_and_module,
    intersection_by_layer_and_module,
    sum_by_layer_and_module,
    union_by_layer_and_module,
)
from comp_rep.pruning.weight_pruning.masked_weights_layernorm import (
    ContinuousMaskedWeightsLayerNorm,
)
from comp_rep.pruning.weight_pruning.masked_weights_linear import (
    ContinuousMaskedWeightsLinear,
)
from comp_rep.pruning.weight_pruning.weight_pruner import WeightPruner


@pytest.fixture
def modelA():
    input_vocabulary_size = 10
    output_vocabulary_size = 5
    layers = 6
    hidden_dim = 64
    dropout = 0.1
    model_hparams = {
        "input_vocabulary_size": input_vocabulary_size,
        "output_vocabulary_size": output_vocabulary_size,
        "num_transformer_layers": layers,
        "hidden_size": hidden_dim,
        "dropout": dropout,
    }
    model = Transformer(**model_hparams)
    pruner = WeightPruner(model, "continuous", {"mask_initial_value": 0.1})
    pruner.activate_ticket()
    change_b_matrix_(model)
    return model


@pytest.fixture
def modelB():
    input_vocabulary_size = 10
    output_vocabulary_size = 5
    layers = 6
    hidden_dim = 64
    dropout = 0.1
    model_hparams = {
        "input_vocabulary_size": input_vocabulary_size,
        "output_vocabulary_size": output_vocabulary_size,
        "num_transformer_layers": layers,
        "hidden_size": hidden_dim,
        "dropout": dropout,
    }
    model = Transformer(**model_hparams)
    pruner = WeightPruner(model, "continuous", {"mask_initial_value": 0.1})
    pruner.activate_ticket()
    change_b_matrix_(model, True)
    return model


@pytest.fixture
def modelC():
    input_vocabulary_size = 10
    output_vocabulary_size = 5
    layers = 6
    hidden_dim = 64
    dropout = 0.1
    model_hparams = {
        "input_vocabulary_size": input_vocabulary_size,
        "output_vocabulary_size": output_vocabulary_size,
        "num_transformer_layers": layers,
        "hidden_size": hidden_dim,
        "dropout": dropout,
    }
    model = Transformer(**model_hparams)
    pruner = WeightPruner(model, "continuous", {"mask_initial_value": 0.1})
    pruner.activate_ticket()
    for module in model.modules():
        if isinstance(module, MaskedLayer):
            b_matrix = torch.tensor(
                [
                    [1] * hidden_dim if i == 0 else [0] * hidden_dim
                    for i in range(0, hidden_dim)
                ]
            )
            module.b_matrix = b_matrix
    return model


def change_b_matrix_(model: Transformer, invert: bool = False):
    for module in model.modules():
        if isinstance(module, MaskedLayer):
            b_matrix = torch.triu(torch.ones(model.hidden_size, model.hidden_size))
            if invert:
                b_matrix = 1 - b_matrix
            module.b_matrix = b_matrix


def test_union_by_layer_and_module(modelA, modelB):
    expected_result = torch.ones(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = union_by_layer_and_module(
        modelA,
        modelB,
        ["encoder", "decoder", "projection"],
        [1],
        [ContinuousMaskedWeightsLayerNorm],
    )

    # Assert that the union works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)
    # Assert that the union is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)
    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_union_by_layer_and_module_encoder_only(modelA, modelB):
    expected_result = torch.ones(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = union_by_layer_and_module(
        modelA, modelB, ["encoder"], [1], [ContinuousMaskedWeightsLayerNorm]
    )

    # Assert that the union works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)

    # Assert that the decoder layer is not affected
    assert not torch.all(new_model.decoder.layers[1].norm_1.b_matrix == expected_result)

    # Assert that the union is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)

    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_intersection_by_layer_and_module(modelA, modelB):
    expected_result = torch.zeros(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = intersection_by_layer_and_module(
        modelA,
        modelB,
        ["encoder", "decoder", "projection"],
        [1],
        [ContinuousMaskedWeightsLayerNorm],
    )

    # Assert that the intersection works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)
    # Assert that the intersection is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)
    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_intersection_by_layer_and_module_encoder_only(modelA, modelB):
    expected_result = torch.zeros(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = intersection_by_layer_and_module(
        modelA, modelB, ["encoder"], [1], [ContinuousMaskedWeightsLayerNorm]
    )

    # Assert that the union works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)

    # Assert that the decoder layer is not affected
    assert not torch.all(new_model.decoder.layers[1].norm_1.b_matrix == expected_result)

    # Assert that the union is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)

    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_difference_by_layer_and_module(modelA, modelC):
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    full_triu_matrix = torch.triu(
        torch.ones(modelA.hidden_size, modelA.hidden_size)
    ).tolist()
    full_triu_matrix[0] = [0] * modelA.hidden_size
    expected_result = torch.tensor(full_triu_matrix)

    new_model = difference_by_layer_and_module(
        modelA,
        modelC,
        ["encoder", "decoder", "projection"],
        [1],
        [ContinuousMaskedWeightsLayerNorm],
    )

    # Assert that the difference works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)
    # Assert that the difference is not applied in another layer
    assert torch.all(new_model.encoder.layers[0].norm_1.b_matrix == unchanged_matrix)
    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_difference_by_layer_and_module_encoder_only(modelA, modelC):
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    full_triu_matrix = torch.triu(
        torch.ones(modelA.hidden_size, modelA.hidden_size)
    ).tolist()
    full_triu_matrix[0] = [0] * modelA.hidden_size
    expected_result = torch.tensor(full_triu_matrix)

    new_model = difference_by_layer_and_module(
        modelA, modelC, ["encoder"], [1], [ContinuousMaskedWeightsLayerNorm]
    )

    # Assert that the union works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)

    # Assert that the decoder layer is not affected
    assert not torch.all(new_model.decoder.layers[1].norm_1.b_matrix == expected_result)

    # Assert that the union is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)

    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )


def test_union_at_two_layers(modelA, modelB):
    expected_result = torch.ones(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = union_by_layer_and_module(
        modelA,
        modelB,
        ["encoder", "decoder", "projection"],
        [2, 4],
        [ContinuousMaskedWeightsLinear],
    )

    # Assert that the union works where it is supposed to
    assert torch.all(
        new_model.decoder.layers[2].cross_attention.query.b_matrix == expected_result
    )
    assert torch.all(
        new_model.encoder.layers[4].self_attention.value.b_matrix == expected_result
    )

    # Assert that the union is not applied in another layer
    assert torch.all(
        new_model.decoder.layers[0].cross_attention.query.b_matrix == unchanged_matrix
    )
    assert torch.all(
        new_model.encoder.layers[5].self_attention.value.b_matrix == unchanged_matrix
    )

    # Assert that no other module type is affected
    assert torch.all(new_model.encoder.layers[2].norm_1.b_matrix == unchanged_matrix)
    assert torch.all(new_model.encoder.layers[4].norm_1.b_matrix == unchanged_matrix)


def test_sum_by_layer_and_module(modelA, modelB):
    expected_result = torch.ones(modelA.hidden_size, modelA.hidden_size)
    unchanged_matrix = torch.triu(torch.ones(modelA.hidden_size, modelA.hidden_size))
    new_model = sum_by_layer_and_module(
        modelA,
        modelB,
        ["encoder", "decoder", "projection"],
        [1],
        [ContinuousMaskedWeightsLayerNorm],
    )

    # Assert that the union works where it is supposed to
    assert torch.all(new_model.encoder.layers[1].norm_1.b_matrix == expected_result)
    assert torch.all(
        new_model.encoder.layers[1].norm_1.weight
        == modelA.encoder.layers[1].norm_1.weight
        + modelB.encoder.layers[1].norm_1.weight
    )
    # Assert that the union is not applied in another layer
    assert not torch.all(new_model.encoder.layers[0].norm_1.b_matrix == expected_result)
    assert torch.all(
        new_model.encoder.layers[0].norm_1.weight
        == modelA.encoder.layers[1].norm_1.weight
    )
    # Assert that no other module type is affected
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.b_matrix == unchanged_matrix
    )
    assert torch.all(
        new_model.encoder.layers[0].self_attention.query.weight
        == modelA.encoder.layers[0].self_attention.query.weight
    )
