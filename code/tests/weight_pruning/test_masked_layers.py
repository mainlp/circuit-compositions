"""
Unit tests to test saving and loading of masked layers
"""

import os

import pytest
import torch
import torch.nn as nn

from comp_rep.pruning.weight_pruning.masked_weights_layernorm import (
    ContinuousMaskedWeightsLayerNorm,
    SampledMaskedWeightsLayerNorm,
)
from comp_rep.pruning.weight_pruning.masked_weights_linear import (
    ContinuousMaskedWeightsLinear,
    SampledMaskedWeightsLinear,
)


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


@pytest.fixture
def sampled_linear() -> SampledMaskedWeightsLinear:
    """
    Fixture to create a SampledMaskedWeightsLinear layer.

    Returns:
        SampledMaskedWeightsLinear: The SampledMaskedWeightsLinear layer.
    """
    input_dim = 4
    output_dim = 2
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))

    return SampledMaskedWeightsLinear(weight=linear_weight, bias=None, ticket=True)


@pytest.fixture
def sampled_layernorm() -> SampledMaskedWeightsLayerNorm:
    """
    Fixture to create a SampledMaskedWeightsLayerNorm layer.

    Returns:
        SampledMaskedWeightsLayerNorm: The SampledMaskedWeightsLayerNorm layer.
    """
    norm_shape = (2,)
    norm_layer_weights = nn.Parameter(torch.randn(norm_shape))

    return SampledMaskedWeightsLayerNorm(
        normalized_shape=norm_shape, weight=norm_layer_weights, bias=None, ticket=True
    )


def test_save_and_load_continuous_linear(
    continuous_linear: ContinuousMaskedWeightsLinear,
) -> None:
    """
    Test saving and loading the ContinuousMaskedWeightsLinear layer.

    Args:
        continuous_linear (ContinuousMaskedWeightsLinear): The ContinuousMaskedWeightsLinear layer.
    """
    input_dim = 4
    output_dim = 2

    x = torch.randn(1, 5, 4)
    initial_s_matrix = continuous_linear.s_matrix.clone()
    initial_output = continuous_linear(x).detach().clone()

    # modify s_matrix and b_matrix
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))
    model_linear_s_matrix = torch.Tensor(
        [
            [0.2, -0.2, -1.0, 0.6],
            [0.5, 0.3, 1.0, 0.0],
        ]
    )

    with torch.no_grad():
        continuous_linear.s_matrix.copy_(model_linear_s_matrix)
    continuous_linear.compute_mask()

    # new output
    modified_output = continuous_linear(x).detach().clone()

    # save the model state_dict
    torch.save(continuous_linear.state_dict(), "test_continuous_linear.pth")

    # create a new model instance and load the state_dict
    loaded_model = ContinuousMaskedWeightsLinear(
        weight=linear_weight, bias=None, ticket=True
    )
    loaded_model.load_state_dict(torch.load("test_continuous_linear.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        continuous_linear.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert not torch.allclose(
        initial_output, output_after
    ), "The initial outputs are equal to the outputs after loading."
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_continuous_linear.pth")


def test_save_and_load_continuous_layernorm(
    continuous_layernorm: ContinuousMaskedWeightsLayerNorm,
) -> None:
    """
    Test saving and loading the ContinuousMaskedWeightsLayerNorm.

    Args:
        continuous_layernorm (ContinuousMaskedWeightsLayerNorm): The ContinuousMaskedWeightsLayerNorm layer.
    """
    norm_shape = (2,)

    x = torch.randn(1, 2)
    initial_s_matrix = continuous_layernorm.s_matrix.clone()
    initial_output = continuous_layernorm(x).detach().clone()

    # modify s_matrix and b_matrix
    layernorm_weight = nn.Parameter(torch.randn(norm_shape))
    model_layernorm_s_matrix = torch.Tensor([0.6, -0.2])

    with torch.no_grad():
        continuous_layernorm.s_matrix.copy_(model_layernorm_s_matrix)
    continuous_layernorm.compute_mask()

    # new output
    modified_output = continuous_layernorm(x).detach().clone()

    # save the model state_dict
    torch.save(continuous_layernorm.state_dict(), "test_continuous_layernorm.pth")

    # create a new model instance and load the state_dict
    loaded_model = ContinuousMaskedWeightsLayerNorm(
        normalized_shape=norm_shape, weight=layernorm_weight, bias=None, ticket=True
    )
    loaded_model.load_state_dict(torch.load("test_continuous_layernorm.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        continuous_layernorm.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert not torch.allclose(
        initial_output, output_after
    ), "The initial outputs are equal to the outputs after loading."
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_continuous_layernorm.pth")


def test_save_and_load_sampled_linear(
    sampled_linear: SampledMaskedWeightsLinear,
) -> None:
    """
    Test saving and loading the SampledMaskedWeightsLinear layer.

    Args:
        sampled_linear (SampledMaskedWeightsLinear): The SampledMaskedWeightsLinear layer.
    """
    input_dim = 4
    output_dim = 2

    x = torch.randn(1, 5, 4)
    initial_s_matrix = sampled_linear.s_matrix.clone()

    # modify s_matrix and b_matrix
    linear_weight = nn.Parameter(torch.randn(output_dim, input_dim))
    model_linear_s_matrix = torch.Tensor(
        [
            [0.2, -0.2, -1.0, 0.6],
            [0.5, 0.3, 1.0, 0.0],
        ]
    )

    with torch.no_grad():
        sampled_linear.s_matrix.copy_(model_linear_s_matrix)
    sampled_linear.compute_mask()

    # new output
    modified_output = sampled_linear(x).detach().clone()

    # save the model state_dict
    torch.save(sampled_linear.state_dict(), "test_sampled_linear.pth")

    # create a new model instance and load the state_dict
    loaded_model = SampledMaskedWeightsLinear(
        weight=linear_weight, bias=None, ticket=True
    )
    loaded_model.load_state_dict(torch.load("test_sampled_linear.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        sampled_linear.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_sampled_linear.pth")


def test_save_and_load_sampled_layernorm(
    sampled_layernorm: SampledMaskedWeightsLayerNorm,
) -> None:
    """
    Test saving and loading the SampledMaskedWeightsLayerNorm.

    Args:
        sampled_layernorm (SampledMaskedWeightsLayerNorm): The SampledMaskedWeightsLayerNorm layer.
    """
    norm_shape = (2,)

    x = torch.randn(1, 2)
    initial_s_matrix = sampled_layernorm.s_matrix.clone()

    # modify s_matrix and b_matrix
    layernorm_weight = nn.Parameter(torch.randn(norm_shape))
    model_layernorm_s_matrix = torch.Tensor([0.6, -0.2])

    with torch.no_grad():
        sampled_layernorm.s_matrix.copy_(model_layernorm_s_matrix)
    sampled_layernorm.compute_mask()

    # new output
    modified_output = sampled_layernorm(x).detach().clone()

    # save the model state_dict
    torch.save(sampled_layernorm.state_dict(), "test_sampled_layernorm.pth")

    # create a new model instance and load the state_dict
    loaded_model = SampledMaskedWeightsLayerNorm(
        normalized_shape=norm_shape, weight=layernorm_weight, bias=None, ticket=True
    )
    loaded_model.load_state_dict(torch.load("test_sampled_layernorm.pth"))
    output_after = loaded_model(x)

    # assert that s_matrices are the same
    assert not torch.allclose(
        initial_s_matrix, loaded_model.s_matrix
    ), "The initial s_matrix is equal to the s_matrix after loading."
    assert torch.allclose(
        sampled_layernorm.s_matrix, loaded_model.s_matrix
    ), "The s_matrix before and after loading does not match."

    # assert that the outputs are the same
    assert torch.allclose(
        modified_output, output_after
    ), "The outputs before and after loading do not match."

    # cleanup
    os.remove("test_sampled_layernorm.pth")
