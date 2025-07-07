"""
Subnetwork set operations
"""

import copy
from typing import Callable, List, Literal, Optional, Type

import torch
from torch import nn

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.utils import (
    get_architecture_block_from_module_name,
    get_current_layer_from_module_name,
)


def complement_(subnetwork: MaskedLayer):
    """
    Inverts the binary mask of the provided subnetwork.

    Args:
        subnetwork (MaskedLayer): The subnetwork to invert the masks for.
    """
    assert subnetwork.ticket is True
    setattr(subnetwork, "b_matrix", ~subnetwork.b_matrix.bool())


def binary_function_(
    subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer, operator: Callable
):
    """
    Replaces the binary mask of subnetwork_A with some operation between its own mask and the mask of subnetwork_B in place.

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """

    assert subnetwork_A.ticket is True
    assert subnetwork_B.ticket is True
    intermediate_result = operator(subnetwork_A.b_matrix, subnetwork_B.b_matrix)
    setattr(subnetwork_A, "b_matrix", intermediate_result.float())


def intersection_(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer):
    """
    Replaces the binary mask of subnetwork_A with the intersection of its own mask and the mask of subnetwork_B in place.

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    binary_function_(subnetwork_A, subnetwork_B, torch.logical_and)


def union_(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer):
    """
    Replaces the mask of subnetwork_A with the union of its mask with the mask of subnetwork_B in place.

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    binary_function_(subnetwork_A, subnetwork_B, torch.logical_or)


def difference_(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer):
    """
    Computes the set difference beween subnetwork_A and subnetwork_B.
    A / B = A ∩ ∁(B)

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    complement_(subnetwork_B)
    intersection_(subnetwork_A, subnetwork_B)


def sum_(subnetwork_A: MaskedLayer, subnetwork_B: MaskedLayer):
    """
    Computes the sum of subnetwork_A and subnetwork_B.
    A = A + B

    Args:
        subnetwork_A (MaskedLayer): The first subnetwork.
        subnetwork_B (MaskedLayer): The second subnetwork.
    """
    union_(subnetwork_A, subnetwork_B)

    # arithmetic addition of weights
    with torch.no_grad():
        setattr(
            subnetwork_A,
            "weight",
            nn.Parameter(subnetwork_A.weight + subnetwork_B.weight),
        )
        if subnetwork_A.bias is not None and subnetwork_B.bias is not None:
            setattr(
                subnetwork_A,
                "bias",
                nn.Parameter(subnetwork_A.bias + subnetwork_B.bias),
            )


def union_model(model_A: Transformer, model_B: Transformer) -> Transformer:
    """
    Performs the union of the b_matrix on all MaskedLayers in the entire provided model

    Args:
        model_A (Transformer): The model to replace the b_matrix of
        model_B (Transformer): The model to calculate the canghe of b_matrix with

    Returns:
        Transformer: The modified model
    """
    model_A = copy.deepcopy(model_A)
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            union_(sub_A, sub_B)
    return model_A


def intersection_model(model_A: Transformer, model_B: Transformer) -> Transformer:
    """
    Performs the intersection of the b_matrix on all MaskedLayers in the entire provided model

    Args:
        model_A (Transformer): The model to replace the b_matrix of
        model_B (Transformer): The model to calculate the canghe of b_matrix with

    Returns:
        Transformer: The modified model
    """
    model_A = copy.deepcopy(model_A)
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            intersection_(sub_A, sub_B)
    return model_A


def difference_model(model_A: Transformer, model_B: Transformer) -> Transformer:
    """
    Performs the difference of the b_matrix on all MaskedLayers in the entire provided model

    Args:
        model_A (Transformer): The model to replace the b_matrix of
        model_B (Transformer): The model to calculate the canghe of b_matrix with

    Returns:
        Transformer: The modified model
    """
    model_A = copy.deepcopy(model_A)
    model_B = copy.deepcopy(model_B)
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            difference_(sub_A, sub_B)
    return model_A


def sum_model(model_A: Transformer, model_B: Transformer) -> Transformer:
    """
    Performs the sum on all MaskedLayers in the entire provided model

    Args:
        model_A (Transformer): The model to replace the weights of
        model_B (Transformer): The model to calculate the change with

    Returns:
        Transformer: The modified model
    """
    model_A = copy.deepcopy(model_A)
    model_B = copy.deepcopy(model_B)
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            sum_(sub_A, sub_B)
    return model_A


def complement_model(subnetwork: Transformer) -> Transformer:
    """
    Performs the complement of the b_matrix on all MaskedLayers in the entire provided model

    Args:
        subnetwork (Transformer): The model to replace the b_matrix of

    Returns:
        Transformer: The modfied model
    """
    subnetwork = copy.deepcopy(subnetwork)
    for sub in subnetwork.modules():
        if isinstance(sub, MaskedLayer):
            complement_(sub)
    return subnetwork


def union_model_(model_A: Transformer, model_B: Transformer):
    """
    Performs the union of the b_matrix on all MaskedLayers in the entire provided model in place

    Args:
        model_A (Transformer): The model to replace the b_matrix of
        model_B (Transformer): The model to calculate the canghe of b_matrix with
    """
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            union_(sub_A, sub_B)


def intersection_model_(model_A: Transformer, model_B: Transformer):
    """
    Performs the intersection of the b_matrix on all MaskedLayers in the entire provided model in place

    Args:
        model_A (Transformer): The model to replace the b_matrix of
        model_B (Transformer): The model to calculate the canghe of b_matrix with
    """
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            intersection_(sub_A, sub_B)


def difference_model_(model_A: Transformer, model_B: Transformer):
    """
    Performs the difference of the b_matrix on all MaskedLayers in the entire provided model in place

    Args:
        model_A (Transformer): The model to replace the b_matrix of
        model_B (Transformer): The model to calculate the canghe of b_matrix with
    """
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            difference_(sub_A, sub_B)


def sum_model_(model_A: Transformer, model_B: Transformer):
    """
    Performs the sum on all MaskedLayers in the entire provided model in place

    Args:
        model_A (Transformer): The model to replace
        model_B (Transformer): The model to calculate the change with
    """
    for sub_A, sub_B in zip(model_A.modules(), model_B.modules()):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            sum_(sub_A, sub_B)


def complement_model_(subnetwork: Transformer):
    """
    Performs the complement of the b_matrix on all MaskedLayers in the entire provided model in place

    Args:
        subnetwork (Transformer): The model to replace the b_matrix of
    """
    for sub in subnetwork.modules():
        if isinstance(sub, MaskedLayer):
            complement_(sub)


def binary_operation_by_layer_and_module(
    model_A: Transformer,
    model_B: Transformer,
    operation: Callable,
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
) -> Transformer:
    """
    Replaces the b_matrix of a MaskedLayer module with result of a binary set operation, at a specified layer and for a specific MaskedLayer type.

    Args:
        model_A (Transformer): The model to replace the b_matrix of
        model_B (Transformer): The model to calculate the canghe of b_matrix with
        operation (Callable): The set operation to perform
        architecture_blocks: (Optional[List[Literal["encoder", "decoder", "projection"]]]): The part of the network to consider.
        layer_idx (Optional[List[int]]): The layers of the model to perform the replacement in. Adding -1 adds the non-layered objects to the list, e.g output norms and projection layer.
        module_types (Optional[List[Type]]: The type of MaskedModule to perform the replacement in

    Returns:
        Transformer
    """
    model_A = copy.deepcopy(model_A)
    for (name_A, sub_A), (name_B, sub_B) in zip(
        model_A.named_modules(), model_B.named_modules()
    ):
        if isinstance(sub_A, MaskedLayer) and isinstance(sub_B, MaskedLayer):
            if architecture_blocks:
                architecture_block = get_architecture_block_from_module_name(name_A)
                if architecture_block not in architecture_blocks:
                    continue

            if layer_idx:
                current_layer = get_current_layer_from_module_name(name_A)
                if current_layer not in layer_idx:
                    continue

            if not module_types:  # If no type is specifed, repalce everything
                module_types = [MaskedLayer]

            for acceptable_type in module_types:
                if isinstance(sub_A, acceptable_type) and isinstance(
                    sub_B, acceptable_type
                ):
                    operation(sub_A, sub_B)
    return model_A


def union_by_layer_and_module(
    model_A: Transformer,
    model_B: Transformer,
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
):
    """
    Replaces the b_matix of model_A with the union of model_A and model_B at a specified layerfor a specificed type of module.
    """
    return binary_operation_by_layer_and_module(
        model_A, model_B, union_, architecture_blocks, layer_idx, module_types
    )


def intersection_by_layer_and_module(
    model_A: Transformer,
    model_B: Transformer,
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
):
    """
    Replaces the b_matix of model_A with the intersection of model_A and model_B at a specified layer for a specificed type of module.
    """
    return binary_operation_by_layer_and_module(
        model_A, model_B, intersection_, architecture_blocks, layer_idx, module_types
    )


def difference_by_layer_and_module(
    model_A: Transformer,
    model_B: Transformer,
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
):
    """
    Replaces the b_matix of model_A with the difference of model_A and model_B at a specified layer for a specificed type of module.
    """
    return binary_operation_by_layer_and_module(
        model_A, model_B, difference_, architecture_blocks, layer_idx, module_types
    )


def sum_by_layer_and_module(
    model_A: Transformer,
    model_B: Transformer,
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
):
    """
    Replaces the b_matix and weights of model_A with the sum of model_A and model_B at a specified layer for a specificed type of module.
    """
    return binary_operation_by_layer_and_module(
        model_A, model_B, sum_, architecture_blocks, layer_idx, module_types
    )
