"""
Subnetwork metrics
"""

import copy
from typing import List, Literal, Optional, Type

from comp_rep.models.model import Transformer
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.subnetwork_set_operations import intersection_, union_
from comp_rep.utils import (
    get_architecture_block_from_module_name,
    get_current_layer_from_module_name,
)


def intersection_remaining_mask(
    masked_layers: List[MaskedLayer], fraction: bool = True
) -> float:
    """
    Calculates the sum or fraction of remaining mask in the intersection of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining mask in the layer, the sum if False. Defaults to True.

    Returns:
        float: The the sum or fraction of remaining mask in the intersection of the masked layers.

    Raises:
        AssertionError: If the list of masked layers is empty.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"

    intersection_layer = copy.deepcopy(masked_layers[0])

    for masked_layer in masked_layers:
        intersection_(intersection_layer, masked_layer)

    return intersection_layer.compute_remaining_mask(fraction)


def union_remaining_mask(
    masked_layers: List[MaskedLayer], fraction: bool = True
) -> float:
    """
    Calculates the sum or fraction of remaining mask in the union of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining mask in the layer, the sum if False. Defaults to True.

    Returns:
        float: The the sum or fraction of remaining mask in the union of the masked layers.

    Raises:
        AssertionError: If the list of masked layers is empty.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"

    union_layer = copy.deepcopy(masked_layers[0])

    for masked_layer in masked_layers:
        union_(union_layer, masked_layer)

    return union_layer.compute_remaining_mask(fraction)


def intersection_over_union(
    masked_layers: List[MaskedLayer], fraction: bool = False
) -> float:
    """
    Calculate the intersection over union of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining mask in the layer, the sum if False. Defaults to False.

    Returns:
        float: The intersection over union of the masked layers.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"

    return intersection_remaining_mask(masked_layers, fraction) / union_remaining_mask(
        masked_layers, fraction
    )


def intersection_over_minimum(
    masked_layers: List[MaskedLayer], fraction: bool = False
) -> float:
    """
    Calculates the intersection over minimum of multiple masked layers.

    Args:
        masked_layers (List[MaskedLayer]): A list of masked layers.
        fraction (bool): If True, computes the fraction of remaining mask in the layer, the sum if False. Defaults to False.

    Returns:
        float: The intersection over minimum of the masked layers.
    """
    assert len(masked_layers) > 0, f"Empty list of masked layers: {masked_layers}!"
    minimum_frac = min(
        [
            masked_layer.compute_remaining_mask(fraction)
            for masked_layer in masked_layers
        ]
    )

    return intersection_remaining_mask(masked_layers, fraction) / minimum_frac


def intersection_remaining_mask_by_layer_and_module(
    model_list: List[Transformer],
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
    fraction: bool = False,
) -> float:
    """
    Calculates the remaining mask of the intersection of dedicated layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        architecture_blocks: (Optional[List[Literal["encoder", "decoder", "projection"]]]): The part of the network to consider.
        layer_idx (Optional[List[int]]): A list of layer indices to consider. If None, all layers are considered.
        module_types (Optional[List[Type]]): A list of module types to consider. If None, all module types are considered.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The remaining mask for the intersection of specified layers and modules.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"

    intersection_mask: List[float] = []
    first_model = model_list[0]

    for module_name, subnetwork in first_model.named_modules():
        if isinstance(subnetwork, MaskedLayer):
            if architecture_blocks:
                architecture_block = get_architecture_block_from_module_name(
                    module_name
                )
                if architecture_block not in architecture_blocks:
                    continue

            if layer_idx:
                current_layer = get_current_layer_from_module_name(module_name)
                if current_layer not in layer_idx:
                    continue

            if not module_types:  # If no type is specifed, compute for everything
                module_types = [MaskedLayer]

            for acceptable_type in module_types:
                if isinstance(subnetwork, acceptable_type):
                    masked_layers = [
                        model.get_submodule(module_name) for model in model_list
                    ]
                    intersection_mask.append(
                        intersection_remaining_mask(
                            masked_layers, fraction  # type: ignore
                        )
                    )

    return sum(intersection_mask) / len(intersection_mask)


def union_remaining_mask_by_layer_and_module(
    model_list: List[Transformer],
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
    fraction: bool = False,
) -> float:
    """
    Calculates the remaining mask of the union of dedicated layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        architecture_blocks: (Optional[List[Literal["encoder", "decoder", "projection"]]]): The part of the network to consider.
        layer_idx (Optional[List[int]]): A list of layer indices to consider. If None, all layers are considered.
        module_types (Optional[List[Type]]): A list of module types to consider. If None, all module types are considered.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The remaining mask for the union of specified layers and modules.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"

    union_mask: List[float] = []
    first_model = model_list[0]

    for module_name, subnetwork in first_model.named_modules():
        if isinstance(subnetwork, MaskedLayer):
            if architecture_blocks:
                architecture_block = get_architecture_block_from_module_name(
                    module_name
                )
                if architecture_block not in architecture_blocks:
                    continue

            if layer_idx:
                current_layer = get_current_layer_from_module_name(module_name)
                if current_layer not in layer_idx:
                    continue

            if not module_types:  # If no type is specifed, compute for everything
                module_types = [MaskedLayer]

            for acceptable_type in module_types:
                if isinstance(subnetwork, acceptable_type):
                    masked_layers = [
                        model.get_submodule(module_name) for model in model_list
                    ]
                    union_mask.append(
                        union_remaining_mask(masked_layers, fraction)  # type: ignore
                    )

    return sum(union_mask) / len(union_mask)


def iou_by_layer_and_module(
    model_list: List[Transformer],
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
    average: bool = False,
    fraction: bool = False,
) -> float:
    """
    Calculates the Intersection over Union (IoU) for dedicated layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        architecture_blocks: (Optional[List[Literal["encoder", "decoder", "projection"]]]): The part of the network to consider.
        layer_idx (Optional[List[int]]): A list of layer indices to consider. If None, all layers are considered.
        module_types (Optional[List[Type]]): A list of module types to consider. If None, all module types are considered.
        average (bool, optional): Whether to compute the average IuO over individual layers, or the global version. Defaults to False.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoU value for the specified layers and modules.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"

    iou: float = 0.0
    intersection_layer: float = 0.0
    union_layer: float = 0.0
    intersection_layers: List[float] = []
    union_layers: List[float] = []

    first_model = model_list[0]

    for module_name, subnetwork in first_model.named_modules():
        if isinstance(subnetwork, MaskedLayer):
            if architecture_blocks:
                architecture_block = get_architecture_block_from_module_name(
                    module_name
                )
                if architecture_block not in architecture_blocks:
                    continue

            if layer_idx:
                current_layer = get_current_layer_from_module_name(module_name)
                if current_layer not in layer_idx:
                    continue

            if not module_types:  # If no type is specifed, compute for everything
                module_types = [MaskedLayer]

            for acceptable_type in module_types:
                if isinstance(subnetwork, acceptable_type):
                    masked_layers = [
                        model.get_submodule(module_name) for model in model_list
                    ]
                    intersection_layer = intersection_remaining_mask(
                        masked_layers, fraction  # type: ignore
                    )
                    union_layer = union_remaining_mask(masked_layers, fraction)  # type: ignore

                    intersection_layers.append(intersection_layer)
                    union_layers.append(union_layer)

    if average:
        for intersection_value, union_value in zip(intersection_layers, union_layers):
            if union_value > 0:
                iou += intersection_value / union_value
            else:
                iou += 1.0
        return iou / len(intersection_layers)
    else:
        if sum(union_layers) > 0:
            return sum(intersection_layers) / sum(union_layers)
        else:
            return 1.0


def iom_by_layer_and_module(
    model_list: List[Transformer],
    architecture_blocks: Optional[
        List[Literal["encoder", "decoder", "projection"]]
    ] = None,
    layer_idx: Optional[List[int]] = None,
    module_types: Optional[List[Type]] = None,
    average: bool = False,
    fraction: bool = False,
) -> float:
    """
    Calculates the Intersection over Minimum (IoM) for dedicated layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        architecture_blocks: (Optional[List[Literal["encoder", "decoder", "projection"]]]): The part of the network to consider.
        layer_idx (Optional[List[int]]): A list of layer indices to consider. If None, all layers are considered.
        module_types (Optional[List[Type]]): A list of module types to consider. If None, all module types are considered.
        average (bool, optional): Whether to compute the average IuM over individual layers, or the global version. Defaults to False.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoM value for the specified layers and modules.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    iom: float = 0.0
    intersection_layers: List[float] = []
    minimum_layers: List[float] = []

    first_model = model_list[0]
    for module_name, subnetwork in first_model.named_modules():
        if isinstance(subnetwork, MaskedLayer):
            if architecture_blocks:
                architecture_block = get_architecture_block_from_module_name(
                    module_name
                )
                if architecture_block not in architecture_blocks:
                    continue

            if layer_idx:
                current_layer = get_current_layer_from_module_name(module_name)
                if current_layer not in layer_idx:
                    continue

            if not module_types:  # If no type is specifed, compute for everything
                module_types = [MaskedLayer]

            for acceptable_type in module_types:
                if isinstance(subnetwork, acceptable_type):
                    masked_layers = [
                        model.get_submodule(module_name) for model in model_list
                    ]
                    intersection_layer = intersection_remaining_mask(
                        masked_layers, fraction  # type: ignore
                    )
                    minimum_masked_layer = min(
                        [
                            masked_layer.compute_remaining_mask(fraction=fraction)
                            for masked_layer in masked_layers
                        ]
                    )
                    intersection_layers.append(intersection_layer)
                    minimum_layers.append(minimum_masked_layer)

    if average:
        for intersection, minimum in zip(intersection_layers, minimum_layers):
            if minimum > 0:
                iom += intersection / minimum
            else:
                iom += 1
        return iom / len(intersection_layers)
    else:
        if sum(minimum_layers) == 0:
            return 1.0
        else:
            return sum(intersection_layers) / sum(minimum_layers)


def intersection_remaining_mask_models(
    model_list: List[Transformer], fraction: bool = False
) -> float:
    """
    Calculates the remaining mask for the intersection of all layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The sum/fraction of remaining mask.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    return intersection_remaining_mask_by_layer_and_module(
        model_list=model_list, fraction=fraction
    )


def union_remaining_mask_models(
    model_list: List[Transformer], fraction: bool = False
) -> float:
    """
    Calculates the remaining mask for the union of all layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The sum/fraction of remaining mask.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    return union_remaining_mask_by_layer_and_module(
        model_list=model_list, fraction=fraction
    )


def iou_models(
    model_list: List[Transformer], average: bool = False, fraction: bool = False
) -> float:
    """
    Calculates the Intersection over Union (IoU) for all layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        average (bool, optional): Whether to compute the average IuO over individual layers, or the global version. Defaults to False.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoU value.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    return iou_by_layer_and_module(
        model_list=model_list, average=average, fraction=fraction
    )


def iom_models(
    model_list: List[Transformer], average: bool = False, fraction: bool = False
) -> float:
    """
    Calculates the Intersection over Minimum (IoM) for all layers and modules in a list of Transformer models.

    Args:
        model_list (List[Transformer]): A list of Transformer models.
        average (bool, optional): Whether to compute the average IuM over individual layers, or the global version. Defaults to False.
        fraction (bool, optional): Whether to compute the fraction of remaining mask in the layer, or the sum. Defaults to False.

    Returns:
        float: The IoM value.

    Raises:
        AssertionError: If the model_list is empty.
    """
    assert len(model_list) > 0, f"Empty list of models: {model_list}!"
    return iom_by_layer_and_module(
        model_list=model_list, average=average, fraction=fraction
    )
