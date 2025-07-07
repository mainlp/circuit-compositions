"""
Modules to find subnetworks via model pruning
"""

import abc
from typing import Any, Literal

import torch.nn as nn

from comp_rep.pruning.activation_pruning.masked_activation_base import (
    ContinuousMaskedActivationLayer,
)
from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.weight_pruning.masked_weights_base import (
    ContinuousMaskedWeightsLayer,
)


class Pruner(abc.ABC):
    """
    A model wrapper that applies a masking strategy for model pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_method: Literal["continuous", "sampled"],
        maskedlayer_kwargs: dict[str, Any],
    ):
        self.model = model
        self.pruning_method = pruning_method
        self.maskedlayer_kwargs = maskedlayer_kwargs
        self.init_model_pruning(pruning_method, maskedlayer_kwargs)

    def freeze_initial_model(self) -> None:
        """
        Freezes the initial model parameters to prevent updates during training.
        """
        for p in self.model.parameters():
            p.requires_grad = False

    @abc.abstractmethod
    def init_model_pruning(
        self,
        pruning_method: Literal["continuous", "sampled"],
        maskedlayer_kwargs: dict[str, Any],
    ) -> None:
        """
        Initializes the model pruning by replacing layers with masked layers.

        Args:
            pruning_method (Literal["continuous", "sampled"]): The pruning method to deploy.
            maskedlayer_kwargs (dict[str, Any]): Additional keyword-arguments for the masked layer.
        """
        pass

    def update_hyperparameters(self):
        """
        Updates the hyperparameters of the underlying Masked modules.

        Return:
            None
        """
        for m in self.model.modules():
            if isinstance(m, ContinuousMaskedWeightsLayer) or isinstance(
                m, ContinuousMaskedActivationLayer
            ):
                m.update_temperature()

    @abc.abstractmethod
    def get_remaining_mask(self) -> dict:
        """
        Computes the macro average remaining mask elements of the masked modules.

        Returns:
            float: the macro average
        """
        pass

    def compute_and_update_masks(self):
        """
        Computes and updates the masks for all MaskedLayer modules in the model.
        """
        for _, m in self.model.named_modules():
            if isinstance(m, MaskedLayer):
                m.compute_mask()

    def activate_ticket(self):
        """
        Activates the ticket for evaluation mode in the mask layers
        """
        for m in self.model.modules():
            if isinstance(m, MaskedLayer):
                m.ticket = True
                m.compute_mask()  # Set b_matrix

    def deactivate_ticket(self):
        """
        Deactivates the ticket for training mode in the mask layers
        """
        for m in self.model.modules():
            if isinstance(m, MaskedLayer):
                m.ticket = False
                m.compute_mask()  # Set b_matrix

    def compute_l1_norm(self):
        """
        Gathers all the L1 Norms
        """
        norms = 0.0
        for m in self.model.modules():
            if isinstance(m, MaskedLayer):
                norms += m.compute_l1_norm()
        return norms
