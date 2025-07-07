"""
Modules to find subnetworks via model weight pruning
"""

from collections import defaultdict
from typing import Any, Literal

import torch
import torch.nn as nn

from comp_rep.pruning.masked_base import MaskedLayer
from comp_rep.pruning.pruner import Pruner
from comp_rep.pruning.weight_pruning.masked_weights_layernorm import (
    ContinuousMaskedWeightsLayerNorm,
    SampledMaskedWeightsLayerNorm,
)
from comp_rep.pruning.weight_pruning.masked_weights_linear import (
    ContinuousMaskedWeightsLinear,
    SampledMaskedWeightsLinear,
)


class WeightPruner(Pruner):
    """
    A model wrapper that applies a masking strategy for model weight pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_method: Literal["continuous", "sampled"],
        maskedlayer_kwargs: dict[str, Any],
    ):
        super(WeightPruner, self).__init__(
            model=model,
            pruning_method=pruning_method,
            maskedlayer_kwargs=maskedlayer_kwargs,
        )

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
        self.freeze_initial_model()

        def replace_linear(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    if pruning_method == "continuous":
                        setattr(
                            module,
                            name,
                            ContinuousMaskedWeightsLinear(
                                child.weight, child.bias, **maskedlayer_kwargs
                            ),
                        )
                    elif self.pruning_method == "sampled":
                        setattr(
                            module,
                            name,
                            SampledMaskedWeightsLinear(
                                child.weight, child.bias, **maskedlayer_kwargs
                            ),
                        )
                    else:
                        raise ValueError("Invalid pruning strategy method provided")
                else:
                    replace_linear(child)

        def replace_layernorm(module: nn.Module) -> None:
            for name, child in module.named_children():
                if isinstance(child, nn.LayerNorm):
                    if pruning_method == "continuous":
                        setattr(
                            module,
                            name,
                            ContinuousMaskedWeightsLayerNorm(
                                child.normalized_shape,
                                child.weight,
                                child.bias,
                                child.eps,
                                **maskedlayer_kwargs,
                            ),
                        )
                    elif pruning_method == "sampled":
                        setattr(
                            module,
                            name,
                            SampledMaskedWeightsLayerNorm(
                                child.normalized_shape,
                                child.weight,
                                child.bias,
                                child.eps,
                                **maskedlayer_kwargs,
                            ),
                        )
                    else:
                        raise ValueError("Invalid pruning strategy method provided")
                else:
                    replace_layernorm(child)

        replace_linear(self.model)
        replace_layernorm(self.model)

    def get_remaining_mask(self) -> dict:
        """
        Computes the macro average remaining weights of the masked modules.

        Returns:
            float: the macro average
        """
        global_remaining_running = []
        layer_remaining_running = defaultdict(list)
        fine_grained_remaining_weights = {}
        for name, m in self.model.named_modules():
            if isinstance(m, MaskedLayer):
                local_remainder = m.compute_remaining_mask()
                name_list = name.split(".")
                try:
                    coder = f"{name_list[0]}_layer_{name_list[2]}"
                except IndexError:
                    coder = name
                layer_remaining_running[coder].append(local_remainder)
                fine_grained_remaining_weights[name] = local_remainder
                global_remaining_running.append(local_remainder)

        global_macro_average = sum(global_remaining_running) / len(
            global_remaining_running
        )
        per_layer_remaining = {
            key: sum(x) / len(x) for key, x in layer_remaining_running.items()
        }
        section_logs = {
            "global_remaining_mask": global_macro_average,
            "pruning_layers/": per_layer_remaining,
            "pruning_finegrained/": fine_grained_remaining_weights,
        }
        return section_logs

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


if __name__ == "__main__":

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    # Define masked layer arguments
    sampled_maskedlayer_kwargs = {"tau": 1.0, "num_masks": 2}

    # Create a simple model
    model = SimpleModel()
    print(f"Toy model: \n{model}")

    sampled_masked_model = WeightPruner(
        model,
        pruning_method="sampled",
        maskedlayer_kwargs=sampled_maskedlayer_kwargs,
    )
    print(f"Sampled Masked model: \n{model}")

    # Create dummy input data
    input_data = torch.randn(18, 10)
    sampled_output_data = model(input_data)
    print(f"in tensor: \n{input_data.shape}")
    print(f"Sampled out tensor: \n{sampled_output_data.shape}")

    # continuous mask
    cont_maskedlayer_kwargs = {
        "mask_initial_value": 1.0,
        "ticket": False,
    }
    model = SimpleModel()
    print(f"Toy model: \n{model}")

    cont_masked_model = WeightPruner(
        model,
        pruning_method="continuous",
        maskedlayer_kwargs=cont_maskedlayer_kwargs,
    )
    print(f"Continuous Masked model: \n{model}")

    cont_output_data = model(input_data)
    print(f"in tensor: \n{input_data.shape}")
    print(f"Continuous out tensor: \n{cont_output_data.shape}")
