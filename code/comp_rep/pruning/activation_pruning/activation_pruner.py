"""
Modules to find subnetworks via model activation pruning
"""

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Literal

import torch
import torch.nn as nn

from comp_rep.models.model import FeedForward, MultiHeadAttention
from comp_rep.pruning.activation_pruning.masked_activation_base import (
    ContinuousMaskedActivationLayer,
    MaskedActivationLayer,
)
from comp_rep.pruning.pruner import Pruner

PRUNED_NODES = [FeedForward, MultiHeadAttention]
CURR_FILE_PATH = Path(__file__).resolve()
MEAN_ABLATION_VALUES_PATH = CURR_FILE_PATH.parents[1] / "mean_ablation_values"


class ActivationPruner(Pruner):
    """
    A model wrapper that applies a masking strategy for model activation pruning.
    """

    def __init__(
        self,
        model: nn.Module,
        pruning_method: Literal["continuous", "sampled"],
        ablation_value: Literal["zero", "mean"],
        subtask: str,
        maskedlayer_kwargs: dict[str, Any],
    ):

        self.ablation_value = ablation_value
        self.subtask = subtask
        super(ActivationPruner, self).__init__(
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
        if self.ablation_value == "mean":
            from comp_rep.utils import load_json  # Prevent circular import

            full_path = (
                MEAN_ABLATION_VALUES_PATH
                / f"{self.subtask.lower()}_mean_ablation_values.json"
            )
            ablation_data = load_json(full_path)

        def replace_activation_layer(module: nn.Module, parent_name: str) -> None:
            for name, child in module.named_children():
                parents = f"{parent_name}.{name}"
                if any(
                    isinstance(child, pruning_node) for pruning_node in PRUNED_NODES
                ):
                    if self.ablation_value == "mean":
                        try:
                            ablation_value = torch.tensor(ablation_data[parents])
                        except KeyError:
                            raise KeyError(
                                "Failed to match model name to ablation value key"
                            )
                    else:
                        ablation_value = torch.zeros(1)
                    if pruning_method == "continuous":
                        setattr(
                            module,
                            name,
                            ContinuousMaskedActivationLayer(
                                layer=child,
                                hidden_size=child.hidden_size,
                                ablation_values=ablation_value,
                                **maskedlayer_kwargs,
                            ),
                        )
                    elif self.pruning_method == "sampled":
                        raise ValueError(
                            "Sampled pruning not yet implemented for activation pruning."
                        )
                    else:
                        raise ValueError("Invalid pruning strategy method provided")
                else:
                    replace_activation_layer(child, parents)

        replace_activation_layer(self.model, "model")

    def set_ablation_value(self, ablation_data: Dict) -> None:
        """
        Sets the ablation value for the model.

        Args:
            ablation_data (Dict): A dictionary containing the ablation values for each layer.

        Returns:
            None
        """
        assert (
            self.ablation_value == "mean"
        ), "ablation_value is not 'mean'! Do not set new ablation value."

        def set_ablation_value_layer(module: nn.Module, parent_name: str) -> None:
            for name, child in module.named_children():
                parents = f"{parent_name}.{name}"
                if isinstance(child, ContinuousMaskedActivationLayer):
                    try:
                        ablation_value = torch.tensor(ablation_data[parents])
                    except KeyError:
                        raise KeyError(
                            "Failed to match model name to ablation value key"
                        )
                    setattr(
                        child,
                        "ablation_values",
                        ablation_value.to(child.ablation_values.device),
                    )
                else:
                    set_ablation_value_layer(child, parents)

        set_ablation_value_layer(self.model, "model")

    def get_remaining_mask(self) -> dict:
        """
        Computes the macro average remaining activations of the masked modules.

        Returns:
            float: the macro average
        """
        global_remaining_running = []
        layer_remaining_running = defaultdict(list)
        fine_grained_remaining_activations = {}
        for name, m in self.model.named_modules():
            if isinstance(m, MaskedActivationLayer):
                local_remainder = m.compute_remaining_mask()
                name_list = name.split(".")
                try:
                    coder = f"{name_list[0]}_layer_{name_list[2]}"
                except IndexError:
                    coder = name
                layer_remaining_running[coder].append(local_remainder)
                fine_grained_remaining_activations[name] = local_remainder
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
            "pruning_finegrained/": fine_grained_remaining_activations,
        }
        return section_logs


if __name__ == "__main__":

    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.mlp1 = FeedForward(10)
            self.attn1 = MultiHeadAttention(10, num_heads=2, dropout=0.1)
            self.mlp2 = FeedForward(10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.mlp1(x)
            x = x + self.attn1(x, x, x)
            x = x + self.mlp2(x)
            return x

    # Create a simple model
    model = SimpleModel()
    print(f"Toy model: \n{model}")

    # Create dummy input data
    input_data = torch.randn(18, 10, 10)
    print(f"in tensor: \n{input_data.shape}")

    # continuous mask
    cont_maskedlayer_kwargs = {
        "mask_initial_value": 1.0,
        "ticket": False,
    }
    model = SimpleModel()
    pruning_method: Literal["continuous"] = "continuous"
    subtask = "copy"
    ablation_value: Literal["zero", "mean"] = "zero"
    print(f"Toy model: \n{model}")

    cont_masked_model = ActivationPruner(
        model,
        pruning_method,
        ablation_value,
        subtask,
        maskedlayer_kwargs=cont_maskedlayer_kwargs,
    )
    print(f"Continuous Masked model: \n{model}")

    cont_output_data = model(input_data)
    print(f"in tensor: \n{input_data.shape}")
    print(f"Continuous out tensor: \n{cont_output_data.shape}")
    print(cont_masked_model.get_remaining_mask())
