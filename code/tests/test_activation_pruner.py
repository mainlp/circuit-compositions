from pathlib import Path

import pytest
import torch

from comp_rep.models.model import Transformer
from comp_rep.pruning.activation_pruning.activation_pruner import ActivationPruner
from comp_rep.utils import load_json

CURR_FILE_PATH = Path(__file__).resolve()
MEAN_ABLATION_VALUES_PATH = (
    CURR_FILE_PATH.parents[1]
    / "comp_rep/pruning/mean_ablation_values/copy_mean_ablation_values.json"
)


@pytest.fixture
def modelA():
    return Transformer(10, 10, 6, 512, 0.2)


def test_mean_ablation_loading(modelA):
    ablation_values = load_json(MEAN_ABLATION_VALUES_PATH)
    pruner = ActivationPruner(
        modelA,
        pruning_method="continuous",
        ablation_value="mean",
        subtask="copy",
        maskedlayer_kwargs={},
    )
    value_to_be_loaded = torch.tensor(
        ablation_values["model.encoder.layers.2.feed_forward"]
    )
    loaded_value = pruner.model.encoder.layers[2].feed_forward.ablation_values
    assert torch.all(torch.eq(value_to_be_loaded, loaded_value))


def test_zero_ablation_loading(modelA):
    pruner = ActivationPruner(
        modelA,
        pruning_method="continuous",
        ablation_value="zero",
        subtask="copy",
        maskedlayer_kwargs={},
    )
    value_to_be_loaded = torch.zeros(1)
    loaded_value = pruner.model.encoder.layers[2].feed_forward.ablation_values
    assert torch.all(torch.eq(value_to_be_loaded, loaded_value))
