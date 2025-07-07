"""
Utility functions and modules
"""

import argparse
import gc
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import Lang
from comp_rep.models.lightning_models import LitTransformer
from comp_rep.models.lightning_pruned_models import LitPrunedModel
from comp_rep.models.model import Transformer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class ValidatePredictionPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if ".csv" in str(values):
            parser.error(
                "Only provide the path where you want to store the predictions file"
            )
        Path(values).mkdir(parents=True, exist_ok=True)
        setattr(namespace, self.dest, values)


class ValidateTaskOptions(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        for v in values:
            if v not in POSSIBLE_TASKS:
                parser.error(f"{v} is not a valid task option")
        setattr(namespace, self.dest, values)


class ValidateSavePath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        Path(values).mkdir(parents=True, exist_ok=True)
        if (Path.cwd() / str(values) / "model.ckpt").exists():
            logging.warning("There is already a model file saved on that path!")
        setattr(namespace, self.dest, values)


class ValidateWandbPath(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        Path(values).mkdir(parents=True, exist_ok=True)
        setattr(namespace, self.dest, values)


def keystoint(x: Dict) -> Dict:
    """
    Converts the keys in the input dictionary to integers if they are digits, otherwise keeps them as they are,
    and returns the modified dictionary.

    Args:
        x (Dict): The input dictionary to convert the keys.

    Returns:
        Dict: A dictionary with keys converted to integers if they are digits.
    """
    return {(int(k) if k.isdigit() else k): v for k, v in x.items()}


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generation in torch, numpy, and random libraries.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_tokenizer_dict(input_lang: Lang, output_lang: Lang) -> Dict:
    """
    Serializes the tokenizers

    Args:
        input_lang (Lang): The input language representation.
        output_lang (Lang): The output language representation.

    Returns:
        Dict: The language mapping.
    """
    return {
        "input_language": {
            "index2word": input_lang.index2word,
            "word2index": input_lang.word2index,
        },
        "output_language": {
            "index2word": output_lang.index2word,
            "word2index": output_lang.word2index,
        },
    }


def save_tokenizer(path: Path, tokenizer: dict) -> None:
    """
    Saves the index2word and word2index dicts from the languages

    Args:
        path (Path): The path to the tokenizer file.
        tokenizer (dict): The dictionary representing the tokenizer.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path / "tokenizers.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer, f, ensure_ascii=False, indent=4)


def load_tokenizer(path: Path) -> Dict:
    """
    Loads the index2word and word2index dicts for both languages

    Args:
        path (Path): The path to the tokenizer file.

    Returns:
        Dict: The tokenizer.
    """
    tokenizer_path = path / "tokenizers.json"
    return load_json(tokenizer_path, object_hook=keystoint)


def load_json(path: Path, object_hook=None) -> Dict:
    """
    Loads a JSON file from disk and returns it as a python dict object.

    Args:
        path (Path): The path to the json file.

    Raises:
        IOError

    Returns
        Dict: The serialized python dict.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f, object_hook=object_hook)
    except IOError:
        raise IOError(f"Failed to read JSON file from disk at path: {path}")
    return data


def load_tensor(path: Path) -> torch.Tensor:
    """
    Loads a torch.Tensor from the provided path

    Args:
        path (Path): The path to the tensor file.

    Raises:
        IOError

    Returns:
        torch.Tensor
    """
    try:
        tensor = torch.load(path)
    except IOError:
        raise IOError(f"Failed to load tensor from path: {str(path)}")
    return tensor


def setup_logging(verbosity: int = 1) -> None:
    """
    Set up logging based on the verbosity level.

    Args:
        verbosity (int): Verbosity level.
    """
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    else:
        raise ValueError(
            f"Invalid log-level specified: {verbosity}! Should be 0, 1, or 2."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )


def load_model(
    model_path: Path,
    is_masked: bool,
    model: Optional[nn.Module] = None,
    return_pl: Optional[bool] = False,
):
    """
    Loads a model from a given checkpoint.

    Args:
        model_path (Path): The path to the model checkpoint.
        is_masked (bool): Whether the model is masked or not.
        model (Optional[nn.Module]): The model to load from the checkpoint. Defaults to None.

    Returns:
        nn.Module: The loaded model.
    """
    if is_masked:
        if model is None:
            model = create_transformer_from_checkpoint(model_path)

        pl_pruner = LitPrunedModel.load_from_checkpoint(model_path, model=model)  # type: ignore
        pl_pruner.pruner.activate_ticket()
        pl_pruner.pruner.compute_and_update_masks()
        if return_pl:
            model = pl_pruner
        else:
            model = pl_pruner.model
    else:
        pl_transformer = LitTransformer.load_from_checkpoint(model_path)  # type: ignore
        if return_pl:
            model = pl_transformer
        else:
            model = pl_transformer.model
    return model


def create_transformer_from_checkpoint(model_path: Path) -> nn.Module:
    """
    Creates a transformer model from a given checkpoint.

    Args:
        model_path (Path): The path to the model checkpoint.

    Returns:
        nn.Module: The created transformer model.
    """
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    input_vocabulary_size = vars(checkpoint["hyper_parameters"]["args"])[
        "input_vocabulary_size"
    ]
    output_vocabulary_size = vars(checkpoint["hyper_parameters"]["args"])[
        "output_vocabulary_size"
    ]
    num_transformer_layers = vars(checkpoint["hyper_parameters"]["args"])[
        "num_transformer_layers"
    ]
    hidden_size = vars(checkpoint["hyper_parameters"]["args"])["hidden_size"]
    dropout = vars(checkpoint["hyper_parameters"]["args"])["dropout"]

    base_model = Transformer(
        input_vocabulary_size,
        output_vocabulary_size,
        num_transformer_layers,
        hidden_size,
        dropout,
    ).to(DEVICE)

    return base_model


def save_list_to_csv(file_path: Path, data: List[str]) -> None:
    """
    Save a list of strings to a CSV file.

    Parameters:
    file_path (str): Name of the CSV file to save.
    data (List[str]): List of strings to save.
    """
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", newline="") as csvfile:
        for item in data:
            csvfile.write(item + "\n")


def get_architecture_block_from_module_name(module_name: str) -> str:
    """
    Returns the encoder or decoder association from a torch.named_modules string
    """
    pattern = r"\b(encoder|decoder|projection)\b"
    matches = re.findall(pattern, module_name)
    assert (
        len(matches) == 1
    ), f"No 'encoder', 'decoder', or 'projection' in module name! Can not parse architecture block. {module_name}"
    return matches[0]


def get_current_layer_from_module_name(module_name: str) -> int:
    """
    Returns the layer number from a torch.named_modules string
    """
    if "layers" not in module_name:
        return -1  # Projection or output norm layers
    else:
        pattern = r"\.\d+"
        matches = re.findall(pattern, module_name)
        assert (
            len(matches) == 1
        ), f"Ambiguous module name! Can not parse layer index. {module_name}"
        return int(matches[0][-1])


def free_model(
    model: nn.Module,
) -> None:
    """
    Frees the memory occupied by a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to be freed.

    Returns:
        None
    """
    del model
    torch.cuda.empty_cache()
    gc.collect()
