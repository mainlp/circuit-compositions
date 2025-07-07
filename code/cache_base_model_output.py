"""
Cache base model output, i.e. probability distributions over vocabulary
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.utils import (
    ValidateWandbPath,
    load_model,
    load_tokenizer,
    set_seed,
    setup_logging,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data"
CACHE_DIR = DATA_DIR / "cached_probabilities"


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser("Utility script for caching probabilities")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=1860, help="Random seed.")
    parser.add_argument(
        "--base_model_name", type=str, default="pcfgs_base", help="Name of base model."
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Path to load trained model from.",
    )
    parser.add_argument(
        "--cache_dir",
        action=ValidateWandbPath,
        type=Path,
        help="Path to save cached probabilities at.",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="copy",
        choices=POSSIBLE_TASKS,
        help="Name of subtask on which model has been pruned on.",
    )
    return parser.parse_args()


@torch.no_grad()
def run_caching(model: nn.Module, loader: DataLoader, cache_save_path: Path) -> None:
    """
    Runs caching for a given model on a provided data loader, storing the resulting probabilities at a specified cache save path.

    Args:
        model (nn.Module): The model to run caching for.
        loader (DataLoader): The data loader to use for caching.
        cache_save_path (Path): The path to save the cached probabilities at.

    Returns:
        None
    """
    dataset_probabilities: List[Tensor] = []
    for (
        source_ids,
        source_mask,
        target_ids,
        target_mask,
        source_str,
        target_str,
    ) in tqdm(loader):
        source_ids = source_ids.to(DEVICE)
        source_mask = source_mask.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        target_mask = target_mask.to(DEVICE)
        logits = model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        ).squeeze()  # [seq_len, vocab_size]
        probas = nn.functional.softmax(logits, dim=-1)
        dataset_probabilities.append(probas)

    dataset_probabilities_tensor = torch.stack(dataset_probabilities, dim=0)
    dataset_probabilities_tensor = dataset_probabilities_tensor.detach().cpu()
    torch.save(dataset_probabilities_tensor, cache_save_path)
    logging.info("Finished writing cached probabilities to disk..")


def get_longest_item_of_dataset(dataset: SequenceDataset) -> int:
    """
    Retrieves the length of the longest item in the given dataset.

    Args:
        dataset (SequenceDataset): The dataset to search for the longest item.

    Returns:
        int: The length of the longest item in the dataset.
    """
    max_seen_length = 0
    for idx in range(0, len(dataset)):
        source_ids, target_ids, _, _ = dataset[idx]  # type: ignore
        local_max_length = max(len(source_ids), len(target_ids))
        max_seen_length = max(max_seen_length, local_max_length)
    return max_seen_length


def main() -> None:
    """
    Main script.
    """
    args = parse_args()

    set_seed(args.seed)
    setup_logging(args.verbose)
    logging.info("Utility script for caching probabilities")

    model_dir = args.model_dir / args.base_model_name
    model_file = "base_model.ckpt"
    model_path = model_dir / model_file
    model = load_model(model_path=model_path, is_masked=False)

    tokenizer = load_tokenizer(model_dir)

    data_dir = DATA_DIR if args.subtask == "base_tasks" else DATA_DIR / "function_tasks"
    for set_type in ["train", "test"]:
        logging.info(
            f"Caching probabilities for subtask {args.subtask} and split: {set_type}"
        )
        data_path = data_dir / args.subtask / f"{set_type}.csv"
        dataset = SequenceDataset(data_path, tokenizer=tokenizer)
        longest_sequence = get_longest_item_of_dataset(dataset)
        input_vocabulary_size = len(tokenizer["input_language"]["index2word"])
        output_vocabulary_size = len(tokenizer["output_language"]["index2word"])
        args.input_vocabulary_size = input_vocabulary_size
        args.output_vocabulary_size = output_vocabulary_size

        loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=CollateFunctor(max_length=longest_sequence),
            shuffle=False,
            num_workers=7,
            persistent_workers=True,
        )

        cache_dir = args.cache_dir / f"{args.subtask}"
        os.makedirs(cache_dir, exist_ok=True)
        cache_save_path = cache_dir / f"{args.subtask}_{set_type}.pt"

        if set_type == "train":
            model.train()
        else:
            model.eval()
        run_caching(model, loader, cache_save_path)


if __name__ == "__main__":
    main()
