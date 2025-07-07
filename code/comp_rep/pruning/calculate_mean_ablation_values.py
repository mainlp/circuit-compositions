import argparse
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from nnsight import NNsight
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from comp_rep.constants import MASK_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.utils import load_model, load_tokenizer, set_seed, setup_logging

CURR_FILE_PATH = Path(__file__).resolve()
DATA_DIR = CURR_FILE_PATH.parents[3] / "data" / "function_tasks"
SAVE_DIR = CURR_FILE_PATH.parents[0] / "mean_ablation_values"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        "Utility script for calculating mean ablation values"
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument(
        "--subtask",
        type=str,
        default="append",
        choices=MASK_TASKS,
        help="Name of subtask on which model has been pruned on.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to the model you wish to get the mean ablation values from",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1000,
        help="The number of samples in the distribution you want to mean ablate from",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument(
        "--run_all", action="store_true", help="Run all the functional masks"
    )
    return parser.parse_args()


def get_mean_activations(model: NNsight, data_loader: DataLoader, n: int) -> dict:
    """
    Calculate the mean activations of a given model's layers.

    Args:
        model (NNsight): The model for which to calculate the mean activations.
        data_loader (DataLoader): The data loader used to load the data for the model.
        n (int): The number of samples to use for the calculation.

    Returns:
        dict: A dictionary where the keys are the names of the model's layers and the values are the mean activations of those layers.
    """
    distribution: Dict[str, List[Tensor]] = defaultdict(list)
    sampled: int = 0
    for source_ids, source_mask, target_ids, target_mask, src_str, target_str in tqdm(
        data_loader
    ):
        source_ids = source_ids.to(DEVICE)
        source_mask = source_mask.to(DEVICE)
        target_ids = target_ids.to(DEVICE)
        target_mask = target_mask.to(DEVICE)
        for i in range(len(model.encoder.layers)):
            # The execution of the hook is only run upon exiting the context of the with statement... Hence, We have to repeat the tracer per call.
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                e_ff_output = model.encoder.layers[i].feed_forward.output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                e_sa_output = model.encoder.layers[i].self_attention.nns_output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                d_ff_output = model.decoder.layers[i].feed_forward.output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                d_ca_output = model.decoder.layers[i].cross_attention.nns_output.save()
            with model.trace(
                source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
            ) as _:
                d_sa_output = model.decoder.layers[i].self_attention.nns_output.save()

            distribution[f"model.encoder.layers.{i}.feed_forward"].append(
                torch.mean(e_ff_output, dim=1).detach().cpu().clone()
            )
            distribution[f"model.encoder.layers.{i}.self_attention"].append(
                torch.mean(e_sa_output, dim=1).detach().cpu().clone()
            )
            distribution[f"model.decoder.layers.{i}.self_attention"].append(
                torch.mean(d_sa_output, dim=1).detach().cpu().clone()
            )
            distribution[f"model.decoder.layers.{i}.cross_attention"].append(
                torch.mean(d_ca_output, dim=1).detach().cpu().clone()
            )
            distribution[f"model.decoder.layers.{i}.feed_forward"].append(
                torch.mean(d_ff_output, dim=1).detach().cpu().clone()
            )

        if data_loader.batch_size is not None:
            sampled += data_loader.batch_size
        else:
            logging.warning(
                f"Data loader's batch_size is: {data_loader.batch_size}. Sampled will remain 0."
            )

        if sampled > n:
            break

    for module, values in distribution.items():
        stacked_activations = torch.cat(values, dim=0)
        new_v = torch.mean(stacked_activations, dim=0)
        distribution.update({module: new_v.tolist()})

    return distribution


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    setup_logging(args.verbose)

    if args.run_all:
        tasks_to_run = MASK_TASKS
    else:
        tasks_to_run = [args.subtask]

    model_path = args.model_path / "base_model.ckpt"
    model = load_model(model_path, False)
    model = model.to(DEVICE)
    model.eval()
    model = NNsight(model)
    tokenizer = load_tokenizer(args.model_path)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    for task in tasks_to_run:
        logging.info(f"Calculating mean activation values for task: {task}")
        train_dataset = SequenceDataset(
            DATA_DIR / task / "train.csv", tokenizer=tokenizer
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            collate_fn=CollateFunctor(),
            shuffle=True,
            num_workers=7,
            persistent_workers=True,
        )

        distribution = get_mean_activations(model, train_loader, args.n)

        with open(SAVE_DIR / f"{task}_mean_ablation_values.json", "w") as f:
            json.dump(distribution, f, indent=4)


if __name__ == "__main__":
    main()
