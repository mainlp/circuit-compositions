import argparse
import json
import logging
import os
from pathlib import Path

import torch

from comp_rep.constants import MASK_TASKS
from comp_rep.utils import ValidateTaskOptions, load_model, setup_logging

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
RESULT_DIR = CURR_FILE_DIR / "sparsity"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Evaluation script")

    # General Configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )

    # Eval Configs
    parser.add_argument(
        "--circuit_names",
        nargs="+",
        default=MASK_TASKS,
        action=ValidateTaskOptions,
        help="Task(s) to evaluate model on.",
    )

    # Mask Configs
    parser.add_argument("--save_path", type=Path, help="Path to the saved models.")
    parser.add_argument(
        "--pruning_type",
        type=str,
        choices=["weights", "activations"],
        default="activations",
        help="Pruning type (either 'weights' or 'activations').",
    )
    parser.add_argument(
        "--ablation_value",
        type=str,
        choices=["zero", "mean"],
        default="zero",
        help="Which value to ablate with",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function.
    """
    args = parse_args()

    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(
        f"\nRunning calculation of global sparsity with the config: \n{config_string}"
    )

    sparsity = {}
    for task in args.circuit_names:
        logging.info(f"Getting local sparsity for subtask: {task}")
        model_path = (
            args.save_path
            / f"{task}"
            / f"{args.pruning_type}_continuous_{args.ablation_value}_pruned_model.ckpt"
        )
        model = load_model(
            model_path=model_path, is_masked=True, model=None, return_pl=True
        )
        finegrained_activations = model.pruner.get_remaining_mask()[
            "pruning_finegrained/"
        ]
        sparsity[f"{task}"] = finegrained_activations
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = (
        RESULT_DIR / f"{args.pruning_type}_{args.ablation_value}_local_sparsity.json"
    )
    with open(save_path, "w") as file:
        json.dump(sparsity, file, indent=4)


if __name__ == "__main__":
    main()
