"""
Evaluating the seed overlap of various circuits
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from comp_rep.constants import MASK_TASKS
from comp_rep.pruning.subnetwork_mask_metrics import (
    iom_by_layer_and_module,
    iou_by_layer_and_module,
)
from comp_rep.utils import ValidateTaskOptions, load_model, setup_logging

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data/function_tasks"


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
    parser.add_argument(
        "--circuit_name",
        choices=MASK_TASKS,
        default="copy",
        help="Task(s) to evaluate model on.",
    )

    # Eval Configs
    parser.add_argument("--seed_folder", type=Path, help="Path to the saved models.")
    parser.add_argument(
        "--result_dir", type=Path, help="Path to where the results are saved."
    )
    parser.add_argument(
        "--architecture_blocks",
        nargs="+",
        type=str,
        default=["encoder", "decoder"],
        help="Name of architecture blocks to consider.",
    )
    parser.add_argument(
        "--layer_idx",
        nargs="+",
        default=[0, 1, 2, 3, 4, 5],
        action=ValidateTaskOptions,
        help="Layers to consider.",
    )

    # Mask Configs
    parser.add_argument(
        "--seeds",
        nargs="+",
        help="Which seeds to load",
    )
    parser.add_argument(
        "--pruning_type",
        type=str,
        choices=["weights", "activations"],
        default="activations",
        help="Pruning type (either 'weights' or 'activations').",
    )
    parser.add_argument(
        "--pruning_method", type=str, default="continuous", help="Pruning method."
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
        f"\nRunning circuit overlap evaluation with the config: \n{config_string}"
    )

    seeds = args.seeds
    model_name = f"{args.pruning_type}_{args.pruning_method}_{args.ablation_value}_pruned_model.ckpt"

    model_list = []
    for seed in seeds:
        m_path = args.seed_folder / seed / args.circuit_name / model_name
        m = load_model(model_path=m_path, is_masked=True)
        model_list.append(m)

    iou = iou_by_layer_and_module(
        model_list=model_list,
        architecture_blocks=args.architecture_blocks,
        layer_idx=args.layer_idx,
        average=False,
    )
    iom = iom_by_layer_and_module(
        model_list=model_list,
        architecture_blocks=args.architecture_blocks,
        layer_idx=args.layer_idx,
        average=True,
    )
    result = {
        "iom": iom,
        "iou": iou,
    }

    logging.info(result)

    # save result
    result = dict(result)
    json_dict = json.dumps(result)
    output_path = (
        args.result_dir / f"{args.circuit_name}_{args.ablation_value}_seed_overlap.json"
    )
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
