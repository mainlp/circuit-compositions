"""
Evaluating the node overlap of various circuits
"""

import argparse
import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch

from comp_rep.constants import MASK_TASKS
from comp_rep.pruning.subnetwork_mask_metrics import (
    iom_by_layer_and_module,
    iou_by_layer_and_module,
)
from comp_rep.utils import ValidateTaskOptions, free_model, load_model, setup_logging

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

    # Eval Configs
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
    parser.add_argument("--model_path", type=Path, help="Path to the saved models.")
    parser.add_argument(
        "--circuit_names",
        nargs="+",
        default=MASK_TASKS,
        action=ValidateTaskOptions,
        help="Name of subtask on which model has been pruned on.",
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


def run_circuit_overlap_evaluation(
    model_dir: Path,
    pruning_type: Literal["weights", "activations"],
    pruning_method: Literal["sampled", "continuous"],
    ablation_value: Literal["zero", "mean"],
    circuit_pairs: List[Tuple[str, str]],
    architecture_blocks: List[Literal["encoder", "decoder", "projection"]],
    layer_idx: List[int],
    fraction: bool = True,
    eval_iou: bool = True,
    eval_iom: bool = True,
) -> dict:
    """
    Evaluates the overlap between pairs of circuits based on their pruning configurations.

    Args:
        model_path (Path): The path to the directory containing the saved models.
        pruning_type (Literal["weights", "activations"]): The type of pruning applied to the models.
        pruning_method (Literal["sampled", "continuous"]): The method used for pruning.
        ablation_value (Literal["zero", "mean"]): The value used for ablation.
        circuit_pairs (List[Tuple[str, str]]): A list of pairs of circuit names to evaluate.
        architecture_blocks (List[Literal["encoder", "decoder"]]): A list of architecture blocks to consider.
        layer_idx (List[int]): A list of layer indices to evaluate.
        fraction (bool, optional): Whether to calculate the fraction of overlap. Defaults to True.
        eval_iou (bool, optional): Whether to evaluate the IoU metric. Defaults to True.
        eval_iom (bool, optional): Whether to evaluate the IoM metric. Defaults to True.

    Returns:
        dict: A dictionary containing the overlap metrics for each pair of circuits.
    """
    result: Dict[str, Dict[str, float]] = defaultdict()

    for circuit_one, circuit_two in circuit_pairs:
        logging.info(
            f"Evaluating circuit overlap between: {circuit_one} and {circuit_two}"
        )
        result.setdefault(f"{circuit_one}-{circuit_two}", {})

        # load circuit one
        model_one_directory = model_dir / circuit_one
        model_name = (
            f"{pruning_type}_{pruning_method}_{ablation_value}_pruned_model.ckpt"
        )
        model_one_path = model_one_directory / model_name
        model_one = load_model(model_path=model_one_path, is_masked=True)

        # load circuit two
        model_two_directory = model_dir / circuit_two
        model_two_path = model_two_directory / model_name
        model_two = load_model(model_path=model_two_path, is_masked=True)

        if eval_iou:
            iou = iou_by_layer_and_module(
                model_list=[model_one, model_two],
                architecture_blocks=architecture_blocks,
                layer_idx=layer_idx,
                fraction=fraction,
                average=False,
            )
            result[f"{circuit_one}-{circuit_two}"]["IoU"] = iou

        if eval_iom:
            iom = iom_by_layer_and_module(
                model_list=[model_one, model_two],
                architecture_blocks=architecture_blocks,
                layer_idx=layer_idx,
                fraction=fraction,
                average=True,
            )
            result[f"{circuit_one}-{circuit_two}"]["IoM"] = iom

        free_model(model_one)
        free_model(model_two)
    return result


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

    circuits = args.circuit_names
    circuit_pairs = list(itertools.combinations(circuits, 2))

    result = run_circuit_overlap_evaluation(
        model_dir=args.model_path,
        pruning_type=args.pruning_type,
        pruning_method=args.pruning_method,
        ablation_value=args.ablation_value,
        circuit_pairs=circuit_pairs,
        architecture_blocks=args.architecture_blocks,
        layer_idx=args.layer_idx,
        fraction=True,
        eval_iou=True,
        eval_iom=True,
    )
    logging.info(result)

    # save result
    result = dict(result)
    json_dict = json.dumps(result)
    output_path = (
        args.result_dir
        / f"{args.pruning_type}_{args.pruning_method}_{args.ablation_value}_circuit_overlap_evaluation_results.json"
    )
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
