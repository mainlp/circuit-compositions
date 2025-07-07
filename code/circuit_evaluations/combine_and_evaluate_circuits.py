import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch

from comp_rep.constants import MASK_TASKS
from comp_rep.eval.evaluator import eval_task
from comp_rep.pruning.subnetwork_set_operations import (
    difference_model,
    intersection_model,
    union_model,
)
from comp_rep.utils import (
    ValidateTaskOptions,
    load_model,
    load_tokenizer,
    setup_logging,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
RESULT_DIR = CURR_FILE_DIR / "figures"
DATA_DIR = CURR_FILE_PATH.parents[2] / "data/function_tasks"


ARGS_TO_OPERATION = {
    "union": union_model,
    "difference": difference_model,
    "intersection": intersection_model,
}


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
        "--operation",
        type=str,
        default="union",
        choices=["union", "intersection", "difference"],
        help="Which set operation to apply",
    )

    # Eval Configs
    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        default=MASK_TASKS,
        action=ValidateTaskOptions,
        help="Task(s) to evaluate model on.",
    )

    parser.add_argument(
        "--circuit_names",
        nargs="+",
        default=MASK_TASKS,
        action=ValidateTaskOptions,
        help="Circuits to load",
    )

    # Mask Configs
    parser.add_argument("--model_dir", type=Path, help="Path to the saved models.")
    parser.add_argument(
        "--cache_dir", type=Path, help="Path to the cached probabilities"
    )
    parser.add_argument("--result_dir", type=Path, help="Where to save the results")
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
    logging.info("Test circuit compositions")
    tokenizer_path = (
        args.model_dir / args.circuit_names[0]
    )  # All the circuits for the same base model have the same tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    assert (
        len(args.circuit_names) >= 2
    ), "Need more than one circuit in order to do set operations!"

    model_list = {}
    for circuit in args.circuit_names:
        model_path = (
            args.model_dir
            / circuit
            / f"{args.pruning_type}_continuous_{args.ablation_value}_pruned_model.ckpt"
        )
        model = load_model(
            model_path=model_path, is_masked=True, model=None, return_pl=False
        )
        model_list[circuit] = model
    combination_name = "-".join(list(model_list.keys()))

    new_model = None
    for circuit, model in model_list.items():
        if not new_model:
            new_model = model
        logging.info(f"Applying {args.operation} to {circuit}")
        new_model = ARGS_TO_OPERATION[args.operation](new_model, model)
    new_model_name = "-".join(args.circuit_names)
    model_list[new_model_name] = new_model

    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for circuit, model in model_list.items():
        logging.info(f"Loaded circuit {circuit}...")
        result.setdefault(circuit, {})
        for task_name in args.eval_tasks:
            data_path = DATA_DIR / task_name / "test.csv"
            eval_dict = eval_task(
                task_name=task_name,
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                output_dir=None,
                eval_data_path=data_path,
                cached_probabilities_path=args.cache_dir
                / task_name
                / f"{task_name}_test.pt",
                eval_acc=True,
            )
            result[circuit][task_name] = eval_dict
            logging.info(eval_dict)

    result = dict(result)
    json_dict = json.dumps(result)
    output_path = (
        args.result_dir
        / f"{combination_name}_{args.ablation_value}_set_{args.operation}_results.json"
    )
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
