"""
Evaluate masked models on the individual functions
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Literal

import torch

from comp_rep.constants import MASK_TASKS
from comp_rep.eval.evaluator import eval_task
from comp_rep.utils import (
    ValidateTaskOptions,
    ValidateWandbPath,
    free_model,
    load_json,
    load_model,
    load_tokenizer,
    setup_logging,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[2] / "data/function_tasks"
MEAN_ABLATION_VALUES_PATH = (
    CURR_FILE_PATH.parents[1] / "comp_rep" / "pruning" / "mean_ablation_values"
)


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
        "--eval_tasks",
        nargs="+",
        default=MASK_TASKS,
        action=ValidateTaskOptions,
        help="Task(s) to evaluate model on.",
    )
    parser.add_argument(
        "--cache_dir",
        action=ValidateWandbPath,
        type=Path,
        help="Path to cached probabilities.",
    )
    parser.add_argument(
        "--result_dir", type=Path, help="Path to where the results are saved."
    )
    parser.add_argument(
        "--mask_func_equivalence",
        action="store_true",
        help="Mask function equivalence.",
    )

    # Mask Configs
    parser.add_argument("--model_path", type=Path, help="Path to the saved models.")
    parser.add_argument(
        "--mean_mode",
        choices=["local", "global"],
        default="local",
        help="Activate global mean mode",
    )
    parser.add_argument(
        "--circuit_names",
        nargs="+",
        default=["append"],
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
        default="mean",
        help="Which value to ablate with",
    )
    return parser.parse_args()


def load_and_average_ablation_data(task_list: list):
    ablation_values = []
    for task in task_list:
        print(f"Loading {task}'s ablation values")
        ablation_value_path = (
            MEAN_ABLATION_VALUES_PATH / f"{task}_mean_ablation_values.json"
        )

        ablation_data = load_json(ablation_value_path)
        ablation_values.append(ablation_data)

    mean_dict: Dict[str, list] = {
        key: [] for key in ablation_values[0].keys()
    }  # equal keys
    for task in ablation_values:
        for key, item in task.items():
            mean_dict[key].append(torch.tensor(item))

    for key, item in mean_dict.items():
        item = torch.stack(item, dim=0)
        item = torch.mean(item, dim=0).tolist()
        mean_dict[key] = item

    return mean_dict


def run_circuit_performance_evaluation(
    model_dir: Path,
    cache_dir: Path,
    result_dir: Path,
    pruning_type: Literal["weights", "activations"],
    pruning_method: Literal["sampled", "continuous"],
    ablation_value: Literal["zero", "mean"],
    circuit_names: List[str],
    tasks: List[str],
    eval_acc: bool = True,
    eval_faithfulness: bool = False,
    mask_func_equivalence: bool = False,
    mean_mode: str = "local",
) -> dict:
    """
    Evaluates masked models on the individual functions.

    Args:
        model_dir (Path): The directory of the saved models.
        cache_dir (Path): The directory of the store cached probabilities.
        result_dir (Path): The directory to store evaluation results.
        pruning_type (Literal["weights", "activations"]): The pruning type.
        pruning_method (Literal["sampled", "continuous"]): The pruning method.
        ablation_value (Literal["zero", "mean"]): The value to ablate with.
        circuit_names (List[str]): A list of circuit names to evaluate.
        tasks (List[str]): A list of tasks to evaluate the model on.
        eval_acc (bool, optional): Whether to evaluate accuracy. Defaults to True.
        eval_faithfulness (bool, optional): Whether to evaluate faithfulness. Defaults to False.
        eval_faithfulness (bool, optional): Whether to mask function equivalent tokens. Defaults to False.

    Returns:
        dict: A dictionary containing the evaluation results for each task and circuit.
    """
    result: Dict[str, Dict[str, Dict[str, float]]] = {}

    for mask_name in circuit_names:
        logging.info(f"Evaluating circuit: {mask_name}")
        result.setdefault(mask_name, {})

        # load masked model
        model_directory = model_dir / mask_name
        model_name = (
            f"{pruning_type}_{pruning_method}_{ablation_value}_pruned_model.ckpt"
        )
        model_path = model_directory / model_name

        pl_pruner = load_model(model_path=model_path, is_masked=True, return_pl=True)
        tokenizer = load_tokenizer(model_directory)

        # eval model
        for task_name in tasks:
            data_path = DATA_DIR / task_name / "test.csv"
            output_dir = (
                result_dir / mask_name / f"circuit_{mask_name}_function_{task_name}"
            )

            model = pl_pruner.model

            eval_dict = eval_task(
                task_name=task_name,
                model=model,
                tokenizer=tokenizer,
                device=DEVICE,
                output_dir=output_dir,
                eval_data_path=data_path,
                cached_probabilities_path=cache_dir
                / task_name
                / f"{task_name}_test.pt",
                eval_acc=eval_acc,
                eval_faithfulness=eval_faithfulness,
                mask_func_equivalence=mask_func_equivalence,
                circuit_name=mask_name,
                eval_task_name=task_name,
            )
            result[mask_name][task_name] = eval_dict

        free_model(model)
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
        f"\nRunning circuit performance evaluation with the config: \n{config_string}"
    )

    result = run_circuit_performance_evaluation(
        model_dir=args.model_path,
        cache_dir=args.cache_dir,
        result_dir=args.result_dir,
        pruning_type=args.pruning_type,
        pruning_method=args.pruning_method,
        ablation_value=args.ablation_value,
        circuit_names=args.circuit_names,
        tasks=args.eval_tasks,
        eval_acc=True,
        eval_faithfulness=True,
        mask_func_equivalence=args.mask_func_equivalence,
        mean_mode=args.mean_mode,
    )
    logging.info(result)

    if args.mask_func_equivalence:
        func_equivalence_str = "func_equivalence_mask_"
    else:
        func_equivalence_str = ""

    # save result
    result = dict(result)
    json_dict = json.dumps(result)
    output_path = (
        args.result_dir
        / f"{args.pruning_type}_{args.pruning_method}_{args.ablation_value}_circuit_{func_equivalence_str}performance_evaluation_results.json"
    )
    os.makedirs(args.result_dir, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
