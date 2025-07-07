"""
Evaluate base model on the individual functions
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.eval.evaluator import eval_task
from comp_rep.utils import (
    ValidateTaskOptions,
    load_model,
    load_tokenizer,
    setup_logging,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data" / "function_tasks"
RESULT_DIR = CURR_FILE_DIR / "base_model_evaluations"


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser("Evaluation script")

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument(
        "--eval_tasks",
        nargs="+",
        default=POSSIBLE_TASKS,
        action=ValidateTaskOptions,
        help="Task(s) to evaluate model on.",
    )
    parser.add_argument("--save_path", type=Path, help="Path to the saved base models")
    parser.add_argument("--output_path", type=Path, help="Output path for evaluation.")

    return parser.parse_args()


def run_base_evaluation(
    save_path: Path,
    tasks: list[str],
) -> dict:
    """
    Run the base model evaluation on a list of tasks.

    Args:
        save_path (Path): The path to the directory where the base model is saved.
        tasks (list[str]): A list of task names to evaluate the model on.

    Returns:
        dict: A dictionary containing the evaluation results for each task. The keys are the task names, and the values are lists of accuracy values.

    """
    result = defaultdict(list)

    # load base model
    model_path = save_path / "base_model.ckpt"
    model = load_model(model_path=model_path, is_masked=False)
    tokenizer = load_tokenizer(save_path)

    # eval model
    for task_name in tasks:
        data_path = DATA_DIR / task_name / "test.csv"
        output_dir = RESULT_DIR / f"base_model_function_{task_name}"

        eval_dict = eval_task(
            task_name=task_name,
            model=model,
            tokenizer=tokenizer,
            device=DEVICE,
            eval_data_path=data_path,
            output_dir=output_dir,
            eval_acc=True,
            eval_faithfulness=False,
        )
        task_accuracy = eval_dict["accuracy"]
        result[task_name].append(task_accuracy)
    return result


def main() -> None:
    """
    Main function.
    """
    args = parse_args()

    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning function evaluation with the config: \n{config_string}")

    result = run_base_evaluation(
        args.save_path,
        args.eval_tasks,
    )
    logging.info(result)

    # save result
    result = dict(result)
    json_dict = json.dumps(result)
    output_path = RESULT_DIR / "base_model_results.json"
    with open(output_path, "w") as f:
        f.write(json_dict)


if __name__ == "__main__":
    main()
