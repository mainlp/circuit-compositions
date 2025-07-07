"""
Evaluate trained models and subnetworks.
"""

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.utils import ValidateSavePath, load_model, load_tokenizer, setup_logging

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data"
RESULT_DIR = CURR_FILE_DIR / "predictions"


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

    # Model Configs
    parser.add_argument(
        "--save_path",
        action=ValidateSavePath,
        type=Path,
        help="Path to load trained model from.",
    )
    parser.add_argument(
        "--is_masked", action="store_true", help="Whether the model is pruned."
    )

    # Base Model Configs
    parser.add_argument(
        "--base_model_name", type=str, default="pcfgs_base", help="Name of base model."
    )

    # Pruning Configs
    parser.add_argument(
        "--pruning_task",
        type=str,
        default="append",
        choices=POSSIBLE_TASKS,
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
        "--pruning_method",
        type=str,
        choices=["continuous", "sampled"],
        default="continuous",
        help="Pruning method.",
    )
    parser.add_argument(
        "--ablation_value",
        type=str,
        choices=["zero", "mean"],
        default="zero",
        help="Which value to ablate with",
    )

    # Eval Configs
    parser.add_argument(
        "--eval_task",
        type=str,
        default="append",
        choices=POSSIBLE_TASKS,
        help="Task to evaluate model on.",
    )
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size.")

    return parser.parse_args()


def load_eval_data(path: Path, tokenizer: dict) -> SequenceDataset:
    """
    Load evaluation data from the given path using the provided tokenizer.

    Args:
        path (Path): The path to the evaluation data.
        tokenizer (dict): The tokenizer used to preprocess the data.

    Returns:
        SequenceDataset: The loaded evaluation dataset.
    """
    dataset = SequenceDataset(path, tokenizer)
    return dataset


def main() -> None:
    """
    Main function.
    """
    args = parse_args()

    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning evaluation with the config: \n{config_string}")

    # load model
    if args.is_masked:
        model_dir = args.save_path / args.pruning_task
        model_file = f"{args.pruning_type}_{args.pruning_method}_{args.ablation_value}_pruned_model.ckpt"
        model_path = model_dir / model_file
        prediction_path = (
            RESULT_DIR
            / args.pruning_task
            / f"{args.pruning_type}_{args.pruning_method}"
            / {args.ablation_value}
        )
    else:
        model_dir = args.save_path / args.base_model_name
        model_file = "base_model.ckpt"
        model_path = model_dir / "base_model.ckpt"
        prediction_path = RESULT_DIR / args.base_model_name

    model = load_model(model_path=model_path, is_masked=args.is_masked)
    tokenizer = load_tokenizer(model_dir)

    # load data
    data_dir = (
        DATA_DIR if args.eval_task == "base_tasks" else DATA_DIR / "function_tasks"
    )
    data_path = data_dir / args.eval_task / "test.csv"
    eval_dataset = SequenceDataset(data_path, tokenizer=tokenizer)
    input_vocabulary_size = len(tokenizer["input_language"]["index2word"])
    output_vocabulary_size = len(tokenizer["output_language"]["index2word"])
    args.input_vocabulary_size = input_vocabulary_size
    args.output_vocabulary_size = output_vocabulary_size

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )

    # evaluate
    searcher = GreedySearch(model, eval_dataset.output_language)
    accuracy = evaluate_generation(
        model, searcher, eval_loader, prediction_path, DEVICE
    )

    logging.info(f"Final accuracy was: {accuracy}")


if __name__ == "__main__":
    main()
