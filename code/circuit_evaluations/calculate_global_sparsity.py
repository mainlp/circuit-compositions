import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from comp_rep.constants import MASK_TASKS
from comp_rep.utils import ValidateTaskOptions, load_model, setup_logging

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
RESULT_DIR = CURR_FILE_DIR / "figures"


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


def bar_plot(labels: list, values: list, path: Path) -> None:
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(7, 5))

    sorted_pairs = sorted(zip(values, labels))

    # Unzip the sorted pairs back into separate lists
    values, labels = zip(*sorted_pairs)
    labels = [label.replace("swap_first_last", "swap") for label in labels]
    labels = [label.replace("remove_second", "rm_second") for label in labels]
    labels = [label.replace("remove_first", "rm_first") for label in labels]

    # Plotting the bars with a color gradient
    bars = ax.bar(labels, values, color=plt.cm.Paired(range(len(labels))))

    # Setting y-axis limits from 0 to 100%
    ax.set_ylim(0, 100)

    # Adding labels to the axes
    ax.set_ylabel("Remaining act. (%)", fontsize=12)

    # Adding gridlines for better readability
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right", fontsize=18)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height: .0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # Offset label slightly above the bar
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # Show the plot
    plt.tight_layout()
    plt.savefig(path, format="pdf")
    # plt.show()


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
        model_path = (
            args.save_path
            / f"{task}"
            / f"{args.pruning_type}_continuous_{args.ablation_value}_pruned_model.ckpt"
        )
        model = load_model(
            model_path=model_path, is_masked=True, model=None, return_pl=True
        )
        global_remaining_mask = model.pruner.get_remaining_mask()[
            "global_remaining_mask"
        ]
        sparsity[f"{task}"] = global_remaining_mask * 100
    labels = list(sparsity.keys())
    values = list(sparsity.values())
    os.makedirs(RESULT_DIR, exist_ok=True)
    save_path = (
        RESULT_DIR / f"{args.pruning_type}_{args.ablation_value}_global_sparsity.pdf"
    )
    bar_plot(labels, values, save_path)


if __name__ == "__main__":
    main()
