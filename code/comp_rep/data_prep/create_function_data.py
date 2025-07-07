"""
Generate subtask data for PCFG set.
"""

import argparse
from pathlib import Path

from comp_rep.constants import POSSIBLE_TASKS
from comp_rep.data_prep.generate import MarkovTree, generate_data
from comp_rep.data_prep.operations import ALPHABET, BINARY_FUNC, FUNC_MAP, UNARY_FUNC
from comp_rep.utils import set_seed

CURR_FILE_PATH = Path(__file__).resolve()
DATA_DIR = CURR_FILE_PATH.parents[3] / "data" / "function_tasks"


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1860, help="Random seed.")
    parser.add_argument(
        "--subtask",
        type=str,
        default="append",
        choices=POSSIBLE_TASKS,
        help="Name of subtask on which model has been pruned on.",
    )
    parser.add_argument(
        "--nr_samples", type=int, default=100, help="Number of samples to generate"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Ratio of train split."
    )
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # These are the default settings from Hupkes et al (2020).
    alphabet_ratio = 1
    prob_func = 0.25
    lengths = [2, 3, 4, 5]
    placeholders = False
    omit_brackets = True
    random_probs = False
    if args.subtask in UNARY_FUNC:
        prob_unary = 1.0
        unary_functions = [FUNC_MAP[args.subtask]]
        binary_functions = []
    elif args.subtask in BINARY_FUNC:
        prob_unary = 0.0
        binary_functions = [FUNC_MAP[args.subtask]]
        unary_functions = []
    elif args.subtask == "base_tasks":
        prob_unary = 0.75
        unary_functions = [FUNC_MAP[i] for i in UNARY_FUNC]
        binary_functions = [FUNC_MAP[i] for i in BINARY_FUNC]
    else:
        raise ValueError("Invalid subtask provided as argument")

    alphabet = [
        letter + str(i) for letter in ALPHABET for i in range(1, alphabet_ratio + 1)
    ]
    pcfg_tree_generator = MarkovTree(
        unary_functions=unary_functions,
        binary_functions=binary_functions,
        alphabet=alphabet,
        prob_unary=prob_unary,
        prob_func=prob_func,
        lengths=lengths,
        placeholders=placeholders,
        omit_brackets=omit_brackets,
    )

    generate_data(
        pcfg_tree=pcfg_tree_generator,
        total_samples=args.nr_samples,
        file_dir=DATA_DIR / args.subtask,
        random_probs=random_probs,
        train_ratio=args.train_ratio,
    )


if __name__ == "__main__":
    main()
