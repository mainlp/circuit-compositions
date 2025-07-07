"""
Model pruning script.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracr_data_utils import (
    DataCollatorTracr,
    ErazrTokenizer,
    load_datasets,
)
from tracr_model_utils import get_config_weights_and_vocab

import wandb
from comp_rep.constants import POSSIBLE_TASKS

from comp_rep.loss import get_regularized_loss_from_nano
from comp_rep.models.nanoGPT import GPT
from comp_rep.pruning.activation_pruning.activation_pruner import ActivationPruner
from comp_rep.pruning.activation_pruning.masked_activation_base import (
    ContinuousMaskedActivationLayer,
)
from comp_rep.utils import ValidateSavePath, ValidateWandbPath, set_seed, setup_logging

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data"
CACHE_DIR = DATA_DIR / "cached_probabilities"
SWEEP_DIR = CURR_FILE_DIR / "sweeps"
RESULT_DIR = CURR_FILE_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser("Model Pruning for Subnetwork Identification.")

    # General configs
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbose mode (0: WARNING, 1: INFO, 2: DEBUG)",
    )
    parser.add_argument("--seed", type=int, default=1860, help="Random seed.")
    parser.add_argument(
        "--acc_freq",
        type=int,
        default=100,
        help="Frequency of epochs with which generation accuracy is evaluated.",
    )
    parser.add_argument(
        "--faithfulness_freq",
        type=int,
        default=20,
        help="Frequency of epochs with which task faithfulness is evaluated.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to evaluate the model in addition to training.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Whether to perform a hyperparameter sweep.",
    )

    # Path configs
    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to the Tracr compiled model.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        help="Path to the Tracr data.",
    )
    parser.add_argument(
        "--save_path",
        action=ValidateSavePath,
        type=Path,
        help="Path to save trained model at.",
    )
    parser.add_argument(
        "--wandb_path",
        action=ValidateWandbPath,
        type=Path,
        help="Path to save wandb metadata.",
    )
    parser.add_argument(
        "--cache_dir",
        action=ValidateWandbPath,
        type=Path,
        help="Path to cached probabilities.",
    )

    # Pruning configs
    parser.add_argument(
        "--subtask",
        type=str,
        default="copy",
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
    parser.add_argument("--num_masks", type=int, default=4)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--mask_initial_value", type=float, default=0.2)
    parser.add_argument(
        "--mask_lambda",
        type=float,
        default=1e-5,
        help="Lambda hyperparameter for continuous pruning.",
    )
    parser.add_argument(
        "--max_temp",
        type=int,
        default=100,
        help="Maximum temperature for continuous pruning.",
    )

    # Train parameter configs
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument(
        "--seq_len", type=int, default=5, help="Maximum sequence length."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Validation batch size."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--eta_min", type=float, default=0.0, help="Minimum learning rate."
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.0,
        help="Value for gradient clipping. If 0 no gradient clipping is applied.",
    )
    parser.add_argument(
        "--gradient_clip_alg",
        type=str,
        choices=["norm", "value"],
        default="norm",
        help="Algorithm for gradient clipping.",
    )

    return parser.parse_args()


def get_pruner_kwargs(args: argparse.Namespace) -> Dict:
    """
    Returns a dictionary of keyword arguments for a pruner based on the provided command-line arguments.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        Dict: Keyword arguments for a pruner.
    """
    pruning_methods_kwargs: dict[str, Any] = {}

    if args.pruning_method == "continuous":
        pruning_methods_kwargs["temperature_increase"] = args.max_temp ** (
            1.0 / args.epochs
        )
        pruning_methods_kwargs["mask_initial_value"] = args.mask_initial_value
    elif args.pruning_method == "sampled":
        pruning_methods_kwargs["tau"] = args.tau
        pruning_methods_kwargs["num_masks"] = args.num_masks
    else:
        raise ValueError("Invalid pruning strategy method provided")

    pruner_kwargs = {
        "pruning_method": args.pruning_method,
        "ablation_value": args.ablation_value,
        "subtask": args.subtask,
        "maskedlayer_kwargs": pruning_methods_kwargs,
    }

    if args.pruning_type == "activations":
        pruner_kwargs.update(
            {
                "ablation_value": args.ablation_value,
                "subtask": args.subtask,
            }
        )

    return pruner_kwargs


def print_remaining_compontents(pruner: ActivationPruner):
    """
    Iterates over the model components. For evert ContinuousMaskedActivationLayer, checks
    if the s_matrix of that layer contains any values that are not zero.
    It the matrix contains any non-zero values, it stores that component in a dictionary with the layer name as key
    """
    remaining_mask_elements = {}
    for name, module in pruner.model.transformer.named_modules():
        if isinstance(module, ContinuousMaskedActivationLayer):
            if torch.any(module.b_matrix > 0):
                # remaining_mask_elements[name] = torch.sum(module.s_matrix != 0).item()
                remaining_mask_elements[name] = module.b_matrix
    return remaining_mask_elements


def main() -> None:
    """
    Main script.
    """
    args = parse_args()

    set_seed(args.seed)
    setup_logging(args.verbose)

    # sweep config
    config = vars(args)
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(
        f"\nRunning pruning training loop for the Tracr models with the config: \n{config_string}"
    )

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")

    wandb_logger = wandb.init(
        entity="your-account-here",
        project="circomp-mask-training-tracr",
        name=f"{args.ablation_value}_{args.pruning_type}_{args.pruning_method}_{args.subtask}_{formatted_datetime}",
        config=config,
    )

    # load data
    config, tok_embed, pos_embed, blocks_embeds, vocab, bos, pad, unembedding_mtx = (
        get_config_weights_and_vocab(args.model_path)
    )

    base_model = GPT.from_tracr(
        config, tok_embed, pos_embed, unembedding_mtx, blocks_embeds
    )
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    mask_model = GPT.from_tracr(
        config, tok_embed, pos_embed, unembedding_mtx, blocks_embeds
    )
    mask_model.train()

    raw_datasets = load_datasets(args.data_path, 10000, 100)
    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]
    tokenizer = ErazrTokenizer(vocab, bos, pad)
    collator = DataCollatorTracr(tokenizer=tokenizer, length=args.seq_len)

    temp_increase = args.max_temp ** (1.0 / args.epochs)

    pruner = ActivationPruner(
        model=mask_model,
        pruning_method="continuous",
        subtask=args.subtask,
        ablation_value="zero",
        maskedlayer_kwargs={
            "mask_initial_value": args.mask_initial_value,
            "ticket": False,
            "temperature_increase": temp_increase,
        },
    )
    print(pruner.model)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(mask_model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs * len(train_dataloader),  # total number of iterations
        eta_min=args.eta_min,
    )

    for epoch in tqdm(range(args.epochs)):
        pruner.deactivate_ticket()
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"]

            pruner.compute_and_update_masks()
            logits, loss = pruner.model(input_ids)
            base_logits, base_loss = base_model(input_ids)
            ce_loss, norm_loss, regularized_loss = get_regularized_loss_from_nano(
                pruner, logits, base_logits, args.mask_lambda
            )

            wandb_logger.log(
                {
                    "train_loss": regularized_loss,
                    "cross_entropy_loss": ce_loss,
                    "l1_norm_loss": norm_loss,
                }
            )
            regularized_loss.backward()
            optimizer.step()
            lr_scheduler.step()

        pruner.update_hyperparameters()

        pruner.activate_ticket()
        remaining_mask_elements = pruner.get_remaining_mask()
        wandb_logger.log(remaining_mask_elements)
        if args.eval:
            with torch.no_grad():
                mask_model.eval()
                correct = 0
                for batch in val_dataloader:
                    input_ids = batch["input_ids"]
                    targets = batch["labels"]
                    logits, _ = pruner.model(input_ids)
                    preds = logits.argmax(-1)
                    c = torch.all(preds[:, 1:] == targets[:, 1:]).float()
                    correct += c

            accuracy = correct / len(val_dataset)
            wandb_logger.log({"val_accuracy": accuracy})

    remaining_components = print_remaining_compontents(pruner)
    print(remaining_components)


if __name__ == "__main__":
    main()
