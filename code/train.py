"""
Train encoder-decoder Transformer model on full dataset.
"""

import argparse
import logging
import os
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from comp_rep.data_prep.dataset import CollateFunctor, SequenceDataset
from comp_rep.eval.decoding import GreedySearch
from comp_rep.eval.evaluator import evaluate_generation
from comp_rep.models.lightning_models import LitTransformer
from comp_rep.utils import (
    ValidateSavePath,
    ValidateWandbPath,
    create_tokenizer_dict,
    save_tokenizer,
    set_seed,
    setup_logging,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[1] / "data" / "PCFGS"
RESULT_DIR = CURR_FILE_DIR / "predictions"


def parse_args() -> argparse.Namespace:
    """
    Parses the command line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        "Full Encoder-Decoder Transformer training script."
    )

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
        "--base_model_name", type=str, default="pcfgs_base", help="Name of base model."
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Whether to evaluate the model in addition to training.",
    )

    # Path configs
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

    # Train parameter configs
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=7e-5, help="Learning rate.")
    parser.add_argument(
        "--eta_min", type=float, default=0.0, help="Minimum learning rate."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=64, help="Validation batch size."
    )
    parser.add_argument(
        "--hidden_size", type=int, default=512, help="Size of hidden dimension."
    )
    parser.add_argument("--layers", type=int, default=6, help="Number of layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout parameter.")
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


def main() -> None:
    """
    Main script.
    """
    args = parse_args()

    set_seed(args.seed)
    setup_logging(args.verbose)
    config = vars(args).copy()
    config_string = "\n".join([f"--{k}: {v}" for k, v in config.items()])
    logging.info(f"\nRunning training loop with the config: \n{config_string}")

    wandb_logger = WandbLogger(
            entity="REPLACE_WITH_YOUR_PROJECT_ID",
        project="circomp-base-training",
        config=config,
        save_dir=args.wandb_path,
        mode="disabled",
    )

    # load data
    train_dataset = SequenceDataset(path=DATA_DIR / "train.csv")
    train_tokenizer = create_tokenizer_dict(
        train_dataset.input_language, train_dataset.output_language
    )
    input_vocabulary_size = len(train_tokenizer["input_language"]["index2word"])
    output_vocabulary_size = len(train_tokenizer["output_language"]["index2word"])
    args.input_vocabulary_size = input_vocabulary_size
    args.output_vocabulary_size = output_vocabulary_size
    val_dataset = SequenceDataset(path=DATA_DIR / "test.csv", tokenizer=train_tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=True,
        num_workers=7,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        collate_fn=CollateFunctor(),
        shuffle=False,
        num_workers=7,
        persistent_workers=True,
    )

    # init model
    base_model_dir = args.save_path / args.base_model_name
    args.T_max = args.epochs * len(train_loader)

    pl_transformer = LitTransformer(args)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # checkpoint saving
    model_ckpt_name = "base_model.ckpt"
    model_ckpt_path = base_model_dir / model_ckpt_name
    if os.path.exists(model_ckpt_path):
        logging.warning(
            f"File: {model_ckpt_path} already exists. File will be overwritten."
        )
        os.remove(model_ckpt_path)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=base_model_dir,
        filename=model_ckpt_name.split(".")[0],
        save_top_k=1,
        mode="min",
    )

    # train model
    trainer = L.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_alg,
        max_epochs=args.epochs,
        logger=wandb_logger,
    )
    trainer.fit(pl_transformer, train_loader, val_loader)
    save_tokenizer(base_model_dir, train_tokenizer)

    # evaluate
    if args.eval:
        prediction_path = RESULT_DIR / args.base_model_name
        os.makedirs(prediction_path, exist_ok=True)

        searcher = GreedySearch(pl_transformer.model, val_dataset.output_language)
        accuracy = evaluate_generation(
            pl_transformer.model,
            searcher,
            val_loader,
            predictions_path=prediction_path,
            device=DEVICE,
        )
        logging.info(f"Final accuracy was: {accuracy}")
        wandb.log({"final_accuracy": accuracy})  # type: ignore[attr-defined]

    wandb.finish()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
