"""
Pytorch Lightning module for the Transformer model.
"""

import argparse

import lightning as L
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from comp_rep.loss import get_logits_loss
from comp_rep.models.model import Transformer


class LitTransformer(L.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.model = Transformer(
            input_vocabulary_size=self.args.input_vocabulary_size,
            output_vocabulary_size=self.args.output_vocabulary_size,
            num_transformer_layers=args.layers,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
        )

    def forward(self, batch: tuple):
        """
        Forward pass of the model.

        Args:
            batch (tuple): The input batch.

        Returns:
            torch.Tensor: The logits of the model.
        """
        source_ids, source_mask, target_ids, target_mask, _, _ = batch
        # Left shift the targets so that the last token predicts the EOS
        logits = self.model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        )  # [batch size, max seq len, vocab]
        return logits

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            optim.AdamW: The configured optimizer.
        """
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.args.T_max,
            eta_min=self.args.eta_min,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
            },
        }

    def training_step(self, train_batch: tuple, batch_idx: int):
        """
        Perform a training step.

        Args:
            train_batch (tuple): The training batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the training step.
        """
        _, loss = get_logits_loss(self, train_batch)
        self.log(
            "train_loss",
            loss,
            batch_size=self.args.train_batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int):
        """
        Perform a validation step.

        Args:
            val_batch (tuple): The validation batch.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The loss of the validation step.
        """
        _, loss = get_logits_loss(self, val_batch)
        self.log(
            "val_loss",
            loss,
            batch_size=self.args.val_batch_size,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
