"""
Pytorch Lightning module for the Pruned Transformer model.
"""

import argparse
from typing import Any, Dict, Literal

import lightning as L
import torch.optim as optim
import wandb
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from comp_rep.loss import get_regularized_logits_loss
from comp_rep.pruning.activation_pruning.activation_pruner import ActivationPruner
from comp_rep.pruning.pruner import Pruner
from comp_rep.pruning.weight_pruning.weight_pruner import WeightPruner


class LitPrunedModel(L.LightningModule):
    def __init__(
        self,
        args: argparse.Namespace,
        model: nn.Module,
        pruning_type: Literal["activations", "weights"],
        pruning_kwargs: Dict,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.args = args

        # init model and pruner in class
        self.model = model
        self.pruner = self.init_pruner_by_name(
            pruning_type=pruning_type, model=self.model, pruner_kwargs=pruning_kwargs
        )

    @staticmethod
    def init_pruner_by_name(
        pruning_type: Literal["activations", "weights"],
        model: nn.Module,
        pruner_kwargs: Dict,
    ) -> Pruner:
        """
        Initializes a pruner by its class name.

        Args:
            pruning_type (Literal["activations", "weights"]): The pruning type of the pruner to initialize.
            pruner_kwargs (Dict): Keyword arguments to pass to the pruner's constructor.

        Returns:
            Pruner: An instance of the specified pruner class.
        """
        if pruning_type == "activations":
            return ActivationPruner(model=model, **pruner_kwargs)
        elif pruning_type == "weights":
            return WeightPruner(model=model, **pruner_kwargs)
        else:
            raise ValueError(
                f"Invalid pruning type for pruner: {pruning_type}! Must be 'activations' or 'weights'."
            )

    def forward(self, batch: tuple):
        """
        Forward pass of the model.

        Args:
            batch (tuple): The input batch.

        Returns:
            torch.Tensor: The logits of the model.
        """
        source_ids, source_mask, target_ids, target_mask, target_probabilities, _, _ = (
            batch
        )
        # Left shift the targets so that the last token predicts the EOS
        logits = self.model(
            source_ids, source_mask, target_ids[:, :-1], target_mask[:, :-1]
        )  # [batch size, max seq len, vocab]
        return logits

    def configure_optimizers(self):
        """
        Configures the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.
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
        A single training step in the LightningModule.

        Args:
            train_batch (tuple): The batch of training data.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The loss value calculated during the training step.
        """
        self.pruner.compute_and_update_masks()
        _, cross_entropy_loss, mask_loss, loss = get_regularized_logits_loss(
            self, self.args.mask_lambda, train_batch
        )

        self.log_dict(
            {
                "train_loss": loss,
                "cross_entropy_loss": cross_entropy_loss,
                "l1_norm_loss": mask_loss,
            },
            on_step=True,
            logger=True,
            batch_size=self.args.train_batch_size,
        )
        return loss

    def validation_step(self, val_batch: tuple, batch_idx: int):
        """
        Calculates the validation loss for a given batch of data.

        Args:
            val_batch (tuple): The batch of data to calculate the validation loss on.
            batch_idx (int): The index of the batch.

        Returns:
            float: The calculated validation loss.
        """
        self.pruner.compute_and_update_masks()
        _, _, _, loss = get_regularized_logits_loss(
            self, self.args.mask_lambda, val_batch
        )
        self.log(
            "val_loss",
            loss,
            batch_size=self.args.val_batch_size,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_start(self):
        """
        Callback function that is executed at the start of each training epoch.
        If the pruning method is set to "continuous", the pruner's ticket is deactivated.

        Returns:
            None
        """
        self.pruner.deactivate_ticket()

    def on_validation_epoch_start(self):
        """
        Callback function that is executed at the start of each validation epoch.
        If the pruning method is set to "continuous", the pruner's ticket is activated.

        Returns:
            None
        """
        self.pruner.activate_ticket()

    def on_train_epoch_end(self):
        """
        Updates hyperparameters at the end of a training epoch and logs the average remaining mask elements.
        """
        self.pruner.update_hyperparameters()

        remaining_mask_elements = self.pruner.get_remaining_mask()
        wandb.log(remaining_mask_elements)  # need to use the wandb log here

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Saves the state dictionary, configuration, and type of the pruner model to the checkpoint dictionary.

        Args:
            checkpoint (Dict[str, Any]): The dictionary to save the pruner state, configuration, and type.

        Returns:
            None
        """
        setattr(
            checkpoint["hyper_parameters"]["args"],
            "input_vocabulary_size",
            self.model.input_vocabulary_size,
        )
        setattr(
            checkpoint["hyper_parameters"]["args"],
            "output_vocabulary_size",
            self.model.output_vocabulary_size,
        )
        setattr(
            checkpoint["hyper_parameters"]["args"],
            "num_transformer_layers",
            self.model.num_transformer_layers,
        )
        setattr(
            checkpoint["hyper_parameters"]["args"],
            "hidden_size",
            self.model.hidden_size,
        )
        setattr(checkpoint["hyper_parameters"]["args"], "dropout", self.model.dropout)
