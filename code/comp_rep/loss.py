"""
Loss computations.
"""

import lightning as L
import torch.nn.functional as F

from comp_rep.data_prep.dataset import PAD_TOKEN


def get_logits_loss(
    pl_module: L.LightningModule,
    batch: tuple,
) -> tuple:
    """
    Calculates the logits loss for a given model and batch.

    Args:
        model (L.LightningModule): The model to calculate the logits loss for.
        batch (tuple): The batch of data to calculate the logits loss on.

    Returns:
        tuple: A tuple containing the logits and the calculated loss.
    """
    _, _, target_ids, _, _, _ = batch
    logits = pl_module(batch)

    # Transpose output to [batch size, vocab, seq len] to match the required dims for CE.
    # Also, right shift the targets so that it matches the output order.
    # (at position k the decoder writes to pos k + 1).
    # Ignore pad tokens in loss computation.
    loss = F.cross_entropy(
        logits.transpose(-2, -1), target_ids[:, 1:], ignore_index=PAD_TOKEN
    )
    return logits, loss


def get_regularized_logits_loss(
    pl_module: L.LightningModule,
    mask_lambda: float,
    batch: tuple,
) -> tuple:
    """
    Calculates the regularized logits loss based on the input model, mask lambda, and batch.

    Parameters:
        model (L.LightningModule): The LightningModule model.
        mask_lambda (float): The lambda value for masking.
        batch (tuple): The input batch tuple.

    Returns:
        tuple: A tuple containing logits, cross entropy loss, mask loss, and total loss.
    """
    _, _, target_ids, _, target_probabilities, _, _ = batch

    # first loss term
    logits = pl_module(batch)
    cross_entropy_loss = F.cross_entropy(
        logits.transpose(-2, -1),
        target_probabilities.transpose(-2, -1),
        reduction="none",
    )
    pad_mask = (target_ids[:, 1:] != PAD_TOKEN).float()
    masked_loss = cross_entropy_loss * pad_mask

    if len(masked_loss.shape) == 2:
        seq_loss_sum = masked_loss.sum(dim=-1)
        seq_mask_sum = pad_mask.sum(dim=-1)
        masked_loss = seq_loss_sum / (
            seq_mask_sum + (seq_mask_sum == 0) * 1.0e-10
        )  # avoid division by 0

    non_zero_sequences = (pad_mask.sum(dim=-1) > 0).sum()
    final_cross_entropy_loss = masked_loss.sum(dim=-1) / (
        non_zero_sequences + (non_zero_sequences == 0) * 1.0e-10
    )  # avoid division by 0

    # second loss term
    circuit_norms = pl_module.pruner.compute_l1_norm()
    circuit_norm_loss = mask_lambda * circuit_norms

    # full loss
    loss = final_cross_entropy_loss + circuit_norm_loss
    return logits, final_cross_entropy_loss, circuit_norm_loss, loss
