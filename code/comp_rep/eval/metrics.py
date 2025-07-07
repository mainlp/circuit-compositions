"""
Evaluation metrics
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def jensen_shannon_divergence_from_logits(
    p_logits: Tensor,
    q_probs: Tensor,
    eps: float = 1e-10,
    mask: Optional[Tensor] = None,
    sqrt: bool = False,
) -> float:
    """
    Computes the normalized Jensen-Shannon Divergence (JSD) between two probability distributions.

    Args:
        p_logits (Tensor): The logits of the first probability distribution.
        q_probs (Tensor): The probabilities of the second probability distribution.
        eps (float, optional): A small value added to the probabilities for numerical stability. Defaults to 1e-10.
        mask: (Tensor, optional): A mask for the loss calculation. Defaults to None.
        sqrt: (bool, optional): Whether to take the square-root of the JSD to obtain the Jensen-Shannon distance instead of the divergence. Defaults to False.

    Returns:
        float: The Jensen-Shannon Divergence between the two probability distributions.
    """
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = q_probs

    # Normalize probabilities to ensure they sum to 1
    p_probs = p_probs / p_probs.sum(dim=-1, keepdim=True)
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

    p_probs = p_probs + eps
    q_probs = q_probs + eps

    # compute average distribution M
    m = 0.5 * (p_probs + q_probs)

    # Compute KL divergences
    kl_pm = torch.sum(p_probs * (torch.log(p_probs) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q_probs * (torch.log(q_probs) - torch.log(m)), dim=-1)

    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)  # Shape: [batch_size, seq_len]

    # Mask tokens for cross-task-faithfulness
    if mask is not None:
        jsd = jsd * mask

        # Average over sequence
        if len(jsd.shape) == 2:
            seq_loss_sum = jsd.sum(dim=-1)
            seq_mask_sum = mask.sum(dim=-1)
            jsd = seq_loss_sum / (
                seq_mask_sum + (seq_mask_sum == 0) * eps
            )  # avoid division by 0

        # Average over batch
        non_zero_sequences = (mask.sum(dim=-1) > 0).sum()
        jsd_batch_mean = jsd.sum(dim=-1) / (
            non_zero_sequences + (non_zero_sequences == 0) * eps
        )
    else:
        if len(jsd.shape) == 2:
            jsd = jsd.mean(dim=-1)
        jsd_batch_mean = jsd.mean(dim=-1)

    jsd_normalized = jsd_batch_mean / math.log(2)

    if sqrt:
        return torch.sqrt(jsd_normalized).item()

    return jsd_normalized.item()


def jsd_faithfulness(
    p_logits: Tensor,
    q_probs: Tensor,
    eps: float = 1e-10,
    mask: Optional[Tensor] = None,
    sqrt: bool = False,
) -> float:
    """
    Computes the faithfulness of a probability distribution based on the Jensen-Shannon Divergence (JSD) between two distributions.

    Args:
        p_logits (Tensor): The logits of the first probability distribution.
        q_probs (Tensor): The probabilities of the second probability distribution.
        eps (float, optional): A small value added to the probabilities for numerical stability. Defaults to 1e-10.
        mask: (Tensor, optional): A mask for the loss calculation. Defaults to None.
        sqrt: (bool, optional): Whether to take the square-root of the JSD to obtain the Jensen-Shannon distance instead of the divergence. Defaults to False.

    Returns:
        float: The faithfulness of the probability distribution, calculated as 1 minus the JSD between the two distributions.
    """
    return 1.0 - jensen_shannon_divergence_from_logits(
        p_logits, q_probs, eps=eps, mask=mask, sqrt=sqrt
    )


def kl_divergence(
    p_logits: Tensor,
    q_probs: Tensor,
    eps: float = 1e-10,
    mask: Optional[Tensor] = None,
) -> float:
    """
    Computes the kl_divergence between two probability distributions.

    Args:
        p_logits (Tensor): The logits of the first probability distribution.
        q_probs (Tensor): The probabilities of the second probability distribution.
        eps (float, optional): A small value added to the probabilities for numerical stability. Defaults to 1e-10.
        mask: (Tensor, optional): A mask for the loss calculation. Defaults to None.

    Returns:
        float: The Jensen-Shannon Divergence between the two probability distributions.
    """
    # Convert logits to log-probabilities
    p_log_probs = F.log_softmax(p_logits, dim=-1)

    # Compute element-wise KL divergence
    kl_loss = F.kl_div(p_log_probs, q_probs, reduction="none")
    kl_loss = kl_loss.sum(dim=-1)

    # Mask tokens for cross-task-faithfulness
    if mask is not None:
        kl_loss = kl_loss * mask

        # Average over sequence
        if len(kl_loss.shape) == 2:
            seq_loss_sum = kl_loss.sum(dim=-1)
            seq_mask_sum = mask.sum(dim=-1)
            kl_loss = seq_loss_sum / (
                seq_mask_sum + (seq_mask_sum == 0) * eps
            )  # avoid division by 0

        # Average over batch
        non_zero_sequences = (mask.sum(dim=-1) > 0).sum()
        kl_batch_mean = kl_loss.sum(dim=-1) / (
            non_zero_sequences + (non_zero_sequences == 0) * eps
        )
    else:
        if len(kl_loss.shape) == 2:
            kl_loss = kl_loss.mean(dim=-1)
        kl_batch_mean = kl_loss.mean(dim=-1)

    return kl_batch_mean.item()
