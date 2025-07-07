"""
Unit test for JSD metric
"""

import math

import pytest
import torch
import torch.nn.functional as F

from comp_rep.eval.metrics import jensen_shannon_divergence_from_logits


def test_same_distributions():
    """
    Test that the JSD between identical distributions is approximately zero.
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0]])
    q_probs = F.softmax(p_logits, dim=-1)
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert abs(jsd - 0.0) < 1e-6, "JSD should be zero for identical distributions."


def test_different_distributions():
    """
    Test that the JSD between different distributions is positive.
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0]])
    q_probs = torch.tensor([[0.1, 0.1, 0.8]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert jsd > 0.0, "JSD should be positive for different distributions."


def test_symmetry():
    """
    Test that the JSD is symmetric: JSD(P || Q) == JSD(Q || P).
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0]])
    q_logits = torch.tensor([[2.0, 1.0, 0.5]])
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = F.softmax(q_logits, dim=-1)

    jsd_pq = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    jsd_qp = jensen_shannon_divergence_from_logits(q_logits, p_probs)

    assert abs(jsd_pq - jsd_qp) < 1e-6, "JSD should be symmetric between P and Q."


def test_uniform_distributions():
    """
    Test that the JSD is zero when both distributions are uniform.
    """
    p_logits = torch.log(torch.tensor([[1.0, 1.0, 1.0]]))
    q_probs = torch.tensor([[1 / 3, 1 / 3, 1 / 3]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert abs(jsd - 0.0) < 1e-6, "JSD should be zero for uniform distributions."


def test_normalization():
    """
    Test that the function handles unnormalized logits.
    """
    p_logits = torch.tensor([[10.0, 0.0, -10.0]])
    q_probs = torch.tensor([[0.7, 0.2, 0.1]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert jsd > 0.0, "JSD should handle unnormalized logits and return positive value."


def test_batch_inputs():
    """
    Test that the function correctly handles batch inputs.
    """
    p_logits = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 0.5]])
    q_probs = torch.tensor([[0.2, 0.3, 0.5], [0.3, 0.4, 0.3]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert isinstance(jsd, float), "JSD should return a scalar value for batch inputs."


def test_invalid_inputs():
    """
    Test that the function raises an error for inputs of mismatched sizes.
    """
    p_logits = torch.tensor([[1.0, 2.0]])
    q_probs = torch.tensor([[0.5, 0.5, 0.0]])
    with pytest.raises(RuntimeError):
        jensen_shannon_divergence_from_logits(p_logits, q_probs)


def test_large_values():
    """
    Test the function with large logits and probabilities to check numerical stability.
    """
    p_logits = torch.tensor([[1000.0, 2000.0, 3000.0]])
    q_probs = torch.tensor([[0.0, 0.0, 1.0]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert jsd >= 0.0, "JSD should handle large values without numerical issues."


def test_bounds():
    """
    Test that the JSD between highly different distribution is one.
    """
    p_logits = torch.tensor([[-100000.0, 0.0, -1000000.0]])
    q_probs = torch.tensor([[1.0, 0.0, 0.0]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs)
    assert abs(jsd - 1.0) < 1e-6, "JSD should be 1 for one-hot different distributions."


def test_exact_values():
    """
    Test JSD between two given distributions.
    """
    p_probs = torch.tensor([[0.3, 0.2, 0.0, 0.1, 0.0, 0.4]])
    p_logits = torch.log(p_probs)
    q_probs = torch.tensor([[0.1, 0.0, 0.2, 0.3, 0.0, 0.4]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs, sqrt=False)
    assert abs(jsd - 0.2755) < 1e-4, "JSD should be around 0.2755 for given probs."


def test_exact_values_with_sqrt():
    """
    Test JSD between two given distributions.
    """
    p_probs = torch.tensor([[0.3, 0.2, 0.0, 0.1, 0.0, 0.4]])
    p_logits = torch.log(p_probs)
    q_probs = torch.tensor([[0.1, 0.0, 0.2, 0.3, 0.0, 0.4]])
    jsd = jensen_shannon_divergence_from_logits(p_logits, q_probs, sqrt=True)
    assert (
        abs(jsd - 0.2755**0.5) < 1e-4
    ), "JS-distance should be around 0.2755**0.5 for given probs.."


def test_jsd_with_mask():
    """
    Test the function with a mask that includes zeros and ones.
    """
    p_logits = torch.tensor([[[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]]])
    q_probs = torch.tensor([[[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]]])
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    # Compute expected JSD manually for unmasked positions
    eps = 1e-10
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

    p_probs += eps
    q_probs += eps

    m = 0.5 * (p_probs + q_probs)
    kl_pm = torch.sum(p_probs * (torch.log(p_probs) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q_probs * (torch.log(q_probs) - torch.log(m)), dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)

    jsd = jsd * mask
    seq_loss_sum = jsd.sum(dim=-1)
    seq_mask_sum = mask.sum(dim=-1)
    jsd_seq = seq_loss_sum / (seq_mask_sum + (seq_mask_sum == 0) * eps)

    non_zero_sequences = (mask.sum(dim=-1) > 0).sum()
    jsd_batch_mean = jsd_seq.sum() / (
        non_zero_sequences + (non_zero_sequences == 0) * eps
    )
    expected_jsd = (jsd_batch_mean / math.log(2)).item()

    jsd_value = jensen_shannon_divergence_from_logits(p_logits, q_probs, mask=mask)
    assert abs(jsd_value - expected_jsd) < 1e-6


def test_jsd_with_mask_all_zeros():
    """
    Test the function with a mask that is all zeros.
    """
    p_logits = torch.tensor([[[0.2, 0.8], [0.5, 0.5]]])
    q_probs = torch.tensor([[[0.6, 0.4], [0.3, 0.7]]])
    mask = torch.tensor([[0.0, 0.0]])

    jsd_value = jensen_shannon_divergence_from_logits(p_logits, q_probs, mask=mask)
    expected_jsd = 0.0  # Should handle division by zero and return 0
    assert abs(jsd_value - expected_jsd) < 1e-6


def test_jsd_mask_all_ones_vs_no_mask():
    """
    Test that a mask of all ones yields the same result as no mask.
    """
    p_logits = torch.tensor([[[0.2, 0.8], [0.5, 0.5]]])
    q_probs = torch.tensor([[[0.6, 0.4], [0.3, 0.7]]])
    mask = torch.tensor([[1.0, 1.0]])

    jsd_with_mask = jensen_shannon_divergence_from_logits(p_logits, q_probs, mask=mask)
    jsd_no_mask = jensen_shannon_divergence_from_logits(p_logits, q_probs)

    assert abs(jsd_with_mask - jsd_no_mask) < 1e-6


def test_jsd_mask_some_zero_sequences():
    """
    Test the function when some sequences in the batch have a mask sum of zero.
    """
    p_logits = torch.tensor(
        [
            [[0.2, 0.8], [0.5, 0.5]],  # First sequence
            [[0.3, 0.7], [0.6, 0.4]],  # Second sequence
        ]
    )  # Shape: [2, 2, 2]
    q_probs = torch.tensor(
        [[[0.6, 0.4], [0.3, 0.7]], [[0.5, 0.5], [0.4, 0.6]]]
    )  # Shape: [2, 2, 2]
    mask = torch.tensor(
        [
            [0.0, 0.0],  # First sequence mask sum is zero
            [1.0, 1.0],  # Second sequence mask sum is non-zero
        ]
    )  # Shape: [2, 2]

    # Compute expected JSD manually for the second sequence
    eps = 1e-10
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

    p_probs += eps
    q_probs += eps

    m = 0.5 * (p_probs + q_probs)
    kl_pm = torch.sum(p_probs * (torch.log(p_probs) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q_probs * (torch.log(q_probs) - torch.log(m)), dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)

    jsd = jsd * mask
    seq_loss_sum = jsd.sum(dim=-1)
    seq_mask_sum = mask.sum(dim=-1)
    jsd_seq = seq_loss_sum / (seq_mask_sum + (seq_mask_sum == 0) * eps)

    non_zero_sequences = (mask.sum(dim=-1) > 0).sum()
    jsd_batch_mean = jsd_seq.sum() / (
        non_zero_sequences + (non_zero_sequences == 0) * eps
    )
    expected_jsd = (jsd_batch_mean / math.log(2)).item()

    jsd_value = jensen_shannon_divergence_from_logits(p_logits, q_probs, mask=mask)
    assert abs(jsd_value - expected_jsd) < 1e-6


def test_jsd_mask_with_zeros():
    """
    Test the function with masks containing zeros in different positions.
    """
    p_logits = torch.tensor(
        [[[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]], [[0.3, 0.7], [0.6, 0.4], [0.8, 0.2]]]
    )  # Shape: [2, 3, 2]
    q_probs = torch.tensor(
        [[[0.6, 0.4], [0.3, 0.7], [0.5, 0.5]], [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]]
    )  # Shape: [2, 3, 2]
    mask = torch.tensor(
        [
            [1.0, 0.0, 1.0],  # First sequence: mask out position 1
            [0.0, 1.0, 1.0],  # Second sequence: mask out position 0
        ]
    )  # Shape: [2, 3]

    # Compute expected JSD manually for unmasked positions
    eps = 1e-10
    p_probs = F.softmax(p_logits, dim=-1)
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)

    p_probs += eps
    q_probs += eps

    m = 0.5 * (p_probs + q_probs)
    kl_pm = torch.sum(p_probs * (torch.log(p_probs) - torch.log(m)), dim=-1)
    kl_qm = torch.sum(q_probs * (torch.log(q_probs) - torch.log(m)), dim=-1)
    jsd = 0.5 * (kl_pm + kl_qm)

    jsd = jsd * mask
    seq_loss_sum = jsd.sum(dim=-1)
    seq_mask_sum = mask.sum(dim=-1)
    jsd_seq = seq_loss_sum / (seq_mask_sum + (seq_mask_sum == 0) * eps)

    non_zero_sequences = (mask.sum(dim=-1) > 0).sum()
    jsd_batch_mean = jsd_seq.sum() / (
        non_zero_sequences + (non_zero_sequences == 0) * eps
    )
    expected_jsd = (jsd_batch_mean / math.log(2)).item()

    jsd_value = jensen_shannon_divergence_from_logits(p_logits, q_probs, mask=mask)
    assert abs(jsd_value - expected_jsd) < 1e-6
