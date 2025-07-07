"""
Test kl_divergence score
"""

import pytest
import torch
import torch.nn.functional as F

from comp_rep.eval.metrics import kl_divergence


def test_kl_divergence_no_mask() -> None:
    """
    Test kl_divergence without a mask on simple distributions.
    """
    p_logits = (
        torch.tensor([0.2, 0.5, 0.3]).log().unsqueeze(0)
    )  # Shape: [1, num_classes]
    q_probs = torch.tensor([0.1, 0.7, 0.2]).unsqueeze(0)  # Shape: [1, num_classes]
    # Compute expected KL divergence manually
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    expected_kl = F.kl_div(p_log_probs, q_probs, reduction="batchmean").item()
    kl = kl_divergence(p_logits, q_probs)
    assert torch.isclose(
        torch.tensor(kl), torch.tensor(expected_kl), atol=1e-6
    ), f"Expected KL divergence {expected_kl}, but got {kl}"


def test_kl_divergence_with_mask() -> None:
    """
    Test kl_divergence with a mask on batched distributions.
    """
    p_logits = torch.log(
        torch.tensor(
            [[[0.2, 0.5, 0.3], [0.1, 0.6, 0.3]], [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]]]
        )
    )  # Shape: [batch_size=2, seq_len=2, num_classes=3]
    q_probs = torch.tensor(
        [[[0.1, 0.7, 0.2], [0.2, 0.5, 0.3]], [[0.2, 0.5, 0.3], [0.1, 0.7, 0.2]]]
    )  # Same shape as p_logits
    mask = torch.tensor(
        [
            [1, 0],  # Only consider the first token in the first sequence
            [0, 1],  # Only consider the second token in the second sequence
        ],
        dtype=torch.float32,
    )  # Shape: [batch_size=2, seq_len=2]

    # Compute expected KL divergence manually
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    kl_elements = F.kl_div(p_log_probs, q_probs, reduction="none")  # Shape: [2, 2, 3]
    kl_per_sample = kl_elements.sum(dim=-1)  # Sum over classes, shape: [2, 2]
    kl_masked = kl_per_sample * mask
    total_loss = kl_masked.sum()
    total_mask = mask.sum()
    expected_kl = (total_loss / total_mask).item()

    kl = kl_divergence(p_logits, q_probs, mask=mask)

    assert torch.isclose(
        torch.tensor(kl), torch.tensor(expected_kl), atol=1e-6
    ), f"Expected KL divergence {expected_kl}, but got {kl}"


def test_kl_divergence_zero_mask() -> None:
    """
    Test kl_divergence when the mask sums to zero (no valid tokens).
    """
    p_logits = torch.randn(2, 3, 4)  # Shape: [batch_size=2, seq_len=3, num_classes=4]
    q_probs = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    mask = torch.zeros(2, 3)  # No valid tokens
    kl = kl_divergence(p_logits, q_probs, mask=mask)
    assert kl == 0.0, f"Expected KL divergence 0.0, but got {kl}"


def test_kl_divergence_batch_input_no_mask() -> None:
    """
    Test kl_divergence with batched inputs without a mask.
    """
    p_logits = torch.randn(5, 10)  # Shape: [batch_size=5, num_classes=10]
    q_probs = torch.softmax(torch.randn(5, 10), dim=-1)
    # Compute expected KL divergence using PyTorch's built-in function
    p_log_probs = F.log_softmax(p_logits, dim=-1)
    kl_elements = F.kl_div(p_log_probs, q_probs, reduction="none")
    expected_kl = kl_elements.sum(dim=-1).mean().item()
    kl = kl_divergence(p_logits, q_probs)
    assert torch.isclose(
        torch.tensor(kl), torch.tensor(expected_kl), atol=1e-6
    ), f"Expected KL divergence {expected_kl}, but got {kl}"


def test_kl_divergence_different_shapes() -> None:
    """
    Test kl_divergence with inputs of higher dimensions and corresponding masks.
    """
    p_logits = torch.randn(3, 4, 5)  # Shape: [batch_size=3, seq_len=4, num_classes=5]
    q_probs = torch.softmax(torch.randn(3, 4, 5), dim=-1)
    mask = torch.randint(0, 2, (3, 4)).float()  # Shape: [batch_size=3, seq_len=4]
    kl = kl_divergence(p_logits, q_probs, mask=mask)
    # Ensure kl is a scalar float value
    assert isinstance(kl, float), f"KL divergence should be a float, got {type(kl)}"


def test_kl_divergence_numerical_stability() -> None:
    """
    Test kl_divergence for numerical stability with extremely small probabilities.
    """
    p_logits = torch.full((2, 3), -1e9)  # Very negative logits
    q_probs = torch.full((2, 3), 1e-9)
    q_probs = q_probs / q_probs.sum(dim=-1, keepdim=True)  # Normalize to probabilities
    kl = kl_divergence(p_logits, q_probs)
    assert not torch.isnan(
        torch.tensor(kl)
    ), "KL divergence is NaN due to numerical instability"


def test_kl_divergence_total_mask_zero() -> None:
    """
    Test kl_divergence when the total_mask is zero after masking invalid tokens.
    """
    p_logits = torch.randn(2, 3, 4)  # Shape: [batch_size=2, seq_len=3, num_classes=4]
    q_probs = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    mask = torch.zeros(2, 3)  # No valid tokens
    kl = kl_divergence(p_logits, q_probs, mask=mask)
    assert (
        kl == 0.0
    ), f"Expected KL divergence 0.0 when total_mask is zero, but got {kl}"


def test_kl_divergence_invalid_inputs() -> None:
    """
    Test kl_divergence with invalid inputs to ensure it raises appropriate errors.
    """
    p_logits = torch.randn(2, 3, 4)
    q_probs = torch.randn(2, 3, 5)  # Mismatched num_classes
    with pytest.raises(RuntimeError):
        kl_divergence(p_logits, q_probs)


def test_kl_divergence_mask_broadcasting() -> None:
    """
    Test kl_divergence to ensure mask broadcasting works correctly.
    """
    p_logits = torch.randn(2, 3, 4)
    q_probs = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)  # Shape: [2, 3]
    kl = kl_divergence(p_logits, q_probs, mask=mask)
    # Ensure kl is a scalar float value
    assert isinstance(kl, float), f"KL divergence should be a float, got {type(kl)}"


def test_kl_divergence_large_inputs() -> None:
    """
    Test kl_divergence with large input tensors to evaluate performance.
    """
    p_logits = torch.randn(
        64, 128, 1000
    )  # Shape: [batch_size=64, seq_len=128, num_classes=1000]
    q_probs = torch.softmax(torch.randn(64, 128, 1000), dim=-1)
    mask = torch.randint(0, 2, (64, 128)).float()  # Shape: [64, 128]
    kl = kl_divergence(p_logits, q_probs, mask=mask)
    assert isinstance(kl, float), "KL divergence should be a float for large inputs"


def test_kl_divergence_single_element() -> None:
    """
    Test kl_divergence with single-element tensors.
    """
    p_logits = torch.tensor([[10.0]])  # Shape: [1, 1]
    q_probs = torch.tensor([[1.0]])  # Shape: [1, 1]
    kl = kl_divergence(p_logits, q_probs)
    expected_kl = 0.0  # Since both distributions are the same
    assert torch.isclose(
        torch.tensor(kl), torch.tensor(expected_kl), atol=1e-6
    ), f"Expected KL divergence {expected_kl}, but got {kl}"


def test_kl_divergence_identical_distributions() -> None:
    """
    Test kl_divergence with identical distributions.
    """
    p_logits = torch.randn(2, 3, 4)
    q_probs = torch.softmax(p_logits, dim=-1)
    kl = kl_divergence(p_logits, q_probs)
    expected_kl = 0.0  # KL divergence between identical distributions is zero
    assert torch.isclose(
        torch.tensor(kl), torch.tensor(expected_kl), atol=1e-6
    ), f"Expected KL divergence {expected_kl}, but got {kl}"


def test_kl_divergence_negative_logits() -> None:
    """
    Test kl_divergence with negative logits to ensure stability.
    """
    p_logits = torch.randn(2, 3, 4) - 1000  # Very negative logits
    q_probs = torch.softmax(torch.randn(2, 3, 4), dim=-1)
    kl = kl_divergence(p_logits, q_probs)
    assert not torch.isnan(
        torch.tensor(kl)
    ), "KL divergence is NaN due to negative logits"
