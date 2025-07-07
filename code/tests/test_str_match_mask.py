"""
Unit tests for string match mask computation.
"""

import pytest
import torch

from comp_rep.eval.cross_task_evaluations import is_unary, str_match_mask


def test_basic_functionality():
    str_list_a = [["A1", "B1"], ["D1", "C1"]]
    target_str = ["A1 B1", "D1 E1"]
    seq_len = 3
    expected_mask = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # "A1"=="A1", "B1"=="B1", "EOS"=="EOS"
            [0.0, 1.0, 0.0],  # "foo"=="foo", "bar"!="baz", "EOS"=="EOS"
        ],
        dtype=torch.float32,
    )
    mask = str_match_mask(str_list_a, target_str, seq_len)
    assert torch.equal(mask, expected_mask)


def test_mismatched_batch_size():
    str_list_a = [["A1", "B1"]]
    target_str = ["A1 B1", "extra"]
    seq_len = 3
    with pytest.raises(ValueError, match="Input lists must have the same batch size."):
        str_match_mask(str_list_a, target_str, seq_len)


def test_sequences_longer_than_seq_len():
    str_list_a = [["a", "b", "c", "d"]]
    target_str = ["a b c d e"]
    seq_len = 3
    with pytest.raises(IndexError):
        str_match_mask(str_list_a, target_str, seq_len)


def test_sequences_shorter_than_seq_len():
    str_list_a = [["a"]]
    target_str = ["a"]
    seq_len = 5
    expected_mask = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    mask = str_match_mask(str_list_a, target_str, seq_len)
    assert torch.equal(mask, expected_mask)


def test_empty_sequences():
    str_list_a = [[]]
    target_str = [""]
    seq_len = 3
    expected_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    mask = str_match_mask(str_list_a, target_str, seq_len)
    assert torch.equal(mask, expected_mask)


def test_special_characters():
    str_list_a = [["A1", "B1!"]]
    target_str = ["A1 B1!"]
    seq_len = 3
    expected_mask = torch.tensor(
        [
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mask = str_match_mask(str_list_a, target_str, seq_len)
    assert torch.equal(mask, expected_mask)


def test_extra_spaces_in_target_str():
    str_list_a = [["A1", "B1"]]
    target_str = ["A1   B1"]
    seq_len = 3
    expected_mask = torch.tensor(
        [
            [0.0, 1.0, 1.0],  # The extra spaces result in empty strings during split
        ],
        dtype=torch.float32,
    )
    mask = str_match_mask(str_list_a, target_str, seq_len)
    assert torch.equal(mask, expected_mask)


def test_large_batch_size():
    str_list_a = [["token{}".format(i) for i in range(1000)]]
    target_str = [" ".join(["token{}".format(i) for i in range(1000)])]
    seq_len = 1001
    expected_mask = torch.zeros((1, 1001), dtype=torch.float32)
    expected_mask[0, -1] = 0.0  # EOS token comparison
    mask = str_match_mask(str_list_a, target_str, seq_len)
    assert torch.equal(mask, expected_mask)


def test_special_case_eos():
    str_list_a = [["EOS"]]
    target_str = ["EOS"]
    seq_len = 2
    expected_mask = torch.tensor(
        [
            [0.0, 0.0],  # "EOS"=="EOS", "EOS"=="EOS"
        ],
        dtype=torch.float32,
    )
    mask = str_match_mask(str_list_a, target_str, seq_len)
    assert torch.equal(mask, expected_mask)


def test_is_unary_with_unary_function():
    def unary_function(x):
        return x * 2

    assert is_unary(unary_function) == True


def test_is_unary_with_binary_function():
    def binary_function(x, y):
        return x + y

    assert is_unary(binary_function) == False
