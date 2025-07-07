"""
Functions and modules for cross-task evaluations
"""

from inspect import signature
from pathlib import Path
from typing import Callable, Dict, List, Sequence

import torch

from comp_rep.data_prep.operations import (
    append,
    copy,
    echo,
    prepend,
    remove_first,
    remove_second,
    repeat,
    reverse,
    shift,
    swap_first_last,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CURR_FILE_PATH = Path(__file__).resolve()
CURR_FILE_DIR = CURR_FILE_PATH.parent
DATA_DIR = CURR_FILE_PATH.parents[3] / "data"

FUNC_BY_STR: Dict[str, Callable[..., List[str]]] = {
    "copy": copy,
    "reverse": reverse,
    "shift": shift,
    "echo": echo,
    "swap_first_last": swap_first_last,
    "repeat": repeat,
    "append": append,
    "prepend": prepend,
    "remove_first": remove_first,
    "remove_second": remove_second,
}


def is_unary(func: Callable) -> bool:
    """
    Determine whether a function accepts exactly one parameter.

    Args:
        func (Callable): The function to inspect.

    Returns:
        bool: True if the function accepts exactly one parameter, False otherwise.
    """
    return len(signature(func).parameters) == 1


def src_to_func_target(
    src_str_list: List[str], data_func_name: str, new_func_name: str
) -> List[str]:
    """
    Transforms a list of strings by replacing occurrences of a specific function with a new function.

    Args:
        src_str_list (List[str]): The list of strings to transform.
        data_func_name (str): The name of the function to replace.
        new_func_name (str): The name of the new function to replace with.

    Returns:
        List[str]: The transformed list of strings.
    """
    if data_func_name not in src_str_list:
        return src_str_list

    new_func: Callable[..., List[str]] = FUNC_BY_STR[new_func_name]
    func_index = len(src_str_list) - 1 - src_str_list[::-1].index(data_func_name)

    if "," in src_str_list[func_index + 1 :]:
        comma_index = src_str_list.index(",", func_index + 1)
        first_args = src_str_list[func_index + 1 : comma_index]
        second_args = src_str_list[comma_index + 1 :]

        if "," in second_args:
            second_args_comma_index = second_args.index(",")
            remaining_str_list = second_args[second_args_comma_index:]
            second_args = second_args[:second_args_comma_index]
        else:
            remaining_str_list = []

        if is_unary(new_func):
            result = new_func(first_args + second_args) + remaining_str_list
        else:
            result = new_func(first_args, second_args) + remaining_str_list
    else:
        first_args = src_str_list[func_index + 1 :]
        if is_unary(new_func):
            result = new_func(first_args)
        else:
            result = new_func(first_args, [])

    src_str_list = src_str_list[:func_index] + result

    return src_to_func_target(src_str_list, data_func_name, new_func_name)


def map_src_to_func_target(
    source_str: List[str], circuit_name: str, eval_task_name: str
) -> List[List[str]]:
    """
    Maps each source string to a new function.

    Args:
        source_str (List[str]): The list of source strings.
        circuit_name (str): The name of the circuit.
        eval_task_name (str): The name of the evaluation task.

    Returns:
        List[List[str]]: The list of mapped source strings.
    """
    mapped_source_str: List[List[str]] = []
    for source_sentence in source_str:
        mapped_str = src_to_func_target(
            source_sentence.split(" "), eval_task_name, circuit_name
        )
        mapped_source_str.append(mapped_str)

    return mapped_source_str


def str_match_mask(
    str_list_a: List[List[str]], target_str: Sequence[str], seq_len: int
) -> torch.Tensor:
    """
    Computes a 2-dimensional mask by comparing two lists of lists of strings.
    If elements are the same, denotes it as 0; if different, denotes it as 1.

    Args:
        str_list_a (List[List[str]]): The first list of lists of strings.
        str_list_b (List[List[str]]): The second list of lists of strings.

    Returns:
        torch.Tensor: A 2-dimensional tensor mask of binary values (0s and 1s).
    """
    if len(str_list_a) != len(target_str):
        raise ValueError("Input lists must have the same batch size.")

    batch_size = len(str_list_a)

    # Initialize a tensor to hold the mask
    mask = torch.ones((batch_size, seq_len), dtype=torch.float32)

    for i, (seq_a, str_b) in enumerate(zip(str_list_a, target_str)):
        seq_b = str_b.split(" ")
        seq_b += ["EOS"]
        seq_a += ["EOS"]

        for j, (elem_a, elem_b) in enumerate(zip(seq_a, seq_b)):
            mask[i, j] = 0.0 if elem_a == elem_b else 1.0

    return mask


def compute_str_match_mask(
    source_str: List[str],
    target_str: Sequence[str],
    max_seq_len: int,
    circuit_name: str,
    eval_task_name: str,
) -> torch.Tensor:
    """
    Compute the string match mask for a given set of source strings and target strings.

    Args:
        source_str (List[str]): The list of source strings.
        target_str (Sequence[str]): The sequence of target strings.
        max_seq_len (int): The maximum sequence length.
        circuit_name (str): The name of the circuit.
        eval_task_name (str): The name of the evaluation task.

    Returns:
        torch.Tensor: The computed string match mask.
    """
    mapped_source_str = map_src_to_func_target(source_str, circuit_name, eval_task_name)
    mask = str_match_mask(
        str_list_a=mapped_source_str, target_str=target_str, seq_len=max_seq_len
    )

    return mask
