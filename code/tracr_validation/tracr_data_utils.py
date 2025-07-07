from typing import Any, Dict, List

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_datasets(dataset_path, max_train_samples, max_eval_samples):
    dataset_ = load_from_disk(dataset_path)
    dataset = DatasetDict(
        {
            "train": dataset_,
            "validation": dataset_,
        }
    )
    if max_train_samples is not None and max_train_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train_samples))
    if max_eval_samples is not None and max_eval_samples < len(dataset["validation"]):
        dataset["validation"] = dataset["validation"].select(range(max_eval_samples))
    return dataset


def unstringify(string_list: List[Any]) -> List[int]:
    """
    Convert a list of elements to a list of integers.

    The first element in the list is not converted, the remaining elements are converted to integers.

    Args:
        string_list (List[Any]): The list of strings to convert.

    Returns:
        List[int]: The list of integers.
    """
    for i in range(1, len(string_list)):
        string_list[i] = int(string_list[i])
    return string_list


class ErazrTokenizer:
    def __init__(self, vocab, bos, pad):
        self.vocab = vocab
        self.bos = bos
        self.pad = pad
        self.vocab_dict = vocab
        self.rev_vocab_dict = {v: k for k, v in self.vocab_dict.items()}
        self.bos_token = self.vocab_dict[self.bos]
        self.pad_token = self.vocab_dict[self.pad]

    def pad_or_truncate(self, tokens, max_length, padding=False, truncate=False):
        if truncate and len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif padding and len(tokens) < max_length:
            tokens = tokens + [self.vocab_dict[self.pad]] * (max_length - len(tokens))
        return tokens

    def encode_single(
        self,
        text,
        max_length=None,
        padding=False,
        truncate=False,
        return_tensors=False,
        add_special_tokens=True,
    ):
        if type(text) != list:
            text = text.split()
        if add_special_tokens and (len(text) == 0 or text[0] != self.bos):
            tokens = [self.bos_token]
        else:
            tokens = []
        tokens += [self.vocab_dict.get(t, self.vocab_dict[self.pad]) for t in text]
        if max_length is not None:
            tokens = self.pad_or_truncate(tokens, max_length, padding, truncate)
        if return_tensors == "np":
            return np.array(tokens, dtype=int)
        elif return_tensors == "pt":
            return torch.LongTensor(tokens)
        else:
            return tokens

    def encode(
        self,
        texts,
        max_length=None,
        padding=False,
        truncate=False,
        return_tensors=False,
        add_special_tokens=True,
    ):
        if type(texts) == str:
            texts = [texts]
        encoded = [
            self.encode_single(
                t, max_length, padding, truncate, add_special_tokens=add_special_tokens
            )
            for t in texts
        ]
        max_l = max([len(e) for e in encoded])
        min_l = min([len(e) for e in encoded])
        if min_l != max_l:
            assert (
                padding
            ), "All sequences must have the same length if padding is not enabled."
            encoded = [
                self.pad_or_truncate(e, max_l, padding, truncate) for e in encoded
            ]
        if return_tensors == "np":
            return np.array(encoded, dtype=int)
        elif return_tensors == "pt":
            return torch.LongTensor(encoded)
        else:
            return encoded

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode_single(self, tokens, remove_special_tokens=True, starts_with_bos=False):
        if type(tokens) == torch.Tensor or type(tokens) == np.ndarray:
            tokens = tokens.tolist()
        if starts_with_bos:  # The bos token is somehow not in the unembedding matrix
            tokens[0] = self.bos_token
        if remove_special_tokens:
            tokens = [t for t in tokens if t not in [self.bos_token, self.pad_token]]
        return [self.rev_vocab_dict[t] for t in tokens]

    def decode(self, tokens, remove_special_tokens=True, starts_with_bos=False):
        return [
            self.decode_single(t, remove_special_tokens, starts_with_bos)
            for t in tokens
        ]


class DataCollatorTracr:
    def __init__(
        self,
        tokenizer: AutoTokenizer | PreTrainedTokenizerFast,
        length: int,
    ):
        self.tokenizer = tokenizer
        self.length = length

    def __call__(self, examples: List[Dict[str, List[Any]]]) -> Dict[str, torch.Tensor]:
        """
        Collates a list of examples into a single batch.

        Args:
            examples (List[Dict[str, List[Any]]]): A list of examples.

        Returns:
            (Dict[str, torch.Tensor]): A dictionary containing the input IDs, corrupted input IDs, and labels.
        """
        input_ids_all = []
        corr_input_ids_all = []
        labels_all = []

        for example in examples:
            seq = unstringify(example["seq"])
            target = unstringify(example["target"])
            corr_seq = unstringify(example["corr_seq"])

            input_ids = self.tokenizer([seq], return_tensors="pt")[0]
            corr_input_ids = self.tokenizer([corr_seq], return_tensors="pt")[0]
            labels = self.tokenizer([target], return_tensors="pt")[0]

            assert (
                input_ids.shape[0] == self.length
            ), f"Input length is {input_ids.shape[0]}, expected {self.length}"
            assert (
                corr_input_ids.shape[0] == self.length
            ), f"Corrupted length is {corr_input_ids.shape[0]}, expected {self.length}"
            assert (
                labels.shape[0] == self.length
            ), f"Target length is {labels.shape[0]}, expected {self.length}"

            input_ids_all.append(input_ids)
            corr_input_ids_all.append(corr_input_ids)
            labels_all.append(labels)

        return {
            "input_ids": torch.stack(input_ids_all),
            "corr_input_ids": torch.stack(corr_input_ids_all),
            "labels": torch.stack(labels_all),
        }
