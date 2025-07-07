import pathlib
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CURR_FILE_PATH = Path(__file__).resolve()
DATA_DIR = CURR_FILE_PATH.parents[3] / "data"

Pairs = list[tuple[str, str]]
DatasetItem = tuple[torch.Tensor, torch.Tensor, str, str]
DatasetItemWithProbabilities = tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]
CollatedItem = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str]
CollatedItemWithProbabilities = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, str
]
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2


class Lang:
    def __init__(
        self,
        name: str,
        word2index: Optional[dict] = None,
        index2word: Optional[dict] = None,
    ) -> None:
        if index2word is None:
            index2word = {PAD_TOKEN: "PAD", SOS_TOKEN: "SOS", EOS_TOKEN: "EOS"}
        if word2index is None:
            word2index = {}
        self.name = name
        self.word2count: dict = defaultdict(int)
        self.word2index = word2index
        self.index2word = index2word
        self.n_words = 3  # Count PAD, SOS and EOS tokens

    def add_sentence(self, sentence: str) -> None:
        for word in sentence.split(" "):
            self.add_word(word)

    def add_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def read_pairs(path: pathlib.Path) -> Pairs:
    try:
        lines = open(path, encoding="utf-8").read().strip().split("\n")
    except IOError as e:
        raise e
    pairs = []
    for line in lines:
        inl, outl = line.split(";")
        inl = inl.strip()
        outl = outl.strip()
        pairs.append((inl, outl))
    return pairs


class SequenceDatasetWithProbabilities(Dataset):
    def __init__(
        self,
        path: pathlib.Path,
        probabilities_path: Path,
        tokenizer: Optional[dict] = None,
    ) -> None:
        try:
            self.pairs = read_pairs(path)
        except IOError:
            print(f"Failed to read data file from path: {str(path)}")
            exit(0)

        if tokenizer is not None:
            self.input_language = Lang(
                "INPUT",
                word2index=tokenizer["input_language"]["word2index"],
                index2word=tokenizer["input_language"]["index2word"],
            )
            self.output_language = Lang(
                "OUTPUT",
                word2index=tokenizer["output_language"]["word2index"],
                index2word=tokenizer["output_language"]["index2word"],
            )
        else:
            self.input_language = Lang("INPUT")
            self.output_language = Lang("OUTPUT")
            for ip, op in self.pairs:
                self.input_language.add_sentence(ip)
                self.output_language.add_sentence(op)

        from comp_rep.utils import load_tensor

        self.cached_probabilities = load_tensor(probabilities_path).detach().cpu()

    def __len__(self) -> int:
        return len(self.pairs)

    def indexesFromSentence(self, lang: Lang, sentence: str) -> list[int]:
        return [lang.word2index[word] for word in sentence.split(" ")]

    def __getitem__(self, idx) -> DatasetItemWithProbabilities:
        x, y = self.pairs[idx]
        input_tensor = torch.tensor(
            self.indexesFromSentence(self.input_language, x) + [EOS_TOKEN]
        )
        output_tokens = self.indexesFromSentence(self.output_language, y)
        output_tokens = [SOS_TOKEN] + output_tokens + [EOS_TOKEN]
        output_tensor = torch.tensor(output_tokens)
        output_probabilities = self.cached_probabilities[idx]
        return input_tensor, output_tensor, output_probabilities, x, y


class SequenceDataset(Dataset):
    def __init__(
        self,
        path: pathlib.Path,
        tokenizer: Optional[dict] = None,
    ) -> None:
        try:
            self.pairs = read_pairs(path)
        except IOError:
            print(f"Failed to read data file from path: {str(path)}")
            exit(0)

        if tokenizer is not None:
            self.input_language = Lang(
                "INPUT",
                word2index=tokenizer["input_language"]["word2index"],
                index2word=tokenizer["input_language"]["index2word"],
            )
            self.output_language = Lang(
                "OUTPUT",
                word2index=tokenizer["output_language"]["word2index"],
                index2word=tokenizer["output_language"]["index2word"],
            )
        else:
            self.input_language = Lang("INPUT")
            self.output_language = Lang("OUTPUT")
            for ip, op in self.pairs:
                self.input_language.add_sentence(ip)
                self.output_language.add_sentence(op)

    def __len__(self) -> int:
        return len(self.pairs)

    def indexesFromSentence(self, lang: Lang, sentence: str) -> list[int]:
        return [lang.word2index[word] for word in sentence.split(" ")]

    def __getitem__(self, idx) -> DatasetItem:
        x, y = self.pairs[idx]
        input_tensor = torch.tensor(
            self.indexesFromSentence(self.input_language, x) + [EOS_TOKEN]
        )
        output_tokens = self.indexesFromSentence(self.output_language, y)
        output_tokens = [SOS_TOKEN] + output_tokens + [EOS_TOKEN]
        output_tensor = torch.tensor(output_tokens)
        return input_tensor, output_tensor, x, y


class CollateFunctor:
    def __init__(self, max_length: Optional[int] = None) -> None:
        self.pad_id = PAD_TOKEN
        self.max_length = max_length

    def __call__(self, sentences: list) -> CollatedItem:
        raw_source_ids, raw_target_ids, source_str, target_str = zip(*sentences)
        source_ids, source_mask = self.collate_sentences(raw_source_ids)
        target_ids, target_mask = self.collate_sentences(raw_target_ids)
        return (
            source_ids,
            source_mask,
            target_ids,
            target_mask,
            source_str,
            target_str,
        )

    def collate_sentences(self, sentences: list) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = [sentence.size(0) for sentence in sentences]
        if self.max_length is not None:
            max_length = self.max_length
        else:
            max_length = max(lengths)
        if max_length == 1:  # Some samples are only one token
            max_length += 1

        subword_ids = torch.stack(
            [
                F.pad(sentence, (0, max_length - length), value=self.pad_id)
                for length, sentence in zip(lengths, sentences)
            ]
        )
        attention_mask = subword_ids == self.pad_id
        return subword_ids, attention_mask


class CollateFunctorWithProbabilities:
    def __init__(
        self, probability_mode: bool = False, max_length: Optional[int] = None
    ) -> None:
        self.pad_id = PAD_TOKEN
        self.max_length = max_length

    def __call__(self, sentences: list) -> CollatedItemWithProbabilities:
        raw_source_ids, raw_target_ids, output_probabilities, source_str, target_str = (
            zip(*sentences)
        )
        source_ids, source_mask = self.collate_sentences(raw_source_ids)
        target_ids, target_mask = self.collate_sentences(raw_target_ids)
        truncated_output_probabilities = torch.stack(
            [sample[0 : target_ids.shape[-1] - 1] for sample in output_probabilities]
        )
        return (
            source_ids,
            source_mask,
            target_ids,
            target_mask,
            truncated_output_probabilities,
            source_str,
            target_str,
        )

    def collate_sentences(self, sentences: list) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = [sentence.size(0) for sentence in sentences]
        if self.max_length is not None:
            max_length = self.max_length
        else:
            max_length = max(lengths)
        if max_length == 1:  # Some samples are only one token
            max_length += 1

        subword_ids = torch.stack(
            [
                F.pad(sentence, (0, max_length - length), value=self.pad_id)
                for length, sentence in zip(lengths, sentences)
            ]
        )
        attention_mask = subword_ids == self.pad_id
        return subword_ids, attention_mask
