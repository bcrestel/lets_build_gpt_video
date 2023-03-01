# text_processor.py

from pathlib import Path
from typing import Iterator, List, Optional, Union

import torch


class TextProcessor:
    """Parse a text and provide a few functionalities to work with text"""

    def __init__(self, input_path: Union[str, Path], type: Optional[str] = None):
        # process input_path
        self.input_path = (
            input_path if isinstance(input_path, Path) else Path(input_path)
        )
        if self.input_path.suffix == ".txt" or type == "str":
            self.text = self.read_txt(self.input_path)
        else:
            raise NotImplementedError
        # generate properties about text
        self.all_chars = sorted(list(set(self.text)))
        self._nb_unique_chars = len(self.all_chars)
        # map characters to unique integer
        self.map_string2integer = {ch: i for i, ch in enumerate(self.all_chars)}
        self.map_integer2string = {i: ch for i, ch in enumerate(self.all_chars)}
        # prepare training dataset
        self.data = torch.tensor(
            self.convert_string2integer(self.text), dtype=torch.long
        )
        len_data = len(self.data)
        len_data_train = int(0.9 * len_data)
        self.data_train = self.data[:len_data_train]
        self.data_val = self.data[len_data_train:]

    def convert_string2integer(self, string: str):
        """encoder: take a string, output a list of integers"""
        return [self.map_string2integer[c] for c in string]

    def convert_integer2string(self, integers: List[str]):
        """decoder: take a list of integers, output a string"""
        return "".join([self.map_integer2string[i] for i in integers])

    @property
    def vocab_size(self):
        return self._nb_unique_chars

    @staticmethod
    def read_txt(input_path: Path):
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def get_batch(self, batch_size: int, block_size: int, split: str = "train"):
        """
        Generate a small batch of data of inputs x and targets y

        Arguments:
            batch_size:
            block_size: also called context_length. Max length of tokens a language model can use to generate prediction. For Transformers, they must predict from any context length ranging from 1 to block_size. For a bigram model, it doesn't make any difference.
            split: train or val
        """
        data = self.data_train if split == "train" else self.data_val
        idx = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in idx])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in idx])
        return x, y

    # TODO: Add block_size to iterator_all as this is required by v2.py
    # should be the same as get_batch but in deterministic manner
    def iterator_all(self, batch_size: int, split: str) -> Iterator:
        data = self.data_train if split == "train" else self.data_val
        ii = 0
        for ii in range(0, len(data) - batch_size + 1, batch_size):
            yield data[ii : ii + batch_size], data[ii + 1 : ii + batch_size + 1]
