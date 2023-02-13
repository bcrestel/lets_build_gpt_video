# text_processor.py

from pathlib import Path
from typing import List, Optional, Union


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
        self.chars = sorted(list(set(self.text)))
        self._len_chars = len(self.chars)
        # map characters to unique integer
        self.map_string2integer = {ch: i for i, ch in enumerate(self.chars)}
        self.map_integer2string = {i: ch for i, ch in enumerate(self.chars)}

    def convert_string2integer(self, string: str):
        """encoder: take a string, output a list of integers"""
        return [self.map_string2integer[c] for c in string]

    def convert_integer2string(self, integers: List[str]):
        """decoder: take a list of integers, output a string"""
        return "".join([self.map_integer2string[i] for i in integers])

    @property
    def vocab_size(self):
        return self._len_chars

    @staticmethod
    def read_txt(input_path: Path):
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
