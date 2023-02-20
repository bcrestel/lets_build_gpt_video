# n_gram.py

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional

from src.text_processor import TextProcessor


class BiGram(nn.Module):
    def __init__(self, vocab_size: int, device: Optional[torch.device] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)
        self.device = ("cuda" if torch.cuda.is_available() else 'cpu') if device is None else device

    def forward(self, x: torch.Tensor):
        """x.shape = (B, T)"""
        logits = self.embedding(x)  # shape (B, T, self.vocab_size)
        return logits

    def loss(self, logits: torch.Tensor, y: torch.Tensor):
        """
        logits.shape = (B, T, vocab_size)
        y.shape = (B, T)
        """
        logits = logits.view(-1, self.vocab_size)
        y = y.view(-1)
        return functional.cross_entropy(logits, y)

    # TODO: Continuere here
    #def train(self, epochs: int):
