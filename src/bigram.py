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

    def train(
        self, 
        nb_epochs: int, 
        text: TextProcessor, 
        batch_size: int = 32, 
        block_size: int = 8,
        learning_rate: float = 1e-2
        ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        for ep in range(nb_epochs):
            print(f"Epoch {ep}")
            optimizer.zero_grad()
            x_train, y_train = text.get_batch(
                batch_size=batch_size, 
                block_size=block_size,
                split="train"
                )
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            loss = self.loss(logits=self.forward(x_train), y=y_train)
            loss.grad()
            optimizer.step()

            # Estimate train and validation loss
            # TODO: implement deterministic batch implementation of train and validation losses
            #with torch.no_grad():
                

