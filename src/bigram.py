# n_gram.py

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional

from src.text_processor import TextProcessor


class BiGram(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.vocab_size)

    def forward(self, x: torch.Tensor):
        """x.shape = (B, T)"""
        logits = self.embedding(x)  # shape (B, T, self.vocab_size)
        return logits

    def inference(self, idx: torch.Tensor) -> torch.Tensor:
        """Generate a new index given idx (shape 1)"""
        logits = self.embedding(idx)
        probs = functional.softmax(logits)
        return torch.multinomial(probs, num_samples=1).flatten()

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
        learning_rate: float = 1e-2,
        eval_interval: int = 100,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        for ep in range(nb_epochs):
            optimizer.zero_grad()
            x_train, y_train = text.get_batch(
                batch_size=batch_size, block_size=block_size, split="train"
            )
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            loss = self.loss(logits=self.forward(x_train), y=y_train)
            loss.backward()
            optimizer.step()

            # Estimate train and validation loss
            if ep % eval_interval == 0:
                with torch.no_grad():
                    loss_split = {}
                    for split in ["train", "val"]:
                        text_iterator = text.iterator_all(
                            batch_size=batch_size,
                            split=split,
                        )
                        _all_losses = [
                            self.loss(self.forward(bb[0]), bb[1]).item()
                            for bb in text_iterator
                        ]
                        loss_split[split] = sum(_all_losses)
                    print(
                        f"Epoch {ep}: train_loss = {loss_split['train']}, eval_loss = {loss_split['val']}"
                    )

    def generate(self, max_nb_tokens: int, idx: torch.Tensor):
        idx = idx.to(self.device)
        token_idx = [idx.item()]
        for ii in range(max_nb_tokens):
            new_idx = self.inference(idx)
            token_idx.append(new_idx.item())
            idx = new_idx
        return token_idx

    @property
    def device(self):
        return next(self.parameters()).device
