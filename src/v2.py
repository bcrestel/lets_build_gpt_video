# n_gram.py

from logging import getLogger
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional

from src.text_processor import TextProcessor

logger = getLogger(__name__)


class BiGram(nn.Module):
    def __init__(self, vocab_size: int, dim_token_embedding: int, block_size: int):
        """A bigram model

        Args:
            vocab_size (int): total number of tokens
            dim_token_embedding (int): dimension of the token embeddings, i.e., subspace to represent the tokens
            block_size (int): maximum number of tokens in a model input
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.dim_token_embedding = dim_token_embedding
        self.block_size = block_size
        self.embedding = nn.Embedding(self.vocab_size, dim_token_embedding)
        self.map_token_embedding_to_token = nn.Linear(
            self.dim_token_embedding, self.vocab_size
        )
        self.positional_embedding = nn.Embedding(
            self.block_size, self.dim_token_embedding
        )

    def forward(self, token_idx: torch.Tensor) -> torch.Tensor:
        """Forward pass

        Args:
            token_idx (torch.Tensor): idx of the input token; token_idx.shape = (B, block_size)
                B = batch_size
                block_size = T in original code

        Returns:
            torch.Tensor: logits for the model prediction; has shape (B, self.block_size, self.vocab_size)
        """
        pos_input = torch.arange(self.block_size, device=self.device)
        positional_embeddings = self.embedding(
            pos_input
        )  # shape (block_size, self.dim_token_embedding)
        token_embeddings = self.embedding(
            token_idx
        )  # shape (B, block_size, self.dim_token_embedding)
        input_embeddings = token_embeddings + positional_embeddings
        logits = self.map_token_embedding_to_token(
            input_embeddings
        )  # shape (B, block_size, self.vocab_size)
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
        learning_rate: float = 1e-2,
        nb_batch_eval: int = 200,
        eval_period: int = 300,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        for ep in range(nb_epochs):
            optimizer.zero_grad()
            x_train, y_train = text.get_batch(
                batch_size=batch_size, block_size=self.block_size, split="train"
            )
            x_train = x_train.to(self.device)
            y_train = y_train.to(self.device)
            loss = self.loss(logits=self.forward(x_train), y=y_train)
            loss.backward()
            optimizer.step()

            # Estimate train and validation loss
            if ep % eval_period == 0:
                with torch.no_grad():
                    loss_split = {}
                    for split in ["train", "val"]:
                        _all_losses = []
                        for bb in range(nb_batch_eval):
                            x_, y_ = text.get_batch(
                                batch_size=batch_size,
                                block_size=self.block_size,
                                split=split,
                            )
                            x_ = x_train.to(self.device)
                            y_ = y_train.to(self.device)
                            loss_ = self.loss(logits=self.forward(x_), y=y_)
                            _all_losses.append(loss_.item())
                        loss_split[split] = sum(_all_losses) / (nb_batch_eval)
                    logger.info(
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
