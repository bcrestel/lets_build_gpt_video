import torch.nn as nn

from src.feed_forward_network import FFW
from src.self_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        nb_heads: int,
        dim_token_embedding: int,
        head_size: int,
        max_block_size: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.sa_head = MultiHeadAttention(
            nb_heads=nb_heads,
            dim_token_embedding=dim_token_embedding,
            head_size=head_size,
            max_block_size=max_block_size,
            dropout_rate=dropout_rate,
        )
        self.linear = FFW(
            dim_token_embedding=dim_token_embedding, dropout_rate=dropout_rate
        )
        self.ln1 = nn.LayerNorm(dim_token_embedding)
        self.ln2 = nn.LayerNorm(dim_token_embedding)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, T, C)
        Returns:
            torch.Tensor: shape (B, T, C)
        """
        x = x + self.sa_head(self.ln1(x))
        x = x + self.linear(self.ln2(x))
        return x
