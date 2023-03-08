import torch
import torch.nn as nn
from torch.nn import functional 

class ScaledDotProductSelfAttentionHead(nn.Module):
    def __init__(self, dim_token_embedding: int, head_size: int, max_block_size: int, dropout_rate: float = 0.0):
        """_summary_

        Args:
            dim_token_embedding (int): 
            head_size (int): dimensions of K, Q, V
            max_block_size (int): 
            dropout_rate (float, optional): Apply dropout to the attention weights. Defaults to 0.0 (no dropout). Note dropout rescale entries at training time
        """
        super().__init__()
        self.key_layer = nn.Linear(dim_token_embedding, head_size, bias=False)
        self.query_layer = nn.Linear(dim_token_embedding, head_size, bias=False)
        self.value_layer = nn.Linear(dim_token_embedding, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(max_block_size, max_block_size)))
        self.scaling_factor = head_size ** (-0.5)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): has shape batch, block_size, token_embedding
        """
        _, block_size, _ = x.shape
        K = self.key_layer(x)
        Q = self.query_layer(x)
        weights = (Q @ K.transpose(-2, -1)) * self.scaling_factor # has shape (B, block, emb) @ (B, emb, block) -> (B, block, block)
        mask = self.mask[:block_size, :block_size]
        weights.masked_fill_(mask==0, float("-inf"))
        weights = functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        V = self.value_layer(x)
        output = weights @ V # shape (B, block, block) @ (B, block, emb) -> (B, block, emb)
        return output
