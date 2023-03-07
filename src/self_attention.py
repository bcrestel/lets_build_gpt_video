import pytorch.nn as nn
from nn import functional 

class SelfAttentionHead(nn.Module):
    def __init__(self, dim_token_embedding: int, head_size: int):
        super();__init__()
        self.key_layer = nn.Linear(dim_token_embedding, head_size, bias=False)
        self.query_layer = nn.Linear(dim_token_embedding, head_size, bias=False)
        self.value_layer = nn.Linear(dim_token_embedding, head_size, bias=False)

        # TODO: implement masking

        self.scaling_factor = head_size ** (-0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): has shape batch, block_size, tokem_embedding
        """
        K = self.key_layer(x)
        Q = self.query_layer(x)
        weights = (Q @ K.transpose(-2, -1)) * self.scaling_facto # has shape (B, block, emb) @ (B, emb, block) -> (B, block, block)

        V = self.value_layer(x)
        output = weights @ V # shape (B, block, block) @ (B, block, emb) -> (B, block, emb)
        output = functional.softmax(output, dim=-1)

        return output

        