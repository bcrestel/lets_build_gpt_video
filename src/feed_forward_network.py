import torch.nn as nn


class FFW(nn.Module):
    def __init__(self, dim_token_embedding: int, dropout_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            self.Linear(dim_token_embedding, 4 * dim_token_embedding),
            nn.ReLU(),
            nn.Linear(4 * dim_token_embedding, dim_token_embedding),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)
