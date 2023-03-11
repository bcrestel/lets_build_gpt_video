import pytest
import torch

from src.self_attention import ScaledDotProductSelfAttentionHead

@pytest.mark.parametrize("block_size", [8, 12])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
def test_selfattentionhead(dropout_rate, block_size):
    sa = ScaledDotProductSelfAttentionHead(
        dim_token_embedding=512,
        head_size=64,
        max_block_size=12,
        dropout_rate=dropout_rate,
    )
    x = torch.randn(32, block_size, 512)
    out = sa(x)
    assert out.shape == (32, block_size, 64)


def test_multiheadattention():
