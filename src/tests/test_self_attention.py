import pytest
import torch

from src.self_attention import ScaledDotProductSelfAttentionHead, MultiHeadAttention

@pytest.mark.parametrize("block_size", [8, 12])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
@pytest.mark.parametrize("dim_tok_emb", [256, 512])
@pytest.mark.parametrize("head_size", [32, 64])
def test_selfattentionhead(dropout_rate, block_size, dim_tok_emb, head_size):
    sa = ScaledDotProductSelfAttentionHead(
        dim_token_embedding=dim_tok_emb,
        head_size=head_size,
        max_block_size=12,
        dropout_rate=dropout_rate,
    )
    batch_size = 16
    x = torch.randn(batch_size, block_size, dim_tok_emb)
    out = sa(x)
    assert out.shape == (batch_size, block_size, head_size)


@pytest.mark.parametrize("block_size", [8, 12])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
@pytest.mark.parametrize("dim_tok_emb", [256, 512])
@pytest.mark.parametrize("nb_heads", [6, 12])
@pytest.mark.parametrize("head_size", [64, 128])
def test_multiheadattention(block_size, dropout_rate, dim_tok_emb, nb_heads, head_size):
    ma = MultiHeadAttention(
        nb_heads=nb_heads, 
        dim_token_embedding=dim_tok_emb,
        head_size=head_size,
        max_block_size=12,
        dropout_rate=dropout_rate
    )
    batch_size = 16
    x = torch.randn(batch_size, block_size, dim_tok_emb)
    out = ma(x)
    assert out.shape == (batch_size, block_size, nb_heads*head_size)

