import pytest
import torch

from src.v2 import LanguageModel


@pytest.mark.parametrize("block_size", [8, 12])
@pytest.mark.parametrize("dropout_rate", [0.0, 0.2])
@pytest.mark.parametrize("nb_heads_per_block", [8, 16])
def test_v2(dropout_rate, block_size, nb_heads_per_block):
    vocab_size = 65
    dim_tok_emb = 512
    nb_decoder_blocks = 2
    model = LanguageModel(
        vocab_size=vocab_size,
        dim_token_embedding=dim_tok_emb,
        block_size=block_size,
        nb_decoder_blocks=nb_decoder_blocks,
        nb_heads_per_block=nb_heads_per_block,
        dropout_rate=dropout_rate,
    )
    batch_size = 16
    x = torch.randint(vocab_size, (batch_size, block_size))
    out = model(x)
    assert out.shape == (batch_size, block_size, vocab_size)