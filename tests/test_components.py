import os

import torch

from modeling_gemma import KVCache, PaliGemmaConfig, PaliGemmaForConditionalGeneration
from utils import load_vae_weights
from variational_autoencoder import PaliGemmaVAEDecoder


def test_vae():
    vae = PaliGemmaVAEDecoder(vocab_size=128, embed_dim=512)
    dummy_input = torch.randint(0, 128, (1, 16))
    out = vae(dummy_input)
    assert out.shape == (1, 1, 64, 64)

    if os.path.exists("data/vae-oid"):
        load_vae_weights(vae, "data/vae-oid")
        mask = vae(dummy_input)
        assert mask.shape == (1, 1, 64, 64)
        assert (mask >= 0.0).all() and (mask <= 1.0).all()


def test_model_masking_and_forward():
    config = PaliGemmaConfig(
        vocab_size=257152,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=64,
        vision_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 224,
            "patch_size": 14,
            "projection_dim": 256,
            "num_img_tokens": 256,
        },
    )

    model = PaliGemmaForConditionalGeneration(config)
    batch_size = 1
    seq_len = 256 + 10
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    dummy_input_ids[0, :256] = config.image_token_id
    dummy_pixel_values = torch.randn(batch_size, 3, 224, 224)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64)

    kv_cache = KVCache()
    outputs = model(
        input_ids=dummy_input_ids,
        pixel_values=dummy_pixel_values,
        attention_mask=dummy_attention_mask,
        kv_cache=kv_cache,
    )
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

    next_input_ids = torch.randint(0, 1000, (batch_size, 1))
    next_attention_mask = torch.ones((batch_size, seq_len + 1), dtype=torch.int64)
    decode_outputs = model(
        input_ids=next_input_ids,
        pixel_values=None,
        attention_mask=next_attention_mask,
        kv_cache=kv_cache,
    )
    assert decode_outputs["logits"].shape == (batch_size, 1, config.vocab_size)

    next_input_ids = torch.randint(0, 1000, (batch_size, 1))
    decode_outputs = model(
        input_ids=next_input_ids,
        pixel_values=None,
        attention_mask=None,
        kv_cache=kv_cache,
    )
    assert decode_outputs["logits"].shape == (batch_size, 1, config.vocab_size)