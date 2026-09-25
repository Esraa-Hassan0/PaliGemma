import os
import torch
from modeling_gemma import PaliGemmaConfig, PaliGemmaForConditionalGeneration, KVCache
from variational_autoencoder import PaliGemmaVAEDecoder
from utils import load_vae_weights

def test_vae():
    print("Testing VAE decoder...")
    vae = PaliGemmaVAEDecoder(vocab_size=128, embed_dim=512)
    dummy_input = torch.randint(0, 128, (1, 16))
    out = vae(dummy_input)
    assert out.shape == (1, 1, 64, 64), f"Expected shape (1, 1, 64, 64), got {out.shape}"
    print("VAE decoder shape test PASSED!")

    # Test loading actual weights if directory exists
    if os.path.exists("data/vae-oid"):
        print("Testing loading actual VAE weights from data/vae-oid...")
        load_vae_weights(vae, "data/vae-oid")
        mask = vae(dummy_input)
        assert mask.shape == (1, 1, 64, 64)
        assert (mask >= 0.0).all() and (mask <= 1.0).all()
        print("VAE weight loading and forward test PASSED!")

def test_model_masking_and_forward():
    print("Testing PaliGemma Config and Forward pass...")
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
        }
    )
    
    model = PaliGemmaForConditionalGeneration(config)
    
    # Test prefill phase
    batch_size = 1
    seq_len = 256 + 10 # 256 img tokens + 10 text tokens
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
    
    logits = outputs["logits"]
    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"Unexpected logits shape {logits.shape}"
    print(f"Prefill forward logits shape: {logits.shape} - PASSED!")
    
    # Test decode phase step with explicit attention mask
    next_input_ids = torch.randint(0, 1000, (batch_size, 1))
    next_attention_mask = torch.ones((batch_size, seq_len + 1), dtype=torch.int64)
    decode_outputs = model(
        input_ids=next_input_ids,
        pixel_values=None,
        attention_mask=next_attention_mask,
        kv_cache=kv_cache,
    )
    assert decode_outputs["logits"].shape == (batch_size, 1, config.vocab_size)
    print("Decode step forward (with attention mask) - PASSED!")

    # Test decode phase step with attention_mask=None
    next_input_ids2 = torch.randint(0, 1000, (batch_size, 1))
    decode_outputs2 = model(
        input_ids=next_input_ids2,
        pixel_values=None,
        attention_mask=None,
        kv_cache=kv_cache,
    )
    assert decode_outputs2["logits"].shape == (batch_size, 1, config.vocab_size)
    print("Decode step forward (with attention_mask=None) - PASSED!")

if __name__ == "__main__":
    test_vae()
    test_model_masking_and_forward()
    print("ALL COMPONENT TESTS PASSED!")
