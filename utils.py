import os
import json
import glob
from typing import Tuple
import torch
import numpy as np
from safetensors import safe_open
from transformers import AutoTokenizer
from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig


def load_hf_model(
    model_pth: str, device: str
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(model_pth, padding_side="right")
    assert tokenizer.padding_side == "right"

    safetensors_files = glob.glob(os.path.join(model_pth, "*.safetensors"))

    if not safetensors_files:
        raise FileNotFoundError(
            f"No .safetensors checkpoint files found in '{model_pth}'."
        )

    tensors = {}

    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as file:
            for key in file.keys():
                tensors[key] = file.get_tensor(key)

    with open(os.path.join(model_pth, "config.json")) as f:
        model_config = json.load(f)
        config = PaliGemmaConfig(**model_config)

    # Initialize model on CPU or target device
    model = PaliGemmaForConditionalGeneration(config=config).to(device="cpu")

    if "language_model.lm_head.weight" not in tensors:
        tensors["language_model.lm_head.weight"] = tensors[
            "language_model.model.embed_tokens.weight"
        ]

    model.load_state_dict(tensors, strict=True)
    del tensors

    model = model.to(device=device)
    if device == "cuda":
        torch.cuda.empty_cache()

    model.tie_weights()

    return (model, tokenizer)


def load_vae_weights(vae_pytorch_model, weights_path: str):
    """
    Loads VAE decoder weights from either an extracted directory of .npy files
    (e.g., data/vae-oid) or a consolidated .npz checkpoint file.
    """
    state_dict = {}

    if os.path.isdir(weights_path):
        npy_files = glob.glob(os.path.join(weights_path, "*.npy"))
        if not npy_files:
            raise FileNotFoundError(
                f"No .npy files found in directory '{weights_path}'"
            )
        for f in npy_files:
            name = os.path.basename(f)
            if name == "_vq_vae._embedding.npy":
                state_dict["embedding.weight"] = torch.from_numpy(np.load(f))
            elif name.startswith("decoder.") and name.endswith(".npy"):
                key = name[:-4]  # strip .npy extension
                state_dict[key] = torch.from_numpy(np.load(f))

    elif weights_path.endswith(".npz"):
        weights = np.load(weights_path)
        if "quantizer/codebook" in weights:
            state_dict["embedding.weight"] = torch.from_numpy(
                weights["quantizer/codebook"]
            )
        elif "_vq_vae._embedding" in weights:
            state_dict["embedding.weight"] = torch.from_numpy(
                weights["_vq_vae._embedding"]
            )
        for k in weights.files:
            if k.startswith("decoder."):
                state_dict[k] = torch.from_numpy(weights[k])
    else:
        raise ValueError(f"Unsupported weights path format: '{weights_path}'")

    if state_dict:
        # Cast loaded tensors to model dtype and device
        target_device = vae_pytorch_model.embedding.weight.device
        target_dtype = vae_pytorch_model.embedding.weight.dtype
        state_dict = {
            k: v.to(device=target_device, dtype=target_dtype)
            for k, v in state_dict.items()
        }
        # Load weights into model (strict if full decoder weights present, else non-strict)
        strict = len(state_dict) >= len(vae_pytorch_model.state_dict())
        vae_pytorch_model.load_state_dict(state_dict, strict=strict)
        print(f"Successfully loaded VAE weights ({len(state_dict)} tensors)!")
    else:
        raise RuntimeError(
            f"Could not extract any valid VAE tensors from '{weights_path}'"
        )

    return vae_pytorch_model
