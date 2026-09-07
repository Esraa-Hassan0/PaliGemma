import os
import json
import glob
from typing import Tuple
from safetensors import safe_open
from transformers import AutoTokenizer
from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig

def load_hf_model(model_pth: str, device:str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:

    tokenizer = AutoTokenizer.from_pretrained(model_pth, padding_side="right")
    assert tokenizer.padding_side == "right"

    safetensors_files = glob.glob(os.path.join(model_pth, "*.safetensors"))

    tensors = {}

    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework ="pt", device= "cuda") as file:
            for key in file.keys():
                tensors[key] = file.get_tensor(key)

    with open(os.path.join(model_pth, "config.json")) as f:
        model_config = json.load(f)
        config = PaliGemmaConfig(**model_config)

    model = PaliGemmaForConditionalGeneration(config=config).to(device)

    model.load_state_dict(tensors, strict=False)

    model.tie_weights()
    
    return (model, tokenizer)