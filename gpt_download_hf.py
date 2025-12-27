# Alternative GPT-2 weight downloader using Hugging Face
# This is a more reliable alternative to downloading from OpenAI's servers

import os
import json
import numpy as np
import torch
from transformers import GPT2LMHeadModel
from tqdm import tqdm


def download_and_load_gpt2_hf(model_size="124M", models_dir="gpt2"):
    """
    Download GPT-2 weights from Hugging Face and convert to the expected format.
    
    Args:
        model_size: Size of the model ("124M", "355M", "774M", "1558M")
        models_dir: Directory to save the model
    
    Returns:
        settings: Model configuration dictionary
        params: Model parameters dictionary
    """
    # Map model sizes to Hugging Face model names
    size_to_hf_model = {
        "124M": "gpt2",
        "355M": "gpt2-medium",
        "774M": "gpt2-large",
        "1558M": "gpt2-xl"
    }
    
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")
    
    hf_model_name = size_to_hf_model[model_size]
    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Downloading GPT-2 {model_size} from Hugging Face...")
    
    # Download the model from Hugging Face (force download from HF hub, not local)
    model = GPT2LMHeadModel.from_pretrained(hf_model_name, cache_dir=None)
    config = model.config
    
    # Create settings dictionary matching the expected format
    settings = {
        "n_vocab": config.vocab_size,
        "n_ctx": config.n_positions,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer
    }
    
    # Save settings to hparams.json
    hparams_path = os.path.join(model_dir, "hparams.json")
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    
    print("Converting model parameters...")
    
    # Convert Hugging Face model to the expected parameter format
    params = convert_hf_to_params(model, settings)
    
    # Save the converted parameters as a PyTorch checkpoint
    checkpoint_path = os.path.join(model_dir, "model.pth")
    torch.save(params, checkpoint_path)
    
    print(f"Model successfully downloaded and saved to {model_dir}")
    
    return settings, params


def convert_hf_to_params(hf_model, settings):
    """
    Convert Hugging Face GPT-2 model to the parameter format expected by the custom implementation.
    
    Args:
        hf_model: Hugging Face GPT2LMHeadModel
        settings: Model configuration dictionary
    
    Returns:
        params: Dictionary containing model parameters
    """
    state_dict = hf_model.state_dict()
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    
    # Token and position embeddings
    params["wte"] = state_dict["transformer.wte.weight"].cpu().numpy()
    params["wpe"] = state_dict["transformer.wpe.weight"].cpu().numpy()
    
    # Final layer norm
    params["ln_f"] = {
        "g": state_dict["transformer.ln_f.weight"].cpu().numpy(),
        "b": state_dict["transformer.ln_f.bias"].cpu().numpy()
    }
    
    # Process each transformer block
    for i in range(settings["n_layer"]):
        block_prefix = f"transformer.h.{i}"
        block_params = {}
        
        # Layer norm 1 (before attention)
        block_params["ln_1"] = {
            "g": state_dict[f"{block_prefix}.ln_1.weight"].cpu().numpy(),
            "b": state_dict[f"{block_prefix}.ln_1.bias"].cpu().numpy()
        }
        
        # Attention weights
        # HF uses a single c_attn for Q, K, V projections
        c_attn_weight = state_dict[f"{block_prefix}.attn.c_attn.weight"].cpu().numpy()
        c_attn_bias = state_dict[f"{block_prefix}.attn.c_attn.bias"].cpu().numpy()
        
        block_params["attn"] = {
            "c_attn": {
                "w": c_attn_weight,
                "b": c_attn_bias
            },
            "c_proj": {
                "w": state_dict[f"{block_prefix}.attn.c_proj.weight"].cpu().numpy(),
                "b": state_dict[f"{block_prefix}.attn.c_proj.bias"].cpu().numpy()
            }
        }
        
        # Layer norm 2 (before MLP)
        block_params["ln_2"] = {
            "g": state_dict[f"{block_prefix}.ln_2.weight"].cpu().numpy(),
            "b": state_dict[f"{block_prefix}.ln_2.bias"].cpu().numpy()
        }
        
        # MLP weights
        block_params["mlp"] = {
            "c_fc": {
                "w": state_dict[f"{block_prefix}.mlp.c_fc.weight"].cpu().numpy(),
                "b": state_dict[f"{block_prefix}.mlp.c_fc.bias"].cpu().numpy()
            },
            "c_proj": {
                "w": state_dict[f"{block_prefix}.mlp.c_proj.weight"].cpu().numpy(),
                "b": state_dict[f"{block_prefix}.mlp.c_proj.bias"].cpu().numpy()
            }
        }
        
        params["blocks"][i] = block_params
    
    return params


if __name__ == "__main__":
    # Test the download
    settings, params = download_and_load_gpt2_hf("124M", "gpt2")
    print("\nSettings:", settings)
    print("\nParameter keys:", params.keys())
    print("Number of blocks:", len(params["blocks"]))
