"""
Simple and reliable GPT-2 weight loader using Hugging Face.
This downloads directly from Hugging Face and converts to the format needed.
"""

import os
import json
import torch
from transformers import GPT2LMHeadModel


def load_gpt2_from_hf(model_size="124M", save_dir="gpt2_hf"):
    """
    Load GPT-2 weights from Hugging Face and save in a compatible format.
    
    Args:
        model_size: "124M", "355M", "774M", or "1558M"
        save_dir: Directory to save the converted model
    
    Returns:
        settings: Model configuration dictionary
        params: Model parameters dictionary
    """
    # Map model sizes to Hugging Face model names
    size_to_hf = {
        "124M": "gpt2",
        "355M": "gpt2-medium", 
        "774M": "gpt2-large",
        "1558M": "gpt2-xl"
    }
    
    if model_size not in size_to_hf:
        raise ValueError(f"Model size must be one of {list(size_to_hf.keys())}")
    
    hf_model_name = size_to_hf[model_size]
    
    print(f"Downloading {hf_model_name} from Hugging Face...")
    print("This may take a few minutes on first download...")
    
    # Download model from Hugging Face
    model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    config = model.config
    
    print("Model downloaded successfully!")
    print(f"Config: vocab_size={config.vocab_size}, n_layer={config.n_layer}, n_embd={config.n_embd}")
    
    # Create settings dictionary
    settings = {
        "n_vocab": config.vocab_size,
        "n_ctx": config.n_positions,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer
    }
    
    # Convert to parameter format
    print("Converting parameters...")
    params = hf_to_custom_params(model, settings)
    
    # Save to disk
    model_dir = os.path.join(save_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save settings
    settings_path = os.path.join(model_dir, "hparams.json")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    
    # Save parameters
    params_path = os.path.join(model_dir, "params.pt")
    torch.save(params, params_path)
    
    print(f"Model saved to {model_dir}")
    
    return settings, params


def hf_to_custom_params(hf_model, settings):
    """Convert Hugging Face model to custom parameter format."""
    sd = hf_model.state_dict()
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    
    # Embeddings
    params["wte"] = sd["transformer.wte.weight"].cpu().numpy()
    params["wpe"] = sd["transformer.wpe.weight"].cpu().numpy()
    
    # Final layer norm
    params["ln_f"] = {
        "g": sd["transformer.ln_f.weight"].cpu().numpy(),
        "b": sd["transformer.ln_f.bias"].cpu().numpy()
    }
    
    # Transformer blocks
    for i in range(settings["n_layer"]):
        prefix = f"transformer.h.{i}"
        block = {}
        
        # Layer norm 1
        block["ln_1"] = {
            "g": sd[f"{prefix}.ln_1.weight"].cpu().numpy(),
            "b": sd[f"{prefix}.ln_1.bias"].cpu().numpy()
        }
        
        # Attention
        block["attn"] = {
            "c_attn": {
                "w": sd[f"{prefix}.attn.c_attn.weight"].cpu().numpy(),
                "b": sd[f"{prefix}.attn.c_attn.bias"].cpu().numpy()
            },
            "c_proj": {
                "w": sd[f"{prefix}.attn.c_proj.weight"].cpu().numpy(),
                "b": sd[f"{prefix}.attn.c_proj.bias"].cpu().numpy()
            }
        }
        
        # Layer norm 2
        block["ln_2"] = {
            "g": sd[f"{prefix}.ln_2.weight"].cpu().numpy(),
            "b": sd[f"{prefix}.ln_2.bias"].cpu().numpy()
        }
        
        # MLP
        block["mlp"] = {
            "c_fc": {
                "w": sd[f"{prefix}.mlp.c_fc.weight"].cpu().numpy(),
                "b": sd[f"{prefix}.mlp.c_fc.bias"].cpu().numpy()
            },
            "c_proj": {
                "w": sd[f"{prefix}.mlp.c_proj.weight"].cpu().numpy(),
                "b": sd[f"{prefix}.mlp.c_proj.bias"].cpu().numpy()
            }
        }
        
        params["blocks"][i] = block
    
    return params


if __name__ == "__main__":
    # Test download
    settings, params = load_gpt2_from_hf("124M")
    print("\n✓ Success!")
    print(f"Settings: {settings}")
    print(f"Parameter keys: {list(params.keys())}")
