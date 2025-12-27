"""
Complete solution to load GPT-2 pretrained weights from Hugging Face
and convert them to the format your notebook expects.

This replaces the broken gpt_download.py script.
"""

from transformers import GPT2LMHeadModel
import numpy as np


def download_and_load_gpt2(model_size="124M", models_dir="gpt2"):
    """
    Download GPT-2 weights from Hugging Face and convert to expected format.
    
    This function has the SAME signature as the original gpt_download.py
    so you can use it as a drop-in replacement.
    
    Args:
        model_size: "124M", "355M", "774M", or "1558M"
        models_dir: Directory name (not used, kept for compatibility)
    
    Returns:
        settings: Model configuration dict
        params: Model parameters dict with structure:
            - wte: token embeddings
            - wpe: position embeddings  
            - blocks: list of transformer blocks
            - ln_f: final layer norm
    """
    # Map sizes to Hugging Face model names
    size_to_hf = {
        "124M": "gpt2",
        "355M": "gpt2-medium",
        "774M": "gpt2-large", 
        "1558M": "gpt2-xl"
    }
    
    if model_size not in size_to_hf:
        raise ValueError(f"Model size not in {list(size_to_hf.keys())}")
    
    hf_model_name = size_to_hf[model_size]
    
    print(f"Downloading GPT-2 {model_size} from Hugging Face...")
    print("(This may take a few minutes on first download)")
    
    # Download model from Hugging Face
    model = GPT2LMHeadModel.from_pretrained(hf_model_name)
    config = model.config
    
    print(f"✓ Model downloaded successfully!")
    
    # Create settings dict (same format as original)
    settings = {
        "n_vocab": config.vocab_size,
        "n_ctx": config.n_positions,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer
    }
    
    print(f"Converting parameters...")
    print(f"  - Vocab size: {settings['n_vocab']}")
    print(f"  - Layers: {settings['n_layer']}")
    print(f"  - Embedding dim: {settings['n_embd']}")
    
    # Convert to params dict (same format as original)
    params = convert_hf_to_params(model, settings)
    
    print(f"✓ Conversion complete!")
    
    return settings, params


def convert_hf_to_params(hf_model, settings):
    """
    Convert Hugging Face model to the parameter format expected by
    the custom GPT implementation.
    
    Returns dict with structure matching TensorFlow checkpoint format:
        params = {
            "wte": token_embeddings,
            "wpe": position_embeddings,
            "ln_f": {"g": gamma, "b": beta},
            "blocks": [
                {
                    "ln_1": {"g": gamma, "b": beta},
                    "attn": {
                        "c_attn": {"w": weight, "b": bias},
                        "c_proj": {"w": weight, "b": bias}
                    },
                    "ln_2": {"g": gamma, "b": beta},
                    "mlp": {
                        "c_fc": {"w": weight, "b": bias},
                        "c_proj": {"w": weight, "b": bias}
                    }
                },
                ...
            ]
        }
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
        prefix = f"transformer.h.{i}"
        block = {}
        
        # Layer norm 1 (before attention)
        block["ln_1"] = {
            "g": state_dict[f"{prefix}.ln_1.weight"].cpu().numpy(),
            "b": state_dict[f"{prefix}.ln_1.bias"].cpu().numpy()
        }
        
        # Attention weights
        # Note: HF uses a single c_attn for Q, K, V projections
        block["attn"] = {
            "c_attn": {
                "w": state_dict[f"{prefix}.attn.c_attn.weight"].cpu().numpy(),
                "b": state_dict[f"{prefix}.attn.c_attn.bias"].cpu().numpy()
            },
            "c_proj": {
                "w": state_dict[f"{prefix}.attn.c_proj.weight"].cpu().numpy(),
                "b": state_dict[f"{prefix}.attn.c_proj.bias"].cpu().numpy()
            }
        }
        
        # Layer norm 2 (before MLP)
        block["ln_2"] = {
            "g": state_dict[f"{prefix}.ln_2.weight"].cpu().numpy(),
            "b": state_dict[f"{prefix}.ln_2.bias"].cpu().numpy()
        }
        
        # MLP weights
        block["mlp"] = {
            "c_fc": {
                "w": state_dict[f"{prefix}.mlp.c_fc.weight"].cpu().numpy(),
                "b": state_dict[f"{prefix}.mlp.c_fc.bias"].cpu().numpy()
            },
            "c_proj": {
                "w": state_dict[f"{prefix}.mlp.c_proj.weight"].cpu().numpy(),
                "b": state_dict[f"{prefix}.mlp.c_proj.bias"].cpu().numpy()
            }
        }
        
        params["blocks"][i] = block
    
    return params


# Test function
if __name__ == "__main__":
    print("Testing GPT-2 weight download...")
    settings, params = download_and_load_gpt2("124M")
    
    print("\n" + "="*60)
    print("SUCCESS!")
    print("="*60)
    print(f"\nSettings: {settings}")
    print(f"\nParams keys: {list(params.keys())}")
    print(f"Number of blocks: {len(params['blocks'])}")
    print(f"Token embedding shape: {params['wte'].shape}")
    print(f"Position embedding shape: {params['wpe'].shape}")
