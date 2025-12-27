# Alternative GPT-2 Weight Download Instructions

## Problem
The original `gpt_download.py` script fails with the error:
```
An unexpected error occurred: Downloaded file size does not match remote Content-Length
```

This happens because OpenAI's blob storage servers are unreliable or have changed.

## Solution: Use Hugging Face

I've created an alternative download method using Hugging Face's transformers library, which is more reliable.

### Method 1: Use the Simple Script (Recommended)

In your Jupyter notebook, replace the failing cell with:

```python
from load_gpt2_simple import load_gpt2_from_hf

# Download and load GPT-2 weights from Hugging Face
settings, params = load_gpt2_from_hf(model_size="124M", save_dir="gpt2_hf")

print("Settings:", settings)
print("Params keys:", params.keys())
```

### Method 2: Direct Download in Notebook

Or use this code directly in your notebook:

```python
import os
import json
import torch
from transformers import GPT2LMHeadModel
import numpy as np

def load_gpt2_hf(model_size="124M"):
    """Load GPT-2 from Hugging Face"""
    size_map = {"124M": "gpt2", "355M": "gpt2-medium", "774M": "gpt2-large", "1558M": "gpt2-xl"}
    
    print(f"Downloading {size_map[model_size]} from Hugging Face...")
    model = GPT2LMHeadModel.from_pretrained(size_map[model_size])
    config = model.config
    
    settings = {
        "n_vocab": config.vocab_size,
        "n_ctx": config.n_positions,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer
    }
    
    # Convert to parameter format
    sd = model.state_dict()
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}
    
    params["wte"] = sd["transformer.wte.weight"].cpu().numpy()
    params["wpe"] = sd["transformer.wpe.weight"].cpu().numpy()
    params["ln_f"] = {
        "g": sd["transformer.ln_f.weight"].cpu().numpy(),
        "b": sd["transformer.ln_f.bias"].cpu().numpy()
    }
    
    for i in range(settings["n_layer"]):
        prefix = f"transformer.h.{i}"
        params["blocks"][i] = {
            "ln_1": {
                "g": sd[f"{prefix}.ln_1.weight"].cpu().numpy(),
                "b": sd[f"{prefix}.ln_1.bias"].cpu().numpy()
            },
            "attn": {
                "c_attn": {
                    "w": sd[f"{prefix}.attn.c_attn.weight"].cpu().numpy(),
                    "b": sd[f"{prefix}.attn.c_attn.bias"].cpu().numpy()
                },
                "c_proj": {
                    "w": sd[f"{prefix}.attn.c_proj.weight"].cpu().numpy(),
                    "b": sd[f"{prefix}.attn.c_proj.bias"].cpu().numpy()
                }
            },
            "ln_2": {
                "g": sd[f"{prefix}.ln_2.weight"].cpu().numpy(),
                "b": sd[f"{prefix}.ln_2.bias"].cpu().numpy()
            },
            "mlp": {
                "c_fc": {
                    "w": sd[f"{prefix}.mlp.c_fc.weight"].cpu().numpy(),
                    "b": sd[f"{prefix}.mlp.c_fc.bias"].cpu().numpy()
                },
                "c_proj": {
                    "w": sd[f"{prefix}.mlp.c_proj.weight"].cpu().numpy(),
                    "b": sd[f"{prefix}.mlp.c_proj.bias"].cpu().numpy()
                }
            }
        }
    
    return settings, params

# Use it:
settings, params = load_gpt2_hf("124M")
```

### Method 3: Command Line Download

Run this in your terminal:

```bash
python load_gpt2_simple.py
```

This will download the model to `gpt2_hf/124M/` directory.

## Why This Works Better

1. **Hugging Face is reliable**: Their CDN is fast and stable
2. **Automatic caching**: Downloaded models are cached locally
3. **No size mismatch errors**: Proper download handling
4. **Same format**: Converts to the exact format your code expects

## Requirements

Make sure you have transformers installed:
```bash
pip install transformers
```

(It's already installed in your environment)

## Notes

- First download takes a few minutes (~548MB for 124M model)
- Subsequent loads are instant (uses cache)
- The converted parameters are compatible with your existing code
- You can use the same `settings` and `params` variables as before
