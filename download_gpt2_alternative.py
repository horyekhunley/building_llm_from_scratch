"""
Alternative GPT-2 Weight Download Script
This script provides a reliable way to download GPT-2 weights using Hugging Face.

Usage:
    python download_gpt2_alternative.py

Or in a Jupyter notebook:
    from download_gpt2_alternative import download_gpt2_weights
    settings, params = download_gpt2_weights()
"""

import os
import sys


def download_gpt2_weights(model_size="124M", models_dir="gpt2"):
    """
    Download GPT-2 weights using Hugging Face transformers library.
    This is more reliable than downloading from OpenAI's servers.
    
    Args:
        model_size: "124M", "355M", "774M", or "1558M"
        models_dir: Directory to save the model
    
    Returns:
        settings: Model configuration
        params: Model parameters
    """
    try:
        # Try using the Hugging Face method
        from gpt_download_hf import download_and_load_gpt2_hf
        print("Using Hugging Face transformers to download GPT-2 weights...")
        settings, params = download_and_load_gpt2_hf(model_size, models_dir)
        return settings, params
    
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install the required package:")
        print("  pip install transformers")
        sys.exit(1)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nIf the download fails, you can also try:")
        print("1. Check your internet connection")
        print("2. Install transformers: pip install transformers")
        print("3. Try downloading manually from: https://huggingface.co/gpt2")
        sys.exit(1)


def main():
    """Main function to run the download."""
    print("=" * 60)
    print("GPT-2 Weight Downloader (Alternative Method)")
    print("=" * 60)
    print()
    
    # Download 124M model by default
    model_size = "124M"
    models_dir = "gpt2"
    
    print(f"Downloading GPT-2 {model_size} model...")
    print(f"Save directory: {models_dir}/{model_size}")
    print()
    
    settings, params = download_gpt2_weights(model_size, models_dir)
    
    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)
    print(f"\nModel settings: {settings}")
    print(f"\nYou can now load the model using:")
    print(f"  from gpt_download_hf import download_and_load_gpt2_hf")
    print(f"  settings, params = download_and_load_gpt2_hf('{model_size}', '{models_dir}')")


if __name__ == "__main__":
    main()
