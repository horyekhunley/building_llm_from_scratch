"""
Quick test to verify the download works with a smaller model first.
Then you can download the full 124M model.
"""

from transformers import GPT2LMHeadModel
import time

def test_download():
    """Test with the smallest GPT-2 model"""
    print("Testing download with GPT-2 (124M)...")
    print("This will download ~548MB - please be patient...")
    print("\nDownload progress:")
    
    start = time.time()
    
    # This will show download progress automatically
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    elapsed = time.time() - start
    print(f"\n✓ Download completed in {elapsed:.1f} seconds!")
    print(f"Model has {model.config.n_layer} layers")
    print(f"Embedding dimension: {model.config.n_embd}")
    
    return model

if __name__ == "__main__":
    model = test_download()
    print("\n✓ Success! You can now use load_gpt2_simple.py")
