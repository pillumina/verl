#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM
import os

def save_original_hf_weights(model_path, save_path="./original_hf_weights.pt"):
    """Save original HF model weights for comparison"""
    print(f"Loading original HF model from: {model_path}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)

    # Extract state dict
    state_dict = model.state_dict()

    # Save to file
    torch.save(state_dict, save_path)
    print(f"Original HF weights saved to: {os.path.abspath(save_path)}")
    print(f"Total parameters: {len(state_dict)}")

    # Print some sample parameters for verification
    print("\nSample parameters:")
    count = 0
    for name, param in state_dict.items():
        if count >= 10:  # Only show first 10
            break
        print(f"  {name}: shape={param.shape}, dtype={param.dtype}")
        count += 1

    return state_dict

def compare_weights(original_path, converted_path):
    """Compare original and converted weights"""
    print(f"\n=== Comparing weights ===")
    print(f"Original: {original_path}")
    print(f"Converted: {converted_path}")

    # Load both
    original = torch.load(original_path, map_location='cpu')
    converted = torch.load(converted_path, map_location='cpu')

    print(f"Original keys: {len(original)}")
    print(f"Converted keys: {len(converted)}")

    # Find common keys
    common_keys = set(original.keys()) & set(converted.keys())
    print(f"Common keys: {len(common_keys)}")

    # Compare some key parameters
    print("\nComparing key parameters:")
    key_patterns = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.1.input_layernorm.weight",
    ]

    for pattern in key_patterns:
        if pattern in original and pattern in converted:
            orig = original[pattern]
            conv = converted[pattern]

            print(f"\n{pattern}:")
            print(f"  Shape match: {orig.shape == conv.shape}")
            print(f"  Orig shape: {orig.shape}, Conv shape: {conv.shape}")

            if orig.shape == conv.shape:
                # Compute difference
                diff = torch.abs(orig - conv)
                print(f"  Max diff: {diff.max().item():.6f}")
                print(f"  Mean diff: {diff.mean().item():.6f}")
                print(f"  Orig mean: {orig.mean().item():.6f}, Conv mean: {conv.mean().item():.6f}")
            else:
                print(f"  SHAPE MISMATCH!")
        else:
            print(f"\n{pattern}: NOT FOUND in both")

if __name__ == "__main__":
    # Replace this with your model path
    model_path = "/path/to/your/bailing/model"

    # Save original weights
    save_original_hf_weights(model_path)

    # Compare after you have both files
    # Note: Run this part after you have both original and converted weights
    # compare_weights("./original_hf_weights.pt", "./megatron_converted_weights.pt")