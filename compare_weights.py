#!/usr/bin/env python3

import torch
from transformers import AutoModelForCausalLM
import os
import argparse

def save_original_hf_weights(model_path, save_path="/mnt/sfs_turbo/hyx/original_hf_weights.pt"):
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

def inspect_qkv_tensors(original_path):
    """Inspect and print statistics for source attention.query_key_value.weight tensors"""
    print(f"\n=== Inspecting QKV Tensors ===")
    print(f"Loading: {original_path}")

    original = torch.load(original_path, map_location='cpu')

    # Find all QKV tensors
    qkv_keys = [k for k in original.keys() if 'attention.query_key_value.weight' in k]

    print(f"Found {len(qkv_keys)} QKV tensors")

    for i, key in enumerate(sorted(qkv_keys)):
        tensor = original[key]
        print(f"\n--- QKV Tensor {i+1}/{len(qkv_keys)} ---")
        print(f"Name: {key}")
        print(f"Shape: {tensor.shape}")
        print(f"Data type: {tensor.dtype}")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"Std: {tensor.std().item():.6f}")
        print(f"Min: {tensor.min().item():.6f}")
        print(f"Max: {tensor.max().item():.6f}")

        # Check first few and last few values
        flat_tensor = tensor.view(-1)
        print(f"First 5 values: {flat_tensor[:5].tolist()}")
        print(f"Last 5 values: {flat_tensor[-5:].tolist()}")

        # Check for patterns that might indicate rank-based distribution
        if tensor.shape[0] >= 4:  # At least 4 rows to analyze
            row_means = tensor.mean(dim=1)  # Mean of each row
            print(f"Row means - First 3: {row_means[:3].tolist()}")
            print(f"Row means - Last 3: {row_means[-3:].tolist()}")

            mid_point = tensor.shape[0] // 2
            print(f"DEBUG:   Row means at mid point ({mid_point}): {row_means[mid_point-1:mid_point+2].tolist()}", flush=True)

    print(f"\n=== QKV Inspection Complete ===\n")

def compare_weights(original_path, converted_path, compare_all=False, diff_threshold=1e-6):
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

    # Optional: Compare all layers and find mismatches
    if compare_all:
        print(f"\n=== Comparing ALL layers ===")
        shape_mismatches = []
        diff_mismatches = []
        matched_count = 0

        for key in sorted(common_keys):
            orig = original[key]
            conv = converted[key]

            if orig.shape != conv.shape:
                shape_mismatches.append(key)
                print(f"\n❌ SHAPE MISMATCH: {key}")
                print(f"   Orig: {orig.shape}, Conv: {conv.shape}")
            else:
                diff = torch.abs(orig - conv)
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                if max_diff > diff_threshold:
                    diff_mismatches.append((key, max_diff, mean_diff))
                    print(f"\n⚠️  DIFF MISMATCH: {key}")
                    print(f"   Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
                else:
                    matched_count += 1

        # Summary
        print(f"\n=== ALL LAYERS COMPARISON SUMMARY ===")
        print(f"✅ Matched layers: {matched_count}")
        print(f"❌ Shape mismatches: {len(shape_mismatches)}")
        print(f"⚠️  Diff mismatches (> {diff_threshold}): {len(diff_mismatches)}")

        if shape_mismatches:
            print(f"\nLayers with shape mismatches:")
            for key in shape_mismatches:
                print(f"  - {key}")

        if diff_mismatches:
            print(f"\nTop 10 layers with diff mismatches (sorted by max diff):")
            diff_mismatches.sort(key=lambda x: x[1], reverse=True)
            for key, max_diff, mean_diff in diff_mismatches[:10]:
                print(f"  - {key}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

            if len(diff_mismatches) > 10:
                print(f"  ... and {len(diff_mismatches) - 10} more layers with diff mismatches")
    else:
        # Do the original key_patterns comparison
        print("\n=== Comparing key parameters ===")
        key_patterns = [
            "model.layers.0.attention.dense.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.attention.query_key_value.weight",
            "model.layers.1.attention.query_key_value.weight",
            # "model.layers.1.mlp.experts.0.gate_proj.weight",
            # "model.layers.1.mlp.experts.0.up_proj.weight",
            # "model.layers.1.mlp.experts.0.down_proj.weight",
            "model.layers.1.input_layernorm.weight",
            "lm_head.weight",
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
    parser = argparse.ArgumentParser(description="Compare original HF weights with converted Megatron weights")
    parser.add_argument("--save-original", type=str, help="Save original HF weights from this model path")
    parser.add_argument("--original", type=str, default="/mnt/sfs_turbo/hyx/original_hf_weights.pt",
                       help="Path to original HF weights file")
    parser.add_argument("--converted", type=str, default="/mnt/sfs_turbo/hyx/megatron_converted_weights.pt",
                       help="Path to converted Megatron weights file")
    parser.add_argument("--do-compare", action="store_true",
                       help="Do layers comparison and report mismatches")
    parser.add_argument("--compare-all", action="store_true",
                       help="Compare all layers and report mismatches")
    parser.add_argument("--diff-threshold", type=float, default=1e-6,
                       help="Threshold for considering a difference significant (default: 1e-6)")
    parser.add_argument("--inspect-src-qkv", action="store_true",
                       help="Inspect and print statistics for source attention.query_key_value.weight tensors")
    parser.add_argument("--inspect-converted-qkv", action="store_true",
                       help="Inspect and print statistics for converted attention.query_key_value.weight tensors")


    args = parser.parse_args()

    # Save original weights if requested
    if args.save_original:
        save_original_hf_weights(args.save_original, args.original)

    # Inspect source QKV tensors if requested
    if args.inspect_src_qkv:
        inspect_qkv_tensors(args.original)

    if args.inspect_converted_qkv:
        inspect_qkv_tensors(args.converted)

    # Compare weights
    if args.do_compare:
        compare_weights(args.original, args.converted, args.compare_all, args.diff_threshold)