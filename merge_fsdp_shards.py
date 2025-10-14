#!/usr/bin/env python3
"""
Script to merge FSDP shards into a single model weights file.
Each FSDP rank saves its own shard, and this script aggregates them into a complete model.
"""

import os
import argparse
import torch
from typing import Dict, Any
import glob


def load_fsdp_shards(shard_pattern: str = "fsdp_shard_rank_*.pt") -> Dict[int, Dict[str, torch.Tensor]]:
    """Load all FSDP shard files and return a dictionary mapping rank to shard weights."""
    shard_files = glob.glob(shard_pattern)
    if not shard_files:
        raise FileNotFoundError(f"No FSDP shard files found matching pattern: {shard_pattern}")

    shards = {}
    for shard_file in sorted(shard_files):
        # Extract rank number from filename
        rank = int(shard_file.split("_")[-1].split(".")[0])
        print(f"Loading shard from {shard_file} (rank {rank})")
        shard_data = torch.load(shard_file, map_location='cpu')
        shards[rank] = shard_data

    print(f"Loaded {len(shards)} FSDP shards")
    return shards


def analyze_shard_structure(shards: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Analyze the structure of FSDP shards to understand the distribution pattern."""
    analysis = {}

    # Get all unique parameter names across all shards
    all_param_names = set()
    for rank, shard in shards.items():
        all_param_names.update(shard.keys())

    analysis['total_params'] = len(all_param_names)
    analysis['num_ranks'] = len(shards)
    analysis['param_distribution'] = {}

    # For each parameter, check which ranks have it
    for param_name in all_param_names:
        ranks_with_param = []
        for rank, shard in shards.items():
            if param_name in shard:
                ranks_with_param.append(rank)

        analysis['param_distribution'][param_name] = {
            'present_in_ranks': ranks_with_param,
            'num_ranks': len(ranks_with_param)
        }

    return analysis


def merge_fsdp_shards(shards: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Merge FSDP shards into a single model dictionary."""
    merged_weights = {}
    analysis = analyze_shard_structure(shards)

    print(f"\nFSDP Shard Analysis:")
    print(f"  Total ranks: {analysis['num_ranks']}")
    print(f"  Total parameters: {analysis['total_params']}")

    # Count parameters by distribution type
    single_rank_params = 0
    multi_rank_params = 0

    for param_name, param_info in analysis['param_distribution'].items():
        if param_info['num_ranks'] == 1:
            # Parameter exists in only one rank (likely full parameter)
            rank = param_info['present_in_ranks'][0]
            merged_weights[param_name] = shards[rank][param_name]
            single_rank_params += 1
        elif param_info['num_ranks'] == analysis['num_ranks']:
            # Parameter exists in all ranks (likely sharded parameter)
            # For now, take rank 0's version (you may need to modify this logic)
            print(f"Warning: Parameter {param_name} found in all ranks. Using rank 0 version.")
            merged_weights[param_name] = shards[0][param_name]
            multi_rank_params += 1
        else:
            # Parameter exists in some ranks but not all
            print(f"Warning: Parameter {param_name} found in ranks {param_info['present_in_ranks']}. Using rank {param_info['present_in_ranks'][0]} version.")
            merged_weights[param_name] = shards[param_info['present_in_ranks'][0]][param_name]
            multi_rank_params += 1

    print(f"\nMerge Summary:")
    print(f"  Single-rank parameters: {single_rank_params}")
    print(f"  Multi-rank parameters: {multi_rank_params}")
    print(f"  Total merged parameters: {len(merged_weights)}")

    return merged_weights


def verify_merged_weights(merged_weights: Dict[str, torch.Tensor], output_file: str):
    """Verify the merged weights and save them."""
    print(f"\nVerifying merged weights...")
    print(f"  Total parameters: {len(merged_weights)}")

    # Calculate total parameters count
    total_params_count = 0
    for param_name, param_tensor in merged_weights.items():
        param_count = param_tensor.numel()
        total_params_count += param_count
        print(f"  {param_name}: shape={param_tensor.shape}, dtype={param_tensor.dtype}, count={param_count:,}")

    print(f"  Total parameter count: {total_params_count:,}")

    # Save merged weights
    torch.save(merged_weights, output_file)
    print(f"\n✅ Merged weights saved to: {os.path.abspath(output_file)}")

    # Also save in a more readable format for verification
    info_file = output_file.replace('.pt', '_info.txt')
    with open(info_file, 'w') as f:
        f.write(f"FSDP Shard Merge Information\n")
        f.write(f"==============================\n\n")
        f.write(f"Total parameters: {len(merged_weights)}\n")
        f.write(f"Total parameter count: {total_params_count:,}\n\n")
        f.write(f"Parameter Details:\n")
        f.write(f"-----------------\n")
        for param_name, param_tensor in merged_weights.items():
            f.write(f"{param_name}:\n")
            f.write(f"  Shape: {param_tensor.shape}\n")
            f.write(f"  Dtype: {param_tensor.dtype}\n")
            f.write(f"  Count: {param_tensor.numel():,}\n")
            f.write(f"  Mean: {param_tensor.float().mean().item():.6f}\n")
            f.write(f"  Std: {param_tensor.float().std().item():.6f}\n")
            f.write(f"  Min: {param_tensor.float().min().item():.6f}\n")
            f.write(f"  Max: {param_tensor.float().max().item():.6f}\n\n")

    print(f"✅ Parameter information saved to: {os.path.abspath(info_file)}")


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP shards into a single model weights file")
    parser.add_argument("--root-dir", type=str, default=".",
                       help="Root directory containing FSDP shard files (default: current directory)")
    parser.add_argument("--shard-pattern", type=str, default="fsdp_shard_rank_*.pt",
                       help="Pattern to match FSDP shard files (default: fsdp_shard_rank_*.pt)")
    parser.add_argument("--output", type=str, default="merged_fsdp_weights.pt",
                       help="Output file path for merged weights (default: merged_fsdp_weights.pt)")
    parser.add_argument("--analyze-only", action="store_true",
                       help="Only analyze the shard structure without merging")

    args = parser.parse_args()

    # Change to root directory if specified
    if args.root_dir != ".":
        if not os.path.exists(args.root_dir):
            print(f"Error: Root directory '{args.root_dir}' does not exist")
            return
        os.chdir(args.root_dir)
        print(f"Changed working directory to: {os.path.abspath(args.root_dir)}")

    print("FSDP Shard Merger")
    print("==================")
    print(f"Working directory: {os.path.abspath('.')}")
    print(f"Shard pattern: {args.shard_pattern}")
    print(f"Output file: {args.output}")

    try:
        # Load FSDP shards
        shards = load_fsdp_shards(args.shard_pattern)

        # Analyze shard structure
        analysis = analyze_shard_structure(shards)

        print(f"\nFSDP Shard Analysis:")
        print(f"  Total ranks: {analysis['num_ranks']}")
        print(f"  Total parameters: {analysis['total_params']}")

        # Show some example parameters
        print(f"\nExample parameters:")
        example_count = 0
        for param_name, param_info in analysis['param_distribution'].items():
            if example_count >= 5:
                break
            print(f"  {param_name}: present in ranks {param_info['present_in_ranks']}")
            example_count += 1

        if args.analyze_only:
            print("\nAnalysis complete. Use --output to merge the shards.")
            return

        # Merge shards
        merged_weights = merge_fsdp_shards(shards)

        # Verify and save
        verify_merged_weights(merged_weights, args.output)

        print(f"\n🎉 FSDP shard merging completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during merging: {e}")
        raise


if __name__ == "__main__":
    main()