#!/usr/bin/env python3
"""
Convert .pt weight file to HuggingFace standard format.
This script loads a .pt file and saves it as pytorch_model.bin in the specified output directory.
"""

import os
import argparse
import torch
import shutil


def load_weights(pt_file):
    """Load weights from .pt file."""
    print(f"Loading weights from: {pt_file}")
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Weight file not found: {pt_file}")

    weights = torch.load(pt_file, map_location='cpu')
    print(f"✅ Weights loaded successfully")
    print(f"  Total parameters: {len(weights)}")
    print(f"  Total elements: {sum(p.numel() for p in weights.values()):,}")

    # Show some sample keys
    sample_keys = list(weights.keys())[:5]
    print(f"  Sample keys: {sample_keys}")

    return weights


def save_as_hf_format(weights, output_dir):
    """Save weights in HuggingFace standard format."""
    print(f"\nSaving to HuggingFace format...")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save as pytorch_model.bin (HF standard filename)
    output_file = os.path.join(output_dir, "pytorch_model.bin")
    print(f"Saving to: {output_file}")

    torch.save(weights, output_file)
    print(f"✅ Weights saved in HuggingFace format")

    return output_file


def copy_config_files(output_dir, config_source_dir=None):
    """Copy config files from source directory if specified."""
    if config_source_dir is None:
        print(f"\nNo config source directory specified.")
        print(f"Remember to manually copy config.json and tokenizer files to: {output_dir}")
        return

    if not os.path.exists(config_source_dir):
        print(f"⚠️  Config source directory not found: {config_source_dir}")
        return

    print(f"\nCopying config files from: {config_source_dir}")

    # Common config files to copy
    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model"
    ]

    copied_files = []
    for config_file in config_files:
        src_file = os.path.join(config_source_dir, config_file)
        if os.path.exists(src_file):
            dst_file = os.path.join(output_dir, config_file)
            shutil.copy2(src_file, dst_file)
            copied_files.append(config_file)
            print(f"  Copied: {config_file}")

    if copied_files:
        print(f"✅ Copied {len(copied_files)} config files")
    else:
        print(f"⚠️  No config files found in source directory")


def verify_hf_model(model_dir):
    """Verify that the saved model can be loaded by HuggingFace."""
    print(f"\nVerifying HuggingFace model...")

    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        # Check if config.json exists
        config_file = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_file):
            print(f"❌ config.json not found in {model_dir}")
            print(f"   Model can still be loaded with manual config")
            return True

        # Try to load config
        config = AutoConfig.from_pretrained(model_dir)
        print(f"✅ Config loaded successfully")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Num layers: {config.num_hidden_layers}")
        print(f"  Vocab size: {config.vocab_size}")

        # Try to load model (in CPU mode to save memory)
        print(f"Loading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map='cpu')
        print(f"✅ Model loaded successfully!")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        return True

    except ImportError:
        print(f"⚠️  transformers not installed, skipping verification")
        return True
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False


def print_summary(output_dir):
    """Print summary of the conversion."""
    print(f"\n" + "="*50)
    print(f"CONVERSION SUMMARY")
    print(f"="*50)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # List files in output directory
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"Files created:")
        for file in sorted(files):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024*1024)
                print(f"  {file}: {size_mb:.1f} MB")

    print(f"\nTo use the model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{os.path.abspath(output_dir)}')")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .pt weights to HuggingFace standard format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_pt_to_hf.py weights.pt --output-dir my_model

  # With config files from original model
  python convert_pt_to_hf.py weights.pt --output-dir my_model --config-dir /path/to/original/model

  # Verify the converted model
  python convert_pt_to_hf.py weights.pt --output-dir my_model --verify
        """
    )

    parser.add_argument("input_file", type=str, help="Input .pt file containing model weights")
    parser.add_argument("--output-dir", type=str, default="hf_model",
                       help="Output directory for HuggingFace model (default: hf_model)")
    parser.add_argument("--config-dir", type=str, default=None,
                       help="Directory containing config.json and tokenizer files to copy")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the converted model can be loaded by HuggingFace")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite output directory if it exists")

    args = parser.parse_args()

    print("HuggingFace Format Converter")
    print("="*30)
    print(f"Input: {args.input_file}")
    print(f"Output: {args.output_dir}")

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file not found: {args.input_file}")
        return

    # Check if output directory exists
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        if args.overwrite:
            print(f"⚠️  Output directory exists and will be overwritten")
        else:
            print(f"❌ Error: Output directory {args.output_dir} is not empty")
            print(f"   Use --overwrite to overwrite existing files")
            return

    try:
        # Load weights
        weights = load_weights(args.input_file)

        # Save in HF format
        save_as_hf_format(weights, args.output_dir)

        # Copy config files if specified
        copy_config_files(args.output_dir, args.config_dir)

        # Verify if requested
        if args.verify:
            verify_hf_model(args.output_dir)

        # Print summary
        print_summary(args.output_dir)

        print(f"\n🎉 Conversion completed successfully!")

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()