#!/usr/bin/env python3
"""
Generate proper DRA calibration data for the Update model AMBA conversion.

This script runs the DPVO pipeline using Python ONNX models to collect
representative input samples for the Update model. These samples are
then saved in the format expected by the AMBA eazyai_cvt tool.

Usage:
    python3 generate_update_calibration_data.py \
        onnx_models/update.onnx \
        --bin-dir bin_file \
        --frame 15 \
        --output-dir build/models/AshaCam/calibration_update

This reads the saved reshaped inputs from the C++ binary files and
creates the calibration directory structure:
    calibration_update/
        net/sample_0000.bin
        inp/sample_0000.bin
        corr/sample_0000.bin
        ii/sample_0000.bin
        jj/sample_0000.bin
        kk/sample_0000.bin

For best quantization quality, you should run the C++ code for multiple
frames and save inputs for each. Then point this script at all saved frames.
"""

import argparse
import numpy as np
import os
import sys
from pathlib import Path


def load_binary(filepath: str, dtype=np.float32) -> np.ndarray:
    """Load a binary file as numpy array."""
    with open(filepath, 'rb') as f:
        return np.frombuffer(f.read(), dtype=dtype)


def load_metadata(filepath: str) -> dict:
    """Load metadata from text file."""
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key.strip()] = int(value.strip())
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Generate proper DRA calibration data for Update model AMBA conversion')
    parser.add_argument('model_path', help='Path to update ONNX model (for validation)')
    parser.add_argument('--bin-dir', default='bin_file',
                        help='Directory with saved binary files from C++ (default: bin_file)')
    parser.add_argument('--frames', type=str, default='15',
                        help='Comma-separated list of frame numbers to use (default: 15)')
    parser.add_argument('--output-dir', default='build/models/AshaCam/calibration_update',
                        help='Output directory for calibration data')
    args = parser.parse_args()

    bin_dir = Path(args.bin_dir)
    output_dir = Path(args.output_dir)
    frames = [int(f.strip()) for f in args.frames.split(',')]

    print("=" * 80)
    print("🔧 Generate DRA Calibration Data for Update Model")
    print("=" * 80)
    print(f"  Frames: {frames}")
    print(f"  Bin dir: {bin_dir}")
    print(f"  Output dir: {output_dir}")

    # Create output directories
    input_names = ['net', 'inp', 'corr', 'ii', 'jj', 'kk']
    for name in input_names:
        (output_dir / name).mkdir(parents=True, exist_ok=True)

    sample_count = 0

    for frame in frames:
        meta_file = bin_dir / f"update_metadata_frame{frame}.txt"
        if not meta_file.exists():
            print(f"  ⚠️  Frame {frame}: metadata not found at {meta_file}, skipping")
            continue

        metadata = load_metadata(str(meta_file))
        MAX_EDGE = metadata.get('MAX_EDGE', 360)
        DIM = metadata.get('DIM', 384)
        CORR_DIM = metadata.get('CORR_DIM', 882)
        num_active = metadata.get('num_active', 0)

        # Load saved inputs
        files = {
            'net': (bin_dir / f"update_net_input_frame{frame}.bin", np.float32, DIM * MAX_EDGE),
            'inp': (bin_dir / f"update_inp_input_frame{frame}.bin", np.float32, DIM * MAX_EDGE),
            'corr': (bin_dir / f"update_corr_input_frame{frame}.bin", np.float32, CORR_DIM * MAX_EDGE),
            'ii': (bin_dir / f"update_ii_input_frame{frame}.bin", np.int32, MAX_EDGE),
            'jj': (bin_dir / f"update_jj_input_frame{frame}.bin", np.int32, MAX_EDGE),
            'kk': (bin_dir / f"update_kk_input_frame{frame}.bin", np.int32, MAX_EDGE),
        }

        all_exist = True
        for name, (path, _, _) in files.items():
            if not path.exists():
                print(f"  ⚠️  Frame {frame}: {name} file not found at {path}")
                all_exist = False
        if not all_exist:
            print(f"  Skipping frame {frame}")
            continue

        print(f"\n  📦 Frame {frame} (num_active={num_active}):")

        for name in input_names:
            path, dtype, expected_size = files[name]
            data = load_binary(str(path), dtype)

            if data.size != expected_size:
                print(f"    ⚠️  {name}: size mismatch {data.size} != {expected_size}")
                continue

            # For AMBA DRA calibration, we need fp32 data in the model's input shape
            # The ii/jj/kk inputs were saved as int32 by the C++ code,
            # but the AMBA conversion config expects fp32 (it handles the cast internally)
            if dtype == np.int32:
                data_fp32 = data.astype(np.float32)
            else:
                data_fp32 = data

            # Save as fp32 binary
            out_file = output_dir / name / f"sample_{sample_count:04d}.bin"
            data_fp32.tofile(str(out_file))

            print(f"    {name}: shape=({data.size},), "
                  f"min={data_fp32.min():.4f}, max={data_fp32.max():.4f}, "
                  f"mean={data_fp32.mean():.4f} → {out_file.name}")

        sample_count += 1

    if sample_count == 0:
        print("\n❌ No valid frames found! Make sure you ran the C++ code with TARGET_FRAME set.")
        sys.exit(1)

    print(f"\n✅ Generated {sample_count} calibration sample(s) in {output_dir}")

    # Print the updated YAML config snippet
    print(f"\n{'='*80}")
    print("📝 Updated dpvo_update.yaml config (replace data_prepare section):")
    print(f"{'='*80}")
    
    # Use relative path from the config file location
    # Config is at build/models/AshaCam/config/dpvo_update.yaml
    # Calibration data is at build/models/AshaCam/calibration_update/
    rel_path = "../calibration_update"
    
    yaml_snippet = f"""
data_prepare:
  net_data:
    in_path: {rel_path}/net
    in_file_ext: bin
    out_shape: 1,384,360,1
    out_data_format: fp32
    transforms:
      - class: ReadBinaryFiles
        arguments:
          data_format: float32
          shape: "1,384,360,1"

  inp_data:
    in_path: {rel_path}/inp
    in_file_ext: bin
    out_shape: 1,384,360,1
    out_data_format: fp32
    transforms:
      - class: ReadBinaryFiles
        arguments:
          data_format: float32
          shape: "1,384,360,1"

  corr_data:
    in_path: {rel_path}/corr
    in_file_ext: bin
    out_shape: 1,882,360,1
    out_data_format: fp32
    transforms:
      - class: ReadBinaryFiles
        arguments:
          data_format: float32
          shape: "1,882,360,1"

  ii_data:
    in_path: {rel_path}/ii
    in_file_ext: bin
    out_shape: 1,1,360,1
    out_data_format: fp32
    transforms:
      - class: ReadBinaryFiles
        arguments:
          data_format: float32
          shape: "1,1,360,1"

  jj_data:
    in_path: {rel_path}/jj
    in_file_ext: bin
    out_shape: 1,1,360,1
    out_data_format: fp32
    transforms:
      - class: ReadBinaryFiles
        arguments:
          data_format: float32
          shape: "1,1,360,1"

  kk_data:
    in_path: {rel_path}/kk
    in_file_ext: bin
    out_shape: 1,1,360,1
    out_data_format: fp32
    transforms:
      - class: ReadBinaryFiles
        arguments:
          data_format: float32
          shape: "1,1,360,1"
"""
    print(yaml_snippet)

    print("⚠️  IMPORTANT: After updating the YAML config with proper calibration data,")
    print("    you MUST re-run the AMBA model conversion tool to regenerate the model.")
    print("    The current model was converted with random dummy data, causing severe")
    print("    quantization quality loss.")

    # Also generate more frames if possible
    if sample_count < 10:
        print(f"\n💡 TIP: For best quantization quality, use 10-50+ calibration samples.")
        print(f"   To generate more samples:")
        print(f"   1. Modify C++ to save update inputs for multiple frames (not just TARGET_FRAME)")
        print(f"   2. Run the C++ code")
        print(f"   3. Re-run this script with --frames 5,10,15,20,25,30,35,40,45,50")

    return 0


if __name__ == "__main__":
    sys.exit(main())

