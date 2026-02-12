#!/usr/bin/env python3
"""
Compare AMBA vs Python ONNX update model outputs.

Usage:
    python3 compare_update_onnx_outputs.py onnx_models/update.onnx --frame 15 --bin-dir bin_file

This script:
  1. Loads the reshaped inputs saved by C++ (AMBA or ONNX path) from bin_file/
  2. Feeds them to the Python ONNX update model
  3. Loads the C++ model outputs from bin_file/
  4. Compares the outputs (correlation, max_diff, mean_diff)

This is the definitive diagnostic to determine:
  - Whether the AMBA model inputs match what ONNX expects (input comparison)
  - Whether the AMBA model outputs match ONNX outputs (output comparison)
"""

import argparse
import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path
from typing import Dict, Optional


def load_binary(filename: str, dtype=np.float32) -> np.ndarray:
    """Load binary file as numpy array."""
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data


def load_metadata(filepath: str) -> Dict[str, int]:
    """Load update model metadata from text file."""
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key.strip()] = int(value.strip())
    return metadata


def compute_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation between two arrays."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    if a_flat.std() < 1e-10 or b_flat.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def compare_arrays(name: str, cpp_arr: np.ndarray, py_arr: np.ndarray, 
                   num_active: int, max_edge: int, num_channels: int) -> dict:
    """Compare two arrays and print detailed statistics."""
    print(f"\n{'='*80}")
    print(f"  {name}")
    print(f"{'='*80}")
    
    # Overall comparison
    diff = np.abs(cpp_arr - py_arr)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    corr = compute_correlation(cpp_arr, py_arr)
    
    print(f"  Shape: {cpp_arr.shape}")
    print(f"  C++ : min={cpp_arr.min():.6f}, max={cpp_arr.max():.6f}, mean={cpp_arr.mean():.6f}, std={cpp_arr.std():.6f}")
    print(f"  ONNX: min={py_arr.min():.6f}, max={py_arr.max():.6f}, mean={py_arr.mean():.6f}, std={py_arr.std():.6f}")
    print(f"  Max diff : {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Correlation: {corr:.6f}")
    
    # Active edges only comparison
    if num_channels > 1:
        # Shape: [1, C, H, 1] -> active = [:, :, :num_active, :]
        cpp_active = cpp_arr[:, :, :num_active, :]
        py_active = py_arr[:, :, :num_active, :]
    else:
        cpp_active = cpp_arr[:, :, :num_active, :]
        py_active = py_arr[:, :, :num_active, :]
    
    active_diff = np.abs(cpp_active - py_active)
    active_max_diff = float(np.max(active_diff))
    active_mean_diff = float(np.mean(active_diff))
    active_corr = compute_correlation(cpp_active, py_active)
    
    print(f"\n  Active edges only ({num_active}/{max_edge}):")
    print(f"    Max diff : {active_max_diff:.6e}")
    print(f"    Mean diff: {active_mean_diff:.6e}")
    print(f"    Correlation: {active_corr:.6f}")
    
    # Top 5 differences
    flat_diff = diff.flatten()
    top_indices = np.argsort(flat_diff)[-5:][::-1]
    print(f"\n  Top 5 largest differences:")
    print(f"  {'Index':<20} {'C++ Value':<15} {'ONNX Value':<15} {'Diff':<15}")
    for idx in top_indices:
        multi_idx = np.unravel_index(idx, diff.shape)
        print(f"  {str(multi_idx):<20} {cpp_arr.flat[idx]:<15.6f} {py_arr.flat[idx]:<15.6f} {flat_diff[idx]:<15.6e}")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'correlation': corr,
        'active_max_diff': active_max_diff,
        'active_mean_diff': active_mean_diff,
        'active_correlation': active_corr,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare AMBA vs Python ONNX update model outputs')
    parser.add_argument('model_path', help='Path to update ONNX model')
    parser.add_argument('--frame', type=int, default=15, help='Frame number (default: 15)')
    parser.add_argument('--bin-dir', default='bin_file', help='Directory with binary files (default: bin_file)')
    args = parser.parse_args()
    
    model_path = args.model_path
    frame = args.frame
    bin_dir = Path(args.bin_dir)
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # File paths
    meta_file = bin_dir / f"update_metadata_frame{frame}.txt"
    net_input_file = bin_dir / f"update_net_input_frame{frame}.bin"
    inp_input_file = bin_dir / f"update_inp_input_frame{frame}.bin"
    corr_input_file = bin_dir / f"update_corr_input_frame{frame}.bin"
    ii_input_file = bin_dir / f"update_ii_input_frame{frame}.bin"
    jj_input_file = bin_dir / f"update_jj_input_frame{frame}.bin"
    kk_input_file = bin_dir / f"update_kk_input_frame{frame}.bin"
    net_out_file = bin_dir / f"update_net_out_cpp_frame{frame}.bin"
    d_out_file = bin_dir / f"update_d_out_cpp_frame{frame}.bin"
    w_out_file = bin_dir / f"update_w_out_cpp_frame{frame}.bin"
    
    print("=" * 80)
    print("🔍 AMBA vs Python ONNX — Update Model Comparison")
    print("=" * 80)
    print(f"  Frame: {frame}")
    print(f"  Model: {model_path}")
    print(f"  Bin dir: {bin_dir}")
    
    # Check all files exist
    required_files = {
        'metadata': meta_file,
        'net_input': net_input_file,
        'inp_input': inp_input_file,
        'corr_input': corr_input_file,
        'ii_input': ii_input_file,
        'jj_input': jj_input_file,
        'kk_input': kk_input_file,
    }
    
    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(f"  {name}: {path}")
    
    if missing:
        print(f"\n❌ Missing input files:")
        for m in missing:
            print(m)
        print(f"\nMake sure you ran the C++ code with TARGET_FRAME={frame} first.")
        sys.exit(1)
    
    # Load metadata
    print(f"\n📄 Loading metadata from {meta_file}...")
    metadata = load_metadata(str(meta_file))
    print(f"  {metadata}")
    
    num_active = metadata.get('num_active', 0)
    MAX_EDGE = metadata.get('MAX_EDGE', 360)
    DIM = metadata.get('DIM', 384)
    CORR_DIM = metadata.get('CORR_DIM', 882)
    
    # Load C++ inputs
    print(f"\n📂 Loading saved inputs...")
    net_input = load_binary(str(net_input_file), np.float32)
    inp_input = load_binary(str(inp_input_file), np.float32)
    corr_input = load_binary(str(corr_input_file), np.float32)
    ii_input = load_binary(str(ii_input_file), np.int32)
    jj_input = load_binary(str(jj_input_file), np.int32)
    kk_input = load_binary(str(kk_input_file), np.int32)
    
    print(f"  net_input:  {net_input.shape} (expected: {DIM * MAX_EDGE})")
    print(f"  inp_input:  {inp_input.shape} (expected: {DIM * MAX_EDGE})")
    print(f"  corr_input: {corr_input.shape} (expected: {CORR_DIM * MAX_EDGE})")
    print(f"  ii_input:   {ii_input.shape} (expected: {MAX_EDGE})")
    print(f"  jj_input:   {jj_input.shape} (expected: {MAX_EDGE})")
    print(f"  kk_input:   {kk_input.shape} (expected: {MAX_EDGE})")
    
    # Print index statistics
    print(f"\n📊 Index statistics:")
    print(f"  ii: min={ii_input.min()}, max={ii_input.max()}, "
          f"active=[{ii_input[:num_active].min()}, {ii_input[:num_active].max()}], "
          f"inactive_nonzero={np.count_nonzero(ii_input[num_active:])}")
    print(f"  jj: min={jj_input.min()}, max={jj_input.max()}, "
          f"active=[{jj_input[:num_active].min()}, {jj_input[:num_active].max()}], "
          f"inactive_nonzero={np.count_nonzero(jj_input[num_active:])}")
    print(f"  kk: min={kk_input.min()}, max={kk_input.max()}, "
          f"active=[{kk_input[:num_active].min()}, {kk_input[:num_active].max()}], "
          f"inactive_nonzero={np.count_nonzero(kk_input[num_active:])}")
    
    # Reshape inputs to match model expectations
    net_4d = net_input.reshape(1, DIM, MAX_EDGE, 1).astype(np.float32)
    inp_4d = inp_input.reshape(1, DIM, MAX_EDGE, 1).astype(np.float32)
    corr_4d = corr_input.reshape(1, CORR_DIM, MAX_EDGE, 1).astype(np.float32)
    ii_4d = ii_input.reshape(1, 1, MAX_EDGE, 1).astype(np.int32)
    jj_4d = jj_input.reshape(1, 1, MAX_EDGE, 1).astype(np.int32)
    kk_4d = kk_input.reshape(1, 1, MAX_EDGE, 1).astype(np.int32)
    
    # Load ONNX model
    print(f"\n📦 Loading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    input_shapes = [inp.shape for inp in session.get_inputs()]
    input_types = [inp.type for inp in session.get_inputs()]
    
    print(f"  Inputs:")
    for name, shape, dtype in zip(input_names, input_shapes, input_types):
        print(f"    {name}: shape={shape}, type={dtype}")
    print(f"  Outputs: {output_names}")
    
    # Build input dict
    inputs_dict = {
        input_names[0]: net_4d,
        input_names[1]: inp_4d,
        input_names[2]: corr_4d,
        input_names[3]: ii_4d,
        input_names[4]: jj_4d,
        input_names[5]: kk_4d,
    }
    
    # Run Python ONNX inference
    print(f"\n🚀 Running Python ONNX inference with saved C++ inputs...")
    try:
        outputs_py = session.run(output_names, inputs_dict)
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        sys.exit(1)
    
    net_out_py = outputs_py[0]  # [1, DIM, MAX_EDGE, 1]
    d_out_py = outputs_py[1]    # [1, 2, MAX_EDGE, 1]
    w_out_py = outputs_py[2]    # [1, 2, MAX_EDGE, 1]
    
    print(f"  ✅ ONNX inference done.")
    print(f"  net_out: shape={net_out_py.shape}")
    print(f"  d_out:   shape={d_out_py.shape}")
    print(f"  w_out:   shape={w_out_py.shape}")
    
    # Load C++ outputs (if available)
    has_cpp_outputs = net_out_file.exists() and d_out_file.exists() and w_out_file.exists()
    
    if has_cpp_outputs:
        print(f"\n📂 Loading C++ model outputs...")
        net_out_cpp = load_binary(str(net_out_file), np.float32).reshape(1, DIM, MAX_EDGE, 1)
        d_out_cpp = load_binary(str(d_out_file), np.float32).reshape(1, 2, MAX_EDGE, 1)
        w_out_cpp = load_binary(str(w_out_file), np.float32).reshape(1, 2, MAX_EDGE, 1)
        
        print(f"  net_out_cpp: shape={net_out_cpp.shape}")
        print(f"  d_out_cpp:   shape={d_out_cpp.shape}")
        print(f"  w_out_cpp:   shape={w_out_cpp.shape}")
        
        # Compare C++ outputs vs ONNX outputs
        print(f"\n{'='*80}")
        print(f"  COMPARISON: C++ Model Output vs Python ONNX Output")
        print(f"  (Same inputs, different runtime — tests model conversion quality)")
        print(f"{'='*80}")
        
        results = {}
        results['net_out'] = compare_arrays("net_out [1, 384, 360, 1]", 
                                             net_out_cpp, net_out_py, num_active, MAX_EDGE, DIM)
        results['d_out'] = compare_arrays("d_out [1, 2, 360, 1]", 
                                           d_out_cpp, d_out_py, num_active, MAX_EDGE, 2)
        results['w_out'] = compare_arrays("w_out [1, 2, 360, 1]", 
                                           w_out_cpp, w_out_py, num_active, MAX_EDGE, 2)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"  📊 SUMMARY")
        print(f"{'='*80}")
        print(f"  {'Output':<12} {'Correlation':>12} {'Active Corr':>12} {'Max Diff':>12} {'Mean Diff':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        for name, r in results.items():
            corr_str = f"{r['correlation']:.6f}"
            act_corr_str = f"{r['active_correlation']:.6f}"
            status = "✅" if r['active_correlation'] > 0.95 else "❌"
            print(f"  {status} {name:<10} {corr_str:>12} {act_corr_str:>12} {r['max_diff']:>12.4e} {r['mean_diff']:>12.4e}")
        
        all_good = all(r['active_correlation'] > 0.95 for r in results.values())
        if all_good:
            print(f"\n  ✅ All outputs have >0.95 active-edge correlation. Model conversion is good!")
        else:
            print(f"\n  ❌ Some outputs have <0.95 correlation. Check details above.")
            
            # Diagnostic: Are the inputs the cause?
            print(f"\n  🔬 DIAGNOSTIC HINTS:")
            
            # Check if net_input is all zeros (first iteration)
            net_nonzero = np.count_nonzero(net_input)
            if net_nonzero == 0:
                print(f"    ⚠️  net_input is ALL ZEROS (first iteration). This is expected.")
            else:
                print(f"    ℹ️  net_input has {net_nonzero} non-zero values.")
            
            # Check inactive edge indices
            inactive_ii_nonzero = np.count_nonzero(ii_input[num_active:])
            inactive_jj_nonzero = np.count_nonzero(jj_input[num_active:])
            inactive_kk_nonzero = np.count_nonzero(kk_input[num_active:])
            if inactive_ii_nonzero > 0 or inactive_jj_nonzero > 0 or inactive_kk_nonzero > 0:
                print(f"    ❌ Inactive edges have NON-ZERO indices!")
                print(f"       ii: {inactive_ii_nonzero}, jj: {inactive_jj_nonzero}, kk: {inactive_kk_nonzero}")
                print(f"       This means the AMBA reshapeInput is padding with non-zero values.")
                print(f"       The ONNX path uses zero padding. FIX: match ONNX zero-fill logic.")
            else:
                print(f"    ✅ Inactive edges are all zeros (matching ONNX path).")
            
            # Check for negative kk values
            neg_kk = np.sum(kk_input[:num_active] < 0)
            if neg_kk > 0:
                print(f"    ⚠️  {neg_kk} active edges have negative kk values.")
    else:
        print(f"\n⚠️  No C++ output files found. Cannot compare model outputs.")
        print(f"  Expected:")
        print(f"    {net_out_file}")
        print(f"    {d_out_file}")
        print(f"    {w_out_file}")
    
    # Save Python ONNX outputs for reference
    py_net_out_file = bin_dir / f"update_net_out_py_frame{frame}.bin"
    py_d_out_file = bin_dir / f"update_d_out_py_frame{frame}.bin"
    py_w_out_file = bin_dir / f"update_w_out_py_frame{frame}.bin"
    
    net_out_py.flatten().astype(np.float32).tofile(str(py_net_out_file))
    d_out_py.flatten().astype(np.float32).tofile(str(py_d_out_file))
    w_out_py.flatten().astype(np.float32).tofile(str(py_w_out_file))
    print(f"\n💾 Saved Python ONNX outputs:")
    print(f"  {py_net_out_file}")
    print(f"  {py_d_out_file}")
    print(f"  {py_w_out_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
