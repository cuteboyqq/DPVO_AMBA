#!/usr/bin/env python3
"""
Compare C++ and Python ONNX update model inference outputs from DPVO run.

This script loads C++ ONNX update model inputs/outputs saved during DPVO execution,
runs Python ONNX inference with the same inputs, and compares the outputs.
"""

import numpy as np
import onnxruntime as ort
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ComparisonResult:
    """Result of comparing C++ and Python outputs."""
    name: str
    matches: bool
    max_diff: float
    mean_diff: float
    num_mismatches: int
    total_elements: int
    shape: Tuple[int, ...]

def load_binary_file(filename: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load binary file as numpy array."""
    if not Path(filename).exists():
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def load_metadata(frame: int) -> Dict[str, int]:
    """Load update model metadata from text file."""
    bin_dir = "bin_file"
    metadata_file = os.path.join(bin_dir, f"update_metadata_frame{frame}.txt")
    if not Path(metadata_file).exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = {}
    with open(metadata_file, 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            metadata[key] = int(value)
    return metadata

def format_number(val: float) -> str:
    """Format number for display in table."""
    if abs(val) < 1e-6:
        return f"{val:.2e}"
    elif abs(val) < 0.001:
        return f"{val:.6f}"
    else:
        return f"{val:.6f}"

def compare_outputs(cpp_output: np.ndarray, py_output: np.ndarray, name: str,
                   shape: Tuple[int, ...], atol: float = 1e-4) -> ComparisonResult:
    """Compare C++ and Python outputs."""
    # Reshape to match
    cpp_reshaped = cpp_output.reshape(shape)
    py_reshaped = py_output.reshape(shape)
    
    if cpp_reshaped.shape != py_reshaped.shape:
        return ComparisonResult(
            name=name,
            matches=False,
            max_diff=np.inf,
            mean_diff=np.inf,
            num_mismatches=0,
            total_elements=0,
            shape=shape
        )
    
    # Calculate differences
    diff = np.abs(cpp_reshaped - py_reshaped)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_mismatches = np.sum(diff > atol)
    total_elements = diff.size
    
    matches = np.allclose(cpp_reshaped, py_reshaped, atol=atol)
    
    return ComparisonResult(
        name=name,
        matches=matches,
        max_diff=max_diff,
        mean_diff=mean_diff,
        num_mismatches=num_mismatches,
        total_elements=total_elements,
        shape=shape
    )

def print_comparison_table(results: list[ComparisonResult]) -> None:
    """Print comparison results in a table format."""
    print("\n" + "="*100)
    print("UPDATE MODEL OUTPUT COMPARISON")
    print("="*100)
    print(f"{'Output':<20} {'Status':<12} {'Max Diff':<15} {'Mean Diff':<15} {'Mismatches':<15} {'Shape':<25}")
    print("-"*100)
    
    for result in results:
        status = "✅ MATCH" if result.matches else "❌ MISMATCH"
        mismatch_str = f"{result.num_mismatches}/{result.total_elements}"
        shape_str = str(result.shape)
        if len(shape_str) > 25:
            shape_str = shape_str[:22] + "..."
        
        print(f"{result.name:<20} {status:<12} {format_number(result.max_diff):<15} "
              f"{format_number(result.mean_diff):<15} {mismatch_str:<15} {shape_str:<25}")
    
    print("="*100)

def print_sample_values_table(cpp_data: np.ndarray, py_data: np.ndarray, name: str,
                               shape: Tuple[int, ...], num_samples: int = 20) -> None:
    """Print sample values in a comprehensive table format."""
    cpp_reshaped = cpp_data.reshape(shape)
    py_reshaped = py_data.reshape(shape)
    diff = np.abs(cpp_reshaped - py_reshaped)
    
    # Flatten for easier indexing
    cpp_flat = cpp_reshaped.flatten()
    py_flat = py_reshaped.flatten()
    diff_flat = diff.flatten()
    
    # Get indices for sampling (first N, last N, and some random ones)
    total_size = cpp_flat.size
    num_samples = min(num_samples, total_size)
    
    # Sample indices: first few, last few, and some from middle
    sample_indices = []
    if total_size <= num_samples:
        sample_indices = list(range(total_size))
    else:
        # First few
        sample_indices.extend(range(min(5, total_size)))
        # Last few
        sample_indices.extend(range(max(0, total_size - 5), total_size))
        # Middle samples
        if num_samples > 10:
            step = total_size // (num_samples - 10)
            middle_indices = list(range(5, total_size - 5, step))[:num_samples - 10]
            sample_indices.extend(middle_indices)
        sample_indices = sample_indices[:num_samples]
    
    print(f"\n{'='*100}")
    print(f"SAMPLE VALUES: {name}")
    print(f"{'='*100}")
    print(f"Shape: {shape}, Total elements: {total_size}")
    print(f"{'Index':<15} {'Location':<30} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20}")
    print("-"*100)
    
    for idx in sample_indices:
        # Convert flat index to multi-dimensional index
        orig_idx = np.unravel_index(idx, shape)
        idx_str = ', '.join(map(str, orig_idx))
        
        cpp_val = cpp_flat[idx]
        py_val = py_flat[idx]
        diff_val = diff_flat[idx]
        
        print(f"{idx:<15} {idx_str:<30} {format_number(cpp_val):<20} "
              f"{format_number(py_val):<20} {format_number(diff_val):<20}")
    
    # Print statistics
    max_diff = np.max(diff_flat)
    mean_diff = np.mean(diff_flat)
    print("-"*100)
    print(f"{'Statistics':<15} {'':<30} {'Max Diff':<20} {'Mean Diff':<20}")
    print(f"{'':<15} {'':<30} {format_number(max_diff):<20} {format_number(mean_diff):<20}")
    print("="*100)

def print_sample_values(cpp_data: np.ndarray, py_data: np.ndarray, name: str,
                       shape: Tuple[int, ...], num_samples: int = 5) -> None:
    """Print sample values for debugging (legacy function, calls table version)."""
    print_sample_values_table(cpp_data, py_data, name, shape, num_samples)

def main() -> int:
    """Main function."""
    if len(sys.argv) < 3:
        print("Usage: python3 compare_update_onnx_dpvo.py <update_model.onnx> <frame_number>")
        print("Example: python3 compare_update_onnx_dpvo.py onnx_models/update.onnx 69")
        sys.exit(1)
    
    model_path = sys.argv[1]
    frame = int(sys.argv[2])
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print("="*80)
    print("COMPARING C++ AND PYTHON ONNX UPDATE MODEL OUTPUTS")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Frame: {frame}")
    
    # Load metadata
    print("\n📋 Loading metadata...")
    try:
        metadata = load_metadata(frame)
        num_active = metadata['num_active']
        MAX_EDGE = metadata['MAX_EDGE']
        DIM = metadata['DIM']
        CORR_DIM = metadata['CORR_DIM']
        
        print(f"  Frame: {metadata['frame']}")
        print(f"  num_active: {num_active}")
        print(f"  MAX_EDGE: {MAX_EDGE}")
        print(f"  DIM: {DIM}")
        print(f"  CORR_DIM: {CORR_DIM}")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure C++ code has been run and saved inputs/outputs for this frame.")
        sys.exit(1)
    
    # Load C++ inputs
    print("\n📥 Loading C++ inputs...")
    frame_suffix = str(frame)
    bin_dir = "bin_file"
    try:
        net_input_cpp = load_binary_file(os.path.join(bin_dir, f"update_net_input_frame{frame_suffix}.bin"))
        inp_input_cpp = load_binary_file(os.path.join(bin_dir, f"update_inp_input_frame{frame_suffix}.bin"))
        corr_input_cpp = load_binary_file(os.path.join(bin_dir, f"update_corr_input_frame{frame_suffix}.bin"))
        ii_input_cpp = load_binary_file(os.path.join(bin_dir, f"update_ii_input_frame{frame_suffix}.bin"), dtype=np.int32)
        jj_input_cpp = load_binary_file(os.path.join(bin_dir, f"update_jj_input_frame{frame_suffix}.bin"), dtype=np.int32)
        kk_input_cpp = load_binary_file(os.path.join(bin_dir, f"update_kk_input_frame{frame_suffix}.bin"), dtype=np.int32)
        
        print(f"  ✅ Loaded all input files")
        print(f"    net_input: {net_input_cpp.shape} (expected: {1 * DIM * MAX_EDGE * 1})")
        print(f"    inp_input: {inp_input_cpp.shape} (expected: {1 * DIM * MAX_EDGE * 1})")
        print(f"    corr_input: {corr_input_cpp.shape} (expected: {1 * CORR_DIM * MAX_EDGE * 1})")
        print(f"    ii_input: {ii_input_cpp.shape} (expected: {MAX_EDGE})")
        print(f"    jj_input: {jj_input_cpp.shape} (expected: {MAX_EDGE})")
        print(f"    kk_input: {kk_input_cpp.shape} (expected: {MAX_EDGE})")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure C++ code has been run and saved inputs for this frame.")
        sys.exit(1)
    
    # Load ONNX model
    print(f"\n🤖 Loading ONNX model: {model_path}")
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]
        
        print(f"  Input names: {input_names}")
        print(f"  Output names: {output_names}")
    except Exception as e:
        print(f"❌ Error loading ONNX model: {e}")
        sys.exit(1)
    
    # Prepare inputs for Python ONNX Runtime
    print("\n🔧 Preparing inputs for Python ONNX inference...")
    # Get actual input shapes from the model
    input_shapes = [inp.shape for inp in session.get_inputs()]
    print(f"  Model input shapes: {input_shapes}")
    
    # Reshape inputs according to model expectations
    # Note: Index inputs (ii, jj, kk) may have different shapes than float inputs
    inputs_dict = {}
    
    # Float inputs: net, inp, corr
    inputs_dict[input_names[0]] = net_input_cpp.reshape(input_shapes[0]).astype(np.float32)
    inputs_dict[input_names[1]] = inp_input_cpp.reshape(input_shapes[1]).astype(np.float32)
    inputs_dict[input_names[2]] = corr_input_cpp.reshape(input_shapes[2]).astype(np.float32)
    
    # Index inputs: ii, jj, kk (may be [1, MAX_EDGE] or [1, 1, MAX_EDGE, 1])
    inputs_dict[input_names[3]] = ii_input_cpp.reshape(input_shapes[3]).astype(np.int32)
    inputs_dict[input_names[4]] = jj_input_cpp.reshape(input_shapes[4]).astype(np.int32)
    inputs_dict[input_names[5]] = kk_input_cpp.reshape(input_shapes[5]).astype(np.int32)
    
    print(f"  Prepared input shapes:")
    for i, name in enumerate(input_names):
        print(f"    {name}: {inputs_dict[name].shape} (dtype: {inputs_dict[name].dtype})")
    
    # Run Python ONNX inference
    print("\n🚀 Running Python ONNX inference...")
    try:
        outputs_py = session.run(output_names, inputs_dict)
        
        net_out_py = outputs_py[0]  # [1, DIM, MAX_EDGE, 1]
        d_out_py = outputs_py[1]    # [1, 2, MAX_EDGE, 1]
        w_out_py = outputs_py[2]    # [1, 2, MAX_EDGE, 1]
        
        print(f"  ✅ Inference successful")
        print(f"    net_out shape: {net_out_py.shape}")
        print(f"    d_out shape: {d_out_py.shape}")
        print(f"    w_out shape: {w_out_py.shape}")
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        sys.exit(1)
    
    # Save Python outputs
    print("\n💾 Saving Python outputs...")
    os.makedirs(bin_dir, exist_ok=True)
    net_out_py.flatten().astype(np.float32).tofile(os.path.join(bin_dir, f"update_net_out_py_frame{frame_suffix}.bin"))
    d_out_py.flatten().astype(np.float32).tofile(os.path.join(bin_dir, f"update_d_out_py_frame{frame_suffix}.bin"))
    w_out_py.flatten().astype(np.float32).tofile(os.path.join(bin_dir, f"update_w_out_py_frame{frame_suffix}.bin"))
    print("  ✅ Saved Python outputs")
    
    # Load C++ outputs
    print("\n📥 Loading C++ outputs...")
    try:
        net_out_cpp = load_binary_file(os.path.join(bin_dir, f"update_net_out_cpp_frame{frame_suffix}.bin"))
        d_out_cpp = load_binary_file(os.path.join(bin_dir, f"update_d_out_cpp_frame{frame_suffix}.bin"))
        w_out_cpp = load_binary_file(os.path.join(bin_dir, f"update_w_out_cpp_frame{frame_suffix}.bin"))
        print("  ✅ Loaded all C++ output files")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure C++ code has been run and saved outputs for this frame.")
        sys.exit(1)
    
    # Compare outputs
    print("\n🔍 Comparing outputs...")
    results = []
    
    results.append(compare_outputs(
        net_out_cpp, net_out_py.flatten(),
        "net_out",
        (1, DIM, MAX_EDGE, 1),
        atol=1e-4
    ))
    
    results.append(compare_outputs(
        d_out_cpp, d_out_py.flatten(),
        "d_out",
        (1, 2, MAX_EDGE, 1),
        atol=1e-4
    ))
    
    results.append(compare_outputs(
        w_out_cpp, w_out_py.flatten(),
        "w_out",
        (1, 2, MAX_EDGE, 1),
        atol=1e-4
    ))
    
    # Print comparison table
    print_comparison_table(results)
    
    # Print parseable format for run_all_comparisons.py
    # Calculate overall max_diff and mean_diff across all outputs
    overall_max_diff = max(r.max_diff for r in results)
    overall_mean_diff = sum(r.mean_diff for r in results) / len(results)
    print(f"\n   UPDATE_MODEL_MAX_DIFF={overall_max_diff:.10e}")
    print(f"   UPDATE_MODEL_MEAN_DIFF={overall_mean_diff:.10e}")
    
    # Print sample values for all outputs in table format
    print("\n" + "="*100)
    print("DETAILED SAMPLE VALUES COMPARISON")
    print("="*100)
    
    # Print sample values for net_out
    print_sample_values_table(net_out_cpp, net_out_py.flatten(), "net_out", 
                             (1, DIM, MAX_EDGE, 1), num_samples=20)
    
    # Print sample values for d_out (delta output)
    print_sample_values_table(d_out_cpp, d_out_py.flatten(), "d_out", 
                             (1, 2, MAX_EDGE, 1), num_samples=20)
    
    # Print sample values for w_out (weight output)
    print_sample_values_table(w_out_cpp, w_out_py.flatten(), "w_out", 
                             (1, 2, MAX_EDGE, 1), num_samples=20)
    
    # Print edge-by-edge comparison for d_out and w_out (more readable)
    print(f"\n{'='*100}")
    print("EDGE-BY-EDGE COMPARISON (First 10 Edges)")
    print("="*100)
    
    # d_out comparison (delta: x, y for each edge)
    d_out_cpp_reshaped = d_out_cpp.reshape(1, 2, MAX_EDGE, 1)
    d_out_py_reshaped = d_out_py.reshape(1, 2, MAX_EDGE, 1)
    
    print(f"\n{'='*100}")
    print("D_OUT (Delta Output) - Edge Comparison")
    print("="*100)
    print(f"{'Edge':<10} {'C++ (x, y)':<30} {'Python (x, y)':<30} {'Difference (x, y)':<30}")
    print("-"*100)
    
    num_edges_to_show = min(10, MAX_EDGE)
    for e in range(num_edges_to_show):
        cpp_x = d_out_cpp_reshaped[0, 0, e, 0]
        cpp_y = d_out_cpp_reshaped[0, 1, e, 0]
        py_x = d_out_py_reshaped[0, 0, e, 0]
        py_y = d_out_py_reshaped[0, 1, e, 0]
        diff_x = abs(cpp_x - py_x)
        diff_y = abs(cpp_y - py_y)
        
        print(f"{e:<10} ({format_number(cpp_x)}, {format_number(cpp_y)}){'':<15} "
              f"({format_number(py_x)}, {format_number(py_y)}){'':<15} "
              f"({format_number(diff_x)}, {format_number(diff_y)}){'':<15}")
    
    # w_out comparison (weight: w0, w1 for each edge)
    w_out_cpp_reshaped = w_out_cpp.reshape(1, 2, MAX_EDGE, 1)
    w_out_py_reshaped = w_out_py.reshape(1, 2, MAX_EDGE, 1)
    
    print(f"\n{'='*100}")
    print("W_OUT (Weight Output) - Edge Comparison")
    print("="*100)
    print(f"{'Edge':<10} {'C++ (w0, w1)':<30} {'Python (w0, w1)':<30} {'Difference (w0, w1)':<30}")
    print("-"*100)
    
    for e in range(num_edges_to_show):
        cpp_w0 = w_out_cpp_reshaped[0, 0, e, 0]
        cpp_w1 = w_out_cpp_reshaped[0, 1, e, 0]
        py_w0 = w_out_py_reshaped[0, 0, e, 0]
        py_w1 = w_out_py_reshaped[0, 1, e, 0]
        diff_w0 = abs(cpp_w0 - py_w0)
        diff_w1 = abs(cpp_w1 - py_w1)
        
        print(f"{e:<10} ({format_number(cpp_w0)}, {format_number(cpp_w1)}){'':<15} "
              f"({format_number(py_w0)}, {format_number(py_w1)}){'':<15} "
              f"({format_number(diff_w0)}, {format_number(diff_w1)}){'':<15}")
    
    print("="*100)
    
    # Summary
    print("\n" + "="*80)
    all_match = all(r.matches for r in results)
    if all_match:
        print("✅ ALL OUTPUTS MATCH! C++ ONNX update model inference is correct.")
    else:
        print("❌ SOME OUTPUTS DO NOT MATCH! Check the comparison table above for details.")
    print("="*80)
    
    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())

