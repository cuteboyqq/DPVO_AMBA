#!/usr/bin/env python3
"""
Compare C++ and Python ONNX update model inference outputs from DPVO run.

This script loads C++ ONNX update model inputs/outputs saved during DPVO execution,
runs Python ONNX inference with the same inputs, and compares the outputs.
"""

import numpy as np
import onnxruntime as ort
import sys
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
    metadata_file = f"update_metadata_frame{frame}.txt"
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

def print_sample_values(cpp_data: np.ndarray, py_data: np.ndarray, name: str,
                       shape: Tuple[int, ...], num_samples: int = 5) -> None:
    """Print sample values for debugging."""
    cpp_reshaped = cpp_data.reshape(shape)
    py_reshaped = py_data.reshape(shape)
    
    print(f"\n{'='*80}")
    print(f"SAMPLE VALUES: {name}")
    print(f"{'='*80}")
    
    # Print first few elements
    flat_size = min(num_samples, cpp_data.size)
    print(f"{'Index':<10} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20}")
    print("-"*80)
    
    for i in range(flat_size):
        cpp_val = cpp_data[i]
        py_val = py_data[i]
        diff = abs(cpp_val - py_val)
        print(f"{i:<10} {cpp_val:<20.6f} {py_val:<20.6f} {diff:<20.6e}")

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
    try:
        net_input_cpp = load_binary_file(f"update_net_input_frame{frame_suffix}.bin")
        inp_input_cpp = load_binary_file(f"update_inp_input_frame{frame_suffix}.bin")
        corr_input_cpp = load_binary_file(f"update_corr_input_frame{frame_suffix}.bin")
        ii_input_cpp = load_binary_file(f"update_ii_input_frame{frame_suffix}.bin", dtype=np.int32)
        jj_input_cpp = load_binary_file(f"update_jj_input_frame{frame_suffix}.bin", dtype=np.int32)
        kk_input_cpp = load_binary_file(f"update_kk_input_frame{frame_suffix}.bin", dtype=np.int32)
        
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
    net_out_py.flatten().astype(np.float32).tofile(f"update_net_out_py_frame{frame_suffix}.bin")
    d_out_py.flatten().astype(np.float32).tofile(f"update_d_out_py_frame{frame_suffix}.bin")
    w_out_py.flatten().astype(np.float32).tofile(f"update_w_out_py_frame{frame_suffix}.bin")
    print("  ✅ Saved Python outputs")
    
    # Load C++ outputs
    print("\n📥 Loading C++ outputs...")
    try:
        net_out_cpp = load_binary_file(f"update_net_out_cpp_frame{frame_suffix}.bin")
        d_out_cpp = load_binary_file(f"update_d_out_cpp_frame{frame_suffix}.bin")
        w_out_cpp = load_binary_file(f"update_w_out_cpp_frame{frame_suffix}.bin")
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
    
    # Print sample values for mismatched outputs
    for result in results:
        if not result.matches:
            if result.name == "net_out":
                print_sample_values(net_out_cpp, net_out_py.flatten(), result.name, result.shape)
            elif result.name == "d_out":
                print_sample_values(d_out_cpp, d_out_py.flatten(), result.name, result.shape)
            elif result.name == "w_out":
                print_sample_values(w_out_cpp, w_out_py.flatten(), result.name, result.shape)
    
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

