#!/usr/bin/env python3
"""
Compare C++ and Python ONNX update model inference outputs
"""

import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path
from typing import Dict, Tuple, List

def load_binary_file(filename: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load binary file as numpy array.
    
    Args:
        filename: Path to the binary file to load.
        dtype: NumPy data type for the array elements (default: np.float32).
    
    Returns:
        NumPy array containing the data from the binary file.
    
    Note:
        Reads the entire file into memory and converts it to a NumPy array.
    """
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def load_metadata() -> Dict[str, int]:
    """Load test metadata from test_metadata.txt file.
    
    Returns:
        Dictionary containing test parameters:
            - num_active: Number of active edges in the test
            - MAX_EDGE: Maximum number of edges (model input dimension)
            - DIM: Feature dimension (typically 384)
            - CORR_DIM: Correlation dimension (typically 27)
    
    Note:
        Expects metadata file format: key=value (one per line).
        All values are parsed as integers.
    """
    metadata = {}
    with open('test_metadata.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            metadata[key] = int(value)
    return metadata

def compare_outputs(cpp_output: np.ndarray, py_output: np.ndarray, name: str, 
                   shape: Tuple[int, ...], atol: float = 1e-5) -> bool:
    """Compare C++ and Python outputs.
    
    Args:
        cpp_output: C++ output array (flattened or reshaped).
        py_output: Python output array (flattened or reshaped).
        name: Name/description of the output being compared (for logging).
        shape: Expected shape tuple for reshaping arrays (e.g., (1, DIM, MAX_EDGE, 1)).
        atol: Absolute tolerance for comparison (default: 1e-5).
    
    Returns:
        True if outputs match within tolerance, False otherwise.
    
    Note:
        Reshapes both arrays to the specified shape before comparison.
        Prints detailed statistics including max difference, mean difference,
        and percentage of mismatched elements.
        Shows first 5 mismatches if any are found.
    """
    print(f"\n=== Comparing {name} ===")
    print(f"Shape: {shape}")
    print(f"C++ output shape: {cpp_output.shape}")
    print(f"Python output shape: {py_output.shape}")
    
    # Reshape to match
    cpp_reshaped = cpp_output.reshape(shape)
    py_reshaped = py_output.reshape(shape)
    
    # Compare
    if cpp_reshaped.shape != py_reshaped.shape:
        print(f"❌ Shape mismatch: C++ {cpp_reshaped.shape} vs Python {py_reshaped.shape}")
        return False
    
    # Calculate differences
    diff = np.abs(cpp_reshaped - py_reshaped)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_mismatches = np.sum(diff > atol)
    total_elements = diff.size
    
    print(f"Max difference: {max_diff:.6e}")
    print(f"Mean difference: {mean_diff:.6e}")
    print(f"Mismatches (> {atol}): {num_mismatches} / {total_elements} ({100*num_mismatches/total_elements:.2f}%)")
    
    # Check if they match
    matches = np.allclose(cpp_reshaped, py_reshaped, atol=atol)
    
    if matches:
        print(f"✅ {name} outputs MATCH!")
    else:
        print(f"❌ {name} outputs DO NOT MATCH!")
        
        # Find first few mismatches
        mismatches = np.where(diff > atol)
        if len(mismatches[0]) > 0:
            print(f"\nFirst 5 mismatches:")
            for i in range(min(5, len(mismatches[0]))):
                idx = tuple(m[idx][i] for m, idx in zip(mismatches, range(len(mismatches))))
                print(f"  Index {idx}: C++={cpp_reshaped[idx]:.6f}, Python={py_reshaped[idx]:.6f}, Diff={diff[idx]:.6e}")
    
    return matches

def print_sample_net_out_values(net_out_cpp: np.ndarray, net_out_py: np.ndarray, 
                                 DIM: int, MAX_EDGE: int, num_samples: int = 5) -> None:
    """Print sample net_out values from C++ and Python outputs.
    
    Args:
        net_out_cpp: C++ net_out array (flattened, shape: [DIM * MAX_EDGE]).
        net_out_py: Python net_out array (NCHW format, shape: [1, DIM, MAX_EDGE, 1]).
        DIM: Feature dimension (number of channels).
        MAX_EDGE: Maximum number of edges (spatial dimension).
        num_samples: Number of sample values to print per edge (default: 5).
    
    Note:
        Prints values for the first few edges and channels to help debug differences.
        Shows both C++ and Python values side by side for easy comparison.
    """
    print("\n" + "="*60)
    print("SAMPLE NET_OUT VALUES (C++ vs Python)")
    print("="*60)
    
    # Print values for first 3 edges, first num_samples channels
    num_edges_to_show = min(3, MAX_EDGE)
    
    for edge_idx in range(num_edges_to_show):
        print(f"\nEdge {edge_idx}:")
        print(f"  {'Channel':<10} {'C++ Value':<15} {'Python Value':<15} {'Difference':<15}")
        print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
        
        for channel_idx in range(min(num_samples, DIM)):
            # C++ indexing: [channel * MAX_EDGE + edge]
            idx_cpp = channel_idx * MAX_EDGE + edge_idx
            
            # Python indexing: [batch, channel, edge, spatial]
            idx_py = (0, channel_idx, edge_idx, 0)
            
            cpp_val = net_out_cpp[idx_cpp]
            py_val = net_out_py[idx_py]
            diff = abs(cpp_val - py_val)
            
            print(f"  {channel_idx:<10} {cpp_val:<15.6f} {py_val:<15.6f} {diff:<15.6e}")
    
    # Print summary statistics for all values
    print(f"\nSummary Statistics (all {DIM * MAX_EDGE} values):")
    
    # Reshape C++ to match Python for easier comparison
    net_out_cpp_reshaped = net_out_cpp.reshape(DIM, MAX_EDGE)
    net_out_py_reshaped = net_out_py[0, :, :, 0]  # Remove batch and spatial dims
    
    diff_all = np.abs(net_out_cpp_reshaped - net_out_py_reshaped)
    print(f"  Max difference: {np.max(diff_all):.6e}")
    print(f"  Mean difference: {np.mean(diff_all):.6e}")
    print(f"  Std difference: {np.std(diff_all):.6e}")
    
    # Count mismatches
    mismatches = np.sum(diff_all > 1e-5)
    total = diff_all.size
    print(f"  Mismatches (>1e-5): {mismatches}/{total} ({100*mismatches/total:.2f}%)")

def main() -> int:
    """Main function to compare C++ and Python ONNX update model inference outputs.
    
    Returns:
        Exit code: 0 if all outputs match, 1 otherwise.
    
    Workflow:
        1. Parse command line arguments (update model path)
        2. Load test metadata (num_active, MAX_EDGE, DIM, CORR_DIM)
        3. Load C++ generated inputs (net, inp, corr, ii, jj, kk)
        4. Load ONNX model and prepare inputs
        5. Run Python ONNX inference
        6. Save Python outputs to binary files
        7. Load C++ outputs from binary files
        8. Compare all outputs (net_out, d_out, w_out)
        9. Print sample values for debugging
        10. Return exit code based on comparison results
    
    Note:
        Expects C++ test program to have generated:
        - test_metadata.txt (test parameters)
        - test_net_input.bin, test_inp_input.bin, test_corr_input.bin
        - test_ii_input.bin, test_jj_input.bin, test_kk_input.bin (indices)
        - test_net_out_cpp.bin, test_d_out_cpp.bin, test_w_out_cpp.bin (C++ outputs)
    """
    if len(sys.argv) < 2:
        print("Usage: python3 compare_update_onnx_outputs.py <update_model.onnx>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_metadata()
    num_active = metadata['num_active']
    MAX_EDGE = metadata['MAX_EDGE']
    DIM = metadata['DIM']
    CORR_DIM = metadata['CORR_DIM']
    
    print(f"Test parameters:")
    print(f"  num_active: {num_active}")
    print(f"  MAX_EDGE: {MAX_EDGE}")
    print(f"  DIM: {DIM}")
    print(f"  CORR_DIM: {CORR_DIM}")
    
    # Load C++ inputs
    print("\nLoading C++ inputs...")
    net_input_cpp = load_binary_file('test_net_input.bin')
    inp_input_cpp = load_binary_file('test_inp_input.bin')
    corr_input_cpp = load_binary_file('test_corr_input.bin')
    ii_input_cpp = load_binary_file('test_ii_input.bin')
    jj_input_cpp = load_binary_file('test_jj_input.bin')
    kk_input_cpp = load_binary_file('test_kk_input.bin')
    
    print(f"  net_input shape: {net_input_cpp.shape} (expected: {1 * DIM * MAX_EDGE * 1})")
    print(f"  inp_input shape: {inp_input_cpp.shape} (expected: {1 * DIM * MAX_EDGE * 1})")
    print(f"  corr_input shape: {corr_input_cpp.shape} (expected: {1 * CORR_DIM * MAX_EDGE * 1})")
    
    # Load ONNX model
    print(f"\nLoading ONNX model: {model_path}")
    session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
    
    # Prepare inputs for Python ONNX Runtime
    # Inputs are already in [1, DIM, H, 1] format from C++
    inputs_dict = {
        input_names[0]: net_input_cpp.reshape(1, DIM, MAX_EDGE, 1).astype(np.float32),
        input_names[1]: inp_input_cpp.reshape(1, DIM, MAX_EDGE, 1).astype(np.float32),
        input_names[2]: corr_input_cpp.reshape(1, CORR_DIM, MAX_EDGE, 1).astype(np.float32),
        input_names[3]: ii_input_cpp.reshape(1, 1, MAX_EDGE, 1).astype(np.int32),
        input_names[4]: jj_input_cpp.reshape(1, 1, MAX_EDGE, 1).astype(np.int32),
        input_names[5]: kk_input_cpp.reshape(1, 1, MAX_EDGE, 1).astype(np.int32),
    }
    
    # Run Python ONNX inference
    print("\nRunning Python ONNX inference...")
    outputs_py = session.run(output_names, inputs_dict)
    
    net_out_py = outputs_py[0]  # [1, DIM, MAX_EDGE, 1]
    d_out_py = outputs_py[1]    # [1, 2, MAX_EDGE, 1]
    w_out_py = outputs_py[2]    # [1, 2, MAX_EDGE, 1]
    
    print(f"Python outputs:")
    print(f"  net_out shape: {net_out_py.shape}")
    print(f"  d_out shape: {d_out_py.shape}")
    print(f"  w_out shape: {w_out_py.shape}")
    
    # Save Python outputs
    print("\nSaving Python outputs...")
    net_out_py.flatten().astype(np.float32).tofile('test_net_out_py.bin')
    d_out_py.flatten().astype(np.float32).tofile('test_d_out_py.bin')
    w_out_py.flatten().astype(np.float32).tofile('test_w_out_py.bin')
    
    # Load C++ outputs
    print("\nLoading C++ outputs...")
    net_out_cpp = load_binary_file('test_net_out_cpp.bin')
    d_out_cpp = load_binary_file('test_d_out_cpp.bin')
    w_out_cpp = load_binary_file('test_w_out_cpp.bin')
    
    # Compare outputs
    print("\n" + "="*60)
    print("COMPARING OUTPUTS")
    print("="*60)
    
    all_match = True
    
    # Compare net_out
    all_match &= compare_outputs(
        net_out_cpp, net_out_py.flatten(),
        "net_out",
        (1, DIM, MAX_EDGE, 1),
        atol=1e-4
    )
    
    # Compare d_out
    all_match &= compare_outputs(
        d_out_cpp, d_out_py.flatten(),
        "d_out",
        (1, 2, MAX_EDGE, 1),
        atol=1e-4
    )
    
    # Compare w_out
    all_match &= compare_outputs(
        w_out_cpp, w_out_py.flatten(),
        "w_out",
        (1, 2, MAX_EDGE, 1),
        atol=1e-4
    )
    
    # Summary
    print("\n" + "="*60)
    if all_match:
        print("✅ ALL OUTPUTS MATCH! C++ ONNX inference is correct.")
    else:
        print("❌ OUTPUTS DO NOT MATCH! There may be an issue with C++ ONNX inference.")
    print("="*60)
    
    # Print sample net_out values (detailed comparison)
    print_sample_net_out_values(net_out_cpp, net_out_py, DIM, MAX_EDGE, num_samples=5)
    
    # Print sample delta values (first 3 edges)
    print("\n" + "="*60)
    print("SAMPLE DELTA VALUES (d_out) - First 3 edges")
    print("="*60)
    print(f"{'Edge':<10} {'C++ (x, y)':<25} {'Python (x, y)':<25} {'Difference':<25}")
    print(f"{'-'*10} {'-'*25} {'-'*25} {'-'*25}")
    for e in range(min(3, num_active)):
        idx_cpp_x = 0 * MAX_EDGE + e
        idx_cpp_y = 1 * MAX_EDGE + e
        idx_py_x = (0, 0, e, 0)
        idx_py_y = (0, 1, e, 0)
        
        cpp_x = d_out_cpp[idx_cpp_x]
        cpp_y = d_out_cpp[idx_cpp_y]
        py_x = d_out_py[idx_py_x]
        py_y = d_out_py[idx_py_y]
        diff_x = abs(cpp_x - py_x)
        diff_y = abs(cpp_y - py_y)
        
        print(f"{e:<10} ({cpp_x:.6f}, {cpp_y:.6f}){'':<10} ({py_x:.6f}, {py_y:.6f}){'':<10} ({diff_x:.6e}, {diff_y:.6e})")
    
    # Print sample weight values (first 3 edges)
    print("\n" + "="*60)
    print("SAMPLE WEIGHT VALUES (w_out) - First 3 edges")
    print("="*60)
    print(f"{'Edge':<10} {'C++ (w0, w1)':<25} {'Python (w0, w1)':<25} {'Difference':<25}")
    print(f"{'-'*10} {'-'*25} {'-'*25} {'-'*25}")
    for e in range(min(3, num_active)):
        idx_cpp_w0 = 0 * MAX_EDGE + e
        idx_cpp_w1 = 1 * MAX_EDGE + e
        idx_py_w0 = (0, 0, e, 0)
        idx_py_w1 = (0, 1, e, 0)
        
        cpp_w0 = w_out_cpp[idx_cpp_w0]
        cpp_w1 = w_out_cpp[idx_cpp_w1]
        py_w0 = w_out_py[idx_py_w0]
        py_w1 = w_out_py[idx_py_w1]
        diff_w0 = abs(cpp_w0 - py_w0)
        diff_w1 = abs(cpp_w1 - py_w1)
        
        print(f"{e:<10} ({cpp_w0:.6f}, {cpp_w1:.6f}){'':<10} ({py_w0:.6f}, {py_w1:.6f}){'':<10} ({diff_w0:.6e}, {diff_w1:.6e})")
    
    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())

