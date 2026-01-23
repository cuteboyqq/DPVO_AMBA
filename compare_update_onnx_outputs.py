#!/usr/bin/env python3
"""
Compare C++ and Python ONNX update model inference outputs
"""

import numpy as np
import onnxruntime as ort
import sys
from pathlib import Path

def load_binary_file(filename, dtype=np.float32):
    """Load binary file as numpy array"""
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def load_metadata():
    """Load test metadata"""
    metadata = {}
    with open('test_metadata.txt', 'r') as f:
        for line in f:
            key, value = line.strip().split('=')
            metadata[key] = int(value)
    return metadata

def compare_outputs(cpp_output, py_output, name, shape, atol=1e-5):
    """Compare C++ and Python outputs"""
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

def main():
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
    
    # Print sample values for first few edges
    print("\nSample values (first 3 edges, first 5 channels of net_out):")
    print("Edge 0:")
    for c in range(min(5, DIM)):
        idx_cpp = c * MAX_EDGE + 0
        idx_py = (0, c, 0, 0)
        print(f"  Channel {c}: C++={net_out_cpp[idx_cpp]:.6f}, Python={net_out_py[idx_py]:.6f}")
    
    print("\nSample delta values (first 3 edges):")
    for e in range(min(3, num_active)):
        idx_cpp_x = 0 * MAX_EDGE + e
        idx_cpp_y = 1 * MAX_EDGE + e
        idx_py_x = (0, 0, e, 0)
        idx_py_y = (0, 1, e, 0)
        print(f"  Edge {e}: C++=({d_out_cpp[idx_cpp_x]:.6f}, {d_out_cpp[idx_cpp_y]:.6f}), "
              f"Python=({d_out_py[idx_py_x]:.6f}, {d_out_py[idx_py_y]:.6f})")
    
    return 0 if all_match else 1

if __name__ == "__main__":
    sys.exit(main())

