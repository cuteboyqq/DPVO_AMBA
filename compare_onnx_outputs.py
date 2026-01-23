#!/usr/bin/env python3
"""
Compare C++ ONNX inference outputs with Python ONNX inference outputs.

This script:
1. Loads C++ outputs from fnet_frame0.bin and inet_frame0.bin
2. Runs Python ONNX inference on the same input image
3. Saves Python outputs to fnet_py_frame0.bin and inet_py_frame0.bin
4. Compares C++ and Python outputs and reports differences
"""

import numpy as np
import onnxruntime as ort
import cv2
import sys
import os
from pathlib import Path

def resize_nearest_neighbor_cpp_style(img, target_H, target_W):
    """
    Resize image using nearest neighbor interpolation, matching C++ implementation exactly.
    
    C++ code does:
        scale_x = W / target_W
        scale_y = H / target_H
        src_x = int(x * scale_x)
        src_y = int(y * scale_y)
        src_x = clamp(src_x, 0, W-1)
        src_y = clamp(src_y, 0, H-1)
    """
    H, W = img.shape[:2]
    scale_x = float(W) / float(target_W)
    scale_y = float(H) / float(target_H)
    
    if len(img.shape) == 3:
        C = img.shape[2]
        resized = np.zeros((target_H, target_W, C), dtype=img.dtype)
        for y in range(target_H):
            for x in range(target_W):
                src_x = int(x * scale_x)
                src_y = int(y * scale_y)
                src_x = max(0, min(src_x, W - 1))
                src_y = max(0, min(src_y, H - 1))
                resized[y, x] = img[src_y, src_x]
    else:
        resized = np.zeros((target_H, target_W), dtype=img.dtype)
        for y in range(target_H):
            for x in range(target_W):
                src_x = int(x * scale_x)
                src_y = int(y * scale_y)
                src_x = max(0, min(src_x, W - 1))
                src_y = max(0, min(src_y, H - 1))
                resized[y, x] = img[src_y, src_x]
    
    return resized

def load_image(image_path, use_bgr=False, match_python_dpvo=True):
    """Load and preprocess image for ONNX inference.
    
    Args:
        image_path: Path to image file
        use_bgr: If True, keep BGR format (don't convert to RGB). 
                 If False, convert BGR to RGB (default).
        match_python_dpvo: If True, use Python DPVO preprocessing (OpenCV resize with INTER_LINEAR).
                          If False, use C++ preprocessing (nearest neighbor).
    """
    # Load image (OpenCV loads as BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Get dimensions
    H, W, C = img.shape
    
    # Resize to model input size (typically 528x960)
    model_H = 528
    model_W = 960
    
    if match_python_dpvo:
        # Python DPVO uses OpenCV resize with INTER_LINEAR (default) or INTER_AREA
        # For image streams: cv2.resize(image, None, fx=0.5, fy=0.5) uses INTER_LINEAR
        # For video streams: cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # We'll use INTER_LINEAR to match image stream behavior
        img_resized = cv2.resize(img, (model_W, model_H), interpolation=cv2.INTER_LINEAR)
    else:
        # C++ uses nearest neighbor interpolation with exact integer mapping
        img_resized = resize_nearest_neighbor_cpp_style(img, model_H, model_W)
    
    # Convert BGR to RGB if needed
    # Python DPVO expects RGB format (images are converted from BGR to RGB)
    if not use_bgr:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize: Python DPVO uses: image = 2 * (image / 255.0) - 0.5
    # This is equivalent to: (2 * image - 127.5) / 255.0
    img_normalized = 2.0 * (img_resized.astype(np.float32) / 255.0) - 0.5
    
    # Convert from HWC to CHW format
    img_chw = np.transpose(img_normalized, (2, 0, 1))  # [C, H, W]
    
    # Add batch dimension: [1, C, H, W]
    img_nchw = np.expand_dims(img_chw, axis=0)
    
    return img_nchw, H, W

def run_fnet_inference(session, input_data):
    """Run FNet inference."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    fnet_out = outputs[0]  # Shape: [1, 128, H, W] (NCHW)
    
    return fnet_out

def run_inet_inference(session, input_data):
    """Run INet inference."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    inet_out = outputs[0]  # Shape: [1, 384, H, W] (NCHW)
    
    return inet_out

def compare_outputs(cpp_data, py_data, name, tolerance=1e-5):
    """Compare C++ and Python outputs."""
    # Reshape Python output to match C++ format (remove batch dimension)
    if py_data.ndim == 4:
        py_data_chw = py_data[0]  # Remove batch: [C, H, W]
    else:
        py_data_chw = py_data
    
    # Ensure shapes match
    if cpp_data.shape != py_data_chw.shape:
        print(f"  ❌ ERROR: Shape mismatch for {name}:")
        print(f"    ⚙️  C++ shape: {cpp_data.shape}")
        print(f"    🐍 Python shape: {py_data_chw.shape}")
        return False
    
    # Compare values
    diff = np.abs(cpp_data - py_data_chw)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Count differences
    num_different = np.sum(diff > tolerance)
    total_elements = cpp_data.size
    percent_different = (num_different / total_elements) * 100.0
    
    print(f"  🔍 {name} comparison:")
    print(f"    📊 Max difference: {max_diff:.6e}")
    print(f"    📊 Mean difference: {mean_diff:.6e}")
    print(f"    📊 Elements different (>{tolerance}): {num_different}/{total_elements} ({percent_different:.2f}%)")
    
    if max_diff < tolerance:
        print(f"    ✅ PASS: All values match within tolerance ({tolerance})")
        indices = np.where(diff <= tolerance)
        print(f"    📋 Sample differences (first 5):")
        for i in range(min(5, len(indices[0]))):
            sample_idx = tuple(ind[i] for ind in indices)
            print(f"      [{sample_idx}]: C++={cpp_data[sample_idx]:.10f}, Python={py_data_chw[sample_idx]:.10f}, diff={diff[sample_idx]:.10e}")
        return True
    else:
        print(f"    ❌ FAIL: Values differ beyond tolerance ({tolerance})")
        
        # Show some example differences
        if num_different > 0:
            indices = np.where(diff > tolerance)
            print(f"    📋 Sample differences (first 5):")
            for i in range(min(5, len(indices[0]))):
                sample_idx = tuple(ind[i] for ind in indices)
                print(f"      [{sample_idx}]: C++={cpp_data[sample_idx]:.6f}, Python={py_data_chw[sample_idx]:.6f}, diff={diff[sample_idx]:.6e}")
        
        return False

def main():
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("❌ Usage: python compare_onnx_outputs.py <image_path> <fnet_model_path> <inet_model_path> [fnet_cpp_bin] [inet_cpp_bin]")
        print("📝 Example: python compare_onnx_outputs.py frame0.jpg fnet.onnx inet.onnx")
        print("📝 Example: python compare_onnx_outputs.py frame1.jpg fnet.onnx inet.onnx fnet_frame1.bin inet_frame1.bin")
        sys.exit(1)
    
    image_path = sys.argv[1]
    fnet_model_path = sys.argv[2]
    inet_model_path = sys.argv[3]
    
    # Optional: C++ output file names (default to frame0)
    fnet_cpp_bin = sys.argv[4] if len(sys.argv) > 4 else "fnet_frame0.bin"
    inet_cpp_bin = sys.argv[5] if len(sys.argv) > 5 else "inet_frame0.bin"
    
    # Check if input files exist
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(fnet_model_path):
        print(f"❌ ERROR: FNet model file not found: {fnet_model_path}")
        sys.exit(1)
    if not os.path.exists(inet_model_path):
        print(f"❌ ERROR: INet model file not found: {inet_model_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("🔍 ONNX Inference Output Comparison")
    print("=" * 80)
    print(f"📷 Image: {image_path}")
    print(f"🧠 FNet model: {fnet_model_path}")
    print(f"🧠 INet model: {inet_model_path}")
    print(f"📂 C++ FNet output: {fnet_cpp_bin}")
    print(f"📂 C++ INet output: {inet_cpp_bin}")
    print()
    
    # Load and preprocess image
    # Match Python DPVO preprocessing exactly (OpenCV resize with INTER_LINEAR, RGB format)
    print("🖼️  Loading and preprocessing image...")
    print("  📝 NOTE: Using Python DPVO preprocessing (OpenCV resize with INTER_LINEAR, RGB format)")
    print("  ✅ This matches the Python DPVO repository preprocessing flow")
    input_data_python_dpvo, orig_H, orig_W = load_image(image_path, use_bgr=False, match_python_dpvo=True)
    
    # Also try C++ preprocessing for comparison
    print("  🔄 Also testing C++ preprocessing (nearest neighbor, BGR format) for comparison")
    input_data_cpp_bgr, _, _ = load_image(image_path, use_bgr=True, match_python_dpvo=False)
    input_data_cpp_rgb, _, _ = load_image(image_path, use_bgr=False, match_python_dpvo=False)
    
    print(f"  📐 Original image size: {orig_W}x{orig_H}")
    print(f"  🐍 Python DPVO input shape: {input_data_python_dpvo.shape} (NCHW)")
    print(f"  ⚙️  C++ preprocessing (BGR): {input_data_cpp_bgr.shape} (NCHW)")
    print(f"  ⚙️  C++ preprocessing (RGB): {input_data_cpp_rgb.shape} (NCHW)")
    print()
    
    # Load ONNX models
    print("📦 Loading ONNX models...")
    fnet_session = ort.InferenceSession(fnet_model_path, providers=['CPUExecutionProvider'])
    inet_session = ort.InferenceSession(inet_model_path, providers=['CPUExecutionProvider'])
    
    # Get model input/output info
    fnet_input_shape = fnet_session.get_inputs()[0].shape
    fnet_output_shape = fnet_session.get_outputs()[0].shape
    inet_input_shape = inet_session.get_inputs()[0].shape
    inet_output_shape = inet_session.get_outputs()[0].shape
    
    print(f"  🧠 FNet input shape: {fnet_input_shape}")
    print(f"  🧠 FNet output shape: {fnet_output_shape}")
    print(f"  🧠 INet input shape: {inet_input_shape}")
    print(f"  🧠 INet output shape: {inet_output_shape}")
    print()
    
    # Run inference with Python DPVO preprocessing (primary)
    print("🚀 Running inference...")
    print("  🐍 FNet inference (Python DPVO preprocessing)...")
    fnet_py_dpvo = run_fnet_inference(fnet_session, input_data_python_dpvo)
    print(f"    ✅ Output shape: {fnet_py_dpvo.shape} (NCHW)")
    
    # Also run with C++ preprocessing for comparison
    print("  ⚙️  FNet inference (C++ preprocessing, BGR)...")
    fnet_py_cpp_bgr = run_fnet_inference(fnet_session, input_data_cpp_bgr)
    print(f"    ✅ Output shape: {fnet_py_cpp_bgr.shape} (NCHW)")
    
    print("  ⚙️  FNet inference (C++ preprocessing, RGB)...")
    fnet_py_cpp_rgb = run_fnet_inference(fnet_session, input_data_cpp_rgb)
    print(f"    ✅ Output shape: {fnet_py_cpp_rgb.shape} (NCHW)")
    print()
    
    print("  🐍 INet inference (Python DPVO preprocessing)...")
    inet_py_dpvo = run_inet_inference(inet_session, input_data_python_dpvo)
    print(f"    ✅ Output shape: {inet_py_dpvo.shape} (NCHW)")
    
    print("  ⚙️  INet inference (C++ preprocessing, BGR)...")
    inet_py_cpp_bgr = run_inet_inference(inet_session, input_data_cpp_bgr)
    print(f"    ✅ Output shape: {inet_py_cpp_bgr.shape} (NCHW)")
    
    print("  ⚙️  INet inference (C++ preprocessing, RGB)...")
    inet_py_cpp_rgb = run_inet_inference(inet_session, input_data_cpp_rgb)
    print(f"    ✅ Output shape: {inet_py_cpp_rgb.shape} (NCHW)")
    print()
    
    # Save Python outputs to binary files
    print("💾 Saving Python outputs to binary files...")
    
    # Save Python DPVO outputs (primary - matches Python DPVO repository)
    fnet_py_dpvo_chw = fnet_py_dpvo[0]  # Remove batch dimension: [C, H, W]
    fnet_py_dpvo_chw.tofile('fnet_py_dpvo_frame0.bin')
    print(f"  ✅ Saved fnet_py_dpvo_frame0.bin: shape={fnet_py_dpvo_chw.shape} (CHW), size={fnet_py_dpvo_chw.nbytes} bytes")
    
    inet_py_dpvo_chw = inet_py_dpvo[0]  # Remove batch dimension: [C, H, W]
    inet_py_dpvo_chw.tofile('inet_py_dpvo_frame0.bin')
    print(f"  ✅ Saved inet_py_dpvo_frame0.bin: shape={inet_py_dpvo_chw.shape} (CHW), size={inet_py_dpvo_chw.nbytes} bytes")
    
    # Also save C++ preprocessing outputs for comparison
    fnet_py_cpp_bgr_chw = fnet_py_cpp_bgr[0]
    fnet_py_cpp_bgr_chw.tofile('fnet_py_cpp_bgr_frame0.bin')
    print(f"  ✅ Saved fnet_py_cpp_bgr_frame0.bin: shape={fnet_py_cpp_bgr_chw.shape} (CHW), size={fnet_py_cpp_bgr_chw.nbytes} bytes")
    
    fnet_py_cpp_rgb_chw = fnet_py_cpp_rgb[0]
    fnet_py_cpp_rgb_chw.tofile('fnet_py_cpp_rgb_frame0.bin')
    print(f"  ✅ Saved fnet_py_cpp_rgb_frame0.bin: shape={fnet_py_cpp_rgb_chw.shape} (CHW), size={fnet_py_cpp_rgb_chw.nbytes} bytes")
    
    inet_py_cpp_bgr_chw = inet_py_cpp_bgr[0]
    inet_py_cpp_bgr_chw.tofile('inet_py_cpp_bgr_frame0.bin')
    print(f"  ✅ Saved inet_py_cpp_bgr_frame0.bin: shape={inet_py_cpp_bgr_chw.shape} (CHW), size={inet_py_cpp_bgr_chw.nbytes} bytes")
    
    inet_py_cpp_rgb_chw = inet_py_cpp_rgb[0]
    inet_py_cpp_rgb_chw.tofile('inet_py_cpp_rgb_frame0.bin')
    print(f"  ✅ Saved inet_py_cpp_rgb_frame0.bin: shape={inet_py_cpp_rgb_chw.shape} (CHW), size={inet_py_cpp_rgb_chw.nbytes} bytes")
    print()
    
    # Load C++ outputs
    print("📂 Loading C++ outputs...")
    if not os.path.exists(fnet_cpp_bin):
        print(f"  ❌ ERROR: {fnet_cpp_bin} not found. Run C++ code first to generate this file.")
        sys.exit(1)
    if not os.path.exists(inet_cpp_bin):
        print(f"  ❌ ERROR: {inet_cpp_bin} not found. Run C++ code first to generate this file.")
        sys.exit(1)
    
    # Load C++ outputs (CHW format)
    # We need to know the dimensions - get them from Python output shape
    fnet_C, fnet_H, fnet_W = fnet_py_dpvo_chw.shape
    inet_C, inet_H, inet_W = inet_py_dpvo_chw.shape
    
    fnet_cpp = np.fromfile(fnet_cpp_bin, dtype=np.float32)
    fnet_cpp = fnet_cpp.reshape(fnet_C, fnet_H, fnet_W)
    print(f"  ✅ Loaded {fnet_cpp_bin}: shape={fnet_cpp.shape} (CHW), size={fnet_cpp.nbytes} bytes")
    
    inet_cpp = np.fromfile(inet_cpp_bin, dtype=np.float32)
    inet_cpp = inet_cpp.reshape(inet_C, inet_H, inet_W)
    print(f"  ✅ Loaded {inet_cpp_bin}: shape={inet_cpp.shape} (CHW), size={inet_cpp.nbytes} bytes")
    print()
    
    # Compare outputs
    print("=" * 80)
    print("🔍 Comparing Python DPVO preprocessing outputs with C++ outputs...")
    print("=" * 80)
    print("  📝 (This shows if C++ preprocessing matches Python DPVO preprocessing)")
    print()
    
    fnet_match_dpvo = compare_outputs(fnet_cpp, fnet_py_dpvo, "FNet (Python DPVO preprocessing)", tolerance=1e-5)
    print()
    
    inet_match_dpvo = compare_outputs(inet_cpp, inet_py_dpvo, "INet (Python DPVO preprocessing)", tolerance=1e-5)
    print()
    
    print("=" * 80)
    print("🔍 Comparing C++ preprocessing outputs with C++ outputs...")
    print("=" * 80)
    print("  📝 (This shows if C++ preprocessing matches C++ ONNX inference)")
    print()
    
    fnet_match_cpp_bgr = compare_outputs(fnet_cpp, fnet_py_cpp_bgr, "FNet (C++ preprocessing, BGR)", tolerance=1e-5)
    print()
    
    inet_match_cpp_bgr = compare_outputs(inet_cpp, inet_py_cpp_bgr, "INet (C++ preprocessing, BGR)", tolerance=1e-5)
    print()
    
    fnet_match_cpp_rgb = compare_outputs(fnet_cpp, fnet_py_cpp_rgb, "FNet (C++ preprocessing, RGB)", tolerance=1e-5)
    print()
    
    inet_match_cpp_rgb = compare_outputs(inet_cpp, inet_py_cpp_rgb, "INet (C++ preprocessing, RGB)", tolerance=1e-5)
    print()
    
    # Summary
    print("=" * 80)
    print("📊 Summary:")
    print("=" * 80)
    print(f"🐍 Python DPVO preprocessing vs C++ outputs:")
    print(f"  🧠 FNet: {'✅ MATCH' if fnet_match_dpvo else '❌ DIFFER'}")
    print(f"  🧠 INet: {'✅ MATCH' if inet_match_dpvo else '❌ DIFFER'}")
    print()
    print(f"⚙️  C++ preprocessing vs C++ outputs:")
    print(f"  🧠 FNet (BGR): {'✅ MATCH' if fnet_match_cpp_bgr else '❌ DIFFER'}")
    print(f"  🧠 INet (BGR): {'✅ MATCH' if inet_match_cpp_bgr else '❌ DIFFER'}")
    print(f"  🧠 FNet (RGB): {'✅ MATCH' if fnet_match_cpp_rgb else '❌ DIFFER'}")
    print(f"  🧠 INet (RGB): {'✅ MATCH' if inet_match_cpp_rgb else '❌ DIFFER'}")
    print()
    
    if fnet_match_dpvo and inet_match_dpvo:
        print("✅ SUCCESS: Python DPVO preprocessing matches C++ outputs!")
        print("  💡 This means C++ ONNX inference uses the same preprocessing as Python DPVO.")
        return 0
    elif fnet_match_cpp_bgr and inet_match_cpp_bgr:
        print("✅ SUCCESS: C++ preprocessing (BGR) matches C++ outputs!")
        print("  💡 C++ ONNX uses BGR format with nearest neighbor resize.")
        return 0
    elif fnet_match_cpp_rgb and inet_match_cpp_rgb:
        print("✅ SUCCESS: C++ preprocessing (RGB) matches C++ outputs!")
        print("  💡 C++ ONNX uses RGB format with nearest neighbor resize.")
        return 0
    else:
        print("❌ WARNING: No preprocessing method matches perfectly.")
        print("  🔍 Check the differences above. There may be other preprocessing differences.")
        print()
        print("📝 NOTE: Python DPVO outputs are saved to:")
        print("  📄 fnet_py_dpvo_frame0.bin")
        print("  📄 inet_py_dpvo_frame0.bin")
        print("  💡 These can be compared with Python DPVO repository outputs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

