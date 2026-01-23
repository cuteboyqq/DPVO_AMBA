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
from typing import Tuple, List

def resize_nearest_neighbor_cpp_style(img: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
    """
    Resize image using nearest neighbor interpolation, matching C++ implementation exactly.
    
    Args:
        img: Input image array in HWC format (Height, Width, Channels) or HW format.
        target_H: Target height for the resized image.
        target_W: Target width for the resized image.
    
    Returns:
        Resized image array with shape (target_H, target_W, C) or (target_H, target_W).
    
    Note:
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

def load_image(image_path: str, use_bgr: bool = False, match_python_dpvo: bool = True) -> Tuple[np.ndarray, int, int]:
    """Load and preprocess image for ONNX inference.
    
    Args:
        image_path: Path to the input image file (supports formats readable by OpenCV).
        use_bgr: If True, keep BGR color format (don't convert to RGB). 
                 If False, convert BGR to RGB (default). OpenCV loads images as BGR.
        match_python_dpvo: If True, use Python DPVO preprocessing (OpenCV resize with INTER_LINEAR).
                          If False, use C++ preprocessing (nearest neighbor interpolation).
    
    Returns:
        Tuple containing:
            - Preprocessed image array in NCHW format [1, C, H, W] with values normalized to [-0.5, 1.5]
            - Original image height (int)
            - Original image width (int)
    
    Note:
        Normalization formula: 2 * (image / 255.0) - 0.5
        Model input size is typically 528x960 (HxW).
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

def run_fnet_inference(session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
    """Run FNet inference.
    
    Args:
        session: ONNX Runtime inference session for the FNet model.
        input_data: Preprocessed input image in NCHW format [1, 3, H, W].
    
    Returns:
        FNet output feature map in NCHW format [1, 128, H_out, W_out].
        Output is at 1/4 resolution of input (H_out = H/4, W_out = W/4).
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    fnet_out = outputs[0]  # Shape: [1, 128, H, W] (NCHW)
    
    return fnet_out

def run_inet_inference(session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
    """Run INet inference.
    
    Args:
        session: ONNX Runtime inference session for the INet model.
        input_data: Preprocessed input image in NCHW format [1, 3, H, W].
    
    Returns:
        INet output feature map in NCHW format [1, 384, H_out, W_out].
        Output is at 1/4 resolution of input (H_out = H/4, W_out = W/4).
    """
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    inet_out = outputs[0]  # Shape: [1, 384, H, W] (NCHW)
    
    return inet_out

def compare_outputs(cpp_data: np.ndarray, py_data: np.ndarray, name: str, tolerance: float = 1e-5) -> bool:
    """Compare C++ and Python outputs.
    
    Args:
        cpp_data: C++ output array in CHW format [C, H, W].
        py_data: Python output array in NCHW format [1, C, H, W] or CHW format [C, H, W].
                 Batch dimension will be removed if present.
        name: Name/description of the comparison (for logging purposes).
        tolerance: Maximum allowed difference between values to be considered matching (default: 1e-5).
    
    Returns:
        True if outputs match within tolerance, False otherwise.
    
    Note:
        Prints detailed comparison statistics including max difference, mean difference,
        and percentage of elements that differ beyond tolerance.
    """
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

def parse_arguments() -> Tuple[str, str, str, str, str]:
    """Parse command line arguments and return paths.
    
    Returns:
        Tuple containing:
            - image_path: Path to input image file
            - fnet_model_path: Path to FNet ONNX model file
            - inet_model_path: Path to INet ONNX model file
            - fnet_cpp_bin: Path to C++ FNet output binary file (default: "fnet_frame0.bin")
            - inet_cpp_bin: Path to C++ INet output binary file (default: "inet_frame0.bin")
    
    Note:
        Exits with error code 1 if required arguments are missing.
        Optional arguments (fnet_cpp_bin, inet_cpp_bin) default to frame0 files.
    """
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
    
    return image_path, fnet_model_path, inet_model_path, fnet_cpp_bin, inet_cpp_bin

def validate_inputs(image_path: str, fnet_model_path: str, inet_model_path: str) -> None:
    """Validate that all input files exist.
    
    Args:
        image_path: Path to input image file to validate.
        fnet_model_path: Path to FNet ONNX model file to validate.
        inet_model_path: Path to INet ONNX model file to validate.
    
    Note:
        Exits with error code 1 if any file is missing.
        Prints error messages indicating which files are not found.
    """
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Image file not found: {image_path}")
        sys.exit(1)
    if not os.path.exists(fnet_model_path):
        print(f"❌ ERROR: FNet model file not found: {fnet_model_path}")
        sys.exit(1)
    if not os.path.exists(inet_model_path):
        print(f"❌ ERROR: INet model file not found: {inet_model_path}")
        sys.exit(1)

def print_header(image_path: str, fnet_model_path: str, inet_model_path: str, fnet_cpp_bin: str, inet_cpp_bin: str) -> None:
    """Print comparison header with file paths.
    
    Args:
        image_path: Path to input image file (for display).
        fnet_model_path: Path to FNet ONNX model file (for display).
        inet_model_path: Path to INet ONNX model file (for display).
        fnet_cpp_bin: Path to C++ FNet output binary file (for display).
        inet_cpp_bin: Path to C++ INet output binary file (for display).
    
    Note:
        Prints a formatted header showing all file paths being used in the comparison.
    """
    print("=" * 80)
    print("🔍 ONNX Inference Output Comparison")
    print("=" * 80)
    print(f"📷 Image: {image_path}")
    print(f"🧠 FNet model: {fnet_model_path}")
    print(f"🧠 INet model: {inet_model_path}")
    print(f"📂 C++ FNet output: {fnet_cpp_bin}")
    print(f"📂 C++ INet output: {inet_cpp_bin}")
    print()

def load_and_preprocess_images(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess images with different methods.
    
    Args:
        image_path: Path to input image file to load and preprocess.
    
    Returns:
        Tuple containing three preprocessed image arrays (all in NCHW format [1, C, H, W]):
            - input_data_python_dpvo: Preprocessed using Python DPVO method (INTER_LINEAR resize, RGB)
            - input_data_cpp_bgr: Preprocessed using C++ method (nearest neighbor resize, BGR)
            - input_data_cpp_rgb: Preprocessed using C++ method (nearest neighbor resize, RGB)
    
    Note:
        All three preprocessing methods are tested to identify which matches C++ ONNX inference.
        Prints preprocessing information and input shapes for each method.
    """
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
    
    return input_data_python_dpvo, input_data_cpp_bgr, input_data_cpp_rgb

def load_onnx_models(fnet_model_path: str, inet_model_path: str) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
    """Load ONNX models and print model information.
    
    Args:
        fnet_model_path: Path to FNet ONNX model file (.onnx).
        inet_model_path: Path to INet ONNX model file (.onnx).
    
    Returns:
        Tuple containing:
            - fnet_session: ONNX Runtime inference session for FNet model
            - inet_session: ONNX Runtime inference session for INet model
    
    Note:
        Uses CPUExecutionProvider for inference.
        Prints model input/output shapes for verification.
    """
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
    
    return fnet_session, inet_session

def run_all_inferences(fnet_session: ort.InferenceSession, inet_session: ort.InferenceSession,
                       input_data_python_dpvo: np.ndarray, input_data_cpp_bgr: np.ndarray,
                       input_data_cpp_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run all inference variations and return all outputs.
    
    Args:
        fnet_session: ONNX Runtime inference session for FNet model.
        inet_session: ONNX Runtime inference session for INet model.
        input_data_python_dpvo: Preprocessed image using Python DPVO method [1, C, H, W].
        input_data_cpp_bgr: Preprocessed image using C++ method with BGR format [1, C, H, W].
        input_data_cpp_rgb: Preprocessed image using C++ method with RGB format [1, C, H, W].
    
    Returns:
        Tuple containing six output arrays (all in NCHW format):
            - fnet_py_dpvo: FNet output with Python DPVO preprocessing [1, 128, H_out, W_out]
            - fnet_py_cpp_bgr: FNet output with C++ preprocessing (BGR) [1, 128, H_out, W_out]
            - fnet_py_cpp_rgb: FNet output with C++ preprocessing (RGB) [1, 128, H_out, W_out]
            - inet_py_dpvo: INet output with Python DPVO preprocessing [1, 384, H_out, W_out]
            - inet_py_cpp_bgr: INet output with C++ preprocessing (BGR) [1, 384, H_out, W_out]
            - inet_py_cpp_rgb: INet output with C++ preprocessing (RGB) [1, 384, H_out, W_out]
    
    Note:
        Runs inference for both FNet and INet models with all three preprocessing methods.
        Prints output shapes for each inference run.
    """
    print("🚀 Running inference...")
    
    # FNet inference
    print("  🐍 FNet inference (Python DPVO preprocessing)...")
    fnet_py_dpvo = run_fnet_inference(fnet_session, input_data_python_dpvo)
    print(f"    ✅ Output shape: {fnet_py_dpvo.shape} (NCHW)")
    
    print("  ⚙️  FNet inference (C++ preprocessing, BGR)...")
    fnet_py_cpp_bgr = run_fnet_inference(fnet_session, input_data_cpp_bgr)
    print(f"    ✅ Output shape: {fnet_py_cpp_bgr.shape} (NCHW)")
    
    print("  ⚙️  FNet inference (C++ preprocessing, RGB)...")
    fnet_py_cpp_rgb = run_fnet_inference(fnet_session, input_data_cpp_rgb)
    print(f"    ✅ Output shape: {fnet_py_cpp_rgb.shape} (NCHW)")
    print()
    
    # INet inference
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
    
    return fnet_py_dpvo, fnet_py_cpp_bgr, fnet_py_cpp_rgb, inet_py_dpvo, inet_py_cpp_bgr, inet_py_cpp_rgb

def save_python_outputs(fnet_py_dpvo: np.ndarray, fnet_py_cpp_bgr: np.ndarray, fnet_py_cpp_rgb: np.ndarray,
                        inet_py_dpvo: np.ndarray, inet_py_cpp_bgr: np.ndarray, inet_py_cpp_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Save all Python outputs to binary files.
    
    Args:
        fnet_py_dpvo: FNet output with Python DPVO preprocessing [1, C, H, W].
        fnet_py_cpp_bgr: FNet output with C++ preprocessing (BGR) [1, C, H, W].
        fnet_py_cpp_rgb: FNet output with C++ preprocessing (RGB) [1, C, H, W].
        inet_py_dpvo: INet output with Python DPVO preprocessing [1, C, H, W].
        inet_py_cpp_bgr: INet output with C++ preprocessing (BGR) [1, C, H, W].
        inet_py_cpp_rgb: INet output with C++ preprocessing (RGB) [1, C, H, W].
    
    Returns:
        Tuple containing:
            - fnet_py_dpvo_chw: FNet output in CHW format [C, H, W] (batch dimension removed)
            - inet_py_dpvo_chw: INet output in CHW format [C, H, W] (batch dimension removed)
    
    Note:
        Saves all outputs to binary files (.bin) for later comparison:
        - fnet_py_dpvo_frame0.bin, inet_py_dpvo_frame0.bin (Python DPVO preprocessing)
        - fnet_py_cpp_bgr_frame0.bin, inet_py_cpp_bgr_frame0.bin (C++ preprocessing, BGR)
        - fnet_py_cpp_rgb_frame0.bin, inet_py_cpp_rgb_frame0.bin (C++ preprocessing, RGB)
        Prints file paths and sizes for each saved file.
    """
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
    
    return fnet_py_dpvo_chw, inet_py_dpvo_chw

def load_cpp_outputs(fnet_cpp_bin: str, inet_cpp_bin: str, fnet_py_dpvo_chw: np.ndarray, inet_py_dpvo_chw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Load C++ outputs from binary files.
    
    Args:
        fnet_cpp_bin: Path to C++ FNet output binary file (.bin).
        inet_cpp_bin: Path to C++ INet output binary file (.bin).
        fnet_py_dpvo_chw: Python FNet output in CHW format [C, H, W] (used to infer dimensions).
        inet_py_dpvo_chw: Python INet output in CHW format [C, H, W] (used to infer dimensions).
    
    Returns:
        Tuple containing:
            - fnet_cpp: C++ FNet output array in CHW format [C, H, W]
            - inet_cpp: C++ INet output array in CHW format [C, H, W]
    
    Note:
        Exits with error code 1 if binary files are not found.
        Uses Python output shapes to reshape C++ binary data (assumes same dimensions).
        Prints loaded file information including shapes and sizes.
    """
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
    
    return fnet_cpp, inet_cpp

def run_comparisons(fnet_cpp: np.ndarray, inet_cpp: np.ndarray, fnet_py_dpvo: np.ndarray,
                    fnet_py_cpp_bgr: np.ndarray, fnet_py_cpp_rgb: np.ndarray,
                    inet_py_dpvo: np.ndarray, inet_py_cpp_bgr: np.ndarray, inet_py_cpp_rgb: np.ndarray) -> Tuple[bool, bool, bool, bool, bool, bool]:
    """Run all comparisons and return match results.
    
    Args:
        fnet_cpp: C++ FNet output array in CHW format [C, H, W].
        inet_cpp: C++ INet output array in CHW format [C, H, W].
        fnet_py_dpvo: Python FNet output with DPVO preprocessing [1, C, H, W] or [C, H, W].
        fnet_py_cpp_bgr: Python FNet output with C++ preprocessing (BGR) [1, C, H, W] or [C, H, W].
        fnet_py_cpp_rgb: Python FNet output with C++ preprocessing (RGB) [1, C, H, W] or [C, H, W].
        inet_py_dpvo: Python INet output with DPVO preprocessing [1, C, H, W] or [C, H, W].
        inet_py_cpp_bgr: Python INet output with C++ preprocessing (BGR) [1, C, H, W] or [C, H, W].
        inet_py_cpp_rgb: Python INet output with C++ preprocessing (RGB) [1, C, H, W] or [C, H, W].
    
    Returns:
        Tuple containing six boolean match results:
            - fnet_match_dpvo: True if C++ FNet matches Python DPVO preprocessing
            - inet_match_dpvo: True if C++ INet matches Python DPVO preprocessing
            - fnet_match_cpp_bgr: True if C++ FNet matches Python C++ preprocessing (BGR)
            - inet_match_cpp_bgr: True if C++ INet matches Python C++ preprocessing (BGR)
            - fnet_match_cpp_rgb: True if C++ FNet matches Python C++ preprocessing (RGB)
            - inet_match_cpp_rgb: True if C++ INet matches Python C++ preprocessing (RGB)
    
    Note:
        Compares C++ outputs against all Python preprocessing variations.
        Prints detailed comparison results for each pairing.
        Uses tolerance of 1e-5 for value matching.
    """
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
    
    return fnet_match_dpvo, inet_match_dpvo, fnet_match_cpp_bgr, inet_match_cpp_bgr, fnet_match_cpp_rgb, inet_match_cpp_rgb

def print_summary(fnet_match_dpvo: bool, inet_match_dpvo: bool, fnet_match_cpp_bgr: bool,
                  inet_match_cpp_bgr: bool, fnet_match_cpp_rgb: bool, inet_match_cpp_rgb: bool) -> int:
    """Print final summary and return exit code.
    
    Args:
        fnet_match_dpvo: True if C++ FNet matches Python DPVO preprocessing.
        inet_match_dpvo: True if C++ INet matches Python DPVO preprocessing.
        fnet_match_cpp_bgr: True if C++ FNet matches Python C++ preprocessing (BGR).
        inet_match_cpp_bgr: True if C++ INet matches Python C++ preprocessing (BGR).
        fnet_match_cpp_rgb: True if C++ FNet matches Python C++ preprocessing (RGB).
        inet_match_cpp_rgb: True if C++ INet matches Python C++ preprocessing (RGB).
    
    Returns:
        Exit code: 0 if any preprocessing method matches perfectly, 1 otherwise.
    
    Note:
        Prints a formatted summary table showing match status for all comparisons.
        Determines which preprocessing method (if any) matches C++ outputs.
        Provides guidance on which preprocessing method to use based on results.
    """
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

def main() -> int:
    """Main function to orchestrate the ONNX output comparison workflow.
    
    Returns:
        Exit code: 0 on success, 1 on failure or no match found.
    
    Workflow:
        1. Parse command line arguments
        2. Validate input files exist
        3. Print comparison header
        4. Load and preprocess images with different methods
        5. Load ONNX models
        6. Run all inference variations
        7. Save Python outputs to binary files
        8. Load C++ outputs from binary files
        9. Run comparisons between C++ and Python outputs
        10. Print summary and return exit code
    """
    # Step 1: Parse arguments
    image_path, fnet_model_path, inet_model_path, fnet_cpp_bin, inet_cpp_bin = parse_arguments()
    
    # Step 2: Validate inputs
    validate_inputs(image_path, fnet_model_path, inet_model_path)
    
    # Step 3: Print header
    print_header(image_path, fnet_model_path, inet_model_path, fnet_cpp_bin, inet_cpp_bin)
    
    # Step 4: Load and preprocess images
    input_data_python_dpvo, input_data_cpp_bgr, input_data_cpp_rgb = load_and_preprocess_images(image_path)
    
    # Step 5: Load ONNX models
    fnet_session, inet_session = load_onnx_models(fnet_model_path, inet_model_path)
    
    # Step 6: Run all inferences
    fnet_py_dpvo, fnet_py_cpp_bgr, fnet_py_cpp_rgb, inet_py_dpvo, inet_py_cpp_bgr, inet_py_cpp_rgb = \
        run_all_inferences(fnet_session, inet_session, input_data_python_dpvo, input_data_cpp_bgr, input_data_cpp_rgb)
    
    # Step 7: Save Python outputs
    fnet_py_dpvo_chw, inet_py_dpvo_chw = save_python_outputs(
        fnet_py_dpvo, fnet_py_cpp_bgr, fnet_py_cpp_rgb,
        inet_py_dpvo, inet_py_cpp_bgr, inet_py_cpp_rgb
    )
    
    # Step 8: Load C++ outputs
    fnet_cpp, inet_cpp = load_cpp_outputs(fnet_cpp_bin, inet_cpp_bin, fnet_py_dpvo_chw, inet_py_dpvo_chw)
    
    # Step 9: Run comparisons
    fnet_match_dpvo, inet_match_dpvo, fnet_match_cpp_bgr, inet_match_cpp_bgr, fnet_match_cpp_rgb, inet_match_cpp_rgb = \
        run_comparisons(fnet_cpp, inet_cpp, fnet_py_dpvo, fnet_py_cpp_bgr, fnet_py_cpp_rgb,
                       inet_py_dpvo, inet_py_cpp_bgr, inet_py_cpp_rgb)
    
    # Step 10: Print summary and return
    return print_summary(fnet_match_dpvo, inet_match_dpvo, fnet_match_cpp_bgr, inet_match_cpp_bgr,
                        fnet_match_cpp_rgb, inet_match_cpp_rgb)

if __name__ == "__main__":
    sys.exit(main())

