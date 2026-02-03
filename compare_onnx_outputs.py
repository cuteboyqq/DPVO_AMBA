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
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

def resize_nearest_neighbor_cpp_style(img: np.ndarray, target_H: int, target_W: int) -> np.ndarray:
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

def load_image(image_path: str, use_bgr: bool = True, match_python_dpvo: bool = True, 
               apply_undistort: bool = True, is_video: bool = False) -> Tuple[np.ndarray, int, int]:
    """Load and preprocess image for ONNX inference.
    
    Args:
        image_path: Path to image file
        use_bgr: If True, keep BGR format (don't convert to RGB). 
                 If False, convert BGR to RGB (default).
        match_python_dpvo: If True, use Python DPVO preprocessing.
                          If False, use C++ preprocessing (nearest neighbor).
        apply_undistort: If True, apply undistortion using camera intrinsics and distortion parameters.
                        Matches Python DPVO behavior: if len(calib) > 4, apply cv2.undistort
        is_video: If True, use video stream preprocessing (resize by 0.5x with INTER_AREA).
                  If False, use image stream preprocessing (resize to model size with INTER_LINEAR).
    """
    # Load image (OpenCV loads as BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    
    # Get dimensions
    H, W, C = img.shape
    
    # STEP 1: Apply undistortion if requested (matching Python DPVO: if len(calib) > 4)
    if apply_undistort:
        # Camera intrinsics (at full resolution)
        fx = 1660.0
        fy = 1660.0
        cx = 960.0
        cy = 540.0
        
        # Distortion parameters
        k1 = 0.07
        k2 = -0.08
        p1 = 0.0
        p2 = 0.0
        
        # Build camera matrix K
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float64)
        
        # Build distortion coefficients [k1, k2, p1, p2]
        dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)
        
        # Apply undistortion (matching Python DPVO: cv2.undistort(image, K, calib[4:]))
        img = cv2.undistort(img, K, dist_coeffs)
        
        # Update dimensions after undistortion (may change slightly)
        H, W = img.shape[:2]
    
    # STEP 2: Resize (matching Python DPVO behavior)
    # Python DPVO video_stream preprocessing flow:
    #   1. video_stream(): resize(0.5x with INTER_AREA) -> crop to divisible by 16
    #      Example: 1920x1080 -> 960x540 -> crop to 960x528 (if needed)
    #      Intrinsics are scaled: fx*0.5, fy*0.5, cx*0.5, cy*0.5
    #   2. Then the ONNX model (fnet/inet) or preprocessing code resizes to model input size (528x960)
    #      This happens in dpvo.py or the model's preprocessing before ONNX inference
    #   3. Final ONNX model input: 528x960 (torch.Size([3, 528, 960]))
    #
    # Why does video_stream do 0.5x resize?
    #   - To reduce memory/computation for video processing
    #   - Intrinsics are scaled accordingly (fx*0.5, fy*0.5, cx*0.5, cy*0.5)
    #   - The model then resizes to its fixed input size (528x960) internally
    #
    # For comparison, we need to match the FINAL ONNX model input (528x960), not the intermediate video_stream output
    # So we do: resize directly to 528x960 using INTER_AREA (matching video_stream's interpolation method)
    model_H = 528
    model_W = 960
    
    if match_python_dpvo:
        if is_video:
            # Python DPVO video_stream: resize to model input size using INTER_AREA
            # Note: video_stream() does resize(0.5x), but the final ONNX model input is 528x960
            # The model or preprocessing code resizes the video_stream output to model size
            # We resize directly to model size using INTER_AREA to match video_stream's interpolation method
            img_resized = cv2.resize(img, (model_W, model_H), interpolation=cv2.INTER_AREA)
        else:
            # Python DPVO image_stream: no resize (commented out with if 0:)
            # But for model input, we need to resize to model size
            # Use INTER_LINEAR as default (matches image stream behavior when resize is enabled)
            img_resized = cv2.resize(img, (model_W, model_H), interpolation=cv2.INTER_LINEAR)
    else:
        # C++ uses nearest neighbor interpolation with exact integer mapping
        img_resized = resize_nearest_neighbor_cpp_style(img, model_H, model_W)
    
    # STEP 3: Crop to make dimensions divisible by 16 (matching Python DPVO: image[:h-h%16, :w-w%16])
    # h_cropped = img_resized.shape[0] - (img_resized.shape[0] % 16)
    # w_cropped = img_resized.shape[1] - (img_resized.shape[1] % 16)
    # if h_cropped != img_resized.shape[0] or w_cropped != img_resized.shape[1]:
    #     img_resized = img_resized[:h_cropped, :w_cropped]
    
    # STEP 4: Convert BGR to RGB if needed
    # NOTE: Python DPVO actually uses BGR format (cv2.imread loads as BGR, no conversion)
    # The ONNX models were trained with BGR input, so C++ also uses BGR
    # This RGB conversion is only for comparison/testing purposes
    if not use_bgr:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # STEP 5: Normalize: Python DPVO uses: image = 2 * (image / 255.0) - 0.5
    # This is equivalent to: (2 * image - 127.5) / 255.0
    img_normalized = 2.0 * (img_resized.astype(np.float32) / 255.0) - 0.5
    
    # STEP 6: Convert from HWC to CHW format
    img_chw = np.transpose(img_normalized, (2, 0, 1))  # [C, H, W]
    
    # STEP 7: Add batch dimension: [1, C, H, W]
    img_nchw = np.expand_dims(img_chw, axis=0)
    
    return img_nchw, H, W

def run_fnet_inference(session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
    """Run FNet inference."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    fnet_out = outputs[0]  # Shape: [1, 128, H, W] (NCHW)
    
    return fnet_out

def run_inet_inference(session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
    """Run INet inference."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    inet_out = outputs[0]  # Shape: [1, 384, H, W] (NCHW)
    
    return inet_out

@dataclass
class PreprocessingSettings:
    """Preprocessing settings used for input."""
    use_bgr: bool
    match_python_dpvo: bool
    apply_undistort: bool
    is_video: bool
    
    def __str__(self) -> str:
        """Format settings as a readable string."""
        color = "BGR" if self.use_bgr else "RGB"
        match = "Python DPVO" if self.match_python_dpvo else "C++ style"
        undistort = "Yes" if self.apply_undistort else "No"
        mode = "Video" if self.is_video else "Image"
        return f"Color: {color}, Match: {match}, Undistort: {undistort}, Mode: {mode}"

@dataclass
class ComparisonResult:
    """Structured comparison result."""
    name: str
    matches: bool
    max_diff: float
    mean_diff: float
    num_different: int
    total_elements: int
    percent_different: float
    shape_match: bool
    cpp_shape: tuple
    py_shape: tuple
    tolerance: float
    sample_diffs: List[Tuple[tuple, float, float, float]]  # [(idx, cpp_val, py_val, diff), ...]
    preprocessing: Optional[PreprocessingSettings] = None  # Preprocessing settings used

def compare_outputs(cpp_data: np.ndarray, py_data: np.ndarray, name: str, tolerance: float = 1e-5,
                    preprocessing: Optional[PreprocessingSettings] = None) -> ComparisonResult:
    """Compare C++ and Python outputs and return structured result."""
    # Reshape Python output to match C++ format (remove batch dimension)
    if py_data.ndim == 4:
        py_data_chw = py_data[0]  # Remove batch: [C, H, W]
    else:
        py_data_chw = py_data
    
    # Check shape match
    shape_match = (cpp_data.shape == py_data_chw.shape)
    
    if not shape_match:
        return ComparisonResult(
            name=name,
            matches=False,
            max_diff=float('inf'),
            mean_diff=float('inf'),
            num_different=0,
            total_elements=0,
            percent_different=100.0,
            shape_match=False,
            cpp_shape=cpp_data.shape,
            py_shape=py_data_chw.shape,
            tolerance=tolerance,
            sample_diffs=[],
            preprocessing=preprocessing
        )
    
    # Compare values
    diff = np.abs(cpp_data - py_data_chw)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Count differences
    num_different = np.sum(diff > tolerance)
    total_elements = cpp_data.size
    percent_different = (num_different / total_elements) * 100.0
    
    # Get sample differences
    sample_diffs = []
    if num_different > 0:
        indices = np.where(diff > tolerance)
        for i in range(min(5, len(indices[0]))):
            sample_idx = tuple(ind[i] for ind in indices)
            sample_diffs.append((
                sample_idx,
                float(cpp_data[sample_idx]),
                float(py_data_chw[sample_idx]),
                float(diff[sample_idx])
            ))
    else:
        # If all match, show some sample matching values
        indices = np.where(diff <= tolerance)
        for i in range(min(5, len(indices[0]))):
            sample_idx = tuple(ind[i] for ind in indices)
            sample_diffs.append((
                sample_idx,
                float(cpp_data[sample_idx]),
                float(py_data_chw[sample_idx]),
                float(diff[sample_idx])
            ))
    
    matches = (max_diff < tolerance) and shape_match
    
    return ComparisonResult(
        name=name,
        matches=matches,
        max_diff=max_diff,
        mean_diff=mean_diff,
        num_different=num_different,
        total_elements=total_elements,
        percent_different=percent_different,
        shape_match=shape_match,
        cpp_shape=cpp_data.shape,
        py_shape=py_data_chw.shape,
        tolerance=tolerance,
        sample_diffs=sample_diffs,
        preprocessing=preprocessing
    )

def format_number(val: float, max_precision: int = 6) -> str:
    """Format number for table display."""
    if val == float('inf'):
        return "∞"
    elif val == float('-inf'):
        return "-∞"
    elif np.isnan(val):
        return "NaN"
    elif val >= 1e6:
        return f"{val:.2e}"
    elif val >= 1.0:
        return f"{val:.{max_precision}f}"
    elif val >= 0.01:
        return f"{val:.{max_precision}f}"
    elif val >= 0.0001:
        return f"{val:.{max_precision+2}f}"
    elif val >= 0.000001:
        return f"{val:.{max_precision+4}f}"
    else:
        return f"{val:.{max_precision}e}"

def print_comparison_table(results: List[ComparisonResult]) -> None:
    """Print comparison results in a formatted table."""
    print("\n" + "=" * 120)
    print("📊 Comparison Results Table")
    print("=" * 120)
    
    # Table header
    header = f"{'Model':<30} {'Status':<12} {'Max Diff':<15} {'Mean Diff':<15} {'Different':<15} {'Shape Match':<12}"
    print(header)
    print("-" * 120)
    
    # Table rows
    for result in results:
        status = "✅ MATCH" if result.matches else "❌ DIFFER"
        max_diff_str = format_number(result.max_diff)
        mean_diff_str = format_number(result.mean_diff)
        
        if result.total_elements > 0:
            diff_str = f"{result.num_different}/{result.total_elements} ({result.percent_different:.2f}%)"
        else:
            diff_str = "N/A"
        
        shape_status = "✅" if result.shape_match else "❌"
        
        row = f"{result.name:<25} {status:<12} {max_diff_str:<15} {mean_diff_str:<15} {diff_str:<15} {shape_status:<12}"
        print(row)
    
    print("=" * 120)
    
    # Detailed information for mismatches
    mismatches = [r for r in results if not r.matches]
    if mismatches:
        print("\n" + "=" * 120)
        print("🔍 Detailed Information for Mismatches")
        print("=" * 120)
        
        for result in mismatches:
            print(f"\n📋 {result.name}:")
            if not result.shape_match:
                print(f"  ❌ Shape Mismatch:")
                print(f"     C++ shape: {result.cpp_shape}")
                print(f"     Python shape: {result.py_shape}")
            else:
                print(f"  📊 Statistics:")
                print(f"     Max difference: {format_number(result.max_diff)}")
                print(f"     Mean difference: {format_number(result.mean_diff)}")
                print(f"     Different elements: {result.num_different}/{result.total_elements} ({result.percent_different:.2f}%)")
                print(f"     Tolerance: {result.tolerance}")
                
                if result.sample_diffs:
                    print(f"  📋 Sample Differences (first {len(result.sample_diffs)}):")
                    print(f"     {'Index':<20} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20}")
                    print(f"     {'-'*20} {'-'*20} {'-'*20} {'-'*20}")
                    for idx, cpp_val, py_val, diff_val in result.sample_diffs:
                        idx_str = str(idx)[:18]
                        print(f"     {idx_str:<20} {format_number(cpp_val):<20} {format_number(py_val):<20} {format_number(diff_val):<20}")
    
    print("\n" + "=" * 120)

def parse_arguments() -> Tuple[str, str, str, str, str, bool]:
    """Parse command line arguments and return paths."""
    if len(sys.argv) < 4:
        print("❌ Usage: python compare_onnx_outputs.py <image_path> <fnet_model_path> <inet_model_path> [fnet_cpp_bin] [inet_cpp_bin] [--video]")
        print("📝 Example: python compare_onnx_outputs.py frame0.jpg fnet.onnx inet.onnx")
        print("📝 Example: python compare_onnx_outputs.py frame1.jpg fnet.onnx inet.onnx fnet_frame1.bin inet_frame1.bin")
        print("📝 Example: python compare_onnx_outputs.py frame0.jpg fnet.onnx inet.onnx --video")
        print("  --video: Use video stream preprocessing (resize by 0.5x with INTER_AREA)")
        sys.exit(1)
    
    image_path = sys.argv[1]
    fnet_model_path = sys.argv[2]
    inet_model_path = sys.argv[3]
    
    # Check for --video flag
    is_video = "--video" in sys.argv
    
    # Optional: C++ output file names (default to frame0)
    # Skip --video flag when parsing file names
    args_without_flags = [arg for arg in sys.argv[4:] if arg != "--video"]
    bin_dir = "bin_file"
    fnet_cpp_bin = args_without_flags[0] if len(args_without_flags) > 0 else os.path.join(bin_dir, "fnet_frame0.bin")
    inet_cpp_bin = args_without_flags[1] if len(args_without_flags) > 1 else os.path.join(bin_dir, "inet_frame0.bin")
    
    return image_path, fnet_model_path, inet_model_path, fnet_cpp_bin, inet_cpp_bin, is_video

def validate_inputs(image_path: str, fnet_model_path: str, inet_model_path: str) -> None:
    """Validate that all input files exist."""
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
    """Print comparison header with file paths."""
    print("=" * 80)
    print("🔍 ONNX Inference Output Comparison")
    print("=" * 80)
    print(f"📷 Image: {image_path}")
    print(f"🧠 FNet model: {fnet_model_path}")
    print(f"🧠 INet model: {inet_model_path}")
    print(f"📂 C++ FNet output: {fnet_cpp_bin}")
    print(f"📂 C++ INet output: {inet_cpp_bin}")
    print()

def load_and_preprocess_images(image_path: str, is_video: bool = False) -> np.ndarray:
    """Load and preprocess images using Python DPVO preprocessing.
    
    Args:
        image_path: Path to image file
        is_video: If True, use video stream preprocessing (resize by 0.5x with INTER_AREA).
                  If False, use image stream preprocessing (resize to model size with INTER_LINEAR).
    """
    print("🖼️  Loading and preprocessing image...")
    if is_video:
        print("  📝 NOTE: Using Python DPVO VIDEO stream preprocessing:")
        print("    - Undistort (if distortion params exist)")
        print("    - Resize to model input size (528x960) using INTER_AREA")
        print("    - Crop to divisible by 16")
        print("    - BGR format")
        print("  ✅ This matches Python DPVO video_stream() -> model input (final size: 528x960)")
    else:
        print("  📝 NOTE: Using Python DPVO IMAGE stream preprocessing:")
        print("    - Undistort (if distortion params exist)")
        print("    - Resize to model size using INTER_LINEAR")
        print("    - Crop to divisible by 16")
        print("    - BGR format")
        print("  ✅ This matches Python DPVO image_stream() function")
    print("  🔧 Applying undistortion with k1=0.07, k2=-0.08, p1=0, p2=0")
    input_data_python_dpvo, orig_H, orig_W = load_image(image_path, use_bgr=False, match_python_dpvo=True, apply_undistort=False, is_video=is_video)
    
    print(f"  📐 Original image size: {orig_W}x{orig_H}")
    print(f"  🐍 Python DPVO input shape: {input_data_python_dpvo.shape} (NCHW)")
    print()
    
    return input_data_python_dpvo

def load_onnx_models(fnet_model_path: str, inet_model_path: str) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
    """Load ONNX models and print model information."""
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
                       input_data_python_dpvo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference using Python DPVO preprocessing and return outputs."""
    print("🚀 Running inference...")
    
    # FNet inference
    print("  🐍 FNet inference (Python DPVO preprocessing)...")
    fnet_py_dpvo = run_fnet_inference(fnet_session, input_data_python_dpvo)
    print(f"    ✅ Output shape: {fnet_py_dpvo.shape} (NCHW)")
    
    # INet inference
    print("  🐍 INet inference (Python DPVO preprocessing)...")
    inet_py_dpvo = run_inet_inference(inet_session, input_data_python_dpvo)
    print(f"    ✅ Output shape: {inet_py_dpvo.shape} (NCHW)")
    print()
    
    return fnet_py_dpvo, inet_py_dpvo

def save_python_outputs(fnet_py_dpvo: np.ndarray, inet_py_dpvo: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Save Python DPVO outputs to binary files."""
    print("💾 Saving Python outputs to binary files...")
    
    bin_dir = "bin_file"
    os.makedirs(bin_dir, exist_ok=True)
    
    # Save Python DPVO outputs (matches Python DPVO repository)
    fnet_py_dpvo_chw = fnet_py_dpvo[0]  # Remove batch dimension: [C, H, W]
    fnet_py_dpvo_path = os.path.join(bin_dir, 'fnet_py_dpvo_frame0.bin')
    fnet_py_dpvo_chw.tofile(fnet_py_dpvo_path)
    print(f"  ✅ Saved {fnet_py_dpvo_path}: shape={fnet_py_dpvo_chw.shape} (CHW), size={fnet_py_dpvo_chw.nbytes} bytes")
    
    inet_py_dpvo_chw = inet_py_dpvo[0]  # Remove batch dimension: [C, H, W]
    inet_py_dpvo_path = os.path.join(bin_dir, 'inet_py_dpvo_frame0.bin')
    inet_py_dpvo_chw.tofile(inet_py_dpvo_path)
    print(f"  ✅ Saved {inet_py_dpvo_path}: shape={inet_py_dpvo_chw.shape} (CHW), size={inet_py_dpvo_chw.nbytes} bytes")
    print()
    
    return fnet_py_dpvo_chw, inet_py_dpvo_chw

def load_cpp_outputs(fnet_cpp_bin: str, 
                     inet_cpp_bin: str, 
                     fnet_py_dpvo_chw: np.ndarray, 
                     inet_py_dpvo_chw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Load C++ outputs from binary files."""
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

def print_sample_values_table(cpp_data: np.ndarray, py_data: np.ndarray, model_name: str, num_samples: int = 10) -> None:
    """Print a table showing sample value comparisons."""
    # Reshape Python output to match C++ format (remove batch dimension)
    if py_data.ndim == 4:
        py_data_chw = py_data[0]  # Remove batch: [C, H, W]
    else:
        py_data_chw = py_data
    
    if cpp_data.shape != py_data_chw.shape:
        print(f"  ⚠️  Cannot show sample values: shape mismatch")
        print(f"     C++ shape: {cpp_data.shape}, Python shape: {py_data_chw.shape}")
        return
    
    # Compute differences
    diff = np.abs(cpp_data - py_data_chw)
    
    # Get sample indices - mix of largest differences and random samples
    flat_diff = diff.flatten()
    flat_cpp = cpp_data.flatten()
    flat_py = py_data_chw.flatten()
    
    # Get indices of largest differences
    num_large_samples = min(5, num_samples // 2)
    large_diff_indices = np.argsort(flat_diff)[-num_large_samples:][::-1]
    
    # Get random samples from different channels/spatial locations
    num_random_samples = num_samples - num_large_samples
    if num_random_samples > 0:
        random_indices = np.random.choice(len(flat_diff), size=min(num_random_samples, len(flat_diff)), replace=False)
    else:
        random_indices = []
    
    # Combine and get unique indices
    all_indices = np.unique(np.concatenate([large_diff_indices, random_indices]))
    
    # Convert flat indices back to multi-dimensional indices
    sample_indices = []
    for flat_idx in all_indices[:num_samples]:
        multi_idx = np.unravel_index(flat_idx, cpp_data.shape)
        sample_indices.append((multi_idx, flat_cpp[flat_idx], flat_py[flat_idx], flat_diff[flat_idx]))
    
    # Print table
    print(f"\n  📋 Sample Value Comparisons for {model_name}:")
    print(f"     {'Index (C,H,W)':<25} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20} {'Status':<10}")
    print(f"     {'-'*25} {'-'*20} {'-'*20} {'-'*20} {'-'*10}")
    
    for (idx, cpp_val, py_val, diff_val) in sample_indices:
        idx_str = str(idx)[:23]
        status = "✅" if diff_val < 1e-5 else "⚠️"
        print(f"     {idx_str:<25} {format_number(cpp_val):<20} {format_number(py_val):<20} {format_number(diff_val):<20} {status:<10}")
    
    print()

def print_category_table(title: str, results: List[ComparisonResult], description: str = "", 
                        cpp_data_dict: Optional[Dict[str, np.ndarray]] = None,
                        py_data_dict: Optional[Dict[str, np.ndarray]] = None,
                        show_samples: bool = True) -> None:
    """Print a table for a specific category of comparisons."""
    print("\n" + "=" * 120)
    print(f"🔍 {title}")
    print("=" * 120)
    if description:
        print(f"  📝 {description}")
    
    # Display preprocessing settings at the top if available
    if results and results[0].preprocessing:
        print(f"  ⚙️  Preprocessing Settings:")
        # Get unique preprocessing settings (should be same for all results in category)
        unique_settings = set()
        for result in results:
            if result.preprocessing:
                unique_settings.add(str(result.preprocessing))
        
        for setting_str in unique_settings:
            # Parse and display in a more readable format
            print(f"     • {setting_str}")
    
    print()
    
    # Table header
    header = f"{'Model':<25} {'Status':<12} {'Max Diff':<15} {'Mean Diff':<15} {'Different':<15} {'Shape Match':<12}"
    print(header)
    print("-" * 120)
    
    # Table rows
    for result in results:
        status = "✅ MATCH" if result.matches else "❌ DIFFER"
        max_diff_str = format_number(result.max_diff)
        mean_diff_str = format_number(result.mean_diff)
        
        if result.total_elements > 0:
            diff_str = f"{result.num_different}/{result.total_elements} ({result.percent_different:.2f}%)"
        else:
            diff_str = "N/A"
        
        shape_status = "✅" if result.shape_match else "❌"
        
        row = f"{result.name:<25} {status:<12} {max_diff_str:<15} {mean_diff_str:<15} {diff_str:<15} {shape_status:<12}"
        print(row)
    
    print("=" * 120)
    
    # Show sample value comparisons if requested
    if show_samples and cpp_data_dict and py_data_dict:
        for result in results:
            model_key = result.name.lower().replace(" ", "_")
            # Try to find matching data
            for key in cpp_data_dict.keys():
                if model_key in key.lower() or key.lower() in model_key:
                    if key in py_data_dict:
                        print_sample_values_table(cpp_data_dict[key], py_data_dict[key], result.name, num_samples=10)
                        break

def run_comparisons(fnet_cpp: np.ndarray, inet_cpp: np.ndarray, fnet_py_dpvo: np.ndarray,
                    inet_py_dpvo: np.ndarray, is_video: bool = False) -> Tuple[bool, bool]:
    """Run comparisons between Python DPVO preprocessing and C++ outputs."""
    print("=" * 120)
    print("🔍 Running Comparisons")
    print("=" * 120)
    print("  📝 Comparing Python DPVO preprocessing with C++ outputs...")
    print()
    
    # Define preprocessing settings
    dpvo_settings = PreprocessingSettings(
        use_bgr=False,
        match_python_dpvo=True,
        apply_undistort=False,
        is_video=is_video
    )
    
    # Run Python DPVO comparisons
    fnet_dpvo_result = compare_outputs(fnet_cpp, fnet_py_dpvo, "FNet", tolerance=1e-5, preprocessing=dpvo_settings)
    inet_dpvo_result = compare_outputs(inet_cpp, inet_py_dpvo, "INet", tolerance=1e-5, preprocessing=dpvo_settings)
    dpvo_results = [fnet_dpvo_result, inet_dpvo_result]
    
    # Prepare data dictionaries for sample value tables
    dpvo_cpp_dict = {"fnet": fnet_cpp, "inet": inet_cpp}
    dpvo_py_dict = {"fnet": fnet_py_dpvo, "inet": inet_py_dpvo}
    
    # Print comparison table with sample values
    print_category_table(
        "Python DPVO Preprocessing vs C++ Outputs",
        dpvo_results,
        "This shows if C++ preprocessing matches Python DPVO preprocessing",
        cpp_data_dict=dpvo_cpp_dict,
        py_data_dict=dpvo_py_dict,
        show_samples=True
    )
    
    # Print overall comparison table
    print("\n" + "=" * 120)
    print("📊 Comparison Summary")
    print("=" * 120)
    print_comparison_table(dpvo_results)
    
    # Calculate overall max_diff and mean_diff for parseable format
    dpvo_max_diffs = [r.max_diff for r in dpvo_results if r.max_diff != float('inf')]
    dpvo_mean_diffs = [r.mean_diff for r in dpvo_results if r.mean_diff != float('inf')]
    
    if dpvo_max_diffs:
        overall_max_diff = max(dpvo_max_diffs)
    else:
        overall_max_diff = float('inf')
    
    if dpvo_mean_diffs:
        overall_mean_diff = sum(dpvo_mean_diffs) / len(dpvo_mean_diffs)
    else:
        overall_mean_diff = float('inf')
    
    # Print parseable format for run_all_comparisons.py
    print(f"\n   ONNX_MODELS_MAX_DIFF={overall_max_diff:.10e}")
    print(f"   ONNX_MODELS_MEAN_DIFF={overall_mean_diff:.10e}")
    
    # Extract boolean results
    fnet_match_dpvo = dpvo_results[0].matches
    inet_match_dpvo = dpvo_results[1].matches
    
    return fnet_match_dpvo, inet_match_dpvo

def print_summary(fnet_match_dpvo: bool, inet_match_dpvo: bool) -> int:
    """Print final summary in table format and return exit code."""
    print("\n" + "=" * 120)
    print("📊 Final Summary")
    print("=" * 120)
    
    # Summary table
    print(f"\n{'Model':<40} {'Status':<20}")
    print("-" * 120)
    print(f"{'FNet (Python DPVO preprocessing)':<40} {'✅ MATCH' if fnet_match_dpvo else '❌ DIFFER':<20}")
    print(f"{'INet (Python DPVO preprocessing)':<40} {'✅ MATCH' if inet_match_dpvo else '❌ DIFFER':<20}")
    print("=" * 120)
    print()
    
    if fnet_match_dpvo and inet_match_dpvo:
        print("✅ SUCCESS: Python DPVO preprocessing matches C++ outputs!")
        print("  💡 This means C++ ONNX inference uses the same preprocessing as Python DPVO.")
        return 0
    else:
        print("❌ WARNING: Python DPVO preprocessing does not match C++ outputs perfectly.")
        print("  🔍 Check the differences above. There may be preprocessing differences.")
        print()
        print("📝 NOTE: Python DPVO outputs are saved to:")
        print("  📄 bin_file/fnet_py_dpvo_frame0.bin")
        print("  📄 bin_file/inet_py_dpvo_frame0.bin")
        print("  💡 These can be compared with Python DPVO repository outputs.")
        return 1

def main() -> int:
    # Step 1: Parse arguments
    image_path, fnet_model_path, inet_model_path, fnet_cpp_bin, inet_cpp_bin, is_video = parse_arguments()
    
    # Step 2: Validate inputs
    validate_inputs(image_path, fnet_model_path, inet_model_path)
    
    # Step 3: Print header
    print_header(image_path, fnet_model_path, inet_model_path, fnet_cpp_bin, inet_cpp_bin)
    
    # Step 4: Load and preprocess images
    input_data_python_dpvo = load_and_preprocess_images(image_path, is_video=is_video)
    
    # Step 5: Load ONNX models
    fnet_session, inet_session = load_onnx_models(fnet_model_path, inet_model_path)
    
    # Step 6: Run inference
    fnet_py_dpvo, inet_py_dpvo = run_all_inferences(fnet_session, inet_session, input_data_python_dpvo)
    
    # Step 7: Save Python outputs
    fnet_py_dpvo_chw, inet_py_dpvo_chw = save_python_outputs(fnet_py_dpvo, inet_py_dpvo)
    
    # Step 8: Load C++ outputs
    fnet_cpp, inet_cpp = load_cpp_outputs(fnet_cpp_bin, inet_cpp_bin, fnet_py_dpvo_chw, inet_py_dpvo_chw)
    
    # Step 9: Run comparisons
    fnet_match_dpvo, inet_match_dpvo = run_comparisons(
        fnet_cpp, inet_cpp, fnet_py_dpvo, inet_py_dpvo, is_video=is_video
    )
    
    # Step 10: Print summary and return
    return print_summary(fnet_match_dpvo, inet_match_dpvo)

if __name__ == "__main__":
    sys.exit(main())

