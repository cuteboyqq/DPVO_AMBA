#!/usr/bin/env python3
"""
Compare AMBA model FNet/INet outputs with Python ONNX FNet/INet outputs.

This script:
1. Loads AMBA model outputs from amba_fnet_frame<N>.bin and amba_inet_frame<N>.bin
2. Loads the saved input image from amba_input_image_frame<N>.bin
3. Reads metadata from amba_model_metadata_frame<N>.txt
4. Runs Python ONNX FNet/INet inference on the same input image
5. Compares AMBA and Python outputs element-by-element

Usage:
    python compare_amba_outputs.py <fnet_onnx_model> <inet_onnx_model> [--frame N] [--bin-dir DIR] [--image PATH] [--video] [--tolerance TOL]

Examples:
    # Use saved input image from bin_file/ (default frame=1050)
    python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx

    # Specify frame number
    python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx --frame 69

    # Use an external image instead of saved bin
    python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx --image /path/to/RawFrame_001050.jpg

    # Specify custom bin directory
    python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx --bin-dir /path/to/bin_file
"""

import numpy as np
import onnxruntime as ort
import cv2
import sys
import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


# =============================================================================
# Metadata Parsing
# =============================================================================

def load_metadata(metadata_path: str) -> Dict[str, int]:
    """Load AMBA metadata from text file."""
    metadata = {}
    if not os.path.exists(metadata_path):
        print(f"  ⚠️  Metadata file not found: {metadata_path}")
        return metadata
    
    with open(metadata_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                try:
                    metadata[key.strip()] = int(value.strip())
                except ValueError:
                    metadata[key.strip()] = value.strip()
    
    return metadata


# =============================================================================
# Image Loading & Preprocessing
# =============================================================================

def load_image_from_bin(bin_path: str, H: int, W: int, C: int = 3) -> np.ndarray:
    """Load raw image from binary file (uint8 HWC format, as saved from ea_tensor_data)."""
    data = np.fromfile(bin_path, dtype=np.uint8)
    expected_size = H * W * C
    if data.size != expected_size:
        raise ValueError(f"Image bin size mismatch: got {data.size}, expected {expected_size} ({H}x{W}x{C})")
    # ea_tensor_data stores as HWC (height, width, channels)
    img = data.reshape(H, W, C)
    return img


def load_image_from_file(image_path: str) -> np.ndarray:
    """Load image from an image file (BGR via OpenCV)."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image from {image_path}")
    return img


def preprocess_for_onnx(img: np.ndarray, model_H: int, model_W: int,
                         apply_undistort: bool = True,
                         is_video: bool = False,
                         use_rgb: bool = False) -> np.ndarray:
    """Preprocess image for ONNX FNet/INet inference.
    
    Matches Python DPVO preprocessing:
        1. Undistort (optional)
        2. Resize to model input size
        3. Optional BGR→RGB conversion
        4. Normalize: 2 * (image / 255.0) - 0.5
        5. HWC -> CHW
        6. Add batch dimension -> NCHW
    """
    H, W = img.shape[:2]
    
    # Step 1: Undistort (matching Python DPVO)
    if apply_undistort:
        fx, fy, cx, cy = 1660.0, 1660.0, 960.0, 540.0
        k1, k2, p1, p2 = 0.07, -0.08, 0.0, 0.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)
        img = cv2.undistort(img, K, dist_coeffs)
    
    # Step 2: Resize to model input size
    if is_video:
        img_resized = cv2.resize(img, (model_W, model_H), interpolation=cv2.INTER_AREA)
    else:
        img_resized = cv2.resize(img, (model_W, model_H), interpolation=cv2.INTER_LINEAR)
    
    # Step 3: Optional BGR→RGB conversion
    if use_rgb:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Step 4: Normalize: 2 * (image / 255.0) - 0.5
    img_normalized = 2.0 * (img_resized.astype(np.float32) / 255.0) - 0.5
    
    # Step 5: HWC -> CHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    
    # Step 6: Add batch dimension -> NCHW
    img_nchw = np.expand_dims(img_chw, axis=0)
    
    return img_nchw


# =============================================================================
# ONNX Inference
# =============================================================================

def run_onnx_inference(session: ort.InferenceSession, input_data: np.ndarray, model_name: str) -> np.ndarray:
    """Run ONNX inference and return output tensor."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_data})
    result = outputs[0]  # Shape: [1, C, H, W]
    print(f"  ✅ {model_name} inference done. Output shape: {result.shape}")
    return result


# =============================================================================
# Comparison Utilities
# =============================================================================

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
    amba_shape: tuple
    py_shape: tuple
    tolerance: float
    amba_stats: dict  # min, max, mean, std, zeros
    py_stats: dict    # min, max, mean, std, zeros
    sample_diffs: List[Tuple]  # [(idx, amba_val, py_val, diff), ...]


def compute_stats(data: np.ndarray) -> dict:
    """Compute basic statistics of a numpy array."""
    return {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'zeros': int(np.sum(data == 0)),
        'nans': int(np.sum(np.isnan(data))),
        'infs': int(np.sum(np.isinf(data))),
        'total': int(data.size),
    }


def compare_outputs(amba_data: np.ndarray, py_data: np.ndarray, name: str,
                     tolerance: float = 1e-2) -> ComparisonResult:
    """Compare AMBA and Python outputs element-by-element."""
    # Remove batch dimension from Python output if present
    if py_data.ndim == 4:
        py_data = py_data[0]
    
    shape_match = (amba_data.shape == py_data.shape)
    
    amba_stats = compute_stats(amba_data)
    py_stats = compute_stats(py_data)
    
    if not shape_match:
        return ComparisonResult(
            name=name, matches=False,
            max_diff=float('inf'), mean_diff=float('inf'),
            num_different=0, total_elements=0, percent_different=100.0,
            shape_match=False, amba_shape=amba_data.shape, py_shape=py_data.shape,
            tolerance=tolerance, amba_stats=amba_stats, py_stats=py_stats,
            sample_diffs=[]
        )
    
    diff = np.abs(amba_data - py_data)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    
    different_mask = diff > tolerance
    num_different = int(np.sum(different_mask))
    total_elements = int(amba_data.size)
    percent_different = (num_different / total_elements * 100) if total_elements > 0 else 0
    matches = (num_different == 0)
    
    # Collect sample diffs (top 10 largest)
    flat_diff = diff.flatten()
    top_indices = np.argsort(flat_diff)[-10:][::-1]
    sample_diffs = []
    for flat_idx in top_indices:
        multi_idx = np.unravel_index(flat_idx, amba_data.shape)
        sample_diffs.append((
            multi_idx,
            float(amba_data[multi_idx]),
            float(py_data[multi_idx]),
            float(diff[multi_idx])
        ))
    
    return ComparisonResult(
        name=name, matches=matches,
        max_diff=max_diff, mean_diff=mean_diff,
        num_different=num_different, total_elements=total_elements,
        percent_different=percent_different,
        shape_match=shape_match, amba_shape=amba_data.shape, py_shape=py_data.shape,
        tolerance=tolerance, amba_stats=amba_stats, py_stats=py_stats,
        sample_diffs=sample_diffs
    )


def fmt(val: float) -> str:
    """Format a number for display."""
    if val == float('inf') or val == float('-inf'):
        return "inf"
    if np.isnan(val):
        return "NaN"
    if abs(val) >= 1e6 or (abs(val) < 1e-4 and val != 0):
        return f"{val:.4e}"
    return f"{val:.6f}"


# =============================================================================
# Result Printing
# =============================================================================

def print_stats_table(name: str, stats: dict):
    """Print tensor statistics in a compact table."""
    print(f"    {name}: min={fmt(stats['min'])}, max={fmt(stats['max'])}, "
          f"mean={fmt(stats['mean'])}, std={fmt(stats['std'])}, "
          f"zeros={stats['zeros']}/{stats['total']}, "
          f"nans={stats['nans']}, infs={stats['infs']}")


def print_result(result: ComparisonResult):
    """Print a single comparison result."""
    status = "✅ MATCH" if result.matches else "❌ DIFFER"
    print(f"\n{'='*100}")
    print(f"  {result.name}: {status}")
    print(f"{'='*100}")
    
    if not result.shape_match:
        print(f"  ❌ Shape mismatch: AMBA={result.amba_shape}, Python={result.py_shape}")
        print_stats_table("AMBA", result.amba_stats)
        print_stats_table("Python", result.py_stats)
        return
    
    print(f"  Shape: {result.amba_shape}")
    print(f"  Tolerance: {result.tolerance}")
    print(f"  Max diff:  {fmt(result.max_diff)}")
    print(f"  Mean diff: {fmt(result.mean_diff)}")
    print(f"  Different: {result.num_different}/{result.total_elements} ({result.percent_different:.2f}%)")
    print()
    print_stats_table("AMBA", result.amba_stats)
    print_stats_table("Python", result.py_stats)
    
    if result.sample_diffs:
        print(f"\n  Top {len(result.sample_diffs)} largest differences:")
        print(f"    {'Index (C,H,W)':<25} {'AMBA Value':<18} {'Python Value':<18} {'Difference':<18}")
        print(f"    {'-'*25} {'-'*18} {'-'*18} {'-'*18}")
        for idx, amba_val, py_val, diff_val in result.sample_diffs:
            print(f"    {str(idx):<25} {fmt(amba_val):<18} {fmt(py_val):<18} {fmt(diff_val):<18}")


def print_summary(results: List[ComparisonResult]):
    """Print overall summary table."""
    print(f"\n{'='*100}")
    print("📊 AMBA vs Python ONNX — Summary")
    print(f"{'='*100}")
    print(f"  {'Model':<15} {'Status':<12} {'Max Diff':<15} {'Mean Diff':<15} {'Different':<25} {'Shape':<10}")
    print(f"  {'-'*15} {'-'*12} {'-'*15} {'-'*15} {'-'*25} {'-'*10}")
    for r in results:
        status = "✅ MATCH" if r.matches else "❌ DIFFER"
        shape_ok = "✅" if r.shape_match else "❌"
        diff_str = f"{r.num_different}/{r.total_elements} ({r.percent_different:.2f}%)" if r.total_elements > 0 else "N/A"
        print(f"  {r.name:<15} {status:<12} {fmt(r.max_diff):<15} {fmt(r.mean_diff):<15} {diff_str:<25} {shape_ok:<10}")
    print(f"{'='*100}")
    
    all_match = all(r.matches for r in results)
    if all_match:
        print("\n✅ SUCCESS: AMBA model outputs match Python ONNX outputs within tolerance!")
    else:
        print("\n❌ MISMATCH: AMBA model outputs differ from Python ONNX outputs.")
        print("   This indicates the AMBA model conversion may have introduced differences.")
        print("   Check the detailed comparison above for specific channels/locations.")


# =============================================================================
# Channel-by-channel analysis
# =============================================================================

def compute_mean_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Compute mean correlation across all channels."""
    if a.shape != b.shape:
        return 0.0
    C = a.shape[0]
    corrs = []
    for c in range(C):
        af = a[c].flatten()
        bf = b[c].flatten()
        if np.std(af) > 0 and np.std(bf) > 0:
            corrs.append(np.corrcoef(af, bf)[0, 1])
    return float(np.mean(corrs)) if corrs else 0.0


def run_diagnostic(fnet_bin_path: str, inet_bin_path: str,
                   fnet_session, inet_session,
                   img: np.ndarray, model_H: int, model_W: int,
                   fnet_C: int, fnet_H: int, fnet_W: int,
                   inet_C: int, inet_H: int, inet_W: int,
                   apply_undistort: bool, is_video: bool):
    """Test all NCHW/NHWC × BGR/RGB combinations to find best match."""
    print("\n" + "=" * 100)
    print("🔬 DIAGNOSTIC: Testing all layout × color combinations")
    print("=" * 100)
    print("  This tests 4 combinations to determine:")
    print("  1. Output tensor memory layout: NCHW (standard) vs NHWC (channels-last)")
    print("  2. Model input color ordering: BGR vs RGB")
    print()

    # Load raw AMBA output data
    raw_fnet = np.fromfile(fnet_bin_path, dtype=np.float32)
    raw_inet = np.fromfile(inet_bin_path, dtype=np.float32)

    # Two layout interpretations for FNet [128, 132, 240]
    fnet_nchw = raw_fnet.reshape(fnet_C, fnet_H, fnet_W)
    try:
        fnet_nhwc = raw_fnet.reshape(fnet_H, fnet_W, fnet_C).transpose(2, 0, 1)  # → (C, H, W)
    except ValueError:
        fnet_nhwc = None
        print("  ⚠️  Cannot reshape FNet as NHWC — skipping NHWC test for FNet")

    # Two layout interpretations for INet [384, 132, 240]
    inet_nchw = raw_inet.reshape(inet_C, inet_H, inet_W)
    try:
        inet_nhwc = raw_inet.reshape(inet_H, inet_W, inet_C).transpose(2, 0, 1)  # → (C, H, W)
    except ValueError:
        inet_nhwc = None
        print("  ⚠️  Cannot reshape INet as NHWC — skipping NHWC test for INet")

    # Run ONNX inference with BGR and RGB inputs
    input_bgr = preprocess_for_onnx(img, model_H, model_W,
                                     apply_undistort=apply_undistort,
                                     is_video=is_video, use_rgb=False)
    input_rgb = preprocess_for_onnx(img, model_H, model_W,
                                     apply_undistort=apply_undistort,
                                     is_video=is_video, use_rgb=True)

    py_fnet_bgr = run_onnx_inference(fnet_session, input_bgr, "FNet-BGR")[0]
    py_fnet_rgb = run_onnx_inference(fnet_session, input_rgb, "FNet-RGB")[0]
    py_inet_bgr = run_onnx_inference(inet_session, input_bgr, "INet-BGR")[0]
    py_inet_rgb = run_onnx_inference(inet_session, input_rgb, "INet-RGB")[0]

    # Test all combinations
    print("\n  " + "-" * 90)
    print(f"  {'Model':<8} {'AMBA Layout':<14} {'Python Color':<14} {'Mean Corr':>12} {'Mean Diff':>12} {'Max Diff':>12}")
    print("  " + "-" * 90)

    best_fnet = (None, None, -1.0)
    best_inet = (None, None, -1.0)

    fnet_layouts = [('NCHW', fnet_nchw)]
    if fnet_nhwc is not None:
        fnet_layouts.append(('NHWC→NCHW', fnet_nhwc))

    inet_layouts = [('NCHW', inet_nchw)]
    if inet_nhwc is not None:
        inet_layouts.append(('NHWC→NCHW', inet_nhwc))

    for layout_name, amba_data in fnet_layouts:
        for color_name, py_data in [('BGR', py_fnet_bgr), ('RGB', py_fnet_rgb)]:
            corr = compute_mean_correlation(amba_data, py_data)
            diff = np.abs(amba_data - py_data)
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))
            marker = ""
            if corr > best_fnet[2]:
                best_fnet = (layout_name, color_name, corr)
                marker = " ◀ best"
            print(f"  {'FNet':<8} {layout_name:<14} {color_name:<14} {corr:>12.6f} {mean_diff:>12.6f} {max_diff:>12.6f}{marker}")

    print("  " + "-" * 90)

    for layout_name, amba_data in inet_layouts:
        for color_name, py_data in [('BGR', py_inet_bgr), ('RGB', py_inet_rgb)]:
            corr = compute_mean_correlation(amba_data, py_data)
            diff = np.abs(amba_data - py_data)
            mean_diff = float(np.mean(diff))
            max_diff = float(np.max(diff))
            marker = ""
            if corr > best_inet[2]:
                best_inet = (layout_name, color_name, corr)
                marker = " ◀ best"
            print(f"  {'INet':<8} {layout_name:<14} {color_name:<14} {corr:>12.6f} {mean_diff:>12.6f} {max_diff:>12.6f}{marker}")

    print("  " + "-" * 90)

    print(f"\n  🏆 Best FNet match: layout={best_fnet[0]}, color={best_fnet[1]}, correlation={best_fnet[2]:.6f}")
    print(f"  🏆 Best INet match: layout={best_inet[0]}, color={best_inet[1]}, correlation={best_inet[2]:.6f}")

    if best_fnet[2] > 0.95 or best_inet[2] > 0.95:
        print(f"\n  ✅ High correlation found! The correct configuration is:")
        if best_fnet[2] > 0.95:
            print(f"     FNet: AMBA output is {best_fnet[0]} layout, model expects {best_fnet[1]} input")
        if best_inet[2] > 0.95:
            print(f"     INet: AMBA output is {best_inet[0]} layout, model expects {best_inet[1]} input")
        if best_fnet[0] == 'NHWC→NCHW' or best_inet[0] == 'NHWC→NCHW':
            print(f"\n  ⚠️  ACTION NEEDED: The C++ code reads AMBA output as NCHW but it's actually NHWC!")
            print(f"     Fix fnet.cpp/inet.cpp to transpose the output tensor from NHWC to NCHW.")
        if best_fnet[1] == 'RGB' or best_inet[1] == 'RGB':
            print(f"\n  ⚠️  ACTION NEEDED: The model expects RGB input!")
            print(f"     The C++ code should use ea_cvt_color_resize(BGR2RGB), not ea_crop_resize(BGR).")
            print(f"     The Python comparison should also convert BGR→RGB.")
    else:
        print(f"\n  ⚠️  No combination achieved >0.95 correlation.")
        print(f"     The issue may be in preprocessing (resize algorithm), quantization quality,")
        print(f"     or output tensor pitch/padding. Consider dumping the AMBA preprocessed input")
        print(f"     tensor for comparison.")

    return best_fnet, best_inet


def load_amba_preprocessed_input(bin_path: str, meta_path: str) -> Optional[np.ndarray]:
    """Load AMBA preprocessed model input tensor from bin + metadata files.
    
    Returns a uint8 numpy array in CHW format, or None if files don't exist.
    """
    if not os.path.exists(bin_path) or not os.path.exists(meta_path):
        return None
    
    # Load metadata
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, val = line.split('=', 1)
                meta[key.strip()] = int(val.strip())
    
    C = meta.get('C', 3)
    H = meta.get('H', 0)
    W = meta.get('W', 0)
    pitch_bytes = meta.get('pitch_bytes', W)
    total_bytes = meta.get('total_bytes', C * H * W)
    
    if H == 0 or W == 0:
        return None
    
    # Load raw binary data
    raw = np.fromfile(bin_path, dtype=np.uint8)
    
    # Determine element size: if total_bytes == C * H * W → uint8, if 4x → float32
    expected_uint8 = C * H * pitch_bytes  # pitch_bytes includes padding
    expected_float32 = C * H * pitch_bytes  # pitch is in bytes for any type
    
    # Check if it's uint8 or float32 by comparing total_bytes with C*H*W
    bytes_per_pixel = total_bytes / (C * H * W) if (C * H * W > 0) else 1
    
    if abs(bytes_per_pixel - 1.0) < 0.1:
        # uint8 format
        dtype = np.uint8
        elem_size = 1
    elif abs(bytes_per_pixel - 4.0) < 0.1:
        # float32 format
        dtype = np.float32
        elem_size = 4
        raw = np.frombuffer(raw.tobytes(), dtype=np.float32)
    else:
        print(f"  ⚠️  Unknown element size: {bytes_per_pixel:.2f} bytes/pixel")
        return None
    
    pitch_elements = pitch_bytes // elem_size
    
    # Reconstruct dense CHW tensor (strip pitch padding if needed)
    if pitch_elements == W:
        # No padding, simple reshape
        tensor = raw[:C * H * W].reshape(C, H, W)
    else:
        # Pitch padding present — read row by row
        tensor = np.zeros((C, H, W), dtype=dtype)
        for c in range(C):
            for y in range(H):
                row_offset = (c * H + y) * pitch_elements
                tensor[c, y, :] = raw[row_offset:row_offset + W]
    
    return tensor


def run_input_diagnostic(bin_dir: str, frame: int,
                         fnet_session, inet_session,
                         img: np.ndarray, model_H: int, model_W: int,
                         amba_fnet: np.ndarray, amba_inet: np.ndarray,
                         apply_undistort: bool, is_video: bool):
    """Compare AMBA preprocessed input with Python preprocessing,
    then feed AMBA input directly to ONNX to isolate model conversion quality.
    """
    print("\n" + "=" * 100)
    print("🔬 INPUT TENSOR DIAGNOSTIC")
    print("=" * 100)
    print("  This diagnostic:")
    print("  1. Compares AMBA preprocessed input pixels with Python-preprocessed pixels")
    print("  2. Feeds the AMBA input directly to ONNX to isolate model quality from preprocessing")
    print()
    
    # Load AMBA preprocessed inputs
    fnet_input_bin = os.path.join(bin_dir, f"amba_fnet_preprocessed_frame{frame}.bin")
    fnet_input_meta = os.path.join(bin_dir, f"amba_fnet_preprocessed_meta_frame{frame}.txt")
    
    amba_input = load_amba_preprocessed_input(fnet_input_bin, fnet_input_meta)
    
    if amba_input is None:
        print(f"  ❌ AMBA preprocessed input not found!")
        print(f"     Expected files:")
        print(f"       {fnet_input_bin}")
        print(f"       {fnet_input_meta}")
        print(f"     To generate these, rebuild and re-run the C++ app with TARGET_FRAME={frame}.")
        print(f"     The code in fnet.cpp now saves the preprocessed input tensor after ea_crop_resize.")
        return
    
    print(f"  ✅ Loaded AMBA preprocessed input: shape={amba_input.shape}, dtype={amba_input.dtype}")
    print(f"     Range: [{amba_input.min()}, {amba_input.max()}], mean={amba_input.mean():.2f}")
    
    # ── Part 1: Compare input pixels ──
    print("\n  ─── Part 1: Compare preprocessing (AMBA vs Python pixel values) ───")
    
    # Python preprocessing: resize, NO color conversion, NO normalization (keep uint8 for comparison)
    H, W = img.shape[:2]
    if apply_undistort:
        fx, fy, cx, cy = 1660.0, 1660.0, 960.0, 540.0
        k1, k2, p1, p2 = 0.07, -0.08, 0.0, 0.0
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)
        img = cv2.undistort(img, K, dist_coeffs)
    
    if is_video:
        img_resized = cv2.resize(img, (model_W, model_H), interpolation=cv2.INTER_AREA)
    else:
        img_resized = cv2.resize(img, (model_W, model_H), interpolation=cv2.INTER_LINEAR)
    
    # Convert HWC → CHW for comparison
    py_input_bgr = np.transpose(img_resized, (2, 0, 1)).astype(np.uint8)  # CHW, BGR
    py_input_rgb = np.transpose(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.uint8)
    
    if amba_input.dtype == np.uint8:
        # Compare AMBA input with Python BGR and RGB
        for name, py_input in [("BGR", py_input_bgr), ("RGB", py_input_rgb)]:
            if amba_input.shape != py_input.shape:
                print(f"    {name}: Shape mismatch AMBA={amba_input.shape} vs Python={py_input.shape}")
                continue
            diff = np.abs(amba_input.astype(np.int16) - py_input.astype(np.int16))
            pix_corr = compute_mean_correlation(amba_input.astype(np.float32), py_input.astype(np.float32))
            print(f"    vs Python {name}: max_diff={diff.max()}, mean_diff={diff.mean():.4f}, "
                  f"exact_match={np.sum(diff == 0)}/{amba_input.size} "
                  f"({100.0 * np.sum(diff == 0) / amba_input.size:.1f}%), "
                  f"corr={pix_corr:.6f}")
    elif amba_input.dtype == np.float32:
        # AMBA input is already float — check if it's normalized
        print(f"    AMBA input is float32, range [{amba_input.min():.4f}, {amba_input.max():.4f}]")
        if amba_input.max() <= 2.0 and amba_input.min() >= -1.0:
            print(f"    → Appears to be normalized (range ~[-0.5, 1.5])")
        elif amba_input.max() <= 255.0 and amba_input.min() >= 0.0:
            print(f"    → Appears to be raw float pixels (range [0, 255])")
    
    # ── Part 2: Feed AMBA input to ONNX ──
    print("\n  ─── Part 2: Feed AMBA input directly to ONNX (isolate model quality) ───")
    print("  This tests: if we give ONNX the EXACT same pixels as AMBA, how close are outputs?")
    
    # Normalize AMBA input for ONNX: (pixel - 63.75) / 127.5 = 2*(pixel/255) - 0.5
    if amba_input.dtype == np.uint8:
        amba_normalized = 2.0 * (amba_input.astype(np.float32) / 255.0) - 0.5
    else:
        amba_normalized = amba_input  # Already float, might already be normalized
    
    amba_nchw = np.expand_dims(amba_normalized, axis=0)  # Add batch dim → [1, C, H, W]
    
    # Run ONNX with AMBA's exact preprocessed input (BGR order)
    print(f"\n  Feeding AMBA input (BGR, normalized) to ONNX...")
    py_fnet_from_amba_bgr = run_onnx_inference(fnet_session, amba_nchw, "FNet-from-AMBA-BGR")[0]
    py_inet_from_amba_bgr = run_onnx_inference(inet_session, amba_nchw, "INet-from-AMBA-BGR")[0]
    
    # Also try swapping B↔R channels (in case model expects RGB)
    amba_rgb = amba_normalized.copy()
    amba_rgb[0], amba_rgb[2] = amba_normalized[2].copy(), amba_normalized[0].copy()
    amba_rgb_nchw = np.expand_dims(amba_rgb, axis=0)
    
    print(f"  Feeding AMBA input (RGB-swapped, normalized) to ONNX...")
    py_fnet_from_amba_rgb = run_onnx_inference(fnet_session, amba_rgb_nchw, "FNet-from-AMBA-RGB")[0]
    py_inet_from_amba_rgb = run_onnx_inference(inet_session, amba_rgb_nchw, "INet-from-AMBA-RGB")[0]
    
    # Compare: AMBA output vs ONNX-from-AMBA-input
    print(f"\n  ─── Results: AMBA output vs ONNX(AMBA-input) ───")
    print(f"  {'Test':<35} {'Model':<8} {'Mean Corr':>12} {'Mean Diff':>12} {'Max Diff':>12}")
    print(f"  {'-'*35} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    
    for test_name, amba_out, onnx_out, model_name in [
        ("AMBA-input BGR → ONNX", amba_fnet, py_fnet_from_amba_bgr, "FNet"),
        ("AMBA-input RGB → ONNX", amba_fnet, py_fnet_from_amba_rgb, "FNet"),
        ("AMBA-input BGR → ONNX", amba_inet, py_inet_from_amba_bgr, "INet"),
        ("AMBA-input RGB → ONNX", amba_inet, py_inet_from_amba_rgb, "INet"),
    ]:
        corr = compute_mean_correlation(amba_out, onnx_out)
        diff = np.abs(amba_out - onnx_out)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))
        marker = " ◀ GOOD" if corr > 0.95 else (" ◀ ok" if corr > 0.85 else "")
        print(f"  {test_name:<35} {model_name:<8} {corr:>12.6f} {mean_diff:>12.6f} {max_diff:>12.6f}{marker}")
    
    # Also compare: Python-preprocessed-input → ONNX vs AMBA output (for reference)
    print(f"\n  ─── Reference: Python-preprocessed-input → ONNX vs AMBA output ───")
    print(f"  (This repeats the standard comparison for context)")
    
    for use_rgb, color_name in [(False, "BGR"), (True, "RGB")]:
        py_prep = preprocess_for_onnx(img, model_H, model_W,
                                       apply_undistort=apply_undistort,
                                       is_video=is_video, use_rgb=use_rgb)
        py_fnet_ref = run_onnx_inference(fnet_session, py_prep, f"FNet-Python-{color_name}")[0]
        py_inet_ref = run_onnx_inference(inet_session, py_prep, f"INet-Python-{color_name}")[0]
        
        for model_name, amba_out, onnx_out in [("FNet", amba_fnet, py_fnet_ref), ("INet", amba_inet, py_inet_ref)]:
            corr = compute_mean_correlation(amba_out, onnx_out)
            diff = np.abs(amba_out - onnx_out)
            print(f"  {'Python-' + color_name + ' → ONNX':<35} {model_name:<8} {corr:>12.6f} "
                  f"{float(np.mean(diff)):>12.6f} {float(np.max(diff)):>12.6f}")
    
    # Summary interpretation
    print(f"\n  ─── Interpretation ───")
    print(f"  • If 'AMBA-input → ONNX' correlation is HIGH (>0.95):")
    print(f"      The model conversion/quantization is GOOD. Fix the preprocessing pipeline.")
    print(f"  • If 'AMBA-input → ONNX' correlation is LOW (<0.90):")
    print(f"      The model conversion/quantization is the problem.")
    print(f"      → Reconvert the model. Check: DRA color (BGR vs RGB), calibration quality.")
    print(f"  • If BGR gives higher correlation than RGB:")
    print(f"      The AMBA model was calibrated for BGR. Match the ONNX model color accordingly.")
    print(f"  • If RGB gives higher correlation than BGR:")
    print(f"      The model expects RGB. Fix AMBA runtime to convert BGR→RGB.")


def print_channel_analysis(amba_data: np.ndarray, py_data: np.ndarray, name: str,
                           num_channels: int = 10):
    """Print per-channel comparison for the first N channels."""
    if amba_data.shape != py_data.shape:
        return
    
    C = amba_data.shape[0]
    print(f"\n  📋 Per-channel analysis for {name} (first {min(num_channels, C)} of {C} channels):")
    print(f"    {'Ch':<5} {'AMBA mean':<14} {'Py mean':<14} {'Max diff':<14} {'Mean diff':<14} {'Corr':<10}")
    print(f"    {'-'*5} {'-'*14} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    
    for c in range(min(num_channels, C)):
        amba_ch = amba_data[c]
        py_ch = py_data[c]
        ch_diff = np.abs(amba_ch - py_ch)
        
        # Compute correlation
        amba_flat = amba_ch.flatten()
        py_flat = py_ch.flatten()
        if np.std(amba_flat) > 0 and np.std(py_flat) > 0:
            corr = np.corrcoef(amba_flat, py_flat)[0, 1]
        else:
            corr = 0.0
        
        print(f"    {c:<5} {fmt(float(np.mean(amba_ch))):<14} {fmt(float(np.mean(py_ch))):<14} "
              f"{fmt(float(np.max(ch_diff))):<14} {fmt(float(np.mean(ch_diff))):<14} {corr:.6f}")


def print_border_analysis(amba_data: np.ndarray, py_data: np.ndarray, name: str,
                          border_width: int = 3):
    """Analyze error distribution: border pixels vs interior pixels.
    
    Many resize algorithm differences manifest at the edges.
    If interior correlation >> border correlation, the root cause is
    border handling in the resize algorithm (not quantization).
    """
    if amba_data.shape != py_data.shape or amba_data.ndim != 3:
        return
    
    C, H, W = amba_data.shape
    b = border_width
    
    if H <= 2 * b or W <= 2 * b:
        return
    
    # Interior slice: exclude border_width pixels from each edge
    amba_interior = amba_data[:, b:H-b, b:W-b]
    py_interior = py_data[:, b:H-b, b:W-b]
    
    # Border: create mask
    border_mask = np.ones((H, W), dtype=bool)
    border_mask[b:H-b, b:W-b] = False
    
    # Compute per-channel correlations for interior
    interior_corrs = []
    border_corrs = []
    interior_diffs = []
    border_diffs = []
    
    for c in range(C):
        # Interior
        ai = amba_interior[c].flatten()
        pi = py_interior[c].flatten()
        if np.std(ai) > 0 and np.std(pi) > 0:
            interior_corrs.append(np.corrcoef(ai, pi)[0, 1])
        interior_diffs.append(float(np.mean(np.abs(ai - pi))))
        
        # Border
        ab = amba_data[c][border_mask].flatten()
        pb = py_data[c][border_mask].flatten()
        if np.std(ab) > 0 and np.std(pb) > 0:
            border_corrs.append(np.corrcoef(ab, pb)[0, 1])
        border_diffs.append(float(np.mean(np.abs(ab - pb))))
    
    int_corr = float(np.mean(interior_corrs)) if interior_corrs else 0.0
    brd_corr = float(np.mean(border_corrs)) if border_corrs else 0.0
    int_diff = float(np.mean(interior_diffs))
    brd_diff = float(np.mean(border_diffs))
    
    n_interior = (H - 2*b) * (W - 2*b)
    n_border = H * W - n_interior
    
    print(f"\n  🔲 Border vs Interior analysis for {name} (border={b}px):")
    print(f"    {'Region':<20} {'Pixels/ch':<12} {'Mean Corr':>12} {'Mean Diff':>12}")
    print(f"    {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    print(f"    {'Interior':<20} {n_interior:<12} {int_corr:>12.6f} {int_diff:>12.6f}")
    print(f"    {'Border':<20} {n_border:<12} {brd_corr:>12.6f} {brd_diff:>12.6f}")
    print(f"    {'All':<20} {H*W:<12} "
          f"{float(np.mean(interior_corrs + border_corrs)) if (interior_corrs + border_corrs) else 0.0:>12.6f} "
          f"{float(np.mean(np.abs(amba_data - py_data))):>12.6f}")
    
    if int_corr > brd_corr + 0.05:
        print(f"    ⚠️  Interior correlation ({int_corr:.4f}) >> Border ({brd_corr:.4f})")
        print(f"       → Resize border handling is a significant error source.")
    elif int_corr < 0.95:
        print(f"    ℹ️  Both interior ({int_corr:.4f}) and border ({brd_corr:.4f}) have low correlation.")
        print(f"       → Issue is likely INT8 quantization, not just resize borders.")


def run_resize_sensitivity_test(onnx_session, img: np.ndarray,
                                model_H: int, model_W: int, name: str,
                                apply_undistort: bool, is_video: bool):
    """Test how sensitive the ONNX model output is to different resize algorithms.
    
    This reveals if the model is inherently sensitive to resize differences,
    which would explain gaps between AMBA (hardware bilinear) and Python (cv2).
    """
    methods = [
        (cv2.INTER_LINEAR,  "INTER_LINEAR (default)"),
        (cv2.INTER_AREA,    "INTER_AREA"),
        (cv2.INTER_CUBIC,   "INTER_CUBIC"),
        (cv2.INTER_NEAREST, "INTER_NEAREST"),
        (cv2.INTER_LANCZOS4,"INTER_LANCZOS4"),
    ]
    
    # Generate reference with INTER_LINEAR + RGB
    ref_input = preprocess_for_onnx(img, model_H, model_W,
                                     apply_undistort=apply_undistort,
                                     is_video=False, use_rgb=True)
    ref_output = onnx_session.run(None, {onnx_session.get_inputs()[0].name: ref_input})[0][0]
    
    print(f"\n  🔄 Resize sensitivity test for {name}:")
    print(f"    Reference: INTER_LINEAR + RGB")
    print(f"    {'Method':<28} {'Mean Corr':>12} {'Mean Diff':>12} {'Max Diff':>12}")
    print(f"    {'-'*28} {'-'*12} {'-'*12} {'-'*12}")
    
    for method, method_name in methods:
        # Custom preprocessing with specific resize method
        img_work = img.copy()
        if apply_undistort:
            fx, fy, cx, cy = 1660.0, 1660.0, 960.0, 540.0
            k1, k2, p1, p2 = 0.07, -0.08, 0.0, 0.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            dist_coeffs = np.array([k1, k2, p1, p2], dtype=np.float64)
            img_work = cv2.undistort(img_work, K, dist_coeffs)
        
        img_resized = cv2.resize(img_work, (model_W, model_H), interpolation=method)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = 2.0 * (img_resized.astype(np.float32) / 255.0) - 0.5
        img_chw = np.transpose(img_normalized, (2, 0, 1))
        test_input = np.expand_dims(img_chw, axis=0)
        
        test_output = onnx_session.run(None, {onnx_session.get_inputs()[0].name: test_input})[0][0]
        
        corr = compute_mean_correlation(ref_output, test_output)
        diff = np.abs(ref_output - test_output)
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))
        
        marker = " (ref)" if method == cv2.INTER_LINEAR else ""
        print(f"    {method_name:<28} {corr:>12.6f} {mean_diff:>12.6f} {max_diff:>12.6f}{marker}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare AMBA FNet/INet outputs with Python ONNX inference outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: load saved AMBA outputs and input image from bin_file/
  python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx

  # Specify frame number
  python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx --frame 69

  # Use an external image instead of saved bin
  python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx --image /path/to/frame.jpg

  # Use video stream preprocessing
  python compare_amba_outputs.py onnx_models/fnet.onnx onnx_models/inet.onnx --video
        """
    )
    parser.add_argument('fnet_onnx', help='Path to FNet ONNX model')
    parser.add_argument('inet_onnx', help='Path to INet ONNX model')
    parser.add_argument('--frame', type=int, default=1050,
                        help='TARGET_FRAME number (default: 1050, matching target_frame.cpp)')
    parser.add_argument('--bin-dir', default='bin_file',
                        help='Directory containing AMBA output bin files (default: bin_file)')
    parser.add_argument('--image', default=None,
                        help='Path to input image file (overrides saved bin image)')
    parser.add_argument('--video', action='store_true',
                        help='Use video stream preprocessing (INTER_AREA resize)')
    parser.add_argument('--tolerance', type=float, default=1e-2,
                        help='Tolerance for matching (default: 0.01)')
    parser.add_argument('--no-undistort', action='store_true',
                        help='Skip undistortion in Python preprocessing')
    parser.add_argument('--channels', type=int, default=10,
                        help='Number of channels to show in per-channel analysis (default: 10)')
    parser.add_argument('--diagnose', action='store_true',
                        help='Run diagnostic: test all NCHW/NHWC × BGR/RGB combinations')
    parser.add_argument('--input-diag', action='store_true',
                        help='Run input tensor diagnostic: compare AMBA preprocessed input with Python, '
                             'and feed AMBA input directly to ONNX to isolate model quality')
    parser.add_argument('--resize-test', action='store_true',
                        help='Run resize sensitivity test: compare ONNX output with different '
                             'resize interpolation methods to gauge model sensitivity')
    return parser.parse_args()


def main():
    args = parse_args()
    
    frame = args.frame
    bin_dir = args.bin_dir
    
    # Construct file paths
    fnet_bin_path = os.path.join(bin_dir, f"amba_fnet_frame{frame}.bin")
    inet_bin_path = os.path.join(bin_dir, f"amba_inet_frame{frame}.bin")
    image_bin_path = os.path.join(bin_dir, f"amba_input_image_frame{frame}.bin")
    metadata_path = os.path.join(bin_dir, f"amba_model_metadata_frame{frame}.txt")
    
    # ─────────────────────────────────────────────────────────────────────
    # Print header
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 100)
    print("🔍 AMBA vs Python ONNX — FNet/INet Output Comparison")
    print("=" * 100)
    print(f"  📋 Frame:          {frame}")
    print(f"  🧠 FNet ONNX:      {args.fnet_onnx}")
    print(f"  🧠 INet ONNX:      {args.inet_onnx}")
    print(f"  📂 AMBA FNet bin:   {fnet_bin_path}")
    print(f"  📂 AMBA INet bin:   {inet_bin_path}")
    print(f"  📷 Input image:    {args.image or image_bin_path}")
    print(f"  📄 Metadata:       {metadata_path}")
    print(f"  🎯 Tolerance:      {args.tolerance}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 1: Validate files
    # ─────────────────────────────────────────────────────────────────────
    if not os.path.exists(fnet_bin_path):
        print(f"❌ ERROR: AMBA FNet output not found: {fnet_bin_path}")
        print(f"   Run the C++ app with TARGET_FRAME={frame} to generate this file.")
        sys.exit(1)
    if not os.path.exists(inet_bin_path):
        print(f"❌ ERROR: AMBA INet output not found: {inet_bin_path}")
        sys.exit(1)
    if not os.path.exists(args.fnet_onnx):
        print(f"❌ ERROR: FNet ONNX model not found: {args.fnet_onnx}")
        sys.exit(1)
    if not os.path.exists(args.inet_onnx):
        print(f"❌ ERROR: INet ONNX model not found: {args.inet_onnx}")
        sys.exit(1)
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 2: Load metadata
    # ─────────────────────────────────────────────────────────────────────
    print("📄 Loading metadata...")
    metadata = load_metadata(metadata_path)
    if metadata:
        print(f"  Metadata: {metadata}")
    
    # Get dimensions from metadata or use defaults
    input_H = metadata.get('input_image_H', 1080)
    input_W = metadata.get('input_image_W', 1920)
    model_H = metadata.get('model_input_H', 528)
    model_W = metadata.get('model_input_W', 960)
    fnet_C = metadata.get('fnet_output_C', 128)
    fnet_H = metadata.get('fnet_output_H', 132)
    fnet_W = metadata.get('fnet_output_W', 240)
    inet_C = metadata.get('inet_output_C', 384)
    inet_H = metadata.get('inet_output_H', 132)
    inet_W = metadata.get('inet_output_W', 240)
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 3: Load AMBA outputs
    # ─────────────────────────────────────────────────────────────────────
    print("📂 Loading AMBA model outputs...")
    amba_fnet = np.fromfile(fnet_bin_path, dtype=np.float32)
    expected_fnet_size = fnet_C * fnet_H * fnet_W
    if amba_fnet.size != expected_fnet_size:
        print(f"  ⚠️  FNet size mismatch: got {amba_fnet.size}, expected {expected_fnet_size}")
        print(f"      Attempting to infer shape from file size...")
        # Try to infer shape
        if amba_fnet.size % (fnet_H * fnet_W) == 0:
            fnet_C = amba_fnet.size // (fnet_H * fnet_W)
            print(f"      Inferred C={fnet_C}")
    amba_fnet = amba_fnet.reshape(fnet_C, fnet_H, fnet_W)
    print(f"  ✅ AMBA FNet: shape={amba_fnet.shape}, "
          f"min={amba_fnet.min():.6f}, max={amba_fnet.max():.6f}, mean={amba_fnet.mean():.6f}")
    
    amba_inet = np.fromfile(inet_bin_path, dtype=np.float32)
    expected_inet_size = inet_C * inet_H * inet_W
    if amba_inet.size != expected_inet_size:
        print(f"  ⚠️  INet size mismatch: got {amba_inet.size}, expected {expected_inet_size}")
        if amba_inet.size % (inet_H * inet_W) == 0:
            inet_C = amba_inet.size // (inet_H * inet_W)
            print(f"      Inferred C={inet_C}")
    amba_inet = amba_inet.reshape(inet_C, inet_H, inet_W)
    print(f"  ✅ AMBA INet: shape={amba_inet.shape}, "
          f"min={amba_inet.min():.6f}, max={amba_inet.max():.6f}, mean={amba_inet.mean():.6f}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 4: Load and preprocess input image
    # ─────────────────────────────────────────────────────────────────────
    print("🖼️  Loading and preprocessing input image...")
    if args.image:
        # Use external image file
        img = load_image_from_file(args.image)
        print(f"  📷 Loaded from file: {args.image}, shape={img.shape}")
    elif os.path.exists(image_bin_path):
        # Use saved image from AMBA
        img = load_image_from_bin(image_bin_path, input_H, input_W, C=3)
        print(f"  📷 Loaded from bin: {image_bin_path}, shape={img.shape}")
        # Note: Image from ea_tensor_data may be in RGB format (AMBA does BGR2RGB internally)
        # We need to check and possibly convert back to BGR for OpenCV-based preprocessing
        print(f"  ⚠️  Note: Saved image from ea_tensor_data may be RGB. "
              f"If results are off, try --image with the original image file.")
    else:
        print(f"❌ ERROR: No input image available.")
        print(f"   Either provide --image <path> or ensure {image_bin_path} exists.")
        sys.exit(1)
    
    # Preprocess for ONNX — use RGB to match model training
    # Evidence: fnet_onnx.cpp explicitly converts BGR→RGB before ONNX inference (line 246)
    input_data = preprocess_for_onnx(
        img, model_H, model_W,
        apply_undistort=not args.no_undistort,
        is_video=args.video,
        use_rgb=True
    )
    print(f"  ✅ Preprocessed input shape: {input_data.shape} (NCHW, RGB)")
    print(f"     Undistort: {not args.no_undistort}, Video mode: {args.video}, Color: RGB")
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 5: Load ONNX models and run inference
    # ─────────────────────────────────────────────────────────────────────
    print("📦 Loading ONNX models...")
    fnet_session = ort.InferenceSession(args.fnet_onnx, providers=['CPUExecutionProvider'])
    inet_session = ort.InferenceSession(args.inet_onnx, providers=['CPUExecutionProvider'])
    
    fnet_input_shape = fnet_session.get_inputs()[0].shape
    fnet_output_shape = fnet_session.get_outputs()[0].shape
    inet_input_shape = inet_session.get_inputs()[0].shape
    inet_output_shape = inet_session.get_outputs()[0].shape
    print(f"  FNet: input={fnet_input_shape}, output={fnet_output_shape}")
    print(f"  INet: input={inet_input_shape}, output={inet_output_shape}")
    print()
    
    print("🚀 Running Python ONNX inference...")
    py_fnet = run_onnx_inference(fnet_session, input_data, "FNet")
    py_inet = run_onnx_inference(inet_session, input_data, "INet")
    
    # Remove batch dimension
    py_fnet_chw = py_fnet[0]  # [C, H, W]
    py_inet_chw = py_inet[0]  # [C, H, W]
    print(f"  Python FNet: shape={py_fnet_chw.shape}, "
          f"min={py_fnet_chw.min():.6f}, max={py_fnet_chw.max():.6f}, mean={py_fnet_chw.mean():.6f}")
    print(f"  Python INet: shape={py_inet_chw.shape}, "
          f"min={py_inet_chw.min():.6f}, max={py_inet_chw.max():.6f}, mean={py_inet_chw.mean():.6f}")
    print()
    
    # Save Python outputs for reference
    os.makedirs(bin_dir, exist_ok=True)
    py_fnet_path = os.path.join(bin_dir, f"python_fnet_frame{frame}.bin")
    py_inet_path = os.path.join(bin_dir, f"python_inet_frame{frame}.bin")
    py_fnet_chw.tofile(py_fnet_path)
    py_inet_chw.tofile(py_inet_path)
    print(f"💾 Saved Python outputs: {py_fnet_path}, {py_inet_path}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────
    # Step 6: Compare AMBA vs Python
    # ─────────────────────────────────────────────────────────────────────
    print("🔍 Comparing AMBA vs Python ONNX outputs...")
    fnet_result = compare_outputs(amba_fnet, py_fnet_chw, "FNet", tolerance=args.tolerance)
    inet_result = compare_outputs(amba_inet, py_inet_chw, "INet", tolerance=args.tolerance)
    
    # Print detailed results
    print_result(fnet_result)
    print_channel_analysis(amba_fnet, py_fnet_chw, "FNet", num_channels=args.channels)
    print_border_analysis(amba_fnet, py_fnet_chw, "FNet", border_width=3)
    
    print_result(inet_result)
    print_channel_analysis(amba_inet, py_inet_chw, "INet", num_channels=args.channels)
    print_border_analysis(amba_inet, py_inet_chw, "INet", border_width=3)
    
    # Print summary
    print_summary([fnet_result, inet_result])
    
    # Parseable output
    print(f"\n   AMBA_FNET_MAX_DIFF={fnet_result.max_diff:.10e}")
    print(f"   AMBA_FNET_MEAN_DIFF={fnet_result.mean_diff:.10e}")
    print(f"   AMBA_INET_MAX_DIFF={inet_result.max_diff:.10e}")
    print(f"   AMBA_INET_MEAN_DIFF={inet_result.mean_diff:.10e}")
    
    # ─────────────────────────────────────────────────────────────────────
    # Optional: Resize sensitivity test
    # ─────────────────────────────────────────────────────────────────────
    if args.resize_test:
        print("\n" + "=" * 100)
        print("🔄 RESIZE SENSITIVITY TEST")
        print("=" * 100)
        print("  This tests how sensitive the ONNX model is to resize algorithm differences.")
        print("  If the model is highly sensitive, AMBA hardware resize vs cv2 resize")
        print("  could explain the correlation gap.\n")
        run_resize_sensitivity_test(fnet_session, img, model_H, model_W, "FNet",
                                     apply_undistort=not args.no_undistort,
                                     is_video=args.video)
        run_resize_sensitivity_test(inet_session, img, model_H, model_W, "INet",
                                     apply_undistort=not args.no_undistort,
                                     is_video=args.video)
    
    # ─────────────────────────────────────────────────────────────────────
    # Optional: Input tensor diagnostic
    # ─────────────────────────────────────────────────────────────────────
    if args.input_diag:
        run_input_diagnostic(
            bin_dir=bin_dir,
            frame=frame,
            fnet_session=fnet_session,
            inet_session=inet_session,
            img=img,
            model_H=model_H, model_W=model_W,
            amba_fnet=amba_fnet,
            amba_inet=amba_inet,
            apply_undistort=not args.no_undistort,
            is_video=args.video,
        )
    
    # ─────────────────────────────────────────────────────────────────────
    # Optional: Run diagnostic to test all layout × color combinations
    # ─────────────────────────────────────────────────────────────────────
    if args.diagnose:
        run_diagnostic(
            fnet_bin_path=fnet_bin_path,
            inet_bin_path=inet_bin_path,
            fnet_session=fnet_session,
            inet_session=inet_session,
            img=img,
            model_H=model_H, model_W=model_W,
            fnet_C=fnet_C, fnet_H=fnet_H, fnet_W=fnet_W,
            inet_C=inet_C, inet_H=inet_H, inet_W=inet_W,
            apply_undistort=not args.no_undistort,
            is_video=args.video,
        )
    
    return 0 if (fnet_result.matches and inet_result.matches) else 1


if __name__ == "__main__":
    sys.exit(main())

