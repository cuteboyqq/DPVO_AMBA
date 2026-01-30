#!/usr/bin/env python3
"""
Compare C++ reproject intermediate values (Gij, Ji, Jj, Jz) with Python reproject.

This script:
1. Loads C++ saved intermediate values (Ti, Tj, Gij, Ji, Jj, Jz) from binary files
2. Loads C++ input data (poses, patches, intrinsics, ii, jj, kk)
3. Calls Python's transform function with jacobian=True
4. Compares Gij, Ji, Jj, Jz in detail
"""

import numpy as np
import torch
import sys
import os
import argparse
from pathlib import Path
try:
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None
    print("Warning: scipy not available. SE3 comparison will be limited.")

# Add DPVO Python path
dpvo_path = Path("/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx")
if not dpvo_path.exists():
    raise FileNotFoundError(f"DPVO_onnx directory not found: {dpvo_path}")

if str(dpvo_path) not in sys.path:
    sys.path.insert(0, str(dpvo_path))

# Import from dpvo.projective_ops
from dpvo.projective_ops import transform as reproject_python
try:
    from dpvo.lietorch import SE3 as SE3_Python
except ImportError:
    try:
        from lietorch import SE3 as SE3_Python
    except ImportError:
        SE3_Python = None
        print("Warning: Could not import lietorch.SE3. Will use torch tensors directly.")


def load_binary_float(filename):
    """Load float32 array from binary file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    data = np.fromfile(filename, dtype=np.float32)
    return data


def load_binary_int32(filename):
    """Load int32 array from binary file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    data = np.fromfile(filename, dtype=np.int32)
    return data


def load_se3_object(filename):
    """Load SE3 object from binary file [7] format: [tx, ty, tz, qx, qy, qz, qw]."""
    data = load_binary_float(filename)
    if len(data) != 7:
        raise ValueError(f"Expected 7 floats for SE3, got {len(data)}")
    return data


def load_matrix(filename, rows, cols):
    """Load matrix from binary file (row-major format).
    
    Args:
        filename: Binary file path
        rows: Number of rows
        cols: Number of columns
    
    Returns:
        numpy.ndarray: Matrix [rows, cols]
    """
    data = load_binary_float(filename)
    expected_size = rows * cols
    if len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} floats for {rows}x{cols} matrix, got {len(data)}")
    matrix = data.reshape(rows, cols)
    return matrix


def load_poses(filename, N):
    """Load poses from binary file [N, 7] format: [tx, ty, tz, qx, qy, qz, qw]."""
    data = load_binary_float(filename)
    poses = data.reshape(N, 7)
    return poses


def load_patches(filename, N, M, P):
    """Load patches from binary file [N*M, 3, P, P]."""
    data = load_binary_float(filename)
    patches = data.reshape(N * M, 3, P, P)
    return patches


def load_intrinsics(filename, N):
    """Load intrinsics from binary file [N, 4] format: [fx, fy, cx, cy]."""
    data = load_binary_float(filename)
    intrinsics = data.reshape(N, 4)
    return intrinsics


def load_edge_indices(ii_file, jj_file, kk_file, num_active):
    """Load edge indices from binary files."""
    ii = load_binary_int32(ii_file)[:num_active]
    jj = load_binary_int32(jj_file)[:num_active]
    kk = load_binary_int32(kk_file)[:num_active]
    return ii, jj, kk


def convert_poses_to_se3(poses_np):
    """Convert numpy poses [N, 7] to Python SE3 objects.
    
    Args:
        poses_np: numpy array [N, 7] format: [tx, ty, tz, qx, qy, qz, qw]
    
    Returns:
        torch.Tensor: SE3 poses [N, 7] (same format)
    """
    return torch.from_numpy(poses_np).float()


def convert_patches_to_torch(patches_np, N, M, P):
    """Convert numpy patches [N*M, 3, P, P] to torch [N, M, 3, P, P]."""
    patches_torch = torch.from_numpy(patches_np).float()
    patches_torch = patches_torch.reshape(N, M, 3, P, P)
    return patches_torch


def convert_intrinsics_to_torch(intrinsics_np):
    """Convert numpy intrinsics [N, 4] to torch."""
    return torch.from_numpy(intrinsics_np).float()


def se3_to_matrix(se3_data):
    """Convert SE3 data [tx, ty, tz, qx, qy, qz, qw] to 4x4 transformation matrix.
    
    Args:
        se3_data: numpy array [7] format: [tx, ty, tz, qx, qy, qz, qw]
    
    Returns:
        numpy.ndarray: 4x4 transformation matrix
    """
    t = se3_data[:3]
    q = se3_data[3:]
    
    # Normalize quaternion
    q_norm = q / np.linalg.norm(q)
    
    # Convert quaternion to rotation matrix
    if R is not None:
        rot = R.from_quat([q_norm[0], q_norm[1], q_norm[2], q_norm[3]])  # [x, y, z, w]
        R_mat = rot.as_matrix()
    else:
        # Manual quaternion to rotation matrix conversion
        x, y, z, w = q_norm
        R_mat = np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t
    
    return T


def compare_se3(cpp_se3, py_se3, name="SE3", tolerance=1e-5):
    """Compare two SE3 objects.
    
    Args:
        cpp_se3: C++ SE3 [7] format: [tx, ty, tz, qx, qy, qz, qw]
        py_se3: Python SE3 [7] format: [tx, ty, tz, qx, qy, qz, qw] or SE3 object
        name: Name for display
        tolerance: Comparison tolerance
    
    Returns:
        dict: Comparison results
    """
    # Convert Python SE3 to numpy if needed
    if isinstance(py_se3, np.ndarray):
        # Already a numpy array
        py_se3_np = np.array(py_se3).flatten()  # Ensure it's 1D
    elif isinstance(py_se3, torch.Tensor):
        py_se3_np = py_se3.cpu().numpy().flatten()
    elif hasattr(py_se3, 'data'):
        # SE3 object or similar with .data attribute
        try:
            data = py_se3.data
            if isinstance(data, torch.Tensor):
                py_se3_np = data.cpu().numpy().flatten()
            else:
                py_se3_np = np.array(data).flatten()
        except AttributeError:
            py_se3_np = np.array(py_se3).flatten()
    else:
        py_se3_np = np.array(py_se3).flatten()
    
    # Ensure both are [7] arrays
    if len(cpp_se3) != 7 or len(py_se3_np) != 7:
        raise ValueError(f"SE3 must be [7], got C++: {len(cpp_se3)}, Python: {len(py_se3_np)}")
    
    # Compare translation
    t_cpp = cpp_se3[:3]
    t_py = py_se3_np[:3]
    t_diff = np.abs(t_cpp - t_py)
    t_max_diff = np.max(t_diff)
    t_mean_diff = np.mean(t_diff)
    
    # Compare quaternion (handle sign ambiguity)
    q_cpp = cpp_se3[3:]
    q_py = py_se3_np[3:]
    
    # Normalize quaternions
    q_cpp_norm = q_cpp / np.linalg.norm(q_cpp)
    q_py_norm = q_py / np.linalg.norm(q_py)
    
    # Handle quaternion sign ambiguity (q and -q represent same rotation)
    q_diff1 = np.abs(q_cpp_norm - q_py_norm)
    q_diff2 = np.abs(q_cpp_norm + q_py_norm)
    q_diff = np.minimum(q_diff1, q_diff2)
    q_max_diff = np.max(q_diff)
    q_mean_diff = np.mean(q_diff)
    
    # Compare rotation matrices
    T_cpp = se3_to_matrix(cpp_se3)
    T_py = se3_to_matrix(py_se3_np)
    R_cpp = T_cpp[:3, :3]
    R_py = T_py[:3, :3]
    R_diff = np.abs(R_cpp - R_py)
    R_max_diff = np.max(R_diff)
    R_mean_diff = np.mean(R_diff)
    
    # Compare translation parts of transformation matrices
    t_T_cpp = T_cpp[:3, 3]
    t_T_py = T_py[:3, 3]
    t_T_diff = np.abs(t_T_cpp - t_T_py)
    t_T_max_diff = np.max(t_T_diff)
    
    # Overall match
    translation_match = t_max_diff < tolerance
    rotation_match = R_max_diff < tolerance
    overall_match = translation_match and rotation_match
    
    return {
        'name': name,
        'cpp': cpp_se3,
        'py': py_se3_np,
        't_max_diff': t_max_diff,
        't_mean_diff': t_mean_diff,
        'q_max_diff': q_max_diff,
        'q_mean_diff': q_mean_diff,
        'R_max_diff': R_max_diff,
        'R_mean_diff': R_mean_diff,
        't_T_max_diff': t_T_max_diff,
        'translation_match': translation_match,
        'rotation_match': rotation_match,
        'overall_match': overall_match
    }


def compare_jacobians(cpp_jac, py_jac, name="Jacobian", tolerance=1e-4):
    """Compare two Jacobian matrices.
    
    Args:
        cpp_jac: C++ Jacobian [rows, cols] (row-major)
        py_jac: Python Jacobian [rows, cols] or torch.Tensor
        name: Name for display
        tolerance: Comparison tolerance
    
    Returns:
        dict: Comparison results
    """
    # Convert Python Jacobian to numpy if needed
    if isinstance(py_jac, torch.Tensor):
        py_jac_np = py_jac.cpu().numpy()
    else:
        py_jac_np = np.array(py_jac)
    
    # Ensure shapes match
    if cpp_jac.shape != py_jac_np.shape:
        raise ValueError(f"Jacobian shapes don't match: C++ {cpp_jac.shape} vs Python {py_jac_np.shape}")
    
    # Compute differences
    diff = np.abs(cpp_jac - py_jac_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Relative difference
    abs_cpp = np.abs(cpp_jac)
    abs_py = np.abs(py_jac_np)
    max_abs = np.maximum(abs_cpp, abs_py)
    rel_diff = np.where(max_abs > 1e-8, diff / max_abs, diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Match status
    match = max_diff < tolerance
    
    return {
        'name': name,
        'cpp': cpp_jac,
        'py': py_jac_np,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'match': match
    }


def print_se3_comparison(result):
    """Print SE3 comparison results."""
    print(f"\n{'='*80}")
    print(f"SE3 COMPARISON: {result['name']}")
    print(f"{'='*80}")
    
    print(f"\nTranslation (t):")
    print(f"  C++: [{result['cpp'][0]:.6f}, {result['cpp'][1]:.6f}, {result['cpp'][2]:.6f}]")
    print(f"  Py:  [{result['py'][0]:.6f}, {result['py'][1]:.6f}, {result['py'][2]:.6f}]")
    print(f"  Max Diff: {result['t_max_diff']:.6e}, Mean Diff: {result['t_mean_diff']:.6e}")
    print(f"  {'✅ MATCH' if result['translation_match'] else '❌ MISMATCH'}")
    
    print(f"\nQuaternion (q):")
    print(f"  C++: [{result['cpp'][3]:.6f}, {result['cpp'][4]:.6f}, {result['cpp'][5]:.6f}, {result['cpp'][6]:.6f}]")
    print(f"  Py:  [{result['py'][3]:.6f}, {result['py'][4]:.6f}, {result['py'][5]:.6f}, {result['py'][6]:.6f}]")
    print(f"  Max Diff: {result['q_max_diff']:.6e}, Mean Diff: {result['q_mean_diff']:.6e}")
    
    print(f"\nRotation Matrix (R):")
    print(f"  Max Diff: {result['R_max_diff']:.6e}, Mean Diff: {result['R_mean_diff']:.6e}")
    print(f"  {'✅ MATCH' if result['rotation_match'] else '❌ MISMATCH'}")
    
    print(f"\nOverall: {'✅ MATCH' if result['overall_match'] else '❌ MISMATCH'}")


def print_jacobian_comparison(result):
    """Print Jacobian comparison results."""
    print(f"\n{'='*80}")
    print(f"JACOBIAN COMPARISON: {result['name']}")
    print(f"{'='*80}")
    
    print(f"\nShape: {result['cpp'].shape}")
    print(f"Max Absolute Diff: {result['max_diff']:.6e}")
    print(f"Mean Absolute Diff: {result['mean_diff']:.6e}")
    print(f"Max Relative Diff: {result['max_rel_diff']:.6e}")
    print(f"Mean Relative Diff: {result['mean_rel_diff']:.6e}")
    
    print(f"\nValues:")
    print(f"  C++ Jacobian:")
    print(f"    {result['cpp']}")
    print(f"  Python Jacobian:")
    print(f"    {result['py']}")
    print(f"  Difference:")
    print(f"    {result['cpp'] - result['py']}")
    
    print(f"\n{'✅ MATCH' if result['match'] else '❌ MISMATCH'}")


def compare_coordinates(cpp_coords, py_coords, name="Coordinates", tolerance=1e-4):
    """Compare reprojected coordinates.
    
    Args:
        cpp_coords: C++ coordinates [2, P, P] or [P, P, 2] or flattened
        py_coords: Python coordinates [2, P, P] or [P, P, 2] or similar
        name: Name for display
        tolerance: Comparison tolerance
    
    Returns:
        dict: Comparison results
    """
    # Ensure both are numpy arrays
    cpp_coords = np.array(cpp_coords)
    if isinstance(py_coords, torch.Tensor):
        py_coords = py_coords.cpu().numpy()
    else:
        py_coords = np.array(py_coords)
    
    # Handle different shapes
    # C++ format: [2, P, P] (u channel, v channel)
    # Python might return: [P, P, 2] or [2, P, P] or [1, 1, P, P, 2] etc.
    
    # Flatten and reshape to [2, P, P] if needed
    if cpp_coords.ndim == 1:
        # Assume flattened [2*P*P]
        P = int(np.sqrt(len(cpp_coords) // 2))
        cpp_coords = cpp_coords.reshape(2, P, P)
    elif cpp_coords.ndim == 3 and cpp_coords.shape[0] != 2:
        # Might be [P, P, 2]
        if cpp_coords.shape[2] == 2:
            cpp_coords = np.transpose(cpp_coords, (2, 0, 1))  # [2, P, P]
    
    # Handle Python coordinates
    if py_coords.ndim == 5:
        # [1, 1, P, P, 2] or [1, 1, 2, P, P]
        py_coords = py_coords.squeeze()
    if py_coords.ndim == 4:
        # [1, P, P, 2] or [1, 2, P, P]
        py_coords = py_coords.squeeze(0)
    if py_coords.ndim == 3:
        if py_coords.shape[2] == 2:
            # [P, P, 2] -> [2, P, P]
            py_coords = np.transpose(py_coords, (2, 0, 1))
        elif py_coords.shape[0] != 2:
            # Might be [P, P, 2] in wrong order
            if py_coords.shape[-1] == 2:
                py_coords = np.transpose(py_coords, (2, 0, 1))
    
    # Ensure shapes match
    if cpp_coords.shape != py_coords.shape:
        raise ValueError(f"Coordinate shapes don't match: C++ {cpp_coords.shape} vs Python {py_coords.shape}")
    
    # Compute differences
    diff = np.abs(cpp_coords - py_coords)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Relative difference
    abs_cpp = np.abs(cpp_coords)
    abs_py = np.abs(py_coords)
    max_abs = np.maximum(abs_cpp, abs_py)
    rel_diff = np.where(max_abs > 1e-8, diff / max_abs, diff)
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Match status
    match = max_diff < tolerance
    
    return {
        'name': name,
        'cpp': cpp_coords,
        'py': py_coords,
        'diff': diff,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'max_rel_diff': max_rel_diff,
        'mean_rel_diff': mean_rel_diff,
        'match': match
    }


def print_coordinates_comparison(result, P, show_all=False):
    """Print coordinates comparison results.
    
    Args:
        result: Comparison result dict from compare_coordinates
        P: Patch size
        show_all: If True, show all pixel values; if False, show summary and center pixel
    """
    print(f"\n{'='*80}")
    print(f"COORDINATES COMPARISON: {result['name']}")
    print(f"{'='*80}")
    
    print(f"\nShape: {result['cpp'].shape}")
    print(f"Max Absolute Diff: {result['max_diff']:.6f}")
    print(f"Mean Absolute Diff: {result['mean_diff']:.6f}")
    print(f"Max Relative Diff: {result['max_rel_diff']:.6e}")
    print(f"Mean Relative Diff: {result['mean_rel_diff']:.6e}")
    
    # Show center pixel comparison
    center_y = P // 2
    center_x = P // 2
    u_cpp_center = result['cpp'][0, center_y, center_x]
    v_cpp_center = result['cpp'][1, center_y, center_x]
    u_py_center = result['py'][0, center_y, center_x]
    v_py_center = result['py'][1, center_y, center_x]
    u_diff_center = result['diff'][0, center_y, center_x]
    v_diff_center = result['diff'][1, center_y, center_x]
    
    print(f"\nCenter Pixel [{center_y}, {center_x}]:")
    print(f"  C++: u={u_cpp_center:.6f}, v={v_cpp_center:.6f}")
    print(f"  Py:  u={u_py_center:.6f}, v={v_py_center:.6f}")
    print(f"  Diff: u={u_diff_center:.6f}, v={v_diff_center:.6f}")
    
    if show_all:
        print(f"\nAll Pixel Values:")
        print(f"  C++ Coordinates (u channel):")
        print(f"    {result['cpp'][0]}")
        print(f"  Python Coordinates (u channel):")
        print(f"    {result['py'][0]}")
        print(f"  Difference (u channel):")
        print(f"    {result['diff'][0]}")
        print(f"  C++ Coordinates (v channel):")
        print(f"    {result['cpp'][1]}")
        print(f"  Python Coordinates (v channel):")
        print(f"    {result['py'][1]}")
        print(f"  Difference (v channel):")
        print(f"    {result['diff'][1]}")
    else:
        # Show summary statistics per channel
        u_max_diff = np.max(result['diff'][0])
        u_mean_diff = np.mean(result['diff'][0])
        v_max_diff = np.max(result['diff'][1])
        v_mean_diff = np.mean(result['diff'][1])
        
        print(f"\nPer-Channel Statistics:")
        print(f"  u channel: max_diff={u_max_diff:.6f}, mean_diff={u_mean_diff:.6f}")
        print(f"  v channel: max_diff={v_max_diff:.6f}, mean_diff={v_mean_diff:.6f}")
    
    print(f"\n{'✅ MATCH' if result['match'] else '❌ MISMATCH'}")


def load_cpp_intermediate_values(bin_dir, frame_num, edge_idx):
    """Load C++ intermediate values (Ti, Tj, Gij, Ji, Jj, Jz) from binary files.
    
    Args:
        bin_dir: Directory containing binary files
        frame_num: Frame number
        edge_idx: Edge index
    
    Returns:
        tuple: (ti_cpp, tj_cpp, gij_cpp, ji_cpp, jj_cpp, jz_cpp)
    """
    print(f"\n{'='*80}")
    print("LOADING C++ INTERMEDIATE VALUES")
    print(f"{'='*80}")
    
    frame_suffix = str(frame_num)
    edge_suffix = str(edge_idx)
    
    ti_file = os.path.join(bin_dir, f"reproject_Ti_frame{frame_suffix}_edge{edge_suffix}.bin")
    tj_file = os.path.join(bin_dir, f"reproject_Tj_frame{frame_suffix}_edge{edge_suffix}.bin")
    gij_file = os.path.join(bin_dir, f"reproject_Gij_frame{frame_suffix}_edge{edge_suffix}.bin")
    ji_file = os.path.join(bin_dir, f"reproject_Ji_frame{frame_suffix}_edge{edge_suffix}.bin")
    jj_file = os.path.join(bin_dir, f"reproject_Jj_frame{frame_suffix}_edge{edge_suffix}.bin")
    jz_file = os.path.join(bin_dir, f"reproject_Jz_frame{frame_suffix}_edge{edge_suffix}.bin")
    
    if not all(os.path.exists(f) for f in [ti_file, tj_file, gij_file, ji_file, jj_file, jz_file]):
        missing = [f for f in [ti_file, tj_file, gij_file, ji_file, jj_file, jz_file] if not os.path.exists(f)]
        raise FileNotFoundError(f"Missing C++ intermediate files: {missing}")
    
    ti_cpp = load_se3_object(ti_file)
    tj_cpp = load_se3_object(tj_file)
    gij_cpp = load_se3_object(gij_file)
    ji_cpp = load_matrix(ji_file, 2, 6)  # [2, 6] Jacobian w.r.t. pose i
    jj_cpp = load_matrix(jj_file, 2, 6)  # [2, 6] Jacobian w.r.t. pose j
    jz_cpp = load_matrix(jz_file, 2, 1)  # [2, 1] Jacobian w.r.t. inverse depth
    
    print(f"✅ Loaded C++ intermediate values for edge {edge_idx}")
    print(f"   Ti: {ti_cpp}")
    print(f"   Tj: {tj_cpp}")
    print(f"   Gij: {gij_cpp}")
    print(f"   Ji shape: {ji_cpp.shape}")
    print(f"   Jj shape: {jj_cpp.shape}")
    print(f"   Jz shape: {jz_cpp.shape}")
    
    return ti_cpp, tj_cpp, gij_cpp, ji_cpp, jj_cpp, jz_cpp


def load_cpp_input_data(bin_dir, frame_num, edge_idx):
    """Load C++ input data (poses, patches, intrinsics, edge indices, coordinates).
    
    Args:
        bin_dir: Directory containing binary files
        frame_num: Frame number
        edge_idx: Edge index
    
    Returns:
        tuple: (poses_cpp, patches_cpp, intrinsics_cpp, ii_cpp, jj_cpp_idx, kk_cpp, 
                coords_cpp_full, N, M, P, num_active, i, j, k)
    """
    print(f"\n{'='*80}")
    print("LOADING C++ INPUT DATA")
    print(f"{'='*80}")
    
    frame_suffix = str(frame_num)
    
    poses_file = os.path.join(bin_dir, f"reproject_poses_frame{frame_suffix}.bin")
    patches_file = os.path.join(bin_dir, f"reproject_patches_frame{frame_suffix}.bin")
    intrinsics_file = os.path.join(bin_dir, f"reproject_intrinsics_frame{frame_suffix}.bin")
    ii_file = os.path.join(bin_dir, f"reproject_ii_frame{frame_suffix}.bin")
    jj_file_idx = os.path.join(bin_dir, f"reproject_jj_frame{frame_suffix}.bin")
    kk_file = os.path.join(bin_dir, f"reproject_kk_frame{frame_suffix}.bin")
    
    if not all(os.path.exists(f) for f in [poses_file, patches_file, intrinsics_file, ii_file, jj_file_idx, kk_file]):
        missing = [f for f in [poses_file, patches_file, intrinsics_file, ii_file, jj_file_idx, kk_file] if not os.path.exists(f)]
        raise FileNotFoundError(f"Missing C++ input files: {missing}")
    
    # Load edge indices and infer dimensions
    num_active = load_binary_int32(ii_file).shape[0]
    ii_cpp, jj_cpp_idx, kk_cpp = load_edge_indices(ii_file, jj_file_idx, kk_file, num_active)
    
    # Infer dimensions from file sizes FIRST (needed to compute i correctly)
    poses_data = load_binary_float(poses_file)
    N = len(poses_data) // 7
    
    # Try to infer M and P from patches file
    patches_data = load_binary_float(patches_file)
    # Patches format: [N*M, 3, P, P]
    total_patches = len(patches_data) // (3 * 3 * 3)  # Assume P=3 initially
    M = total_patches // N if N > 0 else 50
    P = 3  # Default patch size
    
    # Verify P by checking if patches_data size matches
    expected_size = N * M * 3 * P * P
    if len(patches_data) != expected_size:
        # Try P=8 (larger patch size)
        P = 8
        total_patches = len(patches_data) // (3 * P * P)
        M = total_patches // N if N > 0 else 50
        expected_size = N * M * 3 * P * P
        if len(patches_data) != expected_size:
            # Fall back to default
            P = 3
            M = 50
            print(f"⚠️  Warning: Could not infer M and P exactly, using defaults M={M}, P={P}")
    
    # NOW compute frame and patch indices for this edge (after M is inferred)
    k = kk_cpp[edge_idx]
    i = k // M  # source frame (using correct M)
    j = jj_cpp_idx[edge_idx]  # target frame
    
    # Load data
    poses_cpp = load_poses(poses_file, N)
    patches_cpp = load_patches(patches_file, N, M, P)
    intrinsics_cpp = load_intrinsics(intrinsics_file, N)
    
    # Load C++ reprojected coordinates
    coords_file = os.path.join(bin_dir, f"reproject_coords_frame{frame_suffix}.bin")
    coords_cpp_full = None
    if os.path.exists(coords_file):
        coords_cpp_full = load_binary_float(coords_file)
        print(f"✅ Loaded C++ reprojected coordinates from {coords_file}")
        print(f"   Total size: {len(coords_cpp_full)} floats (expected: {num_active * 2 * P * P})")
    else:
        print(f"⚠️  Warning: C++ coordinates file not found: {coords_file}")
        print(f"   Will skip coordinates comparison")
    
    print(f"✅ Loaded C++ input data")
    print(f"   N (frames): {N}")
    print(f"   M (patches per frame): {M}")
    print(f"   P (patch size): {P}")
    print(f"   num_active: {num_active}")
    print(f"   Edge {edge_idx}: kk={k}, i={i}, j={j}")
    
    return (poses_cpp, patches_cpp, intrinsics_cpp, ii_cpp, jj_cpp_idx, kk_cpp,
            coords_cpp_full, N, M, P, num_active, i, j, k)


def prepare_python_inputs(poses_cpp, patches_cpp, intrinsics_cpp, i, j, k, N, M, P):
    """Prepare Python reproject inputs from C++ data.
    
    Args:
        poses_cpp: C++ poses [N, 7]
        patches_cpp: C++ patches [N*M, 3, P, P]
        intrinsics_cpp: C++ intrinsics [N, 4]
        i: Source frame index
        j: Target frame index
        k: Global patch index
        N: Number of frames
        M: Patches per frame
        P: Patch size
    
    Returns:
        tuple: (poses_batch, patches_batch, intrinsics_batch, ii_single, jj_single, kk_single)
    """
    print(f"\n{'='*80}")
    print("PREPARING PYTHON INPUTS")
    print(f"{'='*80}")
    
    # Convert poses to Python SE3 format if needed
    # Python's lietorch.SE3 expects [N, 7] format: [tx, ty, tz, qw, qx, qy, qz]
    # C++ uses [tx, ty, tz, qx, qy, qz, qw]
    poses_py_format = np.zeros((N, 7))
    poses_py_format[:, :3] = poses_cpp[:, :3]  # Translation
    poses_py_format[:, 3] = poses_cpp[:, 6]     # qw
    poses_py_format[:, 4:7] = poses_cpp[:, 3:6]  # qx, qy, qz
    
    patches_torch = convert_patches_to_torch(patches_cpp, N, M, P)
    intrinsics_torch = convert_intrinsics_to_torch(intrinsics_cpp)
    
    # Prepare batch inputs
    # SE3 constructor expects [batch, N, 7] format
    poses_torch = torch.from_numpy(poses_py_format).float()
    poses_torch_batch = poses_torch.unsqueeze(0)  # [1, N, 7]
    
    if SE3_Python is not None:
        poses_batch = SE3_Python(poses_torch_batch)
    else:
        poses_batch = poses_torch_batch  # [1, N, 7]
    
    patches_batch = patches_torch.reshape(N * M, 3, P, P).unsqueeze(0)  # [1, N*M, 3, P, P]
    intrinsics_batch = intrinsics_torch.unsqueeze(0)  # [1, N, 4]
    
    # Edge indices for this specific edge
    ii_single = torch.tensor([i]).long()
    jj_single = torch.tensor([j]).long()
    kk_single = torch.tensor([k]).long()
    
    print(f"✅ Prepared Python inputs")
    print(f"   poses_batch type: {type(poses_batch)}")
    if isinstance(poses_batch, torch.Tensor):
        print(f"   poses_batch shape: {poses_batch.shape}")
    print(f"   patches_batch shape: {patches_batch.shape}")
    print(f"   intrinsics_batch shape: {intrinsics_batch.shape}")
    print(f"   ii={i}, jj={j}, kk={k}")
    
    return poses_batch, patches_batch, intrinsics_batch, ii_single, jj_single, kk_single


def extract_python_jacobians(ji_py, jj_py, jz_py, P):
    """Extract Python Jacobians at center pixel.
    
    Args:
        ji_py: Python Ji Jacobian tensor
        jj_py: Python Jj Jacobian tensor
        jz_py: Python Jz Jacobian tensor
        P: Patch size
    
    Returns:
        tuple: (ji_py_center, jj_py_center, jz_py_center) as numpy arrays
    """
    center_y = P // 2
    center_x = P // 2
    
    # Python Jacobians can have different shapes:
    # - [1, 1, 2, 6] if called with single edge (our case)
    # - [1, num_active, 2, 6] if called with multiple edges
    # - [1, num_active, 2, P, P, 6] if broadcast to all pixels
    
    # Extract Jacobians for center pixel
    if len(ji_py.shape) == 4:
        if ji_py.shape[1] == 1:  # [1, 1, 2, 6] - single edge call
            ji_py_center = ji_py[0, 0].cpu().numpy()  # [2, 6]
            jj_py_center = jj_py[0, 0].cpu().numpy()  # [2, 6]
            jz_py_center = jz_py[0, 0].cpu().numpy()  # [2, 1] or [2]
        else:  # [1, num_active, 2, 6] - multiple edges
            ji_py_center = ji_py[0, 0].cpu().numpy()  # [2, 6]
            jj_py_center = jj_py[0, 0].cpu().numpy()  # [2, 6]
            jz_py_center = jz_py[0, 0].cpu().numpy()  # [2, 1] or [2]
    elif len(ji_py.shape) == 6:  # [1, num_active, 2, P, P, 6]
        ji_py_center = ji_py[0, 0, :, center_y, center_x].cpu().numpy()  # [2, 6]
        jj_py_center = jj_py[0, 0, :, center_y, center_x].cpu().numpy()  # [2, 6]
        jz_py_center = jz_py[0, 0, :, center_y, center_x].cpu().numpy()  # [2, 1] or [2]
    elif len(ji_py.shape) == 5:  # [1, num_active, P, P, 2, 6] or similar
        ji_py_center = ji_py[0, 0, center_y, center_x].cpu().numpy()  # [2, 6]
        jj_py_center = jj_py[0, 0, center_y, center_x].cpu().numpy()  # [2, 6]
        jz_py_center = jz_py[0, 0, center_y, center_x].cpu().numpy()  # [2, 1] or [2]
    else:
        raise ValueError(f"Unexpected ji_py shape: {ji_py.shape}")
    
    # Ensure jz_py_center is [2, 1]
    if jz_py_center.ndim == 1:
        jz_py_center = jz_py_center.reshape(2, 1)
    elif jz_py_center.shape == (2,):
        jz_py_center = jz_py_center.reshape(2, 1)
    
    print(f"✅ Extracted Python Jacobians at center pixel [{center_y}, {center_x}]")
    print(f"   ji_py_center shape: {ji_py_center.shape}")
    print(f"   jj_py_center shape: {jj_py_center.shape}")
    print(f"   jz_py_center shape: {jz_py_center.shape}")
    
    return ji_py_center, jj_py_center, jz_py_center


def compute_python_gij(poses_batch, i, j, poses_cpp):
    """Compute Python Gij = Tj * Ti^-1.
    
    Args:
        poses_batch: Python poses (SE3 object or tensor) - used for Gij computation
        i: Source frame index
        j: Target frame index
        poses_cpp: C++ poses [N, 7] in C++ format - used for Ti/Tj extraction (same as C++ input)
    
    Returns:
        tuple: (gij_py, ti_py, tj_py) in C++ format [7]
    """
    # Ti and Tj should come from the same input poses that C++ used
    # C++ saves Ti = poses_cpp[i] and Tj = poses_cpp[j]
    ti_py = poses_cpp[i].copy()  # [tx, ty, tz, qx, qy, qz, qw] - same as C++
    tj_py = poses_cpp[j].copy()  # [tx, ty, tz, qx, qy, qz, qw] - same as C++
    
    # Compute Gij using Python SE3 operations for comparison
    if SE3_Python is not None:
        # poses_batch is SE3 object with shape [1, N, 7]
        # Access individual poses - try both indexing methods
        try:
            ti_py_se3 = poses_batch[0, i]  # Try batch indexing first
            tj_py_se3 = poses_batch[0, j]
        except (IndexError, TypeError):
            # Fallback: direct indexing
            ti_py_se3 = poses_batch[i]
            tj_py_se3 = poses_batch[j]
        
        gij_py_se3 = tj_py_se3 * ti_py_se3.inv()
        
        # Extract data and ensure it's 1D [7]
        gij_py_data_raw = gij_py_se3.data.cpu().numpy()
        # Squeeze all dimensions to get 1D array
        gij_py_data = np.array(gij_py_data_raw).flatten()
        if len(gij_py_data) != 7:
            raise ValueError(f"Expected gij_py_data to have length 7, got {len(gij_py_data)} with shape {gij_py_data_raw.shape}")
        
        # Convert to C++ format: [tx, ty, tz, qx, qy, qz, qw]
        gij_py = np.zeros(7)
        gij_py[:3] = gij_py_data[:3]  # Translation
        gij_py[3:6] = gij_py_data[4:7]  # qx, qy, qz
        gij_py[6] = gij_py_data[3]  # qw
    else:
        # Manual computation using scipy
        if R is None:
            raise RuntimeError("Cannot compute Gij without lietorch or scipy")
        # Compute Gij manually using the same poses_cpp that C++ used
        Ti = se3_to_matrix(ti_py)
        Tj = se3_to_matrix(tj_py)
        Gij_mat = Tj @ np.linalg.inv(Ti)
        # Extract translation and rotation
        gij_t = Gij_mat[:3, 3]
        gij_R = Gij_mat[:3, :3]
        # Convert rotation matrix to quaternion
        rot = R.from_matrix(gij_R)
        gij_q = rot.as_quat()  # [x, y, z, w]
        gij_py = np.concatenate([gij_t, gij_q])
    
    print(f"✅ Computed Python Gij")
    print(f"   Gij: {gij_py}")
    
    return gij_py, ti_py, tj_py


def extract_python_coordinates(coords_py):
    """Extract Python coordinates and normalize to [2, P, P] format.
    
    Args:
        coords_py: Python coordinates (various formats)
    
    Returns:
        numpy.ndarray: Coordinates in [2, P, P] format
    """
    coords_py_edge = coords_py
    if isinstance(coords_py_edge, torch.Tensor):
        coords_py_edge = coords_py_edge.cpu().numpy()
    
    original_shape = coords_py_edge.shape
    print(f"   Original coords_py shape: {original_shape}")
    
    # Handle different shapes
    if coords_py_edge.ndim == 5:
        # [1, 1, P, P, 2] or [1, 1, 2, P, P] or [1, 1, P, P, channels]
        # Check which dimension has size 2 (channels)
        if coords_py_edge.shape[-1] == 2:
            # [1, 1, P, P, 2] -> [P, P, 2] -> [2, P, P]
            coords_py_edge = coords_py_edge.squeeze()
            if coords_py_edge.ndim == 3 and coords_py_edge.shape[2] == 2:
                coords_py_edge = np.transpose(coords_py_edge, (2, 0, 1))
        elif coords_py_edge.shape[2] == 2:
            # [1, 1, 2, P, P] -> [2, P, P]
            coords_py_edge = coords_py_edge.squeeze()
        else:
            # Try squeezing and see what we get
            coords_py_edge = coords_py_edge.squeeze()
    
    if coords_py_edge.ndim == 4:
        # [1, P, P, 2] or [1, 2, P, P]
        if coords_py_edge.shape[-1] == 2:
            # [1, P, P, 2] -> [P, P, 2] -> [2, P, P]
            coords_py_edge = coords_py_edge.squeeze(0)
            if coords_py_edge.ndim == 3 and coords_py_edge.shape[2] == 2:
                coords_py_edge = np.transpose(coords_py_edge, (2, 0, 1))
        elif coords_py_edge.shape[1] == 2:
            # [1, 2, P, P] -> [2, P, P]
            coords_py_edge = coords_py_edge.squeeze(0)
        else:
            coords_py_edge = coords_py_edge.squeeze(0)
    
    if coords_py_edge.ndim == 3:
        if coords_py_edge.shape[2] == 2:
            # [P, P, 2] -> [2, P, P]
            coords_py_edge = np.transpose(coords_py_edge, (2, 0, 1))
        elif coords_py_edge.shape[0] == 2:
            # Already [2, P, P]
            pass
        elif coords_py_edge.shape[0] == 3 and coords_py_edge.shape[1] == 3 and coords_py_edge.shape[2] == 3:
            # This is [P, P, P] which is wrong - might be depth channel included
            # Python might return [P, P, 3] with depth, but we only need [u, v]
            # Check if last dimension has depth - if so, take first 2 channels
            print(f"   Warning: Got shape [P, P, P], might need to extract u,v channels")
            # For now, assume it's actually [P, P, 2] but shape detection is wrong
            # Try to reshape or check actual content
            pass
    
    # Final check: ensure we have [2, P, P]
    if coords_py_edge.ndim == 3:
        if coords_py_edge.shape[0] == 2:
            # Already [2, P, P] - correct format
            pass
        elif coords_py_edge.shape[2] == 2:
            # [P, P, 2] -> [2, P, P]
            coords_py_edge = np.transpose(coords_py_edge, (2, 0, 1))
        elif coords_py_edge.shape[0] == 3 and coords_py_edge.shape[1] == 3:
            # Might be [P, P, 3] or [3, P, P] - check last dimension
            if coords_py_edge.shape[2] == 3:
                # [P, P, 3] - might include depth, extract first 2 channels (u, v)
                coords_py_edge = coords_py_edge[:, :, :2]
                coords_py_edge = np.transpose(coords_py_edge, (2, 0, 1))
            elif coords_py_edge.shape[0] == 3:
                # [3, P, P] - might include depth, extract first 2 channels
                coords_py_edge = coords_py_edge[:2, :, :]
    
    # Verify final shape
    if coords_py_edge.ndim != 3 or coords_py_edge.shape[0] != 2:
        raise ValueError(f"Failed to extract coordinates in [2, P, P] format. Final shape: {coords_py_edge.shape}")
    
    print(f"✅ Extracted Python coordinates")
    print(f"   Final coords_py_edge shape: {coords_py_edge.shape}")
    
    return coords_py_edge


def call_python_reproject(poses_batch, patches_batch, intrinsics_batch, 
                          ii_single, jj_single, kk_single, P, i, j, poses_cpp):
    """Call Python reproject function and extract all results.
    
    Args:
        poses_batch: Python poses batch
        patches_batch: Python patches batch
        intrinsics_batch: Python intrinsics batch
        ii_single: Source frame index tensor
        jj_single: Target frame index tensor
        kk_single: Patch index tensor
        P: Patch size
        i: Source frame index (for Gij computation)
        j: Target frame index (for Gij computation)
        poses_cpp: C++ poses [N, 7] in C++ format (same input as C++ used)
    
    Returns:
        tuple: (gij_py, ti_py, tj_py, ji_py_center, jj_py_center, jz_py_center, coords_py_edge)
    """
    print(f"\n{'='*80}")
    print("CALLING PYTHON REPROJECT WITH JACOBIAN")
    print(f"{'='*80}")
    
    try:
        result = reproject_python(
            poses_batch,
            patches_batch,
            intrinsics_batch,
            ii_single,
            jj_single,
            kk_single,
            depth=True,
            valid=True,
            jacobian=True
        )
        
        # Python reproject return format with jacobian=True:
        # - Returns 3 values: (coords, depth/valid, (ji, jj, jz))
        #   When depth=True: (coords, depth, (ji, jj, jz))
        #   When valid=True: (coords, valid, (ji, jj, jz))
        #   When both=True: (coords, depth, (ji, jj, jz)) - depth takes precedence
        if len(result) == 3:
            coords_py, depth_or_valid_py, jacobians = result
            ji_py, jj_py, jz_py = jacobians
            # Since we set depth=True, the second value should be depth
            depth_py = depth_or_valid_py
            valid_py = None  # Not returned when depth=True
        elif len(result) == 4:
            # Some versions might return (coords, depth, valid, (ji, jj, jz))
            coords_py, depth_py, valid_py, jacobians = result
            ji_py, jj_py, jz_py = jacobians
        else:
            raise ValueError(f"Unexpected return format from Python reproject: {len(result)} values, got: {result}")
        
        print(f"✅ Python reproject returned")
        print(f"   coords_py shape: {coords_py.shape}")
        print(f"   depth_py shape: {depth_py.shape if depth_py is not None else None}")
        print(f"   valid_py shape: {valid_py.shape if valid_py is not None else None}")
        print(f"   ji_py shape: {ji_py.shape if ji_py is not None else None}")
        print(f"   jj_py shape: {jj_py.shape if jj_py is not None else None}")
        print(f"   jz_py shape: {jz_py.shape if jz_py is not None else None}")
        
        # Extract Jacobians
        ji_py_center, jj_py_center, jz_py_center = extract_python_jacobians(ji_py, jj_py, jz_py, P)
        
        # Compute Gij (Ti and Tj come from same input poses_cpp that C++ used)
        gij_py, ti_py, tj_py = compute_python_gij(poses_batch, i, j, poses_cpp)
        
        # Extract coordinates
        coords_py_edge = extract_python_coordinates(coords_py)
        
        return gij_py, ti_py, tj_py, ji_py_center, jj_py_center, jz_py_center, coords_py_edge
        
    except Exception as e:
        print(f"❌ Error calling Python reproject: {e}")
        import traceback
        traceback.print_exc()
        raise


def compare_all_results(ti_cpp, tj_cpp, gij_cpp, ji_cpp, jj_cpp, jz_cpp,
                       ti_py, tj_py, gij_py, ji_py_center, jj_py_center, jz_py_center,
                       coords_cpp_full, coords_py_edge, edge_idx, P, tolerance):
    """Compare all intermediate values between C++ and Python.
    
    Args:
        ti_cpp, tj_cpp, gij_cpp: C++ SE3 objects
        ji_cpp, jj_cpp, jz_cpp: C++ Jacobians
        ti_py, tj_py, gij_py: Python SE3 objects
        ji_py_center, jj_py_center, jz_py_center: Python Jacobians
        coords_cpp_full: C++ coordinates array (or None)
        coords_py_edge: Python coordinates array (or None)
        edge_idx: Edge index
        P: Patch size
        tolerance: Comparison tolerance
    
    Returns:
        dict: Comparison results with 'all_match' key
    """
    print(f"\n{'='*80}")
    print("COMPARING RESULTS")
    print(f"{'='*80}")
    
    # Compare SE3 objects
    ti_result = compare_se3(ti_cpp, ti_py, "Ti", tolerance)
    print_se3_comparison(ti_result)
    
    tj_result = compare_se3(tj_cpp, tj_py, "Tj", tolerance)
    print_se3_comparison(tj_result)
    
    gij_result = compare_se3(gij_cpp, gij_py, "Gij", tolerance)
    print_se3_comparison(gij_result)
    
    # Compare Jacobians
    ji_result = compare_jacobians(ji_cpp, ji_py_center, "Ji", tolerance)
    print_jacobian_comparison(ji_result)
    
    jj_result = compare_jacobians(jj_cpp, jj_py_center, "Jj", tolerance)
    print_jacobian_comparison(jj_result)
    
    jz_result = compare_jacobians(jz_cpp, jz_py_center, "Jz", tolerance)
    print_jacobian_comparison(jz_result)
    
    # Compare coordinates if available
    coords_result = None
    if coords_cpp_full is not None and coords_py_edge is not None:
        # Extract C++ coordinates for this edge
        # Format: [num_active, 2, P, P] flattened
        edge_base = edge_idx * 2 * P * P
        if edge_base + 2 * P * P <= len(coords_cpp_full):
            coords_cpp_edge_flat = coords_cpp_full[edge_base:edge_base + 2 * P * P]
            coords_cpp_edge = coords_cpp_edge_flat.reshape(2, P, P)  # [2, P, P]
            
            print(f"\n✅ Extracted C++ coordinates for edge {edge_idx}")
            print(f"   coords_cpp_edge shape: {coords_cpp_edge.shape}")
            
            # Compare coordinates
            try:
                coords_result = compare_coordinates(coords_cpp_edge, coords_py_edge, "Reprojected Coordinates", tolerance)
                print_coordinates_comparison(coords_result, P, show_all=False)
            except Exception as e:
                print(f"⚠️  Error comparing coordinates: {e}")
                coords_result = None
        else:
            print(f"⚠️  Warning: Edge {edge_idx} coordinates out of bounds in C++ file")
            print(f"   File size: {len(coords_cpp_full)}, Expected: {edge_base + 2 * P * P}")
    elif coords_cpp_full is None:
        print(f"\n⚠️  Skipping coordinates comparison: C++ coordinates file not found")
    elif coords_py_edge is None:
        print(f"\n⚠️  Skipping coordinates comparison: Python coordinates not available")
    
    # Compute overall match
    all_match = (
        ti_result['overall_match'] and
        tj_result['overall_match'] and
        gij_result['overall_match'] and
        ji_result['match'] and
        jj_result['match'] and
        jz_result['match']
    )
    
    if coords_result is not None:
        all_match = all_match and coords_result['match']
    
    return {
        'ti_result': ti_result,
        'tj_result': tj_result,
        'gij_result': gij_result,
        'ji_result': ji_result,
        'jj_result': jj_result,
        'jz_result': jz_result,
        'coords_result': coords_result,
        'all_match': all_match
    }


def print_summary(results):
    """Print comparison summary.
    
    Args:
        results: Dictionary with comparison results from compare_all_results
    """
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"Ti Match: {'✅' if results['ti_result']['overall_match'] else '❌'}")
    print(f"Tj Match: {'✅' if results['tj_result']['overall_match'] else '❌'}")
    print(f"Gij Match: {'✅' if results['gij_result']['overall_match'] else '❌'}")
    print(f"Ji Match: {'✅' if results['ji_result']['match'] else '❌'}")
    print(f"Jj Match: {'✅' if results['jj_result']['match'] else '❌'}")
    print(f"Jz Match: {'✅' if results['jz_result']['match'] else '❌'}")
    if results['coords_result'] is not None:
        print(f"Coords Match: {'✅' if results['coords_result']['match'] else '❌'}")
    else:
        print(f"Coords Match: ⚠️  (not compared - file not found)")
    print(f"\nOverall: {'✅ ALL MATCH' if results['all_match'] else '❌ SOME MISMATCHES'}")


def main():
    parser = argparse.ArgumentParser(description="Compare C++ and Python reproject intermediate values")
    parser.add_argument("--frame", type=int, required=True, help="Frame number to compare")
    parser.add_argument("--edge", type=int, default=0, help="Edge index to compare (default: 0)")
    parser.add_argument("--bin-dir", type=str, default="bin_file", help="Directory containing binary files")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Comparison tolerance (default: 1e-4)")
    args = parser.parse_args()
    
    frame_num = args.frame
    edge_idx = args.edge
    bin_dir = args.bin_dir
    tolerance = args.tolerance
    
    print(f"{'='*80}")
    print(f"REPROJECT INTERMEDIATE VALUES COMPARISON")
    print(f"{'='*80}")
    print(f"Frame: {frame_num}")
    print(f"Edge: {edge_idx}")
    print(f"Binary directory: {bin_dir}")
    print(f"Tolerance: {tolerance}")
    print(f"{'='*80}")
    
    # Load C++ intermediate values
    ti_cpp, tj_cpp, gij_cpp, ji_cpp, jj_cpp, jz_cpp = load_cpp_intermediate_values(
        bin_dir, frame_num, edge_idx
    )
    
    # Load C++ input data
    (poses_cpp, patches_cpp, intrinsics_cpp, ii_cpp, jj_cpp_idx, kk_cpp,
     coords_cpp_full, N, M, P, num_active, i, j, k) = load_cpp_input_data(
        bin_dir, frame_num, edge_idx
    )
    
    # Prepare Python inputs
    poses_batch, patches_batch, intrinsics_batch, ii_single, jj_single, kk_single = prepare_python_inputs(
        poses_cpp, patches_cpp, intrinsics_cpp, i, j, k, N, M, P
    )
    
    # Call Python reproject and extract results
    # Pass poses_cpp so Ti/Tj use the same input poses that C++ used
    gij_py, ti_py, tj_py, ji_py_center, jj_py_center, jz_py_center, coords_py_edge = call_python_reproject(
        poses_batch, patches_batch, intrinsics_batch,
        ii_single, jj_single, kk_single, P, i, j, poses_cpp
    )
    
    # Compare all results
    results = compare_all_results(
        ti_cpp, tj_cpp, gij_cpp, ji_cpp, jj_cpp, jz_cpp,
        ti_py, tj_py, gij_py, ji_py_center, jj_py_center, jz_py_center,
        coords_cpp_full, coords_py_edge, edge_idx, P, tolerance
    )
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()

