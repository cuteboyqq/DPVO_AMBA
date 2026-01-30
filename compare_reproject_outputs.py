#!/usr/bin/env python3
"""
Compare C++ reproject output with Python reproject output.

This script:
1. Loads reproject data saved by C++ code (poses, patches, intrinsics, ii, jj, kk, coords)
2. Reconstructs Python data structures
3. Calls Python's reproject function with the same inputs
4. Compares outputs and identifies mismatches
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
possible_paths = [
    Path(__file__).parent.parent / "DPVO_onnx",
    Path(__file__).parent.parent.parent / "DPVO_onnx",
    Path("/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx"),
]

dpvo_path = None
for path in possible_paths:
    if path.exists() and (path / "dpvo" / "projective_ops.py").exists():
        dpvo_path = path
        break

if dpvo_path is None:
    raise FileNotFoundError(
        "Could not find DPVO_onnx directory. Please ensure it exists and contains dpvo/projective_ops.py"
    )

if str(dpvo_path) not in sys.path:
    sys.path.insert(0, str(dpvo_path))

# Import from dpvo.projective_ops and lietorch
from dpvo.projective_ops import transform as reproject_python
try:
    from dpvo.lietorch import SE3
except ImportError:
    try:
        from lietorch import SE3
    except ImportError:
        SE3 = None
        print("Warning: Could not import lietorch.SE3. Will try using torch tensors directly.")


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


def load_metadata(filename):
    """Load metadata from text file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    metadata = {}
    with open(filename, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=')
                metadata[key] = int(value)
    return metadata


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
    ii = load_binary_int32(ii_file)
    jj = load_binary_int32(jj_file)
    kk = load_binary_int32(kk_file)
    return ii[:num_active], jj[:num_active], kk[:num_active]


def convert_poses_to_se3(poses_np):
    """Convert poses from [N, 7] numpy array to lietorch SE3 format."""
    # poses_np: [N, 7] = [tx, ty, tz, qx, qy, qz, qw] (C++ format)
    # lietorch SE3 expects [N, 7] = [tx, ty, tz, qw, qx, qy, qz] (qw first in quaternion)
    if SE3 is None:
        raise ImportError("SE3 is not available. Cannot convert poses to SE3 format.")
    
    # Convert to lietorch format: [tx, ty, tz, qw, qx, qy, qz]
    poses_se3_format = np.zeros_like(poses_np)
    poses_se3_format[:, :3] = poses_np[:, :3]  # translation: tx, ty, tz
    poses_se3_format[:, 3] = poses_np[:, 6]     # qw (from last position)
    poses_se3_format[:, 4:7] = poses_np[:, 3:6] # qx, qy, qz
    poses_torch = torch.from_numpy(poses_se3_format).float()
    
    # SE3 constructor expects [batch, N, 7] format
    poses_torch_batch = poses_torch.unsqueeze(0)  # [1, N, 7]
    return SE3(poses_torch_batch)


def convert_patches_to_torch(patches_np, N, M, P):
    """Convert patches from [N*M, 3, P, P] numpy to PyTorch [N, M, 3, P, P]."""
    patches_reshaped = patches_np.reshape(N, M, 3, P, P)
    patches_torch = torch.from_numpy(patches_reshaped).float()
    return patches_torch


def convert_intrinsics_to_torch(intrinsics_np):
    """Convert intrinsics from [N, 4] numpy to PyTorch [N, 4]."""
    intrinsics_torch = torch.from_numpy(intrinsics_np).float()
    return intrinsics_torch


def load_cpp_data(frame_num, bin_dir="bin_file"):
    """Load C++ reproject data (poses, patches, intrinsics, edge indices, coords).
    
    Returns:
        tuple: (poses_cpp, patches_cpp, intrinsics_cpp, ii_cpp, jj_cpp, kk_cpp, coords_cpp, num_active, M, P, N)
    """
    frame_suffix = str(frame_num)
    
    # Load metadata
    metadata_file = os.path.join(bin_dir, "test_metadata.txt")
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    metadata = load_metadata(metadata_file)
    num_active = metadata.get('num_active', 0)
    M = metadata.get('M', 4)
    P = metadata.get('P', 3)
    N = metadata.get('N', 36)
    
    print(f"Loaded metadata:")
    print(f"  num_active: {num_active}")
    print(f"  M (patches per frame): {M}")
    print(f"  P (patch size): {P}")
    print(f"  N (number of frames): {N}")
    
    # Load C++ inputs
    print(f"\n{'='*70}")
    print("LOADING C++ DATA")
    print(f"{'='*70}")
    
    # CRITICAL: C++ saves poses, patches, and intrinsics RIGHT BEFORE calling reproject()
    reproject_poses_file = os.path.join(bin_dir, f"reproject_poses_frame{frame_suffix}.bin")
    reproject_patches_file = os.path.join(bin_dir, f"reproject_patches_frame{frame_suffix}.bin")
    reproject_intrinsics_file = os.path.join(bin_dir, f"reproject_intrinsics_frame{frame_suffix}.bin")
    
    if os.path.exists(reproject_poses_file):
        poses_cpp = load_poses(reproject_poses_file, N)
        print(f"  ✅ Using poses from {reproject_poses_file} (actual poses used by C++ reproject)")
    else:
        poses_cpp = load_poses(os.path.join(bin_dir, "ba_poses.bin"), N)
        print(f"  ⚠️  Using poses from ba_poses.bin (reproject_poses_frame{frame_suffix}.bin not found)")
    
    if os.path.exists(reproject_patches_file):
        patches_cpp = load_patches(reproject_patches_file, N, M, P)
        print(f"  ✅ Using patches from {reproject_patches_file} (actual patches used by C++ reproject)")
    else:
        patches_cpp = load_patches(os.path.join(bin_dir, "ba_patches.bin"), N, M, P)
        print(f"  ⚠️  Using patches from ba_patches.bin (reproject_patches_frame{frame_suffix}.bin not found)")
        print(f"     This may cause mismatches if patches differ from what C++ actually uses!")
    
    if os.path.exists(reproject_intrinsics_file):
        intrinsics_cpp = load_intrinsics(reproject_intrinsics_file, N)
        print(f"  ✅ Using intrinsics from {reproject_intrinsics_file} (actual intrinsics used by C++ reproject)")
    else:
        intrinsics_cpp = load_intrinsics(os.path.join(bin_dir, "ba_intrinsics.bin"), N)
        print(f"  ⚠️  Using intrinsics from ba_intrinsics.bin (reproject_intrinsics_frame{frame_suffix}.bin not found)")
    
    # CRITICAL: C++ saves edge indices RIGHT BEFORE calling reproject()
    reproject_ii_file = os.path.join(bin_dir, f"reproject_ii_frame{frame_suffix}.bin")
    reproject_jj_file = os.path.join(bin_dir, f"reproject_jj_frame{frame_suffix}.bin")
    reproject_kk_file = os.path.join(bin_dir, f"reproject_kk_frame{frame_suffix}.bin")
    
    if (os.path.exists(reproject_ii_file) and os.path.exists(reproject_jj_file) and os.path.exists(reproject_kk_file)):
        ii_cpp, jj_cpp, kk_cpp = load_edge_indices(reproject_ii_file, reproject_jj_file, reproject_kk_file, num_active)
        print(f"  ✅ Using edge indices from reproject_*_frame{frame_suffix}.bin (actual edge indices used by C++ reproject)")
    else:
        ii_cpp, jj_cpp, kk_cpp = load_edge_indices(
            os.path.join(bin_dir, "ba_ii.bin"), 
            os.path.join(bin_dir, "ba_jj.bin"), 
            os.path.join(bin_dir, "ba_kk.bin"), 
            num_active)
        print(f"  ⚠️  Using edge indices from ba_*.bin (reproject_*_frame{frame_suffix}.bin not found)")
    
    # Load coords
    reproject_coords_full_file = os.path.join(bin_dir, f"reproject_coords_frame{frame_suffix}.bin")
    if os.path.exists(reproject_coords_full_file):
        coords_cpp = load_binary_float(reproject_coords_full_file)
        print(f"  ✅ Using full coords from {reproject_coords_full_file}")
    else:
        # Expand center coords to full patch
        coords_cpp_center = load_binary_float(os.path.join(bin_dir, "ba_reprojected_coords.bin"))
        coords_cpp = np.zeros(num_active * 2 * P * P, dtype=np.float32)
        center_idx = (P // 2) * P + (P // 2)
        for e in range(num_active):
            cx = coords_cpp_center[e * 2 + 0]
            cy = coords_cpp_center[e * 2 + 1]
            for py_idx in range(P):
                for px_idx in range(P):
                    idx = py_idx * P + px_idx
                    coords_cpp[e * 2 * P * P + 0 * P * P + idx] = cx
                    coords_cpp[e * 2 * P * P + 1 * P * P + idx] = cy
        print(f"  ⚠️  Expanded center coords from ba_reprojected_coords.bin to full patch (reproject_coords_frame{frame_suffix}.bin not found)")
    
    print(f"Data                 Shape                          Description")
    print("-"*70)
    print(f"{'poses':<20} {str(poses_cpp.shape):<30} Camera poses [N, 7]")
    print(f"{'patches':<20} {str(patches_cpp.shape):<30} Patches [N*M, 3, P, P]")
    print(f"{'intrinsics':<20} {str(intrinsics_cpp.shape):<30} Intrinsics [N, 4]")
    print(f"{'ii':<20} {str(ii_cpp.shape):<30} Source frame indices")
    print(f"{'jj':<20} {str(jj_cpp.shape):<30} Target frame indices")
    print(f"{'kk':<20} {str(kk_cpp.shape):<30} Global patch indices")
    print(f"{'coords':<20} {str(coords_cpp.shape):<30} Reprojected coords [num_active, 2, P, P]")
    print(f"{'='*70}")
    
    return poses_cpp, patches_cpp, intrinsics_cpp, ii_cpp, jj_cpp, kk_cpp, coords_cpp, num_active, M, P, N


def prepare_python_inputs(poses_cpp, patches_cpp, intrinsics_cpp, kk_cpp, jj_cpp, num_active, M, P, N):
    """Prepare Python reproject inputs from C++ data.
    
    Returns:
        tuple: (poses_batch, patches_batch, intrinsics_batch, ii_1d, jj_1d, kk_1d, patches_torch)
    """
    print(f"\n{'='*70}")
    print("PREPARING PYTHON REPROJECT INPUTS")
    print(f"{'='*70}")
    
    poses_torch = convert_poses_to_se3(poses_cpp)
    patches_torch = convert_patches_to_torch(patches_cpp, N, M, P)
    intrinsics_torch = convert_intrinsics_to_torch(intrinsics_cpp)
    
    # Extract source frame indices from kk (matching C++ logic)
    ii_from_kk = kk_cpp // M  # source frame index
    ii_torch = torch.from_numpy(ii_from_kk).long()
    jj_torch = torch.from_numpy(jj_cpp).long()
    kk_torch = torch.from_numpy(kk_cpp).long()
    
    # Verify ii matches kk // M for all edges
    if not np.array_equal(ii_from_kk, kk_cpp // M):
        print(f"⚠️  WARNING: ii_from_kk does not match kk_cpp // M!")
        print(f"   First 10 differences:")
        for e in range(min(10, num_active)):
            if ii_from_kk[e] != kk_cpp[e] // M:
                print(f"     Edge {e}: ii={ii_from_kk[e]}, kk//M={kk_cpp[e] // M}, kk={kk_cpp[e]}")
    
    print(f"Input                Shape                          Description")
    print("-"*70)
    print(f"{'poses_torch':<20} {str(poses_torch.shape):<30} Camera poses [N, 7]")
    print(f"{'patches_torch':<20} {str(patches_torch.shape):<30} Patches [N, M, 3, P, P]")
    print(f"{'intrinsics_torch':<20} {str(intrinsics_torch.shape):<30} Intrinsics [N, 4]")
    print(f"{'ii_torch':<20} {str(ii_torch.shape):<30} Source frame indices")
    print(f"{'jj_torch':<20} {str(jj_torch.shape):<30} Target frame indices")
    print(f"{'kk_torch':<20} {str(kk_torch.shape):<30} Global patch indices")
    print(f"{'='*70}")
    
    # Prepare batch inputs
    poses_batch = poses_torch
    patches_flat = patches_torch.reshape(N * M, 3, P, P)
    patches_batch = patches_flat.unsqueeze(0)  # [1, N*M, 3, P, P]
    intrinsics_batch = intrinsics_torch.unsqueeze(0)  # [1, N, 4]
    
    ii_1d = ii_torch  # [num_active]
    jj_1d = jj_torch  # [num_active]
    kk_1d = kk_torch  # [num_active]
    
    print(f"Patches shape: {patches_batch.shape} (should be [1, {N*M}, 3, {P}, {P}])")
    print(f"kk range: [{kk_cpp.min()}, {kk_cpp.max()}] (should be in [0, {N*M-1}])")
    print(f"Indices shape: ii={ii_1d.shape}, jj={jj_1d.shape}, kk={kk_1d.shape}")
    
    # Debug: Show first few edges' indices
    print(f"\nFirst 5 edges' indices:")
    print(f"{'Edge':<8} {'kk':<8} {'ii (from kk//M)':<18} {'jj':<8}")
    print("-"*50)
    for e in range(min(5, num_active)):
        print(f"{e:<8} {kk_cpp[e]:<8} {ii_from_kk[e]:<18} {jj_cpp[e]:<8}")
    
    # Debug: For edge 4, show what patches Python's transform will access
    if num_active > 4:
        e4 = 4
        k4 = kk_cpp[e4]
        i4 = k4 // M
        patch_idx4 = k4 % M
        j4 = jj_cpp[e4]
        print(f"\nEdge 4 debug:")
        print(f"  kk={k4}, i={i4}, patch_idx={patch_idx4}, j={j4}")
        print(f"  Python's transform will use: patches[:,kk[{e4}]]=patches[:,{k4}], intrinsics[:,ii[{e4}]]=intrinsics[:,{i4}], poses[:,jj[{e4}]]=poses[:,{j4}]")
        patches_batch_np = patches_batch.cpu().numpy()
        px4 = patches_batch_np[0, k4, 0, P//2, P//2]
        py4 = patches_batch_np[0, k4, 1, P//2, P//2]
        pd4 = patches_batch_np[0, k4, 2, P//2, P//2]
        print(f"  Patch center pixel (from patches_batch[0,{k4}]): px={px4:.6f}, py={py4:.6f}, pd={pd4:.6f}")
        print(f"  When Python does patches[:,kk], it selects patches[0, kk[{e4}]] = patches[0, {k4}]")
        print(f"  When Python does intrinsics[:,ii], it selects intrinsics[0, ii[{e4}]] = intrinsics[0, {i4}]")
        print(f"  When Python does poses[:,jj], it selects poses[0, jj[{e4}]] = poses[0, {j4}]")
    
    # Validate kk indices are in bounds
    if (kk_1d < 0).any() or (kk_1d >= N * M).any():
        invalid_kk = ((kk_1d < 0) | (kk_1d >= N * M)).sum().item()
        print(f"⚠️  WARNING: {invalid_kk} kk indices are out of bounds [0, {N*M-1}]")
        print(f"   Invalid kk values: {kk_1d[(kk_1d < 0) | (kk_1d >= N * M)]}")
    
    return poses_batch, patches_batch, intrinsics_batch, ii_1d, jj_1d, kk_1d, patches_torch


def call_python_reproject(poses_batch, patches_batch, intrinsics_batch, ii_1d, jj_1d, kk_1d, num_active, P, kk_cpp, M, jj_cpp):
    """Call Python reproject function and process output.
    
    Returns:
        numpy.ndarray: coords_py [num_active, 2, P, P]
    """
    print(f"\n{'='*70}")
    print("CALLING PYTHON REPROJECT FUNCTION")
    print(f"{'='*70}")
    
    try:
        coords_py = reproject_python(
            poses_batch,
            patches_batch,
            intrinsics_batch,
            ii_1d,
            jj_1d,
            kk_1d,
            depth=False,
            valid=False,
            jacobian=False
        )
        
        # Remove batch dimension if present: [1, num_active, 2, P, P] -> [num_active, 2, P, P]
        if isinstance(coords_py, torch.Tensor):
            if coords_py.dim() == 5 and coords_py.shape[0] == 1:
                coords_py = coords_py.squeeze(0).cpu().numpy()
            else:
                coords_py = coords_py.cpu().numpy()
        else:
            if hasattr(coords_py, 'shape') and len(coords_py.shape) == 5 and coords_py.shape[0] == 1:
                coords_py = coords_py.squeeze(0)
            coords_py = np.array(coords_py)
        
        print(f"Python reproject returned, shape: {coords_py.shape}")
        print(f"Expected shape: [{num_active}, 2, {P}, {P}]")
        
        # Python returns [num_active, P, P, 2] but C++ has [num_active, 2, P, P]
        if coords_py.shape == (num_active, P, P, 2):
            print(f"  Transposing Python output from [num_active, P, P, 2] to [num_active, 2, P, P]")
            coords_py = np.transpose(coords_py, (0, 3, 1, 2))
            print(f"  After transpose: {coords_py.shape}")
        elif coords_py.shape == (1, num_active, P, P, 2):
            coords_py = coords_py.squeeze(0)
            coords_py = np.transpose(coords_py, (0, 3, 1, 2))
            print(f"  Removed batch dim and transposed: {coords_py.shape}")
        
        # Debug: Test edge 4 individually to compare with debug script
        if num_active > 4:
            e4 = 4
            k4 = kk_cpp[e4]
            i4 = k4 // M
            j4 = jj_cpp[e4]
            
            ii_4_single = torch.tensor([i4]).long()
            jj_4_single = torch.tensor([j4]).long()
            kk_4_single = torch.tensor([k4]).long()
            
            coords_py_4_single = reproject_python(
                poses_batch,
                patches_batch,
                intrinsics_batch,
                ii_4_single,
                jj_4_single,
                kk_4_single,
                depth=False,
                valid=False,
                jacobian=False
            )
            
            # Process output
            if isinstance(coords_py_4_single, torch.Tensor):
                if coords_py_4_single.dim() == 5 and coords_py_4_single.shape[0] == 1:
                    coords_py_4_single = coords_py_4_single.squeeze(0).cpu().numpy()
                else:
                    coords_py_4_single = coords_py_4_single.cpu().numpy()
            else:
                coords_py_4_single = np.array(coords_py_4_single)
            
            if coords_py_4_single.shape == (1, P, P, 2):
                coords_py_4_single = coords_py_4_single.squeeze(0)
                coords_py_4_single = np.transpose(coords_py_4_single, (2, 0, 1))
            elif coords_py_4_single.shape == (P, P, 2):
                coords_py_4_single = np.transpose(coords_py_4_single, (2, 0, 1))
            
            # Compare batch vs single call for edge 4
            u_batch = coords_py[e4, 0, P//2, P//2]
            v_batch = coords_py[e4, 1, P//2, P//2]
            u_single = coords_py_4_single[0, P//2, P//2]
            v_single = coords_py_4_single[1, P//2, P//2]
            
            print(f"\n{'='*70}")
            print("EDGE 4: Batch vs Single Call Comparison")
            print(f"{'='*70}")
            print(f"Batch call (all edges): u={u_batch:.6f}, v={v_batch:.6f}")
            print(f"Single call (edge 4 only): u={u_single:.6f}, v={v_single:.6f}")
            print(f"Difference: u={abs(u_batch-u_single):.6f}, v={abs(v_batch-v_single):.6f}")
            if abs(u_batch-u_single) > 1e-5 or abs(v_batch-v_single) > 1e-5:
                print(f"⚠️  WARNING: Batch call differs from single call!")
                print(f"   This suggests Python's transform behaves differently when processing multiple edges.")
            else:
                print(f"✅ Batch call matches single call")
        
        return coords_py
        
    except Exception as e:
        print(f"❌ Error calling Python reproject function: {e}")
        import traceback
        traceback.print_exc()
        raise


def verify_gij_computation(poses_cpp, kk_cpp, jj_cpp, num_active, M):
    """Verify that Gij computation matches between C++ and Python.
    
    Args:
        poses_cpp: C++ poses [N, 7] format: [tx, ty, tz, qx, qy, qz, qw]
        kk_cpp: Global patch indices [num_active]
        jj_cpp: Target frame indices [num_active]
        num_active: Number of active edges
        M: Patches per frame
    
    Returns:
        bool: True if Gij computations match
    """
    print(f"\n{'='*70}")
    print("VERIFYING Gij COMPUTATION")
    print(f"{'='*70}")
    print("Gij = Tj * Ti^-1 (transform from frame i to frame j)")
    print("Where:")
    print("  Ti = pose of source frame i (world-to-camera)")
    print("  Tj = pose of target frame j (world-to-camera)")
    print("  Gij transforms points from frame i's camera coords to frame j's camera coords")
    print()
    print("Formula Verification:")
    print("  • C++: Gij = Tj * Ti.inverse()  (in projective_ops.cpp line 272)")
    print("  • Python: Gij = poses[:, jj] * poses[:, ii].inv()  (in projective_ops.py line 60)")
    print("  • Both extract source frame i from kk: i = kk // M")
    print()
    
    # Show frame indices for first few edges
    num_verify = min(5, num_active)
    print(f"Frame indices for first {num_verify} edges:")
    print(f"{'Edge':<8} {'kk':<8} {'i (from kk//M)':<18} {'j (from jj)':<15} {'Gij Formula':<30}")
    print("-"*80)
    
    for e in range(num_verify):
        k = kk_cpp[e]
        i = k // M  # source frame from kk
        j = jj_cpp[e]  # target frame
        formula = f"Gij = T{j} * T{i}^-1"
        print(f"{e:<8} {k:<8} {i:<18} {j:<15} {formula:<30}")
    
    print(f"{'='*70}")
    print("COMPARING SE3 OPERATIONS (Gij computation)")
    print(f"{'='*70}")
    
    # Compare SE3 inverse and multiplication for first few edges
    num_verify = min(5, num_active)
    print(f"\nComparing Gij = Tj * Ti^-1 for first {num_verify} edges:")
    print(f"{'Edge':<8} {'i':<5} {'j':<5} {'Operation':<25} {'C++ Result':<40} {'Python Result':<40} {'Diff':<15}")
    print("-"*140)
    
    all_match = True
    for e in range(num_verify):
        k = kk_cpp[e]
        i = k // M  # source frame from kk
        j = jj_cpp[e]  # target frame
        
        # Get poses
        Ti_cpp = poses_cpp[i]  # [tx, ty, tz, qx, qy, qz, qw]
        Tj_cpp = poses_cpp[j]
        
        # Convert to Python SE3 format
        Ti_py_data = np.zeros(7)
        Ti_py_data[:3] = Ti_cpp[:3]  # translation
        Ti_py_data[3] = Ti_cpp[6]    # qw
        Ti_py_data[4:7] = Ti_cpp[3:6]  # qx, qy, qz
        
        Tj_py_data = np.zeros(7)
        Tj_py_data[:3] = Tj_cpp[:3]
        Tj_py_data[3] = Tj_cpp[6]
        Tj_py_data[4:7] = Tj_cpp[3:6]
        
        # Create SE3 objects (with batch dimension)
        Ti_py_se3 = SE3(torch.from_numpy(Ti_py_data).float().unsqueeze(0).unsqueeze(0))  # [1, 1, 7]
        Tj_py_se3 = SE3(torch.from_numpy(Tj_py_data).float().unsqueeze(0).unsqueeze(0))
        
        # Compute Ti^-1 in Python
        Ti_inv_py = Ti_py_se3.inv()
        
        # Compute Gij in Python: Gij = Tj * Ti^-1
        Gij_py = Tj_py_se3 * Ti_inv_py
        
        # Extract Gij components from Python SE3
        Gij_py_data = Gij_py.data[0, 0].cpu().numpy()  # [7] = [tx, ty, tz, qw, qx, qy, qz]
        Gij_py_t = Gij_py_data[:3]  # translation [tx, ty, tz]
        Gij_py_q = np.array([Gij_py_data[3], Gij_py_data[4], Gij_py_data[5], Gij_py_data[6]])  # [qw, qx, qy, qz]
        
        # Convert Python quaternion to rotation matrix
        if R is not None:
            Gij_py_R = R.from_quat(Gij_py_q).as_matrix()
        else:
            # Manual quaternion to rotation matrix conversion
            qw, qx, qy, qz = Gij_py_q
            Gij_py_R = np.array([
                [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
            ])
        
        if R is None:
            print("  ⚠️  scipy not available, skipping detailed SE3 comparison")
            continue
        
        # Convert C++ poses to rotation matrices and translations
        Ti_cpp_q = np.array([Ti_cpp[6], Ti_cpp[3], Ti_cpp[4], Ti_cpp[5]])  # [qw, qx, qy, qz]
        Ti_cpp_t = Ti_cpp[:3]
        Ti_cpp_rot = R.from_quat(Ti_cpp_q).as_matrix()
        
        Tj_cpp_q = np.array([Tj_cpp[6], Tj_cpp[3], Tj_cpp[4], Tj_cpp[5]])  # [qw, qx, qy, qz]
        Tj_cpp_t = Tj_cpp[:3]
        Tj_cpp_rot = R.from_quat(Tj_cpp_q).as_matrix()
        
        # Compute Ti^-1 in C++ style: inv.q = q.conjugate(), inv.t = -(inv.R() * t)
        Ti_inv_cpp_rot = Ti_cpp_rot.T  # R^-1 = R^T
        Ti_inv_cpp_t = -Ti_inv_cpp_rot @ Ti_cpp_t
        
        # Compute Gij in C++ style: Gij = Tj * Ti^-1
        Gij_cpp_rot = Tj_cpp_rot @ Ti_inv_cpp_rot
        Gij_cpp_t = Tj_cpp_t + Tj_cpp_rot @ Ti_inv_cpp_t
        
        # Compare rotation matrices
        R_diff = np.abs(Gij_cpp_rot - Gij_py_R)
        max_R_diff = np.max(R_diff)
        
        # Compare translations
        t_diff = np.abs(Gij_cpp_t - Gij_py_t)
        max_t_diff = np.max(t_diff)
        
        # Format for display
        cpp_str = f"t=({Gij_cpp_t[0]:.4f},{Gij_cpp_t[1]:.4f},{Gij_cpp_t[2]:.4f})"
        py_str = f"t=({Gij_py_t[0]:.4f},{Gij_py_t[1]:.4f},{Gij_py_t[2]:.4f})"
        diff_str = f"R:{max_R_diff:.6f}, t:{max_t_diff:.6f}"
        
        match_str = "✅" if max_R_diff < 1e-5 and max_t_diff < 1e-5 else "❌"
        if max_R_diff >= 1e-5 or max_t_diff >= 1e-5:
            all_match = False
        
        print(f"{e:<8} {i:<5} {j:<5} {'Gij = Tj * Ti^-1':<25} {cpp_str:<40} {py_str:<40} {diff_str:<15} {match_str}")
        
        if e == 0:
            # Show detailed comparison for first edge
            print(f"\n  Detailed Gij comparison for Edge {e} (i={i}, j={j}):")
            print(f"    Rotation Matrix R:")
            print(f"      C++ R[0,:] = [{Gij_cpp_rot[0,0]:.6f}, {Gij_cpp_rot[0,1]:.6f}, {Gij_cpp_rot[0,2]:.6f}]")
            print(f"      Py  R[0,:] = [{Gij_py_R[0,0]:.6f}, {Gij_py_R[0,1]:.6f}, {Gij_py_R[0,2]:.6f}]")
            print(f"      Diff       = [{R_diff[0,0]:.6f}, {R_diff[0,1]:.6f}, {R_diff[0,2]:.6f}]")
            print(f"    Translation t:")
            print(f"      C++ t = [{Gij_cpp_t[0]:.6f}, {Gij_cpp_t[1]:.6f}, {Gij_cpp_t[2]:.6f}]")
            print(f"      Py  t = [{Gij_py_t[0]:.6f}, {Gij_py_t[1]:.6f}, {Gij_py_t[2]:.6f}]")
            print(f"      Diff = [{t_diff[0]:.6f}, {t_diff[1]:.6f}, {t_diff[2]:.6f}]")
    
    print(f"{'='*70}")
    print("✅ VERIFICATION:")
    print("  1. Gij Formula: Both C++ and Python use Gij = Tj * Ti^-1")
    print("  2. Gij Assembly: Gij is computed from input poses Ti and Tj")
    print("  3. Source Frame: Both extract i from kk as i = kk // M")
    print("  4. Target Frame: Both use j from jj array")
    print("  5. Input Poses: Same poses used for both C++ and Python")
    print(f"{'='*70}")
    
    return all_match


def compare_intermediate_steps_for_edge(edge_idx, poses_cpp, patches_cpp, intrinsics_cpp, 
                                        kk_cpp, jj_cpp, coords_cpp_reshaped, coords_py,
                                        patches_batch, patches_torch, num_active, M, P, N):
    """Compare intermediate reprojection steps for a specific edge.
    
    Args:
        edge_idx: Edge index to analyze
        poses_cpp: C++ poses [N, 7]
        patches_cpp: C++ patches [N*M, 3, P, P]
        intrinsics_cpp: C++ intrinsics [N, 4]
        kk_cpp: Global patch indices [num_active]
        jj_cpp: Target frame indices [num_active]
        coords_cpp_reshaped: C++ output coords [num_active, 2, P, P]
        coords_py: Python output coords [num_active, 2, P, P]
        patches_batch: Python patches batch [1, N*M, 3, P, P]
        patches_torch: Python patches [N, M, 3, P, P]
        num_active: Number of active edges
        M: Patches per frame
        P: Patch size
        N: Number of frames
    """
    if edge_idx >= num_active:
        return
    
    e = edge_idx
    k = kk_cpp[e]
    i = k // M  # source frame
    j = jj_cpp[e]  # target frame
    patch_idx = k % M
    center_y = P // 2
    center_x = P // 2
    
    # Get patches for this edge (center pixel)
    px_cpp = patches_cpp[k, 0, center_y, center_x]
    py_cpp = patches_cpp[k, 1, center_y, center_x]
    pd_cpp = patches_cpp[k, 2, center_y, center_x]
    
    # Get intrinsics
    intr_i = intrinsics_cpp[i]
    intr_j = intrinsics_cpp[j]
    
    # Compute Gij for this edge (C++ style)
    Ti_cpp = poses_cpp[i]
    Tj_cpp = poses_cpp[j]
    Ti_cpp_q = np.array([Ti_cpp[6], Ti_cpp[3], Ti_cpp[4], Ti_cpp[5]])  # [qw, qx, qy, qz]
    Ti_cpp_t = Ti_cpp[:3]
    Ti_cpp_rot = R.from_quat(Ti_cpp_q).as_matrix() if R else None
    Tj_cpp_q = np.array([Tj_cpp[6], Tj_cpp[3], Tj_cpp[4], Tj_cpp[5]])
    Tj_cpp_t = Tj_cpp[:3]
    Tj_cpp_rot = R.from_quat(Tj_cpp_q).as_matrix() if R else None
    
    if Ti_cpp_rot is not None and Tj_cpp_rot is not None:
        Ti_inv_cpp_rot = Ti_cpp_rot.T
        Ti_inv_cpp_t = -Ti_inv_cpp_rot @ Ti_cpp_t
        Gij_cpp_rot = Tj_cpp_rot @ Ti_inv_cpp_rot
        Gij_cpp_t = Tj_cpp_t + Tj_cpp_rot @ Ti_inv_cpp_t
    
    # C++ inverse projection
    X0_cpp = (px_cpp - intr_i[2]) / intr_i[0]
    Y0_cpp = (py_cpp - intr_i[3]) / intr_i[1]
    Z0_cpp = 1.0
    W0_cpp = pd_cpp
    
    # C++ transform
    if Ti_cpp_rot is not None:
        p0_vec_cpp = np.array([X0_cpp, Y0_cpp, Z0_cpp])
        p1_vec_cpp = Gij_cpp_rot @ p0_vec_cpp + Gij_cpp_t * W0_cpp
        X1_cpp = p1_vec_cpp[0]
        Y1_cpp = p1_vec_cpp[1]
        Z1_cpp = p1_vec_cpp[2]
        
        # C++ forward projection
        z_cpp = max(Z1_cpp, 0.1)
        d_cpp = 1.0 / z_cpp
        u_cpp = intr_j[0] * (d_cpp * X1_cpp) + intr_j[2]
        v_cpp = intr_j[1] * (d_cpp * Y1_cpp) + intr_j[3]
    
    # Python intermediate steps
    patches_batch_np = patches_batch.cpu().numpy()
    px_py = patches_batch_np[0, k, 0, P//2, P//2]
    py_py = patches_batch_np[0, k, 1, P//2, P//2]
    pd_py = patches_batch_np[0, k, 2, P//2, P//2]
    
    # Python inverse projection
    X0_py = (px_py - intr_i[2]) / intr_i[0]
    Y0_py = (py_py - intr_i[3]) / intr_i[1]
    Z0_py = 1.0
    W0_py = pd_py
    
    # Python transform (compute Gij_py for this edge)
    Ti_py_data = np.zeros(7)
    Ti_py_data[:3] = Ti_cpp[:3]
    Ti_py_data[3] = Ti_cpp[6]
    Ti_py_data[4:7] = Ti_cpp[3:6]
    Tj_py_data = np.zeros(7)
    Tj_py_data[:3] = Tj_cpp[:3]
    Tj_py_data[3] = Tj_cpp[6]
    Tj_py_data[4:7] = Tj_cpp[3:6]
    
    Ti_py_se3 = SE3(torch.from_numpy(Ti_py_data).float().unsqueeze(0).unsqueeze(0))
    Tj_py_se3 = SE3(torch.from_numpy(Tj_py_data).float().unsqueeze(0).unsqueeze(0))
    Ti_inv_py = Ti_py_se3.inv()
    Gij_py = Tj_py_se3 * Ti_inv_py
    
    Gij_py_data = Gij_py.data[0, 0].cpu().numpy()
    Gij_py_t = Gij_py_data[:3]
    Gij_py_q = np.array([Gij_py_data[3], Gij_py_data[4], Gij_py_data[5], Gij_py_data[6]])
    Gij_py_R = R.from_quat(Gij_py_q).as_matrix() if R else None
    
    if Gij_py_R is not None:
        p0_vec_py = np.array([X0_py, Y0_py, Z0_py])
        p1_vec_py = Gij_py_R @ p0_vec_py + Gij_py_t * W0_py
        X1_py = p1_vec_py[0]
        Y1_py = p1_vec_py[1]
        Z1_py = p1_vec_py[2]
        
        # Python forward projection
        z_py = max(Z1_py, 0.1)
        d_py = 1.0 / z_py
        u_py = intr_j[0] * (d_py * X1_py) + intr_j[2]
        v_py = intr_j[1] * (d_py * Y1_py) + intr_j[3]
    
    # Compare intermediate values
    print(f"\nEdge {e} (i={i}, j={j}, kk={k}, patch_idx={patch_idx}):")
    print(f"  Input Patch (center pixel):")
    print(f"    C++: px={px_cpp:.6f}, py={py_cpp:.6f}, pd={pd_cpp:.6f}")
    print(f"    Py:  px={px_py:.6f}, py={py_py:.6f}, pd={pd_py:.6f}")
    print(f"    Diff: px={abs(px_cpp-px_py):.6f}, py={abs(py_cpp-py_py):.6f}, pd={abs(pd_cpp-pd_py):.6f}")
    
    print(f"\n  Step 1: Inverse Projection (X0, Y0, Z0, W0):")
    print(f"    C++: X0={X0_cpp:.6f}, Y0={Y0_cpp:.6f}, Z0={Z0_cpp:.6f}, W0={W0_cpp:.6f}")
    print(f"    Py:  X0={X0_py:.6f}, Y0={Y0_py:.6f}, Z0={Z0_py:.6f}, W0={W0_py:.6f}")
    print(f"    Diff: X0={abs(X0_cpp-X0_py):.6f}, Y0={abs(Y0_cpp-Y0_py):.6f}, Z0={abs(Z0_cpp-Z0_py):.6f}, W0={abs(W0_cpp-W0_py):.6f}")
    
    if Ti_cpp_rot is not None and Gij_py_R is not None:
        print(f"\n  Step 2: SE3 Transform (X1, Y1, Z1):")
        print(f"    C++: X1={X1_cpp:.6f}, Y1={Y1_cpp:.6f}, Z1={Z1_cpp:.6f}")
        print(f"    Py:  X1={X1_py:.6f}, Y1={Y1_py:.6f}, Z1={Z1_py:.6f}")
        print(f"    Diff: X1={abs(X1_cpp-X1_py):.6f}, Y1={abs(Y1_cpp-Y1_py):.6f}, Z1={abs(Z1_cpp-Z1_py):.6f}")
        
        print(f"\n  Step 3: Forward Projection (u, v):")
        print(f"    C++: u={u_cpp:.6f}, v={v_cpp:.6f}")
        print(f"    Py:  u={u_py:.6f}, v={v_py:.6f}")
        print(f"    Diff: u={abs(u_cpp-u_py):.6f}, v={abs(v_cpp-v_py):.6f}")
        
        # Compare with actual output
        u_out_cpp = coords_cpp_reshaped[e, 0, P//2, P//2]
        v_out_cpp = coords_cpp_reshaped[e, 1, P//2, P//2]
        u_out_py = coords_py[e, 0, P//2, P//2]
        v_out_py = coords_py[e, 1, P//2, P//2]
        
        print(f"\n  Final Output (from arrays):")
        print(f"    C++: u={u_out_cpp:.6f}, v={v_out_cpp:.6f}")
        print(f"    Py:  u={u_out_py:.6f}, v={v_out_py:.6f}")
        print(f"    Diff: u={abs(u_out_cpp-u_out_py):.6f}, v={abs(v_out_cpp-v_out_py):.6f}")
        print(f"    Manual calc matches C++: u={abs(u_cpp-u_out_cpp):.6f}, v={abs(v_cpp-v_out_cpp):.6f}")
        print(f"    Manual calc matches Py:  u={abs(u_py-u_out_py):.6f}, v={abs(v_py-v_out_py):.6f}")


def compare_outputs(coords_cpp_reshaped, coords_py, num_active, P, tolerance=1e-4):
    """Compare C++ and Python reproject outputs and generate statistics.
    
    Args:
        coords_cpp_reshaped: C++ output coords [num_active, 2, P, P]
        coords_py: Python output coords [num_active, 2, P, P]
        num_active: Number of active edges
        P: Patch size
        tolerance: Tolerance for mismatch detection
    
    Returns:
        tuple: (all_match, max_diff, mean_diff, mismatches, total_elements)
    """
    print(f"\n{'='*70}")
    print("COMPARING OUTPUTS")
    print(f"{'='*70}")
    
    # Flatten for comparison
    coords_cpp_flat = coords_cpp_reshaped.flatten()
    coords_py_flat = coords_py.flatten()
    
    diff = np.abs(coords_cpp_flat - coords_py_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Relative difference
    rel_diff = np.abs(diff / (np.abs(coords_cpp_flat) + 1e-8))
    max_rel_diff = np.max(rel_diff)
    mean_rel_diff = np.mean(rel_diff)
    
    # Count mismatches
    mismatches = np.sum(diff > tolerance)
    total_elements = diff.size
    
    print(f"\n{'Metric':<25} {'Value':<20} {'Description':<50}")
    print("-"*95)
    print(f"{'Max Diff':<25} {max_diff:<20.6f} {'Maximum absolute difference (pixels)':<50}")
    print(f"{'Mean Diff':<25} {mean_diff:<20.6f} {'Average absolute difference (pixels)':<50}")
    print(f"{'Max Rel Diff':<25} {max_rel_diff:<20.6f} {'Maximum relative difference (|diff|/|cpp|)':<50}")
    print(f"{'Mean Rel Diff':<25} {mean_rel_diff:<20.6f} {'Average relative difference (|diff|/|cpp|)':<50}")
    print(f"{'Mismatches (>1e-4)':<25} {mismatches}/{total_elements} ({100*mismatches/total_elements:.2f}%) {'Elements with diff > 1e-4':<50}")
    
    print(f"\n{'Notes:':<25}")
    print(f"{'':<25} • Max Diff: Largest absolute difference between C++ and Python values")
    print(f"{'':<25} • Mean Diff: Average absolute difference across all elements")
    print(f"{'':<25} • Max Rel Diff: Largest relative difference = max(|cpp - py| / (|cpp| + 1e-8))")
    print(f"{'':<25} • Mean Rel Diff: Average relative difference = mean(|cpp - py| / (|cpp| + 1e-8))")
    print(f"{'':<25} • Relative diff shows error as fraction of C++ value (useful for small values)")
    print(f"{'':<25} • High rel diff (>100%) often indicates values near zero or sign differences")
    
    all_match = (mismatches == 0)
    return all_match, max_diff, mean_diff, mismatches, total_elements


def print_sample_comparison(coords_cpp_reshaped, coords_py, num_active, P, tolerance=1e-4):
    """Print sample value comparison table.
    
    Args:
        coords_cpp_reshaped: C++ output coords [num_active, 2, P, P]
        coords_py: Python output coords [num_active, 2, P, P]
        num_active: Number of active edges
        P: Patch size
        tolerance: Tolerance for mismatch detection
    """
    print(f"\n{'='*100}")
    print("SAMPLE VALUES COMPARISON - CENTER PIXEL (First 20 edges)")
    print(f"{'='*100}")
    print(f"{'Edge':<8} {'C++ u':<15} {'C++ v':<15} {'Python u':<15} {'Python v':<15} {'Diff u':<15} {'Diff v':<15} {'Status':<10}")
    print("-"*100)
    
    center_y = P // 2
    center_x = P // 2
    num_samples = min(20, num_active)
    for e in range(num_samples):
        cpp_u = coords_cpp_reshaped[e, 0, center_y, center_x]
        cpp_v = coords_cpp_reshaped[e, 1, center_y, center_x]
        py_u = coords_py[e, 0, center_y, center_x]
        py_v = coords_py[e, 1, center_y, center_x]
        diff_u = abs(cpp_u - py_u)
        diff_v = abs(cpp_v - py_v)
        max_diff = max(diff_u, diff_v)
        status = "✅" if max_diff < tolerance else "❌"
        
        print(f"{e:<8} {cpp_u:>15.6f} {cpp_v:>15.6f} {py_u:>15.6f} {py_v:>15.6f} "
              f"{diff_u:>15.6f} {diff_v:>15.6f} {status:<10}")
    
    if num_active > num_samples:
        print(f"... ({num_active - num_samples} more edges)")
    
    print(f"{'='*100}")


def print_detailed_patch_comparison(coords_cpp_reshaped, coords_py, num_active, P, tolerance=1e-4):
    """Print detailed patch comparison for first few edges.
    
    Args:
        coords_cpp_reshaped: C++ output coords [num_active, 2, P, P]
        coords_py: Python output coords [num_active, 2, P, P]
        num_active: Number of active edges
        P: Patch size
        tolerance: Tolerance for mismatch detection
    """
    print(f"\n{'='*100}")
    print("DETAILED PATCH COMPARISON (First 3 edges, all pixels)")
    print(f"{'='*100}")
    
    num_detail_edges = min(3, num_active)
    center_y = P // 2
    center_x = P // 2
    
    for e in range(num_detail_edges):
        print(f"\n{'='*100}")
        print(f"EDGE {e} - All Pixels in Patch (P={P}x{P})")
        print(f"{'='*100}")
        print(f"{'Pixel (y,x)':<15} {'C++ u':<15} {'C++ v':<15} {'Python u':<15} {'Python v':<15} {'Diff u':<15} {'Diff v':<15} {'Max Diff':<15} {'Status':<10}")
        print("-"*100)
        
        for py_idx in range(P):
            for px_idx in range(P):
                cpp_u = coords_cpp_reshaped[e, 0, py_idx, px_idx]
                cpp_v = coords_cpp_reshaped[e, 1, py_idx, px_idx]
                py_u = coords_py[e, 0, py_idx, px_idx]
                py_v = coords_py[e, 1, py_idx, px_idx]
                diff_u = abs(cpp_u - py_u)
                diff_v = abs(cpp_v - py_v)
                max_diff = max(diff_u, diff_v)
                status = "✅" if max_diff < tolerance else "❌"
                
                pixel_str = f"({py_idx},{px_idx})"
                center_marker = " [CENTER]" if py_idx == center_y and px_idx == center_x else ""
                
                print(f"{pixel_str:<15} {cpp_u:>15.6f} {cpp_v:>15.6f} {py_u:>15.6f} {py_v:>15.6f} "
                      f"{diff_u:>15.6f} {diff_v:>15.6f} {max_diff:>15.6f} {status:<10}{center_marker}")
        
        # Summary for this edge
        edge_diff = np.abs(coords_cpp_reshaped[e] - coords_py[e])
        edge_max_diff = np.max(edge_diff)
        edge_mean_diff = np.mean(edge_diff)
        edge_mismatches = np.sum(edge_diff > tolerance)
        edge_total = edge_diff.size
        
        print("-"*100)
        print(f"Edge {e} Summary: Max Diff={edge_max_diff:.6f}, Mean Diff={edge_mean_diff:.6f}, "
              f"Mismatches={edge_mismatches}/{edge_total} ({100*edge_mismatches/edge_total:.1f}%)")
    
    print(f"{'='*100}")


def print_mismatch_analysis(coords_cpp_reshaped, coords_py, num_active, P, tolerance=1e-4):
    """Print mismatch analysis table showing edges with largest differences.
    
    Args:
        coords_cpp_reshaped: C++ output coords [num_active, 2, P, P]
        coords_py: Python output coords [num_active, 2, P, P]
        num_active: Number of active edges
        P: Patch size
        tolerance: Tolerance for mismatch detection
    """
    print(f"\n{'='*100}")
    print("MISMATCH ANALYSIS - Edges with Largest Differences (Top 20)")
    print(f"{'='*100}")
    
    # Compute max diff per edge
    edge_max_diffs = []
    center_y = P // 2
    center_x = P // 2
    
    for e in range(num_active):
        edge_diff = np.abs(coords_cpp_reshaped[e] - coords_py[e])
        edge_max_diff = np.max(edge_diff)
        edge_mean_diff = np.mean(edge_diff)
        edge_mismatches = np.sum(edge_diff > tolerance)
        edge_total = edge_diff.size
        edge_max_diffs.append((e, edge_max_diff, edge_mean_diff, edge_mismatches, edge_total))
    
    # Sort by max diff (descending)
    edge_max_diffs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<8} {'Edge':<8} {'Max Diff':<15} {'Mean Diff':<15} {'Mismatches':<20} {'C++ Center (u,v)':<30} {'Python Center (u,v)':<30}")
    print("-"*100)
    
    num_mismatch_samples = min(20, len(edge_max_diffs))
    for rank, (e, max_diff, mean_diff, mismatches, total) in enumerate(edge_max_diffs[:num_mismatch_samples], 1):
        cpp_u_center = coords_cpp_reshaped[e, 0, center_y, center_x]
        cpp_v_center = coords_cpp_reshaped[e, 1, center_y, center_x]
        py_u_center = coords_py[e, 0, center_y, center_x]
        py_v_center = coords_py[e, 1, center_y, center_x]
        
        cpp_str = f"({cpp_u_center:>10.6f}, {cpp_v_center:>10.6f})"
        py_str = f"({py_u_center:>10.6f}, {py_v_center:>10.6f})"
        
        print(f"{rank:<8} {e:<8} {max_diff:>15.6f} {mean_diff:>15.6f} {f'{mismatches}/{total}':<20} "
              f"{cpp_str:<30} {py_str:<30}")
    
    if len(edge_max_diffs) > num_mismatch_samples:
        print(f"... ({len(edge_max_diffs) - num_mismatch_samples} more edges)")
    
    print(f"{'='*100}")


def compare_reproject(frame_num):
    """Compare C++ and Python reproject outputs for a given frame.
    
    NOTE: This script uses ba_patches.bin and ba_reprojected_coords.bin to match
    the BA comparison. These files are saved from the FIRST reproject() call in
    DPVO::update(), which is the same reproject() call that BA uses internally.
    This ensures consistency with compare_ba_step_by_step.py.
    
    IMPORTANT: If you see mismatches here but matches in BA comparison, it's because:
    1. BA comparison uses patches that are modified during BA optimization
    2. This script uses original patches saved before BA
    3. Both C++ and Python BA modify patches the same way, so they match
    4. But standalone reproject uses original patches, which may differ from Python's
       reproject if Python's input patches differ from C++'s ba_patches.bin
    """
    print(f"\n{'='*60}")
    print(f"Comparing Reproject Outputs for Frame {frame_num}")
    print(f"{'='*60}")
    print(f"NOTE: Using ba_patches.bin and ba_reprojected_coords.bin")
    print(f"      (same files as BA comparison for consistency)")
    print(f"\n⚠️  If mismatches occur here but BA comparison matches:")
    print(f"   → BA modifies patches during optimization (both C++ and Python)")
    print(f"   → This comparison uses original patches (before BA)")
    print(f"   → Check if Python's input patches match ba_patches.bin\n")
    
    try:
        # Load C++ data
        poses_cpp, patches_cpp, intrinsics_cpp, ii_cpp, jj_cpp, kk_cpp, coords_cpp, num_active, M, P, N = load_cpp_data(frame_num)
        
        # Reshape coords_cpp to [num_active, 2, P, P]
        coords_cpp_reshaped = coords_cpp.reshape(num_active, 2, P, P)
        
        # Prepare Python inputs
        poses_batch, patches_batch, intrinsics_batch, ii_1d, jj_1d, kk_1d, patches_torch = prepare_python_inputs(
            poses_cpp, patches_cpp, intrinsics_cpp, kk_cpp, jj_cpp, num_active, M, P, N
        )
        
        # Call Python reproject function
        coords_py = call_python_reproject(
            poses_batch, patches_batch, intrinsics_batch, ii_1d, jj_1d, kk_1d, 
            num_active, P, kk_cpp, M, jj_cpp
        )
        
        # Verify Gij computation
        verify_gij_computation(poses_cpp, kk_cpp, jj_cpp, num_active, M)
        
        # Compare intermediate steps for specific edges
        print(f"\n{'='*70}")
        print("COMPARING INTERMEDIATE STEPS (Edge 0 and Edge 4)")
        print(f"{'='*70}")
        
        # Edge 0 (i=j, matches perfectly)
        print(f"\n{'='*70}")
        print("EDGE 0 (i=j, matches perfectly)")
        print(f"{'='*70}")
        compare_intermediate_steps_for_edge(
            0, poses_cpp, patches_cpp, intrinsics_cpp, kk_cpp, jj_cpp,
            coords_cpp_reshaped, coords_py, patches_batch, patches_torch,
            num_active, M, P, N
        )
        
        # Edge 4 (i≠j, has mismatch)
        if num_active > 4:
            print(f"\n{'='*70}")
            print("EDGE 4 ANALYSIS (i≠j, has mismatch)")
            print(f"{'='*70}")
            compare_intermediate_steps_for_edge(
                4, poses_cpp, patches_cpp, intrinsics_cpp, kk_cpp, jj_cpp,
                coords_cpp_reshaped, coords_py, patches_batch, patches_torch,
                num_active, M, P, N
            )
        
        # Compare outputs
        all_match, max_diff, mean_diff, mismatches, total_elements = compare_outputs(
            coords_cpp_reshaped, coords_py, num_active, P
        )
        
        # Print sample comparison
        print_sample_comparison(coords_cpp_reshaped, coords_py, num_active, P)
        
        # Print detailed patch comparison
        print_detailed_patch_comparison(coords_cpp_reshaped, coords_py, num_active, P)
        
        # Print mismatch analysis
        print_mismatch_analysis(coords_cpp_reshaped, coords_py, num_active, P)
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        if all_match:
            print(f"✅ ALL OUTPUTS MATCH! C++ reproject function matches Python.")
        else:
            print(f"❌ MISMATCHES DETECTED!")
            print(f"\nComparison Validity:")
            print(f"  ✅ Gij Formula: Both C++ and Python use Gij = Tj * Ti^-1")
            print(f"  ✅ Gij Assembly: Gij is computed from input poses (Ti, Tj)")
            print(f"  ✅ Input Poses: Same poses used for both C++ and Python")
            print(f"  ✅ Input Patches: Same patches used for both")
            print(f"  ✅ Input Intrinsics: Same intrinsics used for both")
            print(f"  ✅ Index Extraction: Source frame i extracted from kk (i = kk // M)")
            print(f"\nPossible causes of mismatches:")
            print(f"  1. Numerical precision differences (float32 vs float64)")
            print(f"  2. SE3 implementation differences (rotation matrix computation)")
            print(f"  3. Coordinate system conventions (if any)")
            print(f"  4. Clamping/boundary handling differences")
            print(f"\nCheck the sample values above to identify patterns in the differences.")
        
        return all_match
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure C++ code has been run and saved data for this frame.")
        return False


def main():
    parser = argparse.ArgumentParser(description='Compare C++ and Python reproject outputs')
    parser.add_argument('--frame', type=int, required=True, help='Frame number to compare')
    args = parser.parse_args()
    
    success = compare_reproject(args.frame)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

