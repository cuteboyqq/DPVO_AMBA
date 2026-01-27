#!/usr/bin/env python3
"""
Compare C++ and Python BA function outputs using the same update model outputs.

This script:
1. Loads update model outputs (d_out, w_out) from compare_update_onnx_outputs.py
2. Loads BA inputs (poses, patches, intrinsics, indices) - needs to be saved separately
3. Computes targets = reprojected_coords + d_out
4. Runs Python BA function
5. Compares with C++ BA outputs (if available)

Workflow:
- First run: Save BA inputs (poses, patches, intrinsics, ii, jj, kk) from C++ code
- Then run this script to compare BA outputs
"""

import numpy as np
import torch
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import os

# Add DPVO_onnx to path
sys.path.insert(0, '/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx')

from dpvo.ba import python_ba_wrapper
from dpvo.lietorch import SE3
from dpvo import projective_ops as pops

def load_binary_file(filename: str, dtype: np.dtype = np.float32) -> np.ndarray:
    """Load binary file as numpy array."""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=dtype)
    return data

def load_metadata() -> Optional[Dict[str, int]]:
    """Load test metadata from test_metadata.txt file (if available)."""
    if not os.path.exists('test_metadata.txt'):
        return None
    metadata = {}
    with open('test_metadata.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                metadata[key] = int(value)
    return metadata

def infer_metadata_from_files() -> Dict[str, int]:
    """Infer metadata from available data files."""
    metadata = {}
    
    # Try to infer from BA input files
    if os.path.exists('ba_poses.bin'):
        poses_data = load_binary_file('ba_poses.bin')
        if poses_data is not None:
            # Each pose is 7 floats (tx, ty, tz, qx, qy, qz, qw)
            N = poses_data.size // 7
            metadata['N'] = N
            print(f"  Inferred N (frames) from ba_poses.bin: {N}")
    
    if os.path.exists('ba_patches.bin'):
        patches_data = load_binary_file('ba_patches.bin')
        if patches_data is not None and 'N' in metadata:
            # Patches are [N*M, 3, P, P]
            # Total size = N * M * 3 * P * P
            # We need to infer M and P
            # Common values: P=3, M=4 or M=80
            total_patches = patches_data.size
            if 'N' in metadata:
                patches_per_frame = total_patches // (metadata['N'] * 3)
                # patches_per_frame = M * P * P
                # Try common P values
                for P_candidate in [3, 5, 7]:
                    M_candidate = patches_per_frame // (P_candidate * P_candidate)
                    if M_candidate * P_candidate * P_candidate == patches_per_frame:
                        metadata['P'] = P_candidate
                        metadata['M'] = M_candidate
                        print(f"  Inferred P (patch size) from ba_patches.bin: {P_candidate}")
                        print(f"  Inferred M (patches per frame) from ba_patches.bin: {M_candidate}")
                        break
    
    if os.path.exists('ba_ii.bin'):
        ii_data = load_binary_file('ba_ii.bin', dtype=np.int32)
        if ii_data is not None:
            num_active = ii_data.size
            metadata['num_active'] = num_active
            print(f"  Inferred num_active from ba_ii.bin: {num_active}")
    
    # Set defaults for missing values
    if 'MAX_EDGE' not in metadata:
        metadata['MAX_EDGE'] = metadata.get('num_active', 100)  # Use num_active as default
    if 'DIM' not in metadata:
        metadata['DIM'] = 384  # Standard DIM value
    if 'CORR_DIM' not in metadata:
        metadata['CORR_DIM'] = 882  # Standard CORR_DIM for P=3: 2*49*3*3 = 882
    
    return metadata

def numpy_to_se3(poses_np: np.ndarray) -> SE3:
    """Convert numpy array [7] (tx, ty, tz, qx, qy, qz, qw) to SE3."""
    t = poses_np[:3]
    q = poses_np[3:]
    # SE3 expects [tx, ty, tz, qw, qx, qy, qz] format
    return SE3(torch.tensor([t[0], t[1], t[2], q[3], q[0], q[1], q[2]], dtype=torch.float32))

def se3_to_numpy(se3: SE3) -> np.ndarray:
    """Convert SE3 to numpy array [7] (tx, ty, tz, qx, qy, qz, qw)."""
    t = se3.t.cpu().numpy()
    q = se3.q.cpu().numpy()
    # Convert from [qw, qx, qy, qz] to [tx, ty, tz, qx, qy, qz, qw]
    return np.array([t[0], t[1], t[2], q[1], q[2], q[3], q[0]])

def compare_poses(poses_cpp: np.ndarray, poses_py: np.ndarray, 
                  num_poses: int, atol: float = 1e-4) -> bool:
    """Compare C++ and Python BA output poses.
    
    Args:
        poses_cpp: C++ poses [num_poses, 7] (tx, ty, tz, qx, qy, qz, qw)
        poses_py: Python poses [num_poses, 7] (tx, ty, tz, qx, qy, qz, qw)
        num_poses: Number of poses to compare
        atol: Absolute tolerance for comparison
    
    Returns:
        True if poses match within tolerance, False otherwise
    """
    print(f"\n=== Comparing BA Output Poses ===")
    print(f"Number of poses: {num_poses}")
    
    if poses_cpp.shape[0] < num_poses or poses_py.shape[0] < num_poses:
        print(f"❌ Shape mismatch: C++ {poses_cpp.shape[0]} vs Python {poses_py.shape[0]} poses")
        return False
    
    # Compare translation components
    t_diff = np.abs(poses_cpp[:num_poses, :3] - poses_py[:num_poses, :3])
    t_max_diff = np.max(t_diff)
    t_mean_diff = np.mean(t_diff)
    
    print(f"\nTranslation differences:")
    print(f"  Max: {t_max_diff:.6e}")
    print(f"  Mean: {t_mean_diff:.6e}")
    
    # Compare quaternion components
    q_diff = np.abs(poses_cpp[:num_poses, 3:] - poses_py[:num_poses, 3:])
    q_max_diff = np.max(q_diff)
    q_mean_diff = np.mean(q_diff)
    
    print(f"\nQuaternion differences:")
    print(f"  Max: {q_max_diff:.6e}")
    print(f"  Mean: {q_mean_diff:.6e}")
    
    # Check if they match
    t_matches = np.allclose(poses_cpp[:num_poses, :3], poses_py[:num_poses, :3], atol=atol)
    q_matches = np.allclose(poses_cpp[:num_poses, 3:], poses_py[:num_poses, 3:], atol=atol)
    
    matches = t_matches and q_matches
    
    if matches:
        print(f"✅ BA output poses MATCH!")
    else:
        print(f"❌ BA output poses DO NOT MATCH!")
        
        # Show first few mismatches
        print(f"\nFirst 3 pose comparisons:")
        for i in range(min(3, num_poses)):
            print(f"\n  Pose {i}:")
            print(f"    C++ t:    ({poses_cpp[i, 0]:.6f}, {poses_cpp[i, 1]:.6f}, {poses_cpp[i, 2]:.6f})")
            print(f"    Python t: ({poses_py[i, 0]:.6f}, {poses_py[i, 1]:.6f}, {poses_py[i, 2]:.6f})")
            print(f"    t diff:   ({t_diff[i, 0]:.6e}, {t_diff[i, 1]:.6e}, {t_diff[i, 2]:.6e})")
            print(f"    C++ q:    ({poses_cpp[i, 3]:.6f}, {poses_cpp[i, 4]:.6f}, {poses_cpp[i, 5]:.6f}, {poses_cpp[i, 6]:.6f})")
            print(f"    Python q: ({poses_py[i, 3]:.6f}, {poses_py[i, 4]:.6f}, {poses_py[i, 5]:.6f}, {poses_py[i, 6]:.6f})")
            print(f"    q diff:   ({q_diff[i, 0]:.6e}, {q_diff[i, 1]:.6e}, {q_diff[i, 2]:.6e}, {q_diff[i, 3]:.6e})")
    
    return matches

def main() -> int:
    """Main function to compare C++ and Python BA outputs."""
    print("=" * 80)
    print("BA OUTPUT COMPARISON: C++ vs Python")
    print("=" * 80)
    
    # Load metadata
    print("\n📁 Loading metadata...")
    metadata = load_metadata()
    
    if metadata is None:
        print("⚠️  test_metadata.txt not found. Inferring from data files...")
        metadata = infer_metadata_from_files()
    
    if not metadata:
        print("❌ Error: Could not load or infer metadata. Please ensure BA input files exist.")
        return 1
    
    num_active = metadata.get('num_active', 0)
    MAX_EDGE = metadata.get('MAX_EDGE', num_active)  # Default to num_active if not specified
    DIM = metadata.get('DIM', 384)
    CORR_DIM = metadata.get('CORR_DIM', 882)
    M = metadata.get('M', 4)  # Patches per frame
    P = metadata.get('P', 3)   # Patch size
    N = metadata.get('N', 10)  # Number of frames
    
    print(f"  num_active: {num_active}")
    print(f"  MAX_EDGE: {MAX_EDGE}")
    print(f"  M (patches per frame): {M}")
    print(f"  P (patch size): {P}")
    print(f"  N (number of frames): {N}")
    
    # Load update model outputs
    print("\n📁 Loading update model outputs...")
    d_out = load_binary_file('test_d_out_py.bin')  # Use Python outputs (or C++ if preferred)
    w_out = load_binary_file('test_w_out_py.bin')
    
    if d_out is None or w_out is None:
        print("❌ Error: Update model outputs not found. Run compare_update_onnx_outputs.py first.")
        return 1
    
    # Reshape outputs
    d_out = d_out.reshape(1, 2, MAX_EDGE, 1)  # [1, 2, MAX_EDGE, 1]
    w_out = w_out.reshape(1, 2, MAX_EDGE, 1)  # [1, 2, MAX_EDGE, 1]
    
    print(f"  d_out shape: {d_out.shape}")
    print(f"  w_out shape: {w_out.shape}")
    
    # Load BA inputs (these need to be saved from C++ code)
    print("\n📁 Loading BA inputs...")
    print("⚠️  NOTE: BA inputs (poses, patches, intrinsics, indices) need to be saved from C++ code.")
    print("   This script expects:")
    print("   - ba_poses.bin: [N, 7] (tx, ty, tz, qx, qy, qz, qw)")
    print("   - ba_patches.bin: [N*M, 3, P, P] (flattened)")
    print("   - ba_intrinsics.bin: [N, 4] (fx, fy, cx, cy)")
    print("   - ba_ii.bin, ba_jj.bin, ba_kk.bin: [num_active] (int32)")
    print("   - ba_reprojected_coords.bin: [num_active, 2] (reprojected coordinates at patch center)")
    
    # Try to load BA inputs
    poses_np = load_binary_file('ba_poses.bin')
    patches_np = load_binary_file('ba_patches.bin')
    intrinsics_np = load_binary_file('ba_intrinsics.bin')
    ii_np = load_binary_file('ba_ii.bin', dtype=np.int32)
    jj_np = load_binary_file('ba_jj.bin', dtype=np.int32)
    kk_np = load_binary_file('ba_kk.bin', dtype=np.int32)
    reprojected_coords_np = load_binary_file('ba_reprojected_coords.bin')
    
    if any(x is None for x in [poses_np, patches_np, intrinsics_np, ii_np, jj_np, kk_np, reprojected_coords_np]):
        print("\n❌ Error: BA input files not found.")
        print("\n📝 To generate BA inputs, you need to:")
        print("   1. Modify C++ code to save BA inputs before calling bundleAdjustment()")
        print("   2. Save poses, patches, intrinsics, indices, and reprojected coordinates")
        print("   3. Then run this script again")
        return 1
    
    # Calculate actual N from file sizes (more reliable than metadata)
    # poses: [N, 7] -> N = poses_size / 7
    # intrinsics: [N, 4] -> N = intrinsics_size / 4
    # patches: [N*M, 3, P, P] -> N = patches_size / (M * 3 * P * P)
    N_from_poses = len(poses_np) // 7
    N_from_intrinsics = len(intrinsics_np) // 4
    N_from_patches = len(patches_np) // (M * 3 * P * P)
    
    # Use the most common N value (should all be the same)
    N_values = [N_from_poses, N_from_intrinsics, N_from_patches]
    N_actual = max(set(N_values), key=N_values.count)
    
    if N_actual != N:
        print(f"\n⚠️  WARNING: Metadata says N={N}, but files indicate N={N_actual}")
        print(f"   Using N={N_actual} from file sizes (poses: {N_from_poses}, intrinsics: {N_from_intrinsics}, patches: {N_from_patches})")
        N = N_actual
    
    # Verify consistency
    if not all(n == N for n in N_values):
        print(f"\n⚠️  WARNING: Inconsistent N values detected!")
        print(f"   poses: {N_from_poses}, intrinsics: {N_from_intrinsics}, patches: {N_from_patches}")
        print(f"   Using N={N} (most common value)")
    
    # Calculate actual num_active from file sizes
    # reprojected_coords: [num_active, 2] -> num_active = len(reprojected_coords_np) / 2
    # ii, jj, kk: [num_active] -> num_active = len(ii_np)
    num_active_from_coords = len(reprojected_coords_np) // 2
    num_active_from_ii = len(ii_np)
    num_active_from_jj = len(jj_np)
    num_active_from_kk = len(kk_np)
    
    # Use the most common num_active value
    num_active_values = [num_active_from_coords, num_active_from_ii, num_active_from_jj, num_active_from_kk]
    num_active_actual = max(set(num_active_values), key=num_active_values.count)
    
    if num_active_actual != num_active:
        print(f"\n⚠️  WARNING: Metadata says num_active={num_active}, but files indicate num_active={num_active_actual}")
        print(f"   Using num_active={num_active_actual} from file sizes (coords: {num_active_from_coords}, ii: {num_active_from_ii}, jj: {num_active_from_jj}, kk: {num_active_from_kk})")
        num_active = num_active_actual
    
    # Verify consistency
    if not all(n == num_active for n in num_active_values):
        print(f"\n⚠️  WARNING: Inconsistent num_active values detected!")
        print(f"   coords: {num_active_from_coords}, ii: {num_active_from_ii}, jj: {num_active_from_jj}, kk: {num_active_from_kk}")
        print(f"   Using num_active={num_active} (most common value)")
    
    # Reshape inputs using actual N and num_active
    poses_np = poses_np.reshape(N, 7)  # [N, 7]
    patches_np = patches_np.reshape(N * M, 3, P, P)  # [N*M, 3, P, P]
    intrinsics_np = intrinsics_np.reshape(N, 4)  # [N, 4]
    reprojected_coords_np = reprojected_coords_np.reshape(num_active, 2)  # [num_active, 2]
    
    print(f"  poses shape: {poses_np.shape}")
    print(f"  patches shape: {patches_np.shape}")
    print(f"  intrinsics shape: {intrinsics_np.shape}")
    print(f"  ii shape: {ii_np.shape}")
    print(f"  jj shape: {jj_np.shape}")
    print(f"  kk shape: {kk_np.shape}")
    print(f"  reprojected_coords shape: {reprojected_coords_np.shape}")
    
    # Convert to PyTorch tensors
    print("\n🔄 Converting to PyTorch tensors...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert poses to SE3 format
    poses_torch = torch.zeros(1, N, 7, device=device, dtype=torch.float32)
    for i in range(N):
        # poses_np[i] is [tx, ty, tz, qx, qy, qz, qw]
        # SE3 expects [tx, ty, tz, qw, qx, qy, qz]
        poses_torch[0, i, 0] = float(poses_np[i, 0])  # tx
        poses_torch[0, i, 1] = float(poses_np[i, 1])  # ty
        poses_torch[0, i, 2] = float(poses_np[i, 2])  # tz
        poses_torch[0, i, 3] = float(poses_np[i, 6])  # qw
        poses_torch[0, i, 4] = float(poses_np[i, 3])  # qx
        poses_torch[0, i, 5] = float(poses_np[i, 4])  # qy
        poses_torch[0, i, 6] = float(poses_np[i, 5])  # qz
    
    patches_torch = torch.from_numpy(patches_np.copy()).to(device).float().unsqueeze(0)  # [1, N*M, 3, P, P]
    intrinsics_torch = torch.from_numpy(intrinsics_np.copy()).to(device).float().unsqueeze(0)  # [1, N, 4]
    
    # Validate indices BEFORE computing targets/weights
    print(f"\n🔍 Validating indices...")
    original_num_active = num_active  # Store original value
    print(f"  ii range: [{ii_np[:num_active].min()}, {ii_np[:num_active].max()}] (should be < {N})")
    print(f"  jj range: [{jj_np[:num_active].min()}, {jj_np[:num_active].max()}] (should be < {N})")
    print(f"  kk range: [{kk_np[:num_active].min()}, {kk_np[:num_active].max()}] (should be < {N*M})")
    
    # Check for out-of-bounds indices
    ii_invalid = np.sum((ii_np[:num_active] < 0) | (ii_np[:num_active] >= N))
    jj_invalid = np.sum((jj_np[:num_active] < 0) | (jj_np[:num_active] >= N))
    kk_invalid = np.sum((kk_np[:num_active] < 0) | (kk_np[:num_active] >= N*M))
    
    # Filter out invalid edges if needed
    valid_mask = None
    if ii_invalid > 0 or jj_invalid > 0 or kk_invalid > 0:
        print(f"\n⚠️  WARNING: Found invalid indices!")
        print(f"   ii invalid: {ii_invalid}/{num_active}")
        print(f"   jj invalid: {jj_invalid}/{num_active}")
        print(f"   kk invalid: {kk_invalid}/{num_active}")
        print(f"   Filtering out invalid edges...")
        
        # Filter out invalid edges
        valid_mask = (
            (ii_np[:num_active] >= 0) & (ii_np[:num_active] < N) &
            (jj_np[:num_active] >= 0) & (jj_np[:num_active] < N) &
            (kk_np[:num_active] >= 0) & (kk_np[:num_active] < N*M)
        )
        num_valid = np.sum(valid_mask)
        print(f"   Valid edges: {num_valid}/{num_active}")
        
        if num_valid == 0:
            print("❌ Error: No valid edges after filtering!")
            return 1
        
        num_active = num_valid
        print(f"   Updated num_active: {num_active}")
    
    # Load targets and weights from C++ saved files (these are what C++ BA actually uses)
    print("\n🎯 Loading targets and weights from C++ saved files...")
    targets_np = load_binary_file('ba_targets.bin')
    weights_np = load_binary_file('ba_weights.bin')
    
    if targets_np is None or weights_np is None:
        print("⚠️  WARNING: ba_targets.bin or ba_weights.bin not found, computing from reprojected_coords + d_out")
        # Fallback: compute targets from reprojected_coords + d_out
        if valid_mask is not None:
            reprojected_coords_np_filtered = reprojected_coords_np[:original_num_active][valid_mask]
            d_out_active = d_out[0, :, :original_num_active, 0][:, valid_mask]  # [2, num_valid]
            w_out_active = w_out[0, :, :original_num_active, 0][:, valid_mask]  # [2, num_valid]
            ii_filtered = ii_np[:original_num_active][valid_mask]
            jj_filtered = jj_np[:original_num_active][valid_mask]
            kk_filtered = kk_np[:original_num_active][valid_mask]
        else:
            reprojected_coords_np_filtered = reprojected_coords_np[:num_active]
            d_out_active = d_out[0, :, :num_active, 0]  # [2, num_active]
            w_out_active = w_out[0, :, :num_active, 0]  # [2, num_active]
            ii_filtered = ii_np[:num_active]
            jj_filtered = jj_np[:num_active]
            kk_filtered = kk_np[:num_active]
        
        reprojected_coords_torch = torch.from_numpy(reprojected_coords_np_filtered).to(device).float()  # [num_active, 2]
        d_out_torch = torch.from_numpy(d_out_active).to(device).float().T  # [num_active, 2]
        
        # Compute targets
        targets_torch = reprojected_coords_torch + d_out_torch  # [num_active, 2]
        targets_torch = targets_torch.unsqueeze(0)  # [1, num_active, 2]
        
        # Extract weights for active edges
        weights_torch = torch.from_numpy(w_out_active).to(device).float().T  # [num_active, 2]
        weights_torch = weights_torch.unsqueeze(0)  # [1, num_active, 2]
    else:
        # Use saved targets and weights (what C++ BA actually uses)
        targets_np = targets_np.reshape(-1, 2)  # [num_active, 2]
        weights_np = weights_np.reshape(-1, 2)  # [num_active, 2]
        
        print(f"  Loaded targets shape: {targets_np.shape}")
        print(f"  Loaded weights shape: {weights_np.shape}")
        
        # Filter if needed
        if valid_mask is not None:
            targets_np_filtered = targets_np[:original_num_active][valid_mask]
            weights_np_filtered = weights_np[:original_num_active][valid_mask]
            ii_filtered = ii_np[:original_num_active][valid_mask]
            jj_filtered = jj_np[:original_num_active][valid_mask]
            kk_filtered = kk_np[:original_num_active][valid_mask]
            num_active = np.sum(valid_mask)
        else:
            targets_np_filtered = targets_np[:num_active]
            weights_np_filtered = weights_np[:num_active]
            ii_filtered = ii_np[:num_active]
            jj_filtered = jj_np[:num_active]
            kk_filtered = kk_np[:num_active]
        
        # Convert to tensors
        targets_torch = torch.from_numpy(targets_np_filtered.copy()).to(device).float().unsqueeze(0)  # [1, num_active, 2]
        
        # CRITICAL: Python DPVO passes weights shape [1, M, 2] to BA (see dpvo.py line 823)
        # Python BA receives [1, M, 2] and uses broadcasting: weights[..., None] makes it [1, M, 2, 1]
        # Then (weights * Ji) broadcasts [1, M, 2, 1] * [1, M, 2, 6] = [1, M, 2, 6]
        # This means Python BA uses both channels separately (w0 for x, w1 for y)
        # C++ also uses both channels separately, so we should pass [1, M, 2] to match Python DPVO behavior
        weights_torch = torch.from_numpy(weights_np_filtered.copy()).to(device).float().unsqueeze(0)  # [1, num_active, 2]
    
    # Convert indices to tensors (after filtering)
    ii_torch = torch.from_numpy(ii_filtered).to(device).long()
    jj_torch = torch.from_numpy(jj_filtered).to(device).long()
    kk_torch = torch.from_numpy(kk_filtered).to(device).long()
    
    print(f"  targets shape: {targets_torch.shape}")
    print(f"  weights shape: {weights_torch.shape}")
    
    # Run Python BA
    print("\n🐍 Running Python BA function...")
    print(f"  Input shapes:")
    print(f"    poses_se3: {poses_torch.shape}")
    print(f"    patches: {patches_torch.shape}")
    print(f"    intrinsics: {intrinsics_torch.shape}")
    print(f"    targets: {targets_torch.shape}")
    print(f"    weights: {weights_torch.shape}")
    print(f"    ii: {ii_torch.shape}, range: [{ii_torch.min().item()}, {ii_torch.max().item()}]")
    print(f"    jj: {jj_torch.shape}, range: [{jj_torch.min().item()}, {jj_torch.max().item()}]")
    print(f"    kk: {kk_torch.shape}, range: [{kk_torch.min().item()}, {kk_torch.max().item()}]")
    
    # Debug: Check weight values
    print(f"  Weight statistics:")
    print(f"    weights min: {weights_torch.min().item():.6f}, max: {weights_torch.max().item():.6f}")
    print(f"    weights mean: {weights_torch.mean().item():.6f}")
    if weights_torch.dim() == 3:  # [1, num_active, 2]
        print(f"    weights channel 0 (x) mean: {weights_torch[0, :, 0].mean().item():.6f}")
        print(f"    weights channel 1 (y) mean: {weights_torch[0, :, 1].mean().item():.6f}")
    
    lmbda = torch.tensor([1e-4], device=device, dtype=torch.float32)
    t0 = 1  # First pose to optimize (0 is fixed)
    t1 = N  # Last pose to optimize
    
    # Convert poses to SE3 format for Python BA
    poses_se3 = SE3(poses_torch)
    
    try:
        new_poses_py = python_ba_wrapper(
            poses_se3,
            patches_torch,
            intrinsics_torch,
            targets_torch,
            weights_torch,
            lmbda,
            ii_torch,
            jj_torch,
            kk_torch,
            PPF=None,
            t0=t0,
            t1=t1,
            iterations=1,
            eff_impl=False
        )
        
        # Convert back to numpy
        poses_py_np = new_poses_py.data.cpu().numpy()[0]  # [N, 7] in format [tx, ty, tz, qw, qx, qy, qz]
        
        # Convert to [tx, ty, tz, qx, qy, qz, qw] format
        poses_py_final = np.zeros((N, 7), dtype=np.float32)
        poses_py_final[:, 0] = poses_py_np[:, 0]  # tx
        poses_py_final[:, 1] = poses_py_np[:, 1]  # ty
        poses_py_final[:, 2] = poses_py_np[:, 2]  # tz
        poses_py_final[:, 3] = poses_py_np[:, 4]  # qx
        poses_py_final[:, 4] = poses_py_np[:, 5]  # qy
        poses_py_final[:, 5] = poses_py_np[:, 6]  # qz
        poses_py_final[:, 6] = poses_py_np[:, 3]  # qw
        
        print(f"✅ Python BA completed successfully")
        print(f"  Output poses shape: {poses_py_final.shape}")
        
        # Save Python BA outputs
        poses_py_final.tofile('ba_poses_py.bin')
        print(f"  Saved Python BA outputs to ba_poses_py.bin")
        
    except Exception as e:
        print(f"❌ Python BA failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Load C++ BA outputs (if available)
    print("\n📁 Loading C++ BA outputs...")
    poses_cpp = load_binary_file('ba_poses_cpp.bin')
    
    if poses_cpp is None:
        print("⚠️  C++ BA outputs not found (ba_poses_cpp.bin).")
        print("   To compare:")
        print("   1. Modify C++ code to save BA outputs after bundleAdjustment()")
        print("   2. Save updated poses to ba_poses_cpp.bin")
        print("   3. Then run this script again")
        return 0
    
    # Reshape C++ poses
    poses_cpp = poses_cpp.reshape(N, 7)  # [N, 7]
    
    # Compare outputs
    print("\n" + "="*80)
    print("COMPARING BA OUTPUTS")
    print("="*80)
    
    matches = compare_poses(poses_cpp, poses_py_final, N, atol=1e-4)
    
    # Summary
    print("\n" + "="*80)
    if matches:
        print("✅ BA OUTPUTS MATCH! C++ and Python BA functions produce identical results.")
    else:
        print("❌ BA OUTPUTS DO NOT MATCH! There may be differences in the BA implementation.")
    print("="*80)
    
    return 0 if matches else 1

if __name__ == "__main__":
    sys.exit(main())

