#!/usr/bin/env python3
"""
Step-by-step comparison of C++ and Python BA implementations.
This script compares each unit/step of the BA algorithm to identify differences.
"""

import numpy as np
import torch
import sys
import os

# Add DPVO_onnx to path
sys.path.insert(0, '/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx')
from dpvo.ba import BA
from dpvo.lietorch import SE3
from dpvo import projective_ops as pops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_binary_file(filename, dtype=np.float32, warn_if_missing=False):
    """Load binary file as numpy array"""
    if not os.path.exists(filename):
        if warn_if_missing:
            print(f"  ⚠️  C++ comparison file not found: {filename}")
        return None
    data = np.fromfile(filename, dtype=dtype)
    return data

def load_int32_file(filename):
    """Load binary file as int32 array"""
    return load_binary_file(filename, dtype=np.int32)

def compare_tensors(name, cpp_val, py_val, rtol=1e-2, atol=1e-3, show_table=False, max_edges=5):
    """Compare two tensors/arrays and report differences"""
    cpp_tensor = torch.from_numpy(cpp_val).to(device).float() if isinstance(cpp_val, np.ndarray) else cpp_val
    py_tensor = py_val if isinstance(py_val, torch.Tensor) else torch.from_numpy(py_val).to(device).float()
    
    if cpp_tensor.shape != py_tensor.shape:
        print(f"  ❌ {name}: Shape mismatch - C++: {cpp_tensor.shape}, Python: {py_tensor.shape}")
        return False
    
    diff = torch.abs(cpp_tensor - py_tensor)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Count how many values exceed tolerance
    exceeds_tol = (diff > (rtol * torch.abs(py_tensor) + atol)).sum().item()
    total_elements = cpp_tensor.numel()
    exceeds_percent = 100.0 * exceeds_tol / total_elements if total_elements > 0 else 0.0
    
    # Check relative difference (for large values)
    py_abs = torch.abs(py_tensor)
    py_max = py_abs.max().item()
    relative_max_diff = max_diff / max(py_max, 1e-10)
    
    is_close = torch.allclose(cpp_tensor, py_tensor, rtol=rtol, atol=atol)
    
    if is_close:
        print(f"  ✅ {name}: MATCH (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
    else:
        # Check if it's just numerical precision (very small differences)
        # For matrix operations, allow slightly larger tolerance
        numerical_precision_threshold = max(1e-4, atol * 10)  # Allow 10x the absolute tolerance
        if max_diff < numerical_precision_threshold and relative_max_diff < 1e-3:
            print(f"  ✅ {name}: MATCH (NUMERICAL PRECISION) (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
                  f"rel_diff={relative_max_diff:.2e}, {exceeds_tol}/{total_elements} exceed tol)")
            print(f"     Note: Differences are very small, likely due to floating-point precision")
            print(f"     Tolerance used: rtol={rtol:.0e}, atol={atol:.0e}")
        else:
            print(f"  ❌ {name}: MISMATCH (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
                  f"rel_diff={relative_max_diff:.2e}, {exceeds_tol}/{total_elements} exceed tol)")
        if cpp_tensor.numel() <= 20:
            print(f"    C++:   {cpp_tensor.cpu().numpy()}")
            print(f"    Python: {py_tensor.cpu().numpy()}")
    
    # Show detailed comparison table if requested (even for matched cases)
    if show_table:
        # Find edge with maximum difference for 2D/3D tensors (if not matched)
        if not is_close and len(cpp_tensor.shape) >= 2:
            diff_reshaped = diff.view(cpp_tensor.shape[0], -1)  # Flatten all but first dimension
            max_diff_per_edge = diff_reshaped.max(dim=1)[0]  # Max diff per edge
            max_edge_idx = max_diff_per_edge.argmax().item()
            max_edge_diff = max_diff_per_edge[max_edge_idx].item()
            
            if max_edge_diff > 1e-5:  # Only show if significant
                print(f"\n  🔍 Edge with maximum difference: Edge {max_edge_idx} (diff={max_edge_diff:.6e})")
        
        print_comparison_table(name, cpp_tensor, py_tensor, max_edges=max_edges)
    
    return is_close

def print_comparison_table(name, cpp_tensor, py_tensor, max_edges=5):
    """Print a detailed comparison table showing values for each edge/channel"""
    cpp_np = cpp_tensor.cpu().numpy()
    py_np = py_tensor.cpu().numpy()
    
    # Determine the shape and how to iterate
    shape = cpp_np.shape
    ndim = len(shape)
    
    if ndim == 1:
        # 1D: [num_active] or [m]
        num_items = shape[0]
        print(f"\n  📊 Detailed Comparison Table for {name}:")
        print(f"  {'Index':<8} {'C++ Value':<18} {'Python Value':<18} {'Difference':<18}")
        print(f"  {'-'*8} {'-'*18} {'-'*18} {'-'*18}")
        for i in range(min(max_edges, num_items)):
            diff = abs(cpp_np[i] - py_np[i])
            print(f"  {i:<8} {cpp_np[i]:<18.6f} {py_np[i]:<18.6f} {diff:<18.6e}")
        if num_items > max_edges:
            print(f"  ... ({num_items - max_edges} more items)")
    
    elif ndim == 2:
        # 2D: [num_active, 2] or [num_active, 6] or [m, 6]
        num_edges = shape[0]
        num_channels = shape[1]
        print(f"\n  📊 Detailed Comparison Table for {name}:")
        for e in range(min(max_edges, num_edges)):
            print(f"\n  Edge {e}:")
            print(f"  {'Channel':<10} {'C++ Value':<18} {'Python Value':<18} {'Difference':<18}")
            print(f"  {'-'*10} {'-'*18} {'-'*18} {'-'*18}")
            for c in range(num_channels):
                diff = abs(cpp_np[e, c] - py_np[e, c])
                print(f"  {c:<10} {cpp_np[e, c]:<18.6f} {py_np[e, c]:<18.6f} {diff:<18.6e}")
        if num_edges > max_edges:
            print(f"\n  ... ({num_edges - max_edges} more edges)")
    
    elif ndim == 3:
        # 3D: [num_active, 6, 2] or [num_active, 6, 6] or [num_active, 1, 2]
        num_edges = shape[0]
        dim1 = shape[1]
        dim2 = shape[2]
        print(f"\n  📊 Detailed Comparison Table for {name}:")
        for e in range(min(max_edges, num_edges)):
            print(f"\n  Edge {e}:")
            
            # For 6x6 matrices (Hessian blocks), show as formatted matrices for better readability
            if dim1 == 6 and dim2 == 6:
                print(f"  C++ {name}[{e}]:")
                print(f"    {'':<8}", end="")
                for j in range(dim2):
                    print(f"Col {j:<10}", end="")
                print()
                for i in range(dim1):
                    print(f"    Row {i}:", end="")
                    for j in range(dim2):
                        print(f"{cpp_np[e, i, j]:>12.6f}", end="")
                    print()
                
                print(f"  Python {name}[{e}]:")
                print(f"    {'':<8}", end="")
                for j in range(dim2):
                    print(f"Col {j:<10}", end="")
                print()
                for i in range(dim1):
                    print(f"    Row {i}:", end="")
                    for j in range(dim2):
                        print(f"{py_np[e, i, j]:>12.6f}", end="")
                    print()
                
                print(f"  Difference matrix:")
                print(f"    {'':<8}", end="")
                for j in range(dim2):
                    print(f"Col {j:<10}", end="")
                print()
                for i in range(dim1):
                    print(f"    Row {i}:", end="")
                    for j in range(dim2):
                        diff = abs(cpp_np[e, i, j] - py_np[e, i, j])
                        print(f"{diff:>12.6e}", end="")
                    print()
            # For [num_active, 6, 2] or [num_active, 1, 2], show as channel table
            elif dim1 <= 6 and dim2 <= 6:
                print(f"  {'Row':<6} {'Col':<6} {'C++ Value':<18} {'Python Value':<18} {'Difference':<18}")
                print(f"  {'-'*6} {'-'*6} {'-'*18} {'-'*18} {'-'*18}")
                for i in range(dim1):
                    for j in range(dim2):
                        diff = abs(cpp_np[e, i, j] - py_np[e, i, j])
                        print(f"  {i:<6} {j:<6} {cpp_np[e, i, j]:<18.6f} {py_np[e, i, j]:<18.6f} {diff:<18.6e}")
            else:
                # For larger matrices, show summary
                print(f"  Shape: [{dim1}, {dim2}] - showing first few values")
                print(f"  {'Row':<6} {'Col':<6} {'C++ Value':<18} {'Python Value':<18} {'Difference':<18}")
                print(f"  {'-'*6} {'-'*6} {'-'*18} {'-'*18} {'-'*18}")
                count = 0
                for i in range(min(6, dim1)):
                    for j in range(min(6, dim2)):
                        diff = abs(cpp_np[e, i, j] - py_np[e, i, j])
                        if diff > 1e-6 or count < 10:  # Show first 10 or significant differences
                            print(f"  {i:<6} {j:<6} {cpp_np[e, i, j]:<18.6f} {py_np[e, i, j]:<18.6f} {diff:<18.6e}")
                            count += 1
        if num_edges > max_edges:
            print(f"\n  ... ({num_edges - max_edges} more edges)")
    
    elif ndim >= 4:
        # Higher dimensions: flatten or show summary
        print(f"\n  📊 Detailed Comparison Table for {name}:")
        print(f"  Shape: {shape}")
        print(f"  Showing first few values:")
        flat_cpp = cpp_np.flatten()
        flat_py = py_np.flatten()
        print(f"  {'Index':<8} {'C++ Value':<18} {'Python Value':<18} {'Difference':<18}")
        print(f"  {'-'*8} {'-'*18} {'-'*18} {'-'*18}")
        for i in range(min(max_edges * 5, len(flat_cpp))):
            diff = abs(flat_cpp[i] - flat_py[i])
            if diff > 1e-6:  # Only show significant differences
                print(f"  {i:<8} {flat_cpp[i]:<18.6f} {flat_py[i]:<18.6f} {diff:<18.6e}")

def load_metadata():
    """Load metadata from test_metadata.txt and infer from file sizes"""
    metadata = {}
    if os.path.exists("test_metadata.txt"):
        with open("test_metadata.txt", "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=")
                    metadata[key] = int(value)
    
    M = metadata.get("M", 4)
    P = metadata.get("P", 3)
    N = metadata.get("N", 10)
    
    # Infer N from file sizes
    poses_data = load_binary_file("ba_poses.bin")
    if poses_data is not None:
        N = len(poses_data) // 7
    
    num_active = metadata.get("num_active", 10)
    coords_data = load_binary_file("ba_reprojected_coords.bin")
    if coords_data is not None:
        num_active = len(coords_data) // 2
    
    return M, P, N, num_active

def load_ba_inputs(M, P, N, num_active):
    """
    Load all BA inputs from binary files.
    
    Returns:
        poses_se3: SE3 poses [1, N, 7]
        patches_torch: Patches [1, N*M, 3, P, P]
        intrinsics_torch: Intrinsics [1, N, 4]
        ii_torch, jj_torch, kk_torch: Indices [num_active]
        targets_torch: Targets [1, num_active, 2]
        weights_torch: Weights [1, num_active, 2]
        bounds: Image bounds tensor [4]
    """
    # Load poses [N, 7]
    poses_data = load_binary_file("ba_poses.bin")
    if poses_data is None:
        raise FileNotFoundError("Failed to load ba_poses.bin")
    poses_np = poses_data.reshape(N, 7)
    poses_torch = torch.from_numpy(poses_np.copy()).to(device).float().unsqueeze(0)
    poses_se3 = SE3(poses_torch)
    
    # Load patches [N*M, 3, P, P]
    patches_data = load_binary_file("ba_patches.bin")
    if patches_data is None:
        raise FileNotFoundError("Failed to load ba_patches.bin")
    patches_np = patches_data.reshape(N * M, 3, P, P)
    patches_torch = torch.from_numpy(patches_np.copy()).to(device).float().unsqueeze(0)
    
    # Load intrinsics [N, 4]
    intrinsics_data = load_binary_file("ba_intrinsics.bin")
    if intrinsics_data is None:
        raise FileNotFoundError("Failed to load ba_intrinsics.bin")
    intrinsics_np = intrinsics_data.reshape(N, 4)
    intrinsics_torch = torch.from_numpy(intrinsics_np.copy()).to(device).float().unsqueeze(0)
    
    # Load indices
    ii_np = load_int32_file("ba_ii.bin")
    jj_np = load_int32_file("ba_jj.bin")
    kk_np = load_int32_file("ba_kk.bin")
    if ii_np is None or jj_np is None or kk_np is None:
        raise FileNotFoundError("Failed to load index files")
    
    ii_torch = torch.from_numpy(ii_np[:num_active]).to(device).long()
    jj_torch = torch.from_numpy(jj_np[:num_active]).to(device).long()
    kk_torch = torch.from_numpy(kk_np[:num_active]).to(device).long()
    
    # Load targets and weights
    targets_data = load_binary_file("ba_targets.bin")
    weights_data = load_binary_file("ba_weights.bin")
    if targets_data is None or weights_data is None:
        raise FileNotFoundError("Failed to load targets/weights")
    
    targets_np = targets_data.reshape(num_active, 2)
    weights_np = weights_data.reshape(num_active, 2)
    targets_torch = torch.from_numpy(targets_np.copy()).to(device).float().unsqueeze(0)
    weights_torch = torch.from_numpy(weights_np.copy()).to(device).float().unsqueeze(0)
    
    # Compute bounds from intrinsics
    H = int(2 * intrinsics_torch[0, :, 3].max().item())
    W = int(2 * intrinsics_torch[0, :, 2].max().item())
    bounds = torch.tensor([0.0, 0.0, W - 1.0, H - 1.0], device=device, dtype=torch.float32)
    
    return poses_se3, patches_torch, intrinsics_torch, ii_torch, jj_torch, kk_torch, targets_torch, weights_torch, bounds

def step1_forward_projection(poses_se3, patches_torch, intrinsics_torch, ii_torch, jj_torch, kk_torch, P, num_active):
    """
    STEP 1: Forward projection + Jacobians
    
    Returns:
        coords_py: Projected coordinates [1, num_active, P, P, 2]
        v_py: Validity mask [1, num_active]
        Ji_py, Jj_py, Jz_py: Jacobians [1, num_active, 2, 6] or [1, num_active, 2, 1]
    """
    print("\n📊 STEP 1: Forward projection + Jacobians")
    print("-" * 80)
    
    coords_py, v_py, (Ji_py, Jj_py, Jz_py) = \
        pops.transform(poses_se3, patches_torch, intrinsics_torch, ii_torch, jj_torch, kk_torch, jacobian=True)
    
    print(f"  Python outputs:")
    print(f"    coords shape: {coords_py.shape}")
    print(f"    v shape: {v_py.shape}, valid_count: {(v_py > 0.5).sum().item()}/{v_py.numel()}")
    print(f"    Ji shape: {Ji_py.shape}")
    print(f"    Jj shape: {Jj_py.shape}")
    print(f"    Jz shape: {Jz_py.shape}")
    
    # Compare coords center
    coords_cpp_data = load_binary_file("ba_reprojected_coords.bin")
    if coords_cpp_data is not None:
        coords_cpp_center = coords_cpp_data.reshape(num_active, 2)
        coords_py_center = coords_py[0, :, P//2, P//2, :].cpu().numpy()
        compare_tensors("coords (center)", coords_cpp_center, coords_py_center)
    
    return coords_py, v_py, Ji_py, Jj_py, Jz_py

def step2_compute_residuals(targets_torch, coords_py, P, num_active, v_py):
    """
    STEP 2: Compute residual at patch center
    
    Returns:
        r_py: Residuals [1, num_active, 2]
    """
    print("\n📊 STEP 2: Compute residual at patch center")
    print("-" * 80)
    
    r_py = targets_torch - coords_py[..., P//2, P//2, :]  # [1, num_active, 2]
    
    print(f"  Python residual shape: {r_py.shape}")
    print(f"  Python residual stats: min={r_py.min().item():.6f}, max={r_py.max().item():.6f}, mean={r_py.mean().item():.6f}")
    
    # Compare with C++ STEP 1 residuals
    r_cpp_data = load_binary_file("ba_step1_residuals.bin", warn_if_missing=True)
    if r_cpp_data is not None:
        r_cpp = r_cpp_data.reshape(num_active, 2)
        r_py_np = r_py[0].cpu().numpy()
        compare_tensors("residuals", r_cpp, r_py_np, show_table=True, max_edges=3)
    
    # Compare validity mask
    v_cpp_data = load_binary_file("ba_step1_validity.bin", warn_if_missing=True)
    if v_cpp_data is not None:
        v_cpp = v_cpp_data
        v_py_np = v_py[0].cpu().numpy()
        compare_tensors("validity mask", v_cpp, v_py_np, show_table=True, max_edges=3)
    
    return r_py

def step3_apply_validity_mask(r_py, coords_py, v_py, weights_torch, P, bounds):
    """
    STEP 3: Apply validity mask (reject large residuals and out-of-bounds projections)
    
    Returns:
        r_py_masked: Masked residuals [1, num_active, 2, 1]
        weights_py_masked: Masked weights [1, num_active, 2, 1]
        v_py: Updated validity mask [1, num_active]
    """
    print("\n📊 STEP 3: Validity mask application")
    print("-" * 80)
    
    # Reject large residuals
    v_py_large = (r_py.norm(dim=-1) < 250).float()
    v_py *= v_py_large
    
    # Reject projections outside image bounds
    coords_center = coords_py[..., P//2, P//2, :]
    in_bounds = (
        (coords_center[..., 0] > bounds[0]) &
        (coords_center[..., 1] > bounds[1]) &
        (coords_center[..., 0] < bounds[2]) &
        (coords_center[..., 1] < bounds[3])
    )
    v_py *= in_bounds.float()
    
    print(f"  Python validity after large residual check: {(v_py_large > 0.5).sum().item()}/{v_py_large.numel()}")
    print(f"  Python validity after bounds check: {(in_bounds > 0.5).sum().item()}/{in_bounds.numel()}")
    print(f"  Python final validity: {(v_py > 0.5).sum().item()}/{v_py.numel()}")
    
    # Apply validity mask
    r_py_masked = (v_py[..., None] * r_py).unsqueeze(dim=-1)  # [1, num_active, 2, 1]
    weights_py_masked = (v_py[..., None] * weights_torch).unsqueeze(dim=-1)  # [1, num_active, 2, 1]
    
    print(f"  Python masked residual shape: {r_py_masked.shape}")
    print(f"  Python masked weights shape: {weights_py_masked.shape}")
    
    return r_py_masked, weights_py_masked, v_py

def step4_build_weighted_jacobians(weights_py_masked, Ji_py, Jj_py, Jz_py, num_active):
    """
    STEP 4: Build weighted Jacobians
    
    Returns:
        wJiT_py: Weighted Jacobian transpose [1, num_active, 6, 2]
        wJjT_py: Weighted Jacobian transpose [1, num_active, 6, 2]
        wJzT_py: Weighted Jacobian transpose [1, num_active, 1, 2]
    """
    print("\n📊 STEP 4: Build weighted Jacobians")
    print("-" * 80)
    
    wJiT_py = (weights_py_masked * Ji_py).transpose(2, 3)  # [1, num_active, 6, 2]
    wJjT_py = (weights_py_masked * Jj_py).transpose(2, 3)  # [1, num_active, 6, 2]
    wJzT_py = (weights_py_masked * Jz_py).transpose(2, 3)  # [1, num_active, 1, 2]
    
    print(f"  Python wJiT shape: {wJiT_py.shape}")
    print(f"  Python wJjT shape: {wJjT_py.shape}")
    print(f"  Python wJzT shape: {wJzT_py.shape}")
    
    # Compare with C++ STEP 2 weighted Jacobians
    wJiT_cpp_data = load_binary_file("ba_step2_wJiT.bin", warn_if_missing=True)
    wJjT_cpp_data = load_binary_file("ba_step2_wJjT.bin", warn_if_missing=True)
    wJzT_cpp_data = load_binary_file("ba_step2_wJzT.bin", warn_if_missing=True)
    weights_masked_cpp_data = load_binary_file("ba_step2_weights_masked.bin", warn_if_missing=True)
    
    if wJiT_cpp_data is not None:
        wJiT_cpp = wJiT_cpp_data.reshape(num_active, 6, 2)
        wJiT_py_np = wJiT_py[0].cpu().numpy()
        
        # Debug: Show Edge 4 (max difference edge) values
        print(f"\n  🔍 Debug Edge 4 (max difference edge):")
        print(f"  C++ wJiT[4] shape: {wJiT_cpp[4].shape}")
        print(f"  Python wJiT[4] shape: {wJiT_py_np[4].shape}")
        print(f"  C++ wJiT[4]:")
        print(f"    {wJiT_cpp[4]}")
        print(f"  Python wJiT[4]:")
        print(f"    {wJiT_py_np[4]}")
        
        # Show intermediate values for Edge 4
        print(f"\n  🔍 Debug Edge 4 intermediate values:")
        print(f"  weights_masked[4]: {weights_py_masked[0, 4, :, 0].cpu().numpy()}")
        print(f"  Ji_py[4] shape: {Ji_py[0, 4].shape}")
        print(f"  Ji_py[4]:")
        print(f"    {Ji_py[0, 4].cpu().numpy()}")
        weighted_before_transpose = (weights_py_masked[0, 4:5] * Ji_py[0, 4:5])
        print(f"  (weights * Ji)[4] before transpose:")
        print(f"    {weighted_before_transpose[0].cpu().numpy()}")
        print(f"  (weights * Ji)[4] after transpose(2,3):")
        print(f"    {wJiT_py[0, 4].cpu().numpy()}")
        
        # Also check Edge 0 for comparison
        print(f"\n  🔍 Debug Edge 0 (first edge):")
        print(f"  C++ wJiT[0]:")
        print(f"    {wJiT_cpp[0]}")
        print(f"  Python wJiT[0]:")
        print(f"    {wJiT_py_np[0]}")
        print(f"  weights_masked[0]: {weights_py_masked[0, 0, :, 0].cpu().numpy()}")
        print(f"  Ji_py[0]:")
        print(f"    {Ji_py[0, 0].cpu().numpy()}")
        
        # Check if there's a real mismatch or just numerical precision
        compare_tensors("wJiT", wJiT_cpp, wJiT_py_np, rtol=1e-3, atol=1e-4, show_table=True, max_edges=2)
    
    if wJjT_cpp_data is not None:
        wJjT_cpp = wJjT_cpp_data.reshape(num_active, 6, 2)
        wJjT_py_np = wJjT_py[0].cpu().numpy()
        # Check if there's a real mismatch or just numerical precision
        compare_tensors("wJjT", wJjT_cpp, wJjT_py_np, rtol=1e-3, atol=1e-4, show_table=True, max_edges=2)
    
    if wJzT_cpp_data is not None:
        wJzT_cpp = wJzT_cpp_data.reshape(num_active, 1, 2)
        wJzT_py_np = wJzT_py[0].cpu().numpy()
        compare_tensors("wJzT", wJzT_cpp, wJzT_py_np, show_table=True, max_edges=2)
    
    if weights_masked_cpp_data is not None:
        weights_masked_cpp = weights_masked_cpp_data.reshape(num_active, 2)
        weights_masked_py_np = weights_py_masked[0, :, :, 0].cpu().numpy()
        compare_tensors("weights_masked", weights_masked_cpp, weights_masked_py_np, show_table=True, max_edges=3)
    
    # Sample values for first edge
    print(f"\n  Sample values (first edge, first pose param):")
    print(f"    wJiT[0,0,0,0] (x-dir): {wJiT_py[0,0,0,0].item():.6f}")
    print(f"    wJiT[0,0,0,1] (y-dir): {wJiT_py[0,0,0,1].item():.6f}")
    print(f"    weights[0,0,0,0]: {weights_py_masked[0,0,0,0].item():.6f}")
    print(f"    weights[0,0,1,0]: {weights_py_masked[0,0,1,0].item():.6f}")
    print(f"    Ji[0,0,0,0]: {Ji_py[0,0,0,0].item():.6f}")
    print(f"    Ji[0,0,1,0]: {Ji_py[0,0,1,0].item():.6f}")
    
    return wJiT_py, wJjT_py, wJzT_py

def step5_compute_hessian_blocks(wJiT_py, wJjT_py, Ji_py, Jj_py, Jz_py, num_active):
    """
    STEP 5: Compute Hessian blocks
    
    Returns:
        Bii_py, Bij_py, Bji_py, Bjj_py: Pose-pose Hessian blocks [1, num_active, 6, 6]
        Eik_py, Ejk_py: Pose-structure Hessian blocks [1, num_active, 6, 1]
    """
    print("\n📊 STEP 5: Compute Hessian blocks")
    print("-" * 80)
    
    Bii_py = torch.matmul(wJiT_py, Ji_py)  # [1, num_active, 6, 6]
    Bij_py = torch.matmul(wJiT_py, Jj_py)  # [1, num_active, 6, 6]
    Bji_py = torch.matmul(wJjT_py, Ji_py)  # [1, num_active, 6, 6]
    Bjj_py = torch.matmul(wJjT_py, Jj_py)  # [1, num_active, 6, 6]
    Eik_py = torch.matmul(wJiT_py, Jz_py)  # [1, num_active, 6, 1]
    Ejk_py = torch.matmul(wJjT_py, Jz_py)  # [1, num_active, 6, 1]
    
    print(f"  Python Bii shape: {Bii_py.shape}")
    print(f"  Python Bij shape: {Bij_py.shape}")
    print(f"  Python Eik shape: {Eik_py.shape}")
    
    # Compare with C++ STEP 3 Hessian blocks
    Bii_cpp_data = load_binary_file("ba_step3_Bii.bin")
    Bij_cpp_data = load_binary_file("ba_step3_Bij.bin")
    Eik_cpp_data = load_binary_file("ba_step3_Eik.bin")
    
    if Bii_cpp_data is not None:
        Bii_cpp = Bii_cpp_data.reshape(num_active, 6, 6)
        Bii_py_np = Bii_py[0].cpu().numpy()
        
        # Debug: Show Edge 94 (max difference edge) values
        diff_Bii = np.abs(Bii_cpp - Bii_py_np)
        max_diff_per_edge = diff_Bii.reshape(num_active, -1).max(axis=1)
        max_edge_idx = max_diff_per_edge.argmax()
        print(f"\n  🔍 Debug Edge {max_edge_idx} (max difference edge for Bii):")
        print(f"  Max diff: {max_diff_per_edge[max_edge_idx]:.6e}")
        
        # Use relaxed tolerance for Hessian blocks (matrix multiplication accumulates small errors)
        # The differences are very small (mostly 0 or 1e-6 to 1e-5), so use same tolerance as weighted Jacobians
        compare_tensors("Bii", Bii_cpp, Bii_py_np, rtol=1e-2, atol=1e-3, show_table=True, max_edges=3)
    
    if Bij_cpp_data is not None:
        Bij_cpp = Bij_cpp_data.reshape(num_active, 6, 6)
        Bij_py_np = Bij_py[0].cpu().numpy()
        
        # Debug: Show Edge 94 (max difference edge) values
        diff_Bij = np.abs(Bij_cpp - Bij_py_np)
        max_diff_per_edge = diff_Bij.reshape(num_active, -1).max(axis=1)
        max_edge_idx = max_diff_per_edge.argmax()
        print(f"\n  🔍 Debug Edge {max_edge_idx} (max difference edge for Bij):")
        print(f"  Max diff: {max_diff_per_edge[max_edge_idx]:.6e}")
        
        # Use relaxed tolerance for Hessian blocks (matrix multiplication accumulates small errors)
        compare_tensors("Bij", Bij_cpp, Bij_py_np, rtol=1e-2, atol=1e-3, show_table=True, max_edges=3)
    
    if Eik_cpp_data is not None:
        Eik_cpp = Eik_cpp_data.reshape(num_active, 6, 1)
        Eik_py_np = Eik_py[0].cpu().numpy()
        compare_tensors("Eik", Eik_cpp, Eik_py_np, show_table=True, max_edges=2)
    
    print(f"\n  Sample Bii[0,0] (first edge):")
    print(f"    {Bii_py[0,0].cpu().numpy()}")
    
    return Bii_py, Bij_py, Bji_py, Bjj_py, Eik_py, Ejk_py

def step6_compute_gradients(wJiT_py, wJjT_py, wJzT_py, r_py_masked, num_active):
    """
    STEP 6: Compute gradients
    
    Returns:
        vi_py: Gradient w.r.t. pose i [1, num_active, 6, 1]
        vj_py: Gradient w.r.t. pose j [1, num_active, 6, 1]
        w_vec_py: Gradient w.r.t. inverse depth [1, num_active]
    """
    print("\n📊 STEP 6: Compute gradients")
    print("-" * 80)
    
    vi_py = torch.matmul(wJiT_py, r_py_masked)  # [1, num_active, 6, 1]
    vj_py = torch.matmul(wJjT_py, r_py_masked)  # [1, num_active, 6, 1]
    w_vec_py = torch.matmul(wJzT_py, r_py_masked).squeeze(dim=-1).squeeze(dim=-1)  # [1, num_active]
    
    print(f"  Python vi shape: {vi_py.shape}")
    print(f"  Python vj shape: {vj_py.shape}")
    print(f"  Python vi[0,0] (first edge): {vi_py[0,0].squeeze().cpu().numpy()}")
    
    # Compare with C++ STEP 4 gradients
    vi_cpp_data = load_binary_file("ba_step4_vi.bin")
    vj_cpp_data = load_binary_file("ba_step4_vj.bin")
    w_vec_cpp_data = load_binary_file("ba_step4_w_vec.bin")
    
    if vi_cpp_data is not None:
        vi_cpp = vi_cpp_data.reshape(num_active, 6)
        vi_py_np = vi_py[0, :, :, 0].cpu().numpy()
        compare_tensors("vi", vi_cpp, vi_py_np, show_table=True, max_edges=3)
    
    if vj_cpp_data is not None:
        vj_cpp = vj_cpp_data.reshape(num_active, 6)
        vj_py_np = vj_py[0, :, :, 0].cpu().numpy()
        compare_tensors("vj", vj_cpp, vj_py_np, show_table=True, max_edges=3)
    
    if w_vec_cpp_data is not None:
        w_vec_cpp = w_vec_cpp_data
        w_vec_py_np = w_vec_py[0].cpu().numpy()
        compare_tensors("w_vec", w_vec_cpp, w_vec_py_np, show_table=True, max_edges=3)
    
    return vi_py, vj_py, w_vec_py

def step7_fix_first_pose(ii_torch, jj_torch, fixedp=1):
    """
    STEP 7: Fix first pose (gauge freedom)
    
    Returns:
        n_py: Total number of poses
        n_adjusted_py: Number of adjustable poses
        ii_py_adjusted, jj_py_adjusted: Adjusted indices
    """
    print("\n📊 STEP 7: Fix first pose (gauge freedom)")
    print("-" * 80)
    
    n_py = max(ii_torch.max().item(), jj_torch.max().item()) + 1
    n_adjusted_py = n_py - fixedp
    
    ii_py_adjusted = ii_torch.clone() - fixedp
    jj_py_adjusted = jj_torch.clone() - fixedp
    
    print(f"  Python n={n_py}, fixedp={fixedp}, n_adjusted={n_adjusted_py}")
    print(f"  Python ii_adjusted range: [{ii_py_adjusted.min().item()}, {ii_py_adjusted.max().item()}]")
    print(f"  Python jj_adjusted range: [{jj_py_adjusted.min().item()}, {jj_py_adjusted.max().item()}]")
    
    return n_py, n_adjusted_py, ii_py_adjusted, jj_py_adjusted

def step8_reindex_structure(kk_torch):
    """
    STEP 8: Reindex structure variables
    
    Returns:
        kx_py: Unique structure variable indices
        kk_py_new: Reindexed structure variable indices
        m_py: Number of structure variables
    """
    print("\n📊 STEP 8: Reindex structure variables")
    print("-" * 80)
    
    kx_py, kk_py_new = torch.unique(kk_torch, return_inverse=True, sorted=True)
    m_py = len(kx_py)
    
    print(f"  Python m (structure vars): {m_py}")
    print(f"  Python kk_new range: [{kk_py_new.min().item()}, {kk_py_new.max().item()}]")
    
    return kx_py, kk_py_new, m_py

def step9_assemble_hessian_b(Bii_py, Bij_py, Bji_py, Bjj_py, ii_py_adjusted, jj_py_adjusted, n_adjusted_py):
    """
    STEP 9: Assemble global pose Hessian B
    
    Returns:
        B_py: Assembled Hessian [1, n_adjusted, n_adjusted, 6, 6]
    """
    print("\n📊 STEP 9: Assemble global pose Hessian B")
    print("-" * 80)
    
    from dpvo.ba import safe_scatter_add_mat
    
    # Debug: Check which edges are included by safe_scatter_add_mat
    # safe_scatter_add_mat filters: v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    v_Bii = (ii_py_adjusted >= 0) & (ii_py_adjusted >= 0) & (ii_py_adjusted < n_adjusted_py) & (ii_py_adjusted < n_adjusted_py)
    v_Bij = (ii_py_adjusted >= 0) & (jj_py_adjusted >= 0) & (ii_py_adjusted < n_adjusted_py) & (jj_py_adjusted < n_adjusted_py)
    v_Bji = (jj_py_adjusted >= 0) & (ii_py_adjusted >= 0) & (jj_py_adjusted < n_adjusted_py) & (ii_py_adjusted < n_adjusted_py)
    v_Bjj = (jj_py_adjusted >= 0) & (jj_py_adjusted >= 0) & (jj_py_adjusted < n_adjusted_py) & (jj_py_adjusted < n_adjusted_py)
    
    print(f"  Python edge filtering for B assembly:")
    print(f"    Bii: {v_Bii.sum().item()}/{len(v_Bii)} edges included")
    print(f"    Bij: {v_Bij.sum().item()}/{len(v_Bij)} edges included")
    print(f"    Bji: {v_Bji.sum().item()}/{len(v_Bji)} edges included")
    print(f"    Bjj: {v_Bjj.sum().item()}/{len(v_Bjj)} edges included")
    
    # Show edges contributing to block [0,0]
    edges_to_00_Bii = torch.where((ii_py_adjusted == 0) & (ii_py_adjusted == 0) & v_Bii)[0]
    edges_to_00_Bij = torch.where((ii_py_adjusted == 0) & (jj_py_adjusted == 0) & v_Bij)[0]
    edges_to_00_Bji = torch.where((jj_py_adjusted == 0) & (ii_py_adjusted == 0) & v_Bji)[0]
    edges_to_00_Bjj = torch.where((jj_py_adjusted == 0) & (jj_py_adjusted == 0) & v_Bjj)[0]
    
    print(f"\n  🔍 Edges contributing to Python B[0,0]:")
    print(f"    Bii: edges {edges_to_00_Bii.tolist()[:10]}")
    print(f"    Bij: edges {edges_to_00_Bij.tolist()[:10]}")
    print(f"    Bji: edges {edges_to_00_Bji.tolist()[:10]}")
    print(f"    Bjj: edges {edges_to_00_Bjj.tolist()[:10]}")
    
    B_py = (
        safe_scatter_add_mat(Bii_py, ii_py_adjusted, ii_py_adjusted, n_adjusted_py, n_adjusted_py).view(1, n_adjusted_py, n_adjusted_py, 6, 6) +
        safe_scatter_add_mat(Bij_py, ii_py_adjusted, jj_py_adjusted, n_adjusted_py, n_adjusted_py).view(1, n_adjusted_py, n_adjusted_py, 6, 6) +
        safe_scatter_add_mat(Bji_py, jj_py_adjusted, ii_py_adjusted, n_adjusted_py, n_adjusted_py).view(1, n_adjusted_py, n_adjusted_py, 6, 6) +
        safe_scatter_add_mat(Bjj_py, jj_py_adjusted, jj_py_adjusted, n_adjusted_py, n_adjusted_py).view(1, n_adjusted_py, n_adjusted_py, 6, 6)
    )
    
    print(f"  Python B shape: {B_py.shape}")
    print(f"  Python B[0,0,0] (first pose block):")
    print(f"    {B_py[0,0,0].cpu().numpy()}")
    
    # Compare with C++ STEP 9 assembled Hessian B
    B_cpp_data = load_binary_file("ba_step9_B.bin", warn_if_missing=True)
    if B_cpp_data is not None:
        B_cpp = B_cpp_data.reshape(6 * n_adjusted_py, 6 * n_adjusted_py)
        B_py_reshaped = B_py[0].permute(0, 2, 1, 3).contiguous().view(6 * n_adjusted_py, 6 * n_adjusted_py).cpu().numpy()
        
        # Debug: Find which block has the maximum difference
        diff_B = np.abs(B_cpp - B_py_reshaped)
        max_diff_per_block = np.zeros((n_adjusted_py, n_adjusted_py))
        for i in range(n_adjusted_py):
            for j in range(n_adjusted_py):
                block_diff = diff_B[6*i:6*(i+1), 6*j:6*(j+1)].max()
                max_diff_per_block[i, j] = block_diff
        
        max_block_i, max_block_j = np.unravel_index(max_diff_per_block.argmax(), max_diff_per_block.shape)
        print(f"\n  🔍 Debug: Block [{max_block_i}, {max_block_j}] has max difference: {max_diff_per_block[max_block_i, max_block_j]:.6e}")
        print(f"  C++ B block [{max_block_i}, {max_block_j}]:")
        print(f"    {B_cpp[6*max_block_i:6*(max_block_i+1), 6*max_block_j:6*(max_block_j+1)]}")
        print(f"  Python B block [{max_block_i}, {max_block_j}]:")
        print(f"    {B_py_reshaped[6*max_block_i:6*(max_block_i+1), 6*max_block_j:6*(max_block_j+1)]}")
        
        # Check which edges contribute to this block in Python
        print(f"\n  🔍 Edges contributing to block [{max_block_i}, {max_block_j}] in Python:")
        edge_indices_py = []
        for e in range(len(ii_py_adjusted)):
            # Check all four contributions: Bii[ii,ii], Bij[ii,jj], Bji[jj,ii], Bjj[jj,jj]
            if ii_py_adjusted[e].item() == max_block_i and ii_py_adjusted[e].item() == max_block_j:
                edge_indices_py.append(('Bii', e))
            if ii_py_adjusted[e].item() == max_block_i and jj_py_adjusted[e].item() == max_block_j:
                edge_indices_py.append(('Bij', e))
            if jj_py_adjusted[e].item() == max_block_i and ii_py_adjusted[e].item() == max_block_j:
                edge_indices_py.append(('Bji', e))
            if jj_py_adjusted[e].item() == max_block_i and jj_py_adjusted[e].item() == max_block_j:
                edge_indices_py.append(('Bjj', e))
        print(f"    Found {len(edge_indices_py)} contributions: {edge_indices_py[:20]}")  # Show first 20
        
        # Also check which edges contribute to block [0,0] specifically
        print(f"\n  🔍 Edges contributing to block [0, 0] in Python:")
        edge_indices_00_py = []
        for e in range(len(ii_py_adjusted)):
            if ii_py_adjusted[e].item() == 0 and ii_py_adjusted[e].item() == 0:
                edge_indices_00_py.append(('Bii', e, ii_py_adjusted[e].item(), jj_py_adjusted[e].item()))
            if ii_py_adjusted[e].item() == 0 and jj_py_adjusted[e].item() == 0:
                edge_indices_00_py.append(('Bij', e, ii_py_adjusted[e].item(), jj_py_adjusted[e].item()))
            if jj_py_adjusted[e].item() == 0 and ii_py_adjusted[e].item() == 0:
                edge_indices_00_py.append(('Bji', e, ii_py_adjusted[e].item(), jj_py_adjusted[e].item()))
            if jj_py_adjusted[e].item() == 0 and jj_py_adjusted[e].item() == 0:
                edge_indices_00_py.append(('Bjj', e, ii_py_adjusted[e].item(), jj_py_adjusted[e].item()))
        print(f"    Found {len(edge_indices_00_py)} contributions to [0,0]: {edge_indices_00_py[:20]}")
        
        # Show Python Bii/Bij/Bji/Bjj for edges that contribute to [0,0]
        if edge_indices_00_py:
            print(f"\n  🔍 Python Hessian blocks for edges contributing to [0,0]:")
            for contrib_type, e, ii_val, jj_val in edge_indices_00_py[:5]:
                if contrib_type == 'Bii':
                    print(f"    Edge {e}: Bii[{ii_val},{jj_val}] =")
                    print(f"      {Bii_py[0, e].cpu().numpy()}")
                elif contrib_type == 'Bij':
                    print(f"    Edge {e}: Bij[{ii_val},{jj_val}] =")
                    print(f"      {Bij_py[0, e].cpu().numpy()}")
                elif contrib_type == 'Bji':
                    print(f"    Edge {e}: Bji[{ii_val},{jj_val}] =")
                    print(f"      {Bji_py[0, e].cpu().numpy()}")
                elif contrib_type == 'Bjj':
                    print(f"    Edge {e}: Bjj[{ii_val},{jj_val}] =")
                    print(f"      {Bjj_py[0, e].cpu().numpy()}")
        
        compare_tensors("B (assembled)", B_cpp, B_py_reshaped)
    
    return B_py

def step10_assemble_coupling_e(Eik_py, Ejk_py, ii_py_adjusted, jj_py_adjusted, kk_py_new, n_adjusted_py, m_py):
    """
    STEP 10: Assemble pose-structure coupling E
    
    Returns:
        E_py: Pose-structure coupling [1, n_adjusted, m, 6, 1]
    """
    print("\n📊 STEP 10: Assemble pose-structure coupling E")
    print("-" * 80)
    
    from dpvo.ba import safe_scatter_add_mat
    
    E_py = (
        safe_scatter_add_mat(Eik_py, ii_py_adjusted, kk_py_new, n_adjusted_py, m_py).view(1, n_adjusted_py, m_py, 6, 1) +
        safe_scatter_add_mat(Ejk_py, jj_py_adjusted, kk_py_new, n_adjusted_py, m_py).view(1, n_adjusted_py, m_py, 6, 1)
    )
    
    print(f"  Python E shape: {E_py.shape}")
    
    # Compare with C++ STEP 10 pose-structure coupling E
    E_cpp_data = load_binary_file("ba_step10_E.bin", warn_if_missing=True)
    if E_cpp_data is not None:
        E_cpp = E_cpp_data.reshape(6 * n_adjusted_py, m_py)
        E_py_reshaped = E_py[0].permute(0, 2, 1, 3).contiguous().view(6 * n_adjusted_py, m_py).cpu().numpy()
        compare_tensors("E (pose-structure)", E_cpp, E_py_reshaped)
    
    return E_py

def step11_structure_hessian_c(wJzT_py, Jz_py, kk_py_new, m_py):
    """
    STEP 11: Structure Hessian C (diagonal)
    
    Returns:
        C_py: Structure Hessian [1, m, 1, 1]
    """
    print("\n📊 STEP 11: Structure Hessian C")
    print("-" * 80)
    
    from dpvo.ba import safe_scatter_add_vec
    
    C_py = safe_scatter_add_vec(torch.matmul(wJzT_py, Jz_py), kk_py_new, m_py)
    
    print(f"  Python C shape: {C_py.shape}")
    print(f"  Python C stats: min={C_py.min().item():.6f}, max={C_py.max().item():.6f}, mean={C_py.mean().item():.6f}")
    
    # Compare with C++ STEP 11 structure Hessian C
    C_cpp_data = load_binary_file("ba_step11_C.bin", warn_if_missing=True)
    if C_cpp_data is not None:
        C_cpp = C_cpp_data
        C_py_np = C_py[0, :, 0, 0].cpu().numpy()
        compare_tensors("C (structure Hessian)", C_cpp, C_py_np, show_table=True, max_edges=5)
    
    return C_py

def step12_assemble_gradients(vi_py, vj_py, wJzT_py, r_py_masked, ii_py_adjusted, jj_py_adjusted, kk_py_new, n_adjusted_py, m_py):
    """
    STEP 12: Assemble gradient vectors
    
    Returns:
        v_py_grad: Assembled pose gradients [1, n_adjusted, 1, 6, 1]
        w_py_grad: Assembled structure gradients [1, m, 1, 1]
    """
    print("\n📊 STEP 12: Assemble gradient vectors")
    print("-" * 80)
    
    from dpvo.ba import safe_scatter_add_vec
    
    v_py_grad = (
        safe_scatter_add_vec(vi_py, ii_py_adjusted, n_adjusted_py).view(1, n_adjusted_py, 1, 6, 1) +
        safe_scatter_add_vec(vj_py, jj_py_adjusted, n_adjusted_py).view(1, n_adjusted_py, 1, 6, 1)
    )
    
    w_py_grad = safe_scatter_add_vec(torch.matmul(wJzT_py, r_py_masked), kk_py_new, m_py)
    
    print(f"  Python v_grad shape: {v_py_grad.shape}")
    print(f"  Python w_grad shape: {w_py_grad.shape}")
    print(f"  Python v_grad[0,0] (first pose): {v_py_grad[0,0,0].squeeze().cpu().numpy()}")
    
    # Compare with C++ STEP 11 assembled gradients
    v_grad_cpp_data = load_binary_file("ba_step11_v_grad.bin", warn_if_missing=True)
    w_grad_cpp_data = load_binary_file("ba_step11_w_grad.bin", warn_if_missing=True)
    
    if v_grad_cpp_data is not None:
        v_grad_cpp = v_grad_cpp_data
        v_grad_py_np = v_py_grad[0, :, 0, :, 0].cpu().numpy()
        v_grad_py_np_flat = v_grad_py_np.flatten()
        compare_tensors("v_grad", v_grad_cpp, v_grad_py_np_flat, show_table=True, max_edges=3)
    
    if w_grad_cpp_data is not None:
        w_grad_cpp = w_grad_cpp_data
        w_grad_py_np = w_py_grad[0, :, 0, 0].cpu().numpy()
        compare_tensors("w_grad", w_grad_cpp, w_grad_py_np, show_table=True, max_edges=3)
    
    return v_py_grad, w_py_grad

def step13_levenberg_marquardt(C_py, lmbda_val=1e-4):
    """
    STEP 13: Levenberg-Marquardt damping
    
    Returns:
        Q_py: Inverse structure Hessian [1, m, 1, 1]
    """
    print("\n📊 STEP 13: Levenberg-Marquardt damping")
    print("-" * 80)
    
    if isinstance(lmbda_val, torch.Tensor):
        lmbda_py = lmbda_val.reshape(*C_py.shape)
    else:
        lmbda_py = lmbda_val
    
    Q_py = 1.0 / (C_py + lmbda_py)
    
    print(f"  Python Q shape: {Q_py.shape}")
    print(f"  Python Q stats: min={Q_py.min().item():.6f}, max={Q_py.max().item():.6f}, mean={Q_py.mean().item():.6f}")
    
    # Compare with C++ STEP 13 Q
    Q_cpp_data = load_binary_file("ba_step13_Q.bin", warn_if_missing=True)
    if Q_cpp_data is not None:
        Q_cpp = Q_cpp_data
        Q_py_np = Q_py[0, :, 0, 0].cpu().numpy()
        compare_tensors("Q", Q_cpp, Q_py_np, show_table=True, max_edges=5)
    
    return Q_py

def step14_schur_complement(B_py, E_py, Q_py, v_py_grad, w_py_grad, n_adjusted_py):
    """
    STEP 14: Schur complement
    
    Returns:
        S_py: Schur complement [1, n_adjusted, n_adjusted, 6, 6]
        y_py: RHS vector [1, n_adjusted, 1, 6, 1]
    """
    print("\n📊 STEP 14: Schur complement")
    print("-" * 80)
    
    from dpvo.ba import block_matmul
    
    EQ_py = E_py * Q_py[:, None]  # E * C^-1
    S_py = B_py - block_matmul(EQ_py, E_py.permute(0, 2, 1, 4, 3))  # B - E * C^-1 * E^T
    y_py = v_py_grad - block_matmul(EQ_py, w_py_grad.unsqueeze(dim=2))  # v - E * C^-1 * w
    
    print(f"  Python EQ shape: {EQ_py.shape}")
    print(f"  Python S shape: {S_py.shape}")
    print(f"  Python y shape: {y_py.shape}")
    print(f"  Python y.norm(): {y_py.norm().item():.6f}")
    
    # Compare with C++ STEP 14 Schur complement S and RHS y
    S_cpp_data = load_binary_file("ba_step14_S.bin", warn_if_missing=True)
    y_cpp_data = load_binary_file("ba_step14_y.bin", warn_if_missing=True)
    
    if S_cpp_data is not None:
        S_cpp = S_cpp_data.reshape(6 * n_adjusted_py, 6 * n_adjusted_py)
        S_py_reshaped = S_py[0].permute(0, 2, 1, 3).contiguous().view(6 * n_adjusted_py, 6 * n_adjusted_py).cpu().numpy()
        compare_tensors("S (Schur complement)", S_cpp, S_py_reshaped)
    
    if y_cpp_data is not None:
        y_cpp = y_cpp_data
        y_py_np = y_py[0, :, 0, :, 0].cpu().numpy()
        y_py_np_flat = y_py_np.flatten()
        compare_tensors("y (RHS)", y_cpp, y_py_np_flat)
    
    return S_py, y_py

def step15_solve_pose_increments(S_py, y_py, ep=100.0, n_adjusted_py=None):
    """
    STEP 15: Solve for pose increments
    
    Returns:
        dX_py: Pose increments (raw output from block_solve, before reshaping)
    """
    print("\n📊 STEP 15: Solve for pose increments")
    print("-" * 80)
    
    from dpvo.ba import block_solve
    
    dX_py = block_solve(S_py, y_py, ep=ep, lm=1e-4)
    
    print(f"  Python dX shape (raw): {dX_py.shape}")
    print(f"  Python dX.norm(): {dX_py.norm().item():.6f}")
    
    # Reshape for display/comparison
    dX_py_reshaped = dX_py.view(1, -1, 6)
    print(f"  Python dX shape (reshaped): {dX_py_reshaped.shape}")
    print(f"  Python dX[0,0] (first pose): {dX_py_reshaped[0,0].cpu().numpy()}")
    
    # Compare with C++ STEP 15 solution dX
    if n_adjusted_py is not None:
        dX_cpp_data = load_binary_file("ba_step15_dX.bin")
        if dX_cpp_data is not None:
            dX_cpp = dX_cpp_data
            dX_py_np = dX_py_reshaped[0].cpu().numpy()
            dX_py_np_flat = dX_py_np.flatten()
            compare_tensors("dX (solution)", dX_cpp, dX_py_np_flat, show_table=True, max_edges=3)
    
    return dX_py  # Return raw output, not reshaped

def step16_back_substitute(Q_py, w_py_grad, E_py, dX_py, n_adjusted_py, m_py):
    """
    STEP 16: Back-substitute structure increments
    
    Args:
        dX_py: Raw output from block_solve, shape [1, n, 1, 6, 1]
    
    Returns:
        dZ_py: Structure increments [1, m, 1, 1]
    """
    print("\n📊 STEP 16: Back-substitute structure increments")
    print("-" * 80)
    
    from dpvo.ba import block_matmul
    
    # Python BA: dZ = Q * (w - block_matmul(E.permute(0, 2, 1, 4, 3), dX).squeeze(dim=-1))
    # dX from block_solve has shape [1, n, 1, 6, 1]
    print(f"  dX_py shape (raw from block_solve): {dX_py.shape}")
    
    # dX should already be [1, n, 1, 6, 1] from block_solve
    # But ensure it's the right shape
    if len(dX_py.shape) == 5:
        dX_py_for_matmul = dX_py  # Already correct shape
    elif len(dX_py.shape) == 3:
        # [1, n, 6] -> [1, n, 1, 6, 1]
        dX_py_for_matmul = dX_py.view(1, n_adjusted_py, 1, 6, 1)
    else:
        raise ValueError(f"Unexpected dX_py shape: {dX_py.shape}, expected 5D [1, n, 1, 6, 1] or 3D [1, n, 6]")
    
    E_permuted = E_py.permute(0, 2, 1, 4, 3)  # [1, n, m, 6, 1] -> [1, m, n, 1, 6]
    print(f"  E_permuted shape: {E_permuted.shape}")
    print(f"  dX_py_for_matmul shape: {dX_py_for_matmul.shape}")
    
    # block_matmul([1, m, n, 1, 6], [1, n, 1, 6, 1]) -> [1, m, 1, 1, 1]
    ET_dX = block_matmul(E_permuted, dX_py_for_matmul)
    print(f"  ET_dX shape (after block_matmul): {ET_dX.shape}")
    
    # Python BA: .squeeze(dim=-1) removes the last dimension
    # ET_dX is [1, m, 1, 1, 1], after squeeze(-1) -> [1, m, 1, 1]
    ET_dX_squeezed = ET_dX.squeeze(dim=-1)
    print(f"  ET_dX_squeezed shape (after squeeze(-1)): {ET_dX_squeezed.shape}")
    
    # w_py_grad is [1, m, 1, 1], ET_dX_squeezed is [1, m, 1, 1]
    # Q_py is [1, m, 1, 1]
    print(f"  w_py_grad shape: {w_py_grad.shape}")
    print(f"  Q_py shape: {Q_py.shape}")
    
    dZ_py = Q_py * (w_py_grad - ET_dX_squeezed)
    dZ_py = dZ_py.view(1, -1, 1, 1)
    
    print(f"  Python dZ shape: {dZ_py.shape}")
    print(f"  Python dZ.norm(): {dZ_py.norm().item():.6f}")
    
    # Verify dZ has correct size
    if dZ_py.shape[1] != m_py:
        raise ValueError(f"dZ shape mismatch: expected m={m_py}, got {dZ_py.shape[1]}")
    
    # Compare with C++ STEP 16 solution dZ
    dZ_cpp_data = load_binary_file("ba_step16_dZ.bin")
    if dZ_cpp_data is not None:
        dZ_cpp = dZ_cpp_data
        dZ_py_np = dZ_py[0, :, 0, 0].cpu().numpy()
        compare_tensors("dZ (solution)", dZ_cpp, dZ_py_np, show_table=True, max_edges=5)
    
    return dZ_py

def step17_apply_updates(poses_se3, patches_torch, dX_py, dZ_py, kx_py, fixedp, n_adjusted_py):
    """
    STEP 17: Apply updates to poses and patches
    
    Returns:
        poses_py_updated: Updated poses
        patches_py_updated: Updated patches
    """
    print("\n📊 STEP 17: Apply updates")
    print("-" * 80)
    
    from dpvo.ba import pose_retr, disp_retr
    
    # Update patches
    x_py, y_py_patch, disps_py = patches_torch.unbind(dim=2)
    disps_py_updated = disp_retr(disps_py, dZ_py, kx_py).clamp(min=1e-3, max=10.0)
    patches_py_updated = torch.stack([x_py, y_py_patch, disps_py_updated], dim=2)
    
    # Update poses
    poses_py_updated = pose_retr(poses_se3, dX_py, fixedp + torch.arange(n_adjusted_py, device=device))
    
    print(f"  Python updated poses shape: {poses_py_updated.data.shape}")
    print(f"  Python updated patches shape: {patches_py_updated.shape}")
    
    return poses_py_updated, patches_py_updated

def compare_final_outputs(poses_py_updated, N):
    """
    Compare final BA outputs between C++ and Python
    
    Args:
        poses_py_updated: Python updated poses
        N: Number of poses
    """
    print("\n" + "=" * 80)
    print("COMPARING FINAL OUTPUTS")
    print("=" * 80)
    
    # Load C++ BA outputs
    poses_cpp_data = load_binary_file("ba_poses_cpp.bin")
    if poses_cpp_data is not None:
        poses_cpp_np = poses_cpp_data.reshape(N, 7)
        poses_cpp_torch = torch.from_numpy(poses_cpp_np.copy()).to(device).float()
        poses_py_torch = poses_py_updated.data[0]  # Remove batch dimension
        
        print("\n📊 Final Pose Comparison:")
        for i in range(min(3, N)):
            print(f"\n  Pose {i}:")
            t_diff = torch.abs(poses_cpp_torch[i, :3] - poses_py_torch[i, :3])
            q_diff = torch.abs(poses_cpp_torch[i, 3:] - poses_py_torch[i, 3:])
            print(f"    Translation diff: max={t_diff.max().item():.6f}, mean={t_diff.mean().item():.6f}")
            print(f"    Quaternion diff: max={q_diff.max().item():.6f}, mean={q_diff.mean().item():.6f}")
            if t_diff.max().item() > 0.01 or q_diff.max().item() > 0.01:
                print(f"    C++ t:   {poses_cpp_torch[i, :3].cpu().numpy()}")
                print(f"    Python t: {poses_py_torch[i, :3].cpu().numpy()}")
                print(f"    C++ q:   {poses_cpp_torch[i, 3:].cpu().numpy()}")
                print(f"    Python q: {poses_py_torch[i, 3:].cpu().numpy()}")

def main():
    print("=" * 80)
    print("STEP-BY-STEP BA COMPARISON: C++ vs Python")
    print("=" * 80)
    
    # Load metadata
    print("\n📁 Loading metadata...")
    M, P, N, num_active = load_metadata()
    print(f"  N={N}, M={M}, P={P}, num_active={num_active}")
    
    # Load BA inputs
    print("\n📁 Loading BA inputs...")
    try:
        poses_se3, patches_torch, intrinsics_torch, ii_torch, jj_torch, kk_torch, targets_torch, weights_torch, bounds = \
            load_ba_inputs(M, P, N, num_active)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 1
    
    print(f"\n🔍 STEP-BY-STEP COMPARISON")
    print("=" * 80)
    
    # STEP 1: Forward projection + Jacobians
    coords_py, v_py, Ji_py, Jj_py, Jz_py = step1_forward_projection(
        poses_se3, patches_torch, intrinsics_torch, ii_torch, jj_torch, kk_torch, P, num_active
    )
    
    # STEP 2: Compute residual at patch center
    r_py = step2_compute_residuals(targets_torch, coords_py, P, num_active, v_py)
    
    # STEP 3: Apply validity mask
    r_py_masked, weights_py_masked, v_py = step3_apply_validity_mask(
        r_py, coords_py, v_py, weights_torch, P, bounds
    )
    
    # STEP 4: Build weighted Jacobians
    wJiT_py, wJjT_py, wJzT_py = step4_build_weighted_jacobians(
        weights_py_masked, Ji_py, Jj_py, Jz_py, num_active
    )
    
    # STEP 5: Compute Hessian blocks
    Bii_py, Bij_py, Bji_py, Bjj_py, Eik_py, Ejk_py = step5_compute_hessian_blocks(
        wJiT_py, wJjT_py, Ji_py, Jj_py, Jz_py, num_active
    )
    
    # STEP 6: Compute gradients
    vi_py, vj_py, w_vec_py = step6_compute_gradients(
        wJiT_py, wJjT_py, wJzT_py, r_py_masked, num_active
    )
    
    # STEP 7: Fix first pose (gauge freedom)
    n_py, n_adjusted_py, ii_py_adjusted, jj_py_adjusted = step7_fix_first_pose(ii_torch, jj_torch)
    
    # STEP 8: Reindex structure variables
    kx_py, kk_py_new, m_py = step8_reindex_structure(kk_torch)
    
    # STEP 9: Assemble global pose Hessian B
    B_py = step9_assemble_hessian_b(
        Bii_py, Bij_py, Bji_py, Bjj_py, ii_py_adjusted, jj_py_adjusted, n_adjusted_py
    )
    
    # STEP 10: Assemble pose-structure coupling E
    E_py = step10_assemble_coupling_e(
        Eik_py, Ejk_py, ii_py_adjusted, jj_py_adjusted, kk_py_new, n_adjusted_py, m_py
    )
    
    # STEP 11: Structure Hessian C
    C_py = step11_structure_hessian_c(wJzT_py, Jz_py, kk_py_new, m_py)
    
    # STEP 12: Assemble gradient vectors
    v_py_grad, w_py_grad = step12_assemble_gradients(
        vi_py, vj_py, wJzT_py, r_py_masked, ii_py_adjusted, jj_py_adjusted, kk_py_new, n_adjusted_py, m_py
    )
    
    # STEP 13: Levenberg-Marquardt damping
    Q_py = step13_levenberg_marquardt(C_py)
    
    # STEP 14: Schur complement
    S_py, y_py = step14_schur_complement(B_py, E_py, Q_py, v_py_grad, w_py_grad, n_adjusted_py)
    
    # STEP 15: Solve for pose increments (returns raw output, not reshaped)
    dX_py_raw = step15_solve_pose_increments(S_py, y_py, ep=100.0, n_adjusted_py=n_adjusted_py)
    
    # STEP 16: Back-substitute structure increments (uses raw dX)
    dZ_py = step16_back_substitute(Q_py, w_py_grad, E_py, dX_py_raw, n_adjusted_py, m_py)
    
    # Reshape dX for STEP 17 (matching Python BA: dX.view(b, -1, 6))
    dX_py = dX_py_raw.view(1, -1, 6)
    
    # STEP 17: Apply updates
    poses_py_updated, patches_py_updated = step17_apply_updates(
        poses_se3, patches_torch, dX_py, dZ_py, kx_py, fixedp=1, n_adjusted_py=n_adjusted_py
    )
    
    # Compare final outputs
    compare_final_outputs(poses_py_updated, N)
    
    print("\n" + "=" * 80)
    print("STEP-BY-STEP COMPARISON COMPLETE")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
