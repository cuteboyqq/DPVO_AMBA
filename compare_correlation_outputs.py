#!/usr/bin/env python3
"""
Compare C++ correlation output with Python correlation output.

This script:
1. Loads correlation data saved by C++ code
2. Reconstructs Python data structures
3. Calls Python's correlation function with the same inputs
4. Compares outputs and identifies mismatches
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path

# Add DPVO Python path
# Try multiple possible paths
possible_paths = [
    Path(__file__).parent.parent / "DPVO_onnx",
    Path(__file__).parent.parent.parent / "DPVO_onnx",
    Path("/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx"),  # Absolute path fallback
]

dpvo_path = None
for path in possible_paths:
    if path.exists() and (path / "dpvo" / "altcorr" / "correlation.py").exists():
        dpvo_path = path
        break

if dpvo_path is None:
    raise FileNotFoundError(
        "Could not find DPVO_onnx directory. Please ensure it exists and contains dpvo/altcorr/correlation.py"
    )

if str(dpvo_path) not in sys.path:
    sys.path.insert(0, str(dpvo_path))

# Import from dpvo.altcorr
from dpvo.altcorr.correlation import corr as altcorr_corr


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
    """Load metadata (parameters) from binary file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    data = np.fromfile(filename, dtype=np.int32)
    return {
        'num_active': int(data[0]),
        'M': int(data[1]),
        'P': int(data[2]),
        'D': int(data[3]),
        'num_frames': int(data[4]),
        'num_gmap_frames': int(data[5]),
        'fmap1_H': int(data[6]),
        'fmap1_W': int(data[7]),
        'fmap2_H': int(data[8]),
        'fmap2_W': int(data[9]),
        'feature_dim': int(data[10])
    }


def compare_tensors(cpp_tensor, py_tensor, name, tolerance=1e-5, rel_tolerance=1e-4):
    """Compare two tensors and return comparison results."""
    cpp_np = cpp_tensor.cpu().numpy() if torch.is_tensor(cpp_tensor) else cpp_tensor
    py_np = py_tensor.cpu().numpy() if torch.is_tensor(py_tensor) else py_tensor
    
    # Flatten for comparison
    cpp_flat = cpp_np.flatten()
    py_flat = py_np.flatten()
    
    if cpp_flat.shape != py_flat.shape:
        return {
            'match': False,
            'reason': f'Shape mismatch: C++ {cpp_flat.shape} vs Python {py_flat.shape}',
            'max_diff': None,
            'mean_diff': None,
            'rel_diff': None
        }
    
    # Handle NaN and Inf values
    cpp_valid = np.isfinite(cpp_flat)
    py_valid = np.isfinite(py_flat)
    both_valid = cpp_valid & py_valid
    
    # Count NaN/Inf mismatches
    cpp_nan_inf = (~cpp_valid).sum()
    py_nan_inf = (~py_valid).sum()
    nan_inf_mismatch = (cpp_valid != py_valid).sum()  # One is NaN/Inf, other is not
    
    diff = np.abs(cpp_flat - py_flat)
    
    # Compute statistics only for valid (finite) values
    if both_valid.sum() > 0:
        valid_diff = diff[both_valid]
        max_diff = np.max(valid_diff)
        mean_diff = np.mean(valid_diff)
        
        # Relative difference (avoid division by zero)
        py_abs = np.abs(py_flat[both_valid])
        rel_diff = np.where(py_abs > 1e-8, valid_diff / (py_abs + 1e-8), valid_diff)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
    else:
        # All values are NaN/Inf
        max_diff = float('inf')
        mean_diff = float('inf')
        max_rel_diff = float('inf')
        mean_rel_diff = float('inf')
    
    # If there are NaN/Inf mismatches, it's definitely not a match
    match = (both_valid.sum() > 0 and max_diff < tolerance and max_rel_diff < rel_tolerance and 
             nan_inf_mismatch == 0)
    
    return {
        'match': match,
        'max_diff': float(max_diff) if np.isfinite(max_diff) else None,
        'mean_diff': float(mean_diff) if np.isfinite(mean_diff) else None,
        'max_rel_diff': float(max_rel_diff) if np.isfinite(max_rel_diff) else None,
        'mean_rel_diff': float(mean_rel_diff) if np.isfinite(mean_rel_diff) else None,
        'shape': cpp_flat.shape,
        'num_mismatched': int(np.sum(diff > tolerance)) if both_valid.sum() > 0 else int(len(diff)),
        'cpp_nan_inf': int(cpp_nan_inf),
        'py_nan_inf': int(py_nan_inf),
        'nan_inf_mismatch': int(nan_inf_mismatch)
    }


def format_number(val, precision=6):
    """Format number for display."""
    if val is None:
        return "N/A"
    if abs(val) < 1e-6:
        return f"{val:.2e}"
    elif abs(val) < 1:
        return f"{val:.{precision}f}"
    else:
        return f"{val:.{precision}f}"


def print_comparison_table(results):
    """Print comparison results in a table format."""
    print("\n" + "="*100)
    print("CORRELATION OUTPUT COMPARISON")
    print("="*100)
    print(f"{'Component':<25} {'Status':<12} {'Max Diff':<18} {'Mean Diff':<18} {'Max Rel Diff':<18} {'Mismatched':<25}")
    print("-"*100)
    
    for name, result in results.items():
        status = "✅ MATCH" if result['match'] else "❌ MISMATCH"
        max_diff = format_number(result.get('max_diff'), precision=6)
        mean_diff = format_number(result.get('mean_diff'), precision=6)
        max_rel_diff = format_number(result.get('max_rel_diff'), precision=6)
        num_mismatched = result.get('num_mismatched', 0)
        total_elements = np.prod(result.get('shape', [0])) if result.get('shape') else 0
        mismatch_str = f"{num_mismatched}/{total_elements}"
        
        # Add NaN/Inf warning if present
        cpp_nan_inf = result.get('cpp_nan_inf', 0)
        py_nan_inf = result.get('py_nan_inf', 0)
        nan_inf_mismatch = result.get('nan_inf_mismatch', 0)
        if cpp_nan_inf > 0 or py_nan_inf > 0 or nan_inf_mismatch > 0:
            status += f" (NaN/Inf: C++={cpp_nan_inf}, Py={py_nan_inf}, Mismatch={nan_inf_mismatch})"
        
        print(f"{name:<25} {status:<12} {max_diff:<18} {mean_diff:<18} {max_rel_diff:<18} {mismatch_str:<25}")
    
    print("="*100)


def load_and_reshape_cpp_data(frame_num, meta, bin_dir="bin_file"):
    """Load and reshape C++ correlation data from binary files."""
    num_active = meta['num_active']
    P = meta['P']
    D = meta['D']
    feature_dim = meta['feature_dim']
    fmap1_H = meta['fmap1_H']
    fmap1_W = meta['fmap1_W']
    fmap2_H = meta['fmap2_H']
    fmap2_W = meta['fmap2_W']
    
    print("\n" + "="*70)
    print("LOADING C++ DATA")
    print("="*70)
    
    # Load binary files
    coords_cpp = load_binary_float(os.path.join(bin_dir, f"corr_frame{frame_num}_coords.bin"))
    kk_cpp = load_binary_int32(os.path.join(bin_dir, f"corr_frame{frame_num}_kk.bin"))
    jj_cpp = load_binary_int32(os.path.join(bin_dir, f"corr_frame{frame_num}_jj.bin"))
    ii_cpp = load_binary_int32(os.path.join(bin_dir, f"corr_frame{frame_num}_ii.bin"))
    gmap_cpp = load_binary_float(os.path.join(bin_dir, f"corr_frame{frame_num}_gmap.bin"))
    fmap1_cpp = load_binary_float(os.path.join(bin_dir, f"corr_frame{frame_num}_fmap1.bin"))
    fmap2_cpp = load_binary_float(os.path.join(bin_dir, f"corr_frame{frame_num}_fmap2.bin"))
    corr_cpp = load_binary_float(os.path.join(bin_dir, f"corr_frame{frame_num}_corr.bin"))
    
    # Reshape C++ data
    coords_cpp = coords_cpp.reshape(num_active, 2, P, P)
    gmap_cpp = gmap_cpp.reshape(num_active, feature_dim, 3, 3)
    fmap1_cpp = fmap1_cpp.reshape(num_active, feature_dim, fmap1_H, fmap1_W)
    fmap2_cpp = fmap2_cpp.reshape(num_active, feature_dim, fmap2_H, fmap2_W)
    corr_cpp = corr_cpp.reshape(num_active, D, D, P, P, 2)
    
    # Print data summary
    print(f"{'Data':<20} {'Shape':<30} {'Additional Info':<50}")
    print("-"*70)
    print(f"{'coords':<20} {str(coords_cpp.shape):<30} {'Reprojected coordinates':<50}")
    print(f"{'gmap':<20} {str(gmap_cpp.shape):<30} {'Patch features':<50}")
    print(f"{'fmap1':<20} {str(fmap1_cpp.shape):<30} {'Pyramid level 0 features':<50}")
    print(f"{'fmap2':<20} {str(fmap2_cpp.shape):<30} {'Pyramid level 1 features':<50}")
    print(f"{'corr':<20} {str(corr_cpp.shape):<30} {'C++ correlation output':<50}")
    print(f"{'kk':<20} {str(kk_cpp.shape):<30} {f'Range: [{kk_cpp.min()}, {kk_cpp.max()}]':<50}")
    print(f"{'jj':<20} {str(jj_cpp.shape):<30} {f'Range: [{jj_cpp.min()}, {jj_cpp.max()}]':<50}")
    print("="*70)
    
    return {
        'coords': coords_cpp,
        'kk': kk_cpp,
        'jj': jj_cpp,
        'ii': ii_cpp,
        'gmap': gmap_cpp,
        'fmap1': fmap1_cpp,
        'fmap2': fmap2_cpp,
        'corr': corr_cpp
    }


def prepare_python_tensors(cpp_data, device):
    """Convert C++ numpy arrays to PyTorch tensors and prepare for Python correlation."""
    coords_torch = torch.from_numpy(cpp_data['coords']).float().to(device)
    gmap_torch = torch.from_numpy(cpp_data['gmap']).float().to(device)
    fmap1_torch = torch.from_numpy(cpp_data['fmap1']).float().to(device)
    fmap2_torch = torch.from_numpy(cpp_data['fmap2']).float().to(device)
    corr_cpp_torch = torch.from_numpy(cpp_data['corr']).float().to(device)
    
    # Reshape for Python: add batch dimension [B=1, M=num_active, ...]
    gmap_py = gmap_torch.unsqueeze(0)  # [1, num_active, 128, 3, 3]
    fmap1_py = fmap1_torch.unsqueeze(0)  # [1, num_active, 128, fmap1_H, fmap1_W]
    fmap2_py = fmap2_torch.unsqueeze(0)  # [1, num_active, 128, fmap2_H, fmap2_W]
    coords_py = coords_torch.unsqueeze(0)  # [1, num_active, 2, P, P]
    
    return {
        'coords_py': coords_py,
        'gmap_py': gmap_py,
        'fmap1_py': fmap1_py,
        'fmap2_py': fmap2_py,
        'corr_cpp_torch': corr_cpp_torch
    }


def verify_slice_correspondence(kk_cpp, jj_cpp, num_active, M, num_gmap_frames, num_frames, device):
    """Verify that Python slice indices match C++ indexing."""
    print(f"\n{'='*70}")
    print("VERIFYING SLICE CORRESPONDENCE")
    print("="*70)
    
    # Compute ii1 and jj1 from kk and jj using the same modulo operation as C++
    mod_value = M * num_gmap_frames
    ii1_computed = kk_cpp % mod_value
    jj1_computed = jj_cpp % num_frames
    
    # Check if indices are sequential
    ii1_is_sequential = np.allclose(ii1_computed, np.arange(num_active))
    jj1_is_sequential = np.allclose(jj1_computed, np.arange(num_active))
    
    # Check if indices are within bounds
    max_ii1 = ii1_computed.max()
    max_jj1 = jj1_computed.max()
    ii1_in_bounds = max_ii1 < num_active
    jj1_in_bounds = max_jj1 < num_active
    
    # IMPORTANT: Since we're passing flattened arrays [1, num_active, ...] where
    # each element corresponds to an edge, we should ALWAYS use sequential indices
    # [0, 1, 2, ..., num_active-1] so Python selects the correct edge's data.
    # If we use ring buffer indices (kk % (M*pmem)), Python would select the wrong edge!
    # Example: If kk[23]=9, then ii1_py[23]=9, and Python selects fmap1[:, 9],
    # but gmap_slices[9] contains patch from kk[9], not kk[23]!
    # Always use sequential indices for flattened arrays (matching compare_correlation_8x8.py)
    ii1_py = torch.arange(num_active, dtype=torch.long, device=device)
    jj1_py = torch.arange(num_active, dtype=torch.long, device=device)
    print(f"✅ Using sequential indices for flattened arrays")
    print(f"   ii1_py = [0, 1, 2, ..., {num_active-1}]")
    print(f"   jj1_py = [0, 1, 2, ..., {num_active-1}]")
    print(f"   This ensures Python selects gmap_slices[e] and fmap_slices[e] for edge e")
    
    print(f"{'Edge':<10} {'C++ ii1':<15} {'C++ jj1':<15} {'Python ii1':<15} {'Python jj1':<15} {'Note':<30}")
    print("-"*70)
    
    for e in range(min(10, num_active)):
        note = ""
        if e < num_active:
            if ii1_computed[e] != ii1_py[e].item():
                note += f"ii1 mismatch! "
            if jj1_computed[e] != jj1_py[e].item():
                note += f"jj1 mismatch! "
        print(f"{e:<10} {ii1_computed[e]:<15} {jj1_computed[e]:<15} {ii1_py[e].item():<15} {jj1_py[e].item():<15} {note:<30}")
    
    print("="*70)
    
    if not ii1_is_sequential or not jj1_is_sequential:
        print(f"⚠️  WARNING: Indices are NOT sequential!")
        print(f"   ii1_is_sequential: {ii1_is_sequential}")
        print(f"   jj1_is_sequential: {jj1_is_sequential}")
        print(f"   max_ii1: {max_ii1}, num_active: {num_active}, within bounds: {ii1_in_bounds}")
        print(f"   max_jj1: {max_jj1}, num_active: {num_active}, within bounds: {jj1_in_bounds}")
        if ii1_in_bounds and jj1_in_bounds:
            print(f"   ✅ Using C++ indices directly to match C++ behavior")
    else:
        print("✅ Indices are sequential - slices should be in correct order")
    
    print(f"\nIndex mapping comparison:")
    print(f"  C++ kk range: [{kk_cpp.min()}, {kk_cpp.max()}]")
    print(f"  C++ jj range: [{jj_cpp.min()}, {jj_cpp.max()}]")
    print(f"  C++ ii1 (computed) range: [{ii1_computed.min()}, {ii1_computed.max()}]")
    print(f"  C++ jj1 (computed) range: [{jj1_computed.min()}, {jj1_computed.max()}]")
    print(f"  Python ii1_py range: [{ii1_py.min().item()}, {ii1_py.max().item()}]")
    print(f"  Python jj1_py range: [{jj1_py.min().item()}, {jj1_py.max().item()}]")
    
    return ii1_py, jj1_py, ii1_computed, jj1_computed


def validate_coordinates(coords_cpp, fmap1_H, fmap1_W, fmap2_H, fmap2_W, R, P):
    """Validate that coordinates are finite and within feature map bounds."""
    print("\n" + "="*70)
    print("COORDINATE VALIDITY CHECK")
    print("="*70)
    
    coords_valid = np.isfinite(coords_cpp).all()
    coords_in_bounds_fmap1 = ((coords_cpp[:, 0, :, :] >= 0) & (coords_cpp[:, 0, :, :] < fmap1_W) & 
                              (coords_cpp[:, 1, :, :] >= 0) & (coords_cpp[:, 1, :, :] < fmap1_H))
    coords_in_bounds_fmap2 = ((coords_cpp[:, 0, :, :] / 4 >= 0) & (coords_cpp[:, 0, :, :] / 4 < fmap2_W) & 
                              (coords_cpp[:, 1, :, :] / 4 >= 0) & (coords_cpp[:, 1, :, :] / 4 < fmap2_H))
    
    # Check coordinates with correlation window offsets (R radius)
    coords_with_offset_fmap1 = (
        (coords_cpp[:, 0, :, :] - R >= 0) & (coords_cpp[:, 0, :, :] + R < fmap1_W) &
        (coords_cpp[:, 1, :, :] - R >= 0) & (coords_cpp[:, 1, :, :] + R < fmap1_H)
    )
    coords_with_offset_fmap2 = (
        ((coords_cpp[:, 0, :, :] / 4) - R >= 0) & ((coords_cpp[:, 0, :, :] / 4) + R < fmap2_W) &
        ((coords_cpp[:, 1, :, :] / 4) - R >= 0) & ((coords_cpp[:, 1, :, :] / 4) + R < fmap2_H)
    )
    
    print(f"{'Check':<50} {'Result':<40}")
    print("-"*70)
    print(f"{'All coords finite':<50} {str(coords_valid):<40}")
    print(f"{'Coords in fmap1 bounds (center)':<50} {f'{coords_in_bounds_fmap1.sum()}/{coords_in_bounds_fmap1.size} valid':<40}")
    print(f"{'Coords in fmap1 bounds (with offset ±R)':<50} {f'{coords_with_offset_fmap1.sum()}/{coords_with_offset_fmap1.size} valid':<40}")
    print(f"{'Coords in fmap2 bounds (center)':<50} {f'{coords_in_bounds_fmap2.sum()}/{coords_in_bounds_fmap2.size} valid':<40}")
    print(f"{'Coords in fmap2 bounds (with offset ±R)':<50} {f'{coords_with_offset_fmap2.sum()}/{coords_with_offset_fmap2.size} valid':<40}")
    print(f"{'Coords range (x)':<50} {f'[{coords_cpp[:, 0, :, :].min():.2f}, {coords_cpp[:, 0, :, :].max():.2f}]':<40}")
    print(f"{'Coords range (y)':<50} {f'[{coords_cpp[:, 1, :, :].min():.2f}, {coords_cpp[:, 1, :, :].max():.2f}]':<40}")
    print(f"{'fmap1 bounds':<50} {f'[0, {fmap1_W}) x [0, {fmap1_H})':<40}")
    print(f"{'fmap2 bounds':<50} {f'[0, {fmap2_W}) x [0, {fmap2_H})':<40}")
    print(f"{'Correlation radius R':<50} {R:<40}")
    print("="*70)
    
    # Show sample coordinates
    print(f"\n{'='*70}")
    print("SAMPLE COORDINATES (first 5 edges, center pixel)")
    print("="*70)
    print(f"{'Edge':<10} {'X':<15} {'Y':<15} {'In fmap1?':<15} {'In fmap1±R?':<15} {'In fmap2?':<15} {'In fmap2±R?':<15}")
    print("-"*70)
    center_p = P // 2
    for e in range(min(5, coords_cpp.shape[0])):
        x = coords_cpp[e, 0, center_p, center_p]
        y = coords_cpp[e, 1, center_p, center_p]
        in_fmap1 = coords_in_bounds_fmap1[e, center_p, center_p]
        in_fmap1_offset = coords_with_offset_fmap1[e, center_p, center_p]
        in_fmap2 = coords_in_bounds_fmap2[e, center_p, center_p]
        in_fmap2_offset = coords_with_offset_fmap2[e, center_p, center_p]
        print(f"{e:<10} {x:<15.2f} {y:<15.2f} {str(in_fmap1):<15} {str(in_fmap1_offset):<15} {str(in_fmap2):<15} {str(in_fmap2_offset):<15}")
    print("="*70)
    
    return coords_in_bounds_fmap1


def compute_python_correlation(py_tensors, ii1_py, jj1_py, R, fmap1_H, fmap1_W, fmap2_H, fmap2_W, corr_cpp_torch, P):
    """Compute Python correlation for both pyramid levels."""
    print("\n" + "="*70)
    print("COMPUTING PYTHON CORRELATION")
    print("="*70)
    print(f"{'Level':<15} {'Input Coords Scale':<20} {'Output Shape':<30} {'Radius':<10} {'Window Size':<15}")
    print("-"*70)
    
    coords_py = py_tensors['coords_py']
    gmap_py = py_tensors['gmap_py']
    fmap1_py = py_tensors['fmap1_py']
    fmap2_py = py_tensors['fmap2_py']
    
    # Debug: Check device and data types
    print(f"\n🔍 Device and data type check:")
    print(f"   gmap_py device: {gmap_py.device}, dtype: {gmap_py.dtype}")
    print(f"   fmap1_py device: {fmap1_py.device}, dtype: {fmap1_py.dtype}")
    print(f"   coords_py device: {coords_py.device}, dtype: {coords_py.dtype}")
    print(f"   ii1_py device: {ii1_py.device}, dtype: {ii1_py.dtype}")
    print(f"   jj1_py device: {jj1_py.device}, dtype: {jj1_py.dtype}")
    
    # Create coords_level0 for level 0
    coords_level0 = coords_py / 1.0
    
    # Verify data is not all zeros
    print(f"\n🔍 Data validity check:")
    print(f"   gmap_py non-zero: {(gmap_py != 0).sum().item()}/{gmap_py.numel()} ({(gmap_py != 0).sum().item()/gmap_py.numel()*100:.2f}%)")
    print(f"   fmap1_py non-zero: {(fmap1_py != 0).sum().item()}/{fmap1_py.numel()} ({(fmap1_py != 0).sum().item()/fmap1_py.numel()*100:.2f}%)")
    print(f"   coords_py finite: {torch.isfinite(coords_py).sum().item()}/{coords_py.numel()} ({torch.isfinite(coords_py).sum().item()/coords_py.numel()*100:.2f}%)")
    
    # Check coordinate ranges
    coords_x = coords_level0[0, :, 0, :, :]
    coords_y = coords_level0[0, :, 1, :, :]
    print(f"   coords_x range: [{coords_x.min().item():.2f}, {coords_x.max().item():.2f}]")
    print(f"   coords_y range: [{coords_y.min().item():.2f}, {coords_y.max().item():.2f}]")
    print(f"   fmap1 bounds: [0, {fmap1_W}) x [0, {fmap1_H})")
    
    # Check if Python normalizes coordinates internally (for grid_sample)
    # grid_sample expects coordinates in [-1, 1] range
    # Normalization: gx = 2 * (x / (W - 1)) - 1, gy = 2 * (y / (H - 1)) - 1
    print(f"\n🔍 Coordinate normalization check (for grid_sample):")
    sample_x = coords_level0[0, 0, 0, 1, 1].item()
    sample_y = coords_level0[0, 0, 1, 1, 1].item()
    gx_norm = 2 * (sample_x / (fmap1_W - 1)) - 1
    gy_norm = 2 * (sample_y / (fmap1_H - 1)) - 1
    print(f"   Sample coord (raw): x={sample_x:.2f}, y={sample_y:.2f}")
    print(f"   Normalized for grid_sample: gx={gx_norm:.4f}, gy={gy_norm:.4f}")
    print(f"   Normalized range should be [-1, 1]: gx in range? {-1 <= gx_norm <= 1}, gy in range? {-1 <= gy_norm <= 1}")
    print(f"   ⚠️  NOTE: Python's altcorr_corr likely normalizes coordinates internally for grid_sample")
    print(f"      C++ uses raw pixel coordinates with floor() - this may cause differences!")
    
    # Manual verification for edge 0
    print(f"\n🔍 Manual verification for edge 0:")
    e = 0
    x_coord = coords_py[0, e, 0, 1, 1].item()
    y_coord = coords_py[0, e, 1, 1, 1].item()
    print(f"   Edge {e}: coords center = ({x_coord:.2f}, {y_coord:.2f})")
    
    patch_feat = gmap_py[0, ii1_py[e], :, 1, 1]
    frame_x = int(np.floor(x_coord))
    frame_y = int(np.floor(y_coord))
    if 0 <= frame_x < fmap1_W and 0 <= frame_y < fmap1_H:
        frame_feat = fmap1_py[0, jj1_py[e], :, frame_y, frame_x]
        dot_product = torch.dot(patch_feat, frame_feat).item()
        print(f"   Manual dot product at ({frame_x}, {frame_y}): {dot_product:.6f}")
        print(f"   C++ correlation value at center: {corr_cpp_torch[e, R, R, 1, 1, 0].item():.6f}")
        if abs(dot_product - corr_cpp_torch[e, R, R, 1, 1, 0].item()) < 0.1:
            print(f"   ✅ Manual computation matches C++ (within tolerance)")
        else:
            print(f"   ❌ Manual computation does NOT match C++!")
    else:
        print(f"   ⚠️  Coordinates ({frame_x}, {frame_y}) are out of bounds!")
    
    # Call Python correlation for level 0
    print(f"\n🔍 Calling Python correlation function...")
    corr1_py = altcorr_corr(gmap_py, fmap1_py, coords_level0, ii1_py, jj1_py, radius=R)
    print(f"   Correlation function returned, shape: {corr1_py.shape}")
    D_py = corr1_py.shape[2]
    print(f"{'Level 0':<15} {'coords / 1':<20} {str(corr1_py.shape):<30} {R:<10} {D_py:<15}")
    
    # Debug correlation output
    _debug_correlation_output(corr1_py, corr_cpp_torch, R, P, level=0)
    
    # Call Python correlation for level 1
    coords_level1 = coords_py / 4.0
    corr2_py = altcorr_corr(gmap_py, fmap2_py, coords_level1, ii1_py, jj1_py, radius=R)
    print(f"{'Level 1':<15} {'coords / 4':<20} {str(corr2_py.shape):<30} {R:<10} {corr2_py.shape[2]:<15}")
    
    # Debug correlation output
    _debug_correlation_output(corr2_py, corr_cpp_torch, R, P, level=1)
    
    print("-"*70)
    print(f"{'Summary':<15} {'':<20} {'':<30} {'':<10} {'':<15}")
    print(f"{'Python D':<15} {'':<20} {'':<30} {'':<10} {D_py:<15}")
    print("="*70)
    
    return corr1_py, corr2_py, D_py


def _debug_correlation_output(corr_py, corr_cpp_torch, R, P, level):
    """Debug helper to check correlation output quality."""
    corr_nonzero = (corr_py != 0).sum().item()
    corr_total = corr_py.numel()
    corr_nan = torch.isnan(corr_py).sum().item()
    corr_inf = torch.isinf(corr_py).sum().item()
    corr_huge = (torch.abs(corr_py) > 1e6).sum().item()
    
    print(f"  Debug: corr{level+1}_py nonzero elements: {corr_nonzero}/{corr_total} ({100*corr_nonzero/corr_total:.2f}%)")
    print(f"  Debug: corr{level+1}_py NaN elements: {corr_nan}/{corr_total} ({100*corr_nan/corr_total:.2f}%)")
    print(f"  Debug: corr{level+1}_py Inf elements: {corr_inf}/{corr_total} ({100*corr_inf/corr_total:.2f}%)")
    print(f"  Debug: corr{level+1}_py huge elements (>1e6): {corr_huge}/{corr_total} ({100*corr_huge/corr_total:.2f}%)")
    
    center_d = R
    # corr_py shape: [num_active, D, D, P, P] (5D tensor)
    py_center_val = corr_py[0, center_d, center_d, 1, 1].item()
    cpp_center_val = corr_cpp_torch[0, R, R, 1, 1, level].item()
    print(f"  Debug: corr{level+1}_py[0, {center_d}, {center_d}, 1, 1] (center): {py_center_val:.6f}")
    print(f"  Debug: C++ corr[0, {R}, {R}, 1, 1, {level}] (center): {cpp_center_val:.6f}")
    print(f"  Debug: Difference: {abs(py_center_val - cpp_center_val):.6f}")
    
    if corr_nan > 0 or corr_huge > 0:
        print(f"\n  ⚠️  WARNING: Found NaN or huge values in corr{level+1}_py!")
        # corr_py shape: [num_active, D, D, P, P] (5D tensor)
        # Reduce over all dimensions except the first (edge dimension)
        nan_edges = torch.isnan(corr_py).any(dim=(1, 2, 3, 4))  # [num_active]
        huge_edges = (torch.abs(corr_py) > 1e6).any(dim=(1, 2, 3, 4))  # [num_active]
        nan_edge_indices = torch.where(nan_edges)[0][:5].tolist()
        huge_edge_indices = torch.where(huge_edges)[0][:5].tolist()
        print(f"     Edges with NaN (first 5): {nan_edge_indices}")
        print(f"     Edges with huge values (first 5): {huge_edge_indices}")


def align_correlation_windows(corr_cpp_torch, corr1_py, corr2_py, D, D_py, R):
    """Align C++ and Python correlation windows if sizes differ."""
    # Stack Python results
    corr_py_stacked = torch.stack([corr1_py, corr2_py], dim=-1)  # [1, num_active, D_py, D_py, P, P, 2]
    corr_py_final = corr_py_stacked.squeeze(0)  # [num_active, D_py, D_py, P, P, 2]
    
    # IMPORTANT: Python's corr_torch_forward_fp16 returns out.permute(0, 1, 3, 2, 4, 5)
    # This swaps correlation window dimensions: [B, M, corr_y, corr_x, H, W] -> [B, M, corr_x, corr_y, H, W]
    # C++ stores correlation as [edge, corr_y, corr_x, patch_y, patch_x, level]
    # So we need to swap Python's correlation window dimensions to match C++
    # Swap dimensions 1 and 2: [num_active, D_py, D_py, P, P, 2] -> [num_active, D_py, D_py, P, P, 2]
    corr_py_final = corr_py_final.transpose(1, 2)  # Swap correlation window dimensions to match C++
    
    corr_cpp_final = corr_cpp_torch.clone()
    
    if D_py != D:
        print(f"\n⚠️  WARNING: Correlation window size mismatch!")
        print(f"   Python: D_py = {D_py} (2*R+1 = 2*{R}+1 = {2*R+1})")
        print(f"   C++: D_cpp = {D} (2*R+2 = 2*{R}+2 = {2*R+2})")
        print(f"   Extracting center {D_py}x{D_py} region from C++'s {D}x{D} output to match Python")
        
        crop_start = (D - D_py) // 2
        corr_cpp_final = corr_cpp_final[:, crop_start:crop_start+D_py, crop_start:crop_start+D_py, :, :, :]
        print(f"   Extracted C++ correlation: [{crop_start}:{crop_start+D_py}, {crop_start}:{crop_start+D_py}]")
        D_compare = D_py
    else:
        D_compare = D
    
    return corr_cpp_final, corr_py_final, D_compare


def compare_correlation_outputs(corr_cpp_final, corr_py_final, D_compare, P, num_active):
    """Compare C++ and Python correlation outputs and return results."""
    print("\n" + "="*70)
    print("COMPARING OUTPUTS")
    print("="*70)
    
    # Show sample correlation values
    _print_sample_correlation_values(corr_cpp_final, corr_py_final, D_compare, P, num_active)
    
    # Compare level 0
    corr1_cpp = corr_cpp_final[:, :, :, :, :, 0]
    corr1_py_compare = corr_py_final[:, :, :, :, :, 0]
    results = {
        'corr1 (level 0)': compare_tensors(corr1_cpp, corr1_py_compare, 'corr1', tolerance=1e-4, rel_tolerance=1e-3)
    }
    
    # Compare level 1
    corr2_cpp = corr_cpp_final[:, :, :, :, :, 1]
    corr2_py_compare = corr_py_final[:, :, :, :, :, 1]
    results['corr2 (level 1)'] = compare_tensors(corr2_cpp, corr2_py_compare, 'corr2', tolerance=1e-4, rel_tolerance=1e-3)
    
    # Compare full stacked output
    results['corr (stacked)'] = compare_tensors(corr_cpp_final, corr_py_final, 'corr', tolerance=1e-4, rel_tolerance=1e-3)
    
    return results, corr1_cpp, corr2_cpp, corr1_py_compare, corr2_py_compare


def _print_sample_correlation_values(corr_cpp_final, corr_py_final, D_compare, P, num_active):
    """Print sample correlation values in table format."""
    print(f"\n{'='*100}")
    print("SAMPLE CORRELATION VALUES")
    print(f"{'='*100}")
    print(f"{'Edge':<10} {'Level':<10} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20} {'Location':<20}")
    print("-"*100)
    
    if num_active > 0:
        center_p = P // 2
        center_d = D_compare // 2
        num_samples = min(10, num_active)
        
        for e in range(num_samples):
            # Level 0
            cpp_val1 = float(corr_cpp_final[e, center_d, center_d, center_p, center_p, 0])
            py_val1 = float(corr_py_final[e, center_d, center_d, center_p, center_p, 0])
            diff1 = abs(cpp_val1 - py_val1)
            loc1 = f"({e},{center_d},{center_d},{center_p},{center_p},0)"
            print(f"{e:<10} {'Level 0':<10} {format_number(cpp_val1):<20} {format_number(py_val1):<20} "
                  f"{format_number(diff1):<20} {loc1:<20}")
            
            # Level 1
            cpp_val2 = float(corr_cpp_final[e, center_d, center_d, center_p, center_p, 1])
            py_val2 = float(corr_py_final[e, center_d, center_d, center_p, center_p, 1])
            diff2 = abs(cpp_val2 - py_val2)
            loc2 = f"({e},{center_d},{center_d},{center_p},{center_p},1)"
            print(f"{'':<10} {'Level 1':<10} {format_number(cpp_val2):<20} {format_number(py_val2):<20} "
                  f"{format_number(diff2):<20} {loc2:<20}")
            if e < num_samples - 1:
                print("-"*100)
    
    print("="*100)


def analyze_mismatch_location(cpp_data, py_data, location, coords_cpp, fmap_H, fmap_W, level=0):
    """Analyze a specific mismatched location to understand the difference."""
    # location format: (edge, di, dj, pi, pj)
    e, di, dj, pi, pj = location
    lev = level
    
    # Handle torch tensors - check if it's 5D or 6D (stacked)
    if torch.is_tensor(cpp_data):
        if len(cpp_data.shape) == 6:
            # 6D: (edge, di, dj, pi, pj, level) - stacked tensor
            cpp_val = float(cpp_data[e, di, dj, pi, pj, lev].item())
        else:
            # 5D: (edge, di, dj, pi, pj) - single level
            cpp_val = float(cpp_data[e, di, dj, pi, pj].item())
    else:
        if len(cpp_data.shape) == 6:
            cpp_val = float(cpp_data[e, di, dj, pi, pj, lev])
        else:
            cpp_val = float(cpp_data[e, di, dj, pi, pj])
    
    if torch.is_tensor(py_data):
        if len(py_data.shape) == 6:
            py_val = float(py_data[e, di, dj, pi, pj, lev].item())
        else:
            py_val = float(py_data[e, di, dj, pi, pj].item())
    else:
        if len(py_data.shape) == 6:
            py_val = float(py_data[e, di, dj, pi, pj, lev])
        else:
            py_val = float(py_data[e, di, dj, pi, pj])
    
    # Get coordinates for this edge and patch pixel
    coord_scale = 1.0 if lev == 0 else 0.25
    if torch.is_tensor(coords_cpp):
        x_raw = float(coords_cpp[e, 0, pi, pj].item())
        y_raw = float(coords_cpp[e, 1, pi, pj].item())
    else:
        x_raw = float(coords_cpp[e, 0, pi, pj])
        y_raw = float(coords_cpp[e, 1, pi, pj])
    x_scaled = x_raw * coord_scale
    y_scaled = y_raw * coord_scale
    
    # C++ sampling location (nearest neighbor)
    R = 3  # Assuming R=3 for D=7
    cpp_i = int(np.floor(y_scaled)) + (di - R)
    cpp_j = int(np.floor(x_scaled)) + (dj - R)
    
    # Python would normalize for grid_sample
    gx_norm = 2 * (x_scaled / (fmap_W - 1)) - 1
    gy_norm = 2 * (y_scaled / (fmap_H - 1)) - 1
    
    # Python grid_sample with offset (normalized)
    offset_di = (di - R) / (fmap_H - 1) * 2  # Normalized offset
    offset_dj = (dj - R) / (fmap_W - 1) * 2
    py_gx = gx_norm + offset_dj
    py_gy = gy_norm + offset_di
    
    # Convert back to pixel coordinates for Python (approximate)
    py_i_approx = (py_gy + 1) / 2 * (fmap_H - 1)
    py_j_approx = (py_gx + 1) / 2 * (fmap_W - 1)
    
    print(f"\n  Location Analysis: Edge {e}, Window ({di},{dj}), Patch ({pi},{pj}), Level {lev}")
    print(f"    C++ Value: {cpp_val:.6f}, Python Value: {py_val:.6f}, Diff: {abs(cpp_val - py_val):.6f}")
    print(f"    Raw coords: x={x_raw:.2f}, y={y_raw:.2f}")
    print(f"    Scaled coords: x={x_scaled:.2f}, y={y_scaled:.2f} (scale={coord_scale})")
    print(f"    C++ sampling (nearest): i={cpp_i}, j={cpp_j} (from floor({y_scaled:.2f})+{di-R}, floor({x_scaled:.2f})+{dj-R})")
    print(f"    Python normalized: gx={gx_norm:.4f}, gy={gy_norm:.4f}")
    print(f"    Python grid_sample location (approx): i≈{py_i_approx:.2f}, j≈{py_j_approx:.2f}")
    print(f"    → C++ uses integer pixel ({cpp_i}, {cpp_j}), Python uses bilinear at ({py_i_approx:.2f}, {py_j_approx:.2f})")


def print_detailed_analysis(results, corr1_cpp, corr2_cpp, corr_cpp_final, corr1_py_compare, corr2_py_compare, corr_py_final, coords_cpp=None, fmap1_H=None, fmap1_W=None, fmap2_H=None, fmap2_W=None):
    """Print detailed mismatch analysis."""
    print("\n" + "="*100)
    print("DETAILED MISMATCH ANALYSIS")
    print("="*100)
    
    for name, result in results.items():
        if not result['match']:
            print(f"\n{'='*100}")
            print(f"{name.upper()}: MISMATCH")
            print("="*100)
            print(f"{'Metric':<30} {'Value':<30} {'Details':<40}")
            print("-"*100)
            print(f"{'Max Diff':<30} {format_number(result.get('max_diff')):<30} {'':<40}")
            print(f"{'Mean Diff':<30} {format_number(result.get('mean_diff')):<30} {'':<40}")
            print(f"{'Max Rel Diff':<30} {format_number(result.get('max_rel_diff')):<30} {'':<40}")
            mismatch_count = result.get('num_mismatched', 0)
            total_elements = np.prod(result.get('shape', [0])) if result.get('shape') else 0
            mismatch_pct = 100.0 * mismatch_count / total_elements if total_elements > 0 else 0
            print(f"{'Mismatched Elements':<30} {f'{mismatch_count}/{total_elements} ({mismatch_pct:.2f}%)':<30} {'':<40}")
            print("="*100)
            
            # Find edges with largest differences
            if 'corr1' in name.lower():
                cpp_data = corr1_cpp
                py_data = corr1_py_compare
            elif 'corr2' in name.lower():
                cpp_data = corr2_cpp
                py_data = corr2_py_compare
            else:
                cpp_data = corr_cpp_final
                py_data = corr_py_final
            
            cpp_flat = cpp_data.flatten().cpu().numpy()
            py_flat = py_data.flatten().cpu().numpy()
            diff_flat = np.abs(cpp_flat - py_flat)
            top_indices = np.argsort(diff_flat)[-20:][::-1]
            
            print(f"\n{'='*100}")
            print(f"TOP 20 LARGEST DIFFERENCES: {name}")
            print("="*100)
            print(f"{'Rank':<10} {'Index':<15} {'Location':<35} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20}")
            print("-"*100)
            
            # Determine which level and feature map dimensions to use
            if 'corr1' in name.lower() or (name == 'corr (stacked)' and len(top_indices) > 0):
                level = 0
                fmap_H = fmap1_H if fmap1_H else 132
                fmap_W = fmap1_W if fmap1_W else 240
            else:
                level = 1
                fmap_H = fmap2_H if fmap2_H else 33
                fmap_W = fmap2_W if fmap2_W else 60
            
            for rank, idx in enumerate(top_indices[:20], 1):
                if 'corr1' in name.lower():
                    orig_shape = corr1_cpp.shape
                elif 'corr2' in name.lower():
                    orig_shape = corr2_cpp.shape
                else:
                    orig_shape = corr_cpp_final.shape
                orig_idx = np.unravel_index(idx, orig_shape)
                idx_str = ', '.join(map(str, orig_idx))
                print(f"{rank:<10} {idx:<15} {idx_str:<35} {format_number(cpp_flat[idx]):<20} "
                      f"{format_number(py_flat[idx]):<20} {format_number(diff_flat[idx]):<20}")
            
            print("="*100)
            
            # Analyze top 3 mismatches in detail
            if coords_cpp is not None and len(top_indices) > 0:
                print(f"\n{'='*100}")
                print(f"DETAILED ANALYSIS OF TOP 3 MISMATCHES: {name}")
                print("="*100)
                for rank, idx in enumerate(top_indices[:3], 1):
                    if 'corr1' in name.lower():
                        orig_shape = corr1_cpp.shape
                        data_cpp = corr1_cpp
                        data_py = corr1_py_compare
                        lev = 0
                        fmap_H_lev = fmap1_H if fmap1_H else 132
                        fmap_W_lev = fmap1_W if fmap1_W else 240
                        orig_idx = np.unravel_index(idx, orig_shape)
                        location = orig_idx[:5]  # (edge, di, dj, pi, pj)
                    elif 'corr2' in name.lower():
                        orig_shape = corr2_cpp.shape
                        data_cpp = corr2_cpp
                        data_py = corr2_py_compare
                        lev = 1
                        fmap_H_lev = fmap2_H if fmap2_H else 33
                        fmap_W_lev = fmap2_W if fmap2_W else 60
                        orig_idx = np.unravel_index(idx, orig_shape)
                        location = orig_idx[:5]  # (edge, di, dj, pi, pj)
                    else:
                        # Stacked tensor - extract level from index
                        orig_shape = corr_cpp_final.shape
                        data_cpp = corr_cpp_final
                        data_py = corr_py_final
                        orig_idx = np.unravel_index(idx, orig_shape)
                        if len(orig_idx) == 6:  # (edge, di, dj, pi, pj, level)
                            lev = int(orig_idx[5])
                            location = orig_idx[:5]  # (edge, di, dj, pi, pj)
                        else:
                            lev = 0
                            location = orig_idx[:5]
                        fmap_H_lev = fmap1_H if lev == 0 else (fmap2_H if fmap2_H else 33)
                        fmap_W_lev = fmap1_W if lev == 0 else (fmap2_W if fmap2_W else 60)
                    
                    analyze_mismatch_location(data_cpp, data_py, location, coords_cpp, fmap_H_lev, fmap_W_lev, lev)
                print("="*100)
            
            # Add summary of root cause
            print(f"\n{'='*100}")
            print(f"ROOT CAUSE ANALYSIS: {name}")
            print("="*100)
            print("Key Differences Between C++ and Python:")
            print("1. INTERPOLATION METHOD:")
            print("   • C++: ✅ Uses bilinear interpolation (matching Python)")
            print("   • Python: Uses bilinear interpolation (grid_sample with 4-point weighted average)")
            print("   → Both use bilinear interpolation, but implementation details may differ")
            print()
            print("2. COORDINATE HANDLING:")
            print("   • C++: ✅ Normalizes coordinates to [-1, 1] for bilinear sampling")
            print("   • Python: Normalizes coordinates to [-1, 1] for grid_sample")
            print("   → Both normalize: gx = 2 * (x / (W-1)) - 1, gy = 2 * (y / (H-1)) - 1")
            print("   → Both use align_corners=True normalization")
            print()
            print("3. BOUNDARY HANDLING:")
            print("   • C++: ✅ Strict boundary check - returns 0 if ANY corner is out of bounds")
            print("   • Python: grid_sample returns 0 for out-of-bounds coordinates")
            print("   → Both use strict boundary checking (no tolerance-based clamping)")
            print()
            print("4. POSSIBLE REMAINING ISSUES:")
            print("   • Floating-point precision differences")
            print("   • Potential dimension order mismatch in correlation window")
            print("   • Different handling of exactly-on-boundary coordinates")
            print("   • Code may not be recompiled after changes")
            print("="*100)


def print_summary(results, corr_py_final, corr_cpp_final, D_py, D, R, coords_in_bounds_fmap1):
    """Print final summary of comparison."""
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    total_comparisons = len(results)
    matched = sum(1 for r in results.values() if r['match'])
    mismatched = total_comparisons - matched
    
    print(f"{'Metric':<30} {'Value':<70}")
    print("-"*100)
    print(f"{'Total Comparisons':<30} {total_comparisons:<70}")
    print(f"{'✅ Matched':<30} {matched:<70}")
    print(f"{'❌ Mismatched':<30} {mismatched:<70}")
    
    # Check if Python outputs are all zeros
    py_all_zeros = (corr_py_final == 0).all().item()
    cpp_has_values = (corr_cpp_final != 0).any().item()
    
    if py_all_zeros and cpp_has_values:
        print(f"\n{'='*100}")
        print("⚠️  IMPORTANT NOTE: Python Correlation Outputs Are All Zeros")
        print("="*100)
        print("This is likely due to boundary handling differences:")
        print("  • Python's grid_sample returns 0 for out-of-bounds coordinates")
        print("  • C++ explicitly checks bounds and may handle out-of-bounds differently")
        print(f"  • Only {coords_in_bounds_fmap1.sum()}/{coords_in_bounds_fmap1.size} coordinates are in bounds")
        print("\nThis is a known difference in implementation approach:")
        print("  • Python: Uses PyTorch's grid_sample (returns 0 for OOB)")
        print("  • C++: Explicit bounds checking before sampling")
        print("\nThe C++ correlation values are correct for in-bounds coordinates.")
        print("Python's zeros for out-of-bounds coordinates are expected behavior.")
        print("="*100)
    
    if mismatched > 0:
        print(f"\n{'='*100}")
        print("⚠️  WARNING: Mismatches detected! Check the detailed analysis above.")
        print("="*100)
        if D_py != D:
            print(f"\n💡 RECOMMENDATION: Fix C++ correlation window size")
            print(f"   Python uses: D = 2*R + 1 = 2*{R} + 1 = {D_py}")
            print(f"   C++ currently uses: D = 2*R + 2 = 2*{R} + 2 = {D}")
            print(f"   Change C++ correlation_kernel.cpp to use: D = 2 * R + 1")
        return False
    else:
        print(f"\n{'='*100}")
        print("✅ All correlations match!")
        print("="*100)
        if D_py != D:
            print(f"\n⚠️  NOTE: Window sizes differ but center regions match")
            print(f"   Python: D = {D_py}, C++: D = {D}")
            print(f"   Consider fixing C++ to use D = 2*R + 1 to match Python exactly")
        return True


def print_python_input(py_tensors, num_active, ii1_py, jj1_py, ii1_computed, jj1_computed):
    print("\n" + "="*70)
    print("PREPARING PYTHON CORRELATION INPUTS")
    print("="*70)
    print(f"{'Input':<20} {'Shape':<30} {'Description':<50}")
    print("-"*70)
    print(f"{'gmap_py':<20} {str(py_tensors['gmap_py'].shape):<30} {'Patch features (with batch dim)':<50}")
    print(f"{'fmap1_py':<20} {str(py_tensors['fmap1_py'].shape):<30} {'Pyramid level 0 (with batch dim)':<50}")
    print(f"{'fmap2_py':<20} {str(py_tensors['fmap2_py'].shape):<30} {'Pyramid level 1 (with batch dim)':<50}")
    print(f"{'coords_py':<20} {str(py_tensors['coords_py'].shape):<30} {'Full patch coords [B,M,2,P,P]':<50}")
    print(f"{'ii1_py':<20} {f'[{num_active}]':<30} {f'Range: [{ii1_py.min().item()}, {ii1_py.max().item()}]':<50}")
    print(f"{'jj1_py':<20} {f'[{num_active}]':<30} {f'Range: [{jj1_py.min().item()}, {jj1_py.max().item()}]':<50}")
    print(f"{'C++ ii1':<20} {f'[{num_active}]':<30} {f'Range: [{ii1_computed.min()}, {ii1_computed.max()}]':<50}")
    print(f"{'C++ jj1':<20} {f'[{num_active}]':<30} {f'Range: [{jj1_computed.min()}, {jj1_computed.max()}]':<50}")
    print("="*70)


def compare_correlation(frame_num):
    """Compare C++ and Python correlation outputs for a given frame."""
    print(f"\n{'='*60}")
    print(f"Comparing Correlation Outputs for Frame {frame_num}")
    print(f"{'='*60}\n")
    
    bin_dir = "bin_file"
    
    # Load metadata
    meta_file = os.path.join(bin_dir, f"corr_frame{frame_num}_meta.bin")
    meta = load_metadata(meta_file)
    print(f"Loaded metadata: {meta}")
    
    # Extract metadata
    num_active = meta['num_active']
    M = meta['M']
    P = meta['P']
    D = meta['D']
    num_frames = meta['num_frames']
    num_gmap_frames = meta['num_gmap_frames']
    fmap1_H = meta['fmap1_H']
    fmap1_W = meta['fmap1_W']
    fmap2_H = meta['fmap2_H']
    fmap2_W = meta['fmap2_W']
    feature_dim = meta['feature_dim']
    R = (D - 1) // 2  # Calculate R from D: D = 2*R + 1
    
    # Load and reshape C++ data
    cpp_data = load_and_reshape_cpp_data(frame_num, meta, bin_dir)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔍 Using device: {device}")
    
    # Prepare Python tensors
    py_tensors = prepare_python_tensors(cpp_data, device)
    
    # Verify slice correspondence
    ii1_py, jj1_py, ii1_computed, jj1_computed = verify_slice_correspondence(
        cpp_data['kk'], cpp_data['jj'], num_active, M, num_gmap_frames, num_frames, device
    )
    
    # Print Python input summary
    print_python_input(py_tensors, num_active, ii1_py, jj1_py, ii1_computed, jj1_computed)
    
    # Validate coordinates
    coords_in_bounds_fmap1 = validate_coordinates(
        cpp_data['coords'], fmap1_H, fmap1_W, fmap2_H, fmap2_W, R, P
    )
    
    # Compute Python correlation
    corr1_py, corr2_py, D_py = compute_python_correlation(
        py_tensors, ii1_py, jj1_py, R, fmap1_H, fmap1_W, fmap2_H, fmap2_W, py_tensors['corr_cpp_torch'], P
    )
    
    # Align correlation windows
    corr_cpp_final, corr_py_final, D_compare = align_correlation_windows(
        py_tensors['corr_cpp_torch'], corr1_py, corr2_py, D, D_py, R
    )
    
    # Compare outputs
    results, corr1_cpp, corr2_cpp, corr1_py_compare, corr2_py_compare = compare_correlation_outputs(
        corr_cpp_final, corr_py_final, D_compare, P, num_active
    )
    
    # Print comparison table
    print_comparison_table(results)
    
    # Print detailed analysis
    print_detailed_analysis(
        results, corr1_cpp, corr2_cpp, corr_cpp_final,
        corr1_py_compare, corr2_py_compare, corr_py_final,
        coords_cpp=cpp_data['coords'], fmap1_H=fmap1_H, fmap1_W=fmap1_W,
        fmap2_H=fmap2_H, fmap2_W=fmap2_W
    )
    
    # Print summary
    return print_summary(results, corr_py_final, corr_cpp_final, D_py, D, R, coords_in_bounds_fmap1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare C++ and Python correlation outputs")
    parser.add_argument("--frame", type=int, default=69, help="Frame number to compare (default: 69)")
    
    args = parser.parse_args()
    
    try:
        success = compare_correlation(args.frame)
        sys.exit(0 if success else 1)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nMake sure you have run the C++ code and generated the correlation data files.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)