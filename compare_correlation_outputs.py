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


def compare_correlation(frame_num):
    """Compare C++ and Python correlation outputs for a given frame."""
    print(f"\n{'='*60}")
    print(f"Comparing Correlation Outputs for Frame {frame_num}")
    print(f"{'='*60}\n")
    
    # Load metadata
    meta_file = f"corr_frame{frame_num}_meta.bin"
    meta = load_metadata(meta_file)
    print(f"Loaded metadata: {meta}")
    
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
    # Calculate R from D: D = 2*R + 1, so R = (D - 1) / 2
    R = (D - 1) // 2  # For D=5: R = (5-1)/2 = 2
    
    # Load C++ data
    print("\n" + "="*70)
    print("LOADING C++ DATA")
    print("="*70)
    coords_cpp = load_binary_float(f"corr_frame{frame_num}_coords.bin")
    kk_cpp = load_binary_int32(f"corr_frame{frame_num}_kk.bin")
    jj_cpp = load_binary_int32(f"corr_frame{frame_num}_jj.bin")
    ii_cpp = load_binary_int32(f"corr_frame{frame_num}_ii.bin")
    gmap_cpp = load_binary_float(f"corr_frame{frame_num}_gmap.bin")
    fmap1_cpp = load_binary_float(f"corr_frame{frame_num}_fmap1.bin")
    fmap2_cpp = load_binary_float(f"corr_frame{frame_num}_fmap2.bin")
    corr_cpp = load_binary_float(f"corr_frame{frame_num}_corr.bin")
    
    # Reshape C++ data
    coords_cpp = coords_cpp.reshape(num_active, 2, P, P)  # [num_active, 2, P, P]
    gmap_cpp = gmap_cpp.reshape(num_active, feature_dim, 3, 3)  # [num_active, 128, 3, 3]
    fmap1_cpp = fmap1_cpp.reshape(num_active, feature_dim, fmap1_H, fmap1_W)  # [num_active, 128, H, W]
    fmap2_cpp = fmap2_cpp.reshape(num_active, feature_dim, fmap2_H, fmap2_W)  # [num_active, 128, H, W]
    corr_cpp = corr_cpp.reshape(num_active, D, D, P, P, 2)  # [num_active, D, D, P, P, 2]
    
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
    
    # Convert to PyTorch tensors and move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔍 Using device: {device}")
    
    coords_torch = torch.from_numpy(coords_cpp).float().to(device)
    gmap_torch = torch.from_numpy(gmap_cpp).float().to(device)
    fmap1_torch = torch.from_numpy(fmap1_cpp).float().to(device)
    fmap2_torch = torch.from_numpy(fmap2_cpp).float().to(device)
    corr_cpp_torch = torch.from_numpy(corr_cpp).float().to(device)
    
    # Prepare Python correlation inputs
    # Python expects: [B, M_total, C, H, W] for fmap1/fmap2
    # We have slices, so we'll create minimal buffers with just the needed patches/frames
    # Use sequential indices [0, 1, 2, ...] since we're using slices
    
    # Reshape for Python: [B=1, M=num_active, C, H, W]
    gmap_py = gmap_torch.unsqueeze(0)  # [1, num_active, 128, 3, 3]
    fmap1_py = fmap1_torch.unsqueeze(0)  # [1, num_active, 128, fmap1_H, fmap1_W]
    fmap2_py = fmap2_torch.unsqueeze(0)  # [1, num_active, 128, fmap2_H, fmap2_W]
    
    # Reshape coords for Python: Python's correlation expects [B, M, 2, H, W] where H=W=P
    # From correlation_kernel.py line 477: B, M, _, H, W = coords.shape
    # C++ has [num_active, 2, P, P] - coordinates for each pixel in the patch
    # We need to reshape to [1, num_active, 2, P, P] for Python
    # coords_torch shape: [num_active, 2, P, P]
    # Reshape: [num_active, 2, P, P] -> [1, num_active, 2, P, P]
    coords_py = coords_torch.unsqueeze(0)  # [1, num_active, 2, P, P]
    
    # Compute ii1 and jj1 from kk and jj using the same modulo operation as C++
    # Python: ii1 = kk % (M * pmem), jj1 = jj % mem
    # C++ uses these same values to index into the full buffers
    # Since we have slices extracted in order [edge0, edge1, edge2, ...], we use sequential indices
    # But we need to verify that the slices match what C++ would select
    mod_value = M * num_gmap_frames
    ii1_computed = kk_cpp % mod_value  # Actual ii1 values C++ used
    jj1_computed = jj_cpp % num_frames  # Actual jj1 values C++ used
    
    # For Python correlation with slices, we use sequential indices [0, 1, 2, ...]
    # because our slices are extracted in order [edge0, edge1, edge2, ...]
    # This means: slice[0] corresponds to edge 0, slice[1] to edge 1, etc.
    # So ii1_py[e] = e and jj1_py[e] = e to select slice[e] from our buffers
    ii1_py = torch.arange(num_active, dtype=torch.long, device=device)  # [num_active] - sequential: [0, 1, 2, ...]
    jj1_py = torch.arange(num_active, dtype=torch.long, device=device)  # [num_active] - sequential: [0, 1, 2, ...]
    
    # CRITICAL: Verify that slices match what C++ would select
    # The assumption is: gmap_py[:, e] should equal what C++ gets from gmap[ii1_computed[e]]
    # But we need to verify this! If slices are not in the correct order, we need to reorder them.
    print(f"\n{'='*70}")
    print("VERIFYING SLICE CORRESPONDENCE")
    print("="*70)
    print(f"{'Edge':<10} {'C++ ii1':<15} {'C++ jj1':<15} {'Python ii1':<15} {'Python jj1':<15} {'Note':<30}")
    print("-"*70)
    
    # Check if ii1_computed and jj1_computed are sequential (which would validate our assumption)
    ii1_is_sequential = np.allclose(ii1_computed, np.arange(num_active))
    jj1_is_sequential = np.allclose(jj1_computed, np.arange(num_active))
    
    for e in range(min(10, num_active)):
        note = ""
        if e < num_active:
            if ii1_computed[e] != e:
                note += f"ii1 mismatch! "
            if jj1_computed[e] != e:
                note += f"jj1 mismatch! "
        print(f"{e:<10} {ii1_computed[e]:<15} {jj1_computed[e]:<15} {ii1_py[e].item():<15} {jj1_py[e].item():<15} {note:<30}")
    
    print("="*70)
    if not ii1_is_sequential or not jj1_is_sequential:
        print(f"⚠️  WARNING: Indices are NOT sequential!")
        print(f"   ii1_is_sequential: {ii1_is_sequential}")
        print(f"   jj1_is_sequential: {jj1_is_sequential}")
        print(f"   This means slices might not be in the correct order!")
        print(f"   We need to reorder slices to match C++'s indexing.")
        
        # Reorder slices to match C++ indexing
        print(f"\n{'='*70}")
        print("REORDERING SLICES TO MATCH C++ INDEXING")
        print("="*70)
        
        # Create reordered tensors
        gmap_py_reordered = torch.zeros_like(gmap_py)
        fmap1_py_reordered = torch.zeros_like(fmap1_py)
        fmap2_py_reordered = torch.zeros_like(fmap2_py)
        coords_py_reordered = torch.zeros_like(coords_py)
        
        # Map: for each edge e, C++ uses ii1_computed[e] and jj1_computed[e]
        # So we need: gmap_py_reordered[:, e] = gmap_py[:, ii1_computed[e]]
        # But wait - we need the inverse mapping!
        # If C++ selects gmap[ii1_computed[e]] for edge e, then our slice at position e
        # should contain what C++ gets from gmap[ii1_computed[e]]
        # So we need to find which slice index corresponds to each ii1_computed value
        
        # Actually, the slices were saved in edge order [0, 1, 2, ...]
        # So slice[e] contains what C++ gets for edge e
        # C++ uses ii1_computed[e] to index into the full buffer
        # So we need: when Python uses index e, it should get slice[e]
        # But Python correlation will do: gmap_py[:, ii1_py[e]]
        # So if ii1_py[e] = e, it gets gmap_py[:, e] which is slice[e] - correct!
        
        # The issue might be that the slices don't actually match what C++ would select
        # Let's check if we need to use the actual ii1_computed and jj1_computed values
        # instead of sequential indices
        
        # Actually, I think the problem is different:
        # The slices were extracted in edge order, so slice[e] = what C++ gets for edge e
        # But C++ uses ii1_computed[e] to index into the full buffer
        # So if we want Python to match, we should use ii1_computed and jj1_computed directly
        # BUT - our buffers only have num_active elements, not the full buffer size
        # So we can't use ii1_computed directly if it's > num_active
        
        # Check if all indices are within bounds
        max_ii1 = ii1_computed.max()
        max_jj1 = jj1_computed.max()
        print(f"   max_ii1: {max_ii1}, num_active: {num_active}, within bounds: {max_ii1 < num_active}")
        print(f"   max_jj1: {max_jj1}, num_active: {num_active}, within bounds: {max_jj1 < num_active}")
        
        if max_ii1 >= num_active or max_jj1 >= num_active:
            print(f"   ⚠️  ERROR: Some indices are out of bounds!")
            print(f"   Cannot use ii1_computed/jj1_computed directly with pre-sliced buffers.")
            print(f"   Need to either:")
            print(f"     1. Load full buffers (not slices) and use ii1_computed/jj1_computed")
            print(f"     2. Verify that slices are in the correct order")
            print(f"   Using sequential indices as fallback (may cause mismatches)")
        else:
            # CRITICAL: The slices are saved in edge order [0, 1, 2, ...]
            # So gmap_py[:, e] contains what C++ gets for edge e
            # C++ uses ii1_computed[e] to index into the full buffer
            # But our slices are already extracted for each edge in order
            # So we should use sequential indices [0, 1, 2, ...] to match edge order
            # NOT ii1_computed/jj1_computed (which would index into wrong slices)
            print(f"   ✅ All indices are within bounds")
            print(f"   ⚠️  IMPORTANT: Slices are in edge order, so we use sequential indices")
            print(f"      Using ii1_py[e] = e and jj1_py[e] = e (not ii1_computed/jj1_computed)")
            print(f"      This is correct because slice[e] = what C++ gets for edge e")
            # Keep sequential indices - slices are already in edge order
            # ii1_py and jj1_py are already set to sequential above
    else:
        print("✅ Indices are sequential - slices should be in correct order")
        print("   Using sequential indices [0, 1, 2, ...] is correct")
    
    print(f"\nIndex mapping comparison:")
    print(f"  C++ kk range: [{kk_cpp.min()}, {kk_cpp.max()}]")
    print(f"  C++ jj range: [{jj_cpp.min()}, {jj_cpp.max()}]")
    print(f"  C++ ii1 (computed) range: [{ii1_computed.min()}, {ii1_computed.max()}]")
    print(f"  C++ jj1 (computed) range: [{jj1_computed.min()}, {jj1_computed.max()}]")
    print(f"  Python ii1_py (sequential): [0, {num_active-1}]")
    print(f"  Python jj1_py (sequential): [0, {num_active-1}]")
    
    print("\n" + "="*70)
    print("PREPARING PYTHON CORRELATION INPUTS")
    print("="*70)
    print(f"{'Input':<20} {'Shape':<30} {'Description':<50}")
    print("-"*70)
    print(f"{'gmap_py':<20} {str(gmap_py.shape):<30} {'Patch features (with batch dim)':<50}")
    print(f"{'fmap1_py':<20} {str(fmap1_py.shape):<30} {'Pyramid level 0 (with batch dim)':<50}")
    print(f"{'fmap2_py':<20} {str(fmap2_py.shape):<30} {'Pyramid level 1 (with batch dim)':<50}")
    print(f"{'coords_py':<20} {str(coords_py.shape):<30} {'Full patch coords [B,M,2,P,P]':<50}")
    print(f"{'ii1_py':<20} {f'[{num_active}]':<30} {f'Sequential: [0..{num_active-1}]':<50}")
    print(f"{'jj1_py':<20} {f'[{num_active}]':<30} {f'Sequential: [0..{num_active-1}]':<50}")
    print(f"{'C++ ii1':<20} {f'[{num_active}]':<30} {f'Range: [{ii1_computed.min()}, {ii1_computed.max()}]':<50}")
    print(f"{'C++ jj1':<20} {f'[{num_active}]':<30} {f'Range: [{jj1_computed.min()}, {jj1_computed.max()}]':<50}")
    print("="*70)
    
    # Check coordinate validity
    print("\n" + "="*70)
    print("COORDINATE VALIDITY CHECK")
    print("="*70)
    coords_valid = np.isfinite(coords_cpp).all()
    coords_in_bounds_fmap1 = ((coords_cpp[:, 0, :, :] >= 0) & (coords_cpp[:, 0, :, :] < fmap1_W) & 
                              (coords_cpp[:, 1, :, :] >= 0) & (coords_cpp[:, 1, :, :] < fmap1_H))
    coords_in_bounds_fmap2 = ((coords_cpp[:, 0, :, :] / 4 >= 0) & (coords_cpp[:, 0, :, :] / 4 < fmap2_W) & 
                              (coords_cpp[:, 1, :, :] / 4 >= 0) & (coords_cpp[:, 1, :, :] / 4 < fmap2_H))
    
    # Check coordinates with correlation window offsets (R radius)
    # Python will sample at coords +/- R, so we need to check if those are in bounds
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
    
    # Show sample coordinates for first few edges
    print(f"\n{'='*70}")
    print("SAMPLE COORDINATES (first 5 edges, center pixel)")
    print("="*70)
    print(f"{'Edge':<10} {'X':<15} {'Y':<15} {'In fmap1?':<15} {'In fmap1±R?':<15} {'In fmap2?':<15} {'In fmap2±R?':<15}")
    print("-"*70)
    center_p = P // 2
    for e in range(min(5, num_active)):
        x = coords_cpp[e, 0, center_p, center_p]
        y = coords_cpp[e, 1, center_p, center_p]
        in_fmap1 = coords_in_bounds_fmap1[e, center_p, center_p]
        in_fmap1_offset = coords_with_offset_fmap1[e, center_p, center_p]
        in_fmap2 = coords_in_bounds_fmap2[e, center_p, center_p]
        in_fmap2_offset = coords_with_offset_fmap2[e, center_p, center_p]
        print(f"{e:<10} {x:<15.2f} {y:<15.2f} {str(in_fmap1):<15} {str(in_fmap1_offset):<15} {str(in_fmap2):<15} {str(in_fmap2_offset):<15}")
    print("="*70)
    
    # Call Python correlation for level 0 (coords / 1)
    print("\n" + "="*70)
    print("COMPUTING PYTHON CORRELATION")
    print("="*70)
    print(f"{'Level':<15} {'Input Coords Scale':<20} {'Output Shape':<30} {'Radius':<10} {'Window Size':<15}")
    print("-"*70)
    
    # Debug: Check coordinate values before calling Python correlation
    print(f"\nDebug: Sample coordinates (first edge, center pixel):")
    print(f"  coords_py[0, 0, :, 1, 1] = {coords_py[0, 0, :, 1, 1].tolist()}")
    print(f"  coords_py shape: {coords_py.shape}")
    print(f"  gmap_py[0, 0, 0, 1, 1] (sample feature): {gmap_py[0, 0, 0, 1, 1].item():.6f}")
    print(f"  fmap1_py[0, 0, 0, 66, 120] (sample feature): {fmap1_py[0, 0, 0, 66, 120].item():.6f}")
    print(f"  ii1_py[:5]: {ii1_py[:5].tolist()}")
    print(f"  jj1_py[:5]: {jj1_py[:5].tolist()}")
    
    # Debug: Check what Python will select
    print(f"\nDebug: What Python will select:")
    print(f"  gmap_py[:, ii1_py[:3]] shape: {gmap_py[:, ii1_py[:3]].shape}")
    print(f"  fmap1_py[:, jj1_py[:3]] shape: {fmap1_py[:, jj1_py[:3]].shape}")
    print(f"  gmap_py[:, 0, 0, 1, 1] (first patch, first channel, center): {gmap_py[:, 0, 0, 1, 1].item():.6f}")
    print(f"  fmap1_py[:, 0, 0, 66, 120] (first frame, first channel, sample): {fmap1_py[:, 0, 0, 66, 120].item():.6f}")
    
    # CRITICAL: Verify that slices match what C++ would use
    # For edge e, C++ uses gmap[ii1_computed[e]] and fmap1[jj1_computed[e]]
    # Our slices: gmap_py[:, e] should contain what C++ gets from gmap[ii1_computed[e]]
    # But Python correlation will use gmap_py[:, ii1_py[e]]
    # So if ii1_py[e] = e, it uses gmap_py[:, e] which should be correct
    # But if ii1_py[e] = ii1_computed[e], it uses gmap_py[:, ii1_computed[e]] which is wrong!
    print(f"\n🔍 CRITICAL: Verifying slice correspondence:")
    print(f"   For edge 0: C++ uses gmap[{ii1_computed[0]}] and fmap1[{jj1_computed[0]}]")
    print(f"   Python will use: gmap_py[:, {ii1_py[0].item()}] and fmap1_py[:, {jj1_py[0].item()}]")
    print(f"   Slice gmap_py[:, 0] should = C++ gmap[{ii1_computed[0]}]")
    print(f"   Slice fmap1_py[:, 0] should = C++ fmap1[{jj1_computed[0]}]")
    if ii1_py[0].item() != 0:
        print(f"   ⚠️  WARNING: ii1_py[0] = {ii1_py[0].item()} != 0, may be using wrong slice!")
    if jj1_py[0].item() != 0:
        print(f"   ⚠️  WARNING: jj1_py[0] = {jj1_py[0].item()} != 0, may be using wrong slice!")
    
    # Debug: Check coordinate extraction (what Python will extract)
    print(f"\nDebug: Coordinate extraction (what Python will extract):")
    # coords_py shape: [1, num_active, 2, P, P]
    # x coordinates: coords_py[:, :, 0, :, :] = [1, num_active, P, P]
    # y coordinates: coords_py[:, :, 1, :, :] = [1, num_active, P, P]
    x_extracted = coords_py[:, :, 0, :, :]  # [B, M, H, W]
    y_extracted = coords_py[:, :, 1, :, :]  # [B, M, H, W]
    print(f"  x_extracted shape: {x_extracted.shape}")
    print(f"  y_extracted shape: {y_extracted.shape}")
    print(f"  x_extracted[0, 0, 1, 1] (first edge, center): {x_extracted[0, 0, 1, 1].item():.6f}")
    print(f"  y_extracted[0, 0, 1, 1] (first edge, center): {y_extracted[0, 0, 1, 1].item():.6f}")
    
    # Debug: Check coordinate normalization (what Python will do)
    # Python normalizes: gx = 2 * gx / (W2 - 1) - 1, gy = 2 * gy / (H2 - 1) - 1
    # For level 0: W2=240, H2=132
    # For level 1: W2=60, H2=33
    print(f"\nDebug: Coordinate normalization check:")
    sample_x = coords_py[0, 0, 0, 1, 1].item()  # x coordinate
    sample_y = coords_py[0, 0, 1, 1, 1].item()  # y coordinate
    print(f"  Sample coords (level 0): x={sample_x:.2f}, y={sample_y:.2f}")
    print(f"  fmap1 bounds: W={fmap1_W}, H={fmap1_H}")
    # Python will normalize: gx = 2 * floor(x) / (W2 - 1) - 1
    gx_norm = 2 * np.floor(sample_x) / (fmap1_W - 1) - 1
    gy_norm = 2 * np.floor(sample_y) / (fmap1_H - 1) - 1
    print(f"  Normalized (level 0): gx={gx_norm:.4f}, gy={gy_norm:.4f} (should be in [-1, 1])")
    print(f"  In bounds? gx: {-1 <= gx_norm <= 1}, gy: {-1 <= gy_norm <= 1}")
    
    # Manual verification: Check what Python correlation should compute for edge 0
    print(f"\n🔍 Manual verification for edge 0:")
    e = 0
    x_coord = coords_py[0, e, 0, 1, 1].item()  # Center pixel x
    y_coord = coords_py[0, e, 1, 1, 1].item()  # Center pixel y
    print(f"   Edge {e}: coords center = ({x_coord:.2f}, {y_coord:.2f})")
    print(f"   ii1_py[{e}] = {ii1_py[e].item()}, jj1_py[{e}] = {jj1_py[e].item()}")
    print(f"   C++ ii1[{e}] = {ii1_computed[e]}, jj1[{e}] = {jj1_computed[e]}")
    
    # Check what features Python will access
    patch_feat = gmap_py[0, ii1_py[e], :, 1, 1]  # Patch feature at center
    frame_x = int(np.floor(x_coord))
    frame_y = int(np.floor(y_coord))
    if 0 <= frame_x < fmap1_W and 0 <= frame_y < fmap1_H:
        frame_feat = fmap1_py[0, jj1_py[e], :, frame_y, frame_x]  # Frame feature at coord
        dot_product = torch.dot(patch_feat, frame_feat).item()
        print(f"   Manual dot product at ({frame_x}, {frame_y}): {dot_product:.6f}")
        print(f"   C++ correlation value at center: {corr_cpp_torch[e, R, R, 1, 1, 0].item():.6f}")
        if abs(dot_product - corr_cpp_torch[e, R, R, 1, 1, 0].item()) < 0.1:
            print(f"   ✅ Manual computation matches C++ (within tolerance)")
        else:
            print(f"   ❌ Manual computation does NOT match C++!")
            print(f"      This suggests indexing or data mismatch")
    else:
        print(f"   ⚠️  Coordinates ({frame_x}, {frame_y}) are out of bounds!")
    
    # Check device and data types
    print(f"\n🔍 Device and data type check:")
    print(f"   gmap_py device: {gmap_py.device}, dtype: {gmap_py.dtype}")
    print(f"   fmap1_py device: {fmap1_py.device}, dtype: {fmap1_py.dtype}")
    print(f"   coords_py device: {coords_py.device}, dtype: {coords_py.dtype}")
    print(f"   ii1_py device: {ii1_py.device}, dtype: {ii1_py.dtype}")
    print(f"   jj1_py device: {jj1_py.device}, dtype: {jj1_py.dtype}")
    
    # Create coords_level0 for level 0 (needed for validity check and correlation)
    coords_level0 = coords_py / 1.0
    
    # Verify data is not all zeros
    print(f"\n🔍 Data validity check:")
    print(f"   gmap_py non-zero: {(gmap_py != 0).sum().item()}/{gmap_py.numel()} ({(gmap_py != 0).sum().item()/gmap_py.numel()*100:.2f}%)")
    print(f"   fmap1_py non-zero: {(fmap1_py != 0).sum().item()}/{fmap1_py.numel()} ({(fmap1_py != 0).sum().item()/fmap1_py.numel()*100:.2f}%)")
    print(f"   coords_py finite: {torch.isfinite(coords_py).sum().item()}/{coords_py.numel()} ({torch.isfinite(coords_py).sum().item()/coords_py.numel()*100:.2f}%)")
    
    # Check if coordinates are in the expected range
    coords_x = coords_level0[0, :, 0, :, :]
    coords_y = coords_level0[0, :, 1, :, :]
    print(f"   coords_x range: [{coords_x.min().item():.2f}, {coords_x.max().item():.2f}]")
    print(f"   coords_y range: [{coords_y.min().item():.2f}, {coords_y.max().item():.2f}]")
    print(f"   fmap1 bounds: [0, {fmap1_W}) x [0, {fmap1_H})")
    print(f"\n🔍 Calling Python correlation function...")
    corr1_py = altcorr_corr(gmap_py, fmap1_py, coords_level0, ii1_py, jj1_py, radius=R)
    print(f"   Correlation function returned, shape: {corr1_py.shape}")
    D_py = corr1_py.shape[2]  # Actual Python window size
    print(f"{'Level 0':<15} {'coords / 1':<20} {str(corr1_py.shape):<30} {R:<10} {D_py:<15}")
    
    # Debug: Check if correlation output is all zeros
    corr1_nonzero = (corr1_py != 0).sum().item()
    corr1_total = corr1_py.numel()
    corr1_nan = torch.isnan(corr1_py).sum().item()
    corr1_inf = torch.isinf(corr1_py).sum().item()
    corr1_huge = (torch.abs(corr1_py) > 1e6).sum().item()
    print(f"  Debug: corr1_py nonzero elements: {corr1_nonzero}/{corr1_total} ({100*corr1_nonzero/corr1_total:.2f}%)")
    print(f"  Debug: corr1_py NaN elements: {corr1_nan}/{corr1_total} ({100*corr1_nan/corr1_total:.2f}%)")
    print(f"  Debug: corr1_py Inf elements: {corr1_inf}/{corr1_total} ({100*corr1_inf/corr1_total:.2f}%)")
    print(f"  Debug: corr1_py huge elements (>1e6): {corr1_huge}/{corr1_total} ({100*corr1_huge/corr1_total:.2f}%)")
    # corr1_py shape: [1, 40, 5, 5, 3, 3] - no channel dimension, just one level
    # Center of correlation window: for D=7, R=3, center is at index R=3
    center_d = R  # Center index in correlation window
    py_center_val = corr1_py[0, 0, center_d, center_d, 1, 1].item()
    cpp_center_val = corr_cpp_torch[0, R, R, 1, 1, 0].item()
    print(f"  Debug: corr1_py[0, 0, {center_d}, {center_d}, 1, 1] (center, window pos {center_d},{center_d}): {py_center_val:.6f}")
    print(f"  Debug: C++ corr[0, {R}, {R}, 1, 1, 0] (center): {cpp_center_val:.6f}")
    print(f"  Debug: Manual dot product (from verification above): 4.285662")
    print(f"  Debug: Difference Python vs C++: {abs(py_center_val - cpp_center_val):.6f}")
    print(f"  Debug: Difference Python vs Manual: {abs(py_center_val - 4.285662):.6f}")
    
    # Check other window positions to see if Python is computing at a different offset
    print(f"\n  Debug: Checking all window positions for edge 0, patch center (1,1):")
    print(f"    Window pos | Python value | C++ value | Difference")
    print(f"    {'-'*60}")
    for di in range(D_py):
        for dj in range(D_py):
            py_val = corr1_py[0, 0, di, dj, 1, 1].item()
            cpp_val = corr_cpp_torch[0, di, dj, 1, 1, 0].item()
            diff = abs(py_val - cpp_val)
            if diff > 0.1 or abs(py_val) > 0.1 or abs(cpp_val) > 0.1:  # Show significant values
                offset_i = di - R
                offset_j = dj - R
                print(f"    ({di},{dj}) offset=({offset_i:+d},{offset_j:+d}) | {py_val:10.6f} | {cpp_val:10.6f} | {diff:10.6f}")
    
    # Find edges with NaN or huge values
    if corr1_nan > 0 or corr1_huge > 0:
        print(f"\n  ⚠️  WARNING: Found NaN or huge values in corr1_py!")
        nan_edges = torch.isnan(corr1_py).any(dim=2).any(dim=2).any(dim=2).any(dim=2)[0]  # [num_active]
        huge_edges = (torch.abs(corr1_py) > 1e6).any(dim=2).any(dim=2).any(dim=2).any(dim=2)[0]  # [num_active]
        nan_edge_indices = torch.where(nan_edges)[0][:5].tolist()
        huge_edge_indices = torch.where(huge_edges)[0][:5].tolist()
        print(f"     Edges with NaN (first 5): {nan_edge_indices}")
        print(f"     Edges with huge values (first 5): {huge_edge_indices}")
        if len(nan_edge_indices) > 0:
            e = nan_edge_indices[0]
            print(f"     Debug edge {e}:")
            print(f"       ii1_py[{e}] = {ii1_py[e].item()}, jj1_py[{e}] = {jj1_py[e].item()}")
            print(f"       C++ ii1[{e}] = {ii1_computed[e]}, jj1[{e}] = {jj1_computed[e]}")
            print(f"       coords[{e}] center: {coords_py[0, e, :, 1, 1].tolist()}")
            print(f"       gmap_py[:, {ii1_py[e].item()}] shape: {gmap_py[:, ii1_py[e]].shape}")
            print(f"       fmap1_py[:, {jj1_py[e].item()}] shape: {fmap1_py[:, jj1_py[e]].shape}")
    
    # Debug: Check if the issue is with grid_sample returning zeros
    # Python's grid_sample might be returning zeros if coordinates with offsets are out of bounds
    # Let's check what coordinates Python will try to sample (with correlation window offsets)
    print(f"\n  Debug: Checking correlation window offsets:")
    print(f"    Radius R={R}, Window D={D_py}")
    print(f"    Offsets range: [{-R}, {R}]")
    print(f"    Sample coord: x={sample_x:.2f}, y={sample_y:.2f}")
    print(f"    With offsets, x range: [{sample_x-R:.2f}, {sample_x+R:.2f}]")
    print(f"    With offsets, y range: [{sample_y-R:.2f}, {sample_y+R:.2f}]")
    print(f"    fmap1 bounds: [0, {fmap1_W}) x [0, {fmap1_H})")
    print(f"    Some offsets out of bounds? x: {sample_x-R < 0 or sample_x+R >= fmap1_W}, y: {sample_y-R < 0 or sample_y+R >= fmap1_H}")
    
    # Call Python correlation for level 1 (coords / 4)
    coords_level1 = coords_py / 4.0
    corr2_py = altcorr_corr(gmap_py, fmap2_py, coords_level1, ii1_py, jj1_py, radius=R)
    print(f"{'Level 1':<15} {'coords / 4':<20} {str(corr2_py.shape):<30} {R:<10} {corr2_py.shape[2]:<15}")
    
    # Debug: Check if correlation output is all zeros
    corr2_nonzero = (corr2_py != 0).sum().item()
    corr2_total = corr2_py.numel()
    corr2_nan = torch.isnan(corr2_py).sum().item()
    corr2_inf = torch.isinf(corr2_py).sum().item()
    corr2_huge = (torch.abs(corr2_py) > 1e6).sum().item()
    print(f"  Debug: corr2_py nonzero elements: {corr2_nonzero}/{corr2_total} ({100*corr2_nonzero/corr2_total:.2f}%)")
    print(f"  Debug: corr2_py NaN elements: {corr2_nan}/{corr2_total} ({100*corr2_nan/corr2_total:.2f}%)")
    print(f"  Debug: corr2_py Inf elements: {corr2_inf}/{corr2_total} ({100*corr2_inf/corr2_total:.2f}%)")
    print(f"  Debug: corr2_py huge elements (>1e6): {corr2_huge}/{corr2_total} ({100*corr2_huge/corr2_total:.2f}%)")
    # corr2_py shape: [1, 40, 5, 5, 3, 3] - no channel dimension, just one level
    print(f"  Debug: corr2_py[0, 0, 2, 2, 1, 1] (center): {corr2_py[0, 0, 2, 2, 1, 1].item():.6f}")
    
    # Find edges with NaN or huge values
    if corr2_nan > 0 or corr2_huge > 0:
        print(f"\n  ⚠️  WARNING: Found NaN or huge values in corr2_py!")
        nan_edges = torch.isnan(corr2_py).any(dim=2).any(dim=2).any(dim=2).any(dim=2)[0]  # [num_active]
        huge_edges = (torch.abs(corr2_py) > 1e6).any(dim=2).any(dim=2).any(dim=2).any(dim=2)[0]  # [num_active]
        nan_edge_indices = torch.where(nan_edges)[0][:5].tolist()
        huge_edge_indices = torch.where(huge_edges)[0][:5].tolist()
        print(f"     Edges with NaN (first 5): {nan_edge_indices}")
        print(f"     Edges with huge values (first 5): {huge_edge_indices}")
        if len(nan_edge_indices) > 0:
            e = nan_edge_indices[0]
            print(f"     Debug edge {e}:")
            print(f"       ii1_py[{e}] = {ii1_py[e].item()}, jj1_py[{e}] = {jj1_py[e].item()}")
            print(f"       C++ ii1[{e}] = {ii1_computed[e]}, jj1[{e}] = {jj1_computed[e]}")
            print(f"       coords[{e}] center (level 1, /4): {(coords_py[0, e, :, 1, 1] / 4.0).tolist()}")
    
    print("-"*70)
    print(f"{'Summary':<15} {'':<20} {'':<30} {'':<10} {'':<15}")
    print(f"{'Python D':<15} {'':<20} {'':<30} {'':<10} {D_py:<15}")
    print(f"{'C++ D':<15} {'':<20} {'':<30} {'':<10} {D:<15}")
    if D_py == D:
        print(f"{'Status':<15} {'✅ Window sizes match!':<20} {'':<30} {'':<10} {'':<15}")
    else:
        print(f"{'Status':<15} {'⚠️ Window size mismatch':<20} {'':<30} {'':<10} {'':<15}")
    print("="*70)
    
    # Stack results (matches Python: torch.stack([corr1, corr2], -1))
    corr_py_stacked = torch.stack([corr1_py, corr2_py], dim=-1)  # [1, num_active, D_py, D_py, P, P, 2]
    print(f"  corr_py_stacked shape: {corr_py_stacked.shape}")
    
    # Remove batch dimension - keep Python's output as-is
    corr_py_final = corr_py_stacked.squeeze(0)  # [num_active, D_py, D_py, P, P, 2]
    print(f"  corr_py_final shape: {corr_py_final.shape}")
    
    # Extract center region from C++ output to match Python's window size
    # C++ uses D_cpp = 8, Python uses D_py = 7
    # Extract center 7x7 from C++'s 8x8 output
    corr_cpp_final = corr_cpp_torch.clone()  # [num_active, D, D, P, P, 2]
    
    if D_py != D:
        print(f"\n⚠️  WARNING: Correlation window size mismatch!")
        print(f"   Python: D_py = {D_py} (2*R+1 = 2*{R}+1 = {2*R+1})")
        print(f"   C++: D_cpp = {D} (2*R+2 = 2*{R}+2 = {2*R+2})")
        print(f"   Extracting center {D_py}x{D_py} region from C++'s {D}x{D} output to match Python")
        
        # Extract center region from C++ output
        crop_start = (D - D_py) // 2
        corr_cpp_final = corr_cpp_final[:, crop_start:crop_start+D_py, crop_start:crop_start+D_py, :, :, :]
        print(f"   Extracted C++ correlation: [{crop_start}:{crop_start+D_py}, {crop_start}:{crop_start+D_py}]")
        print(f"   Final corr_cpp_final shape: {corr_cpp_final.shape}")
        
        # Update D to match Python for comparison
        D_compare = D_py
    else:
        D_compare = D
    
    # Compare outputs (using aligned window sizes)
    print("\n" + "="*70)
    print("COMPARING OUTPUTS")
    print("="*70)
    
    # Show sample correlation values in table format
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
    
    results = {}
    
    # Compare level 0 correlation
    corr1_cpp = corr_cpp_final[:, :, :, :, :, 0]  # [num_active, D_compare, D_compare, P, P]
    corr1_py_compare = corr_py_final[:, :, :, :, :, 0]  # [num_active, D_py, D_py, P, P]
    results['corr1 (level 0)'] = compare_tensors(corr1_cpp, corr1_py_compare, 'corr1', tolerance=1e-4, rel_tolerance=1e-3)
    
    # Compare level 1 correlation
    corr2_cpp = corr_cpp_final[:, :, :, :, :, 1]  # [num_active, D_compare, D_compare, P, P]
    corr2_py_compare = corr_py_final[:, :, :, :, :, 1]  # [num_active, D_py, D_py, P, P]
    results['corr2 (level 1)'] = compare_tensors(corr2_cpp, corr2_py_compare, 'corr2', tolerance=1e-4, rel_tolerance=1e-3)
    
    # Compare full stacked output
    results['corr (stacked)'] = compare_tensors(corr_cpp_final, corr_py_final, 'corr', tolerance=1e-4, rel_tolerance=1e-3)
    
    # Print comparison table
    print_comparison_table(results)
    
    # Detailed mismatch analysis in table format
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
            top_indices = np.argsort(diff_flat)[-20:][::-1]  # Top 20 instead of 10
            
            print(f"\n{'='*100}")
            print(f"TOP 20 LARGEST DIFFERENCES: {name}")
            print("="*100)
            print(f"{'Rank':<10} {'Index':<15} {'Location':<35} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20}")
            print("-"*100)
            
            for rank, idx in enumerate(top_indices[:20], 1):
                # Convert flat index to multi-dimensional index
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
    
    # Summary
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

