#!/usr/bin/env python3
"""
Compare C++ and Python patchify outputs using the same fnet/inet inputs.

This script:
1. Loads C++ fnet/inet outputs (from .bin files)
2. Runs Python patchify on them
3. Loads C++ patchify outputs (if available)
4. Compares the results
"""

import numpy as np
import torch
import sys
import os

# Add DPVO_onnx to path
sys.path.insert(0, '/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx')

from dpvo.altcorr import correlation as altcorr
from dpvo.utils import coords_grid_with_index

def load_binary_file(filename, dtype=np.float32):
    """Load binary file as numpy array"""
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        return None
    
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=dtype)
    
    print(f"✅ Loaded {filename}: shape={data.shape}, dtype={data.dtype}, "
          f"min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}")
    return data

def reshape_fnet(fnet_data, C=128, H=132, W=240):
    """Reshape fnet from flat array to [C, H, W]"""
    expected_size = C * H * W
    if fnet_data.size != expected_size:
        print(f"⚠️  Warning: fnet size {fnet_data.size} != expected {expected_size}")
        # Try to infer dimensions
        total = fnet_data.size
        if total % C == 0:
            spatial = total // C
            # Try to find H, W that match
            for h in range(100, 200):
                for w in range(200, 300):
                    if h * w == spatial:
                        H, W = h, w
                        print(f"   Inferred dimensions: C={C}, H={H}, W={W}")
                        break
                if h * w == spatial:
                    break
    
    # Reshape to [C, H, W] (CHW format)
    fnet = fnet_data.reshape(C, H, W)
    return fnet, H, W

def reshape_inet(inet_data, C=384, H=132, W=240):
    """Reshape inet from flat array to [C, H, W]"""
    expected_size = C * H * W
    if inet_data.size != expected_size:
        print(f"⚠️  Warning: inet size {inet_data.size} != expected {expected_size}")
        # Try to infer dimensions
        total = inet_data.size
        if total % C == 0:
            spatial = total // C
            for h in range(100, 200):
                for w in range(200, 300):
                    if h * w == spatial:
                        H, W = h, w
                        print(f"   Inferred dimensions: C={C}, H={H}, W={W}")
                        break
                if h * w == spatial:
                    break
    
    # Reshape to [C, H, W] (CHW format)
    inet = inet_data.reshape(C, H, W)
    return inet, H, W

def python_patchify(fmap, imap, coords, P=3, radius=1):
    """
    Run Python patchify on fmap and imap using given coordinates.
    Uses the original Python altcorr.patchify function from DPVO_onnx.
    
    Args:
        fmap: [C, H, W] feature map (128 channels) - numpy array
        imap: [C, H, W] input map (384 channels) - numpy array
        coords: [M, 2] coordinates at feature map resolution - numpy array
        P: patch size (default 3)
        radius: patch radius (default 1, which gives P//2 = 1 for gmap/patches)
    
    Returns:
        gmap: [M, 128, P, P] patches from fmap
        imap_patches: [M, 384, 1, 1] patches from imap (radius=0)
        patches: [M, 3, P, P] coordinate patches
    """
    # Convert to torch tensors
    # Python DPVO uses CUDA, so we need CUDA device for altcorr.patchify
    # (it uses CUDA kernels, not CPU fallback)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert numpy arrays to torch tensors
    # Python Patchifier: fmap shape is [b, n, c, h, w] where b=1, n=1 (batch=1, num_images=1)
    # fmap[0] removes batch dimension: [n, c, h, w] = [1, c, h, w] (still 4D!)
    # Our fmap: [C, H, W] -> need [1, C, H, W] to match Python's fmap[0] shape
    fmap_t = torch.from_numpy(fmap).to(device).unsqueeze(0)  # [1, C, H, W] - matches Python fmap[0]
    imap_t = torch.from_numpy(imap).to(device).unsqueeze(0)  # [1, C, H, W] - matches Python imap[0]
    
    # coords: [M, 2] -> [1, M, 2] (matching Python: coords shape is [n, M, 2] where n=1)
    coords_t = torch.from_numpy(coords).to(device).unsqueeze(0)  # [1, M, 2]
    
    b = 1  # batch size
    n = 1  # number of images (frame count)
    M = coords_t.shape[1]
    
    # Use original Python altcorr.patchify (matches net.py Patchifier.forward exactly)
    # Python Patchifier.forward:
    #   fmap shape: [b, n, c, h, w] = [1, 1, 128, 132, 240]
    #   fmap[0] shape: [n, c, h, w] = [1, 128, 132, 240] (4D!)
    #   coords shape: [n, M, 2] = [1, M, 2]
    #   imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
    #   gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)
    #   patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)
    
    # Extract imap patches (radius=0)
    # imap_t: [1, 384, H, W] matches Python imap[0]: [1, 384, H, W]
    # coords_t: [1, M, 2] matches Python coords: [1, M, 2]
    imap_patches_t = altcorr.patchify(imap_t, coords_t, 0)  # Returns [1, M, 384, 1, 1]
    imap_patches = imap_patches_t.view(b, -1, 384, 1, 1)[0].cpu().numpy()  # [M, 384, 1, 1]
    
    # Extract gmap patches (radius=P//2 = 1 for P=3)
    # fmap_t: [1, 128, H, W] matches Python fmap[0]: [1, 128, H, W]
    # coords_t: [1, M, 2] matches Python coords: [1, M, 2]
    gmap_t = altcorr.patchify(fmap_t, coords_t, P//2)  # Returns [1, M, 128, P, P]
    gmap = gmap_t.view(b, -1, 128, P, P)[0].cpu().numpy()  # [M, 128, P, P]
    
    # Create grid using coords_grid_with_index (matching net.py exactly)
    # Python: disps = torch.ones(b, n, h, w, device="cuda") = torch.ones(1, 1, h, w)
    #         grid, _ = coords_grid_with_index(disps, device=fmap.device)
    #         coords_grid_with_index returns: coords shape [b, n, 3, h, w] = [1, 1, 3, h, w]
    #         grid[0] removes batch dimension: [n, 3, h, w] = [1, 3, h, w] (4D!)
    H, W = fmap.shape[1], fmap.shape[2]
    disps = torch.ones(b, n, H, W, device=device)  # [1, 1, H, W] - matching Python: b=1, n=1
    grid, _ = coords_grid_with_index(disps, device=device)  # [1, 1, 3, H, W] (from torch.stack([x, y, d], dim=2))
    
    # Extract patches (radius=P//2 = 1 for P=3)
    # grid shape: [1, 1, 3, H, W]
    # grid[0] removes batch dimension: [1, 3, H, W] (4D) - matches Python!
    patches_t = altcorr.patchify(grid[0], coords_t, P//2)  # grid[0] is [1, 3, H, W] (4D)
    patches = patches_t.view(b, -1, 3, P, P)[0].cpu().numpy()  # [M, 3, P, P]
    
    return gmap, imap_patches, patches

def print_sample_values(arr, name, max_samples=20):
    """Print sample values from an array in table format"""
    arr = np.asarray(arr)
    
    if arr.size == 0:
        print(f"\n📋 {name}: (empty array)")
        return
    
    # Flatten for sampling
    arr_flat = arr.flatten()
    num_samples = min(max_samples, arr_flat.size)
    
    # Sample evenly across the array
    if arr_flat.size <= max_samples:
        indices = list(range(arr_flat.size))
    else:
        step = arr_flat.size // max_samples
        indices = list(range(0, arr_flat.size, step))[:max_samples]
    
    print(f"\n{'='*80}")
    print(f"{name} Sample Values")
    print(f"{'='*80}")
    print(f"Shape: {arr.shape}, Total elements: {arr.size}")
    print(f"Stats: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, std={arr.std():.6f}")
    print(f"\n{'Index':<15} {'Location':<25} {'Value':<20}")
    print("-"*80)
    
    for idx in indices:
        orig_idx = np.unravel_index(idx, arr.shape)
        idx_str = ', '.join(map(str, orig_idx))
        print(f"{idx:<15} {idx_str:<25} {format_number(arr_flat[idx]):<20}")
    
    print("="*80)

def format_number(val):
    """Format number for display in table"""
    if abs(val) < 1e-6:
        return f"{val:.2e}"
    elif abs(val) < 0.001:
        return f"{val:.6f}"
    else:
        return f"{val:.6f}"

def print_comparison_table(name, arr1, arr2, diff, tolerance=1e-5):
    """Print comparison results in a table format"""
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    num_diff = np.sum(diff > tolerance)
    total = arr1.size
    pct_diff = 100.0 * num_diff / total
    
    # Determine status
    if max_diff < tolerance:
        status = "✅ MATCH"
    elif pct_diff < 1.0:
        status = "⚠️  MOSTLY MATCH"
    else:
        status = "❌ MISMATCH"
    
    print(f"\n{'='*100}")
    print(f"{name.upper()} COMPARISON")
    print(f"{'='*100}")
    print(f"{'Metric':<30} {'Value':<30} {'Details':<40}")
    print("-"*100)
    print(f"{'Shape':<30} {str(arr1.shape):<30} {'':<40}")
    print(f"{'Max Difference':<30} {format_number(max_diff):<30} {'':<40}")
    print(f"{'Mean Difference':<30} {format_number(mean_diff):<30} {'':<40}")
    print(f"{'Mismatches':<30} {f'{num_diff}/{total} ({pct_diff:.2f}%)':<30} {'':<40}")
    print(f"{'Status':<30} {status:<30} {'':<40}")
    
    # Show max difference location
    if arr1.size > 0 and max_diff >= tolerance:
        flat_idx = np.argmax(diff)
        idx = np.unravel_index(flat_idx, arr1.shape)
        idx_str = ', '.join(map(str, idx))
        print(f"{'Max Diff Location':<30} {idx_str:<30} {'':<40}")
        print(f"{'C++ Value':<30} {format_number(arr1[idx]):<30} {'':<40}")
        print(f"{'Python Value':<30} {format_number(arr2[idx]):<30} {'':<40}")
        print(f"{'Difference':<30} {format_number(diff[idx]):<30} {'':<40}")
    
    print("="*100)

def print_sample_values_table(arr1, arr2, name, max_samples=20):
    """Print sample values in a table format"""
    arr1_flat = arr1.flatten()
    arr2_flat = arr2.flatten()
    diff_flat = np.abs(arr1_flat - arr2_flat)
    
    # Get indices for sampling
    num_samples = min(max_samples, arr1_flat.size)
    if arr1_flat.size <= num_samples:
        indices = list(range(arr1_flat.size))
    else:
        step = arr1_flat.size // num_samples
        indices = list(range(0, arr1_flat.size, step))[:num_samples]
    
    print(f"\n{'='*100}")
    print(f"SAMPLE VALUES: {name}")
    print(f"{'='*100}")
    print(f"{'Index':<15} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20} {'Location':<25}")
    print("-"*100)
    
    for idx in indices:
        orig_idx = np.unravel_index(idx, arr1.shape)
        idx_str = ', '.join(map(str, orig_idx))
        cpp_val = arr1_flat[idx]
        py_val = arr2_flat[idx]
        diff_val = diff_flat[idx]
        
        print(f"{idx:<15} {format_number(cpp_val):<20} {format_number(py_val):<20} "
              f"{format_number(diff_val):<20} {idx_str:<25}")
    
    print("="*100)

def compare_arrays(arr1, arr2, name, tolerance=1e-5, show_samples=True):
    """Compare two arrays and print statistics in table format
    
    Returns:
        tuple: (matches: bool, max_diff: float, mean_diff: float)
    """
    arr1 = np.asarray(arr1)
    arr2 = np.asarray(arr2)
    
    if arr1.shape != arr2.shape:
        print(f"\n❌ {name}: Shape mismatch - {arr1.shape} vs {arr2.shape}")
        return False, float('inf'), float('inf')
    
    diff = np.abs(arr1 - arr2)
    
    # Calculate statistics
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # Print comparison table
    print_comparison_table(name, arr1, arr2, diff, tolerance)
    
    if show_samples:
        # Show sample values in table format
        print_sample_values_table(arr1, arr2, name, max_samples=20)
    
    # Determine if they match
    num_diff = np.sum(diff > tolerance)
    total = arr1.size
    pct_diff = 100.0 * num_diff / total
    
    matches = False
    if max_diff < tolerance:
        matches = True
    elif pct_diff < 1.0:
        matches = True
    
    return matches, max_diff, mean_diff

def load_inputs(frame_idx):
    """Load fnet and inet input files"""
    bin_dir = "bin_file"
    fnet_bin = os.path.join(bin_dir, f"fnet_frame{frame_idx}.bin")
    inet_bin = os.path.join(bin_dir, f"inet_frame{frame_idx}.bin")
    
    print(f"\n📁 Loading inputs for frame {frame_idx}...")
    print(f"   FNet: {fnet_bin}")
    print(f"   INet: {inet_bin}")
    
    fnet_data = load_binary_file(fnet_bin)
    inet_data = load_binary_file(inet_bin)
    
    if fnet_data is None or inet_data is None:
        return None, None
    
    # Reshape to [C, H, W]
    fnet, fmap_H, fmap_W = reshape_fnet(fnet_data, C=128)
    inet, imap_H, imap_W = reshape_inet(inet_data, C=384)
    
    print(f"\n📐 Dimensions:")
    print(f"   FMap: [128, {fmap_H}, {fmap_W}]")
    print(f"   IMap: [384, {imap_H}, {imap_W}]")
    
    if fmap_H != imap_H or fmap_W != imap_W:
        print(f"⚠️  Warning: Feature map dimensions don't match!")
        return None, None
    
    return fnet, inet

def load_or_generate_coordinates(frame_idx, fmap_H, fmap_W, M=4):
    """Load C++ coordinates if available, otherwise generate new ones"""
    bin_dir = "bin_file"
    cpp_coords_bin = os.path.join(bin_dir, f"cpp_coords_frame{frame_idx}.bin")
    
    if os.path.exists(cpp_coords_bin):
        print(f"\n📁 Loading C++ coordinates from {cpp_coords_bin}...")
        coords_data = load_binary_file(cpp_coords_bin)
        if coords_data is not None and coords_data.size == M * 2:
            coords = coords_data.reshape(M, 2)
            print(f"✅ Using C++ coordinates (same as C++ patchify)")
            print(f"\n🎯 Using C++ coordinates (at feature map resolution):")
            for m in range(M):
                print(f"   Patch {m}: ({coords[m, 0]:.2f}, {coords[m, 1]:.2f})")
            return coords
        else:
            print(f"⚠️  C++ coordinates file has wrong size, generating new coordinates")
    
    # Generate coordinates (matching C++ random generation)
    np.random.seed(42)  # Use fixed seed for reproducibility
    coords = np.zeros((M, 2), dtype=np.float32)
    for m in range(M):
        coords[m, 0] = 1.0 + float(np.random.randint(0, fmap_W - 2))  # x
        coords[m, 1] = 1.0 + float(np.random.randint(0, fmap_H - 2))  # y
    
    print(f"\n🎯 Generated {M} random coordinates (at feature map resolution):")
    for m in range(M):
        print(f"   Patch {m}: ({coords[m, 0]:.2f}, {coords[m, 1]:.2f})")
    
    return coords

def save_python_outputs(frame_idx, py_gmap, py_imap_patches, py_patches, coords):
    """Save Python patchify outputs to binary files"""
    bin_dir = "bin_file"
    os.makedirs(bin_dir, exist_ok=True)
    
    py_gmap_path = os.path.join(bin_dir, f"python_gmap_frame{frame_idx}.bin")
    py_imap_path = os.path.join(bin_dir, f"python_imap_frame{frame_idx}.bin")
    py_patches_path = os.path.join(bin_dir, f"python_patches_frame{frame_idx}.bin")
    py_coords_path = os.path.join(bin_dir, f"python_coords_frame{frame_idx}.bin")
    
    py_gmap.tofile(py_gmap_path)
    py_imap_patches.tofile(py_imap_path)
    py_patches.tofile(py_patches_path)
    coords.tofile(py_coords_path)
    
    print(f"\n💾 Saved Python outputs:")
    print(f"   {py_gmap_path}")
    print(f"   {py_imap_path}")
    print(f"   {py_patches_path}")
    print(f"   {py_coords_path}")

def load_cpp_outputs(frame_idx, M=4, P=3):
    """Load and reshape C++ patchify outputs"""
    bin_dir = "bin_file"
    cpp_gmap_bin = os.path.join(bin_dir, f"cpp_gmap_frame{frame_idx}.bin")
    cpp_imap_bin = os.path.join(bin_dir, f"cpp_imap_frame{frame_idx}.bin")
    cpp_patches_bin = os.path.join(bin_dir, f"cpp_patches_frame{frame_idx}.bin")
    
    if not os.path.exists(cpp_gmap_bin):
        return None, None, None
    
    print(f"\n📁 Loading C++ outputs...")
    cpp_gmap_data = load_binary_file(cpp_gmap_bin)
    cpp_imap_data = load_binary_file(cpp_imap_bin)
    cpp_patches_data = load_binary_file(cpp_patches_bin)
    
    if cpp_gmap_data is None:
        return None, None, None
    
    # Reshape C++ outputs
    expected_gmap_size = M * 128 * P * P
    expected_imap_size = M * 384 * 1 * 1
    expected_patches_size = M * 3 * P * P
    
    cpp_gmap = None
    cpp_imap = None
    cpp_patches = None
    
    if cpp_gmap_data.size == expected_gmap_size:
        cpp_gmap = cpp_gmap_data.reshape(M, 128, P, P)
    else:
        print(f"⚠️  Warning: C++ gmap size {cpp_gmap_data.size} != expected {expected_gmap_size}")
    
    if cpp_imap_data.size == expected_imap_size:
        cpp_imap = cpp_imap_data.reshape(M, 384, 1, 1)
    else:
        print(f"⚠️  Warning: C++ imap size {cpp_imap_data.size} != expected {expected_imap_size}")
    
    if cpp_patches_data.size == expected_patches_size:
        cpp_patches = cpp_patches_data.reshape(M, 3, P, P)
    else:
        print(f"⚠️  Warning: C++ patches size {cpp_patches_data.size} != expected {expected_patches_size}")
    
    return cpp_gmap, cpp_imap, cpp_patches

def print_first_patch_details(cpp_gmap, py_gmap, cpp_imap, py_imap, cpp_patches, py_patches):
    """Print detailed values for the first patch in table format"""
    print(f"\n{'='*100}")
    print(f"FIRST PATCH (Patch 0) DETAILED VALUES")
    print(f"{'='*100}")
    
    # GMap details - First channel
    print(f"\n{'='*100}")
    print(f"GMAP - First Channel (Channel 0)")
    print(f"{'='*100}")
    print(f"{'Row':<10} {'C++ Values':<45} {'Python Values':<45}")
    print("-"*100)
    P = cpp_gmap.shape[2]
    for y in range(P):
        cpp_vals = ', '.join(f'{cpp_gmap[0, 0, y, x]:.6f}' for x in range(P))
        py_vals = ', '.join(f'{py_gmap[0, 0, y, x]:.6f}' for x in range(P))
        print(f"{y:<10} {cpp_vals:<45} {py_vals:<45}")
    
    # IMap details - First 10 channels
    print(f"\n{'='*100}")
    print(f"IMAP - First 10 Channels")
    print(f"{'='*100}")
    print(f"{'Channel':<10} {'C++ Value':<20} {'Python Value':<20} {'Difference':<20}")
    print("-"*100)
    for c in range(min(10, cpp_imap.shape[1])):
        cpp_val = cpp_imap[0, c, 0, 0]
        py_val = py_imap[0, c, 0, 0]
        diff = abs(cpp_val - py_val)
        print(f"{c:<10} {format_number(cpp_val):<20} {format_number(py_val):<20} {format_number(diff):<20}")
    
    # Patches details - All 3 channels
    print(f"\n{'='*100}")
    print(f"PATCHES - All 3 Channels (x, y, d)")
    print(f"{'='*100}")
    for c in range(3):
        channel_name = ['X', 'Y', 'D'][c]
        print(f"\nChannel {c} ({channel_name}):")
        print(f"{'Row':<10} {'C++ Values':<45} {'Python Values':<45}")
        print("-"*100)
        for y in range(P):
            cpp_vals = ', '.join(f'{cpp_patches[0, c, y, x]:.2f}' for x in range(P))
            py_vals = ', '.join(f'{py_patches[0, c, y, x]:.2f}' for x in range(P))
            print(f"{y:<10} {cpp_vals:<45} {py_vals:<45}")
    
    print("="*100)

def run_comparisons(cpp_gmap, py_gmap, cpp_imap, py_imap, cpp_patches, py_patches):
    """Run detailed comparisons between C++ and Python outputs"""
    print(f"\n🔍 Comparing C++ vs Python outputs...")
    
    # Show first patch details
    print_first_patch_details(cpp_gmap, py_gmap, cpp_imap, py_imap, cpp_patches, py_patches)
    
    # Run comparisons with table output
    gmap_match, gmap_max_diff, gmap_mean_diff = compare_arrays(cpp_gmap, py_gmap, "gmap (patches from fmap)", tolerance=1e-4, show_samples=True)
    imap_match, imap_max_diff, imap_mean_diff = compare_arrays(cpp_imap, py_imap, "imap (patches from inet)", tolerance=1e-4, show_samples=True)
    patches_match, patches_max_diff, patches_mean_diff = compare_arrays(cpp_patches, py_patches, "patches (coordinate patches)", tolerance=1e-4, show_samples=True)
    
    # Summary table
    print(f"\n{'='*100}")
    print(f"SUMMARY")
    print(f"{'='*100}")
    print(f"{'Output':<30} {'Status':<20}")
    print("-"*100)
    print(f"{'gmap':<30} {'✅ MATCH' if gmap_match else '❌ MISMATCH':<20}")
    print(f"{'imap':<30} {'✅ MATCH' if imap_match else '❌ MISMATCH':<20}")
    print(f"{'patches':<30} {'✅ MATCH' if patches_match else '❌ MISMATCH':<20}")
    print("="*100)
    
    # Calculate overall max_diff and mean_diff across all outputs
    max_diffs = [gmap_max_diff, imap_max_diff, patches_max_diff]
    mean_diffs = [gmap_mean_diff, imap_mean_diff, patches_mean_diff]
    
    # Filter out inf values (from shape mismatches)
    max_diffs_valid = [d for d in max_diffs if d != float('inf')]
    mean_diffs_valid = [d for d in mean_diffs if d != float('inf')]
    
    if max_diffs_valid:
        overall_max_diff = max(max_diffs_valid)
    else:
        overall_max_diff = float('inf')
    
    if mean_diffs_valid:
        overall_mean_diff = sum(mean_diffs_valid) / len(mean_diffs_valid)
    else:
        overall_mean_diff = float('inf')
    
    # Print parseable format for run_all_comparisons.py
    print(f"\n   PATCHIFY_MAX_DIFF={overall_max_diff:.10e}")
    print(f"   PATCHIFY_MEAN_DIFF={overall_mean_diff:.10e}")
    
    return gmap_match and imap_match and patches_match

def main():
    print("=" * 80)
    print("PATCHIFY COMPARISON: C++ vs Python")
    print("=" * 80)
    
    # Parse command line arguments
    frame_idx = 0
    if len(sys.argv) > 1:
        frame_idx = int(sys.argv[1])
    
    # Constants
    M = 4  # Number of patches
    P = 3  # Patch size
    radius = P // 2  # radius = 1
    
    # Step 1: Load inputs
    fnet, inet = load_inputs(frame_idx)
    if fnet is None or inet is None:
        print("❌ Failed to load input files")
        return 1
    
    fmap_H, fmap_W = fnet.shape[1], fnet.shape[2]
    
    # Step 2: Load or generate coordinates
    coords = load_or_generate_coordinates(frame_idx, fmap_H, fmap_W, M)
    
    # Step 3: Run Python patchify
    print(f"\n🐍 Running Python patchify...")
    py_gmap, py_imap_patches, py_patches = python_patchify(fnet, inet, coords, P=P, radius=radius)
    
    print(f"   Python gmap shape: {py_gmap.shape}")
    print(f"   Python imap_patches shape: {py_imap_patches.shape}")
    print(f"   Python patches shape: {py_patches.shape}")
    
    # Step 4: Save Python outputs
    save_python_outputs(frame_idx, py_gmap, py_imap_patches, py_patches, coords)
    
    # Step 5: Load C++ outputs and compare
    cpp_gmap, cpp_imap, cpp_patches = load_cpp_outputs(frame_idx, M, P)
    
    if cpp_gmap is not None and cpp_imap is not None and cpp_patches is not None:
        run_comparisons(cpp_gmap, py_gmap, cpp_imap, py_imap_patches, cpp_patches, py_patches)
    else:
        print(f"\n⚠️  C++ outputs not found. To compare:")
        print(f"   1. Run C++ code to generate bin_file/cpp_*_frame{frame_idx}.bin files")
        print(f"   2. Re-run this script")
    
    print(f"\n✅ Python patchify completed!")
    print(f"   Use the saved .bin files in bin_file/ folder to compare with C++ outputs")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

