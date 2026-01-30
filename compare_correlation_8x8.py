#!/usr/bin/env python3
"""
Compare C++ and Python 8x8 correlation values to identify where the mismatch occurs.
This helps determine if the issue is in the 8x8 computation or the bilinear wrapper.
"""

import numpy as np
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add DPVO_onnx to path
dpvo_path = Path(__file__).parent.parent / "DPVO_onnx"
if dpvo_path.exists():
    sys.path.insert(0, str(dpvo_path))

from dpvo.altcorr.correlation_kernel import corr_torch_forward_fp16

def corr_torch_forward_fp16_8x8_only(fmap1, fmap2, coords, ii, jj, radius, device='cuda'):
    """
    Modified version that returns 8x8 correlation BEFORE bilinear wrapper.
    Based on corr_torch_forward_fp16 but stops before the wrapper interpolation.
    """
    B, M, C, H, W = fmap1.shape
    _, _, _, H2, W2 = fmap2.shape
    
    dtype = fmap1.dtype
    chunk_size = 8000
    
    coords = coords.half()
    D = 2 * radius + 2  # 8x8 internal
    
    # output: 8x8 correlation (before wrapper)
    corr_8x8 = torch.empty((B, M, D, D, H, W), device=device, dtype=dtype)
    
    # offsets
    offs = torch.arange(-radius, radius + 2, device=device, dtype=dtype)
    oy, ox = torch.meshgrid(offs, offs, indexing='ij')  # [D,D]
    ox = ox.view(1, 1, D, D, 1, 1)
    oy = oy.view(1, 1, D, D, 1, 1)
    
    # process in chunks
    for m0 in range(0, M, chunk_size):
        m1 = min(m0 + chunk_size, M)
        mc = m1 - m0
        
        ii_c = ii[m0:m1]
        jj_c = jj[m0:m1]
        
        f1 = fmap1[:, ii_c]        # [B, mc, C, H, W]
        f2 = fmap2[:, jj_c]        # [B, mc, C, H2, W2]
        
        x = coords[:, m0:m1, 0]
        y = coords[:, m0:m1, 1]
        
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        
        # grid for sampling
        gx = x0.unsqueeze(2).unsqueeze(2) + ox  # [B, mc, D, D, H, W]
        gy = y0.unsqueeze(2).unsqueeze(2) + oy
        
        gx = 2 * gx / (W2 - 1) - 1
        gy = 2 * gy / (H2 - 1) - 1
        
        grid = torch.stack([gx, gy], dim=-1).view(B * mc, D * D * H * W, 1, 2)
        f2_view = f2.view(B * mc, C, H2, W2)
        
        sampled = F.grid_sample(
            f2_view, grid, mode='bilinear', align_corners=True
        )  # [B*mc, C, D*D*H*W, 1]
        
        sampled = sampled.view(B, mc, C, D, D, H, W)
        
        # dot product over channels
        f1e = f1.unsqueeze(3).unsqueeze(3)  # [B, mc, C, 1, 1, H, W]
        corr_8x8[:, m0:m1] = (f1e * sampled).sum(dim=2)
    
    return corr_8x8

def load_metadata(frame_num, bin_dir="bin_file"):
    """Load metadata from binary or text file."""
    meta_file_txt = Path(bin_dir) / f"corr_frame{frame_num}_metadata.txt"
    meta_file_bin = Path(bin_dir) / f"corr_frame{frame_num}_meta.bin"
    
    meta = {}
    if meta_file_bin.exists():
        # Load from binary file (same format as compare_correlation_outputs.py)
        meta_data = np.fromfile(meta_file_bin, dtype=np.int32)
        if len(meta_data) >= 11:
            meta = {
                'num_active': int(meta_data[0]),
                'M': int(meta_data[1]),
                'P': int(meta_data[2]),
                'D': int(meta_data[3]),
                'num_frames': int(meta_data[4]),
                'num_gmap_frames': int(meta_data[5]),
                'fmap1_H': int(meta_data[6]),
                'fmap1_W': int(meta_data[7]),
                'fmap2_H': int(meta_data[8]),
                'fmap2_W': int(meta_data[9]),
                'feature_dim': int(meta_data[10])
            }
        else:
            raise ValueError(f"Invalid metadata file: expected 11 int32 values, got {len(meta_data)}")
    elif meta_file_txt.exists():
        # Load from text file (fallback)
        with open(meta_file_txt, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    try:
                        meta[key.strip()] = int(value.strip())
                    except ValueError:
                        try:
                            meta[key.strip()] = float(value.strip())
                        except ValueError:
                            meta[key.strip()] = value.strip()
    else:
        raise FileNotFoundError(f"Metadata file not found: {meta_file_bin} or {meta_file_txt}")
    
    return meta


def load_cpp_8x8_buffers(frame_num, meta, bin_dir="bin_file"):
    """Load C++ 8x8 correlation buffers."""
    corr1_8x8_cpp_file = Path(bin_dir) / f"corr_frame{frame_num}_8x8_level0.bin"
    corr2_8x8_cpp_file = Path(bin_dir) / f"corr_frame{frame_num}_8x8_level1.bin"
    
    if not corr1_8x8_cpp_file.exists() or not corr2_8x8_cpp_file.exists():
        print(f"\n❌ C++ 8x8 buffers not found!")
        print(f"   Expected: {corr1_8x8_cpp_file}")
        print(f"   Expected: {corr2_8x8_cpp_file}")
        print(f"\n   Please recompile and run C++ code with TARGET_FRAME={frame_num}")
        return None, None
    
    num_active = meta['num_active']
    P = meta['P']
    D_8x8 = 8  # 2*R+2
    
    corr1_8x8_cpp = np.fromfile(corr1_8x8_cpp_file, dtype=np.float32)
    corr2_8x8_cpp = np.fromfile(corr2_8x8_cpp_file, dtype=np.float32)
    
    corr1_8x8_cpp = corr1_8x8_cpp.reshape(num_active, D_8x8, D_8x8, P, P)
    corr2_8x8_cpp = corr2_8x8_cpp.reshape(num_active, D_8x8, D_8x8, P, P)
    
    print(f"\n✅ Loaded C++ 8x8 buffers:")
    print(f"   Level 0 shape: {corr1_8x8_cpp.shape}")
    print(f"   Level 1 shape: {corr2_8x8_cpp.shape}")
    
    return corr1_8x8_cpp, corr2_8x8_cpp


def load_correlation_inputs(frame_num, meta, bin_dir="bin_file"):
    """Load correlation inputs (coords, gmap, fmap1, fmap2, kk, jj)."""
    coords_file = Path(bin_dir) / f"corr_frame{frame_num}_coords.bin"
    gmap_file = Path(bin_dir) / f"corr_frame{frame_num}_gmap.bin"
    fmap1_file = Path(bin_dir) / f"corr_frame{frame_num}_fmap1.bin"
    fmap2_file = Path(bin_dir) / f"corr_frame{frame_num}_fmap2.bin"
    kk_file = Path(bin_dir) / f"corr_frame{frame_num}_kk.bin"
    jj_file = Path(bin_dir) / f"corr_frame{frame_num}_jj.bin"
    
    coords = np.fromfile(coords_file, dtype=np.float32)
    gmap = np.fromfile(gmap_file, dtype=np.float32)
    fmap1 = np.fromfile(fmap1_file, dtype=np.float32)
    fmap2 = np.fromfile(fmap2_file, dtype=np.float32)
    kk = np.fromfile(kk_file, dtype=np.int32)
    jj = np.fromfile(jj_file, dtype=np.int32)
    
    num_active = meta['num_active']
    P = meta['P']
    
    # Reshape inputs
    coords = coords.reshape(num_active, 2, P, P)
    gmap = gmap.reshape(num_active, 128, P, P)
    fmap1 = fmap1.reshape(num_active, 128, meta['fmap1_H'], meta['fmap1_W'])
    fmap2 = fmap2.reshape(num_active, 128, meta['fmap2_H'], meta['fmap2_W'])
    
    return coords, gmap, fmap1, fmap2, kk, jj


def prepare_python_tensors(coords, gmap, fmap1, fmap2, device):
    """Convert numpy arrays to PyTorch tensors with batch dimension."""
    coords_py = torch.from_numpy(coords).unsqueeze(0).to(device)
    gmap_py = torch.from_numpy(gmap).unsqueeze(0).to(device)
    fmap1_py = torch.from_numpy(fmap1).unsqueeze(0).to(device)
    fmap2_py = torch.from_numpy(fmap2).unsqueeze(0).to(device)
    
    return coords_py, gmap_py, fmap1_py, fmap2_py


def create_sequential_indices(num_active, device):
    """
    Create sequential indices for flattened arrays.
    
    IMPORTANT: Since we're passing flattened arrays [1, num_active, ...] where
    each element corresponds to an edge, we should use sequential indices
    [0, 1, 2, ..., num_active-1] so Python selects the correct edge's data.
    If we use ring buffer indices (kk % (M*pmem)), Python would select the wrong edge!
    Example: If kk[23]=9, then ii1_py[23]=9, and Python selects fmap1[:, 9],
    but gmap_slices[9] contains patch from kk[9], not kk[23]!
    """
    ii1_py = torch.arange(num_active, dtype=torch.long, device=device)
    jj1_py = torch.arange(num_active, dtype=torch.long, device=device)
    return ii1_py, jj1_py


def compute_python_8x8_correlation(gmap_py, fmap1_py, fmap2_py, coords_py, 
                                   ii1_py, jj1_py, radius=3, device='cuda'):
    """Compute Python 8x8 correlation for both pyramid levels."""
    print("\nComputing Python 8x8 correlation...")
    
    corr1_8x8_py = corr_torch_forward_fp16_8x8_only(
        gmap_py, fmap1_py, coords_py / 1.0, ii1_py, jj1_py, radius, device=device
    )
    corr2_8x8_py = corr_torch_forward_fp16_8x8_only(
        gmap_py, fmap2_py, coords_py / 4.0, ii1_py, jj1_py, radius, device=device
    )
    
    # Remove batch dimension: [M, D, D, H, W]
    corr1_8x8_py = corr1_8x8_py[0].cpu().numpy()
    corr2_8x8_py = corr2_8x8_py[0].cpu().numpy()
    
    print(f"Python 8x8 shape: {corr1_8x8_py.shape}")
    
    return corr1_8x8_py, corr2_8x8_py


def compute_differences(corr_cpp, corr_py):
    """Compute absolute differences between C++ and Python correlation values."""
    return np.abs(corr_cpp - corr_py)


def print_comparison_summary(diff1, diff2, level_name="8x8"):
    """Print comparison summary statistics."""
    print("\n" + "="*70)
    print(f"{level_name} Correlation Comparison")
    print("="*70)
    
    print(f"\nLevel 0 ({level_name}):")
    print(f"  Max Diff: {diff1.max():.6f}")
    print(f"  Mean Diff: {diff1.mean():.6f}")
    mismatched1 = (diff1 > 1e-5).sum()
    print(f"  Mismatched: {mismatched1}/{diff1.size} ({mismatched1/diff1.size*100:.2f}%)")
    
    print(f"\nLevel 1 ({level_name}):")
    print(f"  Max Diff: {diff2.max():.6f}")
    print(f"  Mean Diff: {diff2.mean():.6f}")
    mismatched2 = (diff2 > 1e-5).sum()
    print(f"  Mismatched: {mismatched2}/{diff2.size} ({mismatched2/diff2.size*100:.2f}%)")


def print_top_mismatches(corr_cpp, corr_py, diff, top_k=10, level_name="Level 0"):
    """Print top K largest mismatches."""
    print("\n" + "="*70)
    print(f"Top {top_k} Largest 8x8 Differences ({level_name})")
    print("="*70)
    
    diff_flat = diff.flatten()
    top_indices = np.argsort(diff_flat)[-top_k:][::-1]
    
    for idx in top_indices:
        flat_idx = np.unravel_index(idx, diff.shape)
        edge, ii, jj, i0, j0 = flat_idx
        cpp_val = corr_cpp[edge, ii, jj, i0, j0]
        py_val = corr_py[edge, ii, jj, i0, j0]
        diff_val = diff[edge, ii, jj, i0, j0]
        print(f"  Edge {edge}, Window ({ii},{jj}), Patch ({i0},{j0}): "
              f"C++={cpp_val:.6f}, Python={py_val:.6f}, Diff={diff_val:.6f}")


def print_specific_location(corr_cpp, corr_py, diff, edge, corr_ii, corr_jj, 
                           i0, j0, D_8x8=8, level_name="8x8"):
    """Print correlation values at a specific location and surrounding values."""
    print("\n" + "="*70)
    print("Checking Specific Mismatch Locations")
    print("="*70)
    print(f"\nEdge {edge}, Window ({corr_ii},{corr_jj}), Patch ({i0},{j0}) - {level_name} values:")
    
    cpp_val = corr_cpp[edge, corr_ii, corr_jj, i0, j0]
    py_val = corr_py[edge, corr_ii, corr_jj, i0, j0]
    diff_val = diff[edge, corr_ii, corr_jj, i0, j0]
    
    print(f"  C++ {level_name}[{edge}, {corr_ii}, {corr_jj}, {i0}, {j0}]: {cpp_val:.6f}")
    print(f"  Python {level_name}[{edge}, {corr_ii}, {corr_jj}, {i0}, {j0}]: {py_val:.6f}")
    print(f"  Difference: {diff_val:.6f}")
    
    print(f"\n  Surrounding {level_name} values (C++):")
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ii = corr_ii + di
            jj = corr_jj + dj
            if 0 <= ii < D_8x8 and 0 <= jj < D_8x8:
                val = corr_cpp[edge, ii, jj, i0, j0]
                print(f"    ({ii},{jj}): {val:.6f}")
    
    print(f"\n  Surrounding {level_name} values (Python):")
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ii = corr_ii + di
            jj = corr_jj + dj
            if 0 <= ii < D_8x8 and 0 <= jj < D_8x8:
                val = corr_py[edge, ii, jj, i0, j0]
                print(f"    ({ii},{jj}): {val:.6f}")


def compare_8x8_correlation(frame_num, bin_dir="bin_file"):
    """Compare C++ and Python 8x8 correlation values."""
    print("="*70)
    print(f"Comparing 8x8 Correlation for Frame {frame_num}")
    print("="*70)
    
    # Load metadata
    meta = load_metadata(frame_num, bin_dir)
    
    # Load C++ 8x8 buffers
    corr1_8x8_cpp, corr2_8x8_cpp = load_cpp_8x8_buffers(frame_num, meta, bin_dir)
    if corr1_8x8_cpp is None or corr2_8x8_cpp is None:
        return
    
    # Load correlation inputs
    coords, gmap, fmap1, fmap2, kk, jj = load_correlation_inputs(frame_num, meta, bin_dir)
    
    # Prepare PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords_py, gmap_py, fmap1_py, fmap2_py = prepare_python_tensors(
        coords, gmap, fmap1, fmap2, device
    )
    
    # Create sequential indices for flattened arrays
    num_active = meta['num_active']
    ii1_py, jj1_py = create_sequential_indices(num_active, device)
    
    # Compute Python 8x8 correlation
    corr1_8x8_py, corr2_8x8_py = compute_python_8x8_correlation(
        gmap_py, fmap1_py, fmap2_py, coords_py, ii1_py, jj1_py, radius=3, device=device
    )
    
    # Compute differences
    diff1_8x8 = compute_differences(corr1_8x8_cpp, corr1_8x8_py)
    diff2_8x8 = compute_differences(corr2_8x8_cpp, corr2_8x8_py)
    
    # Print comparison summary
    print_comparison_summary(diff1_8x8, diff2_8x8, level_name="8x8")
    
    # Print top mismatches
    print_top_mismatches(corr1_8x8_cpp, corr1_8x8_py, diff1_8x8, top_k=10, level_name="Level 0")
    
    # Print specific location (use a valid edge index, default to center edge or edge 0 if available)
    # Use edge index that exists (num_active - 1 is the last valid edge)
    sample_edge = min(25, num_active - 1)  # Use edge 25 (from top mismatches) or last valid edge
    if num_active > 0:
        print_specific_location(
            corr1_8x8_cpp, corr1_8x8_py, diff1_8x8,
            edge=sample_edge, corr_ii=3, corr_jj=3, i0=1, j0=1, D_8x8=8, level_name="8x8"
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, required=True, help="Frame number")
    parser.add_argument("--bin-dir", type=str, default="bin_file", help="Binary file directory")
    args = parser.parse_args()
    
    compare_8x8_correlation(args.frame, args.bin_dir)

