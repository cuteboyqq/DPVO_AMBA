#!/usr/bin/env python3
"""
Compare C++ and Python DPVO keyframe() function results.

This script:
1. Loads C++ keyframe input data from binary files
2. Runs Python DPVO keyframe() function with that input
3. Saves Python keyframe output to binary files
4. Compares C++ and Python keyframe results
"""

import numpy as np
import torch
import sys
import os
import argparse
from pathlib import Path

# Add DPVO Python path
sys.path.insert(0, '/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx')

try:
    from dpvo.dpvo import DPVO as PythonDPVO
    from dpvo.config import cfg as default_cfg
    from dpvo.lietorch import SE3
    from yacs.config import CfgNode as CN
except ImportError as e:
    print(f"Error importing Python DPVO: {e}")
    print("Make sure DPVO_onnx is in the Python path")
    sys.exit(1)


def load_keyframe_metadata(bin_dir: str, frame_num: int) -> dict:
    """Load keyframe metadata from text file."""
    metadata_file = os.path.join(bin_dir, f"keyframe_metadata_frame{frame_num}.txt")
    metadata = {}
    
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                try:
                    # Try to convert to int or float
                    if '.' in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                except ValueError:
                    metadata[key] = value
    
    return metadata


def load_binary_file(filename: str, dtype=np.float32):
    """Load binary file as numpy array."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    data = np.fromfile(filename, dtype=dtype)
    return data


def load_poses(filename: str, n: int) -> np.ndarray:
    """Load poses [N, 7] format: [tx, ty, tz, qx, qy, qz, qw]."""
    data = load_binary_file(filename, dtype=np.float32)
    assert len(data) == n * 7, f"Expected {n * 7} floats, got {len(data)}"
    return data.reshape(n, 7)


def load_patches(filename: str, n: int, m: int, p: int) -> np.ndarray:
    """Load patches [N*M, 3, P, P]."""
    data = load_binary_file(filename, dtype=np.float32)
    expected_size = n * m * 3 * p * p
    assert len(data) == expected_size, f"Expected {expected_size} floats, got {len(data)}"
    return data.reshape(n * m, 3, p, p)


def load_intrinsics(filename: str, n: int) -> np.ndarray:
    """Load intrinsics [N, 4] format: [fx, fy, cx, cy]."""
    data = load_binary_file(filename, dtype=np.float32)
    assert len(data) == n * 4, f"Expected {n * 4} floats, got {len(data)}"
    return data.reshape(n, 4)


def load_timestamps(filename: str, n: int) -> np.ndarray:
    """Load timestamps [N] (int64)."""
    data = load_binary_file(filename, dtype=np.int64)
    assert len(data) == n, f"Expected {n} int64s, got {len(data)}"
    return data


def load_colors(filename: str, n: int, m: int) -> np.ndarray:
    """Load colors [N, M, 3] (uint8)."""
    data = load_binary_file(filename, dtype=np.uint8)
    assert len(data) == n * m * 3, f"Expected {n * m * 3} uint8s, got {len(data)}"
    return data.reshape(n, m, 3)


def load_index(filename: str, n: int, m: int) -> np.ndarray:
    """Load index [N, M] (int32)."""
    data = load_binary_file(filename, dtype=np.int32)
    assert len(data) == n * m, f"Expected {n * m} int32s, got {len(data)}"
    return data.reshape(n, m)


def load_ix(filename: str, size: int) -> np.ndarray:
    """Load ix [N*M] (int32)."""
    data = load_binary_file(filename, dtype=np.int32)
    assert len(data) == size, f"Expected {size} int32s, got {len(data)}"
    return data


def load_edge_indices(filename: str, num_edges: int) -> np.ndarray:
    """Load edge indices [num_edges] (int32)."""
    data = load_binary_file(filename, dtype=np.int32)
    assert len(data) == num_edges, f"Expected {num_edges} int32s, got {len(data)}"
    return data


def save_poses(filename: str, poses: torch.Tensor, n: int = None):
    """Save poses [N, 7] format: [tx, ty, tz, qx, qy, qz, qw].
    
    Args:
        filename: Output filename
        poses: Poses tensor - can be [N, 7] or [1, N, 7] or property that returns [1, N, 7]
        n: Number of active frames to save (if None, saves all)
    """
    if isinstance(poses, torch.Tensor):
        poses_np = poses.detach().cpu().numpy()
    else:
        poses_np = poses
    
    # Python DPVO poses are [1, N, 7], extract [N, 7]
    if len(poses_np.shape) == 3:
        poses_np = poses_np[0]
    
    # Only save active frames if n is specified
    if n is not None:
        poses_np = poses_np[:n]
    
    poses_np.astype(np.float32).tofile(filename)


def save_patches(filename: str, patches: torch.Tensor, n: int = None, m: int = None):
    """Save patches [N*M, 3, P, P].
    
    Args:
        filename: Output filename
        patches: Patches tensor - can be [N*M, 3, P, P] or [1, N*M, 3, P, P] or property
        n: Number of active frames (if None, saves all)
        m: Number of patches per frame (if None, saves all)
    """
    if isinstance(patches, torch.Tensor):
        patches_np = patches.detach().cpu().numpy()
    else:
        patches_np = patches
    
    # Python DPVO patches are [1, N*M, 3, P, P], extract [N*M, 3, P, P]
    if len(patches_np.shape) == 5:
        patches_np = patches_np[0]
    
    # Only save active frames if n and m are specified
    if n is not None and m is not None:
        patches_np = patches_np[:n * m]
    
    patches_np.astype(np.float32).tofile(filename)


def save_intrinsics(filename: str, intrinsics: torch.Tensor, n: int = None):
    """Save intrinsics [N, 4] format: [fx, fy, cx, cy].
    
    Args:
        filename: Output filename
        intrinsics: Intrinsics tensor - can be [N, 4] or [1, N, 4] or property
        n: Number of active frames (if None, saves all)
    """
    if isinstance(intrinsics, torch.Tensor):
        intrinsics_np = intrinsics.detach().cpu().numpy()
    else:
        intrinsics_np = intrinsics
    
    # Python DPVO intrinsics are [1, N, 4], extract [N, 4]
    if len(intrinsics_np.shape) == 3:
        intrinsics_np = intrinsics_np[0]
    
    # Only save active frames if n is specified
    if n is not None:
        intrinsics_np = intrinsics_np[:n]
    
    intrinsics_np.astype(np.float32).tofile(filename)


def save_timestamps(filename: str, tstamps, n: int = None):
    """Save timestamps [N] (int64).
    
    Args:
        filename: Output filename
        tstamps: Timestamps - can be torch.Tensor or numpy array
        n: Number of active frames (if None, saves all)
    """
    if isinstance(tstamps, torch.Tensor):
        tstamps_np = tstamps.detach().cpu().numpy()
    else:
        tstamps_np = tstamps
    
    # Only save active frames if n is specified
    if n is not None:
        tstamps_np = tstamps_np[:n]
    
    tstamps_np.astype(np.int64).tofile(filename)


def save_colors(filename: str, colors: torch.Tensor, n: int = None):
    """Save colors [N, M, 3] (uint8).
    
    Args:
        filename: Output filename
        colors: Colors tensor - can be torch.Tensor or numpy array
        n: Number of active frames (if None, saves all)
    """
    if isinstance(colors, torch.Tensor):
        colors_np = colors.detach().cpu().numpy()
    else:
        colors_np = colors
    
    # Only save active frames if n is specified
    if n is not None:
        colors_np = colors_np[:n]
    
    colors_np.astype(np.uint8).tofile(filename)


def save_index(filename: str, index: torch.Tensor, n: int = None):
    """Save index [N, M] (int32).
    
    Args:
        filename: Output filename
        index: Index tensor - can be torch.Tensor or numpy array
        n: Number of active frames (if None, saves all)
    """
    if isinstance(index, torch.Tensor):
        index_np = index.detach().cpu().numpy()
    else:
        index_np = index
    
    # Only save active frames if n is specified
    if n is not None:
        index_np = index_np[:n]
    
    index_np.astype(np.int32).tofile(filename)


def save_ix(filename: str, ix: torch.Tensor):
    """Save ix [N*M] (int32)."""
    if isinstance(ix, torch.Tensor):
        ix_np = ix.detach().cpu().numpy()
    else:
        ix_np = ix
    
    ix_np.astype(np.int32).tofile(filename)


def save_edge_indices(filename: str, indices: torch.Tensor):
    """Save edge indices [num_edges] (int32)."""
    if isinstance(indices, torch.Tensor):
        indices_np = indices.detach().cpu().numpy()
    else:
        indices_np = indices
    
    indices_np.astype(np.int32).tofile(filename)


def load_cpp_keyframe_inputs(bin_dir: str, frame_num: int, metadata: dict):
    """Load C++ keyframe input data."""
    frame_suffix = f"frame{frame_num}"
    
    n = metadata['n_before']
    m = metadata['m_before']
    num_edges = metadata['num_edges_before']
    M = 4  # PATCHES_PER_FRAME
    P = 3  # PATCH_SIZE
    
    poses = load_poses(os.path.join(bin_dir, f"keyframe_poses_before_{frame_suffix}.bin"), n)
    patches = load_patches(os.path.join(bin_dir, f"keyframe_patches_before_{frame_suffix}.bin"), n, M, P)
    intrinsics = load_intrinsics(os.path.join(bin_dir, f"keyframe_intrinsics_before_{frame_suffix}.bin"), n)
    tstamps = load_timestamps(os.path.join(bin_dir, f"keyframe_tstamps_before_{frame_suffix}.bin"), n)
    colors = load_colors(os.path.join(bin_dir, f"keyframe_colors_before_{frame_suffix}.bin"), n, M)
    index = load_index(os.path.join(bin_dir, f"keyframe_index_before_{frame_suffix}.bin"), n, M)
    ix = load_ix(os.path.join(bin_dir, f"keyframe_ix_before_{frame_suffix}.bin"), n * M)
    ii = load_edge_indices(os.path.join(bin_dir, f"keyframe_ii_before_{frame_suffix}.bin"), num_edges)
    jj = load_edge_indices(os.path.join(bin_dir, f"keyframe_jj_before_{frame_suffix}.bin"), num_edges)
    kk = load_edge_indices(os.path.join(bin_dir, f"keyframe_kk_before_{frame_suffix}.bin"), num_edges)
    
    return {
        'n': n,
        'm': m,
        'num_edges': num_edges,
        'M': M,
        'P': P,
        'poses': poses,
        'patches': patches,
        'intrinsics': intrinsics,
        'tstamps': tstamps,
        'colors': colors,
        'index': index,
        'ix': ix,
        'ii': ii,
        'jj': jj,
        'kk': kk,
    }


def setup_python_dpvo(cpp_inputs: dict, metadata: dict, network_path: str = None):
    """Setup Python DPVO object with C++ input data.
    
    Note: Python DPVO's constructor requires a network path, but keyframe() doesn't actually use it.
    The network is only needed to set DIM, RES, P attributes. We can work around this by:
    1. Using a dummy network path (if network_path is None)
    2. Or using the provided network path
    """
    # Clone default config and update with C++ values
    cfg = default_cfg.clone()
    cfg.KEYFRAME_INDEX = metadata['KEYFRAME_INDEX']
    cfg.KEYFRAME_THRESH = metadata['KEYFRAME_THRESH']
    cfg.PATCH_LIFETIME = metadata['PATCH_LIFETIME']
    cfg.REMOVAL_WINDOW = metadata['REMOVAL_WINDOW']
    cfg.PATCHES_PER_FRAME = cpp_inputs['M']
    cfg.BUFFER_SIZE = 4096
    cfg.MIXED_PRECISION = False
    cfg.LOOP_CLOSURE = False
    cfg.CLASSIC_LOOP_CLOSURE = False
    cfg.MAX_EDGE_AGE = 360
    cfg.MAX_EDGES = 10000
    
    # Create DPVO object
    # Note: Python DPVO constructor requires network path to set DIM, RES, P attributes,
    # but keyframe() itself doesn't use the network weights. We'll use a dummy path if none provided.
    if network_path is None:
        # Create a minimal dummy network file that Python DPVO can load
        # This is a workaround since keyframe() doesn't actually need the network
        import tempfile
        dummy_network = os.path.join(tempfile.gettempdir(), "dpvo_dummy.pth")
        
        # Create a minimal state dict with just the required structure
        # Python DPVO will extract DIM, RES, P from the network
        dummy_state = {
            'fnet.conv1.weight': torch.randn(32, 3, 3, 3),
            'inet.conv1.weight': torch.randn(32, 3, 3, 3),
            'update.lmbda': torch.tensor(0.0),  # This will be skipped anyway
        }
        torch.save(dummy_state, dummy_network)
        network_path = dummy_network
        print(f"Using dummy network file: {network_path}")
        print("(keyframe() doesn't actually use the network, but Python DPVO constructor requires it)")
    elif not os.path.exists(network_path):
        raise FileNotFoundError(f"Network file not found: {network_path}")
    
    dpvo = PythonDPVO(cfg, network_path, ht=528, wd=960, viz=False)
    
    # Set state from C++ inputs
    n = cpp_inputs['n']
    m = cpp_inputs['m']
    M = cpp_inputs['M']
    P = cpp_inputs['P']
    
    # Convert to torch tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set poses [N, 7] - Python DPVO stores as [N, 7], not [1, N, 7]
    poses_torch = torch.from_numpy(cpp_inputs['poses']).float().to(device)
    dpvo.pg.poses_[:n] = poses_torch
    
    # Set patches [N, M, 3, P, P] - Python DPVO stores as [N, M, 3, P, P] in pg.patches_
    # C++ patches are [N*M, 3, P, P], need to reshape to [N, M, 3, P, P]
    patches_flat = torch.from_numpy(cpp_inputs['patches']).float().to(device)  # [N*M, 3, P, P]
    patches_reshaped = patches_flat.view(n, M, 3, P, P)  # [N, M, 3, P, P]
    dpvo.pg.patches_[:n] = patches_reshaped
    
    # Set intrinsics [N, 4] - Python DPVO stores as [N, 4] in pg.intrinsics_
    intrinsics_torch = torch.from_numpy(cpp_inputs['intrinsics']).float().to(device)
    dpvo.pg.intrinsics_[:n] = intrinsics_torch
    
    # Set timestamps [N] - Python DPVO stores as numpy array, not torch tensor
    tstamps_np = cpp_inputs['tstamps']  # Already numpy int64
    dpvo.pg.tstamps_[:n] = tstamps_np
    
    # Set colors [N, M, 3]
    colors_torch = torch.from_numpy(cpp_inputs['colors']).byte().to(device)
    dpvo.pg.colors_[:n] = colors_torch
    
    # Set index [N, M]
    index_torch = torch.from_numpy(cpp_inputs['index']).long().to(device)
    dpvo.pg.index_[:n] = index_torch
    
    # Set edges
    num_edges = cpp_inputs['num_edges']
    ii_torch = torch.from_numpy(cpp_inputs['ii']).long().to(device)
    jj_torch = torch.from_numpy(cpp_inputs['jj']).long().to(device)
    kk_torch = torch.from_numpy(cpp_inputs['kk']).long().to(device)
    
    dpvo.pg.ii[:num_edges] = ii_torch
    dpvo.pg.jj[:num_edges] = jj_torch
    dpvo.pg.kk[:num_edges] = kk_torch
    dpvo.pg.num_edges = num_edges
    
    # Set counters (Python DPVO uses pg.n and pg.m, but also has dpvo.n and dpvo.m)
    dpvo.pg.n = n
    dpvo.pg.m = m
    dpvo.n = n
    dpvo.m = m
    
    return dpvo


def run_python_keyframe(dpvo: PythonDPVO):
    """Run Python DPVO keyframe() function."""
    # Save state before
    n_before = dpvo.n
    m_before = dpvo.m
    num_edges_before = dpvo.pg.num_edges
    
    # Run keyframe
    dpvo.keyframe()
    
    # Get state after
    n_after = dpvo.n
    m_after = dpvo.m
    num_edges_after = dpvo.pg.num_edges
    
    return {
        'n_before': n_before,
        'm_before': m_before,
        'num_edges_before': num_edges_before,
        'n_after': n_after,
        'm_after': m_after,
        'num_edges_after': num_edges_after,
    }


def save_python_keyframe_outputs(bin_dir: str, frame_num: int, dpvo: PythonDPVO, state: dict):
    """Save Python keyframe output data."""
    frame_suffix = f"frame{frame_num}"
    n_after = state['n_after']
    num_edges_after = state['num_edges_after']
    M = 4
    P = 3
    
    # Save state after keyframe - only save active frames (n_after)
    save_poses(os.path.join(bin_dir, f"keyframe_poses_after_py_{frame_suffix}.bin"), dpvo.pg.poses_, n=n_after)
    save_patches(os.path.join(bin_dir, f"keyframe_patches_after_py_{frame_suffix}.bin"), dpvo.patches, n=n_after, m=M)
    save_intrinsics(os.path.join(bin_dir, f"keyframe_intrinsics_after_py_{frame_suffix}.bin"), dpvo.intrinsics, n=n_after)
    save_timestamps(os.path.join(bin_dir, f"keyframe_tstamps_after_py_{frame_suffix}.bin"), dpvo.pg.tstamps_, n=n_after)
    save_colors(os.path.join(bin_dir, f"keyframe_colors_after_py_{frame_suffix}.bin"), dpvo.pg.colors_, n=n_after)
    save_index(os.path.join(bin_dir, f"keyframe_index_after_py_{frame_suffix}.bin"), dpvo.pg.index_, n=n_after)
    
    # Save ix (flattened index) - only save active frames
    ix_flat = dpvo.pg.index_[:n_after].view(-1)
    save_ix(os.path.join(bin_dir, f"keyframe_ix_after_py_{frame_suffix}.bin"), ix_flat)
    
    # Save edges after
    save_edge_indices(os.path.join(bin_dir, f"keyframe_ii_after_py_{frame_suffix}.bin"), dpvo.pg.ii[:num_edges_after])
    save_edge_indices(os.path.join(bin_dir, f"keyframe_jj_after_py_{frame_suffix}.bin"), dpvo.pg.jj[:num_edges_after])
    save_edge_indices(os.path.join(bin_dir, f"keyframe_kk_after_py_{frame_suffix}.bin"), dpvo.pg.kk[:num_edges_after])


def compare_results(bin_dir: str, frame_num: int, metadata: dict, python_state: dict):
    """Compare C++ and Python keyframe results."""
    frame_suffix = f"frame{frame_num}"
    n_after = python_state['n_after']
    num_edges_after = python_state['num_edges_after']
    M = 4
    P = 3
    
    print("=" * 80)
    print("KEYFRAME COMPARISON RESULTS")
    print("=" * 80)
    print(f"Frame: {frame_num}")
    print()
    
    # Compare metadata - format as table
    print("Metadata Comparison:")
    print("-" * 80)
    print(f"{'Metric':<25} {'C++':<15} {'Python':<15} {'Status':<10}")
    print("-" * 80)
    
    def format_match(cpp_val, py_val):
        match = cpp_val == py_val
        status = "✅ MATCH" if match else "❌ MISMATCH"
        return status
    
    print(f"{'n_before':<25} {metadata['n_before']:<15} {python_state['n_before']:<15} {format_match(metadata['n_before'], python_state['n_before']):<10}")
    print(f"{'m_before':<25} {metadata['m_before']:<15} {python_state['m_before']:<15} {format_match(metadata['m_before'], python_state['m_before']):<10}")
    print(f"{'num_edges_before':<25} {metadata['num_edges_before']:<15} {python_state['num_edges_before']:<15} {format_match(metadata['num_edges_before'], python_state['num_edges_before']):<10}")
    print(f"{'n_after':<25} {metadata['n_after']:<15} {python_state['n_after']:<15} {format_match(metadata['n_after'], python_state['n_after']):<10}")
    print(f"{'m_after':<25} {metadata['m_after']:<15} {python_state['m_after']:<15} {format_match(metadata['m_after'], python_state['m_after']):<10}")
    print(f"{'num_edges_after':<25} {metadata['num_edges_after']:<15} {python_state['num_edges_after']:<15} {format_match(metadata['num_edges_after'], python_state['num_edges_after']):<10}")
    print("-" * 80)
    print()
    
    # Compare poses
    cpp_poses = load_poses(os.path.join(bin_dir, f"keyframe_poses_after_{frame_suffix}.bin"), n_after)
    py_poses = load_poses(os.path.join(bin_dir, f"keyframe_poses_after_py_{frame_suffix}.bin"), n_after)
    
    poses_diff = np.abs(cpp_poses - py_poses)
    poses_max_diff = np.max(poses_diff)
    poses_mean_diff = np.mean(poses_diff)
    
    print(f"Poses Comparison (n={n_after}):")
    print(f"  Max diff: {poses_max_diff:.6e}, Mean diff: {poses_mean_diff:.6e}")
    print(f"  {'✅ MATCH' if poses_max_diff < 1e-4 else '❌ MISMATCH'}")
    print()
    
    # Compare edges - use C++ and Python num_edges_after separately
    cpp_num_edges_after = metadata['num_edges_after']
    py_num_edges_after = python_state['num_edges_after']
    
    cpp_ii = load_edge_indices(os.path.join(bin_dir, f"keyframe_ii_after_{frame_suffix}.bin"), cpp_num_edges_after)
    py_ii = load_edge_indices(os.path.join(bin_dir, f"keyframe_ii_after_py_{frame_suffix}.bin"), py_num_edges_after)
    
    cpp_jj = load_edge_indices(os.path.join(bin_dir, f"keyframe_jj_after_{frame_suffix}.bin"), cpp_num_edges_after)
    py_jj = load_edge_indices(os.path.join(bin_dir, f"keyframe_jj_after_py_{frame_suffix}.bin"), py_num_edges_after)
    
    cpp_kk = load_edge_indices(os.path.join(bin_dir, f"keyframe_kk_after_{frame_suffix}.bin"), cpp_num_edges_after)
    py_kk = load_edge_indices(os.path.join(bin_dir, f"keyframe_kk_after_py_{frame_suffix}.bin"), py_num_edges_after)
    
    # Compare edges - handle different sizes, format as table
    print("Edge Indices Comparison:")
    print("-" * 80)
    print(f"{'Metric':<30} {'C++':<15} {'Python':<15} {'Status':<10}")
    print("-" * 80)
    print(f"{'num_edges_after':<30} {cpp_num_edges_after:<15} {py_num_edges_after:<15} "
          f"{'✅ MATCH' if cpp_num_edges_after == py_num_edges_after else '❌ MISMATCH':<10}")
    print("-" * 80)
    print()
    
    if cpp_num_edges_after != py_num_edges_after:
        print(f"⚠️ Different number of edges: C++={cpp_num_edges_after}, Python={py_num_edges_after}")
        print(f"This indicates a mismatch in edge removal logic!")
        print()
        
        # Compare common edges (up to min size)
        min_size = min(cpp_num_edges_after, py_num_edges_after)
        print(f"Comparing first {min_size} edges:")
        print("-" * 80)
        
        ii_match = np.array_equal(cpp_ii[:min_size], py_ii[:min_size])
        jj_match = np.array_equal(cpp_jj[:min_size], py_jj[:min_size])
        kk_match = np.array_equal(cpp_kk[:min_size], py_kk[:min_size])
        
        print(f"{'Component':<20} {'Status':<15} {'Details':<45}")
        print("-" * 80)
        
        if not ii_match:
            diff_mask = cpp_ii[:min_size] != py_ii[:min_size]
            diff_count = np.sum(diff_mask)
            diff_indices = np.where(diff_mask)[0][:5]
            cpp_vals = cpp_ii[:min_size][diff_mask][:5]
            py_vals = py_ii[:min_size][diff_mask][:5]
            print(f"{'ii (first ' + str(min_size) + ')':<20} {'❌ MISMATCH':<15} "
                  f"{diff_count} mismatches, first 5: C++={list(cpp_vals)}, Py={list(py_vals)}")
        else:
            print(f"{'ii (first ' + str(min_size) + ')':<20} {'✅ MATCH':<15} {'All match'}")
        
        if not jj_match:
            diff_mask = cpp_jj[:min_size] != py_jj[:min_size]
            diff_count = np.sum(diff_mask)
            cpp_vals = cpp_jj[:min_size][diff_mask][:5]
            py_vals = py_jj[:min_size][diff_mask][:5]
            print(f"{'jj (first ' + str(min_size) + ')':<20} {'❌ MISMATCH':<15} "
                  f"{diff_count} mismatches, first 5: C++={list(cpp_vals)}, Py={list(py_vals)}")
        else:
            print(f"{'jj (first ' + str(min_size) + ')':<20} {'✅ MATCH':<15} {'All match'}")
        
        if not kk_match:
            diff_mask = cpp_kk[:min_size] != py_kk[:min_size]
            diff_count = np.sum(diff_mask)
            cpp_vals = cpp_kk[:min_size][diff_mask][:5]
            py_vals = py_kk[:min_size][diff_mask][:5]
            print(f"{'kk (first ' + str(min_size) + ')':<20} {'❌ MISMATCH':<15} "
                  f"{diff_count} mismatches, first 5: C++={list(cpp_vals)}, Py={list(py_vals)}")
        else:
            print(f"{'kk (first ' + str(min_size) + ')':<20} {'✅ MATCH':<15} {'All match'}")
        
        print("-" * 80)
        print()
        
        # Show extra edges in table format
        if cpp_num_edges_after < py_num_edges_after:
            extra_count = py_num_edges_after - cpp_num_edges_after
            print(f"Python has {extra_count} extra edges (not in C++):")
            print("-" * 80)
            print(f"{'Edge':<10} {'ii':<10} {'jj':<10} {'kk':<10}")
            print("-" * 80)
            for i in range(extra_count):
                idx = cpp_num_edges_after + i
                print(f"{i+1:<10} {py_ii[idx]:<10} {py_jj[idx]:<10} {py_kk[idx]:<10}")
        else:
            extra_count = cpp_num_edges_after - py_num_edges_after
            print(f"C++ has {extra_count} extra edges (not in Python):")
            print("-" * 80)
            print(f"{'Edge':<10} {'ii':<10} {'jj':<10} {'kk':<10}")
            print("-" * 80)
            for i in range(extra_count):
                idx = py_num_edges_after + i
                print(f"{i+1:<10} {cpp_ii[idx]:<10} {cpp_jj[idx]:<10} {cpp_kk[idx]:<10}")
        print("-" * 80)
    else:
        # Same size, compare directly
        print("All edges comparison (same count):")
        print("-" * 80)
        
        ii_match = np.array_equal(cpp_ii, py_ii)
        jj_match = np.array_equal(cpp_jj, py_jj)
        kk_match = np.array_equal(cpp_kk, py_kk)
        
        print(f"{'Component':<20} {'Status':<15} {'Details':<45}")
        print("-" * 80)
        
        if not ii_match:
            diff_mask = cpp_ii != py_ii
            diff_count = np.sum(diff_mask)
            cpp_vals = cpp_ii[diff_mask][:10]
            py_vals = py_ii[diff_mask][:10]
            print(f"{'ii':<20} {'❌ MISMATCH':<15} "
                  f"{diff_count} mismatches, first 10: C++={list(cpp_vals)}, Py={list(py_vals)}")
        else:
            print(f"{'ii':<20} {'✅ MATCH':<15} {'All match'}")
        
        if not jj_match:
            diff_mask = cpp_jj != py_jj
            diff_count = np.sum(diff_mask)
            cpp_vals = cpp_jj[diff_mask][:10]
            py_vals = py_jj[diff_mask][:10]
            print(f"{'jj':<20} {'❌ MISMATCH':<15} "
                  f"{diff_count} mismatches, first 10: C++={list(cpp_vals)}, Py={list(py_vals)}")
        else:
            print(f"{'jj':<20} {'✅ MATCH':<15} {'All match'}")
        
        if not kk_match:
            diff_mask = cpp_kk != py_kk
            diff_count = np.sum(diff_mask)
            cpp_vals = cpp_kk[diff_mask][:10]
            py_vals = py_kk[diff_mask][:10]
            print(f"{'kk':<20} {'❌ MISMATCH':<15} "
                  f"{diff_count} mismatches, first 10: C++={list(cpp_vals)}, Py={list(py_vals)}")
        else:
            print(f"{'kk':<20} {'✅ MATCH':<15} {'All match'}")
        
        print("-" * 80)
    print()
    
    # Compare index
    cpp_index = load_index(os.path.join(bin_dir, f"keyframe_index_after_{frame_suffix}.bin"), n_after, M)
    py_index = load_index(os.path.join(bin_dir, f"keyframe_index_after_py_{frame_suffix}.bin"), n_after, M)
    
    index_match = np.array_equal(cpp_index, py_index)
    print("Index Comparison:")
    print("-" * 80)
    print(f"{'Metric':<30} {'C++':<15} {'Python':<15} {'Status':<10}")
    print("-" * 80)
    print(f"{'index [n=' + str(n_after) + ', M=' + str(M) + ']':<30} {'':<15} {'':<15} "
          f"{'✅ MATCH' if index_match else '❌ MISMATCH':<10}")
    if not index_match:
        diff_mask = cpp_index != py_index
        diff_count = np.sum(diff_mask)
        print(f"  Details: {diff_count} mismatches out of {n_after * M} total")
    print("-" * 80)
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare C++ and Python DPVO keyframe() results')
    parser.add_argument('--frame', type=int, required=True, help='Frame number to compare')
    parser.add_argument('--bin-dir', type=str, default='bin_file', help='Binary file directory')
    parser.add_argument('--network', type=str, default=None, 
                       help='Path to Python DPVO network weights file (optional - dummy network will be created if not provided)')
    
    args = parser.parse_args()
    
    bin_dir = args.bin_dir
    frame_num = args.frame
    
    print("=" * 80)
    print("KEYFRAME COMPARISON")
    print("=" * 80)
    print(f"Frame: {frame_num}")
    print(f"Binary directory: {bin_dir}")
    print()
    
    # Load metadata
    print("Loading C++ keyframe metadata...")
    metadata = load_keyframe_metadata(bin_dir, frame_num)
    print(f"  n_before={metadata['n_before']}, m_before={metadata['m_before']}, "
          f"num_edges_before={metadata['num_edges_before']}")
    print()
    
    # Load C++ inputs
    print("Loading C++ keyframe inputs...")
    cpp_inputs = load_cpp_keyframe_inputs(bin_dir, frame_num, metadata)
    print(f"  Loaded poses, patches, intrinsics, timestamps, colors, index, edges")
    print()
    
    # Network path is optional - a dummy network will be created if not provided
    # Note: Python DPVO constructor requires a network path to set DIM, RES, P attributes,
    # but keyframe() itself doesn't use the network weights
    
    # Setup Python DPVO
    print("Setting up Python DPVO with C++ inputs...")
    dpvo = setup_python_dpvo(cpp_inputs, metadata, args.network)
    print(f"  Python DPVO initialized: n={dpvo.n}, m={dpvo.m}, num_edges={dpvo.pg.num_edges}")
    print()
    
    # Run Python keyframe
    print("Running Python keyframe()...")
    python_state = run_python_keyframe(dpvo)
    print(f"  Before: n={python_state['n_before']}, m={python_state['m_before']}, "
          f"num_edges={python_state['num_edges_before']}")
    print(f"  After: n={python_state['n_after']}, m={python_state['m_after']}, "
          f"num_edges={python_state['num_edges_after']}")
    print()
    
    # Save Python outputs
    print("Saving Python keyframe outputs...")
    save_python_keyframe_outputs(bin_dir, frame_num, dpvo, python_state)
    print("  Saved Python outputs to binary files")
    print()
    
    # Compare results
    compare_results(bin_dir, frame_num, metadata, python_state)
    
    print("=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

