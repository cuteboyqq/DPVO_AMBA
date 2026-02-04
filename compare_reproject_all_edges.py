#!/usr/bin/env python3
"""
Compare C++ reproject intermediate values (Gij, Ji, Jj, Jz) with Python reproject for ALL edges.

This script:
1. Loads C++ input data (poses, patches, intrinsics, edge indices)
2. Iterates through all edges
3. For each edge, compares C++ and Python intermediate values
4. Summarizes results in tables showing match/mismatch status for each edge
5. Shows statistics on how many edges matched
"""

import numpy as np
import torch
import sys
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
from io import StringIO

# Import functions from compare_reproject_intermediate.py
from compare_reproject_intermediate import (
    load_binary_float, load_binary_int32, load_se3_object, load_matrix,
    load_poses, load_patches, load_intrinsics, load_edge_indices,
    convert_patches_to_torch, convert_intrinsics_to_torch,
    compare_se3, compare_jacobians, compare_coordinates,
    prepare_python_inputs, call_python_reproject, load_cpp_input_data
)

# Add DPVO Python path
dpvo_path = Path("/home/ali/Projects/GitHub_Code/clean_code/DPVO_onnx")
if not dpvo_path.exists():
    raise FileNotFoundError(f"DPVO_onnx directory not found: {dpvo_path}")

if str(dpvo_path) not in sys.path:
    sys.path.insert(0, str(dpvo_path))

try:
    from dpvo.lietorch import SE3 as SE3_Python
except ImportError:
    try:
        from lietorch import SE3 as SE3_Python
    except ImportError:
        SE3_Python = None
        print("Warning: Could not import lietorch.SE3. Will use torch tensors directly.")


@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@dataclass
class EdgeComparisonResult:
    """Result of comparing one edge."""
    edge_idx: int
    i: int  # Source frame index
    j: int  # Target frame index
    k: int  # Patch index
    ti_match: bool
    tj_match: bool
    gij_match: bool
    ji_match: bool
    jj_match: bool
    jz_match: bool
    coords_match: Optional[bool]
    all_match: bool
    # Max/mean diff values for each component
    ti_max_diff: Optional[float] = None
    ti_mean_diff: Optional[float] = None
    tj_max_diff: Optional[float] = None
    tj_mean_diff: Optional[float] = None
    gij_max_diff: Optional[float] = None  # Max diff from Gij comparison (R_max_diff)
    gij_mean_diff: Optional[float] = None  # Mean diff from Gij comparison (R_mean_diff)
    ji_max_diff: Optional[float] = None
    ji_mean_diff: Optional[float] = None
    jj_max_diff: Optional[float] = None
    jj_mean_diff: Optional[float] = None
    jz_max_diff: Optional[float] = None
    jz_mean_diff: Optional[float] = None
    coords_max_diff: Optional[float] = None
    coords_mean_diff: Optional[float] = None
    coords_error: Optional[str] = None  # Reason why coords weren't compared
    error: Optional[str] = None


def compare_single_edge(
    edge_idx: int,
    bin_dir: str,
    frame_num: int,
    poses_cpp: np.ndarray,
    patches_cpp: np.ndarray,
    intrinsics_cpp: np.ndarray,
    ii_cpp: np.ndarray,
    jj_cpp_idx: np.ndarray,
    kk_cpp: np.ndarray,
    coords_cpp_full: Optional[np.ndarray],
    N: int,
    M: int,
    P: int,
    tolerance: float,
    verbose: bool = False
) -> EdgeComparisonResult:
    """Compare a single edge between C++ and Python.
    
    Args:
        edge_idx: Edge index to compare
        bin_dir: Directory containing binary files
        frame_num: Frame number
        poses_cpp, patches_cpp, intrinsics_cpp: C++ input data
        ii_cpp, jj_cpp_idx, kk_cpp: Edge indices
        coords_cpp_full: C++ coordinates (may be None)
        N, M, P: Dimensions
        tolerance: Comparison tolerance
        verbose: If True, print detailed output
    
    Returns:
        EdgeComparisonResult: Comparison result for this edge
    """
    try:
        # Get edge indices
        i = kk_cpp[edge_idx] // M  # Source frame
        j = jj_cpp_idx[edge_idx]   # Target frame
        k = kk_cpp[edge_idx]        # Patch index
        
        # Load C++ intermediate values for this edge
        frame_suffix = str(frame_num)
        edge_suffix = str(edge_idx)
        
        ti_file = os.path.join(bin_dir, f"reproject_Ti_frame{frame_suffix}_edge{edge_suffix}.bin")
        tj_file = os.path.join(bin_dir, f"reproject_Tj_frame{frame_suffix}_edge{edge_suffix}.bin")
        gij_file = os.path.join(bin_dir, f"reproject_Gij_frame{frame_suffix}_edge{edge_suffix}.bin")
        ji_file = os.path.join(bin_dir, f"reproject_Ji_frame{frame_suffix}_edge{edge_suffix}.bin")
        jj_file = os.path.join(bin_dir, f"reproject_Jj_frame{frame_suffix}_edge{edge_suffix}.bin")
        jz_file = os.path.join(bin_dir, f"reproject_Jz_frame{frame_suffix}_edge{edge_suffix}.bin")
        
        # Check if all required files exist
        required_files = [ti_file, tj_file, gij_file, ji_file, jj_file, jz_file]
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            return EdgeComparisonResult(
                edge_idx=edge_idx,
                i=i,
                j=j,
                k=k,
                ti_match=False,
                tj_match=False,
                gij_match=False,
                ji_match=False,
                jj_match=False,
                jz_match=False,
                coords_match=None,
                coords_error=None,
                all_match=False,
                ti_max_diff=None,
                ti_mean_diff=None,
                tj_max_diff=None,
                tj_mean_diff=None,
                gij_max_diff=None,
                gij_mean_diff=None,
                ji_max_diff=None,
                ji_mean_diff=None,
                jj_max_diff=None,
                jj_mean_diff=None,
                jz_max_diff=None,
                jz_mean_diff=None,
                coords_max_diff=None,
                coords_mean_diff=None,
                error=f"Missing files: {missing_files}"
            )
        
        # Load C++ intermediate values
        ti_cpp = load_se3_object(ti_file)
        tj_cpp = load_se3_object(tj_file)
        gij_cpp = load_se3_object(gij_file)
        ji_cpp = load_matrix(ji_file, 2, 6)
        jj_cpp = load_matrix(jj_file, 2, 6)
        jz_cpp = load_matrix(jz_file, 2, 1)
        
        # Prepare Python inputs for this edge (suppress verbose output unless verbose=True)
        if verbose:
            poses_batch, patches_batch, intrinsics_batch, ii_single, jj_single, kk_single = prepare_python_inputs(
                poses_cpp, patches_cpp, intrinsics_cpp, i, j, k, N, M, P
            )
        else:
            with suppress_stdout():
                poses_batch, patches_batch, intrinsics_batch, ii_single, jj_single, kk_single = prepare_python_inputs(
                    poses_cpp, patches_cpp, intrinsics_cpp, i, j, k, N, M, P
                )
        
        # Call Python reproject (suppress verbose output unless verbose=True)
        coords_py_edge = None  # Initialize to None in case of failure
        try:
            if verbose:
                gij_py, ti_py, tj_py, ji_py_center, jj_py_center, jz_py_center, coords_py_edge = call_python_reproject(
                    poses_batch, patches_batch, intrinsics_batch,
                    ii_single, jj_single, kk_single, P, i, j, poses_cpp
                )
            else:
                with suppress_stdout():
                    gij_py, ti_py, tj_py, ji_py_center, jj_py_center, jz_py_center, coords_py_edge = call_python_reproject(
                        poses_batch, patches_batch, intrinsics_batch,
                        ii_single, jj_single, kk_single, P, i, j, poses_cpp
                    )
        except Exception as e:
            # If call_python_reproject fails, set coords_py_edge to None
            coords_py_edge = None
            if verbose:
                print(f"  ⚠️  Error calling Python reproject for edge {edge_idx}: {e}")
        
        # Compare SE3 objects (without verbose output)
        ti_result = compare_se3(ti_cpp, ti_py, "Ti", tolerance)
        tj_result = compare_se3(tj_cpp, tj_py, "Tj", tolerance)
        gij_result = compare_se3(gij_cpp, gij_py, "Gij", tolerance)
        
        # Compare Jacobians
        ji_result = compare_jacobians(ji_cpp, ji_py_center, "Ji", tolerance)
        jj_result = compare_jacobians(jj_cpp, jj_py_center, "Jj", tolerance)
        jz_result = compare_jacobians(jz_cpp, jz_py_center, "Jz", tolerance)
        
        # Compare coordinates if available
        coords_match = None
        coords_error = None
        coords_max_diff = None
        coords_mean_diff = None
        
        if coords_cpp_full is None:
            coords_error = "C++ coords file missing"
        elif coords_py_edge is None:
            coords_error = "Python coords not returned"
        else:
            try:
                edge_base = edge_idx * 2 * P * P
                if edge_base + 2 * P * P <= len(coords_cpp_full):
                    coords_cpp_edge_flat = coords_cpp_full[edge_base:edge_base + 2 * P * P]
                    coords_cpp_edge = coords_cpp_edge_flat.reshape(2, P, P)
                    coords_result = compare_coordinates(coords_cpp_edge, coords_py_edge, "Coords", tolerance)
                    coords_match = bool(coords_result['match'])  # Convert numpy bool to Python bool
                    coords_max_diff = coords_result.get('max_diff')
                    coords_mean_diff = coords_result.get('mean_diff')
                else:
                    coords_error = f"Edge {edge_idx} out of bounds (file size: {len(coords_cpp_full)}, expected: {edge_base + 2 * P * P})"
            except Exception as e:
                coords_error = f"Exception: {str(e)}"
                if verbose:
                    print(f"  ⚠️  Error comparing coordinates for edge {edge_idx}: {e}")
        
        # Compute overall match
        all_match = (
            ti_result['overall_match'] and
            tj_result['overall_match'] and
            gij_result['overall_match'] and
            ji_result['match'] and
            jj_result['match'] and
            jz_result['match']
        )
        
        if coords_match is not None:
            all_match = all_match and coords_match
        
        return EdgeComparisonResult(
            edge_idx=edge_idx,
            i=i,
            j=j,
            k=k,
            ti_match=ti_result['overall_match'],
            tj_match=tj_result['overall_match'],
            gij_match=gij_result['overall_match'],
            ji_match=ji_result['match'],
            jj_match=jj_result['match'],
            jz_match=jz_result['match'],
            coords_match=coords_match,
            coords_error=coords_error,
            all_match=all_match,
            ti_max_diff=ti_result.get('R_max_diff'),  # Use rotation matrix max diff
            ti_mean_diff=ti_result.get('R_mean_diff'),  # Use rotation matrix mean diff
            tj_max_diff=tj_result.get('R_max_diff'),
            tj_mean_diff=tj_result.get('R_mean_diff'),
            gij_max_diff=gij_result.get('R_max_diff'),  # Use rotation matrix max diff as overall Gij diff
            gij_mean_diff=gij_result.get('R_mean_diff'),  # Use rotation matrix mean diff as overall Gij diff
            ji_max_diff=ji_result.get('max_diff'),
            ji_mean_diff=ji_result.get('mean_diff'),
            jj_max_diff=jj_result.get('max_diff'),
            jj_mean_diff=jj_result.get('mean_diff'),
            jz_max_diff=jz_result.get('max_diff'),
            jz_mean_diff=jz_result.get('mean_diff'),
            coords_max_diff=coords_max_diff,
            coords_mean_diff=coords_mean_diff,
            error=None
        )
        
    except Exception as e:
        return EdgeComparisonResult(
            edge_idx=edge_idx,
            i=-1,
            j=-1,
            k=-1,
            ti_match=False,
            tj_match=False,
            gij_match=False,
            ji_match=False,
            jj_match=False,
            jz_match=False,
            coords_match=None,
            coords_error=None,
            all_match=False,
            ti_max_diff=None,
            ti_mean_diff=None,
            tj_max_diff=None,
            tj_mean_diff=None,
            gij_max_diff=None,
            gij_mean_diff=None,
            ji_max_diff=None,
            ji_mean_diff=None,
            jj_max_diff=None,
            jj_mean_diff=None,
            jz_max_diff=None,
            jz_mean_diff=None,
            coords_max_diff=None,
            coords_mean_diff=None,
            error=str(e)
        )


def format_diff_value(max_diff: Optional[float], mean_diff: Optional[float] = None) -> str:
    """Format max diff value for display."""
    if max_diff is None:
        return "N/A"
    # Always use fixed-point notation with enough decimal places to show small values
    # Use 8 decimal places to show values like 1.53e-05 as 0.00001530
    return f"{max_diff:.8f}"


def print_edge_results_table(results: List[EdgeComparisonResult]) -> None:
    """Print tables showing comparison results for all edges."""
    # Table 1: Match Status
    print(f"\n{'='*120}")
    print("EDGE COMPARISON RESULTS - MATCH STATUS")
    print(f"{'='*120}")
    
    # Table header
    header = f"{'Edge':<8} {'i':<6} {'j':<6} {'k':<8} {'Ti':<6} {'Tj':<6} {'Gij':<6} {'Ji':<6} {'Jj':<6} {'Jz':<6} {'Coords':<8} {'Overall':<10}"
    print(header)
    print("-" * 120)
    
    # Table rows
    for result in results:
        edge_str = str(result.edge_idx)
        i_str = str(result.i) if result.i >= 0 else "N/A"
        j_str = str(result.j) if result.j >= 0 else "N/A"
        k_str = str(result.k) if result.k >= 0 else "N/A"
        
        ti_str = "✅" if result.ti_match else "❌"
        tj_str = "✅" if result.tj_match else "❌"
        gij_str = "✅" if result.gij_match else "❌"
        ji_str = "✅" if result.ji_match else "❌"
        jj_str = "✅" if result.jj_match else "❌"
        jz_str = "✅" if result.jz_match else "❌"
        
        if result.coords_match is not None:
            coords_str = "✅" if result.coords_match else "❌"
        else:
            coords_str = "N/A"
        
        overall_str = "✅ MATCH" if result.all_match else "❌ MISMATCH"
        
        row = f"{edge_str:<8} {i_str:<6} {j_str:<6} {k_str:<8} {ti_str:<6} {tj_str:<6} {gij_str:<6} {ji_str:<6} {jj_str:<6} {jz_str:<6} {coords_str:<8} {overall_str:<10}"
        print(row)
    
    print("=" * 120)
    
    # Table 2: Difference Values
    print(f"\n{'='*150}")
    print("EDGE COMPARISON RESULTS - MAX DIFFERENCE VALUES")
    print(f"{'='*150}")
    
    # Table header for diff values
    diff_header = (f"{'Edge':<8} {'Ti Max Diff':<15} {'Tj Max Diff':<15} {'Gij Max Diff':<15} "
                   f"{'Ji Max Diff':<15} {'Jj Max Diff':<15} {'Jz Max Diff':<15} {'Coords Max Diff':<15}")
    print(diff_header)
    print("-" * 150)
    
    # Table rows for diff values
    for result in results:
        edge_str = str(result.edge_idx)
        
        ti_diff_str = format_diff_value(result.ti_max_diff)
        tj_diff_str = format_diff_value(result.tj_max_diff)
        gij_diff_str = format_diff_value(result.gij_max_diff)
        ji_diff_str = format_diff_value(result.ji_max_diff)
        jj_diff_str = format_diff_value(result.jj_max_diff)
        jz_diff_str = format_diff_value(result.jz_max_diff)
        coords_diff_str = format_diff_value(result.coords_max_diff)
        
        diff_row = (f"{edge_str:<8} {ti_diff_str:<15} {tj_diff_str:<15} {gij_diff_str:<15} "
                    f"{ji_diff_str:<15} {jj_diff_str:<15} {jz_diff_str:<15} {coords_diff_str:<15}")
        print(diff_row)
    
    print("=" * 150)


def print_statistics_table(results: List[EdgeComparisonResult], bin_dir: str = None, frame_num: int = None) -> None:
    """Print statistics table showing how many edges matched."""
    print(f"\n{'='*120}")
    print("COMPARISON STATISTICS")
    print(f"{'='*120}")
    
    total = len(results)
    
    # Count matches for each component
    ti_matched = sum(1 for r in results if r.ti_match)
    tj_matched = sum(1 for r in results if r.tj_match)
    gij_matched = sum(1 for r in results if r.gij_match)
    ji_matched = sum(1 for r in results if r.ji_match)
    jj_matched = sum(1 for r in results if r.jj_match)
    jz_matched = sum(1 for r in results if r.jz_match)
    coords_matched = sum(1 for r in results if r.coords_match is True)
    coords_compared = sum(1 for r in results if r.coords_match is not None)
    coords_skipped = sum(1 for r in results if r.coords_match is None)
    coords_errors = sum(1 for r in results if r.coords_error is not None)
    all_matched = sum(1 for r in results if r.all_match)
    errors = sum(1 for r in results if r.error is not None)
    
    # Statistics table
    print(f"\n{'Component':<20} {'Matched':<15} {'Total':<15} {'Percentage':<15}")
    print("-" * 120)
    
    print(f"{'Ti':<20} {ti_matched:<15} {total:<15} {ti_matched/total*100:.2f}%")
    print(f"{'Tj':<20} {tj_matched:<15} {total:<15} {tj_matched/total*100:.2f}%")
    print(f"{'Gij':<20} {gij_matched:<15} {total:<15} {gij_matched/total*100:.2f}%")
    print(f"{'Ji':<20} {ji_matched:<15} {total:<15} {ji_matched/total*100:.2f}%")
    print(f"{'Jj':<20} {jj_matched:<15} {total:<15} {jj_matched/total*100:.2f}%")
    print(f"{'Jz':<20} {jz_matched:<15} {total:<15} {jz_matched/total*100:.2f}%")
    if coords_compared > 0:
        print(f"{'Coords':<20} {coords_matched:<15} {coords_compared:<15} {coords_matched/coords_compared*100:.2f}%")
    else:
        print(f"{'Coords':<20} {'N/A':<15} {'0':<15} {'N/A':<15}")
    
    if coords_skipped > 0:
        print(f"{'Coords (Skipped)':<20} {coords_skipped:<15} {total:<15} {coords_skipped/total*100:.2f}%")
        # Show why coordinates were skipped (check first few errors)
        if coords_errors > 0:
            sample_errors = [r.coords_error for r in results if r.coords_error is not None][:3]
            unique_errors = list(set(sample_errors))
            print(f"\n  Reasons for skipping coordinates:")
            for err in unique_errors:
                count = sum(1 for r in results if r.coords_error == err)
                print(f"    - {err}: {count} edges")
        else:
            # If no explicit errors but still skipped, check why
            # This happens when coords_cpp_full is None or coords_py_edge is None but no error was set
            # Check a sample of results to see what's happening
            sample_results = results[:min(5, len(results))]
            reasons = []
            for r in sample_results:
                if r.coords_error:
                    reasons.append(r.coords_error)
            if reasons:
                unique_reasons = list(set(reasons))
                print(f"\n  Reasons for skipping coordinates (sample):")
                for reason in unique_reasons:
                    print(f"    - {reason}")
            else:
                print(f"\n  ⚠️  Warning: Coordinates were skipped but no error reason was recorded.")
                print(f"     This may indicate that coords_py_edge was None but error wasn't set.")
                if results:
                    print(f"     First edge coords_error: {results[0].coords_error}")
                    print(f"     First edge coords_match: {results[0].coords_match}")
                # Check if coords_cpp_full was loaded
                if bin_dir and frame_num is not None:
                    coords_file = os.path.join(bin_dir, f"reproject_coords_frame{frame_num}.bin")
                    if os.path.exists(coords_file):
                        print(f"     C++ coords file exists: {coords_file}")
                    else:
                        print(f"     C++ coords file missing: {coords_file}")
    print(f"{'Overall (All)':<20} {all_matched:<15} {total:<15} {all_matched/total*100:.2f}%")
    print(f"{'Errors':<20} {errors:<15} {total:<15} {errors/total*100:.2f}%")
    
    print("=" * 120)
    
    # Summary
    print(f"\nSummary:")
    print(f"  Total edges compared: {total}")
    print(f"  ✅ Fully matched edges: {all_matched} ({all_matched/total*100:.2f}%)")
    print(f"  ❌ Mismatched edges: {total - all_matched} ({(total-all_matched)/total*100:.2f}%)")
    print(f"  ⚠️  Errors: {errors} ({errors/total*100:.2f}%)")
    
    # Calculate overall max diff and mean diff from Gij comparisons
    gij_max_diffs = [r.gij_max_diff for r in results if r.gij_max_diff is not None]
    gij_mean_diffs = [r.gij_mean_diff for r in results if r.gij_mean_diff is not None]
    
    overall_max_diff = max(gij_max_diffs) if gij_max_diffs else None
    overall_mean_diff = sum(gij_mean_diffs) / len(gij_mean_diffs) if gij_mean_diffs else None
    
    # Output parseable format for run_all_comparisons.py
    if overall_max_diff is not None:
        print(f"\nREPROJECT_MAX_DIFF={overall_max_diff:.10e}")
    if overall_mean_diff is not None:
        print(f"REPROJECT_MEAN_DIFF={overall_mean_diff:.10e}")
    
    if all_matched == total:
        print(f"\n🎉 SUCCESS: All edges matched!")
    elif all_matched > 0:
        print(f"\n⚠️  WARNING: {total - all_matched} edges have mismatches")
    else:
        print(f"\n❌ ERROR: No edges matched!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare C++ and Python reproject intermediate values for ALL edges"
    )
    parser.add_argument("--frame", type=int, required=True, help="Frame number to compare")
    parser.add_argument("--bin-dir", type=str, default="bin_file", help="Directory containing binary files")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Comparison tolerance (default: 1e-4)")
    parser.add_argument("--max-edges", type=int, default=None, help="Maximum number of edges to compare (default: all)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output for each edge")
    args = parser.parse_args()
    
    frame_num = args.frame
    bin_dir = args.bin_dir
    tolerance = args.tolerance
    max_edges = args.max_edges
    verbose = args.verbose
    
    print(f"{'='*120}")
    print(f"REPROJECT INTERMEDIATE VALUES COMPARISON - ALL EDGES")
    print(f"{'='*120}")
    print(f"Frame: {frame_num}")
    print(f"Binary directory: {bin_dir}")
    print(f"Tolerance: {tolerance}")
    if max_edges:
        print(f"Max edges: {max_edges}")
    print(f"{'='*120}")
    
    # Load C++ input data
    # We need to load the input files to get dimensions, but we don't need a specific edge yet
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
    
    # Infer dimensions from file sizes
    poses_data = load_binary_float(poses_file)
    N = len(poses_data) // 7
    
    # Try to infer M and P from patches file
    patches_data = load_binary_float(patches_file)
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
    
    print(f"\nLoaded input data:")
    print(f"  N (frames): {N}")
    print(f"  M (patches per frame): {M}")
    print(f"  P (patch size): {P}")
    print(f"  num_active (total edges): {num_active}")
    print(f"\nNote: Verbose output from individual edge comparisons is suppressed.")
    print(f"      Use --verbose flag to see detailed output for each edge.")
    
    # Determine how many edges to compare
    num_edges_to_compare = min(num_active, max_edges) if max_edges else num_active
    print(f"\nComparing {num_edges_to_compare} edges...")
    
    # Compare all edges
    results = []
    for edge_idx in range(num_edges_to_compare):
        if verbose:
            print(f"\n[{edge_idx+1}/{num_edges_to_compare}] Comparing edge {edge_idx}...")
        else:
            if (edge_idx + 1) % 10 == 0:
                print(f"  Progress: {edge_idx+1}/{num_edges_to_compare} edges compared...", end='\r')
        
        result = compare_single_edge(
            edge_idx=edge_idx,
            bin_dir=bin_dir,
            frame_num=frame_num,
            poses_cpp=poses_cpp,
            patches_cpp=patches_cpp,
            intrinsics_cpp=intrinsics_cpp,
            ii_cpp=ii_cpp,
            jj_cpp_idx=jj_cpp_idx,
            kk_cpp=kk_cpp,
            coords_cpp_full=coords_cpp_full,
            N=N,
            M=M,
            P=P,
            tolerance=tolerance,
            verbose=verbose
        )
        
        results.append(result)
    
    if not verbose:
        print()  # New line after progress indicator
    
    # Print results tables
    print_edge_results_table(results)
    print_statistics_table(results, bin_dir=bin_dir, frame_num=frame_num)


if __name__ == "__main__":
    main()

