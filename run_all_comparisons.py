#!/usr/bin/env python3
"""
Run all comparison scripts and summarize results in a table.

This script:
1. Runs all comparison scripts (correlation, reproject, BA, update model, patchify, ONNX)
2. Captures their output and exit codes
3. Parses match/mismatch status from output
4. Creates a summary table showing which components match/mismatch
"""

import subprocess
import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ComparisonResult:
    """Result of running a comparison script."""
    name: str
    script: str
    command: List[str]
    exit_code: int
    success: bool
    match_status: str  # "MATCH", "MISMATCH", "SKIPPED", "ERROR"
    details: str
    max_diff: Optional[float] = None
    mean_diff: Optional[float] = None
    num_mismatched: Optional[int] = None
    total_elements: Optional[int] = None
    output: str = ""


def run_command(cmd: List[str], cwd: Optional[str] = None, timeout: Optional[int] = 300) -> Tuple[int, str]:
    """Run a command and return exit code and output."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return -1, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return -1, f"Error running command: {e}"


def parse_correlation_output(output: str) -> ComparisonResult:
    """Parse correlation comparison output."""
    match_status = "ERROR"
    details = ""
    max_diff = None
    mean_diff = None
    
    # Look for summary section
    if "SUMMARY" in output:
        if "✅ All correlations match!" in output:
            match_status = "MATCH"
        elif "❌ MISMATCH" in output or "Mismatches detected" in output:
            match_status = "MISMATCH"
    
    # Extract max diff from comparison table
    max_diff_match = re.search(r'Max Diff.*?([\d.e+-]+)', output)
    if max_diff_match:
        try:
            max_diff = float(max_diff_match.group(1))
        except ValueError:
            pass
    
    # Extract mean diff
    mean_diff_match = re.search(r'Mean Diff.*?([\d.e+-]+)', output)
    if mean_diff_match:
        try:
            mean_diff = float(mean_diff_match.group(1))
        except ValueError:
            pass
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = "Required files not found"
        else:
            details = "Could not parse output"
    
    return ComparisonResult(
        name="Correlation",
        script="compare_correlation_outputs.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        max_diff=max_diff,
        mean_diff=mean_diff,
        output=output
    )


def parse_reproject_output(output: str, edge: int = None) -> ComparisonResult:
    """Parse reproject intermediate comparison output."""
    match_status = "ERROR"
    details = ""
    
    # Look for summary
    if "SUMMARY" in output:
        if "✅ ALL MATCH" in output:
            match_status = "MATCH"
        elif "❌ SOME MISMATCHES" in output:
            match_status = "MISMATCH"
    
    # Check individual component matches
    if match_status == "ERROR":
        matches = []
        mismatches = []
        if "Ti Match: ✅" in output:
            matches.append("Ti")
        elif "Ti Match: ❌" in output:
            mismatches.append("Ti")
        if "Tj Match: ✅" in output:
            matches.append("Tj")
        elif "Tj Match: ❌" in output:
            mismatches.append("Tj")
        if "Gij Match: ✅" in output:
            matches.append("Gij")
        elif "Gij Match: ❌" in output:
            mismatches.append("Gij")
        if "Ji Match: ✅" in output:
            matches.append("Ji")
        elif "Ji Match: ❌" in output:
            mismatches.append("Ji")
        if "Jj Match: ✅" in output:
            matches.append("Jj")
        elif "Jj Match: ❌" in output:
            mismatches.append("Jj")
        if "Jz Match: ✅" in output:
            matches.append("Jz")
        elif "Jz Match: ❌" in output:
            mismatches.append("Jz")
        if "Coords Match: ✅" in output:
            matches.append("Coords")
        elif "Coords Match: ❌" in output:
            mismatches.append("Coords")
        
        if mismatches:
            match_status = "MISMATCH"
            details = f"Mismatched: {', '.join(mismatches)}"
        elif matches:
            match_status = "MATCH"
            details = f"All matched: {', '.join(matches)}"
    
    # Add edge information to details
    if edge is not None:
        if details:
            details = f"Edge {edge}: {details}"
        else:
            details = f"Edge {edge}"
    else:
        # Try to extract edge from output if not provided
        edge_match = re.search(r'Edge:\s*(\d+)', output)
        if edge_match:
            edge_num = edge_match.group(1)
            if details:
                details = f"Edge {edge_num}: {details}"
            else:
                details = f"Edge {edge_num}"
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = f"Edge {edge if edge is not None else '?'}: Required files not found"
        else:
            details = f"Edge {edge if edge is not None else '?'}: Could not parse output"
    
    return ComparisonResult(
        name="Reproject Intermediate",
        script="compare_reproject_intermediate.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        output=output
    )


def parse_ba_output(output: str) -> ComparisonResult:
    """Parse BA step-by-step comparison output.
    
    Only checks if Final Poses (STEP 17) match. If final poses match, BA is considered MATCH.
    """
    match_status = "ERROR"
    details = ""
    
    # Look specifically for "Final Poses" comparison result
    # The BA script outputs "Final Poses (STEP 17)" in the summary table
    final_poses_found = False
    
    # Check for "Final Poses" in the output
    if "Final Poses" in output or "COMPARING FINAL OUTPUTS" in output:
        final_poses_found = True
        
        # Method 1: Look for "Final Poses (STEP 17)" in the summary table
        # The summary table format is: "Final Poses (STEP 17)" followed by status
        final_poses_pattern = r'Final Poses.*?STEP 17.*?(✅ MATCH|❌ MISMATCH|❌ DIFF)'
        match = re.search(final_poses_pattern, output, re.DOTALL)
        
        if match:
            status_text = match.group(1)
            if "✅ MATCH" in status_text:
                match_status = "MATCH"
                details = "Final poses matched"
            else:
                match_status = "MISMATCH"
                details = "Final poses mismatched"
        
        # Method 2: Look for overall pose match status in the final outputs section
        if match_status == "ERROR":
            # Check for "Overall pose match" line
            overall_match_pattern = r'Overall.*?pose.*?match.*?(✅|❌)'
            match = re.search(overall_match_pattern, output, re.IGNORECASE)
            if match:
                if "✅" in match.group(0):
                    match_status = "MATCH"
                    details = "Final poses matched (overall)"
                else:
                    match_status = "MISMATCH"
                    details = "Final poses mismatched (overall)"
        
        # Method 3: Check the summary table for STEP 17 row
        if match_status == "ERROR":
            # Look for STEP 17 row in the summary table
            # Format: "STEP 17" ... "✅ MATCH" or "❌ MISMATCH"
            step17_pattern = r'STEP 17[^\n]*?(✅ MATCH|❌ MISMATCH)'
            match = re.search(step17_pattern, output, re.DOTALL)
            if match:
                status_text = match.group(1)
                if "✅ MATCH" in status_text:
                    match_status = "MATCH"
                    details = "STEP 17 (final poses) matched"
                else:
                    match_status = "MISMATCH"
                    details = "STEP 17 (final poses) mismatched"
        
        # Method 4: Check if all poses match (from the detailed comparison)
        if match_status == "ERROR":
            # Look for "All poses matched" or similar text
            if re.search(r'all.*?poses.*?match', output, re.IGNORECASE):
                match_status = "MATCH"
                details = "All final poses matched"
            elif re.search(r'pose.*?mismatch', output, re.IGNORECASE):
                match_status = "MISMATCH"
                details = "Some final poses mismatched"
    
    # If we couldn't find final poses specifically, fall back to checking STEP 17
    if match_status == "ERROR" and "STEP 17" in output:
        # Look for STEP 17 status anywhere in output
        step17_pattern = r'STEP 17.*?(✅ MATCH|❌ MISMATCH)'
        match = re.search(step17_pattern, output, re.DOTALL)
        if match:
            status_text = match.group(1)
            if "✅ MATCH" in status_text:
                match_status = "MATCH"
                details = "STEP 17 (final poses) matched"
            else:
                match_status = "MISMATCH"
                details = "STEP 17 (final poses) mismatched"
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = "Required files not found"
        elif final_poses_found:
            # Found final poses section but couldn't determine status
            # Default to checking if there are any mismatches in the summary
            if "❌ MISMATCH" in output:
                match_status = "MISMATCH"
                details = "Some BA steps mismatched (final poses status unclear)"
            else:
                details = "Could not determine final poses match status"
        else:
            details = "Final poses comparison not found"
    
    return ComparisonResult(
        name="Bundle Adjustment",
        script="compare_ba_step_by_step.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        output=output
    )


def parse_update_output(output: str) -> ComparisonResult:
    """Parse update model comparison output."""
    match_status = "ERROR"
    details = ""
    
    # Look for comparison table
    if "UPDATE MODEL OUTPUT COMPARISON" in output:
        # Parse the table to find mismatches
        mismatches = []
        matches = []
        
        # Check each output type from the table
        lines = output.split('\n')
        for line in lines:
            if "net_out" in line.lower() or "netOut" in line.lower():
                if "❌ MISMATCH" in line:
                    mismatches.append("net_out")
                elif "✅ MATCH" in line:
                    matches.append("net_out")
            if "d_out" in line.lower() or "dOut" in line.lower():
                if "❌ MISMATCH" in line:
                    mismatches.append("d_out")
                elif "✅ MATCH" in line:
                    matches.append("d_out")
            if "w_out" in line.lower() or "wOut" in line.lower():
                if "❌ MISMATCH" in line:
                    mismatches.append("w_out")
                elif "✅ MATCH" in line:
                    matches.append("w_out")
        
        if mismatches:
            match_status = "MISMATCH"
            details = f"Mismatched: {', '.join(mismatches)}"
        elif matches:
            match_status = "MATCH"
            details = f"All matched: {', '.join(matches)}"
    else:
        # Fallback: look for match indicators
        if "✅ All outputs match" in output or "All matches: True" in output:
            match_status = "MATCH"
        elif "❌" in output and "MISMATCH" in output:
            match_status = "MISMATCH"
            details = "Some outputs mismatched"
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = "Required files not found"
        else:
            details = "Could not parse output"
    
    return ComparisonResult(
        name="Update Model",
        script="compare_update_onnx_dpvo.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        output=output
    )


def parse_patchify_output(output: str) -> ComparisonResult:
    """Parse patchify comparison output."""
    match_status = "ERROR"
    details = ""
    
    # Look for match indicators
    if "✅" in output and "MATCH" in output:
        match_status = "MATCH"
    elif "❌" in output and "MISMATCH" in output:
        match_status = "MISMATCH"
    
    # Check which components match
    components = []
    if "gmap" in output.lower():
        if "✅" in output and "gmap" in output.lower():
            components.append("gmap✅")
        elif "❌" in output and "gmap" in output.lower():
            components.append("gmap❌")
    if "imap" in output.lower():
        if "✅" in output and "imap" in output.lower():
            components.append("imap✅")
        elif "❌" in output and "imap" in output.lower():
            components.append("imap❌")
    if "patches" in output.lower():
        if "✅" in output and "patches" in output.lower():
            components.append("patches✅")
        elif "❌" in output and "patches" in output.lower():
            components.append("patches❌")
    
    if components:
        details = ", ".join(components)
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = "Required files not found"
        else:
            details = "Could not parse output"
    
    return ComparisonResult(
        name="Patchify",
        script="compare_patchify.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        output=output
    )


def parse_onnx_output(output: str) -> ComparisonResult:
    """Parse ONNX model comparison output."""
    match_status = "ERROR"
    details = ""
    
    # Look for match indicators
    if "✅ All outputs match" in output or "All matches: True" in output:
        match_status = "MATCH"
    elif "❌" in output and "Mismatch" in output:
        match_status = "MISMATCH"
    
    # Check FNet and INet separately
    fnet_match = "FNet" in output and ("✅" in output or "MATCH" in output)
    inet_match = "INet" in output and ("✅" in output or "MATCH" in output)
    
    if fnet_match and inet_match:
        match_status = "MATCH"
        details = "FNet✅, INet✅"
    elif not fnet_match or not inet_match:
        match_status = "MISMATCH"
        mismatches = []
        if not fnet_match:
            mismatches.append("FNet❌")
        if not inet_match:
            mismatches.append("INet❌")
        details = ", ".join(mismatches)
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = "Required files not found"
        else:
            details = "Could not parse output"
    
    return ComparisonResult(
        name="ONNX Models",
        script="compare_onnx_outputs.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        output=output
    )


def run_correlation_comparison(frame: int) -> ComparisonResult:
    """Run correlation comparison."""
    cmd = [sys.executable, "compare_correlation_outputs.py", "--frame", str(frame)]
    exit_code, output = run_command(cmd)
    result = parse_correlation_output(output)
    result.command = cmd
    result.exit_code = exit_code
    result.output = output
    return result


def run_reproject_comparison(frame: int, edge: int = 0) -> ComparisonResult:
    """Run reproject intermediate comparison."""
    cmd = [sys.executable, "compare_reproject_intermediate.py", 
           "--frame", str(frame), "--edge", str(edge)]
    exit_code, output = run_command(cmd)
    result = parse_reproject_output(output, edge=edge)
    result.command = cmd
    result.exit_code = exit_code
    result.output = output
    return result


def run_ba_comparison() -> ComparisonResult:
    """Run BA step-by-step comparison."""
    cmd = [sys.executable, "compare_ba_step_by_step.py"]
    exit_code, output = run_command(cmd)
    result = parse_ba_output(output)
    result.command = cmd
    result.exit_code = exit_code
    result.output = output
    return result


def run_update_comparison(model_path: str, frame: int) -> ComparisonResult:
    """Run update model comparison."""
    if not os.path.exists(model_path):
        return ComparisonResult(
            name="Update Model",
            script="compare_update_onnx_dpvo.py",
            command=[],
            exit_code=-1,
            success=False,
            match_status="SKIPPED",
            details=f"Model file not found: {model_path}",
            output=""
        )
    
    cmd = [sys.executable, "compare_update_onnx_dpvo.py", model_path, str(frame)]
    exit_code, output = run_command(cmd)
    result = parse_update_output(output)
    result.command = cmd
    result.exit_code = exit_code
    result.output = output
    return result


def run_patchify_comparison(frame: int) -> ComparisonResult:
    """Run patchify comparison."""
    cmd = [sys.executable, "compare_patchify.py", str(frame)]
    exit_code, output = run_command(cmd)
    result = parse_patchify_output(output)
    result.command = cmd
    result.exit_code = exit_code
    result.output = output
    return result


def run_onnx_comparison(image_path: str, fnet_model: str, inet_model: str, 
                        fnet_bin: Optional[str] = None, inet_bin: Optional[str] = None) -> ComparisonResult:
    """Run ONNX model comparison."""
    cmd = [sys.executable, "compare_onnx_outputs.py", image_path, fnet_model, inet_model]
    if fnet_bin:
        cmd.append(fnet_bin)
    if inet_bin:
        cmd.append(inet_bin)
    
    exit_code, output = run_command(cmd)
    result = parse_onnx_output(output)
    result.command = cmd
    result.exit_code = exit_code
    result.output = output
    return result


def print_summary_table(results: List[ComparisonResult]):
    """Print a summary table of all comparison results."""
    print("\n" + "="*100)
    print("COMPARISON SUMMARY TABLE")
    print("="*100)
    print(f"{'Component':<30} {'Status':<15} {'Exit Code':<12} {'Details':<43}")
    print("-"*100)
    
    for result in results:
        # Format status with emoji
        status_emoji = {
            "MATCH": "✅ MATCH",
            "MISMATCH": "❌ MISMATCH",
            "SKIPPED": "⚠️  SKIPPED",
            "ERROR": "❌ ERROR"
        }.get(result.match_status, result.match_status)
        
        # Format exit code
        exit_str = str(result.exit_code) if result.exit_code >= 0 else "N/A"
        
        # Truncate details if too long
        details = result.details[:40] + "..." if len(result.details) > 43 else result.details
        
        print(f"{result.name:<30} {status_emoji:<15} {exit_str:<12} {details:<43}")
    
    print("="*100)
    
    # Summary statistics
    total = len(results)
    matched = sum(1 for r in results if r.match_status == "MATCH")
    mismatched = sum(1 for r in results if r.match_status == "MISMATCH")
    skipped = sum(1 for r in results if r.match_status == "SKIPPED")
    errors = sum(1 for r in results if r.match_status == "ERROR")
    
    print(f"\nSummary Statistics:")
    print(f"  Total comparisons: {total}")
    print(f"  ✅ Matched: {matched}")
    print(f"  ❌ Mismatched: {mismatched}")
    print(f"  ⚠️  Skipped: {skipped}")
    print(f"  ❌ Errors: {errors}")
    print("="*100)


def print_detailed_results(results: List[ComparisonResult], show_output: bool = False):
    """Print detailed results for each comparison."""
    print("\n" + "="*100)
    print("DETAILED RESULTS")
    print("="*100)
    
    for result in results:
        print(f"\n{'='*100}")
        print(f"Component: {result.name}")
        print(f"Script: {result.script}")
        print(f"Command: {' '.join(result.command)}")
        print(f"Exit Code: {result.exit_code}")
        print(f"Status: {result.match_status}")
        print(f"Details: {result.details}")
        
        if result.max_diff is not None:
            print(f"Max Diff: {result.max_diff:.6e}")
        if result.mean_diff is not None:
            print(f"Mean Diff: {result.mean_diff:.6e}")
        if result.num_mismatched is not None and result.total_elements is not None:
            print(f"Mismatched: {result.num_mismatched}/{result.total_elements}")
        
        if show_output and result.output:
            print(f"\nOutput (last 500 chars):")
            print(result.output[-500:])
        
        print("="*100)


def main():
    """Main function to run all comparisons."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run all comparison scripts and summarize results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all comparisons for frame 40
  python run_all_comparisons.py --frame 40
  
  # Run with update model
  python run_all_comparisons.py --frame 40 --update-model update.onnx
  
  # Run with ONNX models
  python run_all_comparisons.py --frame 40 --image frame0.jpg --fnet-model fnet.onnx --inet-model inet.onnx
  
  # Skip specific comparisons
  python run_all_comparisons.py --frame 40 --skip-onnx --skip-patchify
        """
    )
    
    parser.add_argument("--frame", type=int, default=40, 
                       help="Frame number for comparisons (default: 40)")
    parser.add_argument("--edge", type=int, default=0,
                       help="Edge index for reproject comparison (default: 0)")
    parser.add_argument("--update-model", type=str, default=None,
                       help="Path to update model ONNX file (optional)")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to image file for ONNX comparison (optional)")
    parser.add_argument("--fnet-model", type=str, default=None,
                       help="Path to FNet ONNX model (optional)")
    parser.add_argument("--inet-model", type=str, default=None,
                       help="Path to INet ONNX model (optional)")
    parser.add_argument("--fnet-bin", type=str, default=None,
                       help="Path to C++ FNet output binary (optional)")
    parser.add_argument("--inet-bin", type=str, default=None,
                       help="Path to C++ INet output binary (optional)")
    parser.add_argument("--skip-correlation", action="store_true",
                       help="Skip correlation comparison")
    parser.add_argument("--skip-reproject", action="store_true",
                       help="Skip reproject intermediate comparison")
    parser.add_argument("--skip-ba", action="store_true",
                       help="Skip BA step-by-step comparison")
    parser.add_argument("--skip-update", action="store_true",
                       help="Skip update model comparison")
    parser.add_argument("--skip-patchify", action="store_true",
                       help="Skip patchify comparison")
    parser.add_argument("--skip-onnx", action="store_true",
                       help="Skip ONNX model comparison")
    parser.add_argument("--show-output", action="store_true",
                       help="Show full output from each comparison")
    parser.add_argument("--save-output", type=str, default=None,
                       help="Save detailed output to file")
    
    args = parser.parse_args()
    
    print("="*100)
    print("RUNNING ALL COMPARISONS")
    print("="*100)
    print(f"Frame: {args.frame}")
    print(f"Edge: {args.edge}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    
    results = []
    
    # 1. Correlation comparison
    if not args.skip_correlation:
        print("\n[1/6] Running correlation comparison...")
        result = run_correlation_comparison(args.frame)
        results.append(result)
        print(f"    Status: {result.match_status}")
    else:
        print("\n[1/6] Skipping correlation comparison")
    
    # 2. Reproject intermediate comparison
    if not args.skip_reproject:
        print("\n[2/6] Running reproject intermediate comparison...")
        result = run_reproject_comparison(args.frame, args.edge)
        results.append(result)
        print(f"    Status: {result.match_status}")
    else:
        print("\n[2/6] Skipping reproject intermediate comparison")
    
    # 3. BA step-by-step comparison
    if not args.skip_ba:
        print("\n[3/6] Running BA step-by-step comparison...")
        result = run_ba_comparison()
        results.append(result)
        print(f"    Status: {result.match_status}")
    else:
        print("\n[3/6] Skipping BA step-by-step comparison")
    
    # 4. Update model comparison
    if not args.skip_update:
        if args.update_model:
            print("\n[4/6] Running update model comparison...")
            result = run_update_comparison(args.update_model, args.frame)
            results.append(result)
            print(f"    Status: {result.match_status}")
        else:
            print("\n[4/6] Skipping update model comparison (--update-model not provided)")
    else:
        print("\n[4/6] Skipping update model comparison")
    
    # 5. Patchify comparison
    if not args.skip_patchify:
        print("\n[5/6] Running patchify comparison...")
        result = run_patchify_comparison(args.frame)
        results.append(result)
        print(f"    Status: {result.match_status}")
    else:
        print("\n[5/6] Skipping patchify comparison")
    
    # 6. ONNX model comparison
    if not args.skip_onnx:
        if args.image and args.fnet_model and args.inet_model:
            print("\n[6/6] Running ONNX model comparison...")
            result = run_onnx_comparison(
                args.image, args.fnet_model, args.inet_model,
                args.fnet_bin, args.inet_bin
            )
            results.append(result)
            print(f"    Status: {result.match_status}")
        else:
            print("\n[6/6] Skipping ONNX model comparison (--image, --fnet-model, --inet-model not all provided)")
    else:
        print("\n[6/6] Skipping ONNX model comparison")
    
    # Print summary table
    print_summary_table(results)
    
    # Print detailed results if requested
    if args.show_output:
        print_detailed_results(results, show_output=True)
    
    # Save output to file if requested
    if args.save_output:
        with open(args.save_output, 'w') as f:
            f.write("="*100 + "\n")
            f.write("COMPARISON RESULTS\n")
            f.write("="*100 + "\n")
            f.write(f"Frame: {args.frame}\n")
            f.write(f"Edge: {args.edge}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")
            
            for result in results:
                f.write(f"\n{'='*100}\n")
                f.write(f"Component: {result.name}\n")
                f.write(f"Script: {result.script}\n")
                f.write(f"Command: {' '.join(result.command)}\n")
                f.write(f"Exit Code: {result.exit_code}\n")
                f.write(f"Status: {result.match_status}\n")
                f.write(f"Details: {result.details}\n")
                f.write(f"\nFull Output:\n{result.output}\n")
                f.write("="*100 + "\n")
        
        print(f"\n✅ Detailed output saved to: {args.save_output}")
    
    # Return exit code based on results
    if any(r.match_status == "ERROR" for r in results):
        return 1
    elif any(r.match_status == "MISMATCH" for r in results):
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())

