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
    
    # Extract max diff and mean diff from comparison table
    # Look for the detailed comparison section
    # Pattern: "Max Diff" followed by a number
    max_diff_patterns = [
        r'Max Diff.*?([\d.e+-]+)',
        r'max_diff.*?([\d.e+-]+)',
        r'Max.*?Diff.*?([\d.e+-]+)'
    ]
    
    for pattern in max_diff_patterns:
        max_diff_match = re.search(pattern, output, re.IGNORECASE)
        if max_diff_match:
            try:
                max_diff = float(max_diff_match.group(1))
                break
            except ValueError:
                continue
    
    # Extract mean diff
    mean_diff_patterns = [
        r'Mean Diff.*?([\d.e+-]+)',
        r'mean_diff.*?([\d.e+-]+)',
        r'Mean.*?Diff.*?([\d.e+-]+)'
    ]
    
    for pattern in mean_diff_patterns:
        mean_diff_match = re.search(pattern, output, re.IGNORECASE)
        if mean_diff_match:
            try:
                mean_diff = float(mean_diff_match.group(1))
                break
            except ValueError:
                continue
    
    # If not found in summary, look in detailed analysis
    if max_diff is None or mean_diff is None:
        # Look for "DETAILED COMPARISON ANALYSIS" section
        detailed_section = re.search(r'DETAILED COMPARISON ANALYSIS.*?Max Diff.*?([\d.e+-]+).*?Mean Diff.*?([\d.e+-]+)', 
                                     output, re.IGNORECASE | re.DOTALL)
        if detailed_section:
            try:
                if max_diff is None:
                    max_diff = float(detailed_section.group(1))
                if mean_diff is None:
                    mean_diff = float(detailed_section.group(2))
            except (ValueError, IndexError):
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
    """Parse reproject all edges comparison output."""
    match_status = "ERROR"
    details = ""
    max_diff = None
    mean_diff = None
    
    # Look for statistics table section
    # The output format from compare_reproject_all_edges.py includes:
    # - "COMPARISON STATISTICS" section with a table
    # - Summary section with "Total edges compared", "Fully matched edges", etc.
    
    # Extract edge match statistics from summary
    # Pattern: "Total edges compared: 50"
    total_edges_match = re.search(r'Total edges compared:\s*(\d+)', output)
    matched_edges_match = re.search(r'✅ Fully matched edges:\s*(\d+)', output)
    mismatched_edges_match = re.search(r'❌ Mismatched edges:\s*(\d+)', output)
    
    total_edges = None
    matched_edges = None
    mismatched_edges = None
    
    if total_edges_match:
        try:
            total_edges = int(total_edges_match.group(1))
        except ValueError:
            pass
    
    if matched_edges_match:
        try:
            matched_edges = int(matched_edges_match.group(1))
        except ValueError:
            pass
    
    if mismatched_edges_match:
        try:
            mismatched_edges = int(mismatched_edges_match.group(1))
        except ValueError:
            pass
    
    # Determine match status based on edge statistics
    if total_edges is not None and matched_edges is not None:
        if matched_edges == total_edges:
            match_status = "MATCH"
            details = f"{matched_edges}/{total_edges} edges matched"
        elif matched_edges > 0:
            match_status = "MISMATCH"
            details = f"{matched_edges}/{total_edges} edges matched"
        else:
            match_status = "MISMATCH"
            details = f"0/{total_edges} edges matched"
    
    # Extract max diff and mean diff from parseable format lines
    # Format: "REPROJECT_MAX_DIFF=1.234567e-03"
    #         "REPROJECT_MEAN_DIFF=5.678901e-04"
    reproject_max_diff_match = re.search(r'REPROJECT_MAX_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
    reproject_mean_diff_match = re.search(r'REPROJECT_MEAN_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
    
    if reproject_max_diff_match:
        try:
            max_diff = float(reproject_max_diff_match.group(1))
        except ValueError:
            pass
    
    if reproject_mean_diff_match:
        try:
            mean_diff = float(reproject_mean_diff_match.group(1))
        except ValueError:
            pass
    
    # Fallback: Check for overall success message
    if match_status == "ERROR":
        if "🎉 SUCCESS: All edges matched!" in output:
            match_status = "MATCH"
            if total_edges:
                details = f"{total_edges}/{total_edges} edges matched"
            else:
                details = "All edges matched"
        elif "⚠️  WARNING:" in output and "edges have mismatches" in output:
            match_status = "MISMATCH"
            # Extract number from warning message
            warning_match = re.search(r'(\d+)\s+edges have mismatches', output)
            if warning_match:
                mismatched_count = int(warning_match.group(1))
                if total_edges:
                    matched_count = total_edges - mismatched_count
                    details = f"{matched_count}/{total_edges} edges matched"
                else:
                    details = f"{mismatched_count} edges mismatched"
        elif "❌ ERROR: No edges matched!" in output:
            match_status = "MISMATCH"
            if total_edges:
                details = f"0/{total_edges} edges matched"
            else:
                details = "No edges matched"
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = "Required files not found"
        else:
            details = "Could not parse output"
    
    return ComparisonResult(
        name="Reproject Intermediate",
        script="compare_reproject_all_edges.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        max_diff=max_diff,
        mean_diff=mean_diff,
        output=output
    )


def parse_ba_output(output: str) -> ComparisonResult:
    """Parse BA step-by-step comparison output.
    
    Only checks if Final Poses (STEP 17) match. If final poses match, BA is considered MATCH.
    """
    match_status = "ERROR"
    details = ""
    max_diff = None
    mean_diff = None
    
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
    
    # Extract max diff and mean diff from final poses comparison
    # Method 1: Extract from parseable format lines added by BA script
    # Format: "BA_FINAL_POSES_MAX_DIFF=1.234567e-03"
    #         "BA_FINAL_POSES_MEAN_DIFF=5.678901e-04"
    if "Final Poses" in output or "COMPARING FINAL OUTPUTS" in output:
        ba_max_diff_match = re.search(r'BA_FINAL_POSES_MAX_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
        ba_mean_diff_match = re.search(r'BA_FINAL_POSES_MEAN_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
        
        if ba_max_diff_match:
            try:
                max_diff = float(ba_max_diff_match.group(1))
            except ValueError:
                pass
        
        if ba_mean_diff_match:
            try:
                mean_diff = float(ba_mean_diff_match.group(1))
            except ValueError:
                pass
        
        # Method 2: Extract from summary table row "Final Poses (STEP 17)"
        # Format: "STEP 17    Final Poses (STEP 17)    ✅ MATCH    0.001234    0.000567"
        if max_diff is None or mean_diff is None:
            # Look for the summary table row with "Final Poses (STEP 17)"
            # The table format has: Step, Component Name, Status, Max Diff, Mean Diff
            # Pattern: "Final Poses (STEP 17)" followed by status, then max_diff, then mean_diff
            summary_row_pattern = r'Final Poses\s*\(STEP\s*17\)[^\n]*?(?:✅|❌)[^\n]*?([\d.e+-]+)\s+([\d.e+-]+)'
            summary_match = re.search(summary_row_pattern, output, re.IGNORECASE | re.DOTALL)
            
            if summary_match:
                try:
                    if max_diff is None:
                        max_diff = float(summary_match.group(1))
                    if mean_diff is None:
                        mean_diff = float(summary_match.group(2))
                except (ValueError, IndexError):
                    pass
        
        # Method 3: Extract from "Overall translation diff: max=..., mean=..." line
        # Format: "Overall translation diff: max=0.001234, mean=0.000567"
        if max_diff is None or mean_diff is None:
            patterns = [
                r'Overall translation diff:\s*max=([\d.e+-]+),\s*mean=([\d.e+-]+)',
                r'Overall translation diff:\s*max\s*=\s*([\d.e+-]+),\s*mean\s*=\s*([\d.e+-]+)',
                r'overall.*?translation.*?diff.*?max\s*=\s*([\d.e+-]+).*?mean\s*=\s*([\d.e+-]+)',
                r'overall.*?translation.*?diff.*?max=([\d.e+-]+).*?mean=([\d.e+-]+)',
            ]
            
            for pattern in patterns:
                translation_diff_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
                if translation_diff_match:
                    try:
                        if max_diff is None:
                            max_diff = float(translation_diff_match.group(1))
                        if mean_diff is None:
                            mean_diff = float(translation_diff_match.group(2))
                        if max_diff is not None and mean_diff is not None:
                            break  # Found both, stop trying other patterns
                    except (ValueError, IndexError):
                        continue
        
        # Method 4: Extract max and mean separately if still not found
        if max_diff is None:
            max_patterns = [
                r'Overall translation diff:\s*max=([\d.e+-]+)',
                r'overall.*?translation.*?diff.*?max\s*=\s*([\d.e+-]+)',
                r'overall.*?t_max.*?=\s*([\d.e+-]+)',
            ]
            for pattern in max_patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    try:
                        max_diff = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        
        if mean_diff is None:
            mean_patterns = [
                r'Overall translation diff:.*?mean=([\d.e+-]+)',
                r'overall.*?translation.*?diff.*?mean\s*=\s*([\d.e+-]+)',
                r'overall.*?t_mean.*?=\s*([\d.e+-]+)',
            ]
            for pattern in mean_patterns:
                match = re.search(pattern, output, re.IGNORECASE)
                if match:
                    try:
                        mean_diff = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
    
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
        max_diff=max_diff,
        mean_diff=mean_diff,
        output=output
    )


def parse_update_output(output: str) -> ComparisonResult:
    """Parse update model comparison output."""
    match_status = "ERROR"
    details = ""
    max_diff = None
    mean_diff = None
    
    # Look for comparison table
    if "UPDATE MODEL OUTPUT COMPARISON" in output:
        # Parse the table to find mismatches
        # Track status for each output type
        output_status = {}  # {output_name: True/False for match/mismatch}
        
        # Check each output type from the table
        lines = output.split('\n')
        for line in lines:
            if "net_out" in line.lower() or "netOut" in line.lower():
                if "❌ MISMATCH" in line:
                    output_status["net_out"] = False
                elif "✅ MATCH" in line:
                    output_status["net_out"] = True
            if "d_out" in line.lower() or "dOut" in line.lower():
                if "❌ MISMATCH" in line:
                    output_status["d_out"] = False
                elif "✅ MATCH" in line:
                    output_status["d_out"] = True
            if "w_out" in line.lower() or "wOut" in line.lower():
                if "❌ MISMATCH" in line:
                    output_status["w_out"] = False
                elif "✅ MATCH" in line:
                    output_status["w_out"] = True
        
        # Build details string with checkmarks/crosses
        detail_parts = []
        for output_name in ["net_out", "d_out", "w_out"]:
            if output_name in output_status:
                status_symbol = "✅" if output_status[output_name] else "❌"
                detail_parts.append(f"{output_name}{status_symbol}")
        
        if detail_parts:
            details = " ".join(detail_parts)
            # Determine overall match status
            if all(output_status.values()):
                match_status = "MATCH"
            else:
                match_status = "MISMATCH"
    else:
        # Fallback: look for match indicators
        if "✅ All outputs match" in output or "All matches: True" in output:
            match_status = "MATCH"
            details = "net_out✅ d_out✅ w_out✅"
        elif "❌" in output and "MISMATCH" in output:
            match_status = "MISMATCH"
            details = "Some outputs mismatched"
    
    # Extract max diff and mean diff from comparison table
    if "UPDATE MODEL OUTPUT COMPARISON" in output:
        # The table format is:
        # Output              Status        Max Diff        Mean Diff       Mismatches      Shape
        # net_out             ✅ MATCH      0.000000        0.000000        0/12345         ...
        # d_out               ✅ MATCH      0.000000        0.000000        0/12345         ...
        # w_out               ✅ MATCH      0.000000        0.000000        0/12345         ...
        
        # Method 1: Extract from parseable format lines added by Update Model script
        # Format: "UPDATE_MODEL_MAX_DIFF=1.234567e-03"
        #         "UPDATE_MODEL_MEAN_DIFF=5.678901e-04"
        update_max_diff_match = re.search(r'UPDATE_MODEL_MAX_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
        update_mean_diff_match = re.search(r'UPDATE_MODEL_MEAN_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
        
        if update_max_diff_match:
            try:
                max_diff = float(update_max_diff_match.group(1))
            except ValueError:
                pass
        
        if update_mean_diff_match:
            try:
                mean_diff = float(update_mean_diff_match.group(1))
            except ValueError:
                pass
        
        # Method 2: Parse table rows if parseable format not found
        if max_diff is None or mean_diff is None:
            # Extract all max_diff and mean_diff values from table rows
            max_diffs = []
            mean_diffs = []
            
            # Parse each line in the table section to extract max_diff and mean_diff
            # The table uses fixed-width columns, so we parse line by line
            lines = output.split('\n')
            in_table = False
            header_found = False
            
            for line in lines:
                if "UPDATE MODEL OUTPUT COMPARISON" in line:
                    in_table = True
                    continue
                if in_table and "="*100 in line and header_found:
                    break  # End of table
                if in_table and "Max Diff" in line and "Mean Diff" in line:
                    header_found = True
                    continue
                if in_table and header_found:
                    # Check if this line contains an output name
                    if any(output_name in line.lower() for output_name in ['net_out', 'd_out', 'w_out']):
                        # Extract numbers from this line
                        # Format: "net_out             ✅ MATCH      0.000000        0.000000        0/12345         ..."
                        # Numbers appear after the status (✅ MATCH or ❌ MISMATCH)
                        # We want the first two numbers after the status
                        numbers = re.findall(r'([\d.e+-]+)', line)
                        if len(numbers) >= 2:
                            try:
                                # Usually the first two numbers after status are max_diff and mean_diff
                                max_diffs.append(float(numbers[0]))
                                mean_diffs.append(float(numbers[1]))
                            except (ValueError, IndexError):
                                continue
            
            # If found multiple outputs, use the maximum max_diff and average mean_diff
            if max_diffs and max_diff is None:
                max_diff = max(max_diffs)
            if mean_diffs and mean_diff is None:
                mean_diff = sum(mean_diffs) / len(mean_diffs)  # Average across all outputs
        
        # Fallback: Try regex pattern matching if line-by-line parsing failed
        if max_diff is None or mean_diff is None:
            # Pattern to match table rows: output_name, status, max_diff, mean_diff
            # Example: "net_out             ✅ MATCH      0.000000        0.000000"
            table_row_patterns = [
                r'(?:net_out|d_out|w_out)\s+(?:✅|❌)\s+MATCH\s+([\d.e+-]+)\s+([\d.e+-]+)',
                r'(?:net_out|d_out|w_out)[^\d]*?([\d.e+-]+)\s+([\d.e+-]+)',
            ]
            
            for pattern in table_row_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
                if matches:
                    temp_max_diffs = []
                    temp_mean_diffs = []
                    for match in matches:
                        try:
                            temp_max_diffs.append(float(match[0]))
                            temp_mean_diffs.append(float(match[1]))
                        except (ValueError, IndexError):
                            continue
                    
                    if temp_max_diffs and max_diff is None:
                        max_diff = max(temp_max_diffs)
                    if temp_mean_diffs and mean_diff is None:
                        mean_diff = sum(temp_mean_diffs) / len(temp_mean_diffs)
                    break  # Found values, stop trying other patterns
    
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
        max_diff=max_diff,
        mean_diff=mean_diff,
        output=output
    )


def parse_patchify_output(output: str) -> ComparisonResult:
    """Parse patchify comparison output."""
    match_status = "ERROR"
    details = ""
    max_diff = None
    mean_diff = None
    
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
    
    # Extract max diff and mean diff from parseable format lines
    # Method 1: Extract from parseable format lines added by Patchify script
    # Format: "PATCHIFY_MAX_DIFF=1.234567e-03"
    #         "PATCHIFY_MEAN_DIFF=5.678901e-04"
    patchify_max_diff_match = re.search(r'PATCHIFY_MAX_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
    patchify_mean_diff_match = re.search(r'PATCHIFY_MEAN_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
    
    if patchify_max_diff_match:
        try:
            max_diff = float(patchify_max_diff_match.group(1))
        except ValueError:
            pass
    
    if patchify_mean_diff_match:
        try:
            mean_diff = float(patchify_mean_diff_match.group(1))
        except ValueError:
            pass
    
    # Method 2: Fallback - extract from comparison tables if parseable format not found
    if max_diff is None or mean_diff is None:
        max_diff_match = re.search(r'Max.*?Diff.*?([\d.e+-]+)', output, re.IGNORECASE)
        mean_diff_match = re.search(r'Mean.*?Diff.*?([\d.e+-]+)', output, re.IGNORECASE)
        
        if max_diff_match and max_diff is None:
            try:
                max_diff = float(max_diff_match.group(1))
            except ValueError:
                pass
        
        if mean_diff_match and mean_diff is None:
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
        name="Patchify",
        script="compare_patchify.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        max_diff=max_diff,
        mean_diff=mean_diff,
        output=output
    )


def parse_keyframe_output(output: str) -> ComparisonResult:
    """Parse keyframe comparison output."""
    match_status = "ERROR"
    details = ""
    max_diff = None
    mean_diff = None
    
    # Extract sections from output to avoid false matches from other sections
    metadata_section = ""
    poses_section = ""
    edges_section = ""
    index_section = ""
    
    # Split output into sections
    if "Metadata Comparison" in output:
        parts = output.split("Metadata Comparison", 1)
        if len(parts) > 1:
            rest = parts[1]
            if "Poses Comparison" in rest:
                metadata_section = rest.split("Poses Comparison")[0]
            else:
                metadata_section = rest
    
    if "Poses Comparison" in output:
        parts = output.split("Poses Comparison", 1)
        if len(parts) > 1:
            rest = parts[1]
            if "Edge Indices Comparison" in rest:
                poses_section = rest.split("Edge Indices Comparison")[0]
            else:
                poses_section = rest
    
    if "Edge Indices Comparison" in output:
        parts = output.split("Edge Indices Comparison", 1)
        if len(parts) > 1:
            rest = parts[1]
            if "Index Comparison" in rest:
                edges_section = rest.split("Index Comparison")[0]
            else:
                edges_section = rest
    
    if "Index Comparison" in output:
        parts = output.split("Index Comparison", 1)
        if len(parts) > 1:
            index_section = parts[1]
    
    # Check metadata comparison - all should match
    metadata_match = True
    if metadata_section:
        if "❌ MISMATCH" in metadata_section:
            metadata_match = False
    
    # Check poses comparison
    poses_match = False
    poses_max_diff = None
    poses_mean_diff = None
    if poses_section:
        # Extract max diff and mean diff from poses comparison
        # Format: "Max diff: 0.000000e+00, Mean diff: 0.000000e+00"
        poses_diff_match = re.search(r'Max diff:\s*([\d.e+-]+),\s*Mean diff:\s*([\d.e+-]+)', poses_section)
        if poses_diff_match:
            try:
                poses_max_diff = float(poses_diff_match.group(1))
                poses_mean_diff = float(poses_diff_match.group(2))
            except ValueError:
                pass
        
        # Check if poses match - look specifically in poses_section
        # Try multiple patterns to catch different formats
        if re.search(r'✅\s*MATCH', poses_section) or "✅ MATCH" in poses_section:
            poses_match = True
        elif re.search(r'❌\s*MISMATCH', poses_section) or "❌ MISMATCH" in poses_section:
            poses_match = False
        # Fallback: check if max diff is very small (likely a match)
        elif poses_max_diff is not None and poses_max_diff < 1e-4:
            poses_match = True
        # Another fallback: if we found the diff values but no explicit match status,
        # assume match if diff is very small
        elif poses_max_diff is not None and poses_max_diff < 1e-3:
            poses_match = True
    
    # Check edge indices comparison
    edges_match = True
    if edges_section:
        # Check if num_edges_after matches
        if "❌ MISMATCH" in edges_section:
            edges_match = False
        # Check individual components (ii, jj, kk) - look for mismatches in edges_section
        if "All edges comparison" in edges_section:
            if "❌ MISMATCH" in edges_section:
                edges_match = False
    
    # Check index comparison
    index_match = True
    if index_section:
        if "❌ MISMATCH" in index_section:
            index_match = False
    
    # Determine overall match status
    if metadata_match and poses_match and edges_match and index_match:
        match_status = "MATCH"
        details_parts = []
        if metadata_match:
            details_parts.append("metadata✅")
        if poses_match:
            details_parts.append("poses✅")
        if edges_match:
            details_parts.append("edges✅")
        if index_match:
            details_parts.append("index✅")
        details = ", ".join(details_parts)
    else:
        match_status = "MISMATCH"
        details_parts = []
        if not metadata_match:
            details_parts.append("metadata❌")
        if not poses_match:
            details_parts.append("poses❌")
        if not edges_match:
            details_parts.append("edges❌")
        if not index_match:
            details_parts.append("index❌")
        details = ", ".join(details_parts) if details_parts else "Mismatches detected"
    
    # Use poses max/mean diff as overall diff values
    max_diff = poses_max_diff
    mean_diff = poses_mean_diff
    
    # If we couldn't find any sections, try a more lenient approach
    if not metadata_section and not poses_section and not edges_section and not index_section:
        # Fallback: check for overall success indicators
        if "Comparison complete!" in output and "✅ MATCH" in output:
            # Count matches vs mismatches
            match_count = output.count("✅ MATCH")
            mismatch_count = output.count("❌ MISMATCH")
            if mismatch_count == 0 and match_count >= 4:  # At least 4 sections should match
                match_status = "MATCH"
                details = "All sections matched"
            elif mismatch_count > 0:
                match_status = "MISMATCH"
                details = f"{mismatch_count} sections mismatched"
    
    if match_status == "ERROR":
        if "File not found" in output or "FileNotFoundError" in output:
            match_status = "SKIPPED"
            details = "Required files not found"
        elif "Error" in output or "Traceback" in output:
            # Script failed with an error
            match_status = "ERROR"
            # Extract error message if possible
            error_match = re.search(r'Error[^\n]*', output)
            if error_match:
                details = error_match.group(0)[:50]
            else:
                details = "Script execution error (check output)"
        else:
            details = "Could not parse output"
    
    return ComparisonResult(
        name="Keyframe",
        script="compare_keyframe.py",
        command=[],
        exit_code=0,  # Will be set by run_keyframe_comparison
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        max_diff=max_diff,
        mean_diff=mean_diff,
        output=output
    )


def parse_onnx_output(output: str) -> ComparisonResult:
    """Parse ONNX model comparison output."""
    match_status = "ERROR"
    details = ""
    max_diff = None
    mean_diff = None
    
    # Look for match indicators
    if "✅ All outputs match" in output or "All matches: True" in output:
        match_status = "MATCH"
    elif "❌" in output and "Mismatch" in output:
        match_status = "MISMATCH"
    
    # First, check the "Final Summary" section which is more reliable
    # Format: "FNet (Python DPVO preprocessing)         ❌ DIFFER" or "✅ MATCH"
    # The summary section ends with a line of "=" characters
    summary_section = re.search(r'📊 Final Summary.*?(?=\n.*?❌ WARNING|\Z)', output, re.DOTALL)
    if summary_section:
        summary_text = summary_section.group(0)
        # Check for FNet status in summary - look for the line with FNet and status
        fnet_summary_line = re.search(r'FNet \(Python DPVO preprocessing\)[^\n]*(?:✅ MATCH|❌ DIFFER)', summary_text)
        if fnet_summary_line:
            if "✅ MATCH" in fnet_summary_line.group(0):
                fnet_match = True
            elif "❌ DIFFER" in fnet_summary_line.group(0):
                fnet_match = False
        # Check for INet status in summary - look for the line with INet and status
        inet_summary_line = re.search(r'INet \(Python DPVO preprocessing\)[^\n]*(?:✅ MATCH|❌ DIFFER)', summary_text)
        if inet_summary_line:
            if "✅ MATCH" in inet_summary_line.group(0):
                inet_match = True
            elif "❌ DIFFER" in inet_summary_line.group(0):
                inet_match = False
    
    # Check FNet and INet separately by looking at their actual status rows
    # The output format is: "FNet" followed by status like "✅ MATCH" or "❌ DIFFER"
    # We need to check the status on the same line or nearby lines
    fnet_match = None  # None = not found, True = match, False = mismatch
    inet_match = None
    
    # Method 1: Look for table rows with FNet/INet and their status
    # Pattern: "FNet" followed by status indicators on the same line
    # The actual format is: "FNet                      ❌ DIFFER" or "FNet                      ✅ MATCH"
    fnet_patterns = [
        r'FNet\s+✅\s+MATCH',  # FNet followed by ✅ MATCH
        r'FNet[^\n]*✅\s+MATCH',  # FNet followed by any chars then ✅ MATCH
        r'FNet[^\n]*❌\s+DIFFER',  # FNet followed by any chars then ❌ DIFFER
        r'^FNet\s+\S+\s+✅\s+MATCH',  # FNet at start of line, then status
        r'^FNet\s+\S+\s+❌\s+DIFFER',  # FNet at start of line, then status
    ]
    inet_patterns = [
        r'INet\s+✅\s+MATCH',  # INet followed by ✅ MATCH
        r'INet[^\n]*✅\s+MATCH',  # INet followed by any chars then ✅ MATCH
        r'INet[^\n]*❌\s+DIFFER',  # INet followed by any chars then ❌ DIFFER
        r'^INet\s+\S+\s+✅\s+MATCH',  # INet at start of line, then status
        r'^INet\s+\S+\s+❌\s+DIFFER',  # INet at start of line, then status
    ]
    
    for pattern in fnet_patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        if match:
            matched_text = match.group(0)
            if "✅ MATCH" in matched_text:
                fnet_match = True
            elif "❌ DIFFER" in matched_text:
                fnet_match = False
            break
    
    for pattern in inet_patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        if match:
            matched_text = match.group(0)
            if "✅ MATCH" in matched_text:
                inet_match = True
            elif "❌ DIFFER" in matched_text:
                inet_match = False
            break
    
    # Method 2: Fallback - check line by line for FNet/INet rows
    if fnet_match is None and "FNet" in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if "FNet" in line:
                if "✅ MATCH" in line:
                    fnet_match = True
                    break
                elif "❌ DIFFER" in line:
                    fnet_match = False
                    break
    
    if inet_match is None and "INet" in output:
        lines = output.split('\n')
        for i, line in enumerate(lines):
            if "INet" in line:
                if "✅ MATCH" in line:
                    inet_match = True
                    break
                elif "❌ DIFFER" in line:
                    inet_match = False
                    break
    
    # Determine overall match status
    if fnet_match is True and inet_match is True:
        match_status = "MATCH"
        details = "FNet✅, INet✅"
    elif fnet_match is False or inet_match is False:
        match_status = "MISMATCH"
        mismatches = []
        if fnet_match is False:
            mismatches.append("FNet❌")
        elif fnet_match is None:
            mismatches.append("FNet?")
        if inet_match is False:
            mismatches.append("INet❌")
        elif inet_match is None:
            mismatches.append("INet?")
        details = ", ".join(mismatches)
    
    # Extract max diff and mean diff from parseable format lines
    # Method 1: Extract from parseable format lines added by ONNX script
    # Format: "ONNX_MODELS_MAX_DIFF=1.234567e-03"
    #         "ONNX_MODELS_MEAN_DIFF=5.678901e-04"
    onnx_max_diff_match = re.search(r'ONNX_MODELS_MAX_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
    onnx_mean_diff_match = re.search(r'ONNX_MODELS_MEAN_DIFF=([\d.e+-]+)', output, re.IGNORECASE)
    
    if onnx_max_diff_match:
        try:
            max_diff = float(onnx_max_diff_match.group(1))
        except ValueError:
            pass
    
    if onnx_mean_diff_match:
        try:
            mean_diff = float(onnx_mean_diff_match.group(1))
        except ValueError:
            pass
    
    # Method 2: Fallback - extract from comparison tables if parseable format not found
    if max_diff is None or mean_diff is None:
        max_diff_match = re.search(r'Max.*?Diff.*?([\d.e+-]+)', output, re.IGNORECASE)
        mean_diff_match = re.search(r'Mean.*?Diff.*?([\d.e+-]+)', output, re.IGNORECASE)
        
        if max_diff_match and max_diff is None:
            try:
                max_diff = float(max_diff_match.group(1))
            except ValueError:
                pass
        
        if mean_diff_match and mean_diff is None:
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
        name="ONNX Models",
        script="compare_onnx_outputs.py",
        command=[],
        exit_code=0,
        success=(match_status != "ERROR"),
        match_status=match_status,
        details=details,
        max_diff=max_diff,
        mean_diff=mean_diff,
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
    """Run reproject all edges comparison."""
    # Use compare_reproject_all_edges.py instead of compare_reproject_intermediate.py
    # The edge parameter is ignored since we compare all edges
    cmd = [sys.executable, "compare_reproject_all_edges.py", 
           "--frame", str(frame), "--tolerance", str(1e-2)]
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


def run_keyframe_comparison(frame: int, network_path: Optional[str] = None) -> ComparisonResult:
    """Run keyframe comparison."""
    cmd = [sys.executable, "compare_keyframe.py", "--frame", str(frame)]
    if network_path:
        cmd.extend(["--network", network_path])
    exit_code, output = run_command(cmd)
    result = parse_keyframe_output(output)
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


def format_diff_value(val: Optional[float]) -> str:
    """Format a difference value for display."""
    if val is None:
        return "N/A"
    if val == 0.0:
        return "0.0"
    elif abs(val) < 1e-6:
        return f"{val:.2e}"
    elif abs(val) < 1.0:
        return f"{val:.6f}"
    elif abs(val) < 1000.0:
        return f"{val:.4f}"
    else:
        return f"{val:.2e}"


def format_status_with_emoji(status: str) -> str:
    """Format match status with emoji indicator."""
    status_emoji_map = {
        "MATCH": "✅ MATCH",
        "MISMATCH": "❌ MISMATCH",
        "SKIPPED": "⚠️  SKIPPED",
        "ERROR": "❌ ERROR"
    }
    return status_emoji_map.get(status, status)


def print_summary_table(results: List[ComparisonResult]):
    """Print a summary table of all comparison results."""
    print("\n" + "="*120)
    print("COMPARISON SUMMARY TABLE")
    print("="*120)
    print(f"{'Component':<30} {'Status':<15} {'Exit Code':<12} {'Max Diff':<15} {'Mean Diff':<15} {'Details':<33}")
    print("-"*120)
    
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
        
        # Format max diff and mean diff
        max_diff_str = format_diff_value(result.max_diff)
        mean_diff_str = format_diff_value(result.mean_diff)
        
        # Truncate details if too long
        details = result.details[:30] + "..." if len(result.details) > 33 else result.details
        
        print(f"{result.name:<30} {status_emoji:<15} {exit_str:<12} {max_diff_str:<15} {mean_diff_str:<15} {details:<33}")
    
    print("="*120)
    
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
                       help="Path to update model ONNX file (default: onnx_models/update.onnx)")
    parser.add_argument("--image", type=str, default=None,
                       help="Path to image file for ONNX comparison (default: build/data/IMG_0482/{frame:05d}.jpg)")
    parser.add_argument("--fnet-model", type=str, default=None,
                       help="Path to FNet ONNX model (default: fnet.onnx)")
    parser.add_argument("--inet-model", type=str, default=None,
                       help="Path to INet ONNX model (default: inet.onnx)")
    parser.add_argument("--fnet-bin", type=str, default=None,
                       help="Path to C++ FNet output binary (default: bin_file/fnet_frame{frame}.bin)")
    parser.add_argument("--inet-bin", type=str, default=None,
                       help="Path to C++ INet output binary (default: bin_file/inet_frame{frame}.bin)")
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
    parser.add_argument("--skip-keyframe", action="store_true",
                       help="Skip keyframe comparison")
    parser.add_argument("--show-output", action="store_true",
                       help="Show full output from each comparison")
    parser.add_argument("--save-output", type=str, default=None,
                       help="Save detailed output to file")
    
    args = parser.parse_args()
    
    # Auto-generate default paths based on frame number if not provided
    if args.image is None:
        args.image = f"build/data/IMG_0482/{args.frame:05d}.jpg"
    
    if args.fnet_model is None:
        args.fnet_model = "fnet.onnx"
    
    if args.inet_model is None:
        args.inet_model = "inet.onnx"
    
    if args.fnet_bin is None:
        args.fnet_bin = f"bin_file/fnet_frame{args.frame}.bin"
    
    if args.inet_bin is None:
        args.inet_bin = f"bin_file/inet_frame{args.frame}.bin"
    
    if args.update_model is None:
        args.update_model = "onnx_models/update.onnx"
    
    print("="*100)
    print("RUNNING ALL COMPARISONS")
    print("="*100)
    print(f"Frame: {args.frame}")
    print(f"Edge: {args.edge}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)
    print(f"Using defaults:")
    print(f"  Image: {args.image}")
    print(f"  FNet model: {args.fnet_model}")
    print(f"  INet model: {args.inet_model}")
    print(f"  FNet bin: {args.fnet_bin}")
    print(f"  INet bin: {args.inet_bin}")
    print(f"  Update model: {args.update_model}")
    print("="*100)
    
    results = []
    
    # 1. ONNX model comparison
    if not args.skip_onnx:
        if args.image and args.fnet_model and args.inet_model:
            print("\n[1/7] Running ONNX model comparison...")
            result = run_onnx_comparison(
                args.image, args.fnet_model, args.inet_model,
                args.fnet_bin, args.inet_bin
            )
            results.append(result)
            print(f"    Status: {format_status_with_emoji(result.match_status)}")
        else:
            print("\n[1/7] Skipping ONNX model comparison (--image, --fnet-model, --inet-model not all provided)")
    else:
        print("\n[1/7] Skipping ONNX model comparison")
    
    
    
    # 2. Patchify comparison
    if not args.skip_patchify:
        print("\n[2/7] Running patchify comparison...")
        result = run_patchify_comparison(args.frame)
        results.append(result)
        print(f"    Status: {format_status_with_emoji(result.match_status)}")
    else:
        print("\n[2/7] Skipping patchify comparison")
    
    
    # 3. Reproject intermediate comparison
    if not args.skip_reproject:
        print("\n[3/7] Running reproject intermediate comparison...")
        result = run_reproject_comparison(args.frame, args.edge)
        results.append(result)
        print(f"    Status: {format_status_with_emoji(result.match_status)}")
    else:
        print("\n[3/7] Skipping reproject intermediate comparison")
    
    # 4. Correlation comparison
    if not args.skip_correlation:
        print("\n[4/7] Running correlation comparison...")
        result = run_correlation_comparison(args.frame)
        results.append(result)
        print(f"    Status: {format_status_with_emoji(result.match_status)}")
    else:
        print("\n[4/7] Skipping correlation comparison")
    
    
    # 5. Update model comparison
    if not args.skip_update:
        if args.update_model:
            print("\n[5/7] Running update model comparison...")
            result = run_update_comparison(args.update_model, args.frame)
            results.append(result)
            print(f"    Status: {format_status_with_emoji(result.match_status)}")
        else:
            print("\n[5/7] Skipping update model comparison (--update-model not provided)")
    else:
        print("\n[5/7] Skipping update model comparison")
    
    
    
    # 6. BA step-by-step comparison
    if not args.skip_ba:
        print("\n[6/7] Running BA step-by-step comparison...")
        result = run_ba_comparison()
        results.append(result)
        print(f"    Status: {format_status_with_emoji(result.match_status)}")
    else:
        print("\n[6/7] Skipping BA step-by-step comparison")
    
    
    
    
    # 7. Keyframe comparison
    if not args.skip_keyframe:
        print("\n[7/7] Running keyframe comparison...")
        # Try to find network path from environment or use default
         # Common locations for DPVO network file
        network_path = os.environ.get("DPVO_NETWORK_PATH", None)
        if not network_path:
            # Try common locations
            possible_paths = [
                "../DPVO_onnx/dpvo.pth",
                "dpvo.pth",
                os.path.expanduser("~/DPVO_onnx/dpvo.pth"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    network_path = path
                    break
        if network_path and not os.path.exists(network_path):
            network_path = None  # Invalid path, let script use dummy network
        result = run_keyframe_comparison(args.frame, network_path)
        results.append(result)
        print(f"    Status: {format_status_with_emoji(result.match_status)}")
        if result.exit_code != 0:
            print(f"    Warning: Script exited with code {result.exit_code}")
            if args.show_output:
                print(f"    Output:\n{result.output}")
    else:
        print("\n[7/7] Skipping keyframe comparison")
    
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

