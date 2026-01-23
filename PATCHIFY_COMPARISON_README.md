# Patchify Comparison Guide

This guide explains how to compare C++ and Python patchify outputs to verify they produce identical results.

## Overview

The patchify function extracts patches from feature maps (fnet/inet outputs) at specified coordinates. We need to verify that C++ and Python produce the same patches when given the same inputs.

## Files

- `compare_patchify.py`: Python script that runs Python patchify and compares with C++ outputs
- `fnet_frame0.bin`: C++ FNet output (already saved by C++ code)
- `inet_frame0.bin`: C++ INet output (already saved by C++ code)
- `cpp_gmap_frame0.bin`: C++ gmap output (saved by C++ patchify)
- `cpp_imap_frame0.bin`: C++ imap output (saved by C++ patchify)
- `cpp_patches_frame0.bin`: C++ patches output (saved by C++ patchify)
- `cpp_coords_frame0.bin`: C++ coordinates used (saved by C++ patchify)
- `python_gmap_frame0.bin`: Python gmap output (saved by comparison script)
- `python_imap_frame0.bin`: Python imap output (saved by comparison script)
- `python_patches_frame0.bin`: Python patches output (saved by comparison script)
- `python_coords_frame0.bin`: Python coordinates used (saved by comparison script)

## Steps

### Step 1: Run C++ Code (First Frame)

Run your C++ DPVO code and process at least frame 0. The C++ code will automatically save:
- `fnet_frame0.bin`
- `inet_frame0.bin`
- `cpp_coords_frame0.bin`
- `cpp_gmap_frame0.bin`
- `cpp_imap_frame0.bin`
- `cpp_patches_frame0.bin`

### Step 2: Run Python Comparison Script

```bash
cd /home/ali/Projects/GitHub_Code/clean_code/DPVO_AMBA
python3 compare_patchify.py [frame_idx]
```

Where `frame_idx` is optional (defaults to 0).

The script will:
1. Load `fnet_frame0.bin` and `inet_frame0.bin`
2. Generate the same random coordinates (using fixed seed for reproducibility)
3. Run Python patchify on the feature maps
4. Save Python outputs to `python_*.bin` files
5. If C++ outputs exist, compare them with Python outputs

### Step 3: Review Comparison Results

The script will print:
- ✅ **MATCH**: Arrays are identical (within tolerance)
- ⚠️ **Mostly matches**: >99% of elements match
- ❌ **MISMATCH**: Significant differences found

## Expected Output Shapes

- **gmap**: `[M, 128, P, P]` = `[4, 128, 3, 3]` - patches from fmap
- **imap**: `[M, 384, 1, 1]` = `[4, 384, 1, 1]` - patches from inet (radius=0)
- **patches**: `[M, 3, P, P]` = `[4, 3, 3, 3]` - coordinate patches (x, y, d)

Where:
- `M = 4`: Number of patches per frame
- `P = 3`: Patch size
- `128`: FNet feature channels
- `384`: INet feature channels

## Coordinate Generation

Both C++ and Python use the same random coordinate generation:
- Coordinates are at **feature map resolution** (not full image resolution)
- Range: `x ∈ [1, fmap_W-2]`, `y ∈ [1, fmap_H-2]`
- Python uses `np.random.seed(42)` for reproducibility
- C++ uses `rand()` (but we save the coordinates it uses)

## Differences to Watch For

1. **Bilinear vs Nearest Neighbor**: 
   - Python uses bilinear interpolation (extracts 4×4, uses 3×3)
   - C++ uses nearest neighbor (extracts 3×3 directly)
   - This may cause small differences at patch boundaries

2. **Coordinate Scaling**:
   - Both should use feature map resolution coordinates
   - No scaling should be applied during patchify

3. **Grid Creation**:
   - Grid should be created at feature map resolution
   - Values: `x ∈ [0, fmap_W-1]`, `y ∈ [0, fmap_H-1]`, `d = 1.0`

## Troubleshooting

### Issue: Shape Mismatch

If shapes don't match, check:
- Feature map dimensions (H, W) - should match between fnet and inet
- Patch size (P) - should be 3
- Number of patches (M) - should be 4

### Issue: Large Differences

If differences are large:
1. Check coordinate values match between C++ and Python
2. Verify feature map values match (fnet/inet outputs)
3. Check if interpolation mode differs (bilinear vs nearest)

### Issue: C++ Outputs Not Found

Make sure:
1. C++ code ran successfully
2. C++ code saved outputs (check for `.bin` files)
3. Files are in the current directory

## Next Steps

After verifying patchify matches:
1. Compare `reproject` function
2. Compare `correlation` function
3. Compare `BA` function (already partially verified)

