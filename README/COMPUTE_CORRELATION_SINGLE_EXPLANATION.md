# computeCorrelationSingle Explanation

## Purpose

`computeCorrelationSingle` computes **correlation volumes** between:
1. **Patch features from a source frame** (stored in `gmap` ring buffer)
2. **Frame features from a target frame** (stored in `pyramid` ring buffer) **at reprojected locations**

## What It Does

### Inputs:
- **`gmap`**: Patch features from previous frames `[num_gmap_frames, M, 128, 3, 3]`
  - Contains 3×3 patches extracted from FNet features
  - Stored in ring buffer (can be from any previous frame)
  
- **`pyramid`**: Full feature map from target frame `[num_frames, 128, fmap_H, fmap_W]`
  - Complete FNet feature map for the target frame
  - Can be current frame or any frame in the buffer
  
- **`coords`**: Reprojected coordinates `[num_active, 2, P, P]`
  - Where each patch pixel should appear in the target frame
  - Computed by `reproject()` function using poses and patches
  
- **`ii1`**: Which patch from `gmap` to use (mapped from `kk`)
- **`jj1`**: Which frame from `pyramid` to use (mapped from `jj`)

### Algorithm:

For each active edge `e`:

1. **Extract source patch features**:
   ```cpp
   int gmap_frame = ii1[e] / M;      // Which frame in gmap ring buffer
   int patch_idx = ii1[e] % M;       // Which patch within that frame
   // Extract patch feature: gmap[gmap_frame][patch_idx][channel][i0][j0]
   ```

2. **Get reprojected location**:
   ```cpp
   float x = coords[...] * coord_scale;  // Reprojected x coordinate
   float y = coords[...] * coord_scale;  // Reprojected y coordinate
   ```

3. **For each pixel (i0, j0) in the patch** and **each offset (corr_ii, corr_jj) in correlation window**:
   ```cpp
   // Sample location in target frame
   int i1 = floor(y) + (corr_ii - R);  // R=3, so offset from -3 to +4
   int j1 = floor(x) + (corr_jj - R);
   
   // Extract frame feature from pyramid
   float f2 = pyramid[pyramid_frame][channel][i1][j1];
   
   // Extract patch feature from gmap
   float f1 = gmap[gmap_frame][patch_idx][channel][i0][j0];
   
   // Compute correlation (dot product over all 128 channels)
   sum += f1 * f2;
   ```

4. **Store correlation value**:
   ```cpp
   corr_out[e][corr_ii][corr_jj][i0][j0] = sum;
   ```

### Output:
- **Shape**: `[num_active, D, D, P, P]` where `D = 8` (correlation window), `P = 3` (patch size)
- **Meaning**: For each edge, for each patch pixel, correlation values at 8×8 offsets around the reprojected location

## Visual Example

```
Source Frame (Frame i)              Target Frame (Frame j)
┌─────────────────┐                ┌─────────────────┐
│                 │                │                 │
│   Patch [3×3]   │  ───reproject──→│  ?  ?  ?  ?  ? │
│   from gmap     │                │  ? [x,y] ?  ?  │  ← Reprojected location
│                 │                │  ?  ?  ?  ?  ? │
└─────────────────┘                └─────────────────┘
     ↓                                    ↓
  Extract patch                    Sample frame features
  features: f1                    at (x,y) + offsets: f2
     ↓                                    ↓
     └─────────── dot product ───────────┘
                    ↓
            Correlation value
```

## Key Points

1. **Yes, it computes correlation between previous frame features and reprojected coordinate features**:
   - `gmap` contains patches from **previous frames** (source)
   - `pyramid` contains features from **target frame** (can be current or previous)
   - `coords` tells us **where** the patch should appear in the target frame

2. **The correlation window (8×8) allows sub-pixel matching**:
   - Reprojected coordinates are often not exact integers
   - The 8×8 window searches around the reprojected location
   - This helps find the best match even if reprojection is slightly off

3. **It's called twice** (once per pyramid level):
   - `computeCorrelationSingle(..., pyramid0, ..., 1.0f, ...)`  → Full resolution (1/4)
   - `computeCorrelationSingle(..., pyramid1, ..., 0.25f, ...)` → Low resolution (1/16)
   - Results are stacked together: `[num_active, D, D, P, P, 2]`

## Why This Matters

The correlation volume tells the Update Model:
- **How well does the patch from frame i match the features at the reprojected location in frame j?**
- **What is the best matching offset?** (from the 8×8 correlation window)
- This information is used to refine the patch position and update poses

## Python Equivalent

```python
# Python: altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
corr1 = altcorr.corr(
    self.gmap,        # Patch features from previous frames
    self.pyramid[0],  # Frame features from target frame
    coords / 1,       # Reprojected coordinates
    ii1,              # Which patches to use
    jj1,              # Which frames to use
    3                 # Correlation radius
)
```

This matches exactly what `computeCorrelationSingle` does!


