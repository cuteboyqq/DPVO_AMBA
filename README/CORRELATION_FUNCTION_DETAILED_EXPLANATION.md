# Correlation Function Detailed Explanation

## Overview

The correlation function computes **how well a patch from one frame matches features at a reprojected location in another frame**. This is the core matching mechanism in DPVO that enables visual odometry.

---

## The Big Picture

**Question**: How do we know if a patch from frame `i` appears at location `(x, y)` in frame `j`?

**Answer**: Compute correlation between:
1. **Patch features** from frame `i` (stored in `gmap`)
2. **Frame features** at location `(x, y)` in frame `j` (stored in `fmap1`/`fmap2`)

**Method**: Dot product over all feature channels (128 channels for FNet features)

---

## Step-by-Step: What Happens

### Step 1: Extract Patch Features (gmap)

**When**: During `patchify.forward()` for each frame

**What**: Extract a 3×3 spatial patch from FNet features

**Example**: Frame 10, Patch 2
```
FNet output: [128 channels, 132 height, 240 width]
Random coordinate: (x=100, y=50) at 1/4 resolution

Extract 3×3 patch around (100, 50):
gmap[10][2] = [128 channels, 3, 3]
              = [128, 3, 3] feature patch
```

**Storage**: `gmap[frame % 36][patch_idx][channel][y][x]`
- `frame % 36`: Ring buffer index (stores up to 36 frames)
- `patch_idx`: Which patch (0-7)
- `channel`: Feature channel (0-127)
- `y, x`: Spatial position within patch (0-2)

---

### Step 2: Reproject Patch to Target Frame

**When**: During `DPVO::update()` → `reproject()`

**What**: Predict where the patch should appear in the target frame

**Example**: Patch from frame 10 → Frame 14
```
Input:
  - Patch coordinates from frame 10: (x=100, y=50) at 1/4 resolution
  - Patch depth: d = 0.5 (inverse depth)
  - Pose from frame 10 to frame 14: T_10_to_14

Process:
  1. Convert to 3D: X = (x - cx) * d, Y = (y - cy) * d, Z = 1/d
  2. Transform: [X', Y', Z'] = T_10_to_14 * [X, Y, Z]
  3. Project: u = fx * X'/Z' + cx, v = fy * Y'/Z' + cy

Output:
  - Reprojected coordinates: (u=120, v=55) at 1/4 resolution
```

**Storage**: `coords[edge][0/1][y][x]`
- `edge`: Which edge (patch-frame pair)
- `0`: x coordinate, `1`: y coordinate
- `y, x`: Pixel position within patch (0-2)

---

### Step 3: Compute Correlation

**When**: During `DPVO::update()` → `computeCorrelation()`

**What**: Compare patch features with frame features at reprojected location

#### 3.1: For Each Edge

**Example**: Edge connecting patch from frame 10 to frame 14

```cpp
// Extract which patch and frame
int gmap_frame = kk[e] / M;      // = 10 / 8 = 1 (frame 10 in ring buffer)
int patch_idx = kk[e] % M;        // = 10 % 8 = 2 (patch 2)
int pyramid_frame = jj[e];        // = 14 (target frame)
```

#### 3.2: For Each Pixel in Patch (3×3 = 9 pixels)

**Example**: Center pixel `(i0=1, j0=1)` of patch

```cpp
// Get reprojected location for this pixel
float x = coords[e][0][i0][j0];  // = 120.0 (reprojected x)
float y = coords[e][1][i0][j0];  // = 55.0 (reprojected y)
```

#### 3.3: For Each Offset in Correlation Window (8×8 = 64 offsets)

**Why**: Reprojected coordinates are often not exact integers. The correlation window searches around the reprojected location to find the best match.

**Example**: Correlation radius `R = 3`, window size `D = 8`

```cpp
// For each offset in [-3, +4] × [-3, +4]
for (int corr_ii = 0; corr_ii < 8; corr_ii++) {
    for (int corr_jj = 0; corr_jj < 8; corr_jj++) {
        // Sample location in target frame
        int i1 = floor(y) + (corr_ii - 3);  // = 55 + (corr_ii - 3)
        int j1 = floor(x) + (corr_jj - 3);  // = 120 + (corr_jj - 3)
        
        // Example: corr_ii=3, corr_jj=3 → i1=55, j1=120 (center)
        // Example: corr_ii=0, corr_jj=0 → i1=52, j1=117 (top-left)
        // Example: corr_ii=7, corr_jj=7 → i1=59, j1=124 (bottom-right)
```

#### 3.4: Dot Product Over All Channels

**The Core Computation**: For each channel, multiply patch feature by frame feature, then sum

```cpp
float sum = 0.0f;
for (int f = 0; f < 128; f++) {  // For each feature channel
    // Extract patch feature from gmap
    float f1 = gmap[gmap_frame][patch_idx][f][i0][j0];
    // Example: gmap[1][2][50][1][1] = 0.123 (channel 50, center pixel)
    
    // Extract frame feature from pyramid at sampled location
    float f2 = pyramid[pyramid_frame][f][i1][j1];
    // Example: pyramid[14][50][55][120] = 0.456 (channel 50, location (55, 120))
    
    // Accumulate dot product
    sum += f1 * f2;  // = 0.123 * 0.456 = 0.056088
}

// Store correlation value
corr_out[e][corr_ii][corr_jj][i0][j0] = sum;
// Example: corr_out[0][3][3][1][1] = 12.345 (high correlation = good match!)
```

**Mathematical Formula**:
```
correlation = Σ (gmap[frame_i][patch][c][y][x] * fmap[frame_j][c][u+offset_y][v+offset_x])
             c=0..127
```

---

## Visual Example

### Setup
- **Source Frame**: Frame 10
- **Target Frame**: Frame 14
- **Patch**: Patch 2 from frame 10
- **Reprojected Location**: (120, 55) in frame 14

### Step-by-Step Visualization

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Extract Patch Features (gmap)                      │
│                                                              │
│ Frame 10 FNet Output: [128, 132, 240]                      │
│   ↓ Extract 3×3 patch at (100, 50)                          │
│                                                              │
│ gmap[10][2] = [128, 3, 3]                                   │
│   Channel 0:  [0.1, 0.2, 0.3]                              │
│                [0.4, 0.5, 0.6]  ← Center pixel (1,1)       │
│                [0.7, 0.8, 0.9]                              │
│   Channel 1:  [0.2, 0.3, 0.4]                              │
│                [0.5, 0.6, 0.7]                              │
│                [0.8, 0.9, 1.0]                              │
│   ... (128 channels total)                                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 2: Reproject to Target Frame                           │
│                                                              │
│ Patch coordinates (frame 10): (100, 50)                    │
│   ↓ Transform using poses                                   │
│ Reprojected coordinates (frame 14): (120, 55)               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Step 3: Compute Correlation                                │
│                                                              │
│ Frame 14 FNet Output: [128, 132, 240]                      │
│                                                              │
│ Correlation Window (8×8) around (120, 55):                  │
│                                                              │
│   ┌─────────────────────────────────────┐                  │
│   │ 52,117  52,118  ...  52,124         │ ← Top row        │
│   │ 53,117  53,118  ...  53,124         │                  │
│   │  ...     ...    ...   ...           │                  │
│   │ 55,117  55,118  ...  55,124         │ ← Center row     │
│   │  ...     ...    ...   ...           │                  │
│   │ 59,117  59,118  ...  59,124         │ ← Bottom row     │
│   └─────────────────────────────────────┘                  │
│                                                              │
│ For each offset (corr_ii, corr_jj):                         │
│   Sample location: (55 + corr_ii - 3, 120 + corr_jj - 3)     │
│                                                              │
│   Dot product over 128 channels:                           │
│     sum = Σ gmap[10][2][c][1][1] * fmap[14][c][i1][j1]     │
│           c=0..127                                          │
│                                                              │
│   Store: corr_out[edge][corr_ii][corr_jj][1][1] = sum       │
└─────────────────────────────────────────────────────────────┘
```

---

## Why Two Pyramid Levels?

The correlation function computes correlation at **two resolutions**:

### Pyramid Level 0 (fmap1): 1/4 Resolution
- **Input**: `fmap1[frame][128][132][240]` (1/4 resolution)
- **Coordinates**: `coords / 1.0` (no scaling)
- **Purpose**: Fine-grained matching for accurate localization

### Pyramid Level 1 (fmap2): 1/16 Resolution
- **Input**: `fmap2[frame][128][33][60]` (1/16 resolution)
- **Coordinates**: `coords / 4.0` (scale down by 4)
- **Purpose**: Coarse matching for robustness to large motions

**Why Both?**
- **Fine resolution**: Captures small details, accurate for small motions
- **Coarse resolution**: Handles large motions, more robust to errors

**Output**: Stacked together: `[num_active, 8, 8, 3, 3, 2]`
- Channel 0: Correlation with `fmap1` (fine)
- Channel 1: Correlation with `fmap2` (coarse)

---

## What Does Correlation Value Mean?

**High Correlation** (e.g., `corr = 15.0`):
- Patch features match well with frame features
- The patch likely appears at this location
- **Good match!**

**Low Correlation** (e.g., `corr = 0.5`):
- Patch features don't match well
- The patch might not appear here, or reprojection is wrong
- **Poor match**

**Zero Correlation** (e.g., `corr = 0.0`):
- No match at all
- Could be:
  - Reprojection is completely wrong
  - Patch is occluded
  - Features are zero (model output issue)

---

## Complete Example: One Edge

**Edge**: Patch from frame 10 → Frame 14

**Inputs**:
- `gmap[10][2]`: `[128, 3, 3]` patch features
- `fmap1[14]`: `[128, 132, 240]` frame features
- `coords[edge]`: Reprojected coordinates `(120, 55)` for center pixel

**Process**:
1. For center pixel `(i0=1, j0=1)` of patch:
   - Reprojected location: `(x=120, y=55)`
   
2. For each offset `(corr_ii, corr_jj)` in `[0, 7] × [0, 7]`:
   - Sample location: `(i1=55+corr_ii-3, j1=120+corr_jj-3)`
   - Compute dot product over 128 channels:
     ```
     sum = Σ gmap[10][2][c][1][1] * fmap1[14][c][i1][j1]
           c=0..127
     ```
   - Store: `corr_out[edge][corr_ii][corr_jj][1][1] = sum`

3. Repeat for all 9 pixels in patch (3×3)

**Output**: `corr_out[edge]`: `[8, 8, 3, 3]` correlation volume
- For each pixel in patch, correlation values at 8×8 offsets

---

## Key Insights

1. **Correlation = Dot Product**: Measures similarity between patch and frame features
2. **Correlation Window**: Searches around reprojected location (±3 pixels) to handle sub-pixel accuracy
3. **Multi-Scale**: Two pyramid levels (fine + coarse) for robustness
4. **Per-Pixel**: Computes correlation for each pixel in the patch (3×3 = 9 pixels)
5. **Per-Offset**: Computes correlation at multiple offsets (8×8 = 64 offsets per pixel)

---

## How It's Used

The correlation volume is passed to the **Update Model** (neural network), which:
1. Analyzes the correlation pattern
2. Predicts:
   - **Delta**: How much to adjust the reprojected coordinates
   - **Weight**: Confidence in the prediction
3. Updates poses and depths via bundle adjustment

**The correlation volume tells the model**: "Here's how well the patch matches at different offsets around the reprojected location. Use this to refine the patch position and camera pose."

---

## Summary

**Correlation Function**:
1. Takes patch features from source frame (`gmap`)
2. Takes frame features from target frame (`fmap1`/`fmap2`)
3. Computes dot product at reprojected location + offsets
4. Outputs correlation volume `[num_active, 8, 8, 3, 3, 2]`

**Purpose**: Match patches across frames to enable visual odometry

**Key**: It's a **feature matching** mechanism that compares neural network features (not raw pixels) to find correspondences between frames.


