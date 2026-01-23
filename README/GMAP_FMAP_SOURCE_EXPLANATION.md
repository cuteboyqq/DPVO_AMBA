# gmap, fmap1, fmap2 Source and Usage Explanation

## Overview

`gmap`, `fmap1`, and `fmap2` are feature maps extracted from neural network outputs and stored in **ring buffers** (circular buffers that overwrite old frames). They do NOT store all 36 frames simultaneously for correlation - correlation is only computed for **specific frame pairs** connected by **edges**.

---

## 1. Where They Come From

### **gmap** (Patch Features)
- **Source**: FNet model output `[128, fmap_H, fmap_W]` (128 channels at 1/4 resolution)
- **Extraction**: `patchify_cpu_safe` extracts 3×3 patches at patch center coordinates
- **Output Shape**: `[M, 128, 3, 3]` = `[8, 128, 3, 3]` per frame
  - 8 patches per frame
  - Each patch is 3×3 pixels
  - 128 feature channels per pixel
- **Storage**: Ring buffer `m_gmap[m_pmem][M][128][3][3]` = `[36][8][128][3][3]`
  - `m_pmem = 36` (BUFFER_SIZE)
  - Each new frame overwrites the oldest frame: `m_gmap[n % 36]`
- **When Updated**: Every frame in `DPVO::run()` → `patchify.forward()` → extracts patches from FNet output

### **fmap1** (Full Feature Map at 1/4 Resolution)
- **Source**: FNet model output `[128, fmap_H, fmap_W]` (e.g., `[128, 132, 240]` for 1920×1080 input)
- **Extraction**: Direct copy from FNet output (no patch extraction)
- **Output Shape**: `[128, fmap1_H, fmap1_W]` = `[128, 132, 240]` per frame
- **Storage**: Ring buffer `m_fmap1[m_mem][128][fmap1_H][fmap1_W]` = `[36][128][132][240]`
  - `m_mem = 36` (BUFFER_SIZE)
  - Each new frame overwrites the oldest frame: `m_fmap1[n % 36]`
- **When Updated**: Every frame in `DPVO::run()` → `patchify.forward()` → copies FNet output

### **fmap2** (Downsampled Feature Map at 1/16 Resolution)
- **Source**: Downsampled from `fmap1` using 4×4 average pooling
- **Extraction**: `DPVO::runAfterPatchify()` → averages 4×4 blocks from `fmap1`
- **Output Shape**: `[128, fmap2_H, fmap2_W]` = `[128, 33, 60]` per frame (1/4 of fmap1)
- **Storage**: Ring buffer `m_fmap2[m_mem][128][fmap2_H][fmap2_W]` = `[36][128][33][60]`
  - `m_mem = 36` (BUFFER_SIZE)
  - Each new frame overwrites the oldest frame: `m_fmap2[n % 36]`
- **When Updated**: Every frame in `DPVO::runAfterPatchify()` → downsamples `fmap1` → `fmap2`

---

## 2. Do They Include 36 Frames?

**Yes, but as a ring buffer (circular buffer):**

- **Ring Buffer Behavior**: The buffers can hold up to 36 frames, but they **overwrite** old frames when full
- **Example**:
  - Frame 0: stored at index `0 % 36 = 0`
  - Frame 1: stored at index `1 % 36 = 1`
  - ...
  - Frame 35: stored at index `35 % 36 = 35`
  - Frame 36: stored at index `36 % 36 = 0` (overwrites frame 0)
  - Frame 37: stored at index `37 % 36 = 1` (overwrites frame 1)
  - etc.

- **Current Sliding Window**: Only `m_pg.m_n` frames (typically 8-10) are actively used in the optimization window
- **Ring Buffer Size**: 36 is the **maximum** number of frames that can be stored, but only frames within the **lifetime window** are used for correlation

---

## 3. Is Correlation Calculated Between All 36 Frames?

**NO!** Correlation is **NOT** calculated between all 36 frames. It's only calculated for **specific frame pairs** connected by **edges**.

### How Edges Work

Edges connect **patches** (from source frames) to **frames** (target frames). They are built by two functions:

#### **edgesForward()** - Forward Edges
```cpp
// Connects patches from older frames to the newest frame
// Example: If m_pg.m_n = 10 and PATCH_LIFETIME = 5:
//   - Patches from frames 5-9 → Frame 9 (newest)
//   - Creates edges: (patch_k, frame_9) for k in [5*M, 9*M)
```

**Purpose**: Track patches from older frames as they appear in the newest frame

#### **edgesBackward()** - Backward Edges
```cpp
// Connects patches from the newest frame to older frames
// Example: If m_pg.m_n = 10 and PATCH_LIFETIME = 5:
//   - Patches from frame 9 (newest) → Frames 5-9
//   - Creates edges: (patch_k, frame_j) for k in [9*M, 10*M) and j in [5, 9]
```

**Purpose**: Track patches from the newest frame as they appeared in older frames

### Correlation Computation

Correlation is computed **only for active edges**:

```cpp
// In DPVO::update()
int num_active = m_pg.m_num_edges;  // Number of edges (NOT 36!)

computeCorrelation(
    m_gmap,        // Source: patch features from ANY frame in ring buffer (via kk)
    m_fmap1,       // Target: full feature map from SPECIFIC frame (via jj)
    m_fmap2,       // Target: downsampled feature map from SPECIFIC frame (via jj)
    coords,        // Reprojected coordinates
    m_pg.m_kk,     // Which patches to use (linear index: frame*M + patch)
    m_pg.m_jj,     // Which target frames to use (frame index in sliding window)
    num_active     // Number of edges (typically 20-50, NOT 36!)
);
```

### Example: Edge Structure

If `m_pg.m_n = 10` (sliding window size) and `PATCH_LIFETIME = 5`:

**Forward Edges** (patches from frames 5-9 → frame 9):
- Edge 0: patch from frame 5 → frame 9
- Edge 1: patch from frame 5 → frame 9
- ...
- Edge 19: patch from frame 9 → frame 9

**Backward Edges** (patches from frame 9 → frames 5-9):
- Edge 20: patch from frame 9 → frame 5
- Edge 21: patch from frame 9 → frame 6
- ...
- Edge 59: patch from frame 9 → frame 9

**Total**: ~40 edges (NOT 36×36 = 1296!)

---

## 4. Data Flow Summary

```
Frame N arrives
    ↓
patchify.forward()
    ├─→ FNet inference → fmap [128, 132, 240]
    │   ├─→ Extract patches → gmap [8, 128, 3, 3] → stored in m_gmap[n % 36]
    │   └─→ Copy full map → fmap1 [128, 132, 240] → stored in m_fmap1[n % 36]
    │
    └─→ INet inference → imap [8, 384] → stored in m_imap[n % 36]

Downsample fmap1 → fmap2 [128, 33, 60] → stored in m_fmap2[n % 36]

Build edges (edgesForward + edgesBackward)
    → Creates ~20-50 edges connecting specific frame pairs

In update():
    For each edge e:
        kk[e] → extract patch from gmap[frame % 36]
        jj[e] → sample features from fmap1[jj[e] % 36] and fmap2[jj[e] % 36]
        → Compute correlation volume
```

---

## 5. Key Points

1. **Ring Buffers**: `gmap`, `fmap1`, `fmap2` are ring buffers (max 36 frames, overwrite old frames)
2. **Not All Frames**: Correlation is **NOT** computed between all 36 frames
3. **Edges Only**: Correlation is computed **only for active edges** (typically 20-50 edges)
4. **Sliding Window**: Only `m_pg.m_n` frames (8-10) are in the active optimization window
5. **Lifetime Window**: Only frames within `PATCH_LIFETIME` (typically 5) are connected by edges
6. **Frame Pairs**: Each edge connects one patch (from source frame) to one frame (target frame)

---

## 6. Python Equivalent

```python
# Python: dpvo.py
# gmap: self.gmap_[n % self.pmem] = gmap  # Ring buffer
# fmap1: self.fmap1_[:, n % self.mem] = F.avg_pool2d(fmap[0], 1, 1)
# fmap2: self.fmap2_[:, n % self.mem] = F.avg_pool2d(fmap[0], 4, 4)

# Edges are built similarly:
#   edgesForward: patches from older frames → newest frame
#   edgesBackward: patches from newest frame → older frames

# Correlation:
#   corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
#   corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
#   # Only computes correlation for edges in ii1, jj1 (NOT all frames!)
```

---

## Summary

- **gmap, fmap1, fmap2**: Stored in ring buffers (max 36 frames, overwrite old frames)
- **Correlation**: Computed only for **active edges** (20-50 edges), NOT all 36 frames
- **Edges**: Connect specific frame pairs based on `edgesForward()` and `edgesBackward()`
- **Sliding Window**: Only `m_pg.m_n` frames (8-10) are actively optimized
- **Lifetime**: Only frames within `PATCH_LIFETIME` (typically 5) are connected

