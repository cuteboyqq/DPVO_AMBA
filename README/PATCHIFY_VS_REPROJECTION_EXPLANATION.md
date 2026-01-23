# Patchify Random Coordinates vs Reprojected Coordinates Explanation

## The Confusion

You're right to be confused! The logic seems contradictory:
- **Patchify**: Uses **random coordinates** to extract patches
- **Update**: Uses **reprojected coordinates** for correlation

But they serve **different purposes** in the DPVO pipeline!

---

## The Two-Stage Process

### Stage 1: Patchify (Initialization) - Random Coordinates

**When**: Every frame in `DPVO::run()` → `patchify.forward()`

**Purpose**: **Initialize new landmarks** (patches) from the current frame

**Process**:
1. Select **M random locations** (e.g., 8 patches) in the current frame
2. Extract patches at these random locations:
   - RGB patches from image
   - Feature patches from FNet (`gmap`)
   - Feature patches from INet (`imap`)
3. **Store** these patches as new landmarks

**Why Random?**
- Random sampling ensures good spatial coverage of the image
- Avoids bias toward specific regions
- Each frame gets a fresh set of landmarks

**What Gets Stored**:
```cpp
// In DPVO::runAfterPatchify()
m_pg.m_patches[n_use][i][0][y][x] = px_pixel_scaled;  // X coordinate (scaled to 1/4 res)
m_pg.m_patches[n_use][i][1][y][x] = py_pixel_scaled;  // Y coordinate (scaled to 1/4 res)
m_pg.m_patches[n_use][i][2][y][x] = depth_value;      // Inverse depth
```

These are the **initial 2D positions** of patches in frame `n_use`.

---

### Stage 2: Update (Tracking) - Reprojected Coordinates

**When**: Every frame in `DPVO::update()` (called after edges are built)

**Purpose**: **Track existing landmarks** across frames

**Process**:
1. For each edge connecting patch `k` from frame `i` to frame `j`:
2. **Reproject** the patch from frame `i` to frame `j`:
   ```cpp
   // In reproject() / transformWithJacobians()
   // Uses stored patch coordinates from frame i:
   float px = m_pg.m_patches[i][patch_idx][0][y][x];  // X coord in frame i
   float py = m_pg.m_patches[i][patch_idx][1][y][x];  // Y coord in frame i
   float pd = m_pg.m_patches[i][patch_idx][2][y][x];  // Inverse depth
   
   // Transforms to 3D, applies pose transform, projects to frame j:
   // → Reprojected coordinates (u, v) in frame j
   ```
3. **Correlate** patch features from frame `i` with features at reprojected location in frame `j`

**Why Reprojected?**
- The patch moved! Camera moved, so the patch appears at a different location in frame `j`
- Reprojection predicts where the patch **should** appear based on:
  - Initial position in frame `i` (from random patchify)
  - Camera pose transformation (from frame `i` to frame `j`)
  - Patch depth estimate

---

## Visual Example

```
Frame 0 (Initialization):
┌─────────────────┐
│                 │
│   ●  ← Random patch at (100, 200)
│                 │
│      ●  ← Random patch at (500, 300)
│                 │
└─────────────────┘
   ↓ Store patches
   m_pg.m_patches[0][0] = (100, 200, depth)
   m_pg.m_patches[0][1] = (500, 300, depth)

Frame 1 (Tracking):
┌─────────────────┐
│                 │
│  Camera moved!  │
│                 │
│     ●  ← Reprojected: patch[0] should be here (150, 250)
│                 │
│           ●  ← Reprojected: patch[1] should be here (550, 350)
│                 │
└─────────────────┘
   ↓ Correlation
   Compare patch[0] features (from frame 0) with features at (150, 250) in frame 1
   Compare patch[1] features (from frame 0) with features at (550, 350) in frame 1
```

---

## Concrete Example: Frame 15 Arrives

**Scenario**: Current frame is 15, `PATCH_LIFETIME = 6`, `M = 8` patches per frame

### Step 1: Patchify (Random Initialization)
- Frame 15 arrives
- Extract **8 random patches** from frame 15
- Store in `m_pg.m_patches[14][0..7]` (0-indexed: frame 15 = index 14)
- Store features in `m_gmap[14 % 36]`, `m_fmap1[14 % 36]`, `m_imap[14 % 36]`

### Step 2: Build Edges

**After processing frame 15, `m_pg.m_n = 15`** (sliding window now contains frames 0-14)

#### Forward Edges (`edgesForward`):
```cpp
r = PATCH_LIFETIME = 6
t0 = M * max(15 - 6, 0) = M * 9  // Patches from frame 9
t1 = M * max(15 - 1, 0) = M * 14 // Patches from frame 14
jj = 15 - 1 = 14  // Target frame is frame 14 (newest before current)
```

**Result**: 
- Patches from frames **9-14** → Frame **14**
- Creates edges: `(patch_k, frame_14)` for `k` in `[9*8, 14*8)` = `[72, 112)`
- Total: **6 frames × 8 patches = 48 forward edges**

#### Backward Edges (`edgesBackward`):
```cpp
r = PATCH_LIFETIME = 6
t0 = M * max(15 - 1, 0) = M * 14  // Patches from frame 14
t1 = M * 15 = M * 15              // Patches from frame 15 (current)
Target frames: max(15 - 6, 0) = 9 to 15 - 1 = 14
```

**Result**:
- Patches from frame **14** → Frames **9-14**
- Creates edges: `(patch_k, frame_j)` for `k` in `[14*8, 15*8)` = `[112, 120)` and `j` in `[9, 14]`
- Total: **8 patches × 6 frames = 48 backward edges**

### Step 3: Update (Correlation)

For each edge `e`:
- **Forward edge example**: `kk[e] = 100` (patch from frame 12), `jj[e] = 14`
  - Read stored coordinates from `m_pg.m_patches[12][4]` (frame 12, patch 4)
  - Reproject to frame 14 → Get reprojected coordinates `(u, v)`
  - Correlate: Compare patch features from `gmap[12]` with features at `(u, v)` in `fmap1[14]`

- **Backward edge example**: `kk[e] = 115` (patch from frame 14), `jj[e] = 10`
  - Read stored coordinates from `m_pg.m_patches[14][3]` (frame 14, patch 3)
  - Reproject to frame 10 → Get reprojected coordinates `(u, v)`
  - Correlate: Compare patch features from `gmap[14]` with features at `(u, v)` in `fmap1[10]`

### Summary for Frame 15:

- **Random patches**: 8 patches extracted from frame 15 (stored as frame index 14)
- **Frame j (target frames)**:
  - **Forward edges**: `j = 14` (single target frame - the newest frame before current)
    - Connects patches from frames **9-14** → frame **14**
  - **Backward edges**: `j ∈ [9, 14]` (range of 6 frames - lifetime window)
    - Connects patches from frame **14** → frames **9-14**
- **Total edges**: ~96 edges (48 forward + 48 backward)
- **Correlation**: Computed for all 96 edges using reprojected coordinates

**Key Point**: Frame `j` is **NOT** just frame 14. It's:
- **Forward edges**: Always `j = m_pg.m_n - 1` (the newest frame before the current one)
- **Backward edges**: `j` ranges from `max(m_pg.m_n - PATCH_LIFETIME, 0)` to `m_pg.m_n - 1` (a range of frames)

So if current frame is 15:
- Forward edges: `j = 14` (single frame)
- Backward edges: `j ∈ [9, 14]` (6 frames in the lifetime window)

---

## Key Insight: Patches Are Landmarks

**Patches are 3D landmarks** that are:
1. **Initialized** once with random 2D positions (in their source frame)
2. **Tracked** across frames using reprojection

The random coordinates are the **starting point** - they define where the landmark was first seen.

The reprojected coordinates are the **prediction** - they predict where the landmark should appear in other frames.

---

## Data Flow

```
Frame N arrives
    ↓
patchify.forward()
    ├─→ Random coordinates: (x1, y1), (x2, y2), ..., (x8, y8)
    ├─→ Extract patches at random locations
    └─→ Store in m_pg.m_patches[N][i][0/1/2] = (x, y, depth)
    
Later, in update():
    For each edge (patch k from frame i → frame j):
        ↓
        reproject()
            ├─→ Read: m_pg.m_patches[i][k][0/1/2] (initial position in frame i)
            ├─→ Transform using poses[i] and poses[j]
            └─→ Output: Reprojected coordinates (u, v) in frame j
        ↓
        computeCorrelation()
            ├─→ Extract patch features from gmap[i][k] (source frame)
            ├─→ Sample features from fmap1[j] at reprojected (u, v) (target frame)
            └─→ Compute correlation volume
        ↓
        Update Model
            ├─→ Uses correlation to refine patch position
            └─→ Updates poses and depths via bundle adjustment
```

---

## Why This Makes Sense

1. **Random Initialization**: Ensures good coverage - patches are distributed across the image
2. **Reprojection**: Predicts where patches should appear based on camera motion
3. **Correlation**: Verifies/refines the prediction by matching features
4. **Bundle Adjustment**: Optimizes poses and depths to minimize reprojection error

The random coordinates are **not** used for correlation - they're used to **initialize** the patches. The reprojected coordinates are used for correlation because they predict where the patches **should** appear in the target frame.

---

## Python Equivalent

```python
# Python: dpvo.py
# Stage 1: Patchify (random initialization)
fmap, gmap, imap, patches, _, clr = \
    self.network.patchify(image, patches_per_image=8, ...)
# patches[:,:,0:2] contains random coordinates

# Stage 2: Update (tracking with reprojection)
coords = self.reproject()  # Reprojects patches using poses
# coords are reprojected coordinates (where patches should appear)

corr = self.corr(coords, ...)  # Correlation at reprojected locations
# Uses reprojected coords, not random coords!
```

---

## Summary

- **Random coordinates** (patchify): Used **once** to initialize patches (landmarks)
- **Stored coordinates** (`m_pg.m_patches`): Initial 2D positions of patches in their source frame
- **Reprojected coordinates** (update): Predicted positions of patches in target frames (computed from poses)
- **Correlation**: Uses reprojected coordinates to match patches across frames

The logic is: **Initialize randomly, track with reprojection!**

