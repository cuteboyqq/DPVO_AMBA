# Points Only Drawing for First Few Frames - Diagnostic

## Problem

Points are only visible for the first few frames in visualization, even though logs show all frames have valid points.

## Root Cause Analysis

### Current Flow

1. **Point Computation** (`computePointCloud()`):
   - Only processes frames currently in the sliding window (`m_pg.m_n`, typically 8-10 frames)
   - Computes points for each patch in each frame
   - Stores points in `m_allPoints[global_frame_idx * M + patch_idx]`

2. **Point Validation**:
   - Checks if `pd > 0` and `pd <= 10.0f` (inverse depth)
   - Checks if coordinates `px`, `py` are valid (within feature map bounds)
   - If invalid, skips point computation but preserves historical points

3. **Point Passing to Viewer** (`updateViewer()`):
   - Passes all historical points: `m_allPoints.data()` with `num_points = num_historical_frames * M`
   - Viewer filters out zero/NaN/Inf points

### Potential Issues

#### Issue 1: Patches Have Invalid Coordinates for Later Frames

**Symptom**: Patches stored with RGB values instead of coordinates, causing coordinate validation to fail.

**Possible Causes**:
- Patches not being updated correctly after BA
- Patch storage being overwritten with RGB values
- Coordinate validation too strict (rejecting valid coordinates)

**Diagnostic**: Check logs for:
```
Point cloud [frame=X, patch=Y]: Invalid coordinates px=..., py=...
```

#### Issue 2: Points Never Computed for Frames That Leave Sliding Window Early

**Symptom**: Frame enters sliding window → patches invalid → points skipped → frame leaves → too late to compute.

**Possible Causes**:
- Patches become invalid before points are computed
- `computePointCloud()` only called for frames in sliding window
- Once frame leaves, patches are no longer accessible

**Solution**: Ensure points are computed immediately when patches are stored (already done in `runAfterPatchify()` line 635).

#### Issue 3: Historical Points Being Overwritten with Zeros

**Symptom**: Points computed initially, but later overwritten with zeros when patches become invalid.

**Current Code**: Code tries to preserve historical points (line 2752, 2772), but might have bugs.

**Check**: Verify that `m_allPoints[global_point_idx]` is NOT being set to zero when patches become invalid.

#### Issue 4: Viewer Filtering Out Valid Points

**Symptom**: Points are computed and stored correctly, but viewer filters them out.

**Current Code**: Viewer filters points with:
- Zero coordinates: `p.x == 0.0f && p.y == 0.0f && p.z == 0.0f`
- NaN/Inf: `!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)`
- Out of bounds: `abs(p.x) > MAX_POINT_DISTANCE` (100000000.0f)

**Check**: Verify viewer is not filtering out valid points incorrectly.

## Diagnostic Logging Added

### 1. Coordinate Validation Logging (`dpvo.cpp` line ~2748-2775)

Logs when patches have invalid coordinates or depth:
```
Point cloud [frame=X, patch=Y]: Invalid depth pd=... (<=0 or >10), skipping
Point cloud [frame=X, patch=Y]: Invalid coordinates px=..., py=... (expected range: ...), skipping
```

### 2. Frame-Level Point Count Logging (`dpvo.cpp` line ~2884-2920)

Logs which frames have zero points:
```
Point cloud: Frames in sliding window with ZERO points: X, Y, Z (these frames have invalid patches)
```

### 3. Viewer Point Count Logging (`dpvo.cpp` line ~3056-3122)

Logs valid points per frame:
```
Viewer update: Valid points per frame (first 5 frames):
  Frame[0]: X valid points
  Frame[1]: Y valid points
  ...
```

## Next Steps

1. **Run the code** and check logs for:
   - Which frames have invalid coordinates/depth
   - Which frames have zero points in sliding window
   - Which frames have zero points when passed to viewer

2. **If frames have invalid coordinates**:
   - Check why patches are stored with RGB values instead of coordinates
   - Verify patch storage in `runAfterPatchify()` is correct
   - Check if patches are being overwritten incorrectly

3. **If frames have valid coordinates but zero points**:
   - Check if coordinate validation is too strict
   - Verify `computePointCloud()` is being called for all frames
   - Check if points are being overwritten with zeros

4. **If points are computed but not visible**:
   - Check viewer filtering logic
   - Verify points are being passed correctly to viewer
   - Check if points are outside viewer's visible range

## Expected Log Output

After running, you should see logs like:
```
Point cloud: Computed X new points, preserved Y existing points, total frames in sliding window: Z
Point cloud: Frames with new points computed: 0, 1, 2, ...
Point cloud: Frames in sliding window with ZERO points: (if any)
Viewer update: Valid points per frame (first 5 frames):
  Frame[0]: M valid points
  Frame[1]: M valid points
  ...
```

If later frames show 0 valid points, check why their patches have invalid coordinates.

