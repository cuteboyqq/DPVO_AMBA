# Viewer Diagnostic: Checking if Visualization is Correct

## Hypothesis

Since `patchify`, `reproject`, `correlation`, and `BA` all match Python, the issue might be in:
1. **How poses are passed to the viewer** (`dpvo.cpp` → `dpvo_viewer.cpp`)
2. **How poses are converted to matrices** (`convertPosesToMatrices`)
3. **How poses are extracted for drawing** (`drawPoses`)

## Current Flow

1. **Pose Storage** (`dpvo.cpp`):
   - `m_pg.m_poses[i]` stores SE3 poses (world-to-camera, T_wc)
   - `m_allPoses[m_counter]` stores historical poses
   - After BA, poses are synced: `m_allPoses[global_idx] = m_pg.m_poses[sw_idx]`

2. **Pose Passing** (`dpvo.cpp` line 2981):
   ```cpp
   m_viewer->updatePoses(m_allPoses.data(), num_historical_frames);
   ```

3. **Pose Conversion** (`dpvo_viewer.cpp` line 205-340):
   - Receives `T_wc` (world-to-camera)
   - Converts to `T_cw = T_wc.inverse()` (camera-to-world)
   - Stores in column-major format: `mat[col*4 + row] = T_cw(row, col)`

4. **Pose Extraction** (`dpvo_viewer.cpp` line 470-472):
   - Extracts camera position: `cam_x = mat[3]`, `cam_y = mat[7]`, `cam_z = mat[11]`
   - This is correct for column-major: `mat[3] = T_cw(0,3) = t_cw.x()`

## Potential Issues

### Issue 1: Matrix Storage Format

**Current Code** (`dpvo_viewer.cpp` line 336-339):
```cpp
mat[0]  = T_cw(0,0); mat[4]  = T_cw(1,0); mat[8]  = T_cw(2,0); mat[12] = T_cw(3,0);
mat[1]  = T_cw(0,1); mat[5]  = T_cw(1,1); mat[9]  = T_cw(2,1); mat[13] = T_cw(3,1);
mat[2]  = T_cw(0,2); mat[6]  = T_cw(1,2); mat[10] = T_cw(2,2); mat[14] = T_cw(3,2);
mat[3]  = T_cw(0,3); mat[7]  = T_cw(1,3); mat[11] = T_cw(2,3); mat[15] = T_cw(3,3);
```

**Verification**: 
- Column-major: `mat[col*4 + row] = T_cw(row, col)`
- `mat[0] = T_cw(0,0)` ✓ (col=0, row=0)
- `mat[4] = T_cw(1,0)` ✓ (col=0, row=1)
- `mat[3] = T_cw(0,3)` ✓ (col=3, row=0) = translation x
- `mat[7] = T_cw(1,3)` ✓ (col=3, row=1) = translation y
- `mat[11] = T_cw(2,3)` ✓ (col=3, row=2) = translation z

**Status**: ✅ **CORRECT** - Matrix storage matches column-major convention.

### Issue 2: Coordinate System Convention

**Current Code** (`dpvo_viewer.cpp` line 209):
```cpp
// SE3 poses are world-to-camera (T_wc), we need camera-to-world (T_cw) for visualization
SE3 T_cw_se3 = T_wc.inverse();
```

**Verification**:
- If `T_wc` transforms world points to camera frame: `p_camera = T_wc * p_world`
- Then `T_cw = T_wc^-1` transforms camera points to world: `p_world = T_cw * p_camera`
- Camera position in world frame = `T_cw.t` = `-R_cw * t_wc`

**Status**: ✅ **CORRECT** - Inverse transformation is correct.

### Issue 3: Pose Extraction

**Current Code** (`dpvo_viewer.cpp` line 470-472):
```cpp
float cam_x = mat[3];   // T_cw(0,3) = t_cw.x()
float cam_y = mat[7];   // T_cw(1,3) = t_cw.y()
float cam_z = mat[11];  // T_cw(2,3) = t_cw.z()
```

**Verification**:
- In column-major format, translation is in column 3 (indices 3, 7, 11)
- `mat[3] = T_cw(0,3)` = `t_cw.x()` ✓
- `mat[7] = T_cw(1,3)` = `t_cw.y()` ✓
- `mat[11] = T_cw(2,3)` = `t_cw.z()` ✓

**Status**: ✅ **CORRECT** - Translation extraction is correct.

## Diagnostic Steps

### Step 1: Add Logging to Verify Poses Being Passed

Add logging in `dpvo.cpp` before calling `updatePoses`:

```cpp
// In dpvo.cpp, before line 2981
if (logger) {
    logger->info("Viewer update: Passing {} poses to viewer", num_historical_frames);
    for (int i = 0; i < std::min(5, num_historical_frames); i++) {
        Eigen::Vector3f t_wc = m_allPoses[i].t;
        Eigen::Quaternionf q_wc = m_allPoses[i].q;
        logger->info("  Pose[{}]: T_wc.t=({:.3f}, {:.3f}, {:.3f}), q=({:.3f}, {:.3f}, {:.3f}, {:.3f})",
                     i, t_wc.x(), t_wc.y(), t_wc.z(), q_wc.x(), q_wc.y(), q_wc.z(), q_wc.w());
    }
}
```

### Step 2: Add Logging in Viewer to Verify Conversion

Add logging in `dpvo_viewer.cpp` in `convertPosesToMatrices`:

```cpp
// In dpvo_viewer.cpp, after line 339
if (i < 5 && logger) {
    logger->info("Viewer convertPosesToMatrices[{}]: T_wc.t=({:.3f}, {:.3f}, {:.3f}), "
                 "T_cw.t=({:.3f}, {:.3f}, {:.3f}), mat[3]={:.3f}, mat[7]={:.3f}, mat[11]={:.3f}",
                 i, t_wc.x(), t_wc.y(), t_wc.z(), t_cw.x(), t_cw.y(), t_cw.z(),
                 mat[3], mat[7], mat[11]);
}
```

### Step 3: Add Logging in drawPoses to Verify Extraction

Add logging in `dpvo_viewer.cpp` in `drawPoses`:

```cpp
// In dpvo_viewer.cpp, after line 472
if (i < 5 && logger) {
    logger->info("Viewer drawPoses[{}]: cam_x={:.3f}, cam_y={:.3f}, cam_z={:.3f}, "
                 "mat[3]={:.3f}, mat[7]={:.3f}, mat[11]={:.3f}",
                 i, cam_x, cam_y, cam_z, mat[3], mat[7], mat[11]);
}
```

## Expected Results

If visualization is correct:
1. **Poses passed**: Should show `T_wc.t` values from BA
2. **Poses converted**: Should show `T_cw.t = -R_cw * t_wc` (inverse transformation)
3. **Poses extracted**: Should match `T_cw.t` values

If there's a mismatch:
- **Poses passed ≠ Poses converted**: Issue in `convertPosesToMatrices`
- **Poses converted ≠ Poses extracted**: Issue in matrix storage/extraction
- **All match but visualization wrong**: Issue in drawing code (pyramid orientation, etc.)

## Conclusion

The viewer code appears mathematically correct. The issue is likely:
1. **Poses themselves are wrong** (but BA/reproject/correlation match Python, so this is unlikely)
2. **Pose synchronization** (`m_allPoses` not updated correctly after BA)
3. **Coordinate system mismatch** (e.g., different conventions between Python and C++)

**Recommendation**: Add the diagnostic logging above to verify where the mismatch occurs.

