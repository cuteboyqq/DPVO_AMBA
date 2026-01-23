# Fixing Visualization Issues

## Current Problems

1. **Poses not updated** - Camera trajectory not showing correctly
2. **Points are wrong** - 3D point cloud is incorrect
3. **Broken images** - Video frame display is corrupted

## Issues Identified

### 1. Point Cloud Computation

The current `computePointCloud()` function has issues:
- The transformation formula might be incorrect
- Need to check if patches are in the right coordinate system
- The depth/inverse depth handling might be wrong

### 2. Image Format

The image conversion from [C,H,W] to [H,W,C] might have issues:
- Need to verify the input image format
- Check if it's RGB or BGR
- Verify the conversion is correct

### 3. Poses Update Frequency

Poses might not be updating because:
- `updateViewer()` is only called after `update()` completes
- Need to ensure poses are updated more frequently
- Check if poses are actually changing

## Debugging Steps

Add logging to verify data:

```cpp
// In computePointCloud(), add logging:
auto logger = spdlog::get("dpvo");
if (logger && i == 0 && k == 0) {
    logger->info("Point cloud: frame={}, patch={}, px={}, py={}, pd={}, fx={}, fy={}", 
                 i, k, px, py, pd, fx, fy);
    logger->info("Point cloud: p0=({}, {}, {}), p_world=({}, {}, {})", 
                 p0.x(), p0.y(), p0.z(), p_world.x(), p_world.y(), p_world.z());
}

// In updateViewer(), add logging:
if (logger) {
    logger->info("Viewer update: n={}, num_points={}, pose[0].t=({}, {}, {})", 
                 m_pg.m_n, num_points, 
                 m_pg.m_poses[0].t.x(), m_pg.m_poses[0].t.y(), m_pg.m_poses[0].t.z());
}
```

## Potential Fixes

### Fix 1: Point Cloud Transformation

The transformation might need to be:
```cpp
// Instead of: p_world = T.R() * p0 + T.t * W0
// Try: p_world = T * (p0 / W0)  (if W0 is inverse depth)
Eigen::Vector3f p_camera = p0 / W0;  // Convert to camera coordinates
Eigen::Vector3f p_world = T.R() * p_camera + T.t;
```

### Fix 2: Image Format

Check if image needs BGR to RGB conversion:
```cpp
// If input is BGR, convert to RGB
for (int c = 0; c < 3; c++) {
    int src_c = (2 - c);  // BGR -> RGB: 0->2, 1->1, 2->0
    // ... rest of conversion
}
```

### Fix 3: Update Frequency

Ensure viewer updates after every frame:
```cpp
// In run(), update viewer after each frame, not just after optimization
if (m_visualizationEnabled) {
    updateViewer();
    // Update image
}
```

