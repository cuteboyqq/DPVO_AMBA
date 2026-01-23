# Python vs C++ DPVO Implementation Comparison

## Critical Differences Found

### 1. Correlation Function (`corr` vs `computeCorrelation`)

**Python (`dpvo.py` line 419-465):**
```python
def corr(self, coords, indicies=None):
    if indicies is not None:
        ii, jj = indicies
    else:
        num_active = self.pg.num_edges
        ii = self.pg.kk[:num_active]  # Patch indices (from kk!)
        jj = self.pg.jj[:num_active]   # Target frame indices
    
    # Map to Feature Memory Indices 
    ii1 = ii % (self.M * self.pmem)  # Selects patches from gmap
    jj1 = jj % (self.mem)             # Selects frames from pyramid
    
    corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
    corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
    
    return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)
```

**C++ (`correlation_kernel.cpp`):**
```cpp
computeCorrelation(
    m_gmap,
    m_fmap1,  // pyramid0
    m_fmap2,  // pyramid1
    coords.data(),
    m_pg.m_ii,  // NOT USED (kept for compatibility)
    m_pg.m_jj,  // Target frame indices
    m_pg.m_kk,  // Linear patch indices
    ...
)

// Inside computeCorrelation:
int ii1 = kk[e] % mod_value;  // mod_value = M * num_gmap_frames
int jj1 = jj[e] % num_frames;
```

**Status:** ✅ **MATCHES** - The index mapping is correct. Python uses `kk` as `ii`, then maps to `ii1`. C++ uses `kk` directly and maps to `ii1`.

### 2. Target Computation

**Python (`dpvo.py` line 806):**
```python
target = coords[...,self.P//2,self.P//2] + delta.float()
```

**C++ (`dpvo.cpp` line 1127-1128):**
```cpp
m_pg.m_target[e * 2 + 0] = cx + dx;
m_pg.m_target[e * 2 + 1] = cy + dy;
```

**Status:** ✅ **MATCHES** - Both extract center pixel from coords and add delta.

### 3. Bundle Adjustment Residual

**Python (`ba.py` line 159):**
```python
r = targets - coords[..., p//2, p//2, :] # reprojection residual
```

**C++ (`ba.cpp` line 91-92):**
```cpp
r[e * 2 + 0] = target_x - cx;
r[e * 2 + 1] = target_y - cy;
```

**Status:** ✅ **MATCHES** - Both compute `target - coords` at patch center.

### 4. BA Bounds Check

**Python (`ba.py` line 165-169):**
```python
in_bounds = \
    (coords[..., p//2, p//2, 0] > bounds[0]) & \
    (coords[..., p//2, p//2, 1] > bounds[1]) & \
    (coords[..., p//2, p//2, 0] < bounds[2]) & \
    (coords[..., p//2, p//2, 1] < bounds[3])
```

**C++ (`ba.cpp` line 149):**
```cpp
if (cx < 0.0f || cy < 0.0f || cx >= m_fmap1_W || cy >= m_fmap1_H) {
    v[e] = 0.0f;
}
```

**Status:** ✅ **MATCHES** - Both check bounds at patch center. C++ uses feature map dimensions (1/4 resolution), which is correct.

### 5. Correlation Output Layout

**Python:**
- `corr1`: Shape from `altcorr.corr` (CUDA kernel)
- `corr2`: Shape from `altcorr.corr` (CUDA kernel)
- `torch.stack([corr1, corr2], -1)`: Stacks along last dimension
- `.view(1, len(ii), -1)`: Flattens to `[1, num_active, D*D*P*P*2]`

**C++:**
- Output: `[num_active, D, D, P, P, 2]` (channel last)
- Reshaped in `reshapeInput` to `[1, 882, 360, 1]` where `882 = 2*7*7*3*3`

**Status:** ⚠️ **NEEDS VERIFICATION** - The layout should be equivalent, but need to verify the channel ordering matches Python's stacked format.

## Potential Issues

### Issue 1: Correlation is Zero
**Symptoms:** Logs show correlation output is mostly zero (504/50688 nonzero values), but each edge has all zeros.

**Possible Causes:**
1. **Coordinates out of bounds**: `within_bounds(i1, j1, fmap_H, fmap_W)` returns false
2. **Invalid gmap/pyramid indices**: `gmap_frame` or `pyramid_frame` out of range
3. **Zero features**: `gmap` or `pyramid` buffers contain zeros
4. **Index mapping error**: `ii1` or `jj1` calculation is wrong

**Diagnostic Added:**
- Logs for first edge showing `kk`, `ii1`, `gmap_frame`, `patch_idx`, `jj`, `pyramid_frame`
- Logs for coordinate values and bounds checks
- Logs for feature values (`f1`, `f2`) and correlation sums

### Issue 2: BA Gradients are Zero
**Symptoms:** Logs show `v_grad_norm=0.000000`, `w_grad_norm=0.000000`, meaning BA won't update poses.

**Possible Causes:**
1. **Residuals are zero**: `target == coords` (poses already optimal)
2. **Weights are zero**: All `m_pg.m_weight[e] == 0`
3. **Jacobians are zero**: Reprojection Jacobians are zero (poses/patches not changing)
4. **Validity mask**: All edges invalidated (`v[e] == 0`)

**Diagnostic Added:**
- Logs for residual stats, weight stats, validity counts
- Sample edge diagnostics showing residual norms and weights

## Recommendations

1. **Run with diagnostic logs** and check:
   - Are coordinates within bounds?
   - Are gmap/pyramid indices valid?
   - Are feature values non-zero?
   - Are correlations being computed correctly?

2. **Verify correlation output layout** matches Python's expected format

3. **Check if correlation channels are interleaved correctly** in the reshape function

4. **Verify BA bounds** use correct resolution (1/4 for feature maps)


