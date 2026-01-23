# Python vs C++ Mathematical Equivalence Check

## Summary
After comparing `dpvo.py` with `dpvo.cpp`, I found **one critical mathematical difference** that was fixed:

### ✅ FIXED: Correlation Coordinate Scaling
- **Python**: `corr1 = altcorr.corr(..., coords / 1, ...)` for pyramid[0], `corr2 = altcorr.corr(..., coords / 4, ...)` for pyramid[1]
- **C++ (before)**: Used `scale = 1.0f` for both channels
- **C++ (after)**: Uses `scale = 1.0f` for channel 0 (fmap1), `scale = 0.25f` for channel 1 (fmap2)
- **Status**: ✅ FIXED

## Verified Equivalences

### 1. Context Slicing (`ctx = self.imap[:, active_kk % (self.M * self.pmem)]`)
- **Python**: `ctx = self.imap[:, active_kk % (self.M * self.pmem)]`
  - `active_kk` is linear patch index: `frame * M + patch`
  - `active_kk % (M * pmem)` gives linear index in ring buffer
  
- **C++**: 
  ```cpp
  int frame = kk_val / M;
  int patch = kk_val % M;
  int imap_frame = frame % m_pmem;
  int imap_offset = imap_idx(imap_frame, patch, 0);
  ```
  - This computes: `((kk_val / M) % pmem) * M + (kk_val % M)`
  - Mathematically equivalent to: `(frame * M + patch) % (M * pmem)` when `patch < M`
  - **Status**: ✅ EQUIVALENT

### 2. Target Computation
- **Python**: `target = coords[...,self.P//2,self.P//2] + delta.float()`
  - Gets center pixel coordinates and adds delta
  
- **C++**:
  ```cpp
  int center_i0 = P / 2;
  int center_j0 = P / 2;
  float cx = coords[e * 2 * P * P + 0 * P * P + center_i0 * P + center_j0];
  float cy = coords[e * 2 * P * P + 1 * P * P + center_i0 * P + center_j0];
  m_pg.m_target[e * 2 + 0] = cx + dx;
  m_pg.m_target[e * 2 + 1] = cy + dy;
  ```
  - **Status**: ✅ EQUIVALENT

### 3. Network Update (`active_net` reuse)
- **Python**: 
  ```python
  active_net = self.pg.net[:, :num_active]  # Reuses existing values
  active_net, (delta, weight, _) = self.network.update(...)
  self.pg.net[:, :num_active] = active_net  # Writes back
  ```
  
- **C++**: 
  - `reshapeInput` copies from `m_pg.m_net` to `net_input` (reuses existing values)
  - Model updates `net_input`
  - `net_out` is written back to `m_pg.m_net`
  - **Status**: ✅ EQUIVALENT (after fix to not zero-fill active edges)

### 4. Reprojection
- **Python**: `coords = pops.transform(...).permute(0, 1, 4, 2, 3).contiguous()`
  - Uses scaled intrinsics (divided by RES=4)
  - Output coordinates are at 1/4 resolution
  
- **C++**: `reproject()` uses `m_pg.m_intrinsics` (scaled by RES=4)
  - Output coordinates are at 1/4 resolution
  - **Status**: ✅ EQUIVALENT

### 5. Correlation Indexing
- **Python**: 
  ```python
  ii1 = ii % (self.M * self.pmem)  # Patch index in gmap
  jj1 = jj % (self.mem)  # Frame index in pyramid
  ```
  
- **C++**:
  ```cpp
  int ii1 = kk[e] % mod_value;  // mod_value = M * num_gmap_frames
  int gmap_frame = ii1 / M;
  int patch_idx = ii1 % M;
  int jj1 = jj[e] % num_frames;
  ```
  - **Status**: ✅ EQUIVALENT

## Remaining Differences (Non-Mathematical)

1. **Error Handling**: C++ has more extensive NaN/Inf checks and validation
2. **Logging**: C++ has more detailed logging for debugging
3. **Memory Layout**: C++ uses flat arrays vs Python's tensor views (mathematically equivalent)

## Conclusion
The C++ implementation is now **mathematically equivalent** to the Python code after fixing the correlation coordinate scaling issue.





