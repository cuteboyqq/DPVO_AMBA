# Correlation Function Comparison: C++ vs Python

## Overview

This document compares the C++ `computeCorrelationSingle` function with Python's `corr_forward_torch` function to verify mathematical equivalence.

## Python Implementation (`corr_forward_torch`)

**Location**: `dpvo/altcorr/correlation_kernel.py` lines 328-385

**Function Signature**:
```python
def corr_forward_torch(R, fmap1, fmap2, coords, us, vs):
    # R: correlation radius (typically 3)
    # fmap1: (B, ?, C, H, W) - Patch features from gmap
    # fmap2: (B, ?, C, H2, W2) - Frame features from pyramid
    # coords: (B, M, 2, H, W) - Reprojected coordinates
    # us: (M,) - Patch indices for fmap1
    # vs: (M,) - Frame indices for fmap2
```

**Key Algorithm**:
```python
for b in range(B):
    for m in range(M):
        ix = int(us[m].item())  # Patch index in gmap
        jx = int(vs[m].item())  # Frame index in pyramid
        
        for i0 in range(H):  # H = P (patch size, typically 3)
            for j0 in range(W):  # W = P (patch size, typically 3)
                x = coords[b, m, 0, i0, j0].item()
                y = coords[b, m, 1, i0, j0].item()
                
                base_i = math.floor(y)
                base_j = math.floor(x)
                
                for ii in range(D):  # D = 2*R + 2 = 8
                    for jj in range(D):
                        i1 = base_i + (ii - R)  # Sampling location in pyramid
                        j1 = base_j + (jj - R)
                        
                        if within_bounds(i1, j1, H2, W2):
                            # Extract patch feature: fmap1[b, ix, :, i0, j0]
                            f1 = fmap1[b, ix, :, i0, j0]  # All channels at patch pixel (i0, j0)
                            # Extract frame feature: fmap2[b, jx, :, i1, j1]
                            f2 = fmap2[b, jx, :, i1, j1]  # All channels at sampling location (i1, j1)
                            s = torch.dot(f1, f2)  # Dot product over channels
                        else:
                            s = 0.0
                        
                        corr[b, m, ii, jj, i0, j0] = s
```

**Python Call Site** (`dpvo.py` line 462-463):
```python
corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
```

**Input Shapes**:
- `self.gmap`: `[B, M*pmem, C, P, P]` = `[1, M*36, 128, 3, 3]` (patch features)
- `self.pyramid[0]`: `[B, mem, C, H1, W1]` = `[1, 36, 128, 132, 240]` (frame features at 1/4 res)
- `self.pyramid[1]`: `[B, mem, C, H2, W2]` = `[1, 36, 128, 33, 60]` (frame features at 1/16 res)
- `coords`: `[B, M, 2, P, P]` = `[1, num_active, 2, 3, 3]` (reprojected coordinates)
- `ii1`: `[M]` - Patch indices (mapped from `kk % (M * pmem)`)
- `jj1`: `[M]` - Frame indices (mapped from `jj % mem`)

## C++ Implementation (`computeCorrelationSingle`)

**Location**: `app/src/correlation_kernel.cpp` lines 286-657

**Function Signature**:
```cpp
void computeCorrelationSingle(
    const float* gmap,        // [num_gmap_frames, M, feature_dim, D_gmap, D_gmap]
    const float* pyramid,     // [num_frames, feature_dim, fmap_H, fmap_W]
    const float* coords,      // [num_active, 2, P, P]
    const int* ii1,           // [num_active] - Patch indices (mapped from kk)
    const int* jj1,           // [num_active] - Frame indices (mapped from jj)
    int num_active, M, P, num_frames, num_gmap_frames,
    int fmap_H, int fmap_W, feature_dim,
    float coord_scale,        // 1.0 for pyramid0, 0.25 for pyramid1
    int radius,
    float* corr_out)          // [num_active, D, D, P, P]
```

**Key Algorithm**:
```cpp
const int R = radius;  // 3
const int D = 2 * R + 2;  // 8
const int D_gmap = 3;  // Patch dimension (from patchify with radius=1)
const int gmap_center_offset = (D_gmap - P) / 2;  // 0 when D_gmap=P=3

for (int e = 0; e < num_active; e++) {
    int ii1_val = ii1[e];
    int jj1_val = jj1[e];
    
    // Extract gmap frame and patch index
    int gmap_frame = ii1_val / M;      // Which frame in gmap ring buffer
    int patch_idx = ii1_val % M;        // Which patch within that frame
    int pyramid_frame = jj1_val;        // Which frame in pyramid
    
    for (int i0 = 0; i0 < P; i0++) {
        for (int j0 = 0; j0 < P; j0++) {
            // Get reprojected coordinates
            float raw_x = coords[e * 2 * P * P + 0 * P * P + i0 * P + j0];
            float raw_y = coords[e * 2 * P * P + 1 * P * P + i0 * P + j0];
            
            // Scale coordinates (1.0 for pyramid0, 0.25 for pyramid1)
            float x = raw_x * coord_scale;
            float y = raw_y * coord_scale;
            
            for (int corr_ii = 0; corr_ii < D; corr_ii++) {
                for (int corr_jj = 0; corr_jj < D; corr_jj++) {
                    // Sampling location in target frame
                    int i1 = static_cast<int>(std::floor(y)) + (corr_ii - R);
                    int j1 = static_cast<int>(std::floor(x)) + (corr_jj - R);
                    
                    if (within_bounds(i1, j1, fmap_H, fmap_W)) {
                        // Extract patch feature from gmap
                        int gmap_i = i0 + gmap_center_offset;  // i0 when offset=0
                        int gmap_j = j0 + gmap_center_offset;  // j0 when offset=0
                        
                        float sum = 0.0f;
                        for (int f = 0; f < feature_dim; f++) {
                            // gmap indexing: [gmap_frame][patch_idx][f][gmap_i][gmap_j]
                            size_t fmap1_idx = gmap_frame * M * feature_dim * D_gmap * D_gmap +
                                               patch_idx * feature_dim * D_gmap * D_gmap +
                                               f * D_gmap * D_gmap +
                                               gmap_i * D_gmap + gmap_j;
                            
                            // pyramid indexing: [pyramid_frame][f][i1][j1]
                            size_t fmap2_idx = pyramid_frame * feature_dim * fmap_H * fmap_W +
                                               f * fmap_H * fmap_W +
                                               i1 * fmap_W + j1;
                            
                            sum += gmap[fmap1_idx] * pyramid[fmap2_idx];
                        }
                    } else {
                        sum = 0.0f;
                    }
                    
                    // Output: [e][corr_ii][corr_jj][i0][j0]
                    size_t out_idx = e * D * D * P * P +
                                     corr_ii * D * P * P +
                                     corr_jj * P * P +
                                     i0 * P + j0;
                    corr_out[out_idx] = sum;
                }
            }
        }
    }
}
```

## Comparison

### ✅ **MATCHING**: Index Mapping

**Python**:
```python
ii1 = ii % (M * pmem)  # ii comes from kk
jj1 = jj % mem
ix = int(us[m].item())  # = ii1[m]
jx = int(vs[m].item())  # = jj1[m]
```

**C++**:
```cpp
ii1[e] = kk[e] % (M * num_gmap_frames);  // Same as Python
jj1[e] = jj[e] % num_frames;              // Same as Python
int gmap_frame = ii1_val / M;             // Extracts frame from linear index
int patch_idx = ii1_val % M;               // Extracts patch from linear index
int pyramid_frame = jj1_val;               // Direct frame index
```

**Status**: ✅ **MATCHES** - Both map `kk` to `ii1` using modulo, then extract frame/patch indices.

### ✅ **MATCHING**: Coordinate Extraction and Scaling

**Python**:
```python
x = coords[b, m, 0, i0, j0].item()  # For pyramid0: coords / 1
y = coords[b, m, 1, i0, j0].item()
# For pyramid1: coords / 4 (done before calling corr)
```

**C++**:
```cpp
float raw_x = coords[e * 2 * P * P + 0 * P * P + i0 * P + j0];
float raw_y = coords[e * 2 * P * P + 1 * P * P + i0 * P + j0];
float x = raw_x * coord_scale;  // 1.0 for pyramid0, 0.25 for pyramid1
float y = raw_y * coord_scale;
```

**Status**: ✅ **MATCHES** - Both extract coordinates at patch pixel `(i0, j0)` and scale them appropriately.

### ✅ **MATCHING**: Sampling Location Computation

**Python**:
```python
base_i = math.floor(y)
base_j = math.floor(x)
i1 = base_i + (ii - R)  # ii ranges [0, D-1], so (ii - R) ranges [-R, D-1-R] = [-3, 4]
j1 = base_j + (jj - R)
```

**C++**:
```cpp
int i1 = static_cast<int>(std::floor(y)) + (corr_ii - R);  // corr_ii ranges [0, D-1]
int j1 = static_cast<int>(std::floor(x)) + (corr_jj - R);
```

**Status**: ✅ **MATCHES** - Both compute sampling locations identically.

### ✅ **MATCHING**: Patch Feature Extraction

**Python**:
```python
f1 = fmap1[b, ix, :, i0, j0]  # All channels at patch pixel (i0, j0)
# Where fmap1 = self.gmap with shape [B, M*pmem, C, P, P]
# ix is the linear patch index (can span multiple frames)
```

**C++**:
```cpp
int gmap_i = i0 + gmap_center_offset;  // = i0 when D_gmap=P=3 (offset=0)
int gmap_j = j0 + gmap_center_offset;  // = j0 when D_gmap=P=3 (offset=0)
// gmap[gmap_frame][patch_idx][f][gmap_i][gmap_j]
```

**Status**: ✅ **MATCHES** - Both extract patch features at pixel `(i0, j0)` from the correct patch.
- Python: `fmap1[b, ix, :, i0, j0]` where `ix` is linear index spanning frames
- C++: `gmap[gmap_frame][patch_idx][f][i0][j0]` where `gmap_frame = ix / M`, `patch_idx = ix % M`
- When `D_gmap = P = 3`, `gmap_center_offset = 0`, so `gmap_i = i0` and `gmap_j = j0`

### ✅ **MATCHING**: Frame Feature Extraction

**Python**:
```python
f2 = fmap2[b, jx, :, i1, j1]  # All channels at sampling location (i1, j1)
# Where fmap2 = self.pyramid[0] or self.pyramid[1]
```

**C++**:
```cpp
// pyramid[pyramid_frame][f][i1][j1]
size_t fmap2_idx = pyramid_frame * feature_dim * fmap_H * fmap_W +
                   f * fmap_H * fmap_W +
                   i1 * fmap_W + j1;
```

**Status**: ✅ **MATCHES** - Both extract frame features at sampling location `(i1, j1)` from the correct frame.

### ✅ **MATCHING**: Dot Product Computation

**Python**:
```python
s = torch.dot(f1, f2)  # Dot product over all channels
```

**C++**:
```cpp
float sum = 0.0f;
for (int f = 0; f < feature_dim; f++) {
    sum += gmap[fmap1_idx] * pyramid[fmap2_idx];
}
```

**Status**: ✅ **MATCHES** - Both compute dot product over all feature channels.

### ✅ **MATCHING**: Output Layout

**Python**:
```python
corr[b, m, ii, jj, i0, j0] = s
# Output shape: [B, M, D, D, H, W] = [1, num_active, 8, 8, 3, 3]
```

**C++**:
```cpp
size_t out_idx = e * D * D * P * P +
                 corr_ii * D * P * P +
                 corr_jj * P * P +
                 i0 * P + j0;
corr_out[out_idx] = sum;
// Output shape: [num_active, D, D, P, P] = [num_active, 8, 8, 3, 3]
```

**Status**: ✅ **MATCHES** - Both store correlation values in the same layout (after accounting for batch dimension in Python).

### ✅ **MATCHING**: Bounds Checking

**Python**:
```python
if within_bounds(i1, j1, H2, W2):
    s = torch.dot(f1, f2)
else:
    s = 0.0
```

**C++**:
```cpp
if (within_bounds(i1, j1, fmap_H, fmap_W)) {
    // Compute correlation
} else {
    sum = 0.0f;  // Already initialized to 0
}
```

**Status**: ✅ **MATCHES** - Both check bounds and set correlation to 0 for out-of-bounds sampling.

## Summary

✅ **The C++ correlation function matches the Python implementation mathematically.**

### Key Equivalences:

1. **Index Mapping**: Both map `kk` → `ii1` using modulo, then extract frame/patch indices
2. **Coordinate Extraction**: Both extract coordinates at patch pixel `(i0, j0)` and scale appropriately
3. **Sampling Location**: Both compute `i1 = floor(y) + (ii - R)`, `j1 = floor(x) + (jj - R)`
4. **Feature Extraction**: Both extract patch features from gmap and frame features from pyramid correctly
5. **Dot Product**: Both compute dot product over all feature channels
6. **Output Layout**: Both store results in the same layout `[edge, D, D, P, P]`

### Minor Differences (Non-functional):

1. **Batch Dimension**: Python has explicit batch dimension `B=1`, C++ doesn't (equivalent to `B=1`)
2. **Index Extraction**: Python uses `int(us[m].item())`, C++ uses `ii1[e]` directly (same value)
3. **gmap_center_offset**: C++ uses `gmap_center_offset` for generality, but it's 0 when `D_gmap=P=3` (matches Python)

## Conclusion

The C++ correlation function is **mathematically equivalent** to the Python implementation. The indexing, coordinate scaling, sampling, and dot product computations all match Python's behavior.

