# Correlation Function Verification: C++ vs Python

## Summary
✅ **The C++ correlation function matches Python's implementation.**

## Python Implementation (`dpvo.py` lines 419-465)

```python
def corr(self, coords, indicies=None):
    if indicies is not None:
        ii, jj = indicies
    else:
        num_active = self.pg.num_edges
        ii = self.pg.kk[:num_active]  # Patch indices (from kk!)
        jj = self.pg.jj[:num_active]  # Target frame indices
    
    # Map to Feature Memory Indices 
    ii1 = ii % (self.M * self.pmem)  # Selects patches from gmap
    jj1 = jj % (self.mem)             # Selects frames from pyramid
    
    corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
    corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
    
    return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)
```

**Key Points:**
1. Uses `kk` (not `ii`) for patch indices
2. Maps `ii1 = kk % (M * pmem)` and `jj1 = jj % mem`
3. Divides coords by 1 for pyramid[0] and by 4 for pyramid[1]
4. Stacks results along last dimension

## C++ Implementation (`correlation_kernel.cpp`)

### Main Function (`computeCorrelation`)

```cpp
void computeCorrelation(
    const float* gmap,           // Patch features ring buffer
    const float* pyramid0,      // Full-res feature pyramid
    const float* pyramid1,      // 1/4-res feature pyramid
    const float* coords,        // Reprojected coordinates
    const int* ii,              // NOT USED (kept for compatibility)
    const int* jj,              // Target frame indices
    const int* kk,              // Linear patch indices (USED)
    ...
)
{
    // Map indices (matches Python)
    for (int e = 0; e < num_active; e++) {
        ii1[e] = kk[e] % mod_value;  // mod_value = M * num_gmap_frames
        jj1[e] = jj[e] % num_frames;
    }
    
    // Call for pyramid0: coord_scale = 1.0 (coords / 1)
    computeCorrelationSingle(..., 1.0f, ...);
    
    // Call for pyramid1: coord_scale = 0.25 (coords / 4)
    computeCorrelationSingle(..., 0.25f, ...);
    
    // Stack results (matches Python: torch.stack([corr1, corr2], -1))
    ...
}
```

**Status:** ✅ **MATCHES** - Index mapping and coordinate scaling are correct.

### Single Pyramid Level (`computeCorrelationSingle`)

**Coordinate Access:**
```cpp
// C++: [num_active, 2, P, P] layout
int coord_x_idx = e * 2 * P * P + 0 * P * P + i0 * P + j0;
int coord_y_idx = e * 2 * P * P + 1 * P * P + i0 * P + j0;
float raw_x = coords[coord_x_idx];
float raw_y = coords[coord_y_idx];
float x = raw_x * coord_scale;  // Scale: 1.0 for pyramid0, 0.25 for pyramid1
float y = raw_y * coord_scale;
```

**CUDA Kernel Equivalent:**
```cpp
// CUDA: coords[n][m][0][i0][j0] where n=0 (batch), m=edge
const float x = coords[n][m][0][i0][j0];  // Already scaled by Python
const float y = coords[n][m][1][i0][j0];
```

**Status:** ✅ **EQUIVALENT** - C++ scales internally, Python scales before CUDA call. Same result.

**Sampling Location:**
```cpp
// C++
int i1 = static_cast<int>(std::floor(y)) + (corr_ii - R);
int j1 = static_cast<int>(std::floor(x)) + (corr_jj - R);
```

```cpp
// CUDA
const int i1 = static_cast<int>(floor(y)) + (ii - R);
const int j1 = static_cast<int>(floor(x)) + (jj - R);
```

**Status:** ✅ **MATCHES** - Identical computation.

**Correlation Computation:**
```cpp
// C++
float sum = 0.0f;
for (int f = 0; f < feature_dim; f++) {
    float f1 = gmap[gmap_idx];      // Patch feature
    float f2 = pyramid[pyramid_idx]; // Frame feature
    sum += f1 * f2;                  // Dot product
}
```

```cpp
// CUDA
scalar_t s = 0;
for (int i=0; i<C; i+=8) {
    scalar_t f1[8]; // Load 8 features at once
    scalar_t f2[8];
    for (int j=0; j<8; j++) s += f1[j] * f2[j];
}
```

**Status:** ✅ **EQUIVALENT** - CUDA unrolls for performance, but computes same dot product.

## Coordinate Layout Verification

**Python Input:** `coords` shape `[B, M, 2, H, W]` = `[1, num_active, 2, P, P]`
- Access: `coords[0][m][0][i0][j0]` for x, `coords[0][m][1][i0][j0]` for y

**C++ Input:** `coords` layout `[num_active, 2, P, P]` (flattened)
- Access: `coords[m * 2 * P * P + 0 * P * P + i0 * P + j0]` for x
- Access: `coords[m * 2 * P * P + 1 * P * P + i0 * P + j0]` for y

**Status:** ✅ **EQUIVALENT** - Same data, different memory layout.

## Index Mapping Verification

**Python:**
```python
ii = self.pg.kk[:num_active]  # Use kk, not ii!
ii1 = ii % (self.M * self.pmem)
jj1 = jj % (self.mem)
```

**C++:**
```cpp
ii1[e] = kk[e] % (M * num_gmap_frames);  // Uses kk directly
jj1[e] = jj[e] % num_frames;
```

**Status:** ✅ **MATCHES** - Both use `kk` for patch indices and map correctly.

## Output Layout Verification

**Python:**
```python
corr1 = altcorr.corr(...)  # Shape: [B, M, D, D, H, W] = [1, num_active, 8, 8, 3, 3]
corr2 = altcorr.corr(...)  # Shape: [1, num_active, 8, 8, 3, 3]
return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)
# Final shape: [1, num_active, D*D*H*W*2] = [1, num_active, 8*8*3*3*2] = [1, num_active, 1152]
```

**C++:**
```cpp
// Output: [num_active, D, D, P, P, 2] (channel last)
// Layout: [e, corr_ii, corr_jj, i0, j0, c]
// Channel 0: corr1, Channel 1: corr2
```

**Status:** ✅ **EQUIVALENT** - C++ uses channel-last layout, Python stacks along last dim. Same data.

## Conclusion

The C++ correlation function is **mathematically and logically equivalent** to Python's implementation:

1. ✅ Uses `kk` (not `ii`) for patch indices
2. ✅ Maps indices correctly: `ii1 = kk % (M * pmem)`, `jj1 = jj % mem`
3. ✅ Scales coordinates correctly: `1.0` for pyramid0, `0.25` for pyramid1
4. ✅ Computes sampling locations identically: `floor(y) + (ii - R)`
5. ✅ Computes correlation as dot product over features
6. ✅ Stacks results correctly (channel-last in C++, last dim in Python)

**If poses are still incorrect, the issue is likely in:**
- Reprojection function (coordinates being computed incorrectly)
- Bundle Adjustment (optimization not converging)
- Intrinsics scaling (as you've already identified)
- Patchify function (patches extracted incorrectly)

The correlation function itself appears to be correct.

