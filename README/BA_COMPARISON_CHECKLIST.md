# C++ BA Function vs Python BA Function - Comparison Checklist

## Critical Areas to Verify

### 1. Residual Computation Order
**Python Expected:**
```python
r = targets - coords[..., p//2, p//2, :]  # Compute residual FIRST
v *= (r.norm(dim=-1) < 250).float()       # Then check residual norm
v *= in_bounds.float()                     # Then check bounds
r = (v[..., None] * r).unsqueeze(dim=-1)  # Finally apply validity mask
```

**C++ Current (lines 78-177):**
- ✅ Computes residual: `r[e * 2 + 0] = target_x - cx` (line 102-103)
- ✅ Checks residual norm: `if (r_norm >= 250.0f)` (line 120)
- ✅ Checks bounds: `if (cx < 0.0f || cy < 0.0f || cx >= m_fmap1_W || cy >= m_fmap1_H)` (line 160)
- ✅ Applies validity mask: `r[e * 2 + 0] *= v[e]` (line 174)
- ⚠️ **ISSUE**: Validity checks happen in TWO separate loops (lines 78-126 and 151-169)
- ⚠️ **ISSUE**: Python does ALL validity checks BEFORE applying mask to residuals

**Status**: ⚠️ **NEEDS REVIEW** - Order might be correct but structure differs

---

### 2. Hessian Block Computation - Patch Aggregation

**Python Expected:**
```python
# Python aggregates over ALL P*P pixels in the patch
Bii = torch.matmul(wJiT, Ji)  # Ji is [num_active, 2, P, P, 6]
Bij = torch.matmul(wJiT, Jj)  # Aggregates over P*P pixels
Bji = torch.matmul(wJjT, Ji)
Bjj = torch.matmul(wJjT, Jj)
Eik = torch.matmul(wJiT, Jz)  # Jz is [num_active, 2, P, P, 1]
Ejk = torch.matmul(wJjT, Jz)
```

**C++ Current (lines 179-273):**
- ⚠️ **CRITICAL**: Extracts Jacobians at patch CENTER only (lines 184-201)
- ⚠️ **CRITICAL**: Computes Hessian blocks using center Jacobians only (lines 254-273)
- ❌ **MISMATCH**: Python aggregates over ALL P*P pixels, C++ uses only center!

**Expected Fix:**
```cpp
// Should aggregate over ALL P*P pixels, not just center
for (int e = 0; e < num_active; e++) {
    Eigen::Matrix<float, 6, 6> Bii_sum = Eigen::Matrix<float, 6, 6>::Zero();
    Eigen::Matrix<float, 6, 6> Bij_sum = Eigen::Matrix<float, 6, 6>::Zero();
    // ... etc
    
    for (int py = 0; py < P; py++) {
        for (int px = 0; px < P; px++) {
            int pixel_idx = py * P + px;
            float pixel_valid = valid[e * P * P + pixel_idx];
            float patch_weight = weights_masked[e];
            float w_pixel = patch_weight * pixel_valid;
            
            // Extract Jacobians for this pixel
            Eigen::Matrix<float, 2, 6> Ji_pixel, Jj_pixel;
            Eigen::Matrix<float, 2, 1> Jz_pixel;
            // ... extract from Ji, Jj, Jz arrays
            
            // Accumulate Hessian blocks
            Bii_sum += w_pixel * Ji_pixel.transpose() * Ji_pixel;
            // ... etc
        }
    }
    Bii[e] = Bii_sum;
    // ... etc
}
```

**Status**: ❌ **CRITICAL MISMATCH** - C++ uses center only, Python aggregates all pixels

---

### 3. Gradient Computation - Patch Aggregation

**Python Expected:**
```python
vi = torch.matmul(wJiT, r)  # r is [num_active, 2, 1], aggregated over P*P
vj = torch.matmul(wJjT, r)
w = torch.matmul(wJzT, r)   # Scalar per edge
```

**C++ Current (lines 282-294):**
- ⚠️ Uses center Jacobians only (from Ji_center, Jj_center, Jz_center)
- ❌ **MISMATCH**: Should aggregate over ALL P*P pixels

**Status**: ❌ **CRITICAL MISMATCH** - Same issue as Hessian blocks

---

### 4. Weight Application

**Python Expected:**
```python
weights = (v[..., None] * weights).unsqueeze(dim=-1)  # Apply validity mask to weights
wJiT = (weights * Ji).transpose(2, 3)  # Weighted Jacobians
```

**C++ Current (lines 172-236):**
- ✅ Creates `weights_masked[e] = m_pg.m_weight[e] * v[e]` (line 176)
- ✅ Uses `weights_masked[e]` for weighted Jacobians (line 216)
- ⚠️ But only applies to center pixel, not all P*P pixels

**Status**: ⚠️ **PARTIAL MATCH** - Weight masking correct, but should apply per-pixel

---

### 5. Validity Mask Application

**Python Expected:**
```python
v *= (r.norm(dim=-1) < 250).float()  # Reject large residuals
v *= in_bounds.float()                # Reject out-of-bounds
r = (v[..., None] * r).unsqueeze(dim=-1)  # Apply to residuals
weights = (v[..., None] * weights).unsqueeze(dim=-1)  # Apply to weights
```

**C++ Current:**
- ✅ Checks residual norm: `if (r_norm >= 250.0f) v[e] = 0.0f` (line 121)
- ✅ Checks bounds: `if (cx < 0.0f || ...) v[e] = 0.0f` (line 161)
- ✅ Applies to residuals: `r[e * 2 + 0] *= v[e]` (line 174)
- ✅ Applies to weights: `weights_masked[e] = m_pg.m_weight[e] * v[e]` (line 176)
- ⚠️ Uses edge-level validity, Python might use pixel-level

**Status**: ✅ **MATCHES** - Logic is correct, but might need pixel-level validity

---

### 6. Damping Formula

**Python Expected:**
```python
A = A + (ep + lm * A) * torch.eye(...)  # ep=100.0, lm=1e-4
```

**C++ Current (lines 536-541):**
```cpp
Eigen::VectorXf S_diag = S.diagonal();
Eigen::MatrixXf S_damped = S;
float lm = 1e-4f;
for (int i = 0; i < 6 * n_adjusted; i++) {
    S_damped(i, i) += ep + lm * S_diag[i];
}
```

**Status**: ✅ **MATCHES** - Formula is correct: `ep + lm * diag(S)`

---

### 7. Parameter Order for retr()

**Python Expected:**
```python
# retr() expects [rx, ry, rz, tx, ty, tz] order
```

**C++ Current (lines 581-594):**
- Uses `dX.segment<6>(6 * idx)` directly
- Need to verify `retr()` expects correct order

**Status**: ⚠️ **NEEDS VERIFICATION** - Check SE3::retr() parameter order

---

### 8. Structure Hessian C Computation

**Python Expected:**
```python
C = sum over edges: wJzT @ Jz  # Scalar per edge, aggregated over P*P pixels
```

**C++ Current (lines 397-411):**
- ⚠️ Uses center Jacobian only: `Jz_center[e * 2 * 1 + ...]`
- ❌ **MISMATCH**: Should aggregate over ALL P*P pixels

**Status**: ❌ **CRITICAL MISMATCH** - Same aggregation issue

---

## Summary of Critical Issues

### ❌ CRITICAL ISSUE #1: Patch Aggregation Missing
**Problem**: C++ BA only uses patch CENTER pixel, Python aggregates over ALL P*P pixels
**Impact**: Hessian blocks, gradients, and structure Hessian are incorrect
**Fix Required**: Loop over all P*P pixels and accumulate contributions

### ⚠️ ISSUE #2: Validity Check Order
**Problem**: Validity checks split across two loops
**Impact**: Might be correct but harder to verify
**Fix**: Consolidate into single loop matching Python order

### ⚠️ ISSUE #3: Pixel-level vs Edge-level Validity
**Problem**: C++ uses edge-level validity `v[e]`, Python might use pixel-level
**Impact**: Need to verify if Python applies validity per-pixel or per-edge
**Fix**: Check Python code to confirm

---

## Recommended Fix Priority

1. **HIGH**: Fix patch aggregation (Issue #1) - This is mathematically incorrect
2. **MEDIUM**: Verify validity check order matches Python exactly
3. **LOW**: Verify pixel-level vs edge-level validity handling

---

## Next Steps

1. **Get Python BA function** to compare line-by-line
2. **Fix patch aggregation** to loop over all P*P pixels
3. **Test with known good poses** to verify correctness
4. **Add unit tests** comparing C++ and Python outputs

