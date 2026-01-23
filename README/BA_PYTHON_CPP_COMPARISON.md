# Python BA vs C++ BA Function - Detailed Comparison

## Step-by-Step Comparison

### 1. Basic Setup
**Python (line 143):**
```python
n = max(ii.max().item(), jj.max().item()) + 1
```

**C++ (lines 39-44):**
```cpp
int n = 0;
for (int e = 0; e < num_active; e++) {
    int i = m_pg.m_kk[e] / M;  // source frame index
    int j = m_pg.m_jj[e];      // target frame index
    n = std::max(n, std::max(i, j) + 1);
}
```
✅ **MATCHES** - Correctly extracts frame indices from kk

---

### 2. Forward Projection + Jacobians
**Python (line 152-153):**
```python
coords, v, (Ji, Jj, Jz) = pops.transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)
```

**C++ (lines 55-63):**
```cpp
reproject(m_pg.m_ii, m_pg.m_jj, m_pg.m_kk, num_active, 
          coords.data(), Ji.data(), Jj.data(), Jz.data(), valid.data());
```
✅ **MATCHES** - Calls reproject function

---

### 3. Residual Computation
**Python (line 159):**
```python
r = targets - coords[..., p//2, p//2, :]  # Compute residual at patch center
```

**C++ (lines 102-103):**
```cpp
r[e * 2 + 0] = target_x - cx;
r[e * 2 + 1] = target_y - cy;
```
✅ **MATCHES** - Computes residual at patch center

---

### 4. Validity Checks
**Python (lines 162, 165-171):**
```python
v *= (r.norm(dim=-1) < 250).float()  # Reject large residuals
in_bounds = (coords[..., p//2, p//2, 0] > bounds[0]) & \
            (coords[..., p//2, p//2, 1] > bounds[1]) & \
            (coords[..., p//2, p//2, 0] < bounds[2]) & \
            (coords[..., p//2, p//2, 1] < bounds[3])
v *= in_bounds.float()
```

**C++ (lines 119-121, 174-179):**
```cpp
if (r_norm >= 250.0f || !std::isfinite(r_norm)) {
    v[e] = 0.0f;
}
// ...
bool in_bounds = (cx > 0.0f) && (cy > 0.0f) && 
                 (cx < static_cast<float>(m_fmap1_W - 1)) && 
                 (cy < static_cast<float>(m_fmap1_H - 1));
if (!in_bounds) {
    v[e] = 0.0f;
}
```
✅ **MATCHES** - Uses same thresholds and bounds logic

---

### 5. Apply Validity Mask
**Python (lines 177-178):**
```python
r = (v[..., None] * r).unsqueeze(dim=-1)
weights = (v[..., None] * weights).unsqueeze(dim=-1)
```

**C++ (lines 191-193):**
```cpp
r[e * 2 + 0] *= v[e];
r[e * 2 + 1] *= v[e];
weights_masked[e] = m_pg.m_weight[e] * v[e];
```
✅ **MATCHES** - Applies validity mask multiplicatively

---

### 6. Build Weighted Jacobians
**Python (lines 183-185):**
```python
wJiT = (weights * Ji).transpose(2, 3)  # [num_active, 6, 2] aggregated over P×P
wJjT = (weights * Jj).transpose(2, 3)
wJzT = (weights * Jz).transpose(2, 3)
```

**C++ (lines 301-304):**
```cpp
Eigen::Matrix<float, 6, 2> wJiT_pixel = w_pixel * Ji_pixel.transpose();
Eigen::Matrix<float, 6, 2> wJjT_pixel = w_pixel * Jj_pixel.transpose();
Eigen::Matrix<float, 1, 2> wJzT_pixel = w_pixel * Jz_pixel.transpose();
```
✅ **MATCHES** - Computes per-pixel, aggregates over P×P

---

### 7. Pose-Pose Hessian Blocks
**Python (lines 190-193):**
```python
Bii = torch.matmul(wJiT, Ji)  # Aggregates over P×P pixels
Bij = torch.matmul(wJiT, Jj)
Bji = torch.matmul(wJjT, Ji)
Bjj = torch.matmul(wJjT, Jj)
```

**C++ (lines 315-327):**
```cpp
Bii_contrib = wJiT_pixel * Ji_pixel;  // Per pixel
Bii[e] += Bii_contrib;  // Accumulate over P×P pixels
```
✅ **MATCHES** - Aggregates over all P×P pixels

---

### 8. Pose-Structure Hessian Blocks
**Python (lines 198-199):**
```python
Eik = torch.matmul(wJiT, Jz)  # Aggregates over P×P pixels
Ejk = torch.matmul(wJjT, Jz)
```

**C++ (lines 330-337):**
```cpp
Eik_contrib = wJiT_pixel * Jz_pixel;  // Per pixel
Eik[e] += Eik_contrib;  // Accumulate over P×P pixels
```
✅ **MATCHES** - Aggregates over all P×P pixels

---

### 9. Gradient Vectors
**Python (lines 204-205, 248):**
```python
vi = torch.matmul(wJiT, r)  # Aggregates over P×P pixels
vj = torch.matmul(wJjT, r)
w = safe_scatter_add_vec(torch.matmul(wJzT, r), kk, m)
```

**C++ (lines 339-347):**
```cpp
vi_contrib = wJiT_pixel * r_vec;  // Per pixel, uses patch center r
vi[e] += vi_contrib;  // Accumulate over P×P pixels
w_vec[e] += (wJzT_pixel * r_vec)(0, 0);  // Accumulate over P×P pixels
```
✅ **MATCHES** - Uses patch center residual for all pixels, aggregates over P×P

---

### 10. Fix First Pose (Gauge Freedom)
**Python (lines 210-215):**
```python
n = n - fixedp
ii = ii - fixedp
jj = jj - fixedp
```

**C++ (lines 367-370):**
```cpp
int i_source = m_pg.m_kk[e] / M;
ii_new[e] = i_source - fixedp;
jj_new[e] = m_pg.m_jj[e] - fixedp;
```
✅ **MATCHES** - Adjusts indices for fixed poses

---

### 11. Reindex Structure Variables
**Python (line 220):**
```python
kx, kk = torch.unique(kk, return_inverse=True, sorted=True)
```

**C++ (lines 388-404):**
```cpp
std::set<int> kk_set;
for (int e = 0; e < num_active; e++) {
    kk_set.insert(m_pg.m_kk[e]);
}
kx.assign(kk_set.begin(), kk_set.end());
std::sort(kx.begin(), kx.end());
// Create mapping and reindex
```
✅ **MATCHES** - Extracts unique kk values and creates mapping

---

### 12. Assemble Global Hessian B
**Python (lines 226-229):**
```python
B = safe_scatter_add_mat(Bii, ii, ii, n, n).view(b, n, n, 6, 6) + \
    safe_scatter_add_mat(Bij, ii, jj, n, n).view(b, n, n, 6, 6) + \
    safe_scatter_add_mat(Bji, jj, ii, n, n).view(b, n, n, 6, 6) + \
    safe_scatter_add_mat(Bjj, jj, jj, n, n).view(b, n, n, 6, 6)
```

**C++ (lines 424-427):**
```cpp
B.block<6, 6>(6 * i, 6 * i) += Bii[e];
B.block<6, 6>(6 * i, 6 * j) += Bij[e];
B.block<6, 6>(6 * j, 6 * i) += Bji[e];
B.block<6, 6>(6 * j, 6 * j) += Bjj[e];
```
✅ **MATCHES** - Scatter-adds blocks to global Hessian

---

### 13. Assemble Pose-Structure Coupling E
**Python (lines 234-235):**
```python
E = safe_scatter_add_mat(Eik, ii, kk, n, m).view(b, n, m, 6, 1) + \
    safe_scatter_add_mat(Ejk, jj, kk, n, m).view(b, n, m, 6, 1)
```

**C++ (lines 446-447):**
```cpp
E.block<6, 1>(6 * i, k) += Eik[e];
E.block<6, 1>(6 * j, k) += Ejk[e];
```
✅ **MATCHES** - Scatter-adds coupling blocks

---

### 14. Structure Hessian C
**Python (line 240):**
```python
C = safe_scatter_add_vec(torch.matmul(wJzT, Jz), kk, m)
```

**C++ (lines 511-519):**
```cpp
float c_contrib = (wJzT_pixel * Jz_pixel)(0, 0);  // Per pixel
C_sum += c_contrib;  // Accumulate over P×P pixels
C[k] += C_sum;  // Scatter to structure variable
```
✅ **MATCHES** - Aggregates over P×P pixels, then scatters

---

### 15. Assemble Gradient Vectors
**Python (lines 245-248):**
```python
v = safe_scatter_add_vec(vi, ii, n).view(b, n, 1, 6, 1) + \
    safe_scatter_add_vec(vj, jj, n).view(b, n, 1, 6, 1)
w = safe_scatter_add_vec(torch.matmul(wJzT, r), kk, m)
```

**C++ (lines 545-553):**
```cpp
v_grad.segment<6>(6 * i) += vi[e];  // Scatter vi
v_grad.segment<6>(6 * j) += vj[e];  // Scatter vj
w_grad[k] += w_vec[e];  // Scatter w
```
✅ **MATCHES** - Scatters gradients to global vectors

---

### 16. Levenberg-Marquardt Damping
**Python (line 257):**
```python
Q = 1.0 / (C + lmbda)
```

**C++ (lines 612-613):**
```cpp
Eigen::VectorXf C_lm = C.array() + lmbda;
Eigen::VectorXf Q = 1.0f / C_lm.array();
```
✅ **MATCHES** - Computes inverse of (C + λ)

---

### 17. Schur Complement
**Python (lines 262, 271-274):**
```python
EQ = E * Q[:, None]  # E * C^-1
S = B - block_matmul(EQ, E.permute(0, 2, 1, 4, 3))
y = v - block_matmul(EQ, w.unsqueeze(dim=2))
```

**C++ (lines 616-620):**
```cpp
Eigen::MatrixXf EQ = E * Q.asDiagonal();  // E * C^-1
Eigen::MatrixXf S = B - EQ * E.transpose();  // B - E * C^-1 * E^T
Eigen::VectorXf y = v_grad - EQ * w_grad;  // v - E * C^-1 * w
```
✅ **MATCHES** - Equivalent matrix operations

---

### 18. Solve for Pose Increments
**Python (line 277):**
```python
dX = block_solve(S, y, ep=ep, lm=1e-4)
```

**C++ (lines 646-650, 664):**
```cpp
S_damped(i, i) += ep + lm * S_diag[i];  // Damping
Eigen::LDLT<Eigen::MatrixXf> solver(S_damped);
dX = solver.solve(y);
```
✅ **MATCHES** - Uses same damping formula, solves linear system

---

### 19. Back-Substitute Structure Increments
**Python (lines 280-282):**
```python
dZ = Q * (w - block_matmul(E.permute(0, 2, 1, 4, 3), dX).squeeze(dim=-1))
```

**C++ (line 672):**
```cpp
dZ = Q.asDiagonal() * (w_grad - E.transpose() * dX);
```
✅ **MATCHES** - Equivalent: E.permute(...) = E.transpose() for [6n, m] matrix

---

### 20. Apply Updates
**Python (lines 290-295):**
```python
disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
patches = torch.stack([x, y, disps], dim=2)
if not structure_only and n > 0:
    poses = pose_retr(poses, dX, fixedp + torch.arange(n))
```

**C++ (lines 711, 753-761):**
```cpp
m_pg.m_poses[pose_idx] = m_pg.m_poses[pose_idx].retr(dx_vec);
// ...
disp = std::max(1e-3f, std::min(10.0f, disp + dZ_val));
```
✅ **MATCHES** - Applies updates with same clamping

---

## Potential Issues Found

### ⚠️ Issue 1: Parameter Order for retr()
**Python:** `pose_retr` uses `poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))`
- The `dx` from solver is in the same order as Jacobians: `[tx, ty, tz, rx, ry, rz]`
- Python's `SE3.retr()` expects `[rx, ry, rz, tx, ty, tz]` (rotation first)

**C++:** Currently passes `dx_vec` directly without reordering
- Jacobians are `[tx, ty, tz, rx, ry, rz]` (translation first)
- `retr()` expects `[rx, ry, rz, tx, ty, tz]` (rotation first)

**Status:** ⚠️ **NEEDS VERIFICATION** - May need reordering, but user reported oscillation when reordered

---

### ✅ Issue 2: Bounds Check
**Fixed:** Changed from `cx >= m_fmap1_W` to `cx < m_fmap1_W - 1` to match Python's strict inequalities

---

## Summary

✅ **All major steps match Python logic:**
1. Residual computation at patch center ✓
2. Validity checks (residual norm, bounds) ✓
3. Validity mask application ✓
4. Hessian aggregation over P×P pixels ✓
5. Gradient aggregation over P×P pixels ✓
6. Structure Hessian aggregation over P×P pixels ✓
7. Schur complement computation ✓
8. Solver with damping ✓
9. Back-substitution ✓
10. Update application ✓

⚠️ **Remaining question:** Parameter order for `retr()` - needs testing to determine if reordering is needed


