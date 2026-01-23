# Reproject and Jacobian Verification: C++ vs Python

## Critical Functions That Feed Into BA

Since BA function matches Python exactly, but poses are wrong, the issue must be in the inputs to BA:
1. **Reprojected coordinates** (`coords`)
2. **Jacobians** (`Ji`, `Jj`, `Jz`)
3. **Targets** (`m_target`)
4. **Weights** (`m_weight`)

---

## 1. Reproject Function: Transform and SE3 Action

### Python (`projective_ops.py` lines 59-65):
```python
Gij = poses[:, jj] * poses[:, ii].inv()
X1 = Gij[:,:,None,None] * X0
```

### C++ (`projective_ops.cpp` line 272):
```cpp
SE3 Gij = Tj * Ti.inverse();  // Transform from frame i to frame j
Eigen::Vector3f p1_vec = Gij.R() * p0_vec + Gij.t * W0_pix;
```

✅ **MATCHES** - Both compute `Gij = Tj * Ti^-1` and apply SE3 action `R*[X,Y,Z] + t*W`

---

## 2. Jacobian Computation: Ja Matrix

### Python (`projective_ops.py` lines 83-88):
```python
Ja = torch.stack([
    H,  o,  o,  o,  Z, -Y,
    o,  H,  o, -Z,  o,  X,
    o,  o,  H,  Y, -X,  o,
    o,  o,  o,  o,  o,  o,
], dim=-1).view(1, len(ii), 4, 6)
```

### C++ (`projective_ops.cpp` lines 669-674):
```cpp
Ja(0, 0) = H;  Ja(0, 4) = Z;   Ja(0, 5) = -Y;
Ja(1, 1) = H;  Ja(1, 3) = -Z;  Ja(1, 5) = X;
Ja(2, 2) = H;  Ja(2, 3) = Y;   Ja(2, 4) = -X;
```

✅ **MATCHES** - Both compute the same 4x6 Jacobian matrix

---

## 3. Jacobian Computation: Jp Matrix

### Python (`projective_ops.py` lines 98-101):
```python
Jp = torch.stack([
     fx*d,     o, -fx*X*d*d,  o,
        o,  fy*d, -fy*Y*d*d,  o,
], dim=-1).view(1, len(ii), 2, 4)
```

### C++ (`projective_ops.cpp` lines 687-692):
```cpp
Jp(0, 0) = fx_j * d_jac;                          // du/dX
Jp(0, 2) = -fx_j * X * d_jac * d_jac;            // du/dZ
Jp(1, 1) = fy_j * d_jac;                          // dv/dY
Jp(1, 2) = -fy_j * Y * d_jac * d_jac;            // dv/dZ
```

✅ **MATCHES** - Both compute the same 2x4 projection Jacobian

---

## 4. Jacobian Computation: Jj (w.r.t. pose j)

### Python (`projective_ops.py` line 103):
```python
Jj = torch.matmul(Jp, Ja)
```

### C++ (`projective_ops.cpp` line 700):
```cpp
Eigen::Matrix<float, 2, 6> Jj = Jp * Ja;
```

✅ **MATCHES** - Both compute `Jj = Jp @ Ja` (2x4 @ 4x6 = 2x6)

---

## 5. Jacobian Computation: Ji (w.r.t. pose i)

### Python (`projective_ops.py` line 104):
```python
Ji = -Gij[:,:,None].adjT(Jj)
```

### C++ (`projective_ops.cpp` line 709):
```cpp
Eigen::Matrix<float, 2, 6> Ji = -Gij.adjointT(Jj);
```

✅ **MATCHES** - Both compute `Ji = -adjT(Jj)`

**Verification of adjointT**:
- Python: `Adj().transpose() * a` where `Adj() = [R, tx*R; 0, R]`
- C++: `J * Ad_T` where `Ad_T = [R^T, -R^T*skew(t); 0, R^T]`
- These are equivalent: `(Adj())^T = Ad_T` ✅

---

## 6. Jacobian Computation: Jz (w.r.t. inverse depth)

### Python (`projective_ops.py` line 106):
```python
Jz = torch.matmul(Jp, Gij.matrix()[...,:,3:])
```

### C++ (`projective_ops.cpp` lines 720-722):
```cpp
Eigen::Matrix4f Gij_mat = Gij.matrix();
Eigen::Vector4f t_col = Gij_mat.col(3);  // Translation column [tx, ty, tz, 1]
Eigen::Matrix<float, 2, 1> Jz = Jp * t_col;
```

✅ **MATCHES** - Both compute `Jz = Jp @ Gij.matrix()[:,3:]` (2x4 @ 4x1 = 2x1)

---

## Potential Issues to Check

1. **Frame Index Extraction**: 
   - Python: `ii` and `jj` are frame indices directly
   - C++: `ii` is patch index mapping, source frame extracted from `kk`
   - **Verify**: Is `i = kk[e] / M` correct?

2. **Patch Coordinate Storage**:
   - Are patch coordinates stored correctly in `m_patches`?
   - Are they at the correct resolution (1/4)?

3. **Intrinsics Scaling**:
   - Are intrinsics scaled correctly to match patch resolution?

4. **Target Computation**:
   - Python: `target = coords[center] + delta`
   - C++: `m_target[e] = coords[center] + delta`
   - **Verify**: Are targets computed correctly?

5. **Weight Application**:
   - Python: Per-residual weights
   - C++: Per-dimension weights (x and y)
   - **Verify**: Is this causing issues?

---

## Next Steps

1. Add detailed logging to compare C++ and Python outputs for:
   - Reprojected coordinates
   - Jacobians (Ji, Jj, Jz)
   - Targets
   - Weights

2. Check frame index extraction logic

3. Verify patch coordinate storage and retrieval

4. Check intrinsics scaling consistency

