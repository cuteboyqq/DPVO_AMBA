# BA Input Functions Comparison: C++ vs Python

This document compares the functions that provide inputs to BA to identify potential issues causing incorrect translation and rotation.

## Functions That Feed Into BA

1. **Reproject/Transform** - Computes reprojected coordinates and Jacobians
2. **Patchify** - Extracts patches from images
3. **Correlation** - Computes correlation volumes (used by update model, not directly by BA)
4. **Update Model** - Predicts target coordinates and weights (provides `m_target` and `m_weight`)

---

## 1. Reproject/Transform Function

### Python (`projective_ops.py` line 60):
```python
Gij = poses[:, jj] * poses[:, ii].inv()
X1 = Gij[:,:,None,None] * X0
```

### C++ (`projective_ops.cpp` line 272):
```cpp
SE3 Gij = Tj * Ti.inverse();  // Transform from frame i to frame j
Eigen::Vector3f p1_vec = Gij.R() * p0_vec + Gij.t * W0_pix;
```

✅ **MATCHES** - Both compute `Gij = Tj * Ti^-1` and apply SE3 action

**Key Question**: Does C++ SE3 action match Python SE3 action?

**Python SE3 action** (from lietorch):
- `SE3 * [X, Y, Z, W]` = `[R*[X,Y,Z] + t*W, W]`

**C++ SE3 action** (line 475):
- `p1_vec = Gij.R() * p0_vec + Gij.t * W0_pix`
- This matches: `R*[X,Y,Z] + t*W`

✅ **MATCHES** - SE3 action is correct

---

## 2. Jacobian Computation

### Python (`projective_ops.py` lines 71-108):
```python
if jacobian:
    p = X1.shape[2]
    X, Y, Z, H = X1[...,p//2,p//2,:].unbind(dim=-1)
    o = torch.zeros_like(H)
    i = torch.zeros_like(H)
    
    fx, fy, cx, cy = intrinsics[:,jj].unbind(dim=-1)
    
    d = torch.zeros_like(Z)
    d[Z.abs() > 0.2] = 1.0 / Z[Z.abs() > 0.2]
    
    Ja = torch.stack([
        H,  o,  o,  o,  Z, -Y,
        o,  H,  o, -Z,  o,  X,
        o,  o,  H,  Y, -X,  o,
        o,  o,  o,  o,  o,  o,
    ], dim=-1).view(1, len(ii), 4, 6)
    
    Jp = torch.stack([
         fx*d,     o, -fx*X*d*d,  o,
            o,  fy*d, -fy*Y*d*d,  o,
    ], dim=-1).view(1, len(ii), 2, 4)
    
    Jj = torch.matmul(Jp, Ja)
    Ji = -Gij[:,:,None].adjT(Jj)
    
    Jz = torch.matmul(Jp, Gij.matrix()[...,:,3:])
```

### C++ (`projective_ops.cpp` lines 500-700+):
Need to check if Jacobian computation matches Python exactly.

**Key Questions**:
1. Is `Ja` computed correctly? (4x6 matrix)
2. Is `Jp` computed correctly? (2x4 matrix)
3. Is `Jj = Jp @ Ja` correct? (2x6 matrix)
4. Is `Ji = -adjT(Jj)` correct? (2x6 matrix)
5. Is `Jz = Jp @ Gij.matrix()[:,3:]` correct? (2x1 matrix)

---

## 3. Patchify Function

### Python (`patchify.py` or `net.py`):
- Extracts patches from feature maps
- Stores coordinates and features

### C++ (`patchify.cpp`):
- Extracts patches from feature maps
- Stores coordinates and features

**Key Questions**:
1. Are patch coordinates stored correctly?
2. Are patch features extracted correctly?
3. Is the coordinate system consistent?

---

## 4. Update Model Outputs

### Python:
- `delta`: [num_active, 2] - predicted offset from current coords
- `weight`: [num_active] - confidence weight per edge

### C++:
- `delta`: [num_active, 2] - predicted offset
- `weight`: [num_active, 2] - per-dimension weights (x and y)

**Key Questions**:
1. Are `target = coords + delta` computed correctly?
2. Are weights applied correctly in BA?

---

## Next Steps

1. **Verify Jacobian computation** - This is critical for BA correctness
2. **Check SE3 adjoint transpose** - Used in `Ji = -adjT(Jj)`
3. **Verify patch storage and retrieval** - Ensure coordinates are correct
4. **Check update model outputs** - Ensure targets are computed correctly

