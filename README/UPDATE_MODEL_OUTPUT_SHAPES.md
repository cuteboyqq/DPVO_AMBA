# Update Model Output Shapes: Python vs C++

## Python Implementation (`net.py`)

### UpdateONNX.forward() Output Shapes

From `UpdateONNX.forward()` (lines 832-878):

```python
# Input: net [B, N, DIM] = [1, 360, 384]

# Outputs before reshape:
d_out = self.d(net)  # [B, N, 2] = [1, 360, 2]
w_out = self.w(net)  # [B, N, 2] = [1, 360, 2]

# After reshape for ONNX export:
if self.export_onnx:
    d_out = d_out.permute(0, 2, 1).unsqueeze(-1)  # [1, 360, 2] -> [1, 2, 360] -> [1, 2, 360, 1]
    w_out = w_out.permute(0, 2, 1).unsqueeze(-1)  # [1, 360, 2] -> [1, 2, 360] -> [1, 2, 360, 1]
```

**Final Output Shapes**:
- `d_out`: `[1, 2, 360, 1]` (NCHW format)
- `w_out`: `[1, 2, 360, 1]` (NCHW format)

Where:
- `N = 1` (batch size)
- `C = 2` (channels: delta_x, delta_y OR weight_x, weight_y)
- `H = 360` (number of edges, `m_maxEdge`)
- `W = 1` (spatial dimension)

---

## C++ Implementation

### Expected Output Shapes

The C++ code expects the same shapes as Python:

```cpp
// From dpvo.cpp line 1062-1064:
// Extract outputs: net_out [1, 384, 360, 1], d_out [1, 2, 360, 1], w_out [1, 2, 360, 1]
```

**Expected Shapes**:
- `d_out`: `[1, 2, 360, 1]` ✅ **MATCHES Python**
- `w_out`: `[1, 2, 360, 1]` ✅ **MATCHES Python**

---

## Indexing Formula

### NCHW Layout: `[N, C, H, W] = [1, 2, 360, 1]`

**General Formula**:
```
index = n * C * H * W + c * H * W + h * W + w
```

**For `d_out` and `w_out`**:
- `n = 0` (batch index)
- `C = 2` (channels)
- `H = 360` (number of edges)
- `W = 1` (spatial dimension)

**For edge `e`, channel `c`**:
```
index = 0 * 2 * 360 * 1 + c * 360 * 1 + e * 1 + 0
     = c * 360 + e
```

**Simplified**:
- Channel 0 (delta_x / weight_x): `index = 0 * 360 + e = e`
- Channel 1 (delta_y / weight_y): `index = 1 * 360 + e = 360 + e`

---

## C++ Extraction Code

### From `dpvo.cpp` (lines 1070-1085):

```cpp
// d_out layout: [N, C, H, W] = [1, 2, m_maxEdge, 1]
// Index: n * C * H * W + c * H * W + h * W + w
// Where: n=0, c=0 or 1, h=e, w=0
int idx0 = 0 * 2 * m_maxEdge * 1 + 0 * m_maxEdge * 1 + e * 1 + 0;
int idx1 = 0 * 2 * m_maxEdge * 1 + 1 * m_maxEdge * 1 + e * 1 + 0;

delta[e * 2 + 0] = pred.dOutBuff[idx0];  // delta_x for edge e
delta[e * 2 + 1] = pred.dOutBuff[idx1];  // delta_y for edge e

// Weight extraction (same indexing)
float w0 = pred.wOutBuff[idx0];  // weight_x for edge e
float w1 = pred.wOutBuff[idx1];  // weight_y for edge e
weight[e] = w0;  // Uses channel 0 (weight_x)
```

**Simplified Formula** (equivalent):
```cpp
int idx0 = e;           // Channel 0: delta_x / weight_x
int idx1 = m_maxEdge + e;  // Channel 1: delta_y / weight_y
```

---

## Verification

### Python Output Shape
```
d_out: float32[1, 2, 360, 1]
w_out: float32[1, 2, 360, 1]
```

### C++ Expected Shape
```cpp
d_out: [1, 2, m_maxEdge, 1]  // where m_maxEdge = 360
w_out: [1, 2, m_maxEdge, 1]  // where m_maxEdge = 360
```

### Indexing Verification

For edge `e = 0`:
- Python: `d_out[0, 0, 0, 0]` = delta_x for edge 0
- C++: `idx0 = 0 * 2 * 360 * 1 + 0 * 360 * 1 + 0 * 1 + 0 = 0` ✅

For edge `e = 0`, channel 1:
- Python: `d_out[0, 1, 0, 0]` = delta_y for edge 0
- C++: `idx1 = 0 * 2 * 360 * 1 + 1 * 360 * 1 + 0 * 1 + 0 = 360` ✅

For edge `e = 100`:
- Python: `d_out[0, 0, 100, 0]` = delta_x for edge 100
- C++: `idx0 = 0 * 2 * 360 * 1 + 0 * 360 * 1 + 100 * 1 + 0 = 100` ✅

---

## Conclusion

✅ **The C++ implementation is CORRECT!**

The output shapes `[1, 2, 360, 1]` match exactly between Python and C++:
- **Python**: `d_out.permute(0, 2, 1).unsqueeze(-1)` produces `[1, 2, 360, 1]`
- **C++**: Expects `[1, 2, m_maxEdge, 1]` where `m_maxEdge = 360`

The indexing formula in C++ correctly extracts:
- Channel 0: `idx = e` (delta_x / weight_x)
- Channel 1: `idx = 360 + e` (delta_y / weight_y)

---

## Note on Weight Extraction

The C++ code currently uses **channel 0** for weight:
```cpp
weight[e] = w0;  // Uses channel 0 (weight_x)
```

However, Python outputs **both channels** (`[1, 2, 360, 1]`), suggesting there might be separate weights for x and y. The C++ code could potentially use:
- `weight_x[e] = w0` (channel 0)
- `weight_y[e] = w1` (channel 1)

But currently, it uses a single weight per edge from channel 0, which matches the original DPVO implementation where weight is a scalar per edge.

