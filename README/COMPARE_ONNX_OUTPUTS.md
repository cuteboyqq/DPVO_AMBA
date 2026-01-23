# Compare C++ and Python ONNX Inference Outputs

This guide explains how to compare C++ ONNX inference outputs with Python ONNX inference outputs.

## Files Created

1. **`compare_onnx_outputs.py`** - Python script to run ONNX inference and compare outputs
2. **`fnet_frame0.bin`** - C++ FNet output (created by C++ code)
3. **`inet_frame0.bin`** - C++ INet output (created by C++ code)
4. **`fnet_py_frame0.bin`** - Python FNet output (created by Python script)
5. **`inet_py_frame0.bin`** - Python INet output (created by Python script)

## Step 1: Run C++ Code

Run your C++ application. It will automatically save the first frame's outputs:
- `fnet_frame0.bin` - FNet output
- `inet_frame0.bin` - INet output

The C++ code will log:
```
[Patchifier] Saved fnet output to fnet_frame0.bin: shape=[1, 128, H, W] (NCHW), X bytes
[Patchifier] Saved inet output to inet_frame0.bin: shape=[1, 384, H, W] (NCHW), X bytes
```

## Step 2: Run Python Comparison Script

```bash
python compare_onnx_outputs.py <image_path> <fnet_model_path> <inet_model_path>
```

**Example:**
```bash
python compare_onnx_outputs.py frame0.jpg models/fnet.onnx models/inet.onnx
```

## What the Script Does

1. **Loads and preprocesses the image**:
   - Loads image from file
   - Resizes to model input size (528x960)
   - Normalizes: `(2 * image - 127.5) / 255.0`
   - Converts to NCHW format: `[1, 3, H, W]`

2. **Runs ONNX inference**:
   - Runs FNet model → `fnet_py` shape: `[1, 128, H, W]`
   - Runs INet model → `inet_py` shape: `[1, 384, H, W]`

3. **Saves Python outputs**:
   - `fnet_py_frame0.bin` - FNet output (CHW format: `[128, H, W]`)
   - `inet_py_frame0.bin` - INet output (CHW format: `[384, H, W]`)

4. **Loads C++ outputs**:
   - `fnet_frame0.bin` → `fnet_cpp` shape: `[128, H, W]`
   - `inet_frame0.bin` → `inet_cpp` shape: `[384, H, W]`

5. **Compares outputs**:
   - Computes max difference, mean difference
   - Counts elements that differ beyond tolerance (1e-5)
   - Shows sample differences if any

## Output Format

### C++ Output Format
- **Storage**: Binary files with `float32` values
- **Layout**: CHW (Channel, Height, Width), row-major
- **FNet**: `[128, fmap_H, fmap_W]` floats
- **INet**: `[384, fmap_H, fmap_W]` floats

### Python Output Format
- **Storage**: Binary files with `float32` values (same as C++)
- **Layout**: CHW (Channel, Height, Width), row-major
- **FNet**: `[128, fmap_H, fmap_W]` floats
- **INet**: `[384, fmap_H, fmap_W]` floats

## Expected Output

If outputs match:
```
================================================================================
Comparing outputs...
================================================================================
  FNet comparison:
    Max difference: 1.234567e-06
    Mean difference: 2.345678e-07
    Elements different (>1e-05): 0/4055040 (0.00%)
    ✓ PASS: All values match within tolerance (1e-05)

  INet comparison:
    Max difference: 1.234567e-06
    Mean difference: 2.345678e-07
    Elements different (>1e-05): 0/12165120 (0.00%)
    ✓ PASS: All values match within tolerance (1e-05)

================================================================================
Summary:
================================================================================
FNet outputs match: ✓ YES
INet outputs match: ✓ YES

✓ All outputs match! C++ and Python ONNX inference produce identical results.
```

If outputs differ:
```
  FNet comparison:
    Max difference: 1.234567e-03
    Mean difference: 2.345678e-04
    Elements different (>1e-05): 12345/4055040 (0.30%)
    ✗ FAIL: Values differ beyond tolerance (1e-05)
    Sample differences (first 5):
      [(0, 0, 0)]: C++=0.123456, Python=0.123789, diff=3.330000e-04
      ...
```

## Requirements

```bash
pip install numpy onnxruntime opencv-python
```

## Troubleshooting

### Issue: "fnet_frame0.bin not found"
**Solution**: Make sure you've run the C++ code first. The files are created in the working directory.

### Issue: Shape mismatch
**Solution**: Check that:
1. C++ and Python are using the same model files
2. Image preprocessing is identical (normalization, resize)
3. Model input/output shapes match

### Issue: Large differences
**Possible causes**:
1. Different ONNX Runtime versions
2. Different input preprocessing
3. Different model files
4. Numerical precision differences (use larger tolerance)

## Manual Comparison

If you want to manually compare the files:

```python
import numpy as np

# Load C++ outputs
fnet_cpp = np.fromfile('fnet_frame0.bin', dtype=np.float32)
fnet_cpp = fnet_cpp.reshape(128, H, W)  # Adjust H, W based on your model

inet_cpp = np.fromfile('inet_frame0.bin', dtype=np.float32)
inet_cpp = inet_cpp.reshape(384, H, W)

# Load Python outputs
fnet_py = np.fromfile('fnet_py_frame0.bin', dtype=np.float32)
fnet_py = fnet_py.reshape(128, H, W)

inet_py = np.fromfile('inet_py_frame0.bin', dtype=np.float32)
inet_py = inet_py.reshape(384, H, W)

# Compare
print("FNet max diff:", np.max(np.abs(fnet_cpp - fnet_py)))
print("INet max diff:", np.max(np.abs(inet_cpp - inet_py)))
print("FNet all close:", np.allclose(fnet_cpp, fnet_py, atol=1e-5))
print("INet all close:", np.allclose(inet_cpp, inet_py, atol=1e-5))
```

