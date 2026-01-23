# ONNX Runtime Setup Guide

## Overview

This guide explains how to enable ONNX Runtime inference for DPVO models to debug issues with AMBA EazyAI conversion.

## Why Use ONNX Runtime?

If you're experiencing incorrect pose/point results and suspect the AMBA model conversion is the issue, you can use ONNX Runtime to run the original `.onnx` models directly. This allows you to:

1. Verify if the issue is with AMBA conversion or C++ implementation
2. Compare results between ONNX Runtime and AMBA EazyAI
3. Debug model inference independently

## Setup Instructions

### 1. Install ONNX Runtime

```bash
# Option 1: Install from package manager (Ubuntu/Debian)
sudo apt-get install libonnxruntime-dev

# Option 2: Build from source
# Download from: https://github.com/microsoft/onnxruntime/releases
# Extract and build, then set ONNXRUNTIME_ROOT environment variable
```

### 2. Update Makefile

Add ONNX Runtime to your Makefile:

```makefile
# Add to CXXFLAGS
CXXFLAGS += -I$(ONNXRUNTIME_ROOT)/include -DUSE_ONNX_RUNTIME

# Add to LDFLAGS
LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime
```

Or if installed system-wide:
```makefile
CXXFLAGS += -DUSE_ONNX_RUNTIME
LDFLAGS += -lonnxruntime
```

### 3. Configure Model Paths

In your config file, set model paths to `.onnx` files:

```json
{
  "fnetModelPath": "/path/to/fnet.onnx",
  "inetModelPath": "/path/to/inet.onnx",
  "updateModelPath": "/path/to/update.onnx",
  "useOnnxRuntime": true
}
```

Or the code will automatically detect `.onnx` extension and use ONNX Runtime.

### 4. Compile

```bash
cd app
make clean
make
```

## Usage

The code will automatically use ONNX Runtime if:
1. `useOnnxRuntime` is set to `true` in config, OR
2. Model paths end with `.onnx` extension

## Implementation Details

### Interface Compatibility

The ONNX Runtime wrappers (`FNetInferenceONNX`, `INetInferenceONNX`, `DPVOUpdateONNX`) provide the same interface as the AMBA versions:

- Same function signatures
- Same input/output formats
- Same error handling

This means the rest of the code doesn't need to change - just swap the implementation.

### Model Input/Output Formats

**FNet/INet:**
- Input: `[1, 3, H, W]` float32, normalized to `[-0.5, 1.5]`
- Output: `[1, C, H/4, W/4]` float32 (C=128 for FNet, C=384 for INet)

**Update Model:**
- Inputs: `net`, `inp`, `corr`, `ii`, `jj`, `kk` (same as AMBA version)
- Outputs: `net_out`, `d_out`, `w_out` (same shapes as AMBA version)

## Troubleshooting

### ONNX Runtime Not Found

If you get linker errors:
```
undefined reference to `Ort::Session::Session(...)`
```

Make sure:
1. ONNX Runtime is installed
2. `-DUSE_ONNX_RUNTIME` is in CXXFLAGS
3. `-lonnxruntime` is in LDFLAGS
4. Library path is correct (`LD_LIBRARY_PATH` if needed)

### Model Loading Fails

Check:
1. Model file exists and is readable
2. Model file is valid ONNX format
3. Model input/output shapes match expected dimensions

### Performance

ONNX Runtime on CPU may be slower than AMBA EazyAI on hardware accelerator. This is expected and acceptable for debugging purposes.

## Next Steps

After verifying ONNX Runtime works correctly:
1. Compare results between ONNX and AMBA
2. If ONNX works but AMBA doesn't → AMBA conversion issue
3. If both fail → C++ implementation issue
4. If both work → Issue is elsewhere in the pipeline


