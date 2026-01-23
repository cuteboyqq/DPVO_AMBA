# ONNX Runtime Implementation Guide

## Overview

This guide explains how to complete the ONNX Runtime integration for DPVO models. The infrastructure is in place - you just need to:

1. Install ONNX Runtime
2. Complete the ONNX wrapper implementations
3. Modify `Patchifier` and `DPVOUpdate` to use ONNX when enabled

## Current Status

✅ **Completed:**
- Config option `useOnnxRuntime` added to `Config_S`
- Auto-detection: If model paths end with `.onnx`, ONNX Runtime is automatically enabled
- Header files created: `fnet_onnx.hpp`, `inet_onnx.hpp`, `update_onnx.hpp`
- Basic implementation skeleton: `fnet_onnx.cpp` (needs ONNX Runtime library)

⏳ **To Complete:**
- Install ONNX Runtime library
- Complete `inet_onnx.cpp` implementation
- Complete `update_onnx.cpp` implementation  
- Modify `Patchifier::setModels()` to use ONNX wrappers when `config->useOnnxRuntime == true`
- Modify `DPVO::setUpdateModel()` to use ONNX wrapper when `config->useOnnxRuntime == true`

## Step 1: Install ONNX Runtime

### Option A: System Package (Ubuntu/Debian)
```bash
# Download from: https://github.com/microsoft/onnxruntime/releases
# Extract to a directory, e.g., /opt/onnxruntime
export ONNXRUNTIME_ROOT=/opt/onnxruntime
```

### Option B: Build from Source
```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --build_shared_lib --parallel
export ONNXRUNTIME_ROOT=$(pwd)
```

## Step 2: Update Makefile

Add ONNX Runtime to compilation:

```makefile
# Add to CXXFLAGS
CXXFLAGS += -I$(ONNXRUNTIME_ROOT)/include -DUSE_ONNX_RUNTIME

# Add to LDFLAGS  
LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime

# Or if installed system-wide:
CXXFLAGS += -DUSE_ONNX_RUNTIME
LDFLAGS += -lonnxruntime
```

## Step 3: Complete ONNX Implementations

### 3.1 Complete `inet_onnx.cpp`

Copy the pattern from `fnet_onnx.cpp` but change:
- Output channels: `384` instead of `128`
- Model path: Use `inetModelPath` instead of `fnetModelPath`
- Logger name: Use `"inet"` instead of `"fnet"`

### 3.2 Complete `update_onnx.cpp`

This is more complex as it has 6 inputs and 3 outputs. Follow the pattern from `fnet_onnx.cpp` but:

```cpp
// Inputs: net, inp, corr, ii, jj, kk
std::vector<const char*> input_names = {
    "net", "inp", "corr", "ii", "jj", "kk"
};

// Outputs: net_out, d_out, w_out
std::vector<const char*> output_names = {
    "net_out", "d_out", "w_out"
};
```

## Step 4: Modify Patchifier

Update `Patchifier::setModels()` in `patchify.cpp`:

```cpp
void Patchifier::setModels(Config_S *fnetConfig, Config_S *inetConfig)
{
    // ... existing logger setup ...
    
    if (fnetConfig != nullptr)
    {
        if (fnetConfig->useOnnxRuntime) {
            #ifdef USE_ONNX_RUNTIME
            m_fnet_onnx = std::make_unique<FNetInferenceONNX>(fnetConfig);
            m_use_onnx_fnet = true;
            #else
            logger->error("ONNX Runtime requested but not compiled. Install ONNX Runtime and compile with -DUSE_ONNX_RUNTIME");
            m_fnet = std::make_unique<FNetInference>(fnetConfig);
            #endif
        } else {
            m_fnet = std::make_unique<FNetInference>(fnetConfig);
        }
    }
    
    // Similar for INet...
}
```

Then update `Patchifier::forward()` to call the appropriate implementation.

## Step 5: Modify DPVOUpdate

Update `DPVO::setUpdateModel()` in `dpvo.cpp`:

```cpp
void DPVO::setUpdateModel(Config_S* config)
{
    if (config->useOnnxRuntime) {
        #ifdef USE_ONNX_RUNTIME
        m_updateModelONNX = std::make_unique<DPVOUpdateONNX>(config);
        m_use_onnx_update = true;
        #else
        logger->error("ONNX Runtime requested but not compiled");
        m_updateModel = std::make_unique<DPVOUpdate>(config, ...);
        #endif
    } else {
        m_updateModel = std::make_unique<DPVOUpdate>(config, ...);
    }
}
```

## Step 6: Update Headers

Add member variables to `Patchifier` and `DPVO`:

```cpp
// In patchify.hpp
#ifdef USE_ONNX_RUNTIME
std::unique_ptr<FNetInferenceONNX> m_fnet_onnx;
std::unique_ptr<INetInferenceONNX> m_inet_onnx;
bool m_use_onnx_fnet = false;
bool m_use_onnx_inet = false;
#endif

// In dpvo.hpp
#ifdef USE_ONNX_RUNTIME
std::unique_ptr<DPVOUpdateONNX> m_updateModelONNX;
bool m_use_onnx_update = false;
#endif
```

## Testing

1. Set model paths to `.onnx` files in config:
   ```
   FnetModelPath = /path/to/fnet.onnx
   InetModelPath = /path/to/inet.onnx
   UpdateModelPath = /path/to/update.onnx
   ```

2. Or explicitly enable:
   ```
   UseOnnxRuntime = true
   ```

3. Run and compare results with Python implementation

## Troubleshooting

### Compilation Errors

**Error**: `onnxruntime_cxx_api.h: No such file or directory`
- **Fix**: Add `-I$(ONNXRUNTIME_ROOT)/include` to CXXFLAGS

**Error**: `undefined reference to Ort::Session::Session(...)`
- **Fix**: Add `-L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime` to LDFLAGS

### Runtime Errors

**Error**: `Model file does not exist`
- **Fix**: Check model paths in config file

**Error**: `Failed to initialize model`
- **Fix**: Verify ONNX model is valid: `python -c "import onnx; onnx.load('model.onnx')"`

**Error**: `Input/output shape mismatch`
- **Fix**: Check model input/output shapes match expected dimensions

## Next Steps

Once ONNX Runtime is working:

1. **Compare Results**: Run same input through both ONNX and AMBA versions
2. **Debug Differences**: If ONNX works but AMBA doesn't → AMBA conversion issue
3. **Fix AMBA**: If ONNX works correctly, use it as reference to fix AMBA conversion

## Files Created

- `app/inc/fnet_onnx.hpp` - FNet ONNX wrapper header
- `app/inc/inet_onnx.hpp` - INet ONNX wrapper header  
- `app/inc/update_onnx.hpp` - Update model ONNX wrapper header
- `app/src/fnet_onnx.cpp` - FNet ONNX wrapper implementation (skeleton)
- `ONNX_RUNTIME_SETUP.md` - Setup instructions
- `ONNX_RUNTIME_IMPLEMENTATION_GUIDE.md` - This file

## Summary

The infrastructure is ready. To complete:

1. ✅ Config option added
2. ✅ Auto-detection implemented
3. ✅ Headers created
4. ⏳ Install ONNX Runtime
5. ⏳ Complete implementations
6. ⏳ Wire up to Patchifier and DPVOUpdate

Once complete, you can easily switch between AMBA and ONNX by changing model paths or config option!


