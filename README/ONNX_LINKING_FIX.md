# Fixing ONNX Runtime Linking Error

## Error
```
undefined reference to `OrtGetApiBase'
```

This means the linker can't find the ONNX Runtime library.

## Solution

### Step 1: Verify Library Exists

```bash
# Check if library file exists
ls -la $ONNXRUNTIME_ROOT/lib/libonnxruntime.so*

# Should see something like:
# libonnxruntime.so -> libonnxruntime.so.1.16.3
# libonnxruntime.so.1.16.3
```

### Step 2: Check Library Path in Makefile

The Makefile should have:
```makefile
LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime
```

### Step 3: Verify Environment Variable

```bash
echo $ONNXRUNTIME_ROOT
# Should output: /tmp/onnxruntime-linux-x64-1.16.3
```

### Step 4: Check Library Architecture

```bash
# Verify library matches your system
file $ONNXRUNTIME_ROOT/lib/libonnxruntime.so.1.16.3

# Should show: ELF 64-bit LSB shared object, x86-64
# (or matching your architecture)
```

### Step 5: Try Explicit Library Path

If the above doesn't work, try linking with explicit path:

```makefile
# In Makefile, change:
LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime

# To:
LDFLAGS += $(ONNXRUNTIME_ROOT)/lib/libonnxruntime.so.1.16.3
```

Or find the exact library name:
```bash
ls -la $ONNXRUNTIME_ROOT/lib/ | grep onnx
```

### Step 6: Set Runtime Library Path

Even if linking succeeds, you need runtime library path:

```bash
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH

# Or add to Makefile (already done):
export LD_LIBRARY_PATH := $(ONNXRUNTIME_ROOT)/lib:$(LD_LIBRARY_PATH)
```

## Quick Test

```bash
# 1. Verify library exists
ls $ONNXRUNTIME_ROOT/lib/libonnxruntime.so*

# 2. Check symbols (should see OrtGetApiBase)
nm -D $ONNXRUNTIME_ROOT/lib/libonnxruntime.so.1.16.3 | grep OrtGetApiBase

# 3. Recompile
make clean
make
```

## Alternative: Use Full Path

If relative path doesn't work, use absolute path in Makefile:

```makefile
ifdef ONNXRUNTIME_ROOT
    CXXFLAGS += -I$(ONNXRUNTIME_ROOT)/include -DUSE_ONNX_RUNTIME
    # Use full path to library
    ONNX_LIB := $(shell find $(ONNXRUNTIME_ROOT)/lib -name "libonnxruntime.so*" | head -1)
    ifneq ($(ONNX_LIB),)
        LDFLAGS += $(ONNX_LIB)
        $(info Found ONNX Runtime library: $(ONNX_LIB))
    else
        $(error ONNX Runtime library not found in $(ONNXRUNTIME_ROOT)/lib)
    endif
    export LD_LIBRARY_PATH := $(ONNXRUNTIME_ROOT)/lib:$(LD_LIBRARY_PATH)
    $(info ONNX Runtime enabled: $(ONNXRUNTIME_ROOT))
endif
```


