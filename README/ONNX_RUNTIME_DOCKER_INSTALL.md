# ONNX Runtime Installation in Docker

## Problem

`libonnxruntime-dev` is not available in standard apt repositories. We need to install ONNX Runtime manually.

## Solution: Download Pre-built Binaries

### Step 1: Download ONNX Runtime

```bash
# Inside Docker container
cd /tmp

# Download Linux x64 build (adjust version/architecture as needed)
# Check latest version at: https://github.com/microsoft/onnxruntime/releases
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz

# Extract
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
cd onnxruntime-linux-x64-1.16.3

# Check structure
ls -la
# Should see: include/, lib/, bin/, etc.
```

### Step 2: Install to System Path (Optional)

```bash
# Copy headers
sudo cp -r include/* /usr/local/include/

# Copy libraries
sudo cp lib/* /usr/local/lib/

# Update library cache
sudo ldconfig
```

### Step 3: Or Use Environment Variable

Instead of installing system-wide, set environment variable:

```bash
export ONNXRUNTIME_ROOT=/tmp/onnxruntime-linux-x64-1.16.3
```

Then update Makefile to use `$(ONNXRUNTIME_ROOT)`.

## Alternative: Build from Source (If Pre-built Doesn't Work)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3 \
    python3-pip \
    libprotobuf-dev \
    protobuf-compiler

# Clone ONNX Runtime
cd /tmp
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime

# Build (this takes a while)
./build.sh --config Release --build_shared_lib --parallel

# Set environment variable
export ONNXRUNTIME_ROOT=/tmp/onnxruntime/build/Linux/Release
```

## Update Makefile

Add to `app/Makefile`:

```makefile
# ONNX Runtime (if ONNXRUNTIME_ROOT is set)
ifdef ONNXRUNTIME_ROOT
    CXXFLAGS += -I$(ONNXRUNTIME_ROOT)/include -DUSE_ONNX_RUNTIME
    LDFLAGS += -L$(ONNXRUNTIME_ROOT)/lib -lonnxruntime
    # Add to library path for runtime
    export LD_LIBRARY_PATH := $(ONNXRUNTIME_ROOT)/lib:$(LD_LIBRARY_PATH)
else
    # Try system-wide installation
    CXXFLAGS += -DUSE_ONNX_RUNTIME
    LDFLAGS += -lonnxruntime
endif
```

## Quick Test Installation

```bash
# Quick test: Download and extract
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
export ONNXRUNTIME_ROOT=/tmp/onnxruntime-linux-x64-1.16.3

# Verify installation
ls $ONNXRUNTIME_ROOT/include/onnxruntime_cxx_api.h
ls $ONNXRUNTIME_ROOT/lib/libonnxruntime.so*

# If files exist, installation is ready!
```

## Dockerfile Integration (Optional)

If you want to bake ONNX Runtime into your Docker image:

```dockerfile
# In your Dockerfile
RUN cd /tmp && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -xzf onnxruntime-linux-x64-1.16.3.tgz && \
    cp -r onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/ && \
    cp onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/ && \
    ldconfig && \
    rm -rf onnxruntime-linux-x64-1.16.3*

ENV ONNXRUNTIME_ROOT=/usr/local
```

## Verify Installation

```bash
# Check if headers are accessible
grep -r "Ort::Session" $ONNXRUNTIME_ROOT/include/onnxruntime_cxx_api.h

# Check if library exists
ls -la $ONNXRUNTIME_ROOT/lib/libonnxruntime.so*

# Test compilation (should not error)
g++ -I$ONNXRUNTIME_ROOT/include -c -o /tmp/test.o -x c++ - <<'EOF'
#include <onnxruntime_cxx_api.h>
int main() { return 0; }
EOF
```

## Common Issues

### Issue: Library not found at runtime

**Fix**: Add to `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$ONNXRUNTIME_ROOT/lib:$LD_LIBRARY_PATH
```

### Issue: Wrong architecture

**Fix**: Check your Docker container architecture:
```bash
uname -m
# x86_64 -> use linux-x64
# aarch64 -> use linux-aarch64
```

### Issue: Version mismatch

**Fix**: Use consistent version. Check available versions:
```bash
# Visit: https://github.com/microsoft/onnxruntime/releases
# Download matching version
```

## Next Steps

Once ONNX Runtime is installed:

1. Set `ONNXRUNTIME_ROOT` environment variable
2. Update Makefile (as shown above)
3. Compile: `make clean && make`
4. Test with `.onnx` model files

