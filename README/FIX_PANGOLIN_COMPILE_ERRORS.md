# Fixing Pangolin Compilation Errors

## Problem

When compiling with Pangolin, you get errors like:
```
error: 'glCopyImageSubDataNV' was not declared in this scope
error: 'glGenRenderbuffers' was not declared in this scope
error: 'glBindRenderbuffer' was not declared in this scope
```

## Root Cause

Pangolin's `gl.hpp` uses OpenGL functions that require:
1. OpenGL extension headers to be available
2. Proper OpenGL version defines
3. Epoxy library (which Pangolin was built with)

## Solution

I've updated the code to fix this:

### 1. Updated `app/inc/dpvo_viewer.hpp`
- Removed explicit `#include <pangolin/gl/gl.h>` 
- Only include `#include <pangolin/pangolin.h>` (it handles everything)

### 2. Updated `app/Makefile`
- Added `-DGL_GLEXT_PROTOTYPES` to CXXFLAGS (enables OpenGL extension prototypes)
- Added `-lepoxy` to LDFLAGS (Pangolin was built with epoxy support)

## Try Building Again

```bash
cd /src/app
make clean
make
```

## If Errors Persist

### Option 1: Include OpenGL headers explicitly

If the above doesn't work, try modifying `app/inc/dpvo_viewer.hpp`:

```cpp
#ifdef ENABLE_PANGOLIN_VIEWER
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <pangolin/pangolin.h>
#ifdef __linux__
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif
#endif
```

### Option 2: Check Pangolin build configuration

Verify how Pangolin was built:

```bash
cd /tmp/Pangolin/build/build
cat CMakeCache.txt | grep -i glew
cat CMakeCache.txt | grep -i epoxy
```

### Option 3: Rebuild Pangolin with GLEW instead of epoxy

If epoxy is causing issues, rebuild Pangolin to use GLEW:

```bash
cd /tmp/Pangolin/build/build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_PANGOLIN_PYTHON=OFF \
    -DBUILD_PANGOLIN_TESTS=OFF \
    -DCMAKE_DISABLE_FIND_PACKAGE_epoxy=ON
make -j$(nproc)
make install
ldconfig
```

Then update Makefile to remove `-lepoxy` from LDFLAGS.

## Current Makefile Settings

```makefile
CXXFLAGS := ... -DENABLE_PANGOLIN_VIEWER -DGL_GLEXT_PROTOTYPES
LDFLAGS := ... -lpangolin -lepoxy -lGL -lGLU -lGLEW
```

The `-DGL_GLEXT_PROTOTYPES` define tells the compiler to include OpenGL extension function prototypes, which Pangolin's gl.hpp needs.

