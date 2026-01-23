# DPVO Code Structure Summary

## 📁 Project Structure

```
DPVO_AMBA/
├── app/
│   ├── inc/          # Header files (.hpp, .h)
│   ├── src/          # Source files (.cpp)
│   └── lib/          # External libraries (eazyai, Eigen)
└── build/            # Build outputs, models, videos
```

---

## 🎯 Core DPVO Modules

### **1. Main Entry Point**
- **`main.cpp` / `main.hpp`**
  - Application entry point
  - Video/image input handling
  - Thread management (DPVO & WNC_APP)
  - Frame processing pipeline

---

### **2. DPVO Core System**
- **`dpvo.cpp` / `dpvo.hpp`**
  - **Main DPVO class** - orchestrates entire pipeline
  - Frame processing (`run()`)
  - Feature extraction & patchification
  - Update loop with network inference
  - Bundle adjustment integration
  - Threading interface for AMBA CV28

---

### **3. Neural Network Inference**
- **`net.cpp` / `net.hpp`**
  - **FNetInference**: Feature network (extracts 128-dim features)
  - **INetInference**: Intensity network (extracts 384-dim features)
  - **DPVOUpdate**: Update network (refines patches & poses)
  - AMBA CV28 model loading & inference
  - Tensor data preparation & conversion

---

### **4. Patch Graph Data Structure**
- **`patch_graph.cpp` / `patch_graph.hpp`**
  - **PatchGraph class**: Stores frames, patches, edges
  - Frame poses (SE3 transformations)
  - Patch states (x, y, inverse depth)
  - Edge connectivity (ii, jj, kk indices)
  - Network state buffers (m_net)

---

### **5. Bundle Adjustment**
- **`ba.cpp` / `ba.hpp`**
  - **Bundle Adjustment solver**
  - Schur complement optimization
  - Pose & structure refinement
  - Levenberg-Marquardt damping
  - Uses Eigen for matrix operations

---

### **6. Projective Operations**
- **`projective_ops.cpp` / `projective_ops.hpp`**
  - 3D point transformations (SE3)
  - Camera projection & reprojection
  - Jacobian computation (Ji, Jj, Jz)
  - Flow magnitude calculation
  - Coordinate transformations

---

### **7. Correlation Computation**
- **`correlation_kernel.cpp` / `correlation_kernel.hpp`**
  - Feature correlation between frames
  - D×D correlation window computation
  - Multi-scale feature matching
  - Correlation volume generation

---

### **8. SE3 Transformations**
- **`se3.cpp` / `se3.h`**
  - SE3 Lie group operations
  - Rotation & translation matrices
  - Adjoint & adjoint transpose
  - Pose composition & inversion

---

## 🔧 Supporting Modules

### **9. Configuration**
- **`config_reader.cpp` / `config_reader.hpp`**
  - Reads YAML/JSON config files
  - Model paths (fnet, inet, update)
  - DPVO parameters (patches, buffer size)
  - Camera intrinsics

- **`dla_config.hpp`**
  - Configuration structure definitions
  - Model paths & settings

---

### **10. Utilities**
- **`utils.cpp` / `utils.hpp`**
  - General utility functions
  - Image processing helpers

- **`img_util.cpp` / `img_util.hpp`**
  - Image format conversion
  - Tensor-to-image conversion

- **`logger.cpp` / `logger.hpp`**
  - Logging system (spdlog)
  - Log level management

---

## 📊 Data Flow

```
Input Image
    ↓
[main.cpp] → Frame Queue
    ↓
[dpvo.cpp] → run()
    ↓
[net.cpp] → FNet/INet → Feature Maps
    ↓
[dpvo.cpp] → Patchification → Patches
    ↓
[correlation_kernel.cpp] → Correlation Volume
    ↓
[net.cpp] → DPVOUpdate → Delta & Weights
    ↓
[ba.cpp] → Bundle Adjustment → Refined Poses & Patches
    ↓
Output: Camera Trajectory
```

---

## 🔑 Key Components

| Component | Purpose |
|----------|---------|
| **DPVO** | Main pipeline orchestrator |
| **PatchGraph** | Data structure for frames/patches/edges |
| **FNet/INet** | Feature extraction networks |
| **DPVOUpdate** | Patch refinement network |
| **Bundle Adjustment** | Non-linear optimization |
| **Correlation** | Feature matching between frames |
| **SE3** | 3D pose representation |

---

## 🎯 Key Features

- ✅ **Multi-threaded processing** for AMBA CV28
- ✅ **Three neural networks**: FNet, INet, Update
- ✅ **Bundle Adjustment** for pose refinement
- ✅ **Patch-based tracking** with correlation
- ✅ **SE3 pose representation** for 3D transformations
- ✅ **Eigen library** for matrix operations
- ✅ **AMBA eazyai** integration for model inference

---

## 📝 File Count Summary

- **Core DPVO**: 8 files (dpvo, net, patch_graph, ba, projective_ops, correlation, se3)
- **Supporting**: 5 files (config, utils, logger, img_util, main)
- **Total DPVO-related**: ~13 files


