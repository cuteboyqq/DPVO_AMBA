# Changelog

## [0.0.4] - 2026-03-04

### Added
- ✅ **3D object detection on viewer**: Show YOLOv8 detections (pedestrians, vehicles) in the 3D view
  - Bbox centers are unprojected to the ground plane using the current frame pose and intrinsics
  - Pedestrians (class 0) drawn as Tesla-style silhouette (head, torso, legs, arms, neck) in blue-gray
  - Vehicles/other classes drawn as wireframe boxes
- ✅ **EnableShow3DDetection** in `app_config.txt`: Set to `0` or `1` to disable/enable 3D detection in the viewer (default: 1)
- ✅ Shared header `detection_3d.hpp` and `yolov8_onnx.hpp` for 3D detection and YOLOv8 ONNX integration

---

## [0.0.3] - 2026-03-02

### Added
- ✅ YOLOv8 object detection integrated with DPVO (runs on same frames as VO; optional overlay in viewer)
- ✅ Enable/disable YOLOv8 via **EnableYOLOv8** in `app_config.txt` (1 = on, 0 = off; path set by **Yolov8ModelPath**)
- ✅ YOLOv8 ONNX support: when **UseOnnxRuntime = 1** and **Yolov8ModelPath** points to a `.onnx` file, inference uses ONNX Runtime (`yolov8_onnx.hpp/cpp`); otherwise AMBA EazyAI is used
- ✅ YOLOv8 inference time logged in yellow and included in main pipeline line: `[DPVO] Frame X time: ... | YOLOv8: X ms`
- ✅ `app_config.txt`: clearer section headers, notes, and format; documented EnableYOLOv8 and Yolov8ModelPath

### Changed
- ✅ Viewer supports YOLOv8 model size for overlay scaling (e.g. ONNX models with non-512×288 input via `setYOLOv8ModelSize`)

---

## [0.0.2] - 2026-02-13

### Fixed
- ✅ AMBA CV28 model now produces correct poses (FNet, INet, Update models)
- ✅ Fixed FNet/INet preprocessing to match ONNX pipeline (OpenCV resize + BGR→RGB)
- ✅ Resolving an issue in FNet / INet / Update model input tensor handling by using pitch aware raw length instead of image width when allocating AMBA CV28 DMA memory
- ✅ Fixed Update model correlation data reshaping (flat copy matching ONNX)
- ✅ Fixed Update model `m_net` initial state (zeros, matching ONNX)
- ✅ Fixed Update model `reshapeInput` zero-fill, index clamping, and inactive edge padding
- ✅ Fixed Update model calibration data (replaced random dummy data with real runtime inputs)
- ✅ Fixed FNet/INet AMBA conversion config (`color: RGB`, correct `mean`/`std`)
- ✅ Fixed wrong poses after frame ~1000 (raised `n_use` safety threshold from 1000 → 99999)
  - Root cause: `n_use > 1000` guard was resetting `n_use = 0`, making poses jump back to origin
- ✅ Fixed Update model `net` hidden state drift (FP16 error compounding in GRU feedback loop)
  - Root cause: `fx16` precision + narrow calibration range caused `net` values to clip after ~350 frames
  - Fix: Switched to `fp16` precision (`act-force-fp16,coeff-force-fp16`) and extended calibration data to frames 0–1700
- ✅ Supports model input size 288×512 (in addition to 528×960)

### Added
- ✅ Multi-frame calibration data collection for Update model (frames 0–1700, every 20 frames, 85 samples)
- ✅ Diagnostic scripts: `compare_amba_outputs.py`, `compare_update_onnx_outputs.py`, `generate_update_calibration_data.py`
- ✅ Debug logging for `net_out` and `w_out` ranges to detect calibration range exceedance
- ✅ Sliding window size logging in keyframe timing (tracks `m_pg.m_n` growth)
- ✅ Viewer frame saving: auto-saves each rendered frame (3D poses + point cloud) as PNG to `viewer_frames/`
  - Avoids hour-long screen recordings when AMBA model inference is slow
  - Convert to video: `ffmpeg -framerate 30 -start_number 8 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4`

### Finished
- ✅ Model input size: 288x512
- 
### Known Issues
- ⚠️ Random patchify timing spikes (~5s) due to disk I/O stalls when inference cache is enabled
- ⚠️ Sequences longer than ~99999 frames may hit the `n_use` safety threshold

---

## [0.0.1] - 2026-02-10

### Finished
- ✅ ONNX Runtime support (FNet, INet, Update models)
- ✅ AMBA CV28 model support
- ✅ Model input size: 528x960
- ✅ Docker environment (setup/build/run scripts)
- ✅ Correct viewer (Pangolin 3D pose/point cloud visualization)
- ✅ Store FNet/INet output as bin files for reuse
- ✅ Optimize correlation function (1415ms → 60ms)

### Known Issues
- ⚠️ ONNX produce wrong poses after frame ~1000
- ⚠️ ONNX models work successful, but AMBA models failed
