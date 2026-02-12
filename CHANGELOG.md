# Changelog

## [0.0.2] - 2026-02-12

### Fixed
- ✅ AMBA CV28 model now produces correct poses (FNet, INet, Update models)
- ✅ Fixed FNet/INet preprocessing to match ONNX pipeline (OpenCV resize + BGR→RGB)
- ✅ Fixed Update model pitch-aware tensor I/O for CV28 32-byte row alignment
- ✅ Fixed Update model correlation data reshaping (flat copy matching ONNX)
- ✅ Fixed Update model `m_net` initial state (zeros, matching ONNX)
- ✅ Fixed Update model `reshapeInput` zero-fill, index clamping, and inactive edge padding
- ✅ Fixed Update model calibration data (replaced random dummy data with real runtime inputs)
- ✅ Fixed FNet/INet AMBA conversion config (`color: RGB`, correct `mean`/`std`)
- ✅ Fixed wrong poses after frame ~1000 (raised `n_use` safety threshold from 1000 → 99999)
  - Root cause: `n_use > 1000` guard was resetting `n_use = 0`, making poses jump back to origin

### Added
- ✅ Multi-frame calibration data collection for Update model (frames 0–600, every 20 frames)
- ✅ Diagnostic scripts: `compare_amba_outputs.py`, `compare_update_onnx_outputs.py`, `generate_update_calibration_data.py`
- ✅ Debug logging for `net_out` and `w_out` ranges to detect calibration range exceedance
- ✅ Sliding window size logging in keyframe timing (tracks `m_pg.m_n` growth)

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
