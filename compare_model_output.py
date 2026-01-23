import numpy as np

H = 132
W = 240

# Load C++ outputs (CHW format)
fnet_cpp = np.fromfile('fnet_frame0.bin', dtype=np.float32)
fnet_cpp = fnet_cpp.reshape(128, H, W)  # [C, H, W]

inet_cpp = np.fromfile('inet_frame0.bin', dtype=np.float32)
inet_cpp = inet_cpp.reshape(384, H, W)  # [C, H, W]

# Compare with Python ONNX output (NCHW format)
fnet_py = ...  # Shape: [1, 128, H, W]
inet_py = ...  # Shape: [1, 384, H, W]

# Reshape Python output to CHW for comparison
fnet_py_chw = fnet_py[0]  # Remove batch dimension: [128, H, W]
inet_py_chw = inet_py[0]  # Remove batch dimension: [384, H, W]

# Compare
np.allclose(fnet_cpp, fnet_py_chw)
np.allclose(inet_cpp, inet_py_chw)