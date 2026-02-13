// =================================================================================================
// Shared TARGET_FRAME constant definition
// =================================================================================================
// This constant controls which frame's data is saved to binary files for comparison with Python.
// Set to the frame number you want to save (0-indexed, e.g., 69 for frame 69).
// Set to -1 to disable saving.
// 
// This constant is declared as extern in target_frame.hpp and defined here.
// 
// This constant is used in:
//   - dpvo.cpp: Save BA input parameters
//   - ba.cpp: Save BA intermediate outputs
//   - patchify.cpp: Save ONNX model outputs (FNet/INet) and patchify outputs (coords, gmap, imap, patches)
//   - update_onnx.cpp: Save update model inputs/outputs
// =================================================================================================

// Note: 'extern' is required here to give the constant external linkage
// Without it, 'const' variables at namespace scope have internal linkage by default
extern const int TARGET_FRAME = -1;  // Change this to save a different frame

