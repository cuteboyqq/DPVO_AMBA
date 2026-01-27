#ifndef TARGET_FRAME_HPP
#define TARGET_FRAME_HPP

// =================================================================================================
// Shared TARGET_FRAME constant for saving debug outputs
// =================================================================================================
// This constant controls which frame's data is saved to binary files for comparison with Python.
// Set to the frame number you want to save (0-indexed, e.g., 77 for frame 77).
// Set to -1 to disable saving.
// 
// This constant is used in:
//   - dpvo.cpp: Save BA input parameters
//   - ba.cpp: Save BA intermediate outputs
//   - patchify.cpp: Save ONNX model outputs (FNet/INet) and patchify outputs (coords, gmap, imap, patches)
// =================================================================================================

static const int TARGET_FRAME = 69;  // Change this to save a different frame

#endif // TARGET_FRAME_HPP

