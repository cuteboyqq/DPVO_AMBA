#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
// fmap   : [C][H][W]
// coords : [M][2]  (x, y)
// gmap   : [M][C][D][D], D = 1 if radius=0, else D = 2*radius + 1 (matches Python altcorr.patchify)

void patchify_cpu(
    const float* fmap,
    const float* coords,
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap
);

void patchify_cpu_safe(
    const float* fmap,    // [C][H][W]
    const float* coords,  // [M][2]
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap           // [M][C][D][D]
);


// -----------------------------------------------------------------------------
// Compute correlation between patch features (gmap) and frame features (pyramid)
// -----------------------------------------------------------------------------
// Computes correlation volumes between patch features and frame features at
// reprojected locations. Used for visual odometry to match patches across frames.
//
// Input Parameters:
//   gmap: [num_gmap_frames, M, feature_dim, D_gmap, D_gmap] - Patch features ring buffer
//         - num_gmap_frames: Frames in ring buffer (e.g., m_pmem = 36)
//         - M: Patches per frame (e.g., 4 or 8)
//         - feature_dim: Feature dimension (128 for FNet)
//         - D_gmap: Patch dimension = 4 (from patchify_cpu_safe with radius=1)
//
//   pyramid0: [num_frames, feature_dim, fmap1_H, fmap1_W] - Feature pyramid at 1/4 resolution
//             - num_frames: Frames in pyramid (e.g., m_mem = 36)
//             - fmap1_H, fmap1_W: Dimensions at 1/4 resolution (e.g., 132x240)
//             - Used for correlation channel 0 (coords scale = 1.0)
//
//   pyramid1: [num_frames, feature_dim, fmap2_H, fmap2_W] - Feature pyramid at 1/16 resolution
//             - Same structure as pyramid0 but at 1/16 resolution
//             - Used for correlation channel 1 (coords scale = 0.25)
//
//   coords: [num_active, 2, P, P] - Reprojected 2D coordinates (u, v) at 1/4 resolution
//           - Channel 0: x coordinates, Channel 1: y coordinates
//
//   ii: [num_active] - Source patch indices (NOT USED, kept for compatibility)
//
//   jj: [num_active] - Target frame indices for pyramid buffers [0, num_frames-1]
//
//   kk: [num_active] - Linear patch indices: kk[e] = gmap_frame * M + patch_idx
//       - gmap_frame = kk[e] / M, patch_idx = kk[e] % M
//
//   num_active: Number of active edges to process
//   M: Patches per frame (PATCHES_PER_FRAME)
//   P: Patch size (typically 3)
//   num_frames: Frames in pyramid buffers (e.g., m_mem)
//   num_gmap_frames: Frames in gmap ring buffer (e.g., m_pmem)
//   fmap1_H, fmap1_W: Dimensions for pyramid0 (1/4 resolution)
//   fmap2_H, fmap2_W: Dimensions for pyramid1 (1/16 resolution)
//   feature_dim: Feature dimension (128 for FNet)
//
// Output:
//   corr_out: [num_active, D, D, P, P, 2] - Correlation volumes
//             - D: Correlation window diameter = 7 (R=3, D = 2*R + 1, matches Python's final output)
//             - Dimension order: [edge, corr_x, corr_y, patch_y, patch_x, level]
//               where corr_x = horizontal offset (dj), corr_y = vertical offset (di)
//             - Matches Python's permute(0,1,3,2,4,5) output format: [B, M, corr_x, corr_y, H, W]
//             - Channel 0: Correlation with pyramid0, Channel 1: Correlation with pyramid1
//             - Each value is dot product between patch and frame features
//
// Matches Python CUDA kernel: corr_forward_kernel
// -----------------------------------------------------------------------------

// Single pyramid level correlation (matches Python: altcorr.corr)
// Output: [num_active, D, D, P, P] (after permute: [num_active, P, P, D, D])
void computeCorrelationSingle(
    const float* gmap,           // [num_gmap_frames, M, feature_dim, D_gmap, D_gmap]
    const float* pyramid,         // [num_frames, feature_dim, fmap_H, fmap_W]
    const float* coords,         // [num_active, 2, P, P] - Reprojected (u, v) coordinates
    const int* ii1,              // [num_active] - Patch indices for gmap (mapped from kk)
    const int* jj1,              // [num_active] - Frame indices for pyramid (mapped from jj)
    int num_active,              // Number of active edges
    int M,                       // Patches per frame (PATCHES_PER_FRAME)
    int P,                       // Patch size (typically 3)
    int num_frames,              // Frames in pyramid buffers (e.g., m_mem)
    int num_gmap_frames,         // Frames in gmap ring buffer (e.g., m_pmem)
    int fmap_H, int fmap_W,      // Dimensions for pyramid
    int feature_dim,             // Feature dimension (128 for FNet)
    float coord_scale,          // Scale factor for coordinates (1.0 for pyramid0, 0.25 for pyramid1)
    int radius,                  // Correlation radius (typically 3)
    float* corr_out,             // Output: [num_active, D, D, P, P] (matches CUDA kernel output)
    float* corr_8x8_out = nullptr);  // Optional: Output 8x8 internal buffer [num_active, 8, 8, P, P] for debugging

// Combined correlation for both pyramid levels (matches Python: torch.stack([corr1, corr2], -1).view(1, len(ii), -1))
// Output: [num_active, D, D, P, P, 2] (channel last) or flattened to [num_active, D*D*P*P*2]
void computeCorrelation(
    const float* gmap,           // [num_gmap_frames, M, feature_dim, D_gmap, D_gmap]
    const float* pyramid0,       // [num_frames, feature_dim, fmap1_H, fmap1_W]
    const float* pyramid1,       // [num_frames, feature_dim, fmap2_H, fmap2_W]
    const float* coords,         // [num_active, 2, P, P] - Reprojected (u, v) coordinates
    const int* ii,               // [num_active] - Source patch indices (NOT USED)
    const int* jj,               // [num_active] - Target frame indices
    const int* kk,               // [num_active] - Linear patch indices (gmap_frame * M + patch_idx)
    int num_active,              // Number of active edges
    int M,                       // Patches per frame (PATCHES_PER_FRAME)
    int P,                       // Patch size (typically 3)
    int num_frames,              // Frames in pyramid buffers (e.g., m_mem)
    int num_gmap_frames,         // Frames in gmap ring buffer (e.g., m_pmem)
    int fmap1_H, int fmap1_W,    // Dimensions for pyramid0 (1/4 resolution)
    int fmap2_H, int fmap2_W,    // Dimensions for pyramid1 (1/16 resolution)
    int feature_dim,             // Feature dimension (128 for FNet)
    float* corr_out,             // Output: [num_active, D, D, P, P, 2]
    int frame_num = -1,          // Optional: Frame number for saving 8x8 debug buffers
    float* corr1_8x8_out = nullptr,  // Optional: Output 8x8 buffer for level 0 [num_active, 8, 8, P, P]
    float* corr2_8x8_out = nullptr);  // Optional: Output 8x8 buffer for level 1 [num_active, 8, 8, P, P]
