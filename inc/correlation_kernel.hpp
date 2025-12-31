#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
// fmap   : [C][H][W]
// coords : [M][2]  (x, y)
// gmap   : [M][C][D][D], D = 2*radius + 2

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


// Computes correlation between gmap and pyramid features given reprojected coords
// Matches Python CUDA kernel: corr_forward_kernel
// 
// Inputs:
//   gmap: [m_pmem * M * feature_dim * D_gmap * D_gmap] (ring buffer of patch features)
//   pyramid0: [num_frames * feature_dim * fmap1_H * fmap1_W] (flattened, full resolution)
//   pyramid1: [num_frames * feature_dim * fmap2_H * fmap2_W] (flattened, 1/4 resolution)
//   coords: [num_active_edges * 2 * P * P] (flattened reprojection coords [num_active][2][P][P])
//   ii: source patch indices (within frame)
//   jj: target frame indices (for pyramid)
//   kk: linear patch indices (frame * M + patch, for gmap frame extraction)
//
// Output:
//   corr_out: [num_active_edges * D * D * P * P * 2]
//             Layout: [num_active, D, D, P, P, 2] (channel last)
//             This matches Python's torch.stack([corr1, corr2], -1) -> [B, M, D, D, H, W, 2]
//             where D = 2*R + 2 = 8 (R=3), P = 3 (patch size)
void computeCorrelation(
    const float* gmap,
    const float* pyramid0,
    const float* pyramid1,
    const float* coords,
    const int* ii,
    const int* jj,
    const int* kk,
    int num_active,
    int M,      // PATCHES_PER_FRAME
    int P,
    int num_frames,  // Number of frames in pyramid buffers (e.g., m_mem)
    int num_gmap_frames,  // Number of frames in gmap ring buffer (e.g., m_pmem)
    int fmap1_H, int fmap1_W,  // Dimensions for pyramid0 (full resolution)
    int fmap2_H, int fmap2_W,  // Dimensions for pyramid1 (1/4 resolution)
    int feature_dim,
    float* corr_out);
