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
// fmap_gmap: [num_patches_total * feature_dim] (flattened)
// pyramid0, pyramid1: [num_frames * feature_dim * fmap_H * fmap_W] (flattened)
// coords: [num_active_edges * 2 * P * P] (flattened reprojection coords)
// ii: source patch indices
// jj: target frame indices
// Output corr: [num_active_edges * 2 * P * P * 2]  (stack corr1 and corr2)
void computeCorrelation(
    const float* gmap,
    const float* pyramid0,
    const float* pyramid1,
    const float* coords,
    const int* ii,
    const int* jj,
    int num_active,
    int M,      // PATCHES_PER_FRAME
    int P,
    int num_frames,  // Number of frames in pyramid buffers (e.g., m_mem)
    int fmap_H, int fmap_W,
    int feature_dim,
    float* corr_out);
