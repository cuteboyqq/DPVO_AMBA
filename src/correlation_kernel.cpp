#include "correlation_kernel.hpp"
#include <cmath>
#include <cstring>
#include <vector>

inline bool within_bounds(int h, int w, int H, int W)
{
    return (h >= 0 && h < H && w >= 0 && w < W);
}

void patchify_cpu(
    const float* fmap,    // [C][H][W]
    const float* coords,  // [M][2]
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap           // [M][C][D][D]
)
{
    const int D = 2 * radius + 2;

    // zero output (matches CUDA behavior)
    std::memset(gmap, 0, sizeof(float) * M * C * D * D);

    for (int m = 0; m < M; m++) {

        const float x = coords[m*2 + 0];
        const float y = coords[m*2 + 1];

        const int cx = static_cast<int>(std::floor(x));
        const int cy = static_cast<int>(std::floor(y));

        for (int ii = 0; ii < D; ii++) {
            for (int jj = 0; jj < D; jj++) {

                const int i = cy + (ii - radius);
                const int j = cx + (jj - radius);

                if (!within_bounds(i, j, H, W))
                    continue;

                for (int c = 0; c < C; c++) {

                    // fmap[c][i][j]
                    const int fmap_idx = 
                            (c * (H * W)) + (i * W) + j;
                    // (c * H + i) * W + j;

                    // gmap[m][c][ii][jj]
                    const int gmap_idx =
                            m * C * D * D +
                            c * D * D +
                            ii * D +
                            jj;

                            // ((m * C + c) * D + ii) * D + jj;

                    gmap[gmap_idx] = fmap[fmap_idx];
                }
            }
        }
    }
}


void patchify_cpu_safe(
    const float* fmap,    // [C][H][W]
    const float* coords,  // [M][2]
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap           // [M][C][D][D]
)
{
    const int D = 2 * radius + 2;
    const int fmap_size = C * H * W;
    const int gmap_size = M * C * D * D;

    // --------------------------------------------------
    // 1. Detect aliasing (in-place call)
    // --------------------------------------------------
    const bool inplace = (fmap == gmap);

    // --------------------------------------------------
    // 2. Prepare safe source pointer
    // --------------------------------------------------
    std::vector<float> fmap_copy;
    const float* src = fmap;

    if (inplace) {
        // Copy entire fmap (DPVO semantics: read original fmap)
        fmap_copy.resize(fmap_size);
        std::memcpy(fmap_copy.data(), fmap,
                    sizeof(float) * fmap_size);
        src = fmap_copy.data();
    }

    // --------------------------------------------------
    // 3. Zero output (DPVO behavior)
    // --------------------------------------------------
    std::memset(gmap, 0, sizeof(float) * gmap_size);

    // --------------------------------------------------
    // 4. Patch extraction
    // --------------------------------------------------
    for (int m = 0; m < M; m++) {

        const int cx = static_cast<int>(std::floor(coords[m*2 + 0]));
        const int cy = static_cast<int>(std::floor(coords[m*2 + 1]));

        const int gmap_m_offset = m * C * D * D;

        for (int c = 0; c < C; c++) {

            const int fmap_c_offset = c * H * W;
            const int gmap_c_offset = gmap_m_offset + c * D * D;

            for (int ii = 0; ii < D; ii++) {
                const int y = cy + ii - radius;
                if ((unsigned)y >= (unsigned)H) continue;

                for (int jj = 0; jj < D; jj++) {
                    const int x = cx + jj - radius;
                    if ((unsigned)x >= (unsigned)W) continue;

                    gmap[gmap_c_offset + ii * D + jj] =
                        src[fmap_c_offset + y * W + x];
                }
            }
        }
    }
}


// inline bool within_bounds(int y, int x, int H, int W) {
//     return y >= 0 && y < H && x >= 0 && x < W;
// }

void computeCorrelation(
    const float* gmap,
    const float* pyramid0,
    const float* pyramid1,
    const float* coords,
    const int* ii,
    const int* jj,
    const int* kk,
    int num_active,
    int M,
    int P,
    int num_frames,  // Number of frames in pyramid buffers
    int num_gmap_frames,  // Number of frames in gmap ring buffer (m_pmem)
    int fmap1_H,    // Height for pyramid0 (full resolution)
    int fmap1_W,    // Width for pyramid0 (full resolution)
    int fmap2_H,    // Height for pyramid1 (1/4 resolution)
    int fmap2_W,    // Width for pyramid1 (1/4 resolution)
    int feature_dim,
    float* corr_out)
{
    // Correlation radius R (typically 3 in DPVO)
    // D = 2*R + 2 is the correlation window diameter
    const int R = 3;  // Correlation radius
    const int D = 2 * R + 2;  // Correlation window diameter (D = 8 for R=3)
    
    // gmap structure: gmap is created by patchify_cpu_safe with radius=1, so D_gmap=4
    // gmap is stored as [num_gmap_frames][M][feature_dim][D_gmap][D_gmap] in the ring buffer
    const int D_gmap = 4;  // From patchify_cpu_safe with radius=1 (m_patch_size/2)
    const int gmap_center_offset = (D_gmap - P) / 2;  // Center the P×P region within D_gmap×D_gmap
    
    // Calculate buffer sizes for bounds checking
    const size_t gmap_total_size = static_cast<size_t>(num_gmap_frames) * M * feature_dim * D_gmap * D_gmap;
    const size_t fmap1_total_size = static_cast<size_t>(num_frames) * feature_dim * fmap1_H * fmap1_W;
    const size_t fmap2_total_size = static_cast<size_t>(num_frames) * feature_dim * fmap2_H * fmap2_W;
    const size_t corr_total_size = static_cast<size_t>(num_active) * D * D * P * P * 2;
    
    // For each active edge
    for (int e = 0; e < num_active; e++) {
        // Validate input indices
        if (e < 0 || e >= num_active) continue;
        if (ii == nullptr || jj == nullptr || kk == nullptr) continue;
        
        // us[m] in Python = which patch from gmap to use (linear index)
        // vs[m] in Python = which frame from pyramid to use (frame index)
        // In C++: 
        //   kk[e] = frame * M + patch (linear patch index, encodes source frame for gmap)
        //   ii[e] = patch index within frame (0 to M-1)
        //   jj[e] = target frame index in pyramid (0 to num_frames-1)
        int patch_idx = ii[e] % M;       // patch index within frame (0 to M-1)
        int gmap_frame = (kk[e] / M) % num_gmap_frames; // source frame index in gmap ring buffer
        int pyramid_frame = jj[e] % num_frames; // target frame index in pyramid (0 to num_frames-1)
        
        // Validate indices
        if (patch_idx < 0 || patch_idx >= M) continue;
        if (gmap_frame < 0 || gmap_frame >= num_gmap_frames) continue;
        if (pyramid_frame < 0 || pyramid_frame >= num_frames) continue;

        for (int c = 0; c < 2; c++) { // two correlation channels: pyramid0, pyramid1
            const float* fmap = (c == 0) ? pyramid0 : pyramid1;
            float scale = (c == 0) ? 1.0f : 0.25f; // coords / 1 or /4
            // Use different dimensions for pyramid0 and pyramid1
            int fmap_H = (c == 0) ? fmap1_H : fmap2_H;
            int fmap_W = (c == 0) ? fmap1_W : fmap2_W;
            size_t fmap_total_size = (c == 0) ? fmap1_total_size : fmap2_total_size;

            // For each pixel in the patch (i0, j0) in Python = (y, x) here
            for (int i0 = 0; i0 < P; i0++) {
                for (int j0 = 0; j0 < P; j0++) {
                    // Get projected coordinate for this pixel
                    // Python: x = coords[n][m][0][i0][j0], y = coords[n][m][1][i0][j0]
                    // C++ coords layout from reproject: [num_active][2][P][P] flattened
                    // coords[e * 2 * P * P + 0 * P * P + i0 * P + j0] = x coordinate
                    // coords[e * 2 * P * P + 1 * P * P + i0 * P + j0] = y coordinate
                    // For correlation channel c (0=pyramid0, 1=pyramid1), we use the same coordinates
                    // but scale them differently (scale=1.0 for c=0, scale=0.25 for c=1)
                    int coord_x_idx = e * 2 * P * P + 0 * P * P + i0 * P + j0;
                    int coord_y_idx = e * 2 * P * P + 1 * P * P + i0 * P + j0;
                    float x = coords[coord_x_idx] * scale;
                    float y = coords[coord_y_idx] * scale;
                    
                    // For each offset in the correlation window (ii, jj) in [0, D) × [0, D)
                    for (int corr_ii = 0; corr_ii < D; corr_ii++) {
                        for (int corr_jj = 0; corr_jj < D; corr_jj++) {
                            // Calculate sampling location in target frame
                            // Python: i1 = floor(y) + (ii - R), j1 = floor(x) + (jj - R)
                            int i1 = static_cast<int>(std::floor(y)) + (corr_ii - R);
                            int j1 = static_cast<int>(std::floor(x)) + (corr_jj - R);
                            
                            // Compute correlation: sum over feature channels
                            // Python: sum over C of fmap1[n][ix][c][i0][j0] * fmap2[n][jx][c][i1][j1]
                            float sum = 0.0f;
                            if (within_bounds(i1, j1, fmap_H, fmap_W)) {
                                // Access gmap: gmap[patch_idx][f][gmap_i][gmap_j]
                                // gmap is [M][feature_dim][D_gmap][D_gmap] for a single frame
                                // Extract center P×P region from D_gmap×D_gmap patch
                                int gmap_i = i0 + gmap_center_offset;
                                int gmap_j = j0 + gmap_center_offset;
                                
                                for (int f = 0; f < feature_dim; f++) {
                                    // gmap layout: [num_gmap_frames][M][feature_dim][D_gmap][D_gmap]
                                    size_t gmap_idx = static_cast<size_t>(gmap_frame) * M * feature_dim * D_gmap * D_gmap +
                                                      static_cast<size_t>(patch_idx) * feature_dim * D_gmap * D_gmap +
                                                      static_cast<size_t>(f) * D_gmap * D_gmap +
                                                      static_cast<size_t>(gmap_i) * D_gmap + static_cast<size_t>(gmap_j);
                                    
                                    // Validate gmap index
                                    if (gmap_idx >= gmap_total_size) continue;
                                    
                                    // fmap layout: [num_frames][feature_dim][fmap_H][fmap_W]
                                    // Use channel-specific dimensions (fmap1_H/fmap1_W for c=0, fmap2_H/fmap2_W for c=1)
                                    size_t fmap_idx = static_cast<size_t>(pyramid_frame) * feature_dim * fmap_H * fmap_W +
                                                      static_cast<size_t>(f) * fmap_H * fmap_W +
                                                      static_cast<size_t>(i1) * fmap_W + static_cast<size_t>(j1);
                                    
                                    // Validate fmap index using channel-specific size
                                    if (fmap_idx >= fmap_total_size) continue;
                                    
                                    sum += gmap[gmap_idx] * fmap[fmap_idx];
                                }
                            }
                            
                            // Store correlation: corr[n][m][ii][jj][i0][j0]
                            // Python CUDA kernel outputs: [B, M, D, D, H, W] per channel
                            // Python then stacks: torch.stack([corr1, corr2], -1) -> [B, M, D, D, H, W, 2]
                            // To match Python's stacking order (channel last), we use: [num_active, D, D, P, P, 2]
                            // This matches Python's final shape before flattening: [B, M, D, D, H, W, 2]
                            size_t out_idx = static_cast<size_t>(e) * D * D * P * P * 2 +
                                             static_cast<size_t>(corr_ii) * D * P * P * 2 +
                                             static_cast<size_t>(corr_jj) * P * P * 2 +
                                             static_cast<size_t>(i0) * P * 2 +
                                             static_cast<size_t>(j0) * 2 +
                                             static_cast<size_t>(c);  // Channel last (matches Python's stack along last dim)
                            
                            // Validate output index
                            if (out_idx < corr_total_size) {
                                corr_out[out_idx] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}
