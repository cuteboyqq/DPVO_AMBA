#include "correlation_kernel.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>
#include <limits>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

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

        const float coord_x = coords[m*2 + 0];
        const float coord_y = coords[m*2 + 1];
        const int cx = static_cast<int>(std::floor(coord_x));
        const int cy = static_cast<int>(std::floor(coord_y));

        // Debug logging for first few patches
        if (m < 3) {
            printf("[patchify_cpu_safe] Patch %d: coords=(%.2f, %.2f), floor=(%d, %d), H=%d, W=%d, radius=%d, D=%d\n",
                   m, coord_x, coord_y, cx, cy, H, W, radius, D);
            fflush(stdout);
        }

        const int gmap_m_offset = m * C * D * D;

        for (int c = 0; c < C; c++) {

            const int fmap_c_offset = c * H * W;
            const int gmap_c_offset = gmap_m_offset + c * D * D;

            for (int ii = 0; ii < D; ii++) {
                const int y = cy + ii - radius;
                if ((unsigned)y >= (unsigned)H) {
                    if (m < 3 && c == 0 && ii == 0) {
                        printf("[patchify_cpu_safe] Patch %d, channel %d: y=%d out of bounds (H=%d)\n", m, c, y, H);
                        fflush(stdout);
                    }
                    continue;
                }

                for (int jj = 0; jj < D; jj++) {
                    const int x = cx + jj - radius;
                    if ((unsigned)x >= (unsigned)W) {
                        if (m < 3 && c == 0 && ii == 0 && jj == 0) {
                            printf("[patchify_cpu_safe] Patch %d, channel %d: x=%d out of bounds (W=%d)\n", m, c, x, W);
                            fflush(stdout);
                        }
                        continue;
                    }

                    int src_idx = fmap_c_offset + y * W + x;
                    int dst_idx = gmap_c_offset + ii * D + jj;
                    
                    // Debug first few samples
                    if (m < 3 && c == 0 && ii == 0 && jj == 0) {
                        printf("[patchify_cpu_safe] Patch %d, channel %d: src[%d]=%f -> gmap[%d]\n",
                               m, c, src_idx, src[src_idx], dst_idx);
                        fflush(stdout);
                    }
                    
                    gmap[dst_idx] = src[src_idx];
                }
            }
        }
    }
}


// inline bool within_bounds(int y, int x, int H, int W) {
//     return y >= 0 && y < H && x >= 0 && x < W;
// }

// -----------------------------------------------------------------------------
// Compute correlation between patch features (gmap) and frame features (pyramid)
// -----------------------------------------------------------------------------
// Purpose: Computes correlation volumes between patch features extracted from source frames
//          and frame features at reprojected locations in target frames. This is used for
//          visual odometry to match patches across frames.
//
// Algorithm:
//   For each active edge e:
//     1. Extract patch features from gmap (source frame, patch index from kk[e])
//     2. Get reprojected coordinates for target frame (from coords)
//     3. For each pixel (i0, j0) in patch and each offset (ii, jj) in correlation window:
//        - Sample frame features at reprojected location + offset
//        - Compute dot product between patch feature and frame feature over all channels
//        - Store correlation value
//     4. Process two pyramid levels: pyramid0 (1/4 res) and pyramid1 (1/16 res)
//
// Input Parameters:
//   gmap: [num_gmap_frames * M * feature_dim * D_gmap * D_gmap] - Ring buffer of patch features
//         Layout: [frame][patch][channel][y][x]
//         - num_gmap_frames: Number of frames in ring buffer (e.g., m_pmem = 36)
//         - M: Patches per frame (e.g., 4 or 8)
//         - feature_dim: Feature dimension (128 for FNet features)
//         - D_gmap: Patch dimension = 4 (from patchify_cpu_safe with radius=1, m_patch_size=3)
//         - Contains patches extracted from source frames using patchify_cpu_safe
//
//   pyramid0: [num_frames * feature_dim * fmap1_H * fmap1_W] - Full resolution feature pyramid
//             Layout: [frame][channel][y][x]
//             - num_frames: Number of frames in pyramid buffer (e.g., m_mem = 36)
//             - feature_dim: Feature dimension (128 for FNet features)
//             - fmap1_H, fmap1_W: Feature map dimensions at 1/4 resolution (e.g., 132x240)
//             - Used for correlation channel 0 (coords scaled by 1.0)
//
//   pyramid1: [num_frames * feature_dim * fmap2_H * fmap2_W] - 1/4 resolution feature pyramid
//             Layout: [frame][channel][y][x]
//             - Same structure as pyramid0 but at 1/16 resolution (fmap2_H, fmap2_W)
//             - Used for correlation channel 1 (coords scaled by 0.25)
//
//   coords: [num_active * 2 * P * P] - Reprojected 2D coordinates
//           Layout: [edge][channel][y][x] where channel 0=x, channel 1=y
//           - num_active: Number of active edges (patch-frame pairs)
//           - P: Patch size (typically 3)
//           - Coordinates are at 1/4 resolution (from reproject function)
//           - Used to sample frame features at reprojected locations
//
//   ii: [num_active] - Source patch indices within frame (NOT USED in current implementation)
//       Kept for compatibility with Python/CUDA interface
//
//   jj: [num_active] - Target frame indices for pyramid buffers
//       Indicates which frame in pyramid0/pyramid1 to sample from
//       Range: [0, num_frames-1]
//
//   kk: [num_active] - Linear patch indices for gmap extraction
//       Encodes: kk[e] = gmap_frame * M + patch_idx
//       - gmap_frame = kk[e] / M (which frame in gmap ring buffer)
//       - patch_idx = kk[e] % M (which patch within that frame)
//       Range: [0, num_gmap_frames * M - 1]
//
//   num_active: Number of active edges to process
//
//   M: Patches per frame (PATCHES_PER_FRAME, typically 4 or 8)
//
//   P: Patch size (typically 3)
//
//   num_frames: Number of frames in pyramid buffers (e.g., m_mem = 36)
//
//   num_gmap_frames: Number of frames in gmap ring buffer (e.g., m_pmem = 36)
//
//   fmap1_H, fmap1_W: Feature map dimensions for pyramid0 at 1/4 resolution
//                     (e.g., 132x240 for 528x960 input)
//
//   fmap2_H, fmap2_W: Feature map dimensions for pyramid1 at 1/16 resolution
//                     (e.g., 33x60 for 528x960 input)
//
//   feature_dim: Feature dimension (128 for FNet features)
//
// Output Parameters:
//   corr_out: [num_active * D * D * P * P * 2] - Correlation volumes
//             Layout: [edge][corr_y][corr_x][patch_y][patch_x][channel]
//             - D: Correlation window diameter = 8 (R=3, D = 2*R + 2)
//             - P: Patch size (typically 3)
//             - Channel 0: Correlation with pyramid0 (1/4 resolution)
//             - Channel 1: Correlation with pyramid1 (1/16 resolution)
//             - Each value is dot product between patch feature and frame feature
//
// Correlation Window:
//   - Radius R = 3 (searches ±3 pixels around reprojected location)
//   - Window size D = 2*R + 2 = 8
//   - For each pixel in patch, computes correlation at 8×8 offsets
//
// Coordinate Scaling:
//   - Reprojected coords are at 1/4 resolution
//   - For pyramid0 (1/4 res): coords used directly (scale = 1.0)
//   - For pyramid1 (1/16 res): coords scaled by 0.25 (coords / 4)
//
// Note: Matches Python CUDA kernel corr_forward_kernel behavior
// -----------------------------------------------------------------------------
void computeCorrelation(
    const float* gmap,           // [num_gmap_frames, M, feature_dim, D_gmap, D_gmap] - Patch features ring buffer
    const float* pyramid0,      // [num_frames, feature_dim, fmap1_H, fmap1_W] - Full-res feature pyramid
    const float* pyramid1,      // [num_frames, feature_dim, fmap2_H, fmap2_W] - 1/4-res feature pyramid
    const float* coords,        // [num_active, 2, P, P] - Reprojected (u, v) coordinates
    const int* ii,              // [num_active] - Source patch indices (NOT USED, kept for compatibility)
    const int* jj,              // [num_active] - Target frame indices for pyramid
    const int* kk,              // [num_active] - Linear patch indices (gmap_frame * M + patch_idx)
    int num_active,             // Number of active edges to process
    int M,                      // Patches per frame (PATCHES_PER_FRAME)
    int P,                      // Patch size (typically 3)
    int num_frames,             // Number of frames in pyramid buffers (e.g., m_mem)
    int num_gmap_frames,        // Number of frames in gmap ring buffer (e.g., m_pmem)
    int fmap1_H, int fmap1_W,   // Dimensions for pyramid0 (1/4 resolution)
    int fmap2_H, int fmap2_W,   // Dimensions for pyramid1 (1/16 resolution)
    int feature_dim,            // Feature dimension (128 for FNet)
    float* corr_out)            // Output: [num_active, D, D, P, P, 2] - Correlation volumes
{
    // Translated from CUDA corr_forward_kernel
    // CUDA signature: corr_forward_kernel(int R, fmap1, fmap2, coords, us, vs, corr)
    
    // Validate inputs
    if (gmap == nullptr || pyramid0 == nullptr || pyramid1 == nullptr || 
        coords == nullptr || ii == nullptr || jj == nullptr || kk == nullptr || corr_out == nullptr) {
        printf("[computeCorrelation] ERROR: Null pointer in inputs\n");
        fflush(stdout);
        return;
    }
    
    if (num_active <= 0 || M <= 0 || P <= 0 || num_frames <= 0 || num_gmap_frames <= 0) {
        printf("[computeCorrelation] ERROR: Invalid dimensions\n");
        fflush(stdout);
        return;
    }
    
    // Correlation radius R (typically 3 in DPVO)
    const int R = 3;
    const int D = 2 * R + 2;  // Correlation window diameter (D = 8 for R=3)
    
    // gmap structure: created by patchify_cpu_safe with radius=1, so D_gmap=4
    const int D_gmap = 4;
    const int gmap_center_offset = (D_gmap - P) / 2;  // Center the P×P region within D_gmap×D_gmap
    
    // Calculate buffer sizes
    const size_t gmap_total_size = static_cast<size_t>(num_gmap_frames) * M * feature_dim * D_gmap * D_gmap;
    const size_t fmap1_total_size = static_cast<size_t>(num_frames) * feature_dim * fmap1_H * fmap1_W;
    const size_t fmap2_total_size = static_cast<size_t>(num_frames) * feature_dim * fmap2_H * fmap2_W;
    const size_t corr_total_size = static_cast<size_t>(num_active) * D * D * P * P * 2;
    
    // Zero output (matches CUDA behavior)
    std::memset(corr_out, 0, sizeof(float) * corr_total_size);
    
    // Main loop: For each active edge (equivalent to CUDA's B * M * H * W * D * D threads)
    for (int e = 0; e < num_active; e++) {
        // Get patch and frame indices (equivalent to CUDA's us[m] and vs[m])
        // Python: ii1 = kk % (M * pmem), jj1 = jj % mem
        int mod_value = M * num_gmap_frames;
        if (mod_value == 0) {
            printf("[computeCorrelation] ERROR: Division by zero! M=%d, num_gmap_frames=%d\n", M, num_gmap_frames);
            fflush(stdout);
            continue;
        }
        
        int ii1 = kk[e] % mod_value;  // Linear patch index in gmap (CUDA's us[m])
        int gmap_frame = ii1 / M;
        int patch_idx = ii1 % M;
        int jj1 = jj[e] % num_frames;  // Frame index in pyramid (CUDA's vs[m])
        int pyramid_frame = jj1;
        
        // Validate indices
        if (patch_idx < 0 || patch_idx >= M || 
            gmap_frame < 0 || gmap_frame >= num_gmap_frames ||
            pyramid_frame < 0 || pyramid_frame >= num_frames) {
            continue;
        }
        
        // Process two correlation channels: pyramid0 (fmap1) and pyramid1 (fmap2)
        // CUDA: fmap1 = patch features, fmap2 = frame features
        // Python: corr1 = altcorr.corr(..., coords / 1, ...) for pyramid[0]
        //         corr2 = altcorr.corr(..., coords / 4, ...) for pyramid[1]
        // Reprojected coordinates are at 1/4 resolution (intrinsics scaled by RES=4)
        // fmap1 is at 1/4 resolution, fmap2 is at 1/16 resolution
        for (int c = 0; c < 2; c++) {
            const float* fmap2 = (c == 0) ? pyramid0 : pyramid1;  // Frame features (target)
            // Python: coords / 1 for pyramid[0], coords / 4 for pyramid[1]
            // Reprojected coords are at 1/4 resolution, so:
            // - For fmap1 (1/4 res): scale = 1.0f (coords / 1)
            // - For fmap2 (1/16 res): scale = 0.25f (coords / 4)
            float scale = (c == 0) ? 1.0f : 0.25f;
            int fmap_H = (c == 0) ? fmap1_H : fmap2_H;
            int fmap_W = (c == 0) ? fmap1_W : fmap2_W;
            
            // For each pixel in the patch (i0, j0) - equivalent to CUDA's H * W loop
            for (int i0 = 0; i0 < P; i0++) {
                for (int j0 = 0; j0 < P; j0++) {
                    // Get reprojected coordinate for target frame (fmap2)
                    // Python: coords / 1 for c=0, coords / 4 for c=1
                    int coord_x_idx = e * 2 * P * P + 0 * P * P + i0 * P + j0;
                    int coord_y_idx = e * 2 * P * P + 1 * P * P + i0 * P + j0;
                    float x = coords[coord_x_idx] * scale;  // scale = 1.0f for c=0, 0.25f for c=1
                    float y = coords[coord_y_idx] * scale;  // scale = 1.0f for c=0, 0.25f for c=1
                    
                    // For each offset in correlation window (ii, jj) - equivalent to CUDA's D * D loop
                    for (int corr_ii = 0; corr_ii < D; corr_ii++) {
                        for (int corr_jj = 0; corr_jj < D; corr_jj++) {
                            // Calculate sampling location in target frame (fmap2)
                            // CUDA: i1 = floor(y) + (ii - R), j1 = floor(x) + (jj - R)
                            int i1 = static_cast<int>(std::floor(y)) + (corr_ii - R);
                            int j1 = static_cast<int>(std::floor(x)) + (corr_jj - R);
                            
                            // Compute correlation: dot product over features (matches CUDA)
                            // CUDA: f1 = fmap1[n][ix][c][i0][j0], f2 = fmap2[n][jx][c][i1][j1]
                            // fmap1 = patch features (from gmap, which contains patches from pyramid)
                            // fmap2 = frame features from pyramid at reprojected location
                            float sum = 0.0f;
                            if (within_bounds(i1, j1, fmap_H, fmap_W)) {
                                // Extract patch feature from gmap (fmap1 equivalent)
                                // gmap layout: [num_gmap_frames][M][feature_dim][D_gmap][D_gmap]
                                // Extract center P×P region: gmap_i = i0 + offset, gmap_j = j0 + offset
                                int gmap_i = i0 + gmap_center_offset;
                                int gmap_j = j0 + gmap_center_offset;
                                
                                // Dot product over feature channels (matches CUDA's loop over C)
                                for (int f = 0; f < feature_dim; f++) {
                                    // fmap1: patch feature from gmap
                                    // CUDA: fmap1[n][ix][c][i0][j0] where ix = us[m] (patch index)
                                    // gmap already contains patches extracted from pyramid, so use it as fmap1
                                    size_t fmap1_idx = static_cast<size_t>(gmap_frame) * M * feature_dim * D_gmap * D_gmap +
                                                       static_cast<size_t>(patch_idx) * feature_dim * D_gmap * D_gmap +
                                                       static_cast<size_t>(f) * D_gmap * D_gmap +
                                                       static_cast<size_t>(gmap_i) * D_gmap + static_cast<size_t>(gmap_j);
                                    
                                    // fmap2: frame feature from pyramid at reprojected location
                                    // CUDA: fmap2[n][jx][c][i1][j1] where jx = vs[m] (frame index)
                                    size_t fmap2_idx = static_cast<size_t>(pyramid_frame) * feature_dim * fmap_H * fmap_W +
                                                       static_cast<size_t>(f) * fmap_H * fmap_W +
                                                       static_cast<size_t>(i1) * fmap_W + static_cast<size_t>(j1);
                                    
                                    // Bounds check and compute correlation
                                    if (fmap1_idx < gmap_total_size && fmap2_idx < (c == 0 ? fmap1_total_size : fmap2_total_size)) {
                                        // CUDA: s += f1 * f2
                                        float f1 = gmap[fmap1_idx];  // Patch feature (fmap1) - from gmap which contains patches
                                        float f2 = fmap2[fmap2_idx]; // Frame feature (fmap2) - from pyramid
                                        sum += f1 * f2;
                                    }
                                }
                            }
                            // If out of bounds, sum remains 0.0f (matches CUDA)
                            
                            // Store correlation (matches CUDA's corr[n][m][ii][jj][i0][j0])
                            // Output layout: [num_active, D, D, P, P, 2] (channel last)
                            size_t out_idx = static_cast<size_t>(e) * D * D * P * P * 2 +
                                             static_cast<size_t>(corr_ii) * D * P * P * 2 +
                                             static_cast<size_t>(corr_jj) * P * P * 2 +
                                             static_cast<size_t>(i0) * P * 2 +
                                             static_cast<size_t>(j0) * 2 +
                                             static_cast<size_t>(c);
                            
                            if (out_idx < corr_total_size) {
                                corr_out[out_idx] = sum;
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Get logger
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    // Log statistics for entire correlation output
    int zero_count = 0;
    int nonzero_count = 0;
    float min_corr = std::numeric_limits<float>::max();
    float max_corr = std::numeric_limits<float>::lowest();
    double sum_corr = 0.0;
    const int sample_count = 20;  // Number of sample values to show
    
    // Count over entire output
    for (size_t i = 0; i < corr_total_size; i++) {
        float val = corr_out[i];
        if (val == 0.0f) {
            zero_count++;
        } else {
            nonzero_count++;
        }
        if (val < min_corr) min_corr = val;
        if (val > max_corr) max_corr = val;
        sum_corr += val;
    }
    
    float mean_corr = corr_total_size > 0 ? static_cast<float>(sum_corr / corr_total_size) : 0.0f;
    
    // Log comprehensive statistics
    logger->info("computeCorrelation: Output statistics - size={}, zero_count={}, nonzero_count={}, min={:.6f}, max={:.6f}, mean={:.6f}",
                 corr_total_size, zero_count, nonzero_count, min_corr, max_corr, mean_corr);
    
    // Log sample values from different parts of the output
    std::string sample_values = "[";
    for (int i = 0; i < sample_count && i < static_cast<int>(corr_total_size); i++) {
        sample_values += std::to_string(corr_out[i]);
        if (i < sample_count - 1 && i < static_cast<int>(corr_total_size) - 1) {
            sample_values += ", ";
        }
    }
    sample_values += "]";
    logger->info("computeCorrelation: First {} sample values: {}", sample_count, sample_values);
    
    // Log some values from middle and end of output
    if (corr_total_size > sample_count * 2) {
        std::string mid_values = "[";
        int mid_start = static_cast<int>(corr_total_size / 2);
        for (int i = 0; i < sample_count && (mid_start + i) < static_cast<int>(corr_total_size); i++) {
            mid_values += std::to_string(corr_out[mid_start + i]);
            if (i < sample_count - 1 && (mid_start + i + 1) < static_cast<int>(corr_total_size)) {
                mid_values += ", ";
            }
        }
        mid_values += "]";
        logger->info("computeCorrelation: Middle {} sample values (starting at index {}): {}", 
                     sample_count, mid_start, mid_values);
    }
    
    // Log per-edge statistics for first few edges
    if (num_active > 0) {
        int edges_to_log = std::min(3, num_active);
        for (int e = 0; e < edges_to_log; e++) {
            int edge_zero = 0;
            int edge_nonzero = 0;
            float edge_min = std::numeric_limits<float>::max();
            float edge_max = std::numeric_limits<float>::lowest();
            double edge_sum = 0.0;
            
            size_t edge_start = static_cast<size_t>(e) * D * D * P * P * 2;
            size_t edge_size = D * D * P * P * 2;
            
            for (size_t i = 0; i < edge_size && (edge_start + i) < corr_total_size; i++) {
                float val = corr_out[edge_start + i];
                if (val == 0.0f) {
                    edge_zero++;
                } else {
                    edge_nonzero++;
                }
                if (val < edge_min) edge_min = val;
                if (val > edge_max) edge_max = val;
                edge_sum += val;
            }
            
            float edge_mean = edge_size > 0 ? static_cast<float>(edge_sum / edge_size) : 0.0f;
            logger->info("computeCorrelation: Edge[{}] stats - zero={}, nonzero={}, min={:.6f}, max={:.6f}, mean={:.6f}",
                         e, edge_zero, edge_nonzero, edge_min, edge_max, edge_mean);
        }
    }
    
    logger->info("computeCorrelation: COMPLETED - processed {} edges", num_active);
}
