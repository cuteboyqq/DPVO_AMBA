#include "correlation_kernel.hpp"
#include "correlation_bilinear_helpers.hpp"  // Bilinear interpolation helpers for grid_sample matching
#include "correlation_file_io.hpp"  // For saving 8x8 buffer
#include "target_frame.hpp"  // TARGET_FRAME constant
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>
#include <limits>
#include <chrono>
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
    // Formula matches Python altcorr.patchify behavior:
    // - radius=0 -> D=1 (single pixel extraction)
    // - radius>0 -> D = 2*radius + 1 (patch extraction)
    const int D = (radius == 0) ? 1 : (2 * radius + 1);

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
    // Formula matches Python altcorr.patchify behavior:
    // - radius=0 -> D=1 (single pixel extraction)
    // - radius>0 -> D = 2*radius + 1 (patch extraction)
    //   Example: radius=1 (P//2 where P=3) -> D=3, producing [M, C, 3, 3] patches
    //   This matches Python: altcorr.patchify(..., P//2).view(..., P, P) where P=3
    const int D = (radius == 0) ? 1 : (2 * radius + 1);
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
//         - D_gmap: Patch dimension = 3 (from patchify_cpu_safe with radius=1, matches Python altcorr.patchify)
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
//   - Window size D = 2*R + 1 = 7 (matches Python's final output after bilinear interpolation)
//   - For each pixel in patch, computes correlation at 7×7 offsets
//   - Note: Python's CUDA kernel uses D = 2*R + 2 = 8 internally, then reduces to D = 7 via bilinear interpolation
//
// Coordinate Scaling:
//   - Reprojected coords are at 1/4 resolution
//   - For pyramid0 (1/4 res): coords used directly (scale = 1.0)
//   - For pyramid1 (1/16 res): coords scaled by 0.25 (coords / 4)
//
// Note: Matches Python CUDA kernel corr_forward_kernel behavior
// -----------------------------------------------------------------------------

// Single pyramid level correlation (matches Python: altcorr.corr)
// This function computes correlation for one pyramid level, matching Python's altcorr.corr call
// Output: [num_active, D, D, P, P] (matches CUDA kernel output before permute)
void computeCorrelationSingle(
    const float* gmap,           // [num_gmap_frames, M, feature_dim, D_gmap, D_gmap] - Patch features ring buffer
    const float* pyramid,         // [num_frames, feature_dim, fmap_H, fmap_W] - Frame features pyramid
    const float* coords,         // [num_active, 2, P, P] - Reprojected (u, v) coordinates
    const int* ii1,              // [num_active] - Patch indices for gmap (mapped from kk: ii1 = kk % (M * pmem))
    const int* jj1,              // [num_active] - Frame indices for pyramid (mapped from jj: jj1 = jj % mem)
    int num_active,              // Number of active edges to process
    int M,                       // Patches per frame (PATCHES_PER_FRAME)
    int P,                       // Patch size (typically 3)
    int num_frames,              // Number of frames in pyramid buffers (e.g., m_mem)
    int num_gmap_frames,         // Number of frames in gmap ring buffer (e.g., m_pmem)
    int fmap_H, int fmap_W,      // Dimensions for pyramid
    int feature_dim,             // Feature dimension (128 for FNet)
    float coord_scale,          // Scale factor for coordinates (1.0 for pyramid0, 0.25 for pyramid1)
    int radius,                  // Correlation radius (typically 3)
    float* corr_out,             // Output: [num_active, D, D, P, P]
    float* corr_8x8_out)         // Optional: Output 8x8 internal buffer [num_active, 8, 8, P, P] for debugging
{
    // Translated from CUDA corr_forward_kernel
    // CUDA signature: corr_forward_kernel(int R, fmap1, fmap2, coords, us, vs, corr)
    // Python: altcorr.corr(fmap1, fmap2, coords, ii1, jj1, radius)
    
    // Validate inputs
    if (gmap == nullptr || pyramid == nullptr || coords == nullptr || 
        ii1 == nullptr || jj1 == nullptr || corr_out == nullptr) {
        printf("[computeCorrelationSingle] ERROR: Null pointer in inputs\n");
        fflush(stdout);
        return;
    }
    
    if (num_active <= 0 || M <= 0 || P <= 0 || num_frames <= 0 || num_gmap_frames <= 0) {
        printf("[computeCorrelationSingle] ERROR: Invalid dimensions\n");
        fflush(stdout);
        return;
    }
    
    const int R = radius;
    // Match Python's corr_torch_forward_fp16: compute 8x8 correlation first, then reduce to 7x7 via bilinear wrapper
    const int D_internal = 2 * R + 2;  // Internal correlation window (D = 8 for R=3, matches Python's 8x8 computation)
    const int D_output = 2 * R + 1;    // Final output size (D = 7 for R=3, matches Python's final 7x7 output)
    
    // gmap structure: created by patchify_cpu_safe with radius=1, so D_gmap=3 (matches Python)
    const int D_gmap = 3;  // D_gmap = 2*radius + 1 = 3 (matches Python: .view(..., P, P) where P=3)
    const int gmap_center_offset = (D_gmap - P) / 2;  // Center the P×P region within D_gmap×D_gmap (0 when D_gmap=P=3)
    
    // Calculate buffer sizes
    const size_t gmap_total_size = static_cast<size_t>(num_gmap_frames) * M * feature_dim * D_gmap * D_gmap;
    const size_t pyramid_total_size = static_cast<size_t>(num_frames) * feature_dim * fmap_H * fmap_W;
    const size_t corr_internal_size = static_cast<size_t>(num_active) * D_internal * D_internal * P * P;
    const size_t corr_output_size = static_cast<size_t>(num_active) * D_output * D_output * P * P;
    
    // Allocate temporary buffer for 8x8 correlation
    std::vector<float> corr_internal(corr_internal_size, 0.0f);
    
    // Zero output (matches CUDA behavior)
    std::memset(corr_out, 0, sizeof(float) * corr_output_size);
    
    // Diagnostic logging setup
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    // Diagnostic: Check input coords array for NaN/Inf values
    if (logger) {
        int coords_total_size = num_active * 2 * P * P;
        int nan_count = 0;
        int inf_count = 0;
        int valid_count = 0;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        // Sample first edge and a few others
        for (int e = 0; e < std::min(num_active, 3); e++) {
            for (int i0 = 0; i0 < P; i0++) {
                for (int j0 = 0; j0 < P; j0++) {
                    int coord_x_idx = e * 2 * P * P + 0 * P * P + i0 * P + j0;
                    int coord_y_idx = e * 2 * P * P + 1 * P * P + i0 * P + j0;
                    if (coord_x_idx < coords_total_size && coord_y_idx < coords_total_size) {
                        float x = coords[coord_x_idx];
                        float y = coords[coord_y_idx];
                        if (!std::isfinite(x)) {
                            if (std::isnan(x)) nan_count++;
                            else if (std::isinf(x)) inf_count++;
                        } else {
                            valid_count++;
                            min_val = std::min(min_val, x);
                            max_val = std::max(max_val, x);
                        }
                        if (!std::isfinite(y)) {
                            if (std::isnan(y)) nan_count++;
                            else if (std::isinf(y)) inf_count++;
                        } else {
                            valid_count++;
                            min_val = std::min(min_val, y);
                            max_val = std::max(max_val, y);
                        }
                    }
                }
            }
        }
    }
    
    // Main loop: For each active edge (equivalent to CUDA's B * M * H * W * D * D threads)
    for (int e = 0; e < num_active; e++) {
        // Get patch and frame indices (equivalent to CUDA's us[m] and vs[m])
        // ii1[e] and jj1[e] are already mapped (from Python: ii1 = kk % (M * pmem), jj1 = jj % mem)
        int ii1_val = ii1[e];
        int jj1_val = jj1[e];
        
        // Extract gmap frame and patch index from ii1
        int gmap_frame = ii1_val / M;
        int patch_idx = ii1_val % M;
        int pyramid_frame = jj1_val;
        
        // Validate indices
        if (patch_idx < 0 || patch_idx >= M || 
            gmap_frame < 0 || gmap_frame >= num_gmap_frames ||
            pyramid_frame < 0 || pyramid_frame >= num_frames) {
            continue;
        }
        
        // For each pixel in the patch (i0, j0) - equivalent to CUDA's H * W loop
        for (int i0 = 0; i0 < P; i0++) {
            for (int j0 = 0; j0 < P; j0++) {
                // Get reprojected coordinate for target frame (scaled)
                int coord_x_idx = e * 2 * P * P + 0 * P * P + i0 * P + j0;
                int coord_y_idx = e * 2 * P * P + 1 * P * P + i0 * P + j0;
                
                // Validate indices
                int coords_total_size = num_active * 2 * P * P;
                if (coord_x_idx < 0 || coord_x_idx >= coords_total_size || 
                    coord_y_idx < 0 || coord_y_idx >= coords_total_size) {
                    continue;  // Skip this pixel if indices are invalid
                }
                
                float raw_x = coords[coord_x_idx];
                float raw_y = coords[coord_y_idx];
                
                // Check if coordinates are NaN/Inf before scaling
                bool is_nan_before_scale = !std::isfinite(raw_x) || !std::isfinite(raw_y);
                
                float x = raw_x * coord_scale;
                float y = raw_y * coord_scale;
                
                // Check if coordinates are NaN/Inf after scaling
                bool is_nan_after_scale = !std::isfinite(x) || !std::isfinite(y);
                
                // Skip this pixel if coordinates are invalid
                if (is_nan_after_scale) {
                    continue;  // Skip correlation computation for invalid coordinates
                }
                
                // Match Python's corr_torch_forward_fp16: convert coordinates to half precision BEFORE floor
                // Python does: coords = coords.half(), then x0 = torch.floor(x)
                // This is critical for matching Python's behavior, especially for coordinates very close to integers
                // (e.g., 30.999988556 becomes 31.000000 after half precision, changing floor from 30 to 31)
                float x_half = float_to_half_to_float(x);
                float y_half = float_to_half_to_float(y);
                
                // Check if half-precision conversion produced infinity (huge out-of-bounds coordinates overflow)
                // Python's grid_sample with infinity coordinates produces NaN, so we skip these cases
                if (!std::isfinite(x_half) || !std::isfinite(y_half)) {
                    continue;  // Skip correlation computation for coordinates that overflow to infinity
                }
                
                // Match Python's corr_torch_forward_fp16: compute 8x8 correlation at integer offsets first
                float x0 = std::floor(x_half);
                float y0 = std::floor(y_half);
                        
                // Step 1: Compute 8x8 correlation at integer offsets (matching Python's internal computation)
                // Python uses offsets: torch.arange(-radius, radius + 2) = [-3, -2, -1, 0, 1, 2, 3, 4]
                for (int corr_ii = 0; corr_ii < D_internal; corr_ii++) {
                    for (int corr_jj = 0; corr_jj < D_internal; corr_jj++) {
                        // Calculate correlation window offset (in pixels) - integer offsets
                        float offset_x = static_cast<float>(corr_jj - R);
                        float offset_y = static_cast<float>(corr_ii - R);
                        
                        // Add offset in pixel space (matching Python: gx = x0 + ox)
                        float gx = x0 + offset_x;
                        float gy = y0 + offset_y;
                        
                        // Normalize coordinates to [-1, 1] range (matching Python: gx = 2 * gx / (W2 - 1) - 1)
                        // Python uses align_corners=True, which uses the same normalization formula
                        float x_norm, y_norm;
                        normalize_coords_for_grid_sample(gx, gy, fmap_H, fmap_W, x_norm, y_norm);
                        
                        bool is_center = (corr_ii == R && corr_jj == R);
                        
                        // Compute correlation: dot product over features using bilinear interpolation
                        // OPTIMIZATION: Precompute bilinear weights and corner addresses ONCE,
                        // then reuse them for all 128 feature channels (was computing 128x redundantly)
                        float sum = 0.0f;
                        
                        // Extract patch feature from gmap (unchanged)
                            int gmap_i = i0 + gmap_center_offset;
                            int gmap_j = j0 + gmap_center_offset;
                        
                        // Precompute bilinear sampling parameters (shared across all channels)
                        float x_pixel_bi = (x_norm + 1.0f) * 0.5f * static_cast<float>(fmap_W - 1);
                        float y_pixel_bi = (y_norm + 1.0f) * 0.5f * static_cast<float>(fmap_H - 1);
                        
                        const float bi_tolerance = 0.5f;
                        bool bi_oob = (x_pixel_bi < -bi_tolerance || x_pixel_bi > static_cast<float>(fmap_W - 1) + bi_tolerance ||
                                       y_pixel_bi < -bi_tolerance || y_pixel_bi > static_cast<float>(fmap_H - 1) + bi_tolerance);
                        
                        if (!bi_oob) {
                            int bx0 = static_cast<int>(std::floor(x_pixel_bi));
                            int by0 = static_cast<int>(std::floor(y_pixel_bi));
                            bx0 = std::max(0, std::min(bx0, fmap_W - 1));
                            by0 = std::max(0, std::min(by0, fmap_H - 1));
                            int bx1 = std::min(bx0 + 1, fmap_W - 1);
                            int by1 = std::min(by0 + 1, fmap_H - 1);
                            
                            x_pixel_bi = std::max(0.0f, std::min(x_pixel_bi, static_cast<float>(fmap_W - 1)));
                            y_pixel_bi = std::max(0.0f, std::min(y_pixel_bi, static_cast<float>(fmap_H - 1)));
                            float bdx = x_pixel_bi - static_cast<float>(bx0);
                            float bdy = y_pixel_bi - static_cast<float>(by0);
                            
                            float bw00 = (1.0f - bdx) * (1.0f - bdy);
                            float bw01 = bdx * (1.0f - bdy);
                            float bw10 = (1.0f - bdx) * bdy;
                            float bw11 = bdx * bdy;
                            
                            // Precompute base offsets for pyramid corners (stride = H * W per channel)
                            size_t frame_offset = static_cast<size_t>(pyramid_frame) * static_cast<size_t>(feature_dim) * static_cast<size_t>(fmap_H) * static_cast<size_t>(fmap_W);
                            size_t hw = static_cast<size_t>(fmap_H) * static_cast<size_t>(fmap_W);
                            size_t off00 = frame_offset + static_cast<size_t>(by0) * fmap_W + bx0;
                            size_t off01 = frame_offset + static_cast<size_t>(by0) * fmap_W + bx1;
                            size_t off10 = frame_offset + static_cast<size_t>(by1) * fmap_W + bx0;
                            size_t off11 = frame_offset + static_cast<size_t>(by1) * fmap_W + bx1;
                            
                            // Gmap base offset (stride = D_gmap * D_gmap per channel)
                            size_t gmap_base = static_cast<size_t>(gmap_frame) * M * feature_dim * D_gmap * D_gmap +
                                               static_cast<size_t>(patch_idx) * feature_dim * D_gmap * D_gmap +
                                               static_cast<size_t>(gmap_i) * D_gmap + static_cast<size_t>(gmap_j);
                            size_t gmap_stride = static_cast<size_t>(D_gmap) * D_gmap;
                            
                            // Dot product over all 128 feature channels (tight inner loop)
                            for (int f = 0; f < feature_dim; f++) {
                                size_t fmap1_idx = gmap_base + static_cast<size_t>(f) * gmap_stride;
                                if (fmap1_idx >= gmap_total_size) continue;
                                float f1 = gmap[fmap1_idx];
                                
                                size_t ch_off = static_cast<size_t>(f) * hw;
                                float f2 = bw00 * pyramid[off00 + ch_off]
                                         + bw01 * pyramid[off01 + ch_off]
                                         + bw10 * pyramid[off10 + ch_off]
                                         + bw11 * pyramid[off11 + ch_off];
                                
                                sum += f1 * f2;
                            }
                        }
                        // If bi_oob, sum stays 0.0 (matches grid_sample padding_mode='zeros')
                        
                        // Store correlation in internal 8x8 buffer
                        // Layout: [num_active, D_internal, D_internal, P, P]
                        // Match Python's permute(0,1,3,2,4,5): swap corr_ii and corr_jj to get [corr_x, corr_y] order
                        // So store as [e, corr_jj, corr_ii, i0, j0] instead of [e, corr_ii, corr_jj, i0, j0]
                        size_t internal_idx = static_cast<size_t>(e) * D_internal * D_internal * P * P +
                                              static_cast<size_t>(corr_jj) * D_internal * P * P +
                                              static_cast<size_t>(corr_ii) * P * P +
                                         static_cast<size_t>(i0) * P +
                                         static_cast<size_t>(j0);
                        
                        if (internal_idx < corr_internal_size) {
                            corr_internal[internal_idx] = sum;
                        }
                    }
                }
                
                // Step 2: Apply bilinear wrapper interpolation to reduce from 8x8 to 7x7 (matching Python)
                // Python's formula: out[i,j] = (1-dx)*(1-dy)*corr[i,j] + dx*(1-dy)*corr[i,j+1] + (1-dx)*dy*corr[i+1,j] + dx*dy*corr[i+1,j+1]
                // Python converts coords to half precision first, then computes dx/dy from half-precision coords
                // dx = (coords[:, :, 0] - torch.floor(coords[:, :, 0])).half()
                // So we need to compute dx/dy from half-precision coordinates
                float dx_raw = x_half - x0;  // Fractional part of x (from half-precision x)
                float dy_raw = y_half - y0;  // Fractional part of y (from half-precision y)
                
                // Convert to half precision and back to float32 (matching Python's .half() conversion)
                float dx = float_to_half_to_float(dx_raw);
                float dy = float_to_half_to_float(dy_raw);
                
                for (int out_ii = 0; out_ii < D_output; out_ii++) {
                    for (int out_jj = 0; out_jj < D_output; out_jj++) {
                        // Bilinear interpolation from 8x8 internal buffer
                        // Internal buffer is stored as [e, corr_jj, corr_ii, i0, j0] (swapped to match Python's final output)
                        // Interpolate from: corr[out_jj, out_ii], corr[out_jj+1, out_ii], corr[out_jj, out_ii+1], corr[out_jj+1, out_ii+1]
                        size_t idx00 = static_cast<size_t>(e) * D_internal * D_internal * P * P +
                                       static_cast<size_t>(out_jj) * D_internal * P * P +
                                       static_cast<size_t>(out_ii) * P * P +
                                       static_cast<size_t>(i0) * P + static_cast<size_t>(j0);
                        size_t idx01 = static_cast<size_t>(e) * D_internal * D_internal * P * P +
                                       static_cast<size_t>(out_jj + 1) * D_internal * P * P +
                                       static_cast<size_t>(out_ii) * P * P +
                                       static_cast<size_t>(i0) * P + static_cast<size_t>(j0);
                        size_t idx10 = static_cast<size_t>(e) * D_internal * D_internal * P * P +
                                       static_cast<size_t>(out_jj) * D_internal * P * P +
                                       static_cast<size_t>(out_ii + 1) * P * P +
                                       static_cast<size_t>(i0) * P + static_cast<size_t>(j0);
                        size_t idx11 = static_cast<size_t>(e) * D_internal * D_internal * P * P +
                                       static_cast<size_t>(out_jj + 1) * D_internal * P * P +
                                       static_cast<size_t>(out_ii + 1) * P * P +
                                       static_cast<size_t>(i0) * P + static_cast<size_t>(j0);
                        
                        // Bilinear interpolation weights
                        float w00 = (1.0f - dx) * (1.0f - dy);
                        float w01 = dx * (1.0f - dy);
                        float w10 = (1.0f - dx) * dy;
                        float w11 = dx * dy;
                        
                        // Get values from internal buffer (with bounds checking)
                        float v00 = (idx00 < corr_internal_size) ? corr_internal[idx00] : 0.0f;
                        float v01 = (idx01 < corr_internal_size) ? corr_internal[idx01] : 0.0f;
                        float v10 = (idx10 < corr_internal_size) ? corr_internal[idx10] : 0.0f;
                        float v11 = (idx11 < corr_internal_size) ? corr_internal[idx11] : 0.0f;
                        
                        // Interpolate
                        float interpolated = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
                        
                        // Store in output buffer (7x7)
                        // Match Python's permute(0,1,3,2,4,5): output as [e, corr_jj, corr_ii, i0, j0] = [e, corr_x, corr_y, i0, j0]
                        size_t out_idx = static_cast<size_t>(e) * D_output * D_output * P * P +
                                         static_cast<size_t>(out_jj) * D_output * P * P +
                                         static_cast<size_t>(out_ii) * P * P +
                                         static_cast<size_t>(i0) * P + static_cast<size_t>(j0);
                        
                        if (out_idx < corr_output_size) {
                            corr_out[out_idx] = interpolated;
                        }
                    }
                }
            }
        }
    }
    
    // Diagnostic: Log summary statistics
    if (logger) {
        int nonzero_count = 0;
        float max_corr = std::numeric_limits<float>::lowest();
        float min_corr = std::numeric_limits<float>::max();
        double sum_corr = 0.0;
        
        for (size_t i = 0; i < corr_output_size; i++) {
            float val = corr_out[i];
            if (val != 0.0f) {
                nonzero_count++;
                if (val > max_corr) max_corr = val;
                if (val < min_corr) min_corr = val;
            }
            sum_corr += val;
        }
    }
    
    // Save 8x8 internal buffer if requested (for debugging)
    if (corr_8x8_out != nullptr) {
        std::memcpy(corr_8x8_out, corr_internal.data(), sizeof(float) * corr_internal_size);
    }
}

// Combined correlation for both pyramid levels (matches Python: torch.stack([corr1, corr2], -1).view(1, len(ii), -1))
// This function calls computeCorrelationSingle twice and stacks the results
// Python equivalent:
//   corr1 = altcorr.corr(self.gmap, self.pyramid[0], coords / 1, ii1, jj1, 3)
//   corr2 = altcorr.corr(self.gmap, self.pyramid[1], coords / 4, ii1, jj1, 3)
//   return torch.stack([corr1, corr2], -1).view(1, len(ii), -1)
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
    float* corr_out,            // Output: [num_active, D, D, P, P, 2] - Correlation volumes
                                  // Format: [edge, corr_x, corr_y, patch_y, patch_x, level] (matches Python's permute output)
    int frame_num,              // Optional: Frame number for saving 8x8 debug buffers
    float* corr1_8x8_out,       // Optional: Output 8x8 buffer for level 0 [num_active, 8, 8, P, P]
    float* corr2_8x8_out)       // Optional: Output 8x8 buffer for level 1 [num_active, 8, 8, P, P]
{
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
    
    const int R = 3;  // Correlation radius (matches Python: altcorr.corr(..., 3))
    // computeCorrelationSingle now computes 8x8 correlation and reduces to 7x7 via bilinear wrapper (matching Python)
    const int D_output = 2 * R + 1;  // Final output size (D = 7 for R=3, matches Python's final 7x7 output)
    
    // Map indices (matches Python: ii1 = kk % (M * pmem), jj1 = jj % mem)
    std::vector<int> ii1(num_active);
    std::vector<int> jj1(num_active);
    
    int mod_value = M * num_gmap_frames;
    if (mod_value == 0) {
        printf("[computeCorrelation] ERROR: Division by zero! M=%d, num_gmap_frames=%d\n", M, num_gmap_frames);
        fflush(stdout);
        return;
    }
    
    for (int e = 0; e < num_active; e++) {
        // Python: ii = self.pg.kk[:num_active], then ii1 = ii % (self.M * self.pmem)
        ii1[e] = kk[e] % mod_value;  // Map to gmap ring buffer
        // Python: jj1 = jj % mem
        jj1[e] = jj[e] % num_frames;  // Map to pyramid ring buffer
    }
    
    // Allocate temporary buffers for individual correlation volumes
    // Each correlation volume: [num_active, D_output, D_output, P, P] where D_output = 7
    const size_t corr_single_size = static_cast<size_t>(num_active) * D_output * D_output * P * P;
    std::vector<float> corr1(corr_single_size);
    std::vector<float> corr2(corr_single_size);
    
    // Allocate buffers for 8x8 internal correlation (for debugging)
    const int D_internal = 2 * R + 2;  // 8x8 internal
    const size_t corr_8x8_size = static_cast<size_t>(num_active) * D_internal * D_internal * P * P;
    std::vector<float> corr1_8x8(corr_8x8_size);
    std::vector<float> corr2_8x8(corr_8x8_size);
    
    // Call computeCorrelationSingle for pyramid0 (matches Python: corr1 = altcorr.corr(..., coords / 1, ...))
    auto corr_t0 = std::chrono::high_resolution_clock::now();
    computeCorrelationSingle(
        gmap, pyramid0, coords,
        ii1.data(), jj1.data(),
        num_active, M, P,
        num_frames, num_gmap_frames,
        fmap1_H, fmap1_W,
        feature_dim,
        1.0f,  // coord_scale = 1.0 (coords / 1)
        R,     // radius = 3
        corr1.data(),
        corr1_8x8.data()  // Save 8x8 internal buffer
    );
    auto corr_t1 = std::chrono::high_resolution_clock::now();
    
    // Call computeCorrelationSingle for pyramid1 (matches Python: corr2 = altcorr.corr(..., coords / 4, ...))
    computeCorrelationSingle(
        gmap, pyramid1, coords,
        ii1.data(), jj1.data(),
        num_active, M, P,
        num_frames, num_gmap_frames,
        fmap2_H, fmap2_W,
        feature_dim,
        0.25f, // coord_scale = 0.25 (coords / 4)
        R,     // radius = 3
        corr2.data(),
        corr2_8x8.data()  // Save 8x8 internal buffer
    );
    auto corr_t2 = std::chrono::high_resolution_clock::now();
    
    {
        auto logger = spdlog::get("dpvo");
        if (logger) {
            double corr1_ms = std::chrono::duration_cast<std::chrono::microseconds>(corr_t1 - corr_t0).count() / 1000.0;
            double corr2_ms = std::chrono::duration_cast<std::chrono::microseconds>(corr_t2 - corr_t1).count() / 1000.0;
            logger->info("[TIMING] Correlation breakdown: Level0({}x{}): {:.1f} ms | Level1({}x{}): {:.1f} ms | edges={}",
                         fmap1_H, fmap1_W, corr1_ms, fmap2_H, fmap2_W, corr2_ms, num_active);
        }
    }
    
    // Copy 8x8 buffers to output if requested
    if (corr1_8x8_out != nullptr) {
        std::memcpy(corr1_8x8_out, corr1_8x8.data(), sizeof(float) * corr_8x8_size);
    }
    if (corr2_8x8_out != nullptr) {
        std::memcpy(corr2_8x8_out, corr2_8x8.data(), sizeof(float) * corr_8x8_size);
    }
    
    // Stack corr1 and corr2 together (matches Python: torch.stack([corr1, corr2], -1))
    // Output layout: [num_active, D, D, P, P, 2] (channel last)
    // Match Python's permute(0,1,3,2,4,5): output as [e, corr_jj, corr_ii, i0, j0, c] = [e, corr_x, corr_y, i0, j0, c]
    // This interleaves the two correlation volumes along the channel dimension
    for (int e = 0; e < num_active; e++) {
        for (int corr_jj = 0; corr_jj < D_output; corr_jj++) {  // corr_x (horizontal offset)
            for (int corr_ii = 0; corr_ii < D_output; corr_ii++) {  // corr_y (vertical offset)
                for (int i0 = 0; i0 < P; i0++) {
                    for (int j0 = 0; j0 < P; j0++) {
                        // Source indices in corr1 and corr2: [e, corr_jj, corr_ii, i0, j0] (already swapped in computeCorrelationSingle)
                        size_t src_idx = static_cast<size_t>(e) * D_output * D_output * P * P +
                                        static_cast<size_t>(corr_jj) * D_output * P * P +
                                        static_cast<size_t>(corr_ii) * P * P +
                                        static_cast<size_t>(i0) * P +
                                        static_cast<size_t>(j0);
                        
                        // Destination indices in stacked output: [e, corr_jj, corr_ii, i0, j0, c] = [e, corr_x, corr_y, i0, j0, c]
                        // Channel 0: corr1, Channel 1: corr2
                        size_t dst_idx_c0 = static_cast<size_t>(e) * D_output * D_output * P * P * 2 +
                                            static_cast<size_t>(corr_jj) * D_output * P * P * 2 +
                                            static_cast<size_t>(corr_ii) * P * P * 2 +
                                            static_cast<size_t>(i0) * P * 2 +
                                            static_cast<size_t>(j0) * 2 +
                                            0;  // Channel 0
                        
                        size_t dst_idx_c1 = static_cast<size_t>(e) * D_output * D_output * P * P * 2 +
                                            static_cast<size_t>(corr_jj) * D_output * P * P * 2 +
                                            static_cast<size_t>(corr_ii) * P * P * 2 +
                                            static_cast<size_t>(i0) * P * 2 +
                                            static_cast<size_t>(j0) * 2 +
                                            1;  // Channel 1
                        
                        if (src_idx < corr_single_size) {
                            corr_out[dst_idx_c0] = corr1[src_idx];  // Channel 0: pyramid0 correlation
                            corr_out[dst_idx_c1] = corr2[src_idx];  // Channel 1: pyramid1 correlation
                        }
                    }
                }
            }
        }
    }
    
    // Log statistics (matches original logging)
    const size_t corr_total_size = static_cast<size_t>(num_active) * D_output * D_output * P * P * 2;
    
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
    const int sample_count = 20;
    
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
    
    logger->info("computeCorrelation: Output statistics - size={}, zero_count={}, nonzero_count={}, min={:.6f}, max={:.6f}, mean={:.6f}",
                 corr_total_size, zero_count, nonzero_count, min_corr, max_corr, mean_corr);
    
    // Log sample values
    std::string sample_values = "[";
    for (int i = 0; i < sample_count && i < static_cast<int>(corr_total_size); i++) {
        sample_values += std::to_string(corr_out[i]);
        if (i < sample_count - 1 && i < static_cast<int>(corr_total_size) - 1) {
            sample_values += ", ";
        }
    }
    sample_values += "]";
    logger->info("computeCorrelation: First {} sample values: {}", sample_count, sample_values);
    
    // Log per-edge statistics for first few edges
    if (num_active > 0) {
        int edges_to_log = std::min(3, num_active);
        for (int e = 0; e < edges_to_log; e++) {
            int edge_zero = 0;
            int edge_nonzero = 0;
            float edge_min = std::numeric_limits<float>::max();
            float edge_max = std::numeric_limits<float>::lowest();
            double edge_sum = 0.0;
            
            size_t edge_start = static_cast<size_t>(e) * D_output * D_output * P * P * 2;
            size_t edge_size = D_output * D_output * P * P * 2;
            
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
    
    logger->info("computeCorrelation: COMPLETED - processed {} edges (corr1 and corr2 stacked)", num_active);
}
