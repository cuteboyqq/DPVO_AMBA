#include "correlation_kernel.hpp"
#include "correlation_bilinear_helpers.hpp"
#include "correlation_file_io.hpp"
#include "target_frame.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>
#include <limits>
#include <chrono>
#include <algorithm>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

inline bool within_bounds(int h, int w, int H, int W)
{
    return (h >= 0 && h < H && w >= 0 && w < W);
}

void patchify_cpu(
    const float* fmap,
    const float* coords,
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap
)
{
    const int D = (radius == 0) ? 1 : (2 * radius + 1);
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
                    const int fmap_idx = (c * (H * W)) + (i * W) + j;
                    const int gmap_idx = m * C * D * D + c * D * D + ii * D + jj;
                    gmap[gmap_idx] = fmap[fmap_idx];
                }
            }
        }
    }
}


void patchify_cpu_safe(
    const float* fmap,
    const float* coords,
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap
)
{
    const int D = (radius == 0) ? 1 : (2 * radius + 1);
    const int fmap_size = C * H * W;
    const int gmap_size = M * C * D * D;

    const bool inplace = (fmap == gmap);
    std::vector<float> fmap_copy;
    const float* src = fmap;

    if (inplace) {
        fmap_copy.resize(fmap_size);
        std::memcpy(fmap_copy.data(), fmap, sizeof(float) * fmap_size);
        src = fmap_copy.data();
    }

    std::memset(gmap, 0, sizeof(float) * gmap_size);

    for (int m = 0; m < M; m++) {
        const float coord_x = coords[m*2 + 0];
        const float coord_y = coords[m*2 + 1];
        const int cx = static_cast<int>(std::floor(coord_x));
        const int cy = static_cast<int>(std::floor(coord_y));

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

                    int src_idx = fmap_c_offset + y * W + x;
                    int dst_idx = gmap_c_offset + ii * D + jj;
                    gmap[dst_idx] = src[src_idx];
                }
            }
        }
    }
}


// =============================================================================
// Optimized correlation kernel
// =============================================================================
//
// Key optimizations vs original:
//   1. Channel-outer loop: iterates channels in the outermost position of the
//      dot-product so that pyramid memory is accessed sequentially through each
//      channel plane (H*W contiguous), avoiding the ~124 KB stride-per-channel
//      cache thrashing of the original offset-outer loop.
//   2. Stack-local 8x8 correlation buffer per patch pixel instead of a large
//      heap-allocated std::vector for the entire batch.
//   3. Gmap features pre-extracted into a contiguous local array (stride 9 →
//      stride 1) for better vectorisation of the inner dot-product.
//   4. Removed normalize→unnormalize coordinate round-trip (was identity).
//   5. Bilinear parameters precomputed once per (e, i0, j0) and reused across
//      all 128 channels.
//   6. All diagnostic / statistics loops removed from the hot path.
// =============================================================================

// ---------------------------------------------------------------------------
// [STUDY NOTE 1] BilinearInfo — precomputed bilinear sampling parameters
// ---------------------------------------------------------------------------
// WHY: In the original code, for every correlation offset (8x8 = 64 offsets),
//      the bilinear interpolation weights and 4 corner addresses were computed
//      INSIDE the 128-channel inner loop. That means the same weights and
//      addresses were recomputed 128 times for each offset.
//
// FIX: Precompute them once per offset into this struct, then reuse across
//      all 128 channel iterations.
//
// MEMORY LAYOUT:
//   off00..off11 = base offsets into the pyramid buffer for the 4 bilinear
//                  corners. To access channel f, add (f * fmap_H * fmap_W).
//                  Example: pyramid[off00 + f * hw] = top-left corner at ch f.
//
//   w00..w11    = bilinear interpolation weights (sum to 1.0):
//                  w00 = (1-dx)(1-dy)  top-left
//                  w01 = dx*(1-dy)     top-right
//                  w10 = (1-dx)*dy     bottom-left
//                  w11 = dx*dy         bottom-right
// ---------------------------------------------------------------------------
struct BilinearInfo {
    bool valid;
    size_t off00, off01, off10, off11;
    float w00, w01, w10, w11;
};

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




// void computeCorrelationSingle(
//     const float* gmap,
//     const float* pyramid,
//     const float* coords,
//     const int* ii1,
//     const int* jj1,
//     int num_active,
//     int M,
//     int P,
//     int num_frames,
//     int num_gmap_frames,
//     int fmap_H, int fmap_W,
//     int feature_dim,
//     float coord_scale,
//     int radius,
//     float* corr_out,
//     float* corr_8x8_out)
// {
//     if (gmap == nullptr || pyramid == nullptr || coords == nullptr ||
//         ii1 == nullptr || jj1 == nullptr || corr_out == nullptr) {
//         return;
//     }
//     if (num_active <= 0 || M <= 0 || P <= 0 || num_frames <= 0 || num_gmap_frames <= 0) {
//         return;
//     }

//     const int R = radius;
//     const int D_internal = 2 * R + 2;   // 8 for R=3
//     const int D_output   = 2 * R + 1;   // 7 for R=3
//     const int D_gmap = 3;
//     const int gmap_center_offset = (D_gmap - P) / 2;

//     const size_t hw = static_cast<size_t>(fmap_H) * fmap_W;
//     const size_t gmap_ch_stride = static_cast<size_t>(D_gmap) * D_gmap;  // stride between channels in gmap
//     const size_t corr_output_size = static_cast<size_t>(num_active) * D_output * D_output * P * P;

//     std::memset(corr_out, 0, sizeof(float) * corr_output_size);

//     const float fW1 = static_cast<float>(fmap_W - 1);
//     const float fH1 = static_cast<float>(fmap_H - 1);
//     const float bi_tolerance = 0.5f;

//     // Optional: copy 8x8 debug output into a caller-provided buffer
//     const size_t corr_8x8_total = corr_8x8_out
//         ? static_cast<size_t>(num_active) * D_internal * D_internal * P * P
//         : 0;

//     {
//         // -------------------------------------------------------------------
//         // [STUDY NOTE 2] Stack buffers — why they replace heap allocations
//         // -------------------------------------------------------------------
//         // gmap_local[128]:
//         //   In gmap, features for one patch pixel are spaced D_gmap*D_gmap = 9
//         //   floats apart between channels (layout: [frame][patch][C][3][3]).
//         //   Strided access (stride 9) prevents CPU auto-vectorization.
//         //   Copying 128 values into a contiguous array gives stride-1 access,
//         //   which enables SIMD and better cache-line utilisation.
//         //
//         // bi_params[64]:
//         //   Stores precomputed bilinear weights + corner offsets for all 8x8
//         //   correlation offsets. Without this, the original code recomputed
//         //   these inside the 128-channel inner loop (128x redundant work).
//         //
//         // local_corr[64]:
//         //   The original code allocated a heap vector of size
//         //   num_active * 64 * P * P (~830 KB for 360 edges, P=3) via
//         //   std::vector<float>. This caused malloc/free per call and poor
//         //   cache locality. Using a 64-float stack array (256 bytes) that is
//         //   reused per (edge, patch_pixel) keeps everything in L1 cache.
//         // -------------------------------------------------------------------
//         float gmap_local[128];
//         BilinearInfo bi_params[8 * 8];
//         float local_corr[8 * 8];

//         for (int e = 0; e < num_active; e++) {
//             const int ii1_val = ii1[e];
//             const int jj1_val = jj1[e];

//             const int gmap_frame   = ii1_val / M;
//             const int patch_idx    = ii1_val % M;
//             const int pyramid_frame = jj1_val;

//             if (patch_idx < 0 || patch_idx >= M ||
//                 gmap_frame < 0 || gmap_frame >= num_gmap_frames ||
//                 pyramid_frame < 0 || pyramid_frame >= num_frames) {
//                 continue;
//             }

//             const size_t frame_offset = static_cast<size_t>(pyramid_frame) * feature_dim * hw;

//             for (int i0 = 0; i0 < P; i0++) {
//                 for (int j0 = 0; j0 < P; j0++) {
//                     // -------------------------------------------------------
//                     // [STUDY NOTE 3] Coordinate handling & half-precision
//                     // -------------------------------------------------------
//                     // coords layout: [num_active, 2, P, P]
//                     //   Channel 0 = x (column), channel 1 = y (row).
//                     //   coord_x_idx jumps over e*(2*P*P) to reach edge e,
//                     //   then i0*P + j0 for the patch pixel.
//                     //   coord_y_idx = coord_x_idx + P*P to reach channel 1.
//                     //
//                     // Half-precision emulation:
//                     //   Python does coords.half() before floor(), which
//                     //   rounds certain values differently in FP16. Example:
//                     //   30.999988 in FP32 → 31.0 in FP16 → floor gives 31
//                     //   instead of 30. We replicate this with
//                     //   float_to_half_to_float() to stay bit-identical.
//                     //
//                     // coord_scale:
//                     //   Level 0 uses scale=1.0 (full resolution).
//                     //   Level 1 uses scale=0.25 (quarter resolution pyramid).
//                     // -------------------------------------------------------
//                     const int coord_x_idx = e * 2 * P * P + i0 * P + j0;
//                     const int coord_y_idx = coord_x_idx + P * P;

//                     const float raw_x = coords[coord_x_idx];
//                     const float raw_y = coords[coord_y_idx];

//                     const float x = raw_x * coord_scale;
//                     const float y = raw_y * coord_scale;

//                     if (!std::isfinite(x) || !std::isfinite(y)) continue;

//                     const float x_half = float_to_half_to_float(x);
//                     const float y_half = float_to_half_to_float(y);

//                     if (!std::isfinite(x_half) || !std::isfinite(y_half)) continue;

//                     const float x0 = std::floor(x_half);
//                     const float y0 = std::floor(y_half);

//                     // -------------------------------------------------------
//                     // [STUDY NOTE 4] Gmap pre-extraction (stride 9 → stride 1)
//                     // -------------------------------------------------------
//                     // gmap layout: [num_gmap_frames, M, feature_dim, D_gmap, D_gmap]
//                     //   = [frames][patches][128][3][3]
//                     //
//                     // To fetch the feature vector for patch pixel (i0,j0),
//                     // we need gmap[frame][patch][f][gmap_i][gmap_j] for all f.
//                     //
//                     // ORIGINAL access pattern:
//                     //   gmap[gmap_base + f * 9]  (stride 9 between channels)
//                     //   This strided access means the CPU loads 9 floats but
//                     //   only uses 1, wasting 8/9 of each cache line fetch.
//                     //   It also prevents auto-vectorization (SIMD needs
//                     //   contiguous data).
//                     //
//                     // OPTIMIZED: copy all 128 channels into gmap_local[f]
//                     //   (stride 1 = sequential). After this, the dot-product
//                     //   loop accesses gmap_local[0], gmap_local[1], ...
//                     //   which is perfect for SIMD and cache lines.
//                     //
//                     // Cost: 128 scattered reads (once per patch pixel).
//                     // Benefit: 128 × 64 = 8192 accesses become stride-1.
//                     // -------------------------------------------------------
//                     const int gmap_i = i0 + gmap_center_offset;
//                     const int gmap_j = j0 + gmap_center_offset;
//                     const size_t gmap_base =
//                         static_cast<size_t>(gmap_frame) * M * feature_dim * gmap_ch_stride +
//                         static_cast<size_t>(patch_idx)  * feature_dim * gmap_ch_stride +
//                         static_cast<size_t>(gmap_i) * D_gmap +
//                         static_cast<size_t>(gmap_j);

//                     for (int f = 0; f < feature_dim; f++) {
//                         gmap_local[f] = gmap[gmap_base + static_cast<size_t>(f) * gmap_ch_stride];
//                     }

//                     // -------------------------------------------------------
//                     // [STUDY NOTE 5] Bilinear precomputation for 8x8 offsets
//                     // -------------------------------------------------------
//                     // For R=3, we sample an 8x8 grid of integer offsets:
//                     //   gx = floor(x_half) + (corr_jj - R), corr_jj ∈ [0..7]
//                     //   gy = floor(y_half) + (corr_ii - R), corr_ii ∈ [0..7]
//                     // giving offsets from -3 to +4 around the coordinate.
//                     //
//                     // INDEXING:
//                     //   idx = corr_jj * D_internal + corr_ii
//                     //   This stores as [x_offset][y_offset], matching the
//                     //   output format after Python's permute(0,1,3,2,4,5)
//                     //   where x and y axes are transposed.
//                     //
//                     // TOLERANCE CHECK:
//                     //   Matches PyTorch's grid_sample with padding_mode='zeros':
//                     //   coordinates outside [-0.5, W-0.5] are set to zero.
//                     //   The tolerance of 0.5 pixels allows sampling at the
//                     //   boundary where bilinear interpolation can still
//                     //   reference valid pixels via clamping.
//                     //
//                     // CLAMP-THEN-FLOOR:
//                     //   px = clamp(gx, 0, W-1), then bx0 = floor(px).
//                     //   Equivalent to the original floor-then-clamp approach
//                     //   but slightly cleaner for edge handling.
//                     //
//                     // NOTE on bilinear weights at integer coordinates:
//                     //   gx and gy are always integers here (floor of half-
//                     //   precision coord + integer offset). So bdx = bdy = 0,
//                     //   making w00 = 1.0 and w01 = w10 = w11 = 0.0.
//                     //   The actual sub-pixel interpolation happens later in
//                     //   the 8x8→7x7 reduction step (Study Note 7).
//                     // -------------------------------------------------------
//                     int num_valid = 0;
//                     for (int corr_ii = 0; corr_ii < D_internal; corr_ii++) {
//                         for (int corr_jj = 0; corr_jj < D_internal; corr_jj++) {
//                             const float gx = x0 + static_cast<float>(corr_jj - R);
//                             const float gy = y0 + static_cast<float>(corr_ii - R);

//                             const int idx = corr_jj * D_internal + corr_ii;
//                             BilinearInfo& bi = bi_params[idx];

//                             if (gx < -bi_tolerance || gx > fW1 + bi_tolerance ||
//                                 gy < -bi_tolerance || gy > fH1 + bi_tolerance) {
//                                 bi.valid = false;
//                                 local_corr[idx] = 0.0f;
//                                 continue;
//                             }
//                             bi.valid = true;
//                             num_valid++;

//                             const float px = std::max(0.0f, std::min(gx, fW1));
//                             const float py = std::max(0.0f, std::min(gy, fH1));

//                             const int bx0 = std::min(static_cast<int>(std::floor(px)), fmap_W - 1);
//                             const int by0 = std::min(static_cast<int>(std::floor(py)), fmap_H - 1);
//                             const int bx1 = std::min(bx0 + 1, fmap_W - 1);
//                             const int by1 = std::min(by0 + 1, fmap_H - 1);

//                             const float bdx = px - static_cast<float>(bx0);
//                             const float bdy = py - static_cast<float>(by0);

//                             bi.w00 = (1.0f - bdx) * (1.0f - bdy);
//                             bi.w01 = bdx * (1.0f - bdy);
//                             bi.w10 = (1.0f - bdx) * bdy;
//                             bi.w11 = bdx * bdy;

//                             bi.off00 = frame_offset + static_cast<size_t>(by0) * fmap_W + bx0;
//                             bi.off01 = frame_offset + static_cast<size_t>(by0) * fmap_W + bx1;
//                             bi.off10 = frame_offset + static_cast<size_t>(by1) * fmap_W + bx0;
//                             bi.off11 = frame_offset + static_cast<size_t>(by1) * fmap_W + bx1;

//                             local_corr[idx] = 0.0f;
//                         }
//                     }

//                     // -------------------------------------------------------
//                     // [STUDY NOTE 6] Channel-outer dot product
//                     //                *** THE key optimization ***
//                     // -------------------------------------------------------
//                     // This is the most performance-critical change.
//                     //
//                     // ORIGINAL (offset-outer) loop structure:
//                     //   for each of 64 offsets:
//                     //     for each of 128 channels (f):
//                     //       sample = pyramid[frame + f*H*W + y*W + x]
//                     //       corr[offset] += gmap[..f..] * sample
//                     //
//                     //   Problem: Each channel f accesses pyramid at offset
//                     //   f * H * W (e.g., for 48x160: stride = 7680 floats =
//                     //   30 KB). Over 64 offsets × 128 channels, the CPU
//                     //   constantly jumps between distant memory locations,
//                     //   thrashing the L1/L2 caches.
//                     //
//                     // OPTIMIZED (channel-outer) loop structure:
//                     //   for each of 128 channels (f):
//                     //     g = gmap_local[f]          // stride-1, see Note 4
//                     //     ch_off = f * H * W         // one channel plane
//                     //     for each of 64 offsets:
//                     //       sample = pyramid[bi.off + ch_off]
//                     //       corr[offset] += g * sample
//                     //
//                     //   Why this is faster:
//                     //   1. All 64 offsets access NEARBY locations within the
//                     //      SAME channel plane (an ~8-pixel neighborhood).
//                     //      These fit in just a few cache lines.
//                     //   2. Moving to the next channel (f+1) adds H*W to
//                     //      all offsets — a sequential stride through memory.
//                     //   3. gmap_local[f] is stride-1 (see Study Note 4).
//                     //
//                     //   Cache behavior comparison:
//                     //     Original: 64 × 128 random jumps of ~30 KB = thrash
//                     //     Optimized: 128 sequential planes, 64 nearby reads
//                     //                per plane = streams nicely through cache
//                     //
//                     // NUMERICAL NOTE:
//                     //   Accumulation order (f=0,1,...,127) is identical to
//                     //   the original, so floating-point results are
//                     //   bit-identical (no reordering of additions).
//                     // -------------------------------------------------------
//                     if (num_valid > 0) {
//                         for (int f = 0; f < feature_dim; f++) {
//                             const float g = gmap_local[f];
//                             const size_t ch_off = static_cast<size_t>(f) * hw;

//                             for (int idx = 0; idx < D_internal * D_internal; idx++) {
//                                 if (!bi_params[idx].valid) continue;
//                                 const BilinearInfo& bi = bi_params[idx];

//                                 const float sample =
//                                     bi.w00 * pyramid[bi.off00 + ch_off] +
//                                     bi.w01 * pyramid[bi.off01 + ch_off] +
//                                     bi.w10 * pyramid[bi.off10 + ch_off] +
//                                     bi.w11 * pyramid[bi.off11 + ch_off];

//                                 local_corr[idx] += g * sample;
//                             }
//                         }
//                     }

//                     // --- Save 8x8 debug buffer if requested ---
//                     if (corr_8x8_out) {
//                         for (int corr_jj = 0; corr_jj < D_internal; corr_jj++) {
//                             for (int corr_ii = 0; corr_ii < D_internal; corr_ii++) {
//                                 const size_t dst = static_cast<size_t>(e) * D_internal * D_internal * P * P +
//                                                    static_cast<size_t>(corr_jj) * D_internal * P * P +
//                                                    static_cast<size_t>(corr_ii) * P * P +
//                                                    static_cast<size_t>(i0) * P + j0;
//                                 if (dst < corr_8x8_total) {
//                                     corr_8x8_out[dst] = local_corr[corr_jj * D_internal + corr_ii];
//                                 }
//                             }
//                         }
//                     }

//                     // -------------------------------------------------------
//                     // [STUDY NOTE 7] Bilinear 8x8 → 7x7 reduction
//                     // -------------------------------------------------------
//                     // The Python code computes correlation at 8x8 integer
//                     // offsets [-R, R+1] = [-3, +4], then uses bilinear
//                     // interpolation with the fractional part of the
//                     // coordinate to produce a 7x7 output grid [-R, R].
//                     //
//                     // dx, dy = fractional parts of the half-precision coords:
//                     //   dx = half(x_half - floor(x_half))
//                     //   dy = half(y_half - floor(y_half))
//                     //   These are also passed through float_to_half_to_float
//                     //   to match the Python .half() behavior.
//                     //
//                     // The 4 weights wr00..wr11 interpolate between adjacent
//                     // integer-offset correlation values:
//                     //   v00 = corr at offset (out_jj,   out_ii  )
//                     //   v01 = corr at offset (out_jj+1, out_ii  )
//                     //   v10 = corr at offset (out_jj,   out_ii+1)
//                     //   v11 = corr at offset (out_jj+1, out_ii+1)
//                     //   val = wr00*v00 + wr01*v01 + wr10*v10 + wr11*v11
//                     //
//                     // Output layout: [e, out_jj, out_ii, i0, j0]
//                     //   = [edge, corr_x, corr_y, patch_y, patch_x]
//                     //   This matches Python's permute(0,1,3,2,4,5) output.
//                     // -------------------------------------------------------
//                     const float dx = float_to_half_to_float(x_half - x0);
//                     const float dy = float_to_half_to_float(y_half - y0);

//                     const float wr00 = (1.0f - dx) * (1.0f - dy);
//                     const float wr01 = dx * (1.0f - dy);
//                     const float wr10 = (1.0f - dx) * dy;
//                     const float wr11 = dx * dy;

//                     for (int out_jj = 0; out_jj < D_output; out_jj++) {
//                         for (int out_ii = 0; out_ii < D_output; out_ii++) {
//                             const float v00 = local_corr[out_jj * D_internal + out_ii];
//                             const float v01 = local_corr[(out_jj + 1) * D_internal + out_ii];
//                             const float v10 = local_corr[out_jj * D_internal + (out_ii + 1)];
//                             const float v11 = local_corr[(out_jj + 1) * D_internal + (out_ii + 1)];

//                             const float val = wr00 * v00 + wr01 * v01 + wr10 * v10 + wr11 * v11;

//                             const size_t out_idx =
//                                 static_cast<size_t>(e) * D_output * D_output * P * P +
//                                 static_cast<size_t>(out_jj) * D_output * P * P +
//                                 static_cast<size_t>(out_ii) * P * P +
//                                 static_cast<size_t>(i0) * P + j0;

//                             if (out_idx < corr_output_size) {
//                                 corr_out[out_idx] = val;
//                             }
//                         }
//                     }
//                 } // j0
//             } // i0
//         } // e
//     } // omp parallel
// }


// Combined correlation for both pyramid levels
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
    int num_frames,
    int num_gmap_frames,
    int fmap1_H, int fmap1_W,
    int fmap2_H, int fmap2_W,
    int feature_dim,
    float* corr_out,
    int frame_num,
    float* corr1_8x8_out,
    float* corr2_8x8_out)
{
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

    const int R = 3;
    const int D_output = 2 * R + 1;  // 7

    // Map indices (matches Python: ii1 = kk % (M * pmem), jj1 = jj % mem)
    const int mod_value = M * num_gmap_frames;
    if (mod_value == 0) {
        printf("[computeCorrelation] ERROR: Division by zero! M=%d, num_gmap_frames=%d\n", M, num_gmap_frames);
        fflush(stdout);
        return;
    }

    std::vector<int> ii1(num_active);
    std::vector<int> jj1(num_active);
    for (int e = 0; e < num_active; e++) {
        ii1[e] = kk[e] % mod_value;
        jj1[e] = jj[e] % num_frames;
    }

    // Allocate temporary buffers for individual correlation volumes
    const size_t corr_single_size = static_cast<size_t>(num_active) * D_output * D_output * P * P;
    std::vector<float> corr1(corr_single_size);
    std::vector<float> corr2(corr_single_size);

    // Level 0: pyramid0, coord_scale = 1.0
    auto corr_t0 = std::chrono::high_resolution_clock::now();
    computeCorrelationSingle(
        gmap, pyramid0, coords,
        ii1.data(), jj1.data(),
        num_active, M, P,
        num_frames, num_gmap_frames,
        fmap1_H, fmap1_W,
        feature_dim,
        1.0f, R,
        corr1.data(),
        corr1_8x8_out);
    auto corr_t1 = std::chrono::high_resolution_clock::now();

    // Level 1: pyramid1, coord_scale = 0.25
    computeCorrelationSingle(
        gmap, pyramid1, coords,
        ii1.data(), jj1.data(),
        num_active, M, P,
        num_frames, num_gmap_frames,
        fmap2_H, fmap2_W,
        feature_dim,
        0.25f, R,
        corr2.data(),
        corr2_8x8_out);
    auto corr_t2 = std::chrono::high_resolution_clock::now();

    {
        auto logger = spdlog::get("dpvo");
        if (logger) {
            double corr1_ms = std::chrono::duration_cast<std::chrono::microseconds>(corr_t1 - corr_t0).count() / 1000.0;
            double corr2_ms = std::chrono::duration_cast<std::chrono::microseconds>(corr_t2 - corr_t1).count() / 1000.0;
            logger->info("[TIMING] Correlation: Level0({}x{}): {:.1f} ms | Level1({}x{}): {:.1f} ms | edges={}",
                         fmap1_H, fmap1_W, corr1_ms, fmap2_H, fmap2_W, corr2_ms, num_active);
        }
    }

    // -------------------------------------------------------------------
    // [STUDY NOTE 8] Stacking corr1 and corr2
    // -------------------------------------------------------------------
    // Equivalent to Python's: torch.stack([corr1, corr2], dim=-1)
    //
    // corr1 = correlation from pyramid level 0 (full resolution)
    // corr2 = correlation from pyramid level 1 (quarter resolution)
    //
    // Simple interleaving produces:
    //   [val1_0, val2_0, val1_1, val2_1, val1_2, val2_2, ...]
    //
    // Output shape: [num_active, D, D, P, P, 2]
    //   The last dimension (size 2) is the pyramid level, matching
    //   the stack(..., dim=-1) semantic. The downstream network
    //   (update operator) expects both scales as the final channels.
    // -------------------------------------------------------------------
    for (size_t i = 0; i < corr_single_size; i++) {
        corr_out[i * 2 + 0] = corr1[i];
        corr_out[i * 2 + 1] = corr2[i];
    }
}
