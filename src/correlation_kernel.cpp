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

void computeCorrelationSingle(
    const float* gmap,
    const float* pyramid,
    const float* coords,
    const int* ii1,
    const int* jj1,
    int num_active,
    int M,
    int P,
    int num_frames,
    int num_gmap_frames,
    int fmap_H, int fmap_W,
    int feature_dim,
    float coord_scale,
    int radius,
    float* corr_out,
    float* corr_8x8_out)
{
    if (gmap == nullptr || pyramid == nullptr || coords == nullptr ||
        ii1 == nullptr || jj1 == nullptr || corr_out == nullptr) {
        return;
    }
    if (num_active <= 0 || M <= 0 || P <= 0 || num_frames <= 0 || num_gmap_frames <= 0) {
        return;
    }

    const int R = radius;
    const int D_internal = 2 * R + 2;   // 8 for R=3
    const int D_output   = 2 * R + 1;   // 7 for R=3
    const int D_gmap = 3;
    const int gmap_center_offset = (D_gmap - P) / 2;

    const size_t hw = static_cast<size_t>(fmap_H) * fmap_W;
    const size_t gmap_ch_stride = static_cast<size_t>(D_gmap) * D_gmap;  // stride between channels in gmap
    const size_t corr_output_size = static_cast<size_t>(num_active) * D_output * D_output * P * P;

    std::memset(corr_out, 0, sizeof(float) * corr_output_size);

    const float fW1 = static_cast<float>(fmap_W - 1);
    const float fH1 = static_cast<float>(fmap_H - 1);
    const float bi_tolerance = 0.5f;

    // Optional: copy 8x8 debug output into a caller-provided buffer
    const size_t corr_8x8_total = corr_8x8_out
        ? static_cast<size_t>(num_active) * D_internal * D_internal * P * P
        : 0;

    {
        // -------------------------------------------------------------------
        // [STUDY NOTE 2] Stack buffers — why they replace heap allocations
        // -------------------------------------------------------------------
        // gmap_local[128]:
        //   In gmap, features for one patch pixel are spaced D_gmap*D_gmap = 9
        //   floats apart between channels (layout: [frame][patch][C][3][3]).
        //   Strided access (stride 9) prevents CPU auto-vectorization.
        //   Copying 128 values into a contiguous array gives stride-1 access,
        //   which enables SIMD and better cache-line utilisation.
        //
        // bi_params[64]:
        //   Stores precomputed bilinear weights + corner offsets for all 8x8
        //   correlation offsets. Without this, the original code recomputed
        //   these inside the 128-channel inner loop (128x redundant work).
        //
        // local_corr[64]:
        //   The original code allocated a heap vector of size
        //   num_active * 64 * P * P (~830 KB for 360 edges, P=3) via
        //   std::vector<float>. This caused malloc/free per call and poor
        //   cache locality. Using a 64-float stack array (256 bytes) that is
        //   reused per (edge, patch_pixel) keeps everything in L1 cache.
        // -------------------------------------------------------------------
        float gmap_local[128];
        BilinearInfo bi_params[8 * 8];
        float local_corr[8 * 8];

        for (int e = 0; e < num_active; e++) {
            const int ii1_val = ii1[e];
            const int jj1_val = jj1[e];

            const int gmap_frame   = ii1_val / M;
            const int patch_idx    = ii1_val % M;
            const int pyramid_frame = jj1_val;

            if (patch_idx < 0 || patch_idx >= M ||
                gmap_frame < 0 || gmap_frame >= num_gmap_frames ||
                pyramid_frame < 0 || pyramid_frame >= num_frames) {
                continue;
            }

            const size_t frame_offset = static_cast<size_t>(pyramid_frame) * feature_dim * hw;

            for (int i0 = 0; i0 < P; i0++) {
                for (int j0 = 0; j0 < P; j0++) {
                    // -------------------------------------------------------
                    // [STUDY NOTE 3] Coordinate handling & half-precision
                    // -------------------------------------------------------
                    // coords layout: [num_active, 2, P, P]
                    //   Channel 0 = x (column), channel 1 = y (row).
                    //   coord_x_idx jumps over e*(2*P*P) to reach edge e,
                    //   then i0*P + j0 for the patch pixel.
                    //   coord_y_idx = coord_x_idx + P*P to reach channel 1.
                    //
                    // Half-precision emulation:
                    //   Python does coords.half() before floor(), which
                    //   rounds certain values differently in FP16. Example:
                    //   30.999988 in FP32 → 31.0 in FP16 → floor gives 31
                    //   instead of 30. We replicate this with
                    //   float_to_half_to_float() to stay bit-identical.
                    //
                    // coord_scale:
                    //   Level 0 uses scale=1.0 (full resolution).
                    //   Level 1 uses scale=0.25 (quarter resolution pyramid).
                    // -------------------------------------------------------
                    const int coord_x_idx = e * 2 * P * P + i0 * P + j0;
                    const int coord_y_idx = coord_x_idx + P * P;

                    const float raw_x = coords[coord_x_idx];
                    const float raw_y = coords[coord_y_idx];

                    const float x = raw_x * coord_scale;
                    const float y = raw_y * coord_scale;

                    if (!std::isfinite(x) || !std::isfinite(y)) continue;

                    const float x_half = float_to_half_to_float(x);
                    const float y_half = float_to_half_to_float(y);

                    if (!std::isfinite(x_half) || !std::isfinite(y_half)) continue;

                    const float x0 = std::floor(x_half);
                    const float y0 = std::floor(y_half);

                    // -------------------------------------------------------
                    // [STUDY NOTE 4] Gmap pre-extraction (stride 9 → stride 1)
                    // -------------------------------------------------------
                    // gmap layout: [num_gmap_frames, M, feature_dim, D_gmap, D_gmap]
                    //   = [frames][patches][128][3][3]
                    //
                    // To fetch the feature vector for patch pixel (i0,j0),
                    // we need gmap[frame][patch][f][gmap_i][gmap_j] for all f.
                    //
                    // ORIGINAL access pattern:
                    //   gmap[gmap_base + f * 9]  (stride 9 between channels)
                    //   This strided access means the CPU loads 9 floats but
                    //   only uses 1, wasting 8/9 of each cache line fetch.
                    //   It also prevents auto-vectorization (SIMD needs
                    //   contiguous data).
                    //
                    // OPTIMIZED: copy all 128 channels into gmap_local[f]
                    //   (stride 1 = sequential). After this, the dot-product
                    //   loop accesses gmap_local[0], gmap_local[1], ...
                    //   which is perfect for SIMD and cache lines.
                    //
                    // Cost: 128 scattered reads (once per patch pixel).
                    // Benefit: 128 × 64 = 8192 accesses become stride-1.
                    // -------------------------------------------------------
                    const int gmap_i = i0 + gmap_center_offset;
                    const int gmap_j = j0 + gmap_center_offset;
                    const size_t gmap_base =
                        static_cast<size_t>(gmap_frame) * M * feature_dim * gmap_ch_stride +
                        static_cast<size_t>(patch_idx)  * feature_dim * gmap_ch_stride +
                        static_cast<size_t>(gmap_i) * D_gmap +
                        static_cast<size_t>(gmap_j);

                    for (int f = 0; f < feature_dim; f++) {
                        gmap_local[f] = gmap[gmap_base + static_cast<size_t>(f) * gmap_ch_stride];
                    }

                    // -------------------------------------------------------
                    // [STUDY NOTE 5] Bilinear precomputation for 8x8 offsets
                    // -------------------------------------------------------
                    // For R=3, we sample an 8x8 grid of integer offsets:
                    //   gx = floor(x_half) + (corr_jj - R), corr_jj ∈ [0..7]
                    //   gy = floor(y_half) + (corr_ii - R), corr_ii ∈ [0..7]
                    // giving offsets from -3 to +4 around the coordinate.
                    //
                    // INDEXING:
                    //   idx = corr_jj * D_internal + corr_ii
                    //   This stores as [x_offset][y_offset], matching the
                    //   output format after Python's permute(0,1,3,2,4,5)
                    //   where x and y axes are transposed.
                    //
                    // TOLERANCE CHECK:
                    //   Matches PyTorch's grid_sample with padding_mode='zeros':
                    //   coordinates outside [-0.5, W-0.5] are set to zero.
                    //   The tolerance of 0.5 pixels allows sampling at the
                    //   boundary where bilinear interpolation can still
                    //   reference valid pixels via clamping.
                    //
                    // CLAMP-THEN-FLOOR:
                    //   px = clamp(gx, 0, W-1), then bx0 = floor(px).
                    //   Equivalent to the original floor-then-clamp approach
                    //   but slightly cleaner for edge handling.
                    //
                    // NOTE on bilinear weights at integer coordinates:
                    //   gx and gy are always integers here (floor of half-
                    //   precision coord + integer offset). So bdx = bdy = 0,
                    //   making w00 = 1.0 and w01 = w10 = w11 = 0.0.
                    //   The actual sub-pixel interpolation happens later in
                    //   the 8x8→7x7 reduction step (Study Note 7).
                    // -------------------------------------------------------
                    int num_valid = 0;
                    for (int corr_ii = 0; corr_ii < D_internal; corr_ii++) {
                        for (int corr_jj = 0; corr_jj < D_internal; corr_jj++) {
                            const float gx = x0 + static_cast<float>(corr_jj - R);
                            const float gy = y0 + static_cast<float>(corr_ii - R);

                            const int idx = corr_jj * D_internal + corr_ii;
                            BilinearInfo& bi = bi_params[idx];

                            if (gx < -bi_tolerance || gx > fW1 + bi_tolerance ||
                                gy < -bi_tolerance || gy > fH1 + bi_tolerance) {
                                bi.valid = false;
                                local_corr[idx] = 0.0f;
                                continue;
                            }
                            bi.valid = true;
                            num_valid++;

                            const float px = std::max(0.0f, std::min(gx, fW1));
                            const float py = std::max(0.0f, std::min(gy, fH1));

                            const int bx0 = std::min(static_cast<int>(std::floor(px)), fmap_W - 1);
                            const int by0 = std::min(static_cast<int>(std::floor(py)), fmap_H - 1);
                            const int bx1 = std::min(bx0 + 1, fmap_W - 1);
                            const int by1 = std::min(by0 + 1, fmap_H - 1);

                            const float bdx = px - static_cast<float>(bx0);
                            const float bdy = py - static_cast<float>(by0);

                            bi.w00 = (1.0f - bdx) * (1.0f - bdy);
                            bi.w01 = bdx * (1.0f - bdy);
                            bi.w10 = (1.0f - bdx) * bdy;
                            bi.w11 = bdx * bdy;

                            bi.off00 = frame_offset + static_cast<size_t>(by0) * fmap_W + bx0;
                            bi.off01 = frame_offset + static_cast<size_t>(by0) * fmap_W + bx1;
                            bi.off10 = frame_offset + static_cast<size_t>(by1) * fmap_W + bx0;
                            bi.off11 = frame_offset + static_cast<size_t>(by1) * fmap_W + bx1;

                            local_corr[idx] = 0.0f;
                        }
                    }

                    // -------------------------------------------------------
                    // [STUDY NOTE 6] Channel-outer dot product
                    //                *** THE key optimization ***
                    // -------------------------------------------------------
                    // This is the most performance-critical change.
                    //
                    // ORIGINAL (offset-outer) loop structure:
                    //   for each of 64 offsets:
                    //     for each of 128 channels (f):
                    //       sample = pyramid[frame + f*H*W + y*W + x]
                    //       corr[offset] += gmap[..f..] * sample
                    //
                    //   Problem: Each channel f accesses pyramid at offset
                    //   f * H * W (e.g., for 48x160: stride = 7680 floats =
                    //   30 KB). Over 64 offsets × 128 channels, the CPU
                    //   constantly jumps between distant memory locations,
                    //   thrashing the L1/L2 caches.
                    //
                    // OPTIMIZED (channel-outer) loop structure:
                    //   for each of 128 channels (f):
                    //     g = gmap_local[f]          // stride-1, see Note 4
                    //     ch_off = f * H * W         // one channel plane
                    //     for each of 64 offsets:
                    //       sample = pyramid[bi.off + ch_off]
                    //       corr[offset] += g * sample
                    //
                    //   Why this is faster:
                    //   1. All 64 offsets access NEARBY locations within the
                    //      SAME channel plane (an ~8-pixel neighborhood).
                    //      These fit in just a few cache lines.
                    //   2. Moving to the next channel (f+1) adds H*W to
                    //      all offsets — a sequential stride through memory.
                    //   3. gmap_local[f] is stride-1 (see Study Note 4).
                    //
                    //   Cache behavior comparison:
                    //     Original: 64 × 128 random jumps of ~30 KB = thrash
                    //     Optimized: 128 sequential planes, 64 nearby reads
                    //                per plane = streams nicely through cache
                    //
                    // NUMERICAL NOTE:
                    //   Accumulation order (f=0,1,...,127) is identical to
                    //   the original, so floating-point results are
                    //   bit-identical (no reordering of additions).
                    // -------------------------------------------------------
                    if (num_valid > 0) {
                        for (int f = 0; f < feature_dim; f++) {
                            const float g = gmap_local[f];
                            const size_t ch_off = static_cast<size_t>(f) * hw;

                            for (int idx = 0; idx < D_internal * D_internal; idx++) {
                                if (!bi_params[idx].valid) continue;
                                const BilinearInfo& bi = bi_params[idx];

                                const float sample =
                                    bi.w00 * pyramid[bi.off00 + ch_off] +
                                    bi.w01 * pyramid[bi.off01 + ch_off] +
                                    bi.w10 * pyramid[bi.off10 + ch_off] +
                                    bi.w11 * pyramid[bi.off11 + ch_off];

                                local_corr[idx] += g * sample;
                            }
                        }
                    }

                    // --- Save 8x8 debug buffer if requested ---
                    if (corr_8x8_out) {
                        for (int corr_jj = 0; corr_jj < D_internal; corr_jj++) {
                            for (int corr_ii = 0; corr_ii < D_internal; corr_ii++) {
                                const size_t dst = static_cast<size_t>(e) * D_internal * D_internal * P * P +
                                                   static_cast<size_t>(corr_jj) * D_internal * P * P +
                                                   static_cast<size_t>(corr_ii) * P * P +
                                                   static_cast<size_t>(i0) * P + j0;
                                if (dst < corr_8x8_total) {
                                    corr_8x8_out[dst] = local_corr[corr_jj * D_internal + corr_ii];
                                }
                            }
                        }
                    }

                    // -------------------------------------------------------
                    // [STUDY NOTE 7] Bilinear 8x8 → 7x7 reduction
                    // -------------------------------------------------------
                    // The Python code computes correlation at 8x8 integer
                    // offsets [-R, R+1] = [-3, +4], then uses bilinear
                    // interpolation with the fractional part of the
                    // coordinate to produce a 7x7 output grid [-R, R].
                    //
                    // dx, dy = fractional parts of the half-precision coords:
                    //   dx = half(x_half - floor(x_half))
                    //   dy = half(y_half - floor(y_half))
                    //   These are also passed through float_to_half_to_float
                    //   to match the Python .half() behavior.
                    //
                    // The 4 weights wr00..wr11 interpolate between adjacent
                    // integer-offset correlation values:
                    //   v00 = corr at offset (out_jj,   out_ii  )
                    //   v01 = corr at offset (out_jj+1, out_ii  )
                    //   v10 = corr at offset (out_jj,   out_ii+1)
                    //   v11 = corr at offset (out_jj+1, out_ii+1)
                    //   val = wr00*v00 + wr01*v01 + wr10*v10 + wr11*v11
                    //
                    // Output layout: [e, out_jj, out_ii, i0, j0]
                    //   = [edge, corr_x, corr_y, patch_y, patch_x]
                    //   This matches Python's permute(0,1,3,2,4,5) output.
                    // -------------------------------------------------------
                    const float dx = float_to_half_to_float(x_half - x0);
                    const float dy = float_to_half_to_float(y_half - y0);

                    const float wr00 = (1.0f - dx) * (1.0f - dy);
                    const float wr01 = dx * (1.0f - dy);
                    const float wr10 = (1.0f - dx) * dy;
                    const float wr11 = dx * dy;

                    for (int out_jj = 0; out_jj < D_output; out_jj++) {
                        for (int out_ii = 0; out_ii < D_output; out_ii++) {
                            const float v00 = local_corr[out_jj * D_internal + out_ii];
                            const float v01 = local_corr[(out_jj + 1) * D_internal + out_ii];
                            const float v10 = local_corr[out_jj * D_internal + (out_ii + 1)];
                            const float v11 = local_corr[(out_jj + 1) * D_internal + (out_ii + 1)];

                            const float val = wr00 * v00 + wr01 * v01 + wr10 * v10 + wr11 * v11;

                            const size_t out_idx =
                                static_cast<size_t>(e) * D_output * D_output * P * P +
                                static_cast<size_t>(out_jj) * D_output * P * P +
                                static_cast<size_t>(out_ii) * P * P +
                                static_cast<size_t>(i0) * P + j0;

                            if (out_idx < corr_output_size) {
                                corr_out[out_idx] = val;
                            }
                        }
                    }
                } // j0
            } // i0
        } // e
    } // omp parallel
}


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
