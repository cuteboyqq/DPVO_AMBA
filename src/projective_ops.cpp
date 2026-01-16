#include "projective_ops.hpp"
#include <algorithm>
#include <cmath>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#ifdef SPDLOG_USE_SYSLOG
#include <spdlog/sinks/syslog_sink.h>
#endif

namespace pops {

static constexpr float MIN_DEPTH = 0.2f;

// ------------------------------------------------------------
// iproj(): inverse projection
// ------------------------------------------------------------
inline void iproj(
    float x, float y, float d,
    const float* intr,   // fx fy cx cy
    float& X, float& Y, float& Z, float& W)
{
    float fx = intr[0];
    float fy = intr[1];
    float cx = intr[2];
    float cy = intr[3];

    float xn = (x - cx) / fx;
    float yn = (y - cy) / fy;

    X = xn;
    Y = yn;
    Z = 1.0f;
    W = d;
}

// ------------------------------------------------------------
// proj(): projection
// ------------------------------------------------------------
inline void proj(
    float X, float Y, float Z, float W,
    const float* intr,
    float& u, float& v)
{
    float fx = intr[0];
    float fy = intr[1];
    float cx = intr[2];
    float cy = intr[3];

    float z = std::max(Z, 0.1f);
    float d = 1.0f / z;

    u = fx * (d * X) + cx;
    v = fy * (d * Y) + cy;
}

// ------------------------------------------------------------
// transform(): main entry
// ------------------------------------------------------------

void transform(
    const SE3* poses,
    const float* patches_flat,
    const float* intrinsics_flat,
    const int* ii,
    const int* jj,
    const int* kk,
    int num_edges,
    int M,
    int P,
    float* coords_out)
{
    for (int e = 0; e < num_edges; e++) {
        // C++ DPVO semantics (different from Python!):
        //   ii[e] = m_pg.m_index[frame][patch] (patch index mapping, NOT frame index)
        //   jj[e] = target frame index
        //   kk[e] = global patch index (frame * M + patch_idx)
        // 
        // Python semantics:
        //   ii = source frame index (for poses and intrinsics)
        //   jj = target frame index
        //   kk = patch index (for patches[:,kk])
        //
        // CRITICAL: In C++, we extract source frame from kk, NOT from ii!
        int j = jj[e];  // target frame index
        int k = kk[e];  // global patch index (frame * M + patch_idx)
        
        // Extract source frame and patch from global patch index k
        int i = k / M;  // source frame index (extracted from kk)
        int patch_idx = k % M;  // patch index within frame
        
        // Transform from frame i to frame j
        const SE3& Ti = poses[i];
        const SE3& Tj = poses[j];
        SE3 Gij = Tj * Ti.inverse();  // Transform from frame i to frame j

        const float* intr_i = &intrinsics_flat[i * 4]; // fx,fy,cx,cy (source frame intrinsics)
        const float* intr_j = &intrinsics_flat[j * 4];  // fx,fy,cx,cy (target frame intrinsics)

        for (int c = 0; c < 3; c++) {} // placeholder if needed later

        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {

                int idx = y * P + x;

                // Use source frame i and patch_idx to index patches
                float px = patches_flat[((i * M + patch_idx) * 3 + 0) * P * P + idx];
                float py = patches_flat[((i * M + patch_idx) * 3 + 1) * P * P + idx];
                float pd = patches_flat[((i * M + patch_idx) * 3 + 2) * P * P + idx];

                // Inverse projection: X0 = [X, Y, Z, W] in homogeneous coordinates
                float X0 = (px - intr_i[2]) / intr_i[0];
                float Y0 = (py - intr_i[3]) / intr_i[1];
                float Z0 = 1.0f;
                float W0 = pd; // inverse depth

                // Transform point: X1 = Gij * X0 (SE3 action on homogeneous coordinates)
                // SE3 act4 formula: X1 = R * [X, Y, Z] + t * W
                Eigen::Vector3f p0_vec(X0, Y0, Z0);
                Eigen::Vector3f p1_vec = Gij.R() * p0_vec + Gij.t * W0;

                // Project: use Z component directly (not Z/W since W is inverse depth)
                float z = std::max(p1_vec.z(), 0.1f);
                float d = 1.0f / z;

                float u = intr_j[0] * (d * p1_vec.x()) + intr_j[2];
                float v = intr_j[1] * (d * p1_vec.y()) + intr_j[3];

                // Store coordinates
                int base = e * 2 * P * P;
                coords_out[base + 0 * P * P + idx] = u;
                coords_out[base + 1 * P * P + idx] = v;
            }
        }
    }
}


// ------------------------------------------------------------
// transformWithJacobians(): transform with Jacobian computation
// ------------------------------------------------------------
// -----------------------------------------------------------------------------
// transformWithJacobians: Reproject patches from source frame i to target frame j
// -----------------------------------------------------------------------------
// Input Parameters:
//   poses: [N] - SE3 camera poses for all frames
//   patches_flat: [N*M*3*P*P] - Flattened patches with inverse depth
//                Patches are stored at 1/4 resolution (scaled by RES=4)
//                Layout: [frame][patch][channel][y][x] where channel 0=x, 1=y, 2=inverse_depth
//   intrinsics_flat: [N*4] - Camera intrinsics [fx, fy, cx, cy] for each frame
//                  **CRITICAL: Intrinsics are SCALED to 1/4 resolution (divided by RES=4)**
//                  Example: If original fx=1660, stored fx=415 (1660/4)
//                  This matches Python: intrinsics / RES where RES=4
//   ii, jj, kk: Edge indices (see comments below for C++ semantics)
//   num_edges: Number of edges to process
//   M: Patches per frame
//   P: Patch size (typically 3)
//
// Output Parameters:
//   coords_out: [num_edges, 2, P, P] - Reprojected 2D coordinates (u, v)
//              **CRITICAL: Coordinates are at 1/4 RESOLUTION (img_W/4, img_H/4)**
//              This matches the feature map resolution (fmap1_H, fmap1_W)
//              Layout: [edge][channel][y][x] where channel 0=u, channel 1=v
//              Example: For 1920x1080 image, coords are in range [0, 480] x [0, 270]
//              Python equivalent: coords from pops.transform() using scaled intrinsics
//   Ji_out, Jj_out, Jz_out: Jacobians for bundle adjustment (optional)
//   valid_out: Validity mask (1.0=valid, 0.0=invalid)
//
// Resolution Details:
//   - Input patches: Stored at 1/4 resolution (px, py scaled by RES=4)
//   - Input intrinsics: Scaled to 1/4 resolution (fx, fy, cx, cy divided by RES=4)
//   - Output coords: At 1/4 resolution (matching feature map resolution)
//   - This matches Python behavior where intrinsics are divided by RES=4
// -----------------------------------------------------------------------------
void transformWithJacobians(
    const SE3* poses,
    const float* patches_flat,
    const float* intrinsics_flat,
    const int* ii,
    const int* jj,
    const int* kk,
    int num_edges,
    int M,
    int P,
    float* coords_out,      // Output: [num_edges, 2, P, P] - Reprojected (u, v) at 1/4 resolution
    float* Ji_out,          // [num_edges, 2, P, P, 6] flattened
    float* Jj_out,          // [num_edges, 2, P, P, 6] flattened
    float* Jz_out,          // [num_edges, 2, P, P, 1] flattened
    float* valid_out)       // [num_edges, P, P] flattened
{
    for (int e = 0; e < num_edges; e++) {
        // C++ DPVO semantics (different from Python!):
        //   ii[e] = m_pg.m_index[frame][patch] (patch index mapping, NOT frame index)
        //   jj[e] = target frame index
        //   kk[e] = global patch index (frame * M + patch_idx)
        // 
        // Python semantics:
        //   ii = source frame index (for poses and intrinsics)
        //   jj = target frame index
        //   kk = patch index (for patches[:,kk])
        //
        // CRITICAL: In C++, we extract source frame from kk, NOT from ii!
        int j = jj[e];  // target frame index
        int k = kk[e];  // global patch index (frame * M + patch_idx)
        
        // Extract source frame and patch from global patch index k
        int i = k / M;  // source frame index (extracted from kk)
        int patch_idx = k % M;  // patch index within frame
        
        // Transform from frame i to frame j
        const SE3& Ti = poses[i];
        const SE3& Tj = poses[j];
        SE3 Gij = Tj * Ti.inverse();  // Transform from frame i to frame j

        const float* intr_i = &intrinsics_flat[i * 4]; // fx,fy,cx,cy (source frame intrinsics)
        const float* intr_j = &intrinsics_flat[j * 4];  // fx,fy,cx,cy (target frame intrinsics)
        
        float fx_j = intr_j[0];
        float fy_j = intr_j[1];
        float cx_j = intr_j[2];
        float cy_j = intr_j[3];

        // Get patch center coordinates
        // Use source frame i and patch_idx to index patches
        int center_idx = (P / 2) * P + (P / 2);
        int patch_base_idx = ((i * M + patch_idx) * 3 + 0) * P * P + center_idx;
        
        // Validate patch index bounds (safety check - reasonable bounds)
        if (i < 0 || i >= 100 || patch_idx < 0 || patch_idx >= M) {
            // Invalid frame or patch index - set coordinates to NaN to mark as invalid
            for (int y = 0; y < P; y++) {
                for (int x = 0; x < P; x++) {
                    int idx = y * P + x;
                    int base = e * 2 * P * P;
                    coords_out[base + 0 * P * P + idx] = std::numeric_limits<float>::quiet_NaN();
                    coords_out[base + 1 * P * P + idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
            if (valid_out) {
                for (int y = 0; y < P; y++) {
                    for (int x = 0; x < P; x++) {
                        int idx = y * P + x;
                        valid_out[e * P * P + idx] = 0.0f;
                    }
                }
            }
            continue;  // Skip this edge
        }
        
        float px = patches_flat[patch_base_idx];
        float py = patches_flat[patch_base_idx + P * P];  // Channel 1 offset
        float pd = patches_flat[patch_base_idx + 2 * P * P];  // Channel 2 offset
        
        // Validate patch data (check for NaN/Inf and reasonable depth)
        if (!std::isfinite(px) || !std::isfinite(py) || !std::isfinite(pd) || pd <= 0.0f || pd > 100.0f) {
            // Invalid patch data - mark as invalid
            for (int y = 0; y < P; y++) {
                for (int x = 0; x < P; x++) {
                    int idx = y * P + x;
                    int base = e * 2 * P * P;
                    coords_out[base + 0 * P * P + idx] = std::numeric_limits<float>::quiet_NaN();
                    coords_out[base + 1 * P * P + idx] = std::numeric_limits<float>::quiet_NaN();
                }
            }
            if (valid_out) {
                for (int y = 0; y < P; y++) {
                    for (int x = 0; x < P; x++) {
                        int idx = y * P + x;
                        valid_out[e * P * P + idx] = 0.0f;
                    }
                }
            }
            continue;  // Skip this edge
        }

        // Inverse projection at center
        float X0 = (px - intr_i[2]) / intr_i[0];
        float Y0 = (py - intr_i[3]) / intr_i[1];
        float Z0 = 1.0f;
        float W0 = pd; // inverse depth

        // Transform point: X1 = Gij * X0 (SE3 action on homogeneous coordinates)
        // SE3 act4 formula: X1 = R * [X, Y, Z] + t * W
        Eigen::Vector3f p0_vec(X0, Y0, Z0);
        Eigen::Vector3f p1_vec = Gij.R() * p0_vec + Gij.t * W0;
        
        // For Jacobian computation, we need the transformed point
        float X1 = p1_vec.x();
        float Y1 = p1_vec.y();
        float Z1 = p1_vec.z();
        float H1 = W0; // homogeneous coordinate (inverse depth)

        // Project
        float z = std::max(Z1, 0.1f);
        float d = 1.0f / z;

        float u = fx_j * (d * X1) + cx_j;
        float v = fy_j * (d * Y1) + cy_j;

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
        
        // Diagnostic: Log first edge, first pixel
        bool log_first_pixel = (e == 0);
        
        // Base index for this edge's coordinates
        int base = e * 2 * P * P;
        
        // Store coordinates for all patch pixels
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int idx = y * P + x;
                
                // Get patch pixel coordinates
                // Use source frame i and patch_idx to index patches
                float px_pix = patches_flat[((i * M + patch_idx) * 3 + 0) * P * P + idx];
                float py_pix = patches_flat[((i * M + patch_idx) * 3 + 1) * P * P + idx];
                float pd_pix = patches_flat[((i * M + patch_idx) * 3 + 2) * P * P + idx];

                // Diagnostic: Log first pixel values
                if (log_first_pixel && y == 0 && x == 0 && logger) {
                    logger->info("transformWithJacobians: Edge[0] pixel[0][0] - i={}, patch_idx={}, "
                                 "px_pix={:.2f}, py_pix={:.2f}, pd_pix={:.6f}, "
                                 "intr_i=[{:.2f}, {:.2f}, {:.2f}, {:.2f}], "
                                 "intr_j=[{:.2f}, {:.2f}, {:.2f}, {:.2f}]",
                                 i, patch_idx, px_pix, py_pix, pd_pix,
                                 intr_i[0], intr_i[1], intr_i[2], intr_i[3],
                                 intr_j[0], intr_j[1], intr_j[2], intr_j[3]);
                }

                // Inverse projection: returns homogeneous coordinates [X, Y, Z, W]
                // where X, Y are normalized coordinates, Z=1, W=inverse_depth
                float X0_pix = (px_pix - intr_i[2]) / intr_i[0];
                float Y0_pix = (py_pix - intr_i[3]) / intr_i[1];
                float Z0_pix = 1.0f;
                float W0_pix = pd_pix; // inverse depth

                // Diagnostic: Log inverse projection
                if (log_first_pixel && y == 0 && x == 0 && logger) {
                    logger->info("transformWithJacobians: Edge[0] inverse_proj - X0={:.6f}, Y0={:.6f}, Z0={:.6f}, W0={:.6f}",
                                 X0_pix, Y0_pix, Z0_pix, W0_pix);
                }

                // Transform point: SE3 action on homogeneous coordinates [X, Y, Z, W]
                // Python: X1 = Gij * X0 where X0 = [X, Y, Z, W] in homogeneous coordinates
                // SE3 act4 formula (from lietorch/include/se3.h):
                //   X1 = R * [X, Y, Z] + t * W
                //   Y1 = ...
                //   Z1 = ...
                //   W1 = W (unchanged)
                // This is NOT R * [X/W, Y/W, Z/W] + t!
                Eigen::Vector3f p0_vec(X0_pix, Y0_pix, Z0_pix); // [X, Y, Z] part
                Eigen::Vector3f p1_vec = Gij.R() * p0_vec + Gij.t * W0_pix; // Transform: R*[X,Y,Z] + t*W
                float X1_pix = p1_vec.x();
                float Y1_pix = p1_vec.y();
                float Z1_pix = p1_vec.z();
                float H1_pix = W0_pix; // W unchanged

                // Diagnostic: Log transform
                if (log_first_pixel && y == 0 && x == 0 && logger) {
                    Eigen::Vector3f t_i = poses[i].t;
                    Eigen::Vector3f t_j = poses[j].t;
                    logger->info("transformWithJacobians: Edge[0] transform - "
                                 "pose_i.t=({:.3f}, {:.3f}, {:.3f}), pose_j.t=({:.3f}, {:.3f}, {:.3f}), "
                                 "Gij.t=({:.6f}, {:.6f}, {:.6f}), "
                                 "p0_vec=({:.6f}, {:.6f}, {:.6f}), p1_vec=({:.6f}, {:.6f}, {:.6f}), "
                                 "X1=({:.6f}, {:.6f}, {:.6f}, {:.6f})",
                                 t_i.x(), t_i.y(), t_i.z(),
                                 t_j.x(), t_j.y(), t_j.z(),
                                 Gij.t.x(), Gij.t.y(), Gij.t.z(),
                                 p0_vec.x(), p0_vec.y(), p0_vec.z(),
                                 p1_vec.x(), p1_vec.y(), p1_vec.z(),
                                 X1_pix, Y1_pix, Z1_pix, H1_pix);
                }

                // Project
                // Check if point is behind camera (Z < 0.1) - reject if so
                bool is_valid = (Z1_pix >= 0.1f);
                
                float u_pix, v_pix;
                if (is_valid) {
                    float z_pix = Z1_pix;
                    float d_pix = 1.0f / z_pix;

                    // Compute projection
                    float u_pix_computed = fx_j * (d_pix * X1_pix) + cx_j;
                    float v_pix_computed = fy_j * (d_pix * Y1_pix) + cy_j;
                    
                    // Check if computed values are finite before bounds check
                    bool computed_is_finite = std::isfinite(u_pix_computed) && std::isfinite(v_pix_computed);
                    
                    // Diagnostic: Log computed values before bounds check
                    if (log_first_pixel && y == 0 && x == 0 && logger) {
                        logger->info("transformWithJacobians: Edge[0] projection computation - "
                                     "z_pix={:.6f}, d_pix={:.6f}, "
                                     "X1={:.6f}, Y1={:.6f}, Z1={:.6f}, "
                                     "d_pix*X1={:.6f}, d_pix*Y1={:.6f}, "
                                     "u_pix_computed={:.2f}, v_pix_computed={:.2f}, "
                                     "computed_is_finite={}",
                                     z_pix, d_pix,
                                     X1_pix, Y1_pix, Z1_pix,
                                     d_pix * X1_pix, d_pix * Y1_pix,
                                     u_pix_computed, v_pix_computed,
                                     computed_is_finite);
                    }

                    // Check if projection is reasonable (within reasonable bounds for feature map)
                    // Feature map is at 1/4 resolution, so bounds are roughly [0, fmap_W*4] and [0, fmap_H*4]
                    // But we allow some margin for sub-pixel accuracy
                    float max_u = fx_j * 20.0f + cx_j;  // Allow up to 20x normalized image width
                    float max_v = fy_j * 20.0f + cy_j;  // Allow up to 20x normalized image height
                    
                    bool out_of_bounds = (std::abs(u_pix_computed) > max_u || std::abs(v_pix_computed) > max_v);
                    
                    if (out_of_bounds || !computed_is_finite) {
                        // Projection is way out of bounds or invalid - likely due to bad poses
                        is_valid = false;
                        u_pix = std::numeric_limits<float>::quiet_NaN();
                        v_pix = std::numeric_limits<float>::quiet_NaN();
                        if (log_first_pixel && y == 0 && x == 0 && logger) {
                            logger->warn("transformWithJacobians: Edge[0] projection out of reasonable bounds - "
                                         "u_pix_computed={:.2f}, v_pix_computed={:.2f}, "
                                         "max_u={:.2f}, max_v={:.2f}, "
                                         "out_of_bounds={}, computed_is_finite={}. "
                                         "X1=({:.6f}, {:.6f}, {:.6f}), d_pix={:.6f}. "
                                         "This suggests poses may be incorrect.",
                                         u_pix_computed, v_pix_computed,
                                         max_u, max_v,
                                         out_of_bounds, computed_is_finite,
                                         X1_pix, Y1_pix, Z1_pix, d_pix);
                        }
                    } else {
                        u_pix = u_pix_computed;
                        v_pix = v_pix_computed;
                        // Diagnostic: Log valid projection
                        if (log_first_pixel && y == 0 && x == 0 && logger) {
                            logger->info("transformWithJacobians: Edge[0] projection - "
                                         "z_pix={:.6f}, d_pix={:.6f}, u_pix={:.2f}, v_pix={:.2f}",
                                         z_pix, d_pix, u_pix, v_pix);
                        }
                    }
                } else {
                    // Point is behind camera or too close
                    u_pix = std::numeric_limits<float>::quiet_NaN();
                    v_pix = std::numeric_limits<float>::quiet_NaN();
                    if (log_first_pixel && y == 0 && x == 0 && logger) {
                        logger->warn("transformWithJacobians: Edge[0] point behind camera - "
                                     "Z1={:.6f} < 0.1, X1=({:.2f}, {:.2f}, {:.2f}). "
                                     "This indicates incorrect poses or transform.",
                                     Z1_pix, X1_pix, Y1_pix, Z1_pix);
                    }
                }

                // Store coordinates
                coords_out[base + 0 * P * P + idx] = u_pix;
                coords_out[base + 1 * P * P + idx] = v_pix;
                
                // Validity: Z > 0.2 and projection is valid
                if (valid_out) {
                    valid_out[e * P * P + idx] = (is_valid && Z1_pix > 0.2f) ? 1.0f : 0.0f;
                }
            }
        }

        // Compute Jacobians at patch center only
        // Python computes at center: X1[...,p//2,p//2,:]
        float X = X1;
        float Y = Y1;
        float Z = Z1;
        float H = H1;

        // Depth check for Jacobian computation
        float d_jac = 0.0f;
        if (std::abs(Z) > 0.2f) {
            d_jac = 1.0f / Z;
        }

        // Ja: Jacobian of transformed point w.r.t. SE3 parameters [4, 6]
        // For SE3: [H, o, o, o, Z, -Y] for first row, etc.
        Eigen::Matrix<float, 4, 6> Ja;
        Ja.setZero();
        Ja(0, 0) = H;  Ja(0, 4) = Z;   Ja(0, 5) = -Y;
        Ja(1, 1) = H;  Ja(1, 3) = -Z;  Ja(1, 5) = X;
        Ja(2, 2) = H;  Ja(2, 3) = Y;   Ja(2, 4) = -X;
        // Row 3 (homogeneous) is all zeros

        // Jp: Jacobian of projection w.r.t. 3D point [2, 4]
        Eigen::Matrix<float, 2, 4> Jp;
        Jp.setZero();
        Jp(0, 0) = fx_j * d_jac;
        Jp(0, 2) = -fx_j * X * d_jac * d_jac;
        Jp(1, 1) = fy_j * d_jac;
        Jp(1, 2) = -fy_j * Y * d_jac * d_jac;

        // Jj: Jacobian w.r.t. pose j = Jp @ Ja [2, 6]
        Eigen::Matrix<float, 2, 6> Jj = Jp * Ja;

        // Ji: Jacobian w.r.t. pose i = -Gij.adjT(Jj) [2, 6]
        Eigen::Matrix<float, 2, 6> Ji = -Gij.adjointT(Jj);

        // Jz: Jacobian w.r.t. inverse depth = Jp @ Gij.matrix()[...,:,3:] [2, 1]
        // Gij.matrix()[...,:,3:] is the translation column [4, 1]
        Eigen::Matrix4f Gij_mat = Gij.matrix();
        Eigen::Vector4f t_col = Gij_mat.col(3); // translation column
        Eigen::Matrix<float, 2, 1> Jz = Jp * t_col;

        // Store Jacobians for all patch pixels (currently using center values)
        // TODO: Could compute per-pixel Jacobians if needed
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int idx = y * P + x;
                int base = e * 2 * P * P * 6; // [num_edges, 2, P, P, 6]
                
                // Ji: [num_edges, 2, P, P, 6]
                for (int c = 0; c < 2; c++) {
                    for (int param = 0; param < 6; param++) {
                        int ji_idx = base + c * P * P * 6 + idx * 6 + param;
                        Ji_out[ji_idx] = Ji(c, param);
                    }
                }
                
                // Jj: [num_edges, 2, P, P, 6]
                for (int c = 0; c < 2; c++) {
                    for (int param = 0; param < 6; param++) {
                        int jj_idx = base + c * P * P * 6 + idx * 6 + param;
                        Jj_out[jj_idx] = Jj(c, param);
                    }
                }
                
                // Jz: [num_edges, 2, P, P, 1]
                int jz_base = e * 2 * P * P * 1;
                for (int c = 0; c < 2; c++) {
                    int jz_idx = jz_base + c * P * P * 1 + idx * 1;
                    Jz_out[jz_idx] = Jz(c, 0);
                }
            }
        }
    }
}

// ------------------------------------------------------------
// flow_mag(): Compute flow magnitude for motion estimation
// ------------------------------------------------------------
void flow_mag(
    const SE3* poses,
    const float* patches_flat,
    const float* intrinsics_flat,
    const int* ii,
    const int* jj,
    const int* kk,
    int num_edges,
    int M,
    int P,
    float beta,
    float* flow_out,
    float* valid_out)
{
    // Allocate temporary buffers for coordinates
    std::vector<float> coords0(num_edges * 2 * P * P);  // transform from i to i (identity)
    std::vector<float> coords1(num_edges * 2 * P * P);   // transform from i to j (full)
    std::vector<float> coords2(num_edges * 2 * P * P);   // transform from i to j (translation only)
    std::vector<float> valid_temp(num_edges * P * P);    // validity mask
    
    // coords0: transform from frame i to frame i (identity - original coordinates)
    // We can use transform() with jj = ii
    std::vector<int> jj_identity(num_edges);
    for (int e = 0; e < num_edges; e++) {
        jj_identity[e] = ii[e];  // target = source (identity transform)
    }
    transform(poses, patches_flat, intrinsics_flat, ii, jj_identity.data(), kk, num_edges, M, P, coords0.data());
    
    // coords1: transform from frame i to frame j (full transform)
    transform(poses, patches_flat, intrinsics_flat, ii, jj, kk, num_edges, M, P, coords1.data());
    
    // coords2: transform from frame i to frame j (translation only)
    // We need to modify the transform to use translation only (identity rotation)
    for (int e = 0; e < num_edges; e++) {
        int i = ii[e];
        int j = jj[e];
        int k = kk[e];
        
        const SE3& Ti = poses[i];
        const SE3& Tj = poses[j];
        SE3 Gij = Tj * Ti.inverse();
        
        // Create translation-only transform: keep translation, set rotation to identity
        SE3 Gij_tonly;
        Gij_tonly.t = Gij.t;  // Keep translation
        Gij_tonly.q = Eigen::Quaternionf::Identity();  // Identity rotation
        
        const float* intr_i = &intrinsics_flat[i * 4];
        const float* intr_j = &intrinsics_flat[j * 4];
        
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int idx = y * P + x;
                
                // Inverse projection
                float px = patches_flat[((i * M + k) * 3 + 0) * P * P + idx];
                float py = patches_flat[((i * M + k) * 3 + 1) * P * P + idx];
                float pd = patches_flat[((i * M + k) * 3 + 2) * P * P + idx];
                
                float X0 = (px - intr_i[2]) / intr_i[0];
                float Y0 = (py - intr_i[3]) / intr_i[1];
                float Z0 = 1.0f;
                float W0 = pd;
                
                // Transform with translation only (no rotation)
                Eigen::Vector3f p0(X0, Y0, Z0);
                Eigen::Vector3f p1 = p0 + Gij_tonly.t * W0;  // Only translation, no rotation
                
                // Project
                float z = std::max(p1.z(), 0.1f);
                float d = 1.0f / z;
                
                float u = intr_j[0] * (d * p1.x()) + intr_j[2];
                float v = intr_j[1] * (d * p1.y()) + intr_j[3];
                
                // Store coordinates
                int base = e * 2 * P * P;
                coords2[base + 0 * P * P + idx] = u;
                coords2[base + 1 * P * P + idx] = v;
                
                // Compute validity (Z > 0.2)
                if (valid_out != nullptr) {
                    valid_temp[e * P * P + idx] = (p1.z() > 0.2f) ? 1.0f : 0.0f;
                }
            }
        }
    }
    
    // Compute flow magnitudes: beta * flow1 + (1-beta) * flow2
    // flow1 = norm(coords1 - coords0)
    // flow2 = norm(coords2 - coords0)
    for (int e = 0; e < num_edges; e++) {
        float sum_flow = 0.0f;
        float sum_valid = 0.0f;
        
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int idx = y * P + x;
                int base = e * 2 * P * P;
                
                // Get coordinates
                float u0 = coords0[base + 0 * P * P + idx];
                float v0 = coords0[base + 1 * P * P + idx];
                float u1 = coords1[base + 0 * P * P + idx];
                float v1 = coords1[base + 1 * P * P + idx];
                float u2 = coords2[base + 0 * P * P + idx];
                float v2 = coords2[base + 1 * P * P + idx];
                
                // Compute flow1 and flow2
                float dx1 = u1 - u0;
                float dy1 = v1 - v0;
                float flow1 = std::sqrt(dx1 * dx1 + dy1 * dy1);
                
                float dx2 = u2 - u0;
                float dy2 = v2 - v0;
                float flow2 = std::sqrt(dx2 * dx2 + dy2 * dy2);
                
                // Combined flow
                float flow = beta * flow1 + (1.0f - beta) * flow2;
                
                // Check validity (use coords1 validity)
                bool valid = true;
                if (valid_out != nullptr) {
                    valid = (valid_temp[e * P * P + idx] > 0.5f);
                }
                
                if (valid) {
                    sum_flow += flow;
                    sum_valid += 1.0f;
                }
            }
        }
        
        // Mean flow over valid pixels
        flow_out[e] = (sum_valid > 0.0f) ? (sum_flow / sum_valid) : 0.0f;
        
        if (valid_out != nullptr) {
            valid_out[e] = (sum_valid > 0.0f) ? 1.0f : 0.0f;
        }
    }
}

} // namespace pops
