#include "projective_ops.hpp"
#include "ba_file_io.hpp"
#include "target_frame.hpp"
#include "patch_graph.hpp"  // For PatchGraph::N
#include <algorithm>
#include <cmath>
#include <future>
#include <mutex>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <sys/stat.h>  // For mkdir
#include <sys/types.h>
#ifdef SPDLOG_USE_SYSLOG
#include <spdlog/sinks/syslog_sink.h>
#endif

// =====================================================================
// Async debug file writer — buffers data during processing, then writes
// all files in a background thread so the main loop doesn't stall.
// =====================================================================
static std::future<void> g_asyncWriteFuture;
static std::mutex        g_asyncWriteMutex;

// Wait for any previous async write to finish (call before starting a new batch)
static void waitForPreviousAsyncWrite() {
    std::lock_guard<std::mutex> lock(g_asyncWriteMutex);
    if (g_asyncWriteFuture.valid()) {
        g_asyncWriteFuture.get();
    }
}

// Represents a single file-write task
struct DebugWriteTask {
    enum Type { SE3_SAVE, EIGEN_2x6, EIGEN_2x1, FLOAT_ARRAY, COORDS };
    Type type;
    std::string filename;
    // For SE3: 7 floats [tx, ty, tz, qx, qy, qz, qw]
    // For Eigen_2x6: 12 floats (row-major)
    // For Eigen_2x1: 2 floats (row-major)
    std::vector<float> data;
    size_t count = 0;  // for FLOAT_ARRAY: number of floats
};

// Execute a batch of write tasks (runs in background thread)
static void executeWriteTasks(std::vector<DebugWriteTask> tasks) {
    auto logger = spdlog::get("dpvo");
    for (auto& task : tasks) {
        std::ofstream file(task.filename, std::ios::binary);
        if (!file.is_open()) {
            if (logger) logger->error("Async save: Failed to open {}", task.filename);
            continue;
        }
        file.write(reinterpret_cast<const char*>(task.data.data()), task.data.size() * sizeof(float));
        file.close();
        if (logger) {
            switch (task.type) {
                case DebugWriteTask::SE3_SAVE:
                    logger->info("Saved SE3 object to {}", task.filename);
                    break;
                case DebugWriteTask::EIGEN_2x6:
                    logger->info("Saved Eigen matrix (2x6) to {}", task.filename);
                    break;
                case DebugWriteTask::EIGEN_2x1:
                    logger->info("Saved Eigen matrix (2x1) to {}", task.filename);
                    break;
                case DebugWriteTask::FLOAT_ARRAY:
                    logger->info("Saved {} floats to {}", task.count, task.filename);
                    break;
                default:
                    break;
            }
        }
    }
    if (logger) logger->info("Async debug write: {} files written in background", tasks.size());
}

namespace pops {

static constexpr float MIN_DEPTH = 0.2f;

// Helper function to get bin_file directory path and ensure it exists
static std::string get_bin_file_path(const std::string& filename) {
    const std::string bin_dir = "bin_file";
    
    // Create directory if it doesn't exist
    struct stat info;
    if (stat(bin_dir.c_str(), &info) != 0) {
        // Directory doesn't exist, create it
        #ifdef _WIN32
        mkdir(bin_dir.c_str());
        #else
        mkdir(bin_dir.c_str(), 0755);
        #endif
    }
    
    return bin_dir + "/" + filename;
}

// NOTE: save_jacobians() is no longer used — Jacobian saves are now buffered
// in transformWithJacobians() and written asynchronously via DebugWriteTask.
// Kept here for reference only.
#if 0
static void save_jacobians(int frame_num, int edge_idx, 
                           const Eigen::Matrix<float, 2, 6>& Ji,
                           const Eigen::Matrix<float, 2, 6>& Jj,
                           const Eigen::Matrix<float, 2, 1>& Jz)
{
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    std::string frame_suffix = std::to_string(frame_num);
    std::string edge_suffix = std::to_string(edge_idx);
    ba_file_io::save_eigen_matrix(get_bin_file_path("reproject_Ji_frame" + frame_suffix + "_edge" + edge_suffix + ".bin"), Ji, logger);
    ba_file_io::save_eigen_matrix(get_bin_file_path("reproject_Jj_frame" + frame_suffix + "_edge" + edge_suffix + ".bin"), Jj, logger);
    ba_file_io::save_eigen_matrix(get_bin_file_path("reproject_Jz_frame" + frame_suffix + "_edge" + edge_suffix + ".bin"), Jz, logger);
}
#endif

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
// ============================================================================
// transformWithJacobians: Reproject patches from source frame i to target frame j
//                         and compute Jacobians for bundle adjustment
// ============================================================================
// 
// ALGORITHM OVERVIEW:
// -------------------
// This function implements the core reprojection operation in DPVO:
//   1. Read patch coordinates and inverse depth from source frame i
//   2. Inverse project to get 3D point in normalized camera coordinates
//   3. Transform 3D point from frame i to frame j using SE3 poses
//   4. Project transformed 3D point to image coordinates in frame j
//   5. Compute Jacobians w.r.t. poses (i, j) and inverse depth for optimization
//
// MATHEMATICAL FLOW:
// ------------------
// Step 1: Inverse Projection (2D → 3D normalized)
//   Input:  (px, py) = pixel coordinates in frame i (at 1/4 resolution)
//           pd = inverse depth (1/depth)
//   Output: [X0, Y0, Z0=1, W0=pd] in normalized camera coordinates
//   Formula: X0 = (px - cx) / fx, Y0 = (py - cy) / fy, Z0 = 1, W0 = pd
//
// Step 2: SE3 Transform (3D point transformation)
//   Input:  Point [X0, Y0, Z0, W0] in frame i's camera coordinates
//           Ti = pose of frame i (world-to-camera)
//           Tj = pose of frame j (world-to-camera)
//   Output: [X1, Y1, Z1, W1=W0] in frame j's camera coordinates
//   Formula: Gij = Tj * Ti^-1  (transform from frame i to frame j)
//            [X1, Y1, Z1] = Gij.R * [X0, Y0, Z0] + Gij.t * W0
//            W1 = W0 (homogeneous coordinate unchanged)
//
// Step 3: Forward Projection (3D → 2D)
//   Input:  [X1, Y1, Z1] in frame j's camera coordinates
//   Output: (u, v) = pixel coordinates in frame j
//   Formula: depth = 1/Z1, u = fx * (X1/Z1) + cx, v = fy * (Y1/Z1) + cy
//
// Step 4: Jacobian Computation
//   Ji = Jacobian of (u,v) w.r.t. pose i (6 parameters: 3 translation + 3 rotation)
//   Jj = Jacobian of (u,v) w.r.t. pose j (6 parameters)
//   Jz = Jacobian of (u,v) w.r.t. inverse depth (1 parameter)
//
// ============================================================================
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
    float* Ji_out,          // [num_edges, 2, P, P, 6] flattened - Jacobian w.r.t. pose i
    float* Jj_out,          // [num_edges, 2, P, P, 6] flattened - Jacobian w.r.t. pose j
    float* Jz_out,          // [num_edges, 2, P, P, 1] flattened - Jacobian w.r.t. inverse depth
    float* valid_out,       // [num_edges, P, P] flattened - Validity mask (1=valid, 0=invalid)
    int frame_num,          // Optional: frame number for saving intermediate values
    bool save_intermediates) // Optional: whether to save Ti, Tj, Gij, Jacobians
{
    // ========================================================================
    // Async debug file buffer: collect all per-edge saves here, flush after loop
    // ========================================================================
    bool do_save = save_intermediates && frame_num >= 0 && frame_num == TARGET_FRAME;
    std::vector<DebugWriteTask> writeTasks;
    if (do_save) {
        // Reserve space: 3 SE3 + 3 Jacobians per edge + 1 coords at end
        writeTasks.reserve(num_edges * 6 + 2);
    }

    // ========================================================================
    // MAIN LOOP: Process each edge (connection between patch and frame)
    // ========================================================================
    for (int e = 0; e < num_edges; e++) {
        // ====================================================================
        // STEP 0: Extract frame and patch indices
        // ====================================================================
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
        int j = jj[e];  // target frame index (where we want to reproject to)
        int k = kk[e];  // global patch index (frame * M + patch_idx)
        
        // Extract source frame and patch from global patch index k
        int i = k / M;  // source frame index (where patch was originally extracted)
        int patch_idx = k % M;  // patch index within frame (0 to M-1)
        
        // ====================================================================
        // STEP 1: Get poses and compute relative transform Gij
        // ====================================================================
        // Ti = pose of frame i (world-to-camera transformation)
        // Tj = pose of frame j (world-to-camera transformation)
        // Gij = transform from frame i's camera coordinates to frame j's camera coordinates
        // 
        // Mathematical derivation:
        //   Point in world: P_world
        //   Point in frame i: P_i = Ti * P_world  =>  P_world = Ti^-1 * P_i
        //   Point in frame j: P_j = Tj * P_world = Tj * (Ti^-1 * P_i) = (Tj * Ti^-1) * P_i
        //   Therefore: Gij = Tj * Ti^-1
        const SE3& Ti = poses[i];
        const SE3& Tj = poses[j];
        SE3 Gij = Tj * Ti.inverse();  // Transform from frame i to frame j

        // Buffer SE3 saves for async write (no disk I/O during main loop)
        if (do_save) {
            std::string frame_suffix = std::to_string(frame_num);
            std::string edge_suffix = std::to_string(e);
            auto makeSE3Task = [&](const std::string& prefix, const SE3& se3_obj) {
                DebugWriteTask task;
                task.type = DebugWriteTask::SE3_SAVE;
                task.filename = get_bin_file_path(prefix + frame_suffix + "_edge" + edge_suffix + ".bin");
                Eigen::Vector3f t = se3_obj.t;
                Eigen::Quaternionf q = se3_obj.q;
                task.data = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
                return task;
            };
            writeTasks.push_back(makeSE3Task("reproject_Ti_frame", Ti));
            writeTasks.push_back(makeSE3Task("reproject_Tj_frame", Tj));
            writeTasks.push_back(makeSE3Task("reproject_Gij_frame", Gij));
        }

        // ====================================================================
        // STEP 2: Get camera intrinsics for both frames
        // ====================================================================
        // Intrinsics format: [fx, fy, cx, cy] (at 1/4 resolution, already scaled)
        // fx, fy = focal lengths in pixels
        // cx, cy = principal point (image center) in pixels
        const float* intr_i = &intrinsics_flat[i * 4]; // Source frame intrinsics
        const float* intr_j = &intrinsics_flat[j * 4];  // Target frame intrinsics
        
        float fx_j = intr_j[0];  // Focal length X for target frame
        float fy_j = intr_j[1];  // Focal length Y for target frame
        float cx_j = intr_j[2];  // Principal point X for target frame
        float cy_j = intr_j[3];  // Principal point Y for target frame

        // ====================================================================
        // STEP 3: Read patch data from stored patches
        // ====================================================================
        // CRITICAL: These are the INITIAL coordinates from patchify (random initialization)
        // They were stored when the patch was first extracted from frame i
        // Now we use them to reproject the patch to frame j (target frame)
        //
        // Flow:
        //   1. Frame i: Random coordinates → Extract patches → Store in m_pg.m_patches[i]
        //   2. Frame j: Read stored coordinates from frame i → Reproject to frame j → Use for correlation
        //
        // Patch storage format: [num_frames, num_patches, 3, P, P]
        //   Channel 0: X coordinates (px values)
        //   Channel 1: Y coordinates (py values)
        //   Channel 2: Inverse depth (pd values)
        //
        // Use source frame i and patch_idx to index patches
        int center_idx = (P / 2) * P + (P / 2);  // Center pixel index in P×P patch
        int patch_base_idx = ((i * M + patch_idx) * 3 + 0) * P * P + center_idx;
        
        // Validate patch index bounds (safety check - use PatchGraph::N instead of hardcoded 100)
        // PatchGraph::N is now 4096, so we need to check against the actual buffer size
        if (i < 0 || i >= PatchGraph::N || patch_idx < 0 || patch_idx >= M) {
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
        
        // Read patch center coordinates and inverse depth
        float px = patches_flat[patch_base_idx];                    // X coordinate at patch center
        float py = patches_flat[patch_base_idx + P * P];            // Y coordinate at patch center (Channel 1 offset)
        float pd = patches_flat[patch_base_idx + 2 * P * P];        // Inverse depth at patch center (Channel 2 offset)

        // ====================================================================
        // STEP 4: Validate patch data
        // ====================================================================
        // Check for NaN/Inf and reasonable depth values
        // Invalid data indicates corrupted patches or frames that were removed
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

        // ====================================================================
        // STEP 5: Inverse projection at patch center (for Jacobian computation)
        // ====================================================================
        // Convert pixel coordinates (px, py) to normalized camera coordinates
        // This gives us a 3D point in the normalized image plane (Z=1)
        //
        // Formula:
        //   X0 = (px - cx) / fx   (normalized X coordinate)
        //   Y0 = (py - cy) / fy   (normalized Y coordinate)
        //   Z0 = 1                 (normalized depth, arbitrary scale)
        //   W0 = pd                (inverse depth, used for 3D reconstruction)
        float X0 = (px - intr_i[2]) / intr_i[0];  // Normalized X
        float Y0 = (py - intr_i[3]) / intr_i[1];  // Normalized Y
        float Z0 = 1.0f;                           // Normalized Z (arbitrary scale)
        float W0 = pd;                              // Inverse depth (1/actual_depth)

        // ====================================================================
        // STEP 6: Transform 3D point from frame i to frame j
        // ====================================================================
        // Apply SE3 transformation: X1 = Gij * X0
        // SE3 acts on homogeneous coordinates [X, Y, Z, W] as:
        //   [X1, Y1, Z1] = R * [X0, Y0, Z0] + t * W0
        //   W1 = W0 (homogeneous coordinate unchanged)
        //
        // This is NOT the same as: R * [X0/W0, Y0/W0, Z0/W0] + t
        // The W coordinate (inverse depth) is used directly in the transformation!
        Eigen::Vector3f p0_vec(X0, Y0, Z0);  // [X, Y, Z] part of homogeneous coordinates
        Eigen::Vector3f p1_vec = Gij.R() * p0_vec + Gij.t * W0;  // Transform: R*[X,Y,Z] + t*W
        
        // Extract transformed coordinates
        float X1 = p1_vec.x();  // Transformed X in frame j
        float Y1 = p1_vec.y();  // Transformed Y in frame j
        float Z1 = p1_vec.z();  // Transformed Z in frame j (actual depth)
        float H1 = W0;           // Homogeneous coordinate (inverse depth, unchanged)

        // ====================================================================
        // STEP 7: Forward projection (for Jacobian computation at center)
        // ====================================================================
        // Project 3D point [X1, Y1, Z1] back to 2D pixel coordinates [u, v]
        //
        // Formula:
        //   depth = Z1 (actual depth in frame j)
        //   u = fx * (X1 / Z1) + cx   (X pixel coordinate)
        //   v = fy * (Y1 / Z1) + cy   (Y pixel coordinate)
        //
        // Clamp Z to prevent division by zero or negative depths
        float z = std::max(Z1, 0.1f);  // Clamp depth to minimum 0.1
        float d = 1.0f / z;             // Inverse depth in frame j

        float u = fx_j * (d * X1) + cx_j;  // X pixel coordinate
        float v = fy_j * (d * Y1) + cy_j;  // Y pixel coordinate

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
        
        // Diagnostic: Log first edge, first pixel AND edge 4, center pixel
        bool log_first_pixel = (e == 0);
        bool log_edge4_center = (e == 4);
        int center_y = P / 2;  // Center pixel y coordinate
        int center_x = P / 2;  // Center pixel x coordinate
        
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
                
                // Diagnostic: Log Edge 4 center pixel values
                if (log_edge4_center && y == center_y && x == center_x && logger) {
                    logger->info("transformWithJacobians: Edge[4] pixel[{}/{}] - i={}, patch_idx={}, k={}, j={}, "
                                 "px_pix={:.6f}, py_pix={:.6f}, pd_pix={:.6f}, "
                                 "intr_i=[{:.6f}, {:.6f}, {:.6f}, {:.6f}], "
                                 "intr_j=[{:.6f}, {:.6f}, {:.6f}, {:.6f}]",
                                 center_y, center_x, i, patch_idx, k, j,
                                 px_pix, py_pix, pd_pix,
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
                
                // Diagnostic: Log Edge 4 center pixel inverse projection
                if (log_edge4_center && y == center_y && x == center_x && logger) {
                    logger->info("transformWithJacobians: Edge[4] inverse_proj - X0={:.6f}, Y0={:.6f}, Z0={:.6f}, W0={:.6f}",
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
                
                // Diagnostic: Log Edge 4 center pixel transform
                if (log_edge4_center && y == center_y && x == center_x && logger) {
                    Eigen::Vector3f t_i = poses[i].t;
                    Eigen::Vector3f t_j = poses[j].t;
                    logger->info("transformWithJacobians: Edge[4] transform - "
                                 "pose_i.t=({:.6f}, {:.6f}, {:.6f}), pose_j.t=({:.6f}, {:.6f}, {:.6f}), "
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

                // ============================================================
                // STEP 10: Forward projection for this pixel
                // ============================================================
                // Project 3D point [X1_pix, Y1_pix, Z1_pix] to 2D pixel coordinates
                //
                // CRITICAL: Match Python behavior - allow all coords through, let BA handle validation
                // Python's transform() function doesn't check bounds or reject points behind camera.
                // It computes the projection formula even for invalid cases, then BA filters them later.
                // We should do the same: allow all coords through here, BA will filter them.
                
                float u_pix, v_pix;
                
                // Compute depth and inverse depth
                // Use a small epsilon to prevent division by zero, but allow negative/small Z through
                float z_pix = Z1_pix;
                float d_pix = (std::abs(z_pix) > 1e-6f) ? (1.0f / z_pix) : 0.0f; // Handle near-zero Z

                // Project to pixel coordinates using pinhole camera model
                // Formula: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
                // Python computes this even for negative/small Z, producing large or invalid coords
                float u_pix_computed = fx_j * (d_pix * X1_pix) + cx_j;  // X pixel coordinate
                float v_pix_computed = fy_j * (d_pix * Y1_pix) + cy_j;  // Y pixel coordinate
                
                // Check if computed values are finite
                bool computed_is_finite = std::isfinite(u_pix_computed) && std::isfinite(v_pix_computed);
                
                // Diagnostic: Log computed values
                if (log_first_pixel && y == 0 && x == 0 && logger) {
                    logger->info("transformWithJacobians: Edge[0] projection computation - "
                                 "z_pix={:.6f}, d_pix={:.6f}, "
                                 "X1=({:.6f}, {:.6f}, {:.6f}), "
                                 "u_pix_computed={:.2f}, v_pix_computed={:.2f}, "
                                 "computed_is_finite={}",
                                 z_pix, d_pix,
                                 X1_pix, Y1_pix, Z1_pix,
                                 u_pix_computed, v_pix_computed,
                                 computed_is_finite);
                }
                
                // Diagnostic: Log Edge 4 center pixel projection computation
                if (log_edge4_center && y == center_y && x == center_x && logger) {
                    logger->info("transformWithJacobians: Edge[4] projection computation - "
                                 "z_pix={:.6f}, d_pix={:.6f}, "
                                 "X1=({:.6f}, {:.6f}, {:.6f}), "
                                 "u_pix_computed={:.6f}, v_pix_computed={:.6f}, "
                                 "computed_is_finite={}",
                                 z_pix, d_pix,
                                 X1_pix, Y1_pix, Z1_pix,
                                 u_pix_computed, v_pix_computed,
                                 computed_is_finite);
                }

                // Only reject if the computed values are not finite (NaN/Inf)
                // Allow all finite coords through (even if out of bounds or behind camera) - BA will filter them
                if (!computed_is_finite) {
                    // Invalid computed values (NaN/Inf) - mark as invalid
                    u_pix = std::numeric_limits<float>::quiet_NaN();
                    v_pix = std::numeric_limits<float>::quiet_NaN();
                    if (log_first_pixel && y == 0 && x == 0 && logger) {
                        logger->warn("transformWithJacobians: Edge[0] projection has NaN/Inf - "
                                     "u_pix_computed={:.2f}, v_pix_computed={:.2f}, "
                                     "X1=({:.6f}, {:.6f}, {:.6f}), d_pix={:.6f}. "
                                     "This suggests numerical issues.",
                                     u_pix_computed, v_pix_computed,
                                     X1_pix, Y1_pix, Z1_pix, d_pix);
                    }
                } else {
                    // Allow all finite coords through (even if out of bounds or behind camera) - BA will filter them
                    u_pix = u_pix_computed;
                    v_pix = v_pix_computed;
                    // Diagnostic: Log projection (including out-of-bounds ones)
                    if (log_first_pixel && y == 0 && x == 0 && logger) {
                        logger->info("transformWithJacobians: Edge[0] projection - "
                                     "z_pix={:.6f}, d_pix={:.6f}, u_pix={:.2f}, v_pix={:.2f}",
                                     z_pix, d_pix, u_pix, v_pix);
                    }
                }

                // Store coordinates
                coords_out[base + 0 * P * P + idx] = u_pix;
                coords_out[base + 1 * P * P + idx] = v_pix;
                
                // Diagnostic: Log Edge 4 center pixel final stored coordinates
                if (log_edge4_center && y == center_y && x == center_x && logger) {
                    logger->info("transformWithJacobians: Edge[4] final stored coords - "
                                 "u_pix={:.6f}, v_pix={:.6f}, "
                                 "stored at coords_out[{}]={:.6f}, coords_out[{}]={:.6f}",
                                 u_pix, v_pix,
                                 base + 0 * P * P + idx, coords_out[base + 0 * P * P + idx],
                                 base + 1 * P * P + idx, coords_out[base + 1 * P * P + idx]);
                }
                
                // Validity: Match Python behavior - mark as valid if coords are finite
                // Python's transform() marks all finite coords as valid, BA filters them later
                // We do the same: mark as valid if coords are finite (not NaN/Inf)
                if (valid_out) {
                    bool coords_finite = std::isfinite(u_pix) && std::isfinite(v_pix);
                    valid_out[e * P * P + idx] = coords_finite ? 1.0f : 0.0f;
                }
            }
        }

        // ====================================================================
        // STEP 11: Compute Jacobians at patch center
        // ====================================================================
        // Jacobians tell us how pixel coordinates (u, v) change when we perturb:
        //   - Pose i (Ji): How does changing frame i's pose affect reprojection?
        //   - Pose j (Jj): How does changing frame j's pose affect reprojection?
        //   - Inverse depth (Jz): How does changing depth affect reprojection?
        //
        // These are used in bundle adjustment to compute gradients for optimization
        //
        // Use transformed point at patch center for Jacobian computation
        float X = X1;  // Transformed X coordinate
        float Y = Y1;  // Transformed Y coordinate
        float Z = Z1;  // Transformed Z coordinate (depth)
        float H = H1;  // Homogeneous coordinate (inverse depth)

        // Depth check for Jacobian computation (avoid division by zero)
        float d_jac = 0.0f;
        if (std::abs(Z) > 0.2f) {
            d_jac = 1.0f / Z;  // Inverse depth for Jacobian
        }

        // ====================================================================
        // STEP 11a: Ja = Jacobian of transformed point w.r.t. SE3 parameters
        // ====================================================================
        // Ja[4, 6]: How does [X1, Y1, Z1, W1] change when we perturb SE3 parameters?
        // SE3 has 6 parameters: [tx, ty, tz, rx, ry, rz] (3 translation + 3 rotation)
        //
        // Formula from SE3 Lie algebra:
        //   d[X1, Y1, Z1, W1] / d[tx, ty, tz, rx, ry, rz] = Ja
        //
        // Structure:
        //   Row 0 (X): [H, 0, 0, 0, Z, -Y]  (derivative w.r.t. [tx, ty, tz, rx, ry, rz])
        //   Row 1 (Y): [0, H, 0, -Z, 0, X]
        //   Row 2 (Z): [0, 0, H, Y, -X, 0]
        //   Row 3 (W): [0, 0, 0, 0, 0, 0]  (W is unchanged by SE3)
        Eigen::Matrix<float, 4, 6> Ja;
        Ja.setZero();
        Ja(0, 0) = H;  Ja(0, 4) = Z;   Ja(0, 5) = -Y;  // dX/d[tx, ty, tz, rx, ry, rz]
        Ja(1, 1) = H;  Ja(1, 3) = -Z;  Ja(1, 5) = X;   // dY/d[tx, ty, tz, rx, ry, rz]
        Ja(2, 2) = H;  Ja(2, 3) = Y;   Ja(2, 4) = -X;  // dZ/d[tx, ty, tz, rx, ry, rz]
        // Row 3 (W) is all zeros (W is unchanged)

        // ====================================================================
        // STEP 11b: Jp = Jacobian of projection w.r.t. 3D point
        // ====================================================================
        // Jp[2, 4]: How do pixel coordinates (u, v) change when we perturb [X, Y, Z, W]?
        //
        // From projection formula: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
        //   du/dX = fx/Z = fx * d_jac
        //   du/dZ = -fx * X / Z^2 = -fx * X * d_jac^2
        //   dv/dY = fy/Z = fy * d_jac
        //   dv/dZ = -fy * Y / Z^2 = -fy * Y * d_jac^2
        //   du/dY = dv/dX = du/dW = dv/dW = 0
        Eigen::Matrix<float, 2, 4> Jp;
        Jp.setZero();
        Jp(0, 0) = fx_j * d_jac;                          // du/dX
        Jp(0, 2) = -fx_j * X * d_jac * d_jac;            // du/dZ
        Jp(1, 1) = fy_j * d_jac;                          // dv/dY
        Jp(1, 2) = -fy_j * Y * d_jac * d_jac;            // dv/dZ

        // ====================================================================
        // STEP 11c: Jj = Jacobian w.r.t. pose j
        // ====================================================================
        // Jj[2, 6]: How do pixel coordinates change when we perturb pose j?
        // Computed by chain rule: Jj = Jp * Ja
        //   Jj = (d[u,v]/d[X,Y,Z,W]) * (d[X,Y,Z,W]/d[pose_j_params])
        Eigen::Matrix<float, 2, 6> Jj = Jp * Ja;

        // ====================================================================
        // STEP 11d: Ji = Jacobian w.r.t. pose i
        // ====================================================================
        // Ji[2, 6]: How do pixel coordinates change when we perturb pose i?
        // Since Gij = Tj * Ti^-1, changing Ti affects Gij through its inverse
        // Python uses: Ji = -Gij.adjT(Jj)
        // Python's adjT computes: J * Ad (not J * Ad^T like C++ adjointT)
        // C++ adjointT computes: J * Ad^T
        // So Python's adjT is equivalent to: J * Ad = (J * Ad^T) * (Ad^T)^-1 * Ad
        // But simpler: Python's adjT might compute Ad^T * J^T, then transpose
        // Actually, let's try: Ji = -Gij.adjoint(Jj.transpose()).transpose()
        // But adjoint takes a 6-vector, not a matrix. Let's compute manually:
        // Python adjT: J * Ad (where Ad is adjoint matrix)
        // C++ adjointT: J * Ad^T
        // So we need: Ji = -Jj * Ad (not Jj * Ad^T)
        // We can compute this by: Ji = -(Jj * Ad^T) * (Ad^T)^-1 * Ad
        // Or simpler: compute Ad manually and do Jj * Ad
        Eigen::Matrix3f R = Gij.R();
        Eigen::Matrix3f t_hat = SE3::skew(Gij.t);
        Eigen::Matrix3f Zero = Eigen::Matrix3f::Zero();
        Eigen::Matrix<float, 6, 6> Ad;
        Ad.block<3, 3>(0, 0) = R;
        Ad.block<3, 3>(0, 3) = t_hat * R;
        Ad.block<3, 3>(3, 0) = Zero;
        Ad.block<3, 3>(3, 3) = R;
        // Python's adjT computes: J * Ad (not J * Ad^T)
        Eigen::Matrix<float, 2, 6> Ji = -Jj * Ad;

        // ====================================================================
        // STEP 11e: Jz = Jacobian w.r.t. inverse depth
        // ====================================================================
        // Jz[2, 1]: How do pixel coordinates change when we perturb inverse depth?
        // Inverse depth affects the transformation through the W coordinate:
        //   [X1, Y1, Z1] = R * [X0, Y0, Z0] + t * W0
        // So d[X1, Y1, Z1]/dW0 = t (translation part of Gij)
        //
        // Then: Jz = Jp * [tx, ty, tz, 0]^T = Jp * translation_column_of_Gij
        Eigen::Matrix4f Gij_mat = Gij.matrix();
        Eigen::Vector4f t_col = Gij_mat.col(3);  // Translation column [tx, ty, tz, 1]
        Eigen::Matrix<float, 2, 1> Jz = Jp * t_col;

        // Buffer Jacobian saves for async write (no disk I/O during main loop)
        if (do_save) {
            std::string frame_suffix = std::to_string(frame_num);
            std::string edge_suffix = std::to_string(e);
            // Ji [2x6] -> 12 floats row-major
            {
                DebugWriteTask task;
                task.type = DebugWriteTask::EIGEN_2x6;
                task.filename = get_bin_file_path("reproject_Ji_frame" + frame_suffix + "_edge" + edge_suffix + ".bin");
                task.data.resize(12);
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 6; c++)
                        task.data[r * 6 + c] = Ji(r, c);
                writeTasks.push_back(std::move(task));
            }
            // Jj [2x6] -> 12 floats row-major
            {
                DebugWriteTask task;
                task.type = DebugWriteTask::EIGEN_2x6;
                task.filename = get_bin_file_path("reproject_Jj_frame" + frame_suffix + "_edge" + edge_suffix + ".bin");
                task.data.resize(12);
                for (int r = 0; r < 2; r++)
                    for (int c = 0; c < 6; c++)
                        task.data[r * 6 + c] = Jj(r, c);
                writeTasks.push_back(std::move(task));
            }
            // Jz [2x1] -> 2 floats row-major
            {
                DebugWriteTask task;
                task.type = DebugWriteTask::EIGEN_2x1;
                task.filename = get_bin_file_path("reproject_Jz_frame" + frame_suffix + "_edge" + edge_suffix + ".bin");
                task.data = {Jz(0, 0), Jz(1, 0)};
                writeTasks.push_back(std::move(task));
            }
        }

        // ====================================================================
        // STEP 12: Store Jacobians for all patch pixels
        // ====================================================================
        // NOTE: Currently using center values for all pixels (approximation)
        // TODO: Could compute per-pixel Jacobians if needed for higher accuracy
        //
        // Storage format:
        //   Ji_out: [num_edges, 2, P, P, 6] - Jacobian w.r.t. pose i
        //   Jj_out: [num_edges, 2, P, P, 6] - Jacobian w.r.t. pose j
        //   Jz_out: [num_edges, 2, P, P, 1] - Jacobian w.r.t. inverse depth
        //
        // Dimension breakdown:
        //   num_edges = number of edges (connections between patches and frames)
        //   2 = [u, v] pixel coordinates
        //   P, P = patch size (e.g., 3x3)
        //   6 = SE3 parameters [tx, ty, tz, rx, ry, rz]
        //   1 = inverse depth parameter
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int idx = y * P + x;  // Linear index within patch
                int base = e * 2 * P * P * 6;  // Base index for this edge's Jacobians
                
                // Store Ji: Jacobian w.r.t. pose i [2, 6]
                // Format: [num_edges, 2, P, P, 6]
                for (int c = 0; c < 2; c++) {  // c=0 for u, c=1 for v
                    for (int param = 0; param < 6; param++) {  // 6 SE3 parameters
                        int ji_idx = base + c * P * P * 6 + idx * 6 + param;
                        Ji_out[ji_idx] = Ji(c, param);  // Copy from computed Jacobian
                    }
                }
                
                // Store Jj: Jacobian w.r.t. pose j [2, 6]
                // Format: [num_edges, 2, P, P, 6]
                for (int c = 0; c < 2; c++) {  // c=0 for u, c=1 for v
                    for (int param = 0; param < 6; param++) {  // 6 SE3 parameters
                        int jj_idx = base + c * P * P * 6 + idx * 6 + param;
                        Jj_out[jj_idx] = Jj(c, param);  // Copy from computed Jacobian
                    }
                }
                
                // Store Jz: Jacobian w.r.t. inverse depth [2, 1]
                // Format: [num_edges, 2, P, P, 1]
                int jz_base = e * 2 * P * P * 1;  // Base index for Jz (different size)
                for (int c = 0; c < 2; c++) {  // c=0 for u, c=1 for v
                    int jz_idx = jz_base + c * P * P * 1 + idx * 1;
                    Jz_out[jz_idx] = Jz(c, 0);  // Copy from computed Jacobian (only 1 parameter)
                }
            }
        }
    }
    
    // Buffer reprojected coordinates save, then launch all writes in background
    if (do_save) {
        std::string frame_suffix = std::to_string(frame_num);
        size_t total_size = static_cast<size_t>(num_edges) * 2 * P * P;
        DebugWriteTask task;
        task.type = DebugWriteTask::FLOAT_ARRAY;
        task.filename = get_bin_file_path("reproject_coords_frame" + frame_suffix + ".bin");
        task.data.assign(coords_out, coords_out + total_size);
        task.count = total_size;
        writeTasks.push_back(std::move(task));

        // Write all buffered debug files synchronously so logs appear in the correct frame
        auto logger = spdlog::get("dpvo");
        if (logger) {
            logger->info("Writing {} debug files for frame {} (synchronous)",
                        writeTasks.size(), frame_num);
        }
        executeWriteTasks(std::move(writeTasks));
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
