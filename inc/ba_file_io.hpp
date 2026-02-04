#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "eigen_common.h"  // Provides Eigen types (Vector3f, Quaternionf, etc.)
#include "se.hpp"  // Changed from se3.h to se.hpp to match codebase
#include "logger.hpp"

/**
 * Utility functions for saving BA input/output parameters to binary files
 * Used for comparison with Python BA implementation
 */
namespace ba_file_io {

/**
 * Save a float array to a binary file
 * @param filename Output filename
 * @param data Pointer to data array
 * @param num_elements Number of elements to write
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_float_array(const std::string& filename, const float* data, size_t num_elements, 
                            std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), sizeof(float) * num_elements);
    file.close();
    if (logger) logger->info("Saved {} floats to {}", num_elements, filename);
    return true;
}

/**
 * Save an int32 array to a binary file
 * @param filename Output filename
 * @param data Pointer to data array
 * @param num_elements Number of elements to write
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_int32_array(const std::string& filename, const int32_t* data, size_t num_elements,
                            std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    file.write(reinterpret_cast<const char*>(data), sizeof(int32_t) * num_elements);
    file.close();
    if (logger) logger->info("Saved {} int32s to {}", num_elements, filename);
    return true;
}

/**
 * Save poses to binary file [N, 7] format: [tx, ty, tz, qx, qy, qz, qw]
 * @param filename Output filename
 * @param poses Array of SE3 poses
 * @param N Number of poses
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_poses(const std::string& filename, const SE3* poses, int N,
                      std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int i = 0; i < N; i++) {
        Eigen::Vector3f t = poses[i].t;
        Eigen::Quaternionf q = poses[i].q;
        float pose_data[7] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
        file.write(reinterpret_cast<const char*>(pose_data), sizeof(float) * 7);
    }
    
    file.close();
    if (logger) logger->info("Saved {} poses to {}", N, filename);
    return true;
}

/**
 * Save patches to binary file [N*M, 3, P, P]
 * @param filename Output filename
 * @param patches 5D array: patches[frame][patch][channel][y][x]
 * @param N Number of frames
 * @param M Number of patches per frame
 * @param P Patch size (PxP)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
template<typename PatchArray>
inline bool save_patches(const std::string& filename, const PatchArray& patches, int N, int M, int P,
                        std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int c = 0; c < 3; c++) {
                for (int y = 0; y < P; y++) {
                    for (int x = 0; x < P; x++) {
                        float val = patches[i][j][c][y][x];
                        file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                    }
                }
            }
        }
    }
    
    file.close();
    if (logger) logger->info("Saved patches [{}*{}, 3, {}, {}] to {}", N, M, P, P, filename);
    return true;
}

/**
 * Save intrinsics to binary file [N, 4] format: [fx, fy, cx, cy]
 * @param filename Output filename
 * @param intrinsics Array of intrinsics arrays (each is 4 floats)
 * @param N Number of frames
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
template<typename IntrinsicsArray>
inline bool save_intrinsics(const std::string& filename, const IntrinsicsArray& intrinsics, int N,
                           std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int i = 0; i < N; i++) {
        file.write(reinterpret_cast<const char*>(intrinsics[i]), sizeof(float) * 4);
    }
    
    file.close();
    if (logger) logger->info("Saved {} intrinsics to {}", N, filename);
    return true;
}

/**
 * Save edge indices to binary files [num_active] (int32)
 * @param ii_filename Output filename for source frame indices
 * @param jj_filename Output filename for target frame indices
 * @param kk_filename Output filename for patch indices
 * @param kk_array Array of global patch indices (kk[e])
 * @param jj_array Array of target frame indices (jj[e])
 * @param num_active Number of active edges
 * @param M Number of patches per frame (used to extract source frame from kk)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
template<typename KKArray, typename JJArray>
inline bool save_edge_indices(const std::string& ii_filename, const std::string& jj_filename, 
                              const std::string& kk_filename,
                              const KKArray& kk_array, const JJArray& jj_array,
                              int num_active, int M,
                              std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream ii_file(ii_filename, std::ios::binary);
    std::ofstream jj_file(jj_filename, std::ios::binary);
    std::ofstream kk_file(kk_filename, std::ios::binary);
    
    if (!ii_file.is_open() || !jj_file.is_open() || !kk_file.is_open()) {
        if (logger) logger->error("Failed to open index files for writing");
        return false;
    }
    
    for (int e = 0; e < num_active; e++) {
        // Extract source frame index from kk (matching BA logic)
        int i_source = kk_array[e] / M;  // Source frame index
        int32_t ii_val = static_cast<int32_t>(i_source);  // Frame index, not patch index!
        int32_t jj_val = static_cast<int32_t>(jj_array[e]);  // Target frame index
        int32_t kk_val = static_cast<int32_t>(kk_array[e]);  // Global patch index
        
        ii_file.write(reinterpret_cast<const char*>(&ii_val), sizeof(int32_t));
        jj_file.write(reinterpret_cast<const char*>(&jj_val), sizeof(int32_t));
        kk_file.write(reinterpret_cast<const char*>(&kk_val), sizeof(int32_t));
    }
    
    ii_file.close();
    jj_file.close();
    kk_file.close();
    
    if (logger) {
        logger->info("Saved {} indices to {}, {}, {} (ii=frame_index from kk/M, jj=frame_index, kk=patch_index)", 
                    num_active, ii_filename, jj_filename, kk_filename);
    }
    return true;
}

/**
 * Save reprojected coordinates at patch center [num_active, 2]
 * @param filename Output filename
 * @param coords Flat array of coordinates [num_active, 2, P, P]
 * @param num_active Number of active edges
 * @param P Patch size
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_reprojected_coords_center(const std::string& filename, const float* coords,
                                          int num_active, int P,
                                          std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    int center_idx = (P / 2) * P + (P / 2);
    for (int e = 0; e < num_active; e++) {
        float cx = coords[e * 2 * P * P + 0 * P * P + center_idx];
        float cy = coords[e * 2 * P * P + 1 * P * P + center_idx];
        file.write(reinterpret_cast<const char*>(&cx), sizeof(float));
        file.write(reinterpret_cast<const char*>(&cy), sizeof(float));
    }
    
    file.close();
    if (logger) logger->info("Saved {} reprojected coordinates to {}", num_active, filename);
    return true;
}

/**
 * Save full reprojected coordinates [num_active, 2, P, P]
 * @param filename Output filename
 * @param coords Flat array of coordinates [num_active, 2, P, P]
 * @param num_active Number of active edges
 * @param P Patch size
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_reprojected_coords_full(const std::string& filename, const float* coords,
                                         int num_active, int P,
                                         std::shared_ptr<spdlog::logger> logger = nullptr) {
    size_t num_elements = static_cast<size_t>(num_active) * 2 * P * P;
    return save_float_array(filename, coords, num_elements, logger);
}

/**
 * Save targets [num_active, 2] (reprojected_coords + delta from update model)
 * @param filename Output filename
 * @param targets Flat array [num_active * 2] - accessed as targets[e * 2 + 0] and targets[e * 2 + 1]
 * @param num_active Number of active edges
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_targets(const std::string& filename, const float* targets, int num_active,
                        std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int e = 0; e < num_active; e++) {
        float target_x = targets[e * 2 + 0];
        float target_y = targets[e * 2 + 1];
        file.write(reinterpret_cast<const char*>(&target_x), sizeof(float));
        file.write(reinterpret_cast<const char*>(&target_y), sizeof(float));
    }
    
    file.close();
    if (logger) logger->info("Saved {} targets to {}", num_active, filename);
    return true;
}

/**
 * Save weights [num_active, 2] (from update model)
 * @param filename Output filename
 * @param weights 2D array weights[edge][dim]
 * @param num_active Number of active edges
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
template<typename WeightArray>
inline bool save_weights(const std::string& filename, const WeightArray& weights, int num_active,
                        std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int e = 0; e < num_active; e++) {
        float w0 = weights[e][0];
        float w1 = weights[e][1];
        file.write(reinterpret_cast<const char*>(&w0), sizeof(float));
        file.write(reinterpret_cast<const char*>(&w1), sizeof(float));
    }
    
    file.close();
    if (logger) logger->info("Saved {} weights to {}", num_active, filename);
    return true;
}

/**
 * Save metadata to text file
 * @param filename Output filename
 * @param num_active Number of active edges
 * @param MAX_EDGE Maximum number of edges
 * @param DIM Dimension
 * @param CORR_DIM Correlation dimension
 * @param M Number of patches per frame
 * @param P Patch size
 * @param N Number of frames
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_metadata(const std::string& filename, int num_active, int MAX_EDGE, int DIM, 
                         int CORR_DIM, int M, int P, int N,
                         std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    file << "num_active=" << num_active << "\n";
    file << "MAX_EDGE=" << MAX_EDGE << "\n";
    file << "DIM=" << DIM << "\n";
    file << "CORR_DIM=" << CORR_DIM << "\n";
    file << "M=" << M << "\n";
    file << "P=" << P << "\n";
    file << "N=" << N << "\n";
    
    file.close();
    if (logger) {
        logger->info("Saved metadata to {}: num_active={}, MAX_EDGE={}, DIM={}, CORR_DIM={}, M={}, P={}, N={}", 
                    filename, num_active, MAX_EDGE, DIM, CORR_DIM, M, P, N);
    }
    return true;
}

/**
 * Save SE3 transforms for reproject intermediate comparison [num_active, 7]
 * Format: [tx, ty, tz, qx, qy, qz, qw] per edge
 * @param filename Output filename
 * @param transforms Array of SE3 transforms
 * @param num_active Number of active edges
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_se3_transforms(const std::string& filename, const SE3* transforms, int num_active,
                                std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int e = 0; e < num_active; e++) {
        Eigen::Vector3f t = transforms[e].t;
        Eigen::Quaternionf q = transforms[e].q;
        float transform_data[7] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
        file.write(reinterpret_cast<const char*>(transform_data), sizeof(float) * 7);
    }
    
    file.close();
    if (logger) logger->info("Saved {} SE3 transforms to {}", num_active, filename);
    return true;
}

/**
 * Save Jacobians for reproject intermediate comparison
 * @param filename Output filename
 * @param jacobians Flat array [num_active, 2, 6] for Ji/Jj or [num_active, 2, 1] for Jz
 * @param num_active Number of active edges
 * @param jacobian_type "Ji", "Jj", or "Jz"
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_jacobians(const std::string& filename, const float* jacobians, int num_active,
                          const std::string& jacobian_type,
                          std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    if (jacobian_type == "Ji" || jacobian_type == "Jj") {
        // Ji/Jj: [num_active, 2, 6] - Jacobian w.r.t. pose (2 outputs, 6 SE3 params)
        for (int e = 0; e < num_active; e++) {
            for (int c = 0; c < 2; c++) {  // c=0 for u, c=1 for v
                for (int param = 0; param < 6; param++) {  // 6 SE3 parameters
                    int idx = e * 2 * 6 + c * 6 + param;
                    file.write(reinterpret_cast<const char*>(&jacobians[idx]), sizeof(float));
                }
            }
        }
        if (logger) logger->info("Saved {} {} Jacobians [num_active, 2, 6] to {}", num_active, jacobian_type, filename);
    } else if (jacobian_type == "Jz") {
        // Jz: [num_active, 2, 1] - Jacobian w.r.t. inverse depth (2 outputs, 1 param)
        for (int e = 0; e < num_active; e++) {
            for (int c = 0; c < 2; c++) {  // c=0 for u, c=1 for v
                int idx = e * 2 * 1 + c * 1;
                file.write(reinterpret_cast<const char*>(&jacobians[idx]), sizeof(float));
            }
        }
        if (logger) logger->info("Saved {} {} Jacobians [num_active, 2, 1] to {}", num_active, jacobian_type, filename);
    } else {
        if (logger) logger->error("Unknown jacobian_type: {} (expected 'Ji', 'Jj', or 'Jz')", jacobian_type);
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

/**
 * Save a single SE3 object to binary file [7] format: [tx, ty, tz, qx, qy, qz, qw]
 * @param filename Output filename
 * @param se3_obj SE3 object to save
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_se3_object(const std::string& filename, const SE3& se3_obj,
                           std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    Eigen::Vector3f t = se3_obj.t;
    Eigen::Quaternionf q = se3_obj.q;
    float data[7] = {t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w()};
    file.write(reinterpret_cast<const char*>(data), sizeof(float) * 7);
    file.close();
    if (logger) logger->info("Saved SE3 object to {}", filename);
    return true;
}

/**
 * Save an Eigen matrix to binary file (row-major format to match Python)
 * @param filename Output filename
 * @param matrix Eigen matrix to save
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
template<typename MatrixType>
inline bool save_eigen_matrix(const std::string& filename, const MatrixType& matrix,
                             std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    // Eigen matrices are column-major by default. Write row by row to match Python's expected row-major format.
    for (int r = 0; r < matrix.rows(); ++r) {
        for (int c = 0; c < matrix.cols(); ++c) {
            float val = matrix(r, c);
            file.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    file.close();
    if (logger) logger->info("Saved Eigen matrix ({}x{}) to {}", matrix.rows(), matrix.cols(), filename);
    return true;
}

/**
 * Save timestamps to binary file [N] (int64_t)
 * @param filename Output filename
 * @param tstamps Array of timestamps
 * @param N Number of frames
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_timestamps(const std::string& filename, const int64_t* tstamps, int N,
                           std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    file.write(reinterpret_cast<const char*>(tstamps), sizeof(int64_t) * N);
    file.close();
    if (logger) logger->info("Saved {} timestamps to {}", N, filename);
    return true;
}

/**
 * Save colors to binary file [N, M, 3] (uint8_t)
 * @param filename Output filename
 * @param colors 3D array colors[frame][patch][channel]
 * @param N Number of frames
 * @param M Number of patches per frame
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
template<typename ColorArray>
inline bool save_colors(const std::string& filename, const ColorArray& colors, int N, int M,
                       std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            for (int c = 0; c < 3; c++) {
                uint8_t val = colors[i][j][c];
                file.write(reinterpret_cast<const char*>(&val), sizeof(uint8_t));
            }
        }
    }
    
    file.close();
    if (logger) logger->info("Saved colors [{}*{}, 3] to {}", N, M, filename);
    return true;
}

/**
 * Save m_index array to binary file [N, M] (int32)
 * @param filename Output filename
 * @param index 2D array index[frame][patch]
 * @param N Number of frames
 * @param M Number of patches per frame
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
template<typename IndexArray>
inline bool save_index(const std::string& filename, const IndexArray& index, int N, int M,
                      std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int32_t val = static_cast<int32_t>(index[i][j]);
            file.write(reinterpret_cast<const char*>(&val), sizeof(int32_t));
        }
    }
    
    file.close();
    if (logger) logger->info("Saved index [{}*{}] to {}", N, M, filename);
    return true;
}

/**
 * Save m_ix array to binary file [N*M] (int32)
 * @param filename Output filename
 * @param ix Array of frame indices for patches
 * @param size Number of elements (N*M)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_ix(const std::string& filename, const int* ix, int size,
                   std::shared_ptr<spdlog::logger> logger = nullptr) {
    return save_int32_array(filename, reinterpret_cast<const int32_t*>(ix), size, logger);
}

/**
 * Save keyframe metadata to text file
 * @param filename Output filename
 * @param n_before Number of frames before keyframe
 * @param m_before Number of patches before keyframe
 * @param num_edges_before Number of edges before keyframe
 * @param n_after Number of frames after keyframe
 * @param m_after Number of patches after keyframe
 * @param num_edges_after Number of edges after keyframe
 * @param i Motion check frame i
 * @param j Motion check frame j
 * @param k Frame removed (k)
 * @param motion Motion magnitude
 * @param should_remove Whether frame was removed
 * @param KEYFRAME_INDEX Config value
 * @param KEYFRAME_THRESH Config value
 * @param PATCH_LIFETIME Config value
 * @param REMOVAL_WINDOW Config value
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_keyframe_metadata(const std::string& filename,
                                   int n_before, int m_before, int num_edges_before,
                                   int n_after, int m_after, int num_edges_after,
                                   int i, int j, int k, float motion, bool should_remove,
                                   int KEYFRAME_INDEX, int KEYFRAME_THRESH, int PATCH_LIFETIME, int REMOVAL_WINDOW,
                                   std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        if (logger) logger->error("Failed to open {} for writing", filename);
        return false;
    }
    
    file << "n_before=" << n_before << "\n";
    file << "m_before=" << m_before << "\n";
    file << "num_edges_before=" << num_edges_before << "\n";
    file << "n_after=" << n_after << "\n";
    file << "m_after=" << m_after << "\n";
    file << "num_edges_after=" << num_edges_after << "\n";
    file << "i=" << i << "\n";
    file << "j=" << j << "\n";
    file << "k=" << k << "\n";
    file << "motion=" << motion << "\n";
    file << "should_remove=" << (should_remove ? 1 : 0) << "\n";
    file << "KEYFRAME_INDEX=" << KEYFRAME_INDEX << "\n";
    file << "KEYFRAME_THRESH=" << KEYFRAME_THRESH << "\n";
    file << "PATCH_LIFETIME=" << PATCH_LIFETIME << "\n";
    file << "REMOVAL_WINDOW=" << REMOVAL_WINDOW << "\n";
    
    file.close();
    if (logger) {
        logger->info("Saved keyframe metadata to {}", filename);
    }
    return true;
}

} // namespace ba_file_io

