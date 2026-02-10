#include "ba.hpp"
#include "dpvo.hpp"
#include "projective_ops.hpp"
#include "eigen_common.h"
#include "Eigen/Dense"
#include "Eigen/Cholesky"
#include "target_frame.hpp"  // Shared TARGET_FRAME constant
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sys/stat.h>  // For mkdir
#include <sys/types.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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

// =================================================================================================
// BA Intermediate Value Saving Helper Functions
// =================================================================================================

// Helper function to get logger (for saving functions)
static std::shared_ptr<spdlog::logger> get_ba_logger() {
    auto logger = spdlog::get("dpvo");
    if (!logger) {
        #ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
        #else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
        #endif
    }
    return logger;
}

// STEP 1: Save residuals, validity mask, and coords at center
static void save_ba_step1(const float* r, const float* v, const float* coords, 
                          int num_active, int P, int center_idx, 
                          std::shared_ptr<spdlog::logger> logger) {
    std::ofstream r_file(get_bin_file_path("ba_step1_residuals.bin"), std::ios::binary);
    std::ofstream v_file(get_bin_file_path("ba_step1_validity.bin"), std::ios::binary);
    
    // Extract coords at center that BA actually uses for residual computation
    std::vector<float> coords_center_ba(num_active * 2);
    for (int e = 0; e < num_active; e++) {
        coords_center_ba[e * 2 + 0] = coords[e * 2 * P * P + 0 * P * P + center_idx];
        coords_center_ba[e * 2 + 1] = coords[e * 2 * P * P + 1 * P * P + center_idx];
    }
    std::ofstream coords_center_file(get_bin_file_path("ba_step1_coords_center.bin"), std::ios::binary);
    
    if (r_file.is_open() && v_file.is_open() && coords_center_file.is_open()) {
        if (logger) {
            logger->info("[BA] Saving STEP 1 residuals (first 3 edges):");
            for (int e = 0; e < std::min(3, num_active); e++) {
                logger->info("[BA]   Edge[{}]: coords=({:.6f}, {:.6f}), residual=({:.6f}, {:.6f}), validity={:.1f}", 
                            e, coords_center_ba[e * 2 + 0], coords_center_ba[e * 2 + 1],
                            r[e * 2 + 0], r[e * 2 + 1], v[e]);
            }
        }
        r_file.write(reinterpret_cast<const char*>(r), sizeof(float) * num_active * 2);
        v_file.write(reinterpret_cast<const char*>(v), sizeof(float) * num_active);
        coords_center_file.write(reinterpret_cast<const char*>(coords_center_ba.data()), sizeof(float) * num_active * 2);
        r_file.close();
        v_file.close();
        coords_center_file.close();
        if (logger) {
            logger->info("[BA] ✅ Successfully saved STEP 1 - residuals, validity mask, and coords at center");
            logger->info("[BA] File sizes - residuals: {} bytes, validity: {} bytes, coords_center: {} bytes",
                         num_active * 2 * sizeof(float), num_active * sizeof(float), num_active * 2 * sizeof(float));
        }
    } else {
        if (logger) {
            logger->error("[BA] ❌ Failed to open files for STEP 1 saving!");
            logger->error("[BA] r_file.is_open()={}, v_file.is_open()={}, coords_center_file.is_open()={}",
                         r_file.is_open(), v_file.is_open(), coords_center_file.is_open());
        }
    }
}

// STEP 2: Save weighted Jacobians
static void save_ba_step2(const std::vector<Eigen::Matrix<float, 6, 2>>& wJiT,
                          const std::vector<Eigen::Matrix<float, 6, 2>>& wJjT,
                          const std::vector<Eigen::Matrix<float, 1, 2>>& wJzT,
                          const float* weights_masked,
                          const float* Ji_center,
                          const float* Jj_center,
                          int num_active,
                          std::shared_ptr<spdlog::logger> logger) {
    std::ofstream wJiT_file(get_bin_file_path("ba_step2_wJiT.bin"), std::ios::binary);
    std::ofstream wJjT_file(get_bin_file_path("ba_step2_wJjT.bin"), std::ios::binary);
    std::ofstream wJzT_file(get_bin_file_path("ba_step2_wJzT.bin"), std::ios::binary);
    std::ofstream weights_masked_file(get_bin_file_path("ba_step2_weights_masked.bin"), std::ios::binary);
    std::ofstream Ji_center_file(get_bin_file_path("ba_step2_Ji_center.bin"), std::ios::binary);
    std::ofstream Jj_center_file(get_bin_file_path("ba_step2_Jj_center.bin"), std::ios::binary);
    
    if (wJiT_file.is_open() && wJjT_file.is_open() && wJzT_file.is_open() && 
        weights_masked_file.is_open() && Ji_center_file.is_open() && Jj_center_file.is_open()) {
        // Eigen matrices are column-major, but Python expects row-major
        // Write row by row to match Python's expected format
        for (int e = 0; e < num_active; e++) {
            // Write wJiT[e] as [6, 2] in row-major order
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 2; j++) {
                    float val = wJiT[e](i, j);
                    wJiT_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }
            // Write wJjT[e] as [6, 2] in row-major order
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 2; j++) {
                    float val = wJjT[e](i, j);
                    wJjT_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }
            // Write wJzT[e] as [1, 2] in row-major order
            for (int i = 0; i < 1; i++) {
                for (int j = 0; j < 2; j++) {
                    float val = wJzT[e](i, j);
                    wJzT_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }
        }
        weights_masked_file.write(reinterpret_cast<const char*>(weights_masked), sizeof(float) * num_active * 2);
        // Save Ji_center and Jj_center for debugging (row-major: [num_active, 2, 6])
        for (int e = 0; e < num_active; e++) {
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 6; j++) {
                    float val_ji = Ji_center[e * 2 * 6 + i * 6 + j];
                    float val_jj = Jj_center[e * 2 * 6 + i * 6 + j];
                    Ji_center_file.write(reinterpret_cast<const char*>(&val_ji), sizeof(float));
                    Jj_center_file.write(reinterpret_cast<const char*>(&val_jj), sizeof(float));
                }
            }
        }
        wJiT_file.close();
        wJjT_file.close();
        wJzT_file.close();
        weights_masked_file.close();
        Ji_center_file.close();
        Jj_center_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 2 - weighted Jacobians");
    } else {
        if (logger) {
            logger->error("[BA] ❌ Failed to open files for STEP 2 saving!");
        }
    }
}

// STEP 3: Save Hessian blocks
static void save_ba_step3(const std::vector<Eigen::Matrix<float, 6, 6>>& Bii,
                          const std::vector<Eigen::Matrix<float, 6, 6>>& Bij,
                          const std::vector<Eigen::Matrix<float, 6, 6>>& Bji,
                          const std::vector<Eigen::Matrix<float, 6, 6>>& Bjj,
                          const std::vector<Eigen::Matrix<float, 6, 1>>& Eik,
                          const std::vector<Eigen::Matrix<float, 6, 1>>& Ejk,
                          int num_active,
                          std::shared_ptr<spdlog::logger> logger) {
    std::ofstream Bii_file(get_bin_file_path("ba_step3_Bii.bin"), std::ios::binary);
    std::ofstream Bij_file(get_bin_file_path("ba_step3_Bij.bin"), std::ios::binary);
    std::ofstream Bji_file(get_bin_file_path("ba_step3_Bji.bin"), std::ios::binary);
    std::ofstream Bjj_file(get_bin_file_path("ba_step3_Bjj.bin"), std::ios::binary);
    std::ofstream Eik_file(get_bin_file_path("ba_step3_Eik.bin"), std::ios::binary);
    std::ofstream Ejk_file(get_bin_file_path("ba_step3_Ejk.bin"), std::ios::binary);
    
    if (Bii_file.is_open() && Bij_file.is_open() && Bji_file.is_open() && 
        Bjj_file.is_open() && Eik_file.is_open() && Ejk_file.is_open()) {
        // Eigen matrices are column-major, but Python expects row-major
        // Write row by row to match Python's expected format
        for (int e = 0; e < num_active; e++) {
            // Write Bii[e] as [6, 6] in row-major order
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    float val = Bii[e](i, j);
                    Bii_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }
            // Write Bij[e] as [6, 6] in row-major order
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    float val = Bij[e](i, j);
                    Bij_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }
            // Write Bji[e] as [6, 6] in row-major order
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    float val = Bji[e](i, j);
                    Bji_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }
            // Write Bjj[e] as [6, 6] in row-major order
            for (int i = 0; i < 6; i++) {
                for (int j = 0; j < 6; j++) {
                    float val = Bjj[e](i, j);
                    Bjj_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
                }
            }
            // Write Eik[e] as [6, 1] in row-major order (column vector, so just write column)
            for (int i = 0; i < 6; i++) {
                float val = Eik[e](i, 0);
                Eik_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
            // Write Ejk[e] as [6, 1] in row-major order (column vector, so just write column)
            for (int i = 0; i < 6; i++) {
                float val = Ejk[e](i, 0);
                Ejk_file.write(reinterpret_cast<const char*>(&val), sizeof(float));
            }
        }
        Bii_file.close();
        Bij_file.close();
        Bji_file.close();
        Bjj_file.close();
        Eik_file.close();
        Ejk_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 3 - Hessian blocks");
    } else {
        if (logger) {
            logger->error("[BA] ❌ Failed to open files for STEP 3 saving!");
        }
    }
}

// STEP 4: Save gradients
static void save_ba_step4(const std::vector<Eigen::Matrix<float, 6, 1>>& vi,
                          const std::vector<Eigen::Matrix<float, 6, 1>>& vj,
                          const float* w_vec,
                          int num_active,
                          std::shared_ptr<spdlog::logger> logger) {
    std::ofstream vi_file(get_bin_file_path("ba_step4_vi.bin"), std::ios::binary);
    std::ofstream vj_file(get_bin_file_path("ba_step4_vj.bin"), std::ios::binary);
    std::ofstream w_vec_file(get_bin_file_path("ba_step4_w_vec.bin"), std::ios::binary);
    
    if (vi_file.is_open() && vj_file.is_open() && w_vec_file.is_open()) {
        for (int e = 0; e < num_active; e++) {
            vi_file.write(reinterpret_cast<const char*>(vi[e].data()), sizeof(float) * 6);
            vj_file.write(reinterpret_cast<const char*>(vj[e].data()), sizeof(float) * 6);
        }
        w_vec_file.write(reinterpret_cast<const char*>(w_vec), sizeof(float) * num_active);
        vi_file.close();
        vj_file.close();
        w_vec_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 4 - gradients");
    } else {
        if (logger) {
            logger->error("[BA] ❌ Failed to open files for STEP 4 saving!");
        }
    }
}

// STEP 9: Save assembled Hessian B
static void save_ba_step9(const Eigen::MatrixXf& B, int n_adjusted, std::shared_ptr<spdlog::logger> logger) {
    std::ofstream B_file(get_bin_file_path("ba_step9_B.bin"), std::ios::binary);
    if (B_file.is_open()) {
        B_file.write(reinterpret_cast<const char*>(B.data()), sizeof(float) * 6 * n_adjusted * 6 * n_adjusted);
        B_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 9 - assembled Hessian B (size: {}x{})", 
                                 6 * n_adjusted, 6 * n_adjusted);
    } else {
        if (logger) logger->error("[BA] ❌ Failed to open file for STEP 9 saving!");
    }
}

// STEP 10: Save pose-structure coupling E
static void save_ba_step10(const Eigen::MatrixXf& E, int n_adjusted, int m, std::shared_ptr<spdlog::logger> logger) {
    std::ofstream E_file(get_bin_file_path("ba_step10_E.bin"), std::ios::binary);
    if (E_file.is_open()) {
        E_file.write(reinterpret_cast<const char*>(E.data()), sizeof(float) * 6 * n_adjusted * m);
        E_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 10 - pose-structure coupling E (size: {}x{})", 
                                 6 * n_adjusted, m);
    } else {
        if (logger) logger->error("[BA] ❌ Failed to open file for STEP 10 saving!");
    }
}

// STEP 11: Save structure Hessian C
static void save_ba_step11_C(const Eigen::VectorXf& C, int m, std::shared_ptr<spdlog::logger> logger) {
    std::ofstream C_file(get_bin_file_path("ba_step11_C.bin"), std::ios::binary);
    if (C_file.is_open()) {
        C_file.write(reinterpret_cast<const char*>(C.data()), sizeof(float) * m);
        C_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 11 - structure Hessian C (size: {})", m);
    } else {
        if (logger) logger->error("[BA] ❌ Failed to open file for STEP 11 C saving!");
    }
}

// STEP 11: Save assembled gradients
static void save_ba_step11_gradients(const Eigen::VectorXf& v_grad, const Eigen::VectorXf& w_grad, 
                                      int n_adjusted, int m, std::shared_ptr<spdlog::logger> logger) {
    std::ofstream v_grad_file(get_bin_file_path("ba_step11_v_grad.bin"), std::ios::binary);
    std::ofstream w_grad_file(get_bin_file_path("ba_step11_w_grad.bin"), std::ios::binary);
    if (v_grad_file.is_open() && w_grad_file.is_open()) {
        v_grad_file.write(reinterpret_cast<const char*>(v_grad.data()), sizeof(float) * 6 * n_adjusted);
        w_grad_file.write(reinterpret_cast<const char*>(w_grad.data()), sizeof(float) * m);
        v_grad_file.close();
        w_grad_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 11 - assembled gradients v_grad (size: {}) and w_grad (size: {})", 
                                 6 * n_adjusted, m);
    } else {
        if (logger) logger->error("[BA] ❌ Failed to open files for STEP 11 gradients saving!");
    }
}

// STEP 13: Save Q (inverse structure Hessian)
static void save_ba_step13(const Eigen::VectorXf& Q, int m, std::shared_ptr<spdlog::logger> logger) {
    std::ofstream Q_file(get_bin_file_path("ba_step13_Q.bin"), std::ios::binary);
    if (Q_file.is_open()) {
        Q_file.write(reinterpret_cast<const char*>(Q.data()), sizeof(float) * m);
        Q_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 13 - Q (inverse structure Hessian, size: {})", m);
    } else {
        if (logger) logger->error("[BA] ❌ Failed to open file for STEP 13 saving!");
    }
}

// STEP 14: Save Schur complement S and RHS y
static void save_ba_step14(const Eigen::MatrixXf& S, const Eigen::VectorXf& y, 
                           int n_adjusted, std::shared_ptr<spdlog::logger> logger) {
    std::ofstream S_file(get_bin_file_path("ba_step14_S.bin"), std::ios::binary);
    std::ofstream y_file(get_bin_file_path("ba_step14_y.bin"), std::ios::binary);
    if (S_file.is_open() && y_file.is_open()) {
        S_file.write(reinterpret_cast<const char*>(S.data()), sizeof(float) * 6 * n_adjusted * 6 * n_adjusted);
        y_file.write(reinterpret_cast<const char*>(y.data()), sizeof(float) * 6 * n_adjusted);
        S_file.close();
        y_file.close();
        if (logger) logger->info("[BA] ✅ Successfully saved STEP 14 - Schur complement S (size: {}x{}) and RHS y (size: {})", 
                                  6 * n_adjusted, 6 * n_adjusted, 6 * n_adjusted);
    } else {
        if (logger) logger->error("[BA] ❌ Failed to open files for STEP 14 saving!");
    }
}

// STEP 15-16: Save solution dX and dZ
static void save_ba_step15_16(const Eigen::VectorXf& dX, const Eigen::VectorXf& dZ, 
                               std::shared_ptr<spdlog::logger> logger) {
    std::ofstream dX_file(get_bin_file_path("ba_step15_dX.bin"), std::ios::binary);
    std::ofstream dZ_file(get_bin_file_path("ba_step16_dZ.bin"), std::ios::binary);
    if (dX_file.is_open() && dZ_file.is_open()) {
        dX_file.write(reinterpret_cast<const char*>(dX.data()), sizeof(float) * dX.size());
        dZ_file.write(reinterpret_cast<const char*>(dZ.data()), sizeof(float) * dZ.size());
        dX_file.close();
        dZ_file.close();
        if (logger) {
            logger->info("[BA] ✅ Successfully saved STEP 15-16 - solution dX (size: {}) and dZ (size: {})", 
                         dX.size(), dZ.size());
            logger->info("[BA] File sizes - dX: {} bytes, dZ: {} bytes", 
                         dX.size() * sizeof(float), dZ.size() * sizeof(float));
        }
    } else {
        if (logger) {
            logger->error("[BA] ❌ Failed to open files for STEP 15-16 saving!");
            logger->error("[BA] dX_file.is_open()={}, dZ_file.is_open()={}", 
                         dX_file.is_open(), dZ_file.is_open());
        }
    }
}

// =================================================================================================
// Bundle Adjustment Implementation
// Translated from Python BA function, adapted for C++ without PyTorch
// =================================================================================================
void DPVO::bundleAdjustment(float lmbda, float ep, bool structure_only, int fixedp)
{
    auto logger = get_ba_logger();
    const int num_active = m_pg.m_num_edges;
    
    if (num_active == 0) {
        if (logger) {
            logger->info("[BA][F{}] SKIP: num_active=0, no edges to process", m_counter);
        }
        return;
    }

    if (logger) {
        logger->info("[BA] \033[32m========================================\033[0m");
        logger->info("[BA] \033[32mSTEP 3.1: bundleAdjustment() - Bundle Adjustment\033[0m");
        logger->info("[BA] \033[32m========================================\033[0m");
    }
    // Save intermediate BA values for step-by-step comparison at a specific frame
    // TARGET_FRAME is now defined in target_frame.hpp (shared across all files)
    // bundleAdjustment() is a member function, so we can access m_counter directly
    bool save_intermediates = (m_counter == TARGET_FRAME);
    // Enable concise summary logging for all frames
    bool log_post1000 = true;
    
    if (logger) {
        logger->info("[BA] m_counter={}, TARGET_FRAME={}, save_intermediates={}", 
                     m_counter, TARGET_FRAME, save_intermediates);
    }

    const int M = m_cfg.PATCHES_PER_FRAME;
    const int P = m_P;
    const int b = 1; // batch size = 1

    // ---------------------------------------------------------
    // CRITICAL: Filter edges that reference frames outside sliding window
    // ---------------------------------------------------------
    // Edges may reference very old frames (e.g., frame 980) that are no longer in the
    // sliding window (which only has 15-16 frames). These edges cause BA to fail because
    // the poses for those frames don't exist. Filter them out before BA.
    std::vector<bool> edge_valid(num_active, true);
    int filtered_count = 0;
    for (int e = 0; e < num_active; e++) {
        int i = m_pg.m_kk[e] / M;  // source frame index (sliding window index)
        int j = m_pg.m_jj[e];      // target frame index (sliding window index)
        
        // Filter out edges that reference frames outside the sliding window
        if (i >= m_pg.m_n || j >= m_pg.m_n || i < 0 || j < 0) {
            edge_valid[e] = false;
            filtered_count++;
        }
    }
    
    if (filtered_count > 0 && logger) {
        logger->warn("[BA] Filtered out {} edges that reference frames outside sliding window (n={})",
                     filtered_count, m_pg.m_n);
    }
    
    // If all edges were filtered, skip BA
    int valid_edge_count = num_active - filtered_count;
    if (valid_edge_count == 0) {
        if (logger) {
            logger->warn("[BA] All edges filtered out, skipping BA");
        }
        return;
    }

    // ---------------------------------------------------------
    // Basic setup: find number of pose variables
    // ---------------------------------------------------------
    // CRITICAL FIX: Match Python BA exactly!
    // Python: n = max(ii.max().item(), jj.max().item()) + 1
    //   - ii = source frame index
    //   - jj = target frame index
    // In C++:
    //   - m_pg.m_ii[e] = patch index mapping (NOT frame index!)
    //   - m_pg.m_jj[e] = target frame index
    //   - Source frame i is extracted from kk: i = kk[e] / M
    // So we need: n = max(max(i from kk), max(jj)) + 1
    // 
    // CRITICAL: Limit n to m_pg.m_n (actual sliding window size) to prevent optimizing
    // unconstrained poses that have no edges. This prevents BA from optimizing hundreds
    // of poses with 0 constraints, which is slow and causes wrong poses.
    int n_max_from_edges = 0;
    for (int e = 0; e < num_active; e++) {
        if (!edge_valid[e]) continue;  // Skip filtered edges
        
        int i = m_pg.m_kk[e] / M;  // source frame index (extracted from kk, matching reproject logic)
        int j = m_pg.m_jj[e];      // target frame index
        n_max_from_edges = std::max(n_max_from_edges, std::max(i, j) + 1);
    }
    
    // Limit n to actual sliding window size to prevent optimizing unconstrained poses
    int n = std::min(n_max_from_edges, m_pg.m_n);
    
    if (logger && n_max_from_edges > m_pg.m_n) {
        logger->warn("[BA] Limiting n from {} (from edges) to {} (sliding window size) to prevent optimizing unconstrained poses",
                     n_max_from_edges, m_pg.m_n);
    }
    
    if (logger) {
        logger->info("[BA] fixedp={}, n={} (computed from edges), will optimize poses [{}, {}] (matching Python t0/t1 logic)", 
                     fixedp, n, fixedp, n - 1);
    }

    // ---------------------------------------------------------
    // Forward projection (coordinates) + Jacobians
    // ---------------------------------------------------------
    // CRITICAL: Create filtered edge arrays to pass to reproject
    // Only include edges that reference frames within the sliding window
    std::vector<int> ii_filtered, jj_filtered, kk_filtered;
    std::vector<float> target_filtered;
    std::vector<float> weight_filtered;
    std::vector<float> net_filtered;
    ii_filtered.reserve(valid_edge_count);
    jj_filtered.reserve(valid_edge_count);
    kk_filtered.reserve(valid_edge_count);
    target_filtered.reserve(valid_edge_count * 2);
    weight_filtered.reserve(valid_edge_count * 2);
    net_filtered.reserve(valid_edge_count * NET_DIM);
    
    for (int e = 0; e < num_active; e++) {
        if (edge_valid[e]) {
            ii_filtered.push_back(m_pg.m_ii[e]);
            jj_filtered.push_back(m_pg.m_jj[e]);
            kk_filtered.push_back(m_pg.m_kk[e]);
            target_filtered.push_back(m_pg.m_target[e * 2 + 0]);
            target_filtered.push_back(m_pg.m_target[e * 2 + 1]);
            weight_filtered.push_back(m_pg.m_weight[e][0]);
            weight_filtered.push_back(m_pg.m_weight[e][1]);
            for (int d = 0; d < NET_DIM; d++) {
                net_filtered.push_back(m_pg.m_net[e][d]);
            }
        }
    }
    
    std::vector<float> coords(valid_edge_count * 2 * P * P); // [valid_edge_count, 2, P, P]
    std::vector<float> Ji(valid_edge_count * 2 * P * P * 6); // [valid_edge_count, 2, P, P, 6]
    std::vector<float> Jj(valid_edge_count * 2 * P * P * 6); // [valid_edge_count, 2, P, P, 6]
    std::vector<float> Jz(valid_edge_count * 2 * P * P * 1); // [valid_edge_count, 2, P, P, 1]
    std::vector<float> valid(valid_edge_count * P * P); // [valid_edge_count, P, P]
    
    reproject(
        ii_filtered.data(), jj_filtered.data(), kk_filtered.data(), 
        valid_edge_count, 
        coords.data(),
        Ji.data(),  // Jacobian w.r.t. pose i
        Jj.data(),  // Jacobian w.r.t. pose j
        Jz.data(),  // Jacobian w.r.t. inverse depth
        valid.data() // validity mask
    );
    
    // Use valid_edge_count for all subsequent operations (num_active is const)
    // All arrays are now filtered, so use valid_edge_count instead of num_active
    const int num_active_filtered = valid_edge_count;

    // ---------------------------------------------------------
    // Compute residual at patch center
    // ---------------------------------------------------------
    const int p = P;
    const int center_idx = (p / 2) * P + (p / 2); // center pixel index in patch
    
    // Debug: Log coords from BA's reproject() call for comparison
    if (logger && m_counter == TARGET_FRAME) {
        logger->info("[BA] Coords from second reproject() call (first 3 edges):");
        for (int e = 0; e < std::min(3, num_active_filtered); e++) {
            float cx = coords[e * 2 * P * P + 0 * P * P + center_idx];
            float cy = coords[e * 2 * P * P + 1 * P * P + center_idx];
            logger->info("[BA]   Edge[{}]: coords=({:.6f}, {:.6f})", e, cx, cy);
        }
    }
    std::vector<float> r(num_active_filtered * 2); // [num_active_filtered, 2]
    std::vector<float> v(num_active_filtered, 1.0f); // validity mask

    float residual_sum = 0.0f;
    int valid_residuals = 0;

    int nan_residual_count = 0;
    for (int e = 0; e < num_active_filtered; e++) {
        // Extract coordinates at patch center
        float cx = coords[e * 2 * P * P + 0 * P * P + center_idx];
        float cy = coords[e * 2 * P * P + 1 * P * P + center_idx];
        
        // Validate inputs before computing residual
        float target_x = target_filtered[e * 2 + 0];
        float target_y = target_filtered[e * 2 + 1];
        
        // Debug: Log first few edges to diagnose target mismatch (always log for debugging)
        // Expected: residual = targets - coords should equal delta from update model
        // If coords match Python but residuals don't, then targets must be different
        if (logger && e < 5) {
            float residual_x = target_x - cx;
            float residual_y = target_y - cy;
            logger->info("[BA] Edge[{}] - target=({:.6f}, {:.6f}), coords=({:.6f}, {:.6f}), residual=({:.6f}, {:.6f})",
                        e, target_x, target_y, cx, cy, residual_x, residual_y);
        }
        
        if (!std::isfinite(cx) || !std::isfinite(cy) || 
            !std::isfinite(target_x) || !std::isfinite(target_y)) {
            // Invalid input - set residual to zero and invalidate edge
            r[e * 2 + 0] = 0.0f;
            r[e * 2 + 1] = 0.0f;
            v[e] = 0.0f;
            nan_residual_count++;
            if (logger && nan_residual_count <= 5) {
                logger->warn("[BA] Invalid residual[{}]: target=({}, {}), coords=({}, {})", 
                            e, target_x, target_y, cx, cy);
            }
            continue;
        }
        
        // Reprojection residual
        r[e * 2 + 0] = target_x - cx;
        r[e * 2 + 1] = target_y - cy;
        
        // Check if residual itself is NaN/Inf
        if (!std::isfinite(r[e * 2 + 0]) || !std::isfinite(r[e * 2 + 1])) {
            r[e * 2 + 0] = 0.0f;
            r[e * 2 + 1] = 0.0f;
            v[e] = 0.0f;
            nan_residual_count++;
            if (logger && nan_residual_count <= 5) {
                logger->warn("[BA] NaN residual[{}]: target=({}, {}), coords=({}, {}), residual=({}, {})", 
                            e, target_x, target_y, cx, cy, r[e * 2 + 0], r[e * 2 + 1]);
            }
            continue;
        }

        // Reject large residuals
        float r_norm = std::sqrt(r[e * 2 + 0] * r[e * 2 + 0] + r[e * 2 + 1] * r[e * 2 + 1]);
        if (r_norm >= 250.0f || !std::isfinite(r_norm)) {
            v[e] = 0.0f;
        } else {
            residual_sum += r_norm;
            valid_residuals++;
        }
    }
    
    if (logger && nan_residual_count > 0) {
        logger->warn("[BA] {} out of {} edges have NaN/Inf residuals", nan_residual_count, num_active_filtered);
    }
    
    if (logger) {
        logger->info("[BA] Residual stats - valid={}/{}, mean_residual={:.4f}", 
                     valid_residuals, num_active_filtered, 
                     valid_residuals > 0 ? residual_sum / valid_residuals : 0.0f);
        if (valid_residuals > 0 && valid_residuals < 5) {
            // Log first few residuals for debugging
            for (int e = 0; e < std::min(3, num_active_filtered); e++) {
                float r_norm = std::sqrt(r[e * 2 + 0] * r[e * 2 + 0] + r[e * 2 + 1] * r[e * 2 + 1]);
                logger->info("[BA] Residual[{}]: target=({:.2f}, {:.2f}), coords=({:.2f}, {:.2f}), "
                             "residual=({:.4f}, {:.4f}), norm={:.4f}, valid={}",
                             e, target_filtered[e * 2 + 0], target_filtered[e * 2 + 1],
                             coords[e * 2 * P * P + 0 * P * P + center_idx],
                             coords[e * 2 * P * P + 1 * P * P + center_idx],
                             r[e * 2 + 0], r[e * 2 + 1], r_norm, v[e]);
            }
        }
    }
    
    // Compute bounds from intrinsics (matching Python: W = 2 * max(cx), H = 2 * max(cy))
    // Python: bounds = [0.0, 0.0, W - 1.0, H - 1.0] where W = 2 * max(intrinsics[:, 2]), H = 2 * max(intrinsics[:, 3])
    float max_cx = 0.0f;
    float max_cy = 0.0f;
    for (int i = 0; i < n; i++) {
        max_cx = std::max(max_cx, m_pg.m_intrinsics[i][2]);  // cx
        max_cy = std::max(max_cy, m_pg.m_intrinsics[i][3]);  // cy
    }
    float bounds_W = 2.0f * max_cx;  // W = 2 * max(cx)
    float bounds_H = 2.0f * max_cy;   // H = 2 * max(cy)
    float bounds_xmax = bounds_W - 1.0f;  // bounds[2] = W - 1.0
    float bounds_ymax = bounds_H - 1.0f;  // bounds[3] = H - 1.0
    
    if (logger) {
        logger->info("[BA] Computed bounds from intrinsics - max_cx={:.2f}, max_cy={:.2f}, W={:.1f}, H={:.1f}, bounds=[0.0, 0.0, {:.1f}, {:.1f}]",
                     max_cx, max_cy, bounds_W, bounds_H, bounds_xmax, bounds_ymax);
    }
    
    // Continue with remaining validity checks (this was outside the loop, moved here)
    for (int e = 0; e < num_active_filtered; e++) {
        // Extract coordinates at patch center for bounds check
        float cx = coords[e * 2 * P * P + 0 * P * P + center_idx];
        float cy = coords[e * 2 * P * P + 1 * P * P + center_idx];
        
        // Reject projections outside image bounds
        // CRITICAL: Match Python BA bounds check exactly
        // Python: bounds = [0.0, 0.0, W-1.0, H-1.0], checks: cx > 0.0, cy > 0.0, cx < W-1.0, cy < H-1.0
        // C++: Use computed bounds from intrinsics, check: cx > 0.0, cy > 0.0, cx < bounds_xmax, cy < bounds_ymax
        bool out_of_bounds = (cx <= 0.0f || cy <= 0.0f || cx >= bounds_xmax || cy >= bounds_ymax);
        if (out_of_bounds) {
            v[e] = 0.0f;
            // Debug: Log out-of-bounds edges for comparison with Python
            if (logger && e < 10) {
                logger->info("[BA] Edge[{}] out-of-bounds - cx={:.2f}, cy={:.2f}, bounds=[0.0, 0.0, {:.1f}, {:.1f}]",
                            e, cx, cy, bounds_xmax, bounds_ymax);
            }
        }
        
        // Also use validity from transformWithJacobians
        float valid_center = valid[e * P * P + center_idx];
        if (valid_center < 0.5f) {
            v[e] = 0.0f;
        }
    }

    // Post-frame-1000 residual summary (after bounds check so v[] is finalized)
    if (logger && log_post1000) {
        float max_r_norm = 0.0f;
        int oob_count = 0;
        int large_r_count = 0;
        int invalid_count = 0;
        for (int e = 0; e < num_active_filtered; e++) {
            float rn = std::sqrt(r[e * 2 + 0] * r[e * 2 + 0] + r[e * 2 + 1] * r[e * 2 + 1]);
            if (rn > max_r_norm) max_r_norm = rn;
            if (v[e] < 0.5f) {
                float cx_e = coords[e * 2 * P * P + 0 * P * P + center_idx];
                float cy_e = coords[e * 2 * P * P + 1 * P * P + center_idx];
                if (cx_e <= 0.0f || cy_e <= 0.0f || cx_e >= bounds_xmax || cy_e >= bounds_ymax)
                    oob_count++;
                else
                    large_r_count++;
                invalid_count++;
            }
        }
        logger->info("[BA][F{}] residual: edges={}, valid={}, invalid={} (nan={}, oob={}, large_r={}), mean_r={:.4f}, max_r={:.4f}",
                     m_counter, num_active_filtered, valid_residuals, invalid_count, nan_residual_count, oob_count, large_r_count,
                     valid_residuals > 0 ? residual_sum / valid_residuals : 0.0f, max_r_norm);
    }

    // Apply validity mask to residuals and weights
    // Match Python BA: use both weight channels separately (matching Python [1, M, 2] format)
    // Python: weights shape [1, M, 2] -> [1, M, 2, 1] after unsqueeze, then multiply with Ji [1, M, 2, 6]
    // Channel 0 (w0) applies to x-direction, Channel 1 (w1) applies to y-direction
    std::vector<float> weights_masked(num_active_filtered * 2);
    for (int e = 0; e < num_active_filtered; e++) {
        r[e * 2 + 0] *= v[e];
        r[e * 2 + 1] *= v[e];
        float w0 = weight_filtered[e * 2 + 0] * v[e];  // Channel 0: weight for x-direction
        float w1 = weight_filtered[e * 2 + 1] * v[e];  // Channel 1: weight for y-direction
        weights_masked[e * 2 + 0] = w0;  // Weight for x-direction
        weights_masked[e * 2 + 1] = w1;  // Weight for y-direction (matching Python)
    }
    
    // Save STEP 1: Residuals and validity mask
    if (save_intermediates) {
        save_ba_step1(r.data(), v.data(), coords.data(), num_active_filtered, P, center_idx, logger);
    } else {
        if (logger && m_counter % 10 == 0) {  // Log every 10 frames to avoid spam
            logger->debug("[BA] Skipping STEP 1 save (m_counter={} != TARGET_FRAME={})", m_counter, TARGET_FRAME);
        }
    }
    
    // Extract Jacobians at patch center: [num_active_filtered, 2, 6] for Ji, Jj, [num_active_filtered, 2, 1] for Jz
    std::vector<float> Ji_center(num_active_filtered * 2 * 6); // [num_active_filtered, 2, 6]
    std::vector<float> Jj_center(num_active_filtered * 2 * 6); // [num_active_filtered, 2, 6]
    std::vector<float> Jz_center(num_active_filtered * 2 * 1); // [num_active_filtered, 2, 1]
    
    for (int e = 0; e < num_active_filtered; e++) {
        // Extract Jacobians at patch center
        // Ji: [num_active_filtered, 2, P, P, 6] -> [num_active_filtered, 2, 6] at center
        for (int c = 0; c < 2; c++) {
            for (int d = 0; d < 6; d++) {
                int src_idx = e * 2 * P * P * 6 + c * P * P * 6 + center_idx * 6 + d;
                int dst_idx = e * 2 * 6 + c * 6 + d;
                Ji_center[dst_idx] = Ji[src_idx];
                Jj_center[dst_idx] = Jj[src_idx];
            }
        }
        // Jz: [num_active, 2, P, P, 1] -> [num_active, 2, 1] at center
        for (int c = 0; c < 2; c++) {
            int src_idx = e * 2 * P * P * 1 + c * P * P * 1 + center_idx * 1;
            int dst_idx = e * 2 * 1 + c * 1;
            Jz_center[dst_idx] = Jz[src_idx];
        }
    }

    // ---------------------------------------------------------
    // Step 2: Build weighted Jacobians: wJiT, wJjT, wJzT
    // ---------------------------------------------------------
    // Reshape r to [num_active_filtered, 2, 1] for matrix operations
    // wJiT = (weights * Ji).transpose(2, 3) -> [num_active_filtered, 6, 2]
    // wJjT = (weights * Jj).transpose(2, 3) -> [num_active_filtered, 6, 2]
    // wJzT = (weights * Jz).transpose(2, 3) -> [num_active_filtered, 1, 2]
    
    std::vector<Eigen::Matrix<float, 6, 2>> wJiT(num_active_filtered);
    std::vector<Eigen::Matrix<float, 6, 2>> wJjT(num_active_filtered);
    std::vector<Eigen::Matrix<float, 1, 2>> wJzT(num_active_filtered);
    
    for (int e = 0; e < num_active_filtered; e++) {
        // Match Python BA: use both weight channels separately (matching Python [1, M, 2] format)
        // Python: weights shape [1, M, 2] -> [1, M, 2, 1] after unsqueeze, then multiply with Ji [1, M, 2, 6]
        // Channel 0 (w0) applies to x-direction, Channel 1 (w1) applies to y-direction
        float w0 = weights_masked[e * 2 + 0];  // Channel 0: weight for x-direction
        float w1 = weights_masked[e * 2 + 1];  // Channel 1: weight for y-direction
        
        // NOTE: Python BA does NOT skip edges with zero weights - it computes Hessian blocks for all edges
        // Even if weights are zero, the computation should proceed (result will be zero, but consistent with Python)
        // Removing the early skip to match Python behavior
        
        // Ji_center: [num_active_filtered, 2, 6] -> transpose to [6, 2]
        // Jj_center: [num_active_filtered, 2, 6] -> transpose to [6, 2]
        // Jz_center: [num_active_filtered, 2, 1] -> transpose to [1, 2]
        // Apply w0 to x-direction and w1 to y-direction (matching Python broadcasting)
        for (int i = 0; i < 6; i++) {
            wJiT[e](i, 0) = w0 * Ji_center[e * 2 * 6 + 0 * 6 + i];  // x-direction: use w0
            wJiT[e](i, 1) = w1 * Ji_center[e * 2 * 6 + 1 * 6 + i];  // y-direction: use w1
            wJjT[e](i, 0) = w0 * Jj_center[e * 2 * 6 + 0 * 6 + i];  // x-direction: use w0
            wJjT[e](i, 1) = w1 * Jj_center[e * 2 * 6 + 1 * 6 + i];  // y-direction: use w1
        }
        wJzT[e](0, 0) = w0 * Jz_center[e * 2 * 1 + 0 * 1];  // x-direction: use w0
        wJzT[e](0, 1) = w1 * Jz_center[e * 2 * 1 + 1 * 1];  // y-direction: use w1
    }

    // ---------------------------------------------------------
    // Step 3: Compute Hessian blocks
    // ---------------------------------------------------------
    // Bii = wJiT @ Ji [6, 6]
    // Bij = wJiT @ Jj [6, 6]
    // Bji = wJjT @ Ji [6, 6]
    // Bjj = wJjT @ Jj [6, 6]
    // Eik = wJiT @ Jz [6, 1]
    // Ejk = wJjT @ Jz [6, 1]
    std::vector<Eigen::Matrix<float, 6, 6>> Bii(num_active_filtered);
    std::vector<Eigen::Matrix<float, 6, 6>> Bij(num_active_filtered);
    std::vector<Eigen::Matrix<float, 6, 6>> Bji(num_active_filtered);
    std::vector<Eigen::Matrix<float, 6, 6>> Bjj(num_active_filtered);
    std::vector<Eigen::Matrix<float, 6, 1>> Eik(num_active_filtered);
    std::vector<Eigen::Matrix<float, 6, 1>> Ejk(num_active_filtered);
    
    for (int e = 0; e < num_active_filtered; e++) {
        // Ji_center: [2, 6], Jj_center: [2, 6], Jz_center: [2, 1]
        Eigen::Matrix<float, 2, 6> Ji_mat, Jj_mat;
        Eigen::Matrix<float, 2, 1> Jz_mat;
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 6; j++) {
                Ji_mat(i, j) = Ji_center[e * 2 * 6 + i * 6 + j];
                Jj_mat(i, j) = Jj_center[e * 2 * 6 + i * 6 + j];
            }
            Jz_mat(i, 0) = Jz_center[e * 2 * 1 + i * 1];
        }
        
        Bii[e] = wJiT[e] * Ji_mat;
        Bij[e] = wJiT[e] * Jj_mat;
        Bji[e] = wJjT[e] * Ji_mat;
        Bjj[e] = wJjT[e] * Jj_mat;
        Eik[e] = wJiT[e] * Jz_mat;
        Ejk[e] = wJjT[e] * Jz_mat;
    }
    
    // Save STEP 3: Hessian blocks (including Ejk for comparison)
    if (save_intermediates) {
        std::ofstream Ejk_file(get_bin_file_path("ba_step3_Ejk.bin"), std::ios::binary);
        if (Ejk_file.is_open()) {
            Ejk_file.write(reinterpret_cast<const char*>(Ejk.data()), sizeof(float) * num_active_filtered * 6 * 1);
            Ejk_file.close();
            if (logger) logger->info("[BA] Saved STEP 3 - Ejk blocks (size: {}x{}x{})", num_active_filtered, 6, 1);
        }
    }
    
    // Save STEP 2: Weighted Jacobians
    if (save_intermediates) {
        save_ba_step2(wJiT, wJjT, wJzT, weights_masked.data(), Ji_center.data(), Jj_center.data(), num_active_filtered, logger);
    }
    
    // Save STEP 3: Hessian blocks
    if (save_intermediates) {
        save_ba_step3(Bii, Bij, Bji, Bjj, Eik, Ejk, num_active_filtered, logger);
    }

    // ---------------------------------------------------------
    // Step 4: Compute gradients
    // ---------------------------------------------------------
    // vi = wJiT @ r [6, 1]
    // vj = wJjT @ r [6, 1]
    // w = wJzT @ r [1, 1]
    
    std::vector<Eigen::Matrix<float, 6, 1>> vi(num_active_filtered);
    std::vector<Eigen::Matrix<float, 6, 1>> vj(num_active_filtered);
    std::vector<float> w_vec(num_active_filtered);
    
    for (int e = 0; e < num_active_filtered; e++) {
        Eigen::Matrix<float, 2, 1> r_vec;
        r_vec(0, 0) = r[e * 2 + 0];
        r_vec(1, 0) = r[e * 2 + 1];
        
        vi[e] = wJiT[e] * r_vec;
        vj[e] = wJjT[e] * r_vec;
        w_vec[e] = (wJzT[e] * r_vec)(0, 0);
    }
    
    // Save STEP 4: Gradients
    if (save_intermediates) {
        save_ba_step4(vi, vj, w_vec.data(), num_active_filtered, logger);
    }

    // ---------------------------------------------------------
    // Step 5: Fix first pose (gauge freedom)
    // ---------------------------------------------------------
    // CRITICAL FIX: Match Python BA exactly!
    // Python: ii = ii - fixedp; jj = jj - fixedp
    //   - ii = source frame index
    //   - jj = target frame index
    // In C++:
    //   - m_pg.m_ii[e] = patch index mapping (NOT frame index!)
    //   - Source frame i must be extracted from kk: i = kk[e] / M
    //   - m_pg.m_jj[e] = target frame index
    std::vector<int> ii_new(num_active_filtered);
    std::vector<int> jj_new(num_active_filtered);
    
    // Debug: Track which edges connect to which poses
    std::map<int, std::vector<int>> edges_to_pose_i;
    std::map<int, std::vector<int>> edges_to_pose_j;
    
    for (int e = 0; e < num_active_filtered; e++) {
        int i_source = kk_filtered[e] / M;  // Extract source frame index from kk (matching reproject logic)
        ii_new[e] = i_source - fixedp;     // Adjust source frame index for fixed poses
        jj_new[e] = jj_filtered[e] - fixedp; // Adjust target frame index for fixed poses
        
        // Track edges for debugging (only valid edges)
        if (v[e] >= 0.5f) {
            edges_to_pose_i[i_source].push_back(e);
            edges_to_pose_j[jj_filtered[e]].push_back(e);
        }
    }
    
    // CRITICAL: Find the maximum pose index that actually has edges
    // This prevents optimizing poses with 0 constraints (which causes slowdown and wrong poses)
    int max_pose_with_edges = fixedp;
    for (int pose_idx = fixedp; pose_idx < n; pose_idx++) {
        int edges_as_i = edges_to_pose_i[pose_idx].size();
        int edges_as_j = edges_to_pose_j[pose_idx].size();
        if (edges_as_i > 0 || edges_as_j > 0) {
            max_pose_with_edges = pose_idx + 1;  // +1 because n is exclusive
        }
    }
    
    // Limit n to only include poses that have edges
    n = std::min(n, max_pose_with_edges);
    int n_adjusted = n - fixedp; // number of pose variables after fixing
    
    // CRITICAL: Early return if no poses to adjust (all poses are fixed)
    // This prevents creating empty matrices and causing assertion failures
    if (n_adjusted <= 0) {
        if (logger) {
            logger->warn("[BA] Early return - n_adjusted={} (n={}, fixedp={}). All poses are fixed, nothing to optimize.",
                        n_adjusted, n, fixedp);
        }
        return;  // No poses to optimize, skip BA
    }
    
    if (logger) {
        logger->info("[BA] num_active={}, n={} (limited to poses with edges), fixedp={}, n_adjusted={}", 
                     num_active_filtered, n, fixedp, n_adjusted);
        
        if (max_pose_with_edges < m_pg.m_n) {
            logger->warn("[BA] Limited optimization window from {} to {} poses (only poses with edges are optimized)",
                        m_pg.m_n, max_pose_with_edges);
        }
        
        // Debug: Log edge connections for first few poses
        for (int pose_idx = 0; pose_idx < std::min(5, n); pose_idx++) {
            int adjusted_idx = pose_idx - fixedp;
            int edges_as_i = edges_to_pose_i[pose_idx].size();
            int edges_as_j = edges_to_pose_j[pose_idx].size();
            logger->info("[BA] Pose[{}] (adjusted_idx={}) - {} edges as source (i), {} edges as target (j)",
                        pose_idx, adjusted_idx, edges_as_i, edges_as_j);
        }
    }

    // ---------------------------------------------------------
    // Step 6: Reindex structure variables
    // ---------------------------------------------------------
    std::vector<int> kk_new(num_active_filtered);
    std::vector<int> kx; // unique structure indices
    std::map<int, int> kk_to_idx; // mapping from original kk to unique index
    
    // Extract unique kk values
    std::set<int> kk_set;
    for (int e = 0; e < num_active_filtered; e++) {
        kk_set.insert(kk_filtered[e]);
    }
    
    kx.assign(kk_set.begin(), kk_set.end());
    std::sort(kx.begin(), kx.end());
    
    // Create mapping
    for (size_t i = 0; i < kx.size(); i++) {
        kk_to_idx[kx[i]] = static_cast<int>(i);
    }
    
    // Create new kk indices
    for (int e = 0; e < num_active_filtered; e++) {
        kk_new[e] = kk_to_idx[kk_filtered[e]];
    }
    
    int m = static_cast<int>(kx.size()); // number of structure variables

    // ---------------------------------------------------------
    // Step 7: Scatter-add to assemble global Hessian B [n, n, 6, 6]
    // ---------------------------------------------------------
    // B is block-sparse: B[i, j] is a 6x6 block
    // We'll use a dense representation for now (can be optimized later)
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(6 * n_adjusted, 6 * n_adjusted);
    
    // Debug: Count edges per adjusted pose index
    std::vector<int> edge_count_per_pose(n_adjusted, 0);
    
    // Count edges connecting to fixed poses (for debugging)
    int edges_to_fixed_i = 0;
    int edges_to_fixed_j = 0;
    int edges_both_adjustable = 0;
    
    // NOTE: Python's safe_scatter_add_mat filters edges based ONLY on indices:
    //   v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    // It does NOT filter by the validity mask v from residual checks.
    // The validity mask is already applied to weights when computing wJiT/wJjT,
    // so the Hessian blocks Bii/Bij/etc. already have the validity mask incorporated.
    // We should NOT filter by v[e] here - only filter by indices to match Python.
    
    for (int e = 0; e < num_active_filtered; e++) {
        int i = ii_new[e];
        int j = jj_new[e];
        
        // Python's safe_scatter_add_mat filters each block type separately:
        //   Bii: v = (ii >= 0) & (ii >= 0) & (ii < n) & (ii < n)  -> only check i
        //   Bij: v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < n)  -> check both i and j
        //   Bji: v = (jj >= 0) & (ii >= 0) & (jj < n) & (ii < n)  -> check both j and i
        //   Bjj: v = (jj >= 0) & (jj >= 0) & (jj < n) & (jj < n)  -> only check j
        // So edges where one pose is fixed can still contribute to diagonal blocks (Bii/Bjj)
        // of the adjustable pose.
        
        // Bii: add if i is adjustable (i >= 0 && i < n_adjusted)
        if (i >= 0 && i < n_adjusted) {
            B.block<6, 6>(6 * i, 6 * i) += Bii[e];
            edge_count_per_pose[i]++;
        }
        
        // Bij: add if both i and j are adjustable
        if (i >= 0 && i < n_adjusted && j >= 0 && j < n_adjusted) {
            B.block<6, 6>(6 * i, 6 * j) += Bij[e];
        }
        
        // Bji: add if both j and i are adjustable
        if (j >= 0 && j < n_adjusted && i >= 0 && i < n_adjusted) {
            B.block<6, 6>(6 * j, 6 * i) += Bji[e];
        }
        
        // Bjj: add if j is adjustable (j >= 0 && j < n_adjusted)
        if (j >= 0 && j < n_adjusted) {
            B.block<6, 6>(6 * j, 6 * j) += Bjj[e];
            edge_count_per_pose[j]++;
        }
        
        // Count edge types for debugging
        if (i >= 0 && i < n_adjusted && j >= 0 && j < n_adjusted) {
            edges_both_adjustable++;
        } else {
            if (i < 0) edges_to_fixed_i++;
            if (j < 0) edges_to_fixed_j++;
        }
    }
    
    if (logger) {
        logger->info("[BA] Edge statistics - both adjustable: {}, to fixed i: {}, to fixed j: {}, total valid: {}",
                     edges_both_adjustable, edges_to_fixed_i, edges_to_fixed_j, 
                     edges_both_adjustable + edges_to_fixed_i + edges_to_fixed_j);
    }
    
    // Debug: Log edge count per pose after assembly
    if (logger) {
        for (int idx = 0; idx < n_adjusted; idx++) {
            int pose_idx = fixedp + idx;
            logger->info("[BA] Adjusted pose idx={} (global pose_idx={}) has {} edges contributing to Hessian",
                         idx, pose_idx, edge_count_per_pose[idx]);
        }
    }
    
    // Save STEP 9: Assembled Hessian B
    if (save_intermediates) {
        save_ba_step9(B, n_adjusted, logger);
    }

    // ---------------------------------------------------------
    // Step 8: Assemble pose-structure coupling E [n, m, 6, 1]
    // ---------------------------------------------------------
    // E is reshaped to [6n, m] for matrix operations
    // NOTE: Python's safe_scatter_add_mat filters edges based ONLY on indices:
    //   Eik: v = (ii >= 0) & (kk >= 0) & (ii < n) & (kk < m)  -> only check i and k
    //   Ejk: v = (jj >= 0) & (kk >= 0) & (jj < n) & (kk < m)  -> only check j and k
    // It does NOT filter by the validity mask v from residual checks.
    // The validity mask is already applied to weights when computing wJiT/wJjT,
    // so Eik/Ejk already have the validity mask incorporated.
    // We should NOT filter by v[e] here - only filter by indices to match Python.
    Eigen::MatrixXf E = Eigen::MatrixXf::Zero(6 * n_adjusted, m);
    
    // Debug: Track which edges contribute to specific E entries (for debugging mismatches)
    std::vector<std::vector<int>> edges_to_E_ik(6 * n_adjusted * m);  // Flattened: [6*n_adjusted*m]
    std::vector<std::vector<int>> edges_to_E_jk(6 * n_adjusted * m);
    
    for (int e = 0; e < num_active_filtered; e++) {
        int i = ii_new[e];
        int j = jj_new[e];
        int k = kk_new[e];
        
        // Eik: add if i is adjustable and k is valid (matches Python's safe_scatter_add_mat for Eik)
        if (i >= 0 && i < n_adjusted && k >= 0 && k < m) {
            E.block<6, 1>(6 * i, k) += Eik[e];
            // Track which edge contributes to each E entry (for debugging)
            if (save_intermediates && m_counter == TARGET_FRAME) {
                for (int param = 0; param < 6; param++) {
                    int flat_idx = (6 * i + param) * m + k;
                    if (flat_idx < static_cast<int>(edges_to_E_ik.size())) {
                        edges_to_E_ik[flat_idx].push_back(e);
                    }
                }
            }
        }
        
        // Ejk: add if j is adjustable and k is valid (matches Python's safe_scatter_add_mat for Ejk)
        if (j >= 0 && j < n_adjusted && k >= 0 && k < m) {
            E.block<6, 1>(6 * j, k) += Ejk[e];
            // Track which edge contributes to each E entry (for debugging)
            if (save_intermediates && m_counter == TARGET_FRAME) {
                for (int param = 0; param < 6; param++) {
                    int flat_idx = (6 * j + param) * m + k;
                    if (flat_idx < static_cast<int>(edges_to_E_jk.size())) {
                        edges_to_E_jk[flat_idx].push_back(e);
                    }
                }
            }
        }
    }
    
    // Debug: Log edges contributing to E[0, 1] (pose 0, param 0, struct var 1) if this is target frame
    if (save_intermediates && logger) {
        int debug_pose = 0;
        int debug_param = 0;
        int debug_struct = 1;
        int debug_flat_idx = (6 * debug_pose + debug_param) * m + debug_struct;
        if (debug_flat_idx < static_cast<int>(edges_to_E_ik.size()) && 
            debug_flat_idx < static_cast<int>(edges_to_E_jk.size())) {
            logger->info("[BA] Edges contributing to E[pose={}, param={}, struct={}]:", 
                        debug_pose, debug_param, debug_struct);
            
            // Convert vector<int> to string for logging
            std::string eik_edges_str = "[";
            for (size_t i = 0; i < edges_to_E_ik[debug_flat_idx].size(); i++) {
                if (i > 0) eik_edges_str += ", ";
                eik_edges_str += std::to_string(edges_to_E_ik[debug_flat_idx][i]);
            }
            eik_edges_str += "]";
            logger->info("[BA]   Eik edges: {}", eik_edges_str);
            
            std::string ejk_edges_str = "[";
            for (size_t i = 0; i < edges_to_E_jk[debug_flat_idx].size(); i++) {
                if (i > 0) ejk_edges_str += ", ";
                ejk_edges_str += std::to_string(edges_to_E_jk[debug_flat_idx][i]);
            }
            ejk_edges_str += "]";
            logger->info("[BA]   Ejk edges: {}", ejk_edges_str);
            
            for (int e : edges_to_E_ik[debug_flat_idx]) {
                logger->info("[BA]     Edge {}: ii_new={}, kk_new={}, Eik[{}]=({:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f})", 
                            e, ii_new[e], kk_new[e], e,
                            Eik[e](0, 0), Eik[e](1, 0), Eik[e](2, 0), Eik[e](3, 0), Eik[e](4, 0), Eik[e](5, 0));
            }
            for (int e : edges_to_E_jk[debug_flat_idx]) {
                logger->info("[BA]     Edge {}: jj_new={}, kk_new={}, Ejk[{}]=({:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f})", 
                            e, jj_new[e], kk_new[e], e,
                            Ejk[e](0, 0), Ejk[e](1, 0), Ejk[e](2, 0), Ejk[e](3, 0), Ejk[e](4, 0), Ejk[e](5, 0));
            }
        }
    }
    
    // Save STEP 10: Pose-structure coupling E
    if (save_intermediates) {
        save_ba_step10(E, n_adjusted, m, logger);
    }

    // ---------------------------------------------------------
    // Step 9: Structure Hessian C [m] (diagonal)
    // ---------------------------------------------------------
    // C = sum over edges: wJzT @ Jz (scalar per edge)
    Eigen::VectorXf C = Eigen::VectorXf::Zero(m);
    
    for (int e = 0; e < num_active_filtered; e++) {
        if (v[e] < 0.5f) continue;
        
        int k = kk_new[e];
        if (k < 0 || k >= m) continue;
        
        // C[k] += wJzT @ Jz (scalar)
        // wJzT is [1, 2], Jz is [2, 1], result is scalar
        Eigen::Matrix<float, 2, 1> Jz_mat;
        Jz_mat(0, 0) = Jz_center[e * 2 * 1 + 0];
        Jz_mat(1, 0) = Jz_center[e * 2 * 1 + 1];
        C[k] += (wJzT[e] * Jz_mat)(0, 0);
    }
    
    // Save STEP 11: Structure Hessian C
    if (save_intermediates) {
        save_ba_step11_C(C, m, logger);
    }

    // ---------------------------------------------------------
    // Step 10: Schur complement solve
    // ---------------------------------------------------------
    // Assemble gradient vectors
    Eigen::VectorXf v_grad = Eigen::VectorXf::Zero(6 * n_adjusted);
    Eigen::VectorXf w_grad = Eigen::VectorXf::Zero(m);
    
    for (int e = 0; e < num_active_filtered; e++) {
        if (v[e] < 0.5f) continue;
        
        int i = ii_new[e];
        int j = jj_new[e];
        int k = kk_new[e];
        
        // Check for NaN/Inf in gradients before adding
        bool vi_valid = true, vj_valid = true, w_valid = true;
        for (int idx = 0; idx < 6; idx++) {
            if (!std::isfinite(vi[e](idx, 0))) vi_valid = false;
            if (!std::isfinite(vj[e](idx, 0))) vj_valid = false;
        }
        if (!std::isfinite(w_vec[e])) w_valid = false;
        
        if (vi_valid && i >= 0 && i < n_adjusted) {
            v_grad.segment<6>(6 * i) += vi[e];
        }
        if (vj_valid && j >= 0 && j < n_adjusted) {
            v_grad.segment<6>(6 * j) += vj[e];
        }
        if (w_valid && k >= 0 && k < m) {
            w_grad[k] += w_vec[e];
        }
    }
    
    if (logger) {
        // Check for NaN/Inf in assembled gradients
        bool has_nan = false;
        for (int i = 0; i < v_grad.size(); i++) {
            if (!std::isfinite(v_grad[i])) {
                has_nan = true;
                break;
            }
        }
        for (int i = 0; i < w_grad.size(); i++) {
            if (!std::isfinite(w_grad[i])) {
                has_nan = true;
                break;
            }
        }
        
        if (has_nan) {
            logger->warn("[BA] Gradient contains NaN/Inf values! Checking individual components...");
            // Log first few problematic gradients
            for (int e = 0; e < std::min(5, num_active_filtered); e++) {
                bool vi_has_nan = false, vj_has_nan = false;
                for (int idx = 0; idx < 6; idx++) {
                    if (!std::isfinite(vi[e](idx, 0))) vi_has_nan = true;
                    if (!std::isfinite(vj[e](idx, 0))) vj_has_nan = true;
                }
                if (vi_has_nan || vj_has_nan || !std::isfinite(w_vec[e])) {
                    float w0 = weight_filtered[e * 2 + 0];  // Channel 0: weight for x-direction
                    float w1 = weight_filtered[e * 2 + 1];  // Channel 1: weight for y-direction
                    logger->warn("[BA] Edge[{}]: vi_has_nan={}, vj_has_nan={}, w_valid={}, "
                                "r=({:.4f}, {:.4f}), weight=({:.4f}, {:.4f})",
                                e, vi_has_nan, vj_has_nan, std::isfinite(w_vec[e]),
                                r[e * 2 + 0], r[e * 2 + 1], w0, w1);
                }
            }
        }
        
        float v_grad_norm = v_grad.norm();
        float v_grad_max = v_grad.cwiseAbs().maxCoeff();
        float w_grad_norm = w_grad.norm();
        float w_grad_max = w_grad.cwiseAbs().maxCoeff();
        logger->info("[BA] Gradient stats - v_grad_norm={:.6f}, v_grad_max={:.6f}, w_grad_norm={:.6f}, w_grad_max={:.6f}", 
                     v_grad_norm, v_grad_max, w_grad_norm, w_grad_max);
        
        // Check if gradients are zero
        if (v_grad_norm < 1e-6f && w_grad_norm < 1e-6f) {
            logger->warn("[BA] WARNING - Both v_grad and w_grad are near zero! This means BA won't update poses.");
            logger->warn("[BA] Possible causes: 1) Residuals are zero (poses optimal), 2) Weights are zero, 3) Jacobians are zero");
            // Log sample residuals and weights to diagnose
            int sample_count = std::min(5, num_active_filtered);
            for (int e = 0; e < sample_count; e++) {
                float r_norm = std::sqrt(r[e * 2 + 0] * r[e * 2 + 0] + r[e * 2 + 1] * r[e * 2 + 1]);
                float w0 = weight_filtered[e * 2 + 0];  // Channel 0: weight for x-direction
                float w1 = weight_filtered[e * 2 + 1];  // Channel 1: weight for y-direction
                logger->warn("[BA] Sample edge[{}]: residual_norm={:.6f}, weight=({:.6f}, {:.6f}), valid={}", 
                            e, r_norm, w0, w1, v[e] > 0.5f ? 1 : 0);
            }
        }
    }
    
    // Save STEP 11: Assembled gradients
    if (save_intermediates) {
        save_ba_step11_gradients(v_grad, w_grad, n_adjusted, m, logger);
    }
    
    // Post-frame-1000 Hessian and gradient summary
    if (logger && log_post1000) {
        // B matrix stats
        float B_diag_min = B.diagonal().minCoeff();
        float B_diag_max = B.diagonal().maxCoeff();
        float B_norm = B.norm();
        // E matrix stats
        float E_norm = E.norm();
        float E_max = E.cwiseAbs().maxCoeff();
        // C stats
        float C_min = C.minCoeff();
        float C_max = C.maxCoeff();
        logger->info("[BA][F{}] Hessian: n_adj={}, m={}, B_diag=[{:.4f},{:.4f}], B_norm={:.4f}, E_norm={:.4f}, E_max={:.4f}, C=[{:.4f},{:.4f}]",
                     m_counter, n_adjusted, m, B_diag_min, B_diag_max, B_norm, E_norm, E_max, C_min, C_max);
        logger->info("[BA][F{}] gradient: v_grad_norm={:.6f}, v_grad_max={:.6f}, w_grad_norm={:.6f}, w_grad_max={:.6f}",
                     m_counter, v_grad.norm(), v_grad.cwiseAbs().maxCoeff(), w_grad.norm(), w_grad.cwiseAbs().maxCoeff());
    }
    
    // Levenberg-Marquardt damping
    Eigen::VectorXf C_lm = C.array() + lmbda;
    Eigen::VectorXf Q = 1.0f / C_lm.array(); // C^-1 (diagonal)
    
    // Save STEP 13: Q (inverse structure Hessian)
    if (save_intermediates) {
        save_ba_step13(Q, m, logger);
    }

    // Schur complement: S = B - E * C^-1 * E^T
    Eigen::MatrixXf EQ = E * Q.asDiagonal(); // E * C^-1
    Eigen::MatrixXf S = B - EQ * E.transpose();
    
    // RHS: y = v - E * C^-1 * w
    Eigen::VectorXf y = v_grad - EQ * w_grad;
    
    // Save STEP 14: Schur complement S and RHS y
    if (save_intermediates) {
        save_ba_step14(S, y, n_adjusted, logger);
    }
    
    if (logger) {
        float y_norm = y.norm();
        float EQ_w_norm = (EQ * w_grad).norm();
        logger->info("[BA] RHS vector y stats - y.norm()={:.6f}, v_grad.norm()={:.6f}, (EQ*w_grad).norm()={:.6f}", 
                     y_norm, v_grad.norm(), EQ_w_norm);
        if (y_norm < 1e-6f) {
            logger->warn("[BA] y vector is near zero - BA will not update poses! Check residuals and weights.");
        }
    }
    
    // Solve for pose increments: S * dX = y
    Eigen::VectorXf dX;
    Eigen::VectorXf dZ;
    
    if (structure_only || n_adjusted == 0) {
        // Only update structure
        // Python: dZ = (Q * w).view(b, -1, 1, 1)
        dX = Eigen::VectorXf::Zero(6 * n_adjusted);
        dZ = Q.asDiagonal() * w_grad;
    } else {
        // Python: A = A + (ep + lm * A) * torch.eye(n1*p1, device=A.device)
        // where ep=100.0 (from BA call), lm=1e-4 (hardcoded in block_solve)
        // Formula: A[i,i] += ep + lm * A[i,i]
        // This is equivalent to: A[i,i] = A[i,i] * (1 + lm) + ep
        Eigen::MatrixXf S_damped = S;
        float lm = 1e-4f;
        // Match Python exactly: add (ep + lm * S[i,i]) to each diagonal element
        for (int i = 0; i < 6 * n_adjusted; i++) {
            S_damped(i, i) += ep + lm * S(i, i);
        }
        
        // Python uses Cholesky solver (matches Python: CholeskySolver.apply)
        // If Cholesky fails, it returns zeros (matches Python behavior)
        Eigen::LLT<Eigen::MatrixXf> solver(S_damped);
        if (solver.info() != Eigen::Success) {
            // Python: if cholesky fails, return zeros
            if (logger) {
                logger->warn("[BA] Cholesky solver failed with info={}", static_cast<int>(solver.info()));
            }
            dX = Eigen::VectorXf::Zero(6 * n_adjusted);
            dZ = Q.asDiagonal() * w_grad; // Still update structure even if pose solve fails
        } else {
            dX = solver.solve(y);
            if (logger) {
                float y_norm = y.norm();
                float dX_norm_before_check = dX.norm();
                float S_diag_min = S_damped.diagonal().minCoeff();
                float S_diag_max = S_damped.diagonal().maxCoeff();
                float S_cond = S_diag_max / std::max(S_diag_min, 1e-10f);
                logger->info("[BA] Solver success - y.norm()={:.6f}, dX.norm()={:.6f}, S_diag_range=[{:.6f}, {:.6f}], S_cond={:.2e}, ep={:.2f}, lm={:.2e}",
                             y_norm, dX_norm_before_check, S_diag_min, S_diag_max, S_cond, ep, lm);
                if (dX_norm_before_check < 0.1f && y_norm > 10.0f) {
                    logger->warn("[BA] WARNING - Large residual (y.norm()={:.2f}) but small update (dX.norm()={:.4f})! "
                                "This suggests damping might be too high or Hessian is poorly conditioned.",
                                y_norm, dX_norm_before_check);
                }
            }
            // Back-substitute structure increments: dZ = C^-1 * (w - E^T * dX)
            // Python: dZ = Q * (w - block_matmul(E.permute(0, 2, 1, 4, 3), dX).squeeze(dim=-1))
            dZ = Q.asDiagonal() * (w_grad - E.transpose() * dX);
            
            // Save STEP 15-16: Solution dX and dZ
            if (save_intermediates) {
                save_ba_step15_16(dX, dZ, logger);
            }
        }
    }

    // Post-frame-1000 solve result summary
    if (logger && log_post1000) {
        float dX_norm = dX.norm();
        float dX_max = dX.cwiseAbs().maxCoeff();
        float dZ_norm = dZ.norm();
        float dZ_max = dZ.cwiseAbs().maxCoeff();
        float dZ_mean = dZ.mean();
        logger->info("[BA][F{}] solve: dX_norm={:.6f}, dX_max={:.6f}, dZ_norm={:.6f}, dZ_max={:.6f}, dZ_mean={:.6f}",
                     m_counter, dX_norm, dX_max, dZ_norm, dZ_max, dZ_mean);
        // Log per-pose dX breakdown (translation vs rotation magnitudes)
        for (int idx = 0; idx < std::min(n_adjusted, 5); idx++) {
            Eigen::Matrix<float, 6, 1> dx_p = dX.segment<6>(6 * idx);
            float t_mag = std::sqrt(dx_p(0)*dx_p(0) + dx_p(1)*dx_p(1) + dx_p(2)*dx_p(2));
            float r_mag = std::sqrt(dx_p(3)*dx_p(3) + dx_p(4)*dx_p(4) + dx_p(5)*dx_p(5));
            logger->info("[BA][F{}] dX pose[{}] (global={}): t_mag={:.6f}, r_mag={:.6f}, dt=({:.6f},{:.6f},{:.6f}), dr=({:.6f},{:.6f},{:.6f})",
                         m_counter, idx, fixedp + idx, t_mag, r_mag,
                         dx_p(0), dx_p(1), dx_p(2), dx_p(3), dx_p(4), dx_p(5));
        }
    }
    
    // ---------------------------------------------------------
    // Step 11: Apply updates
    // ---------------------------------------------------------
    // Update poses: poses = pose_retr(poses, dX, indices)
    if (!structure_only && n_adjusted > 0) {
        if (logger) {
            float dX_norm = dX.norm();
            float dX_max = dX.cwiseAbs().maxCoeff();
            logger->info("[BA] dX stats - norm={:.6f}, max={:.6f}, size={}", 
                         dX_norm, dX_max, dX.size());
        }
        
        for (int idx = 0; idx < n_adjusted; idx++) {
            int pose_idx = fixedp + idx;
            if (pose_idx >= 0 && pose_idx < n) {
                Eigen::Matrix<float, 6, 1> dx_vec = dX.segment<6>(6 * idx);
                
                // Debug: Log dX values for first few poses
                if (logger && idx < 3) {
                    logger->info("[BA] Pose[{}] (idx={}) dX: t=({:.6f}, {:.6f}, {:.6f}), r=({:.6f}, {:.6f}, {:.6f})",
                                pose_idx, idx,
                                dx_vec(0), dx_vec(1), dx_vec(2),
                                dx_vec(3), dx_vec(4), dx_vec(5));
                }
                
                // Python BA directly passes dX to pose_retr without any validation or clamping
                // Python: poses = pose_retr(poses, dX, fixedp + torch.arange(n))
                // Python retr: Exp(a) * X (no negation, no clamping, no validation)
                // So we match Python exactly by passing dX directly to retr
                
                // CRITICAL FIX: Jacobians are [tx, ty, tz, rx, ry, rz] (translation first)
                // retr() expects [tx, ty, tz, rx, ry, rz] (translation first)
                // So NO REORDERING is needed - pass dx_vec directly!
                // Previous reordering was WRONG and caused incorrect pose updates
                
                // Apply update directly (matches Python: no validation, no clamping, no reverting)
                SE3 pose_before = m_pg.m_poses[pose_idx];
                m_pg.m_poses[pose_idx] = m_pg.m_poses[pose_idx].retr(dx_vec);
                
                // Debug: Log pose before and after for first few poses
                if (logger && idx < 3) {
                    Eigen::Vector3f t_before = pose_before.t;
                    Eigen::Quaternionf q_before = pose_before.q;
                    Eigen::Vector3f t_after = m_pg.m_poses[pose_idx].t;
                    Eigen::Quaternionf q_after = m_pg.m_poses[pose_idx].q;
                    logger->info("[BA] Pose[{}] before: t=({:.6f}, {:.6f}, {:.6f}), q=({:.6f}, {:.6f}, {:.6f}, {:.6f})",
                                pose_idx, t_before.x(), t_before.y(), t_before.z(),
                                q_before.x(), q_before.y(), q_before.z(), q_before.w());
                    logger->info("[BA] Pose[{}] after:  t=({:.6f}, {:.6f}, {:.6f}), q=({:.6f}, {:.6f}, {:.6f}, {:.6f})",
                                pose_idx, t_after.x(), t_after.y(), t_after.z(),
                                q_after.x(), q_after.y(), q_after.z(), q_after.w());
                }
            }
        }
    } else {
        if (logger) {
            logger->warn("[BA] Skipping pose updates - structure_only={}, n_adjusted={}", 
                         structure_only, n_adjusted);
        }
    }
    
    // Summary log for BA saving
    if (logger) {
        if (save_intermediates) {
            logger->info("[BA] ========================================");
            logger->info("[BA] ✅ BA intermediate files saved for frame {} (TARGET_FRAME={})", m_counter, TARGET_FRAME);
            logger->info("[BA] Check bin_file/ directory for ba_step*.bin files");
            logger->info("[BA] ========================================");
        } else {
            if (m_counter % 50 == 0) {  // Log every 50 frames to avoid spam
                logger->debug("[BA] Skipping BA intermediate file saving (m_counter={} != TARGET_FRAME={})", 
                             m_counter, TARGET_FRAME);
            }
        }
    }
    
    // Post-frame-1000: Log pose changes summary
    if (logger && log_post1000 && !structure_only && n_adjusted > 0) {
        float max_t_change = 0.0f;
        float max_r_change = 0.0f;
        int max_t_pose = -1, max_r_pose = -1;
        for (int idx = 0; idx < n_adjusted; idx++) {
            Eigen::Matrix<float, 6, 1> dx_p = dX.segment<6>(6 * idx);
            float t_change = std::sqrt(dx_p(0)*dx_p(0) + dx_p(1)*dx_p(1) + dx_p(2)*dx_p(2));
            float r_change = std::sqrt(dx_p(3)*dx_p(3) + dx_p(4)*dx_p(4) + dx_p(5)*dx_p(5));
            if (t_change > max_t_change) { max_t_change = t_change; max_t_pose = fixedp + idx; }
            if (r_change > max_r_change) { max_r_change = r_change; max_r_pose = fixedp + idx; }
        }
        // Log latest pose (last optimized pose) position
        int last_pose_idx = fixedp + n_adjusted - 1;
        if (last_pose_idx >= 0 && last_pose_idx < n) {
            Eigen::Vector3f t_last = m_pg.m_poses[last_pose_idx].t;
            Eigen::Quaternionf q_last = m_pg.m_poses[last_pose_idx].q;
            logger->info("[BA][F{}] pose update: max_t_change={:.6f} (pose {}), max_r_change={:.6f} (pose {}), "
                         "latest_pose[{}] t=({:.4f},{:.4f},{:.4f}), q=({:.4f},{:.4f},{:.4f},{:.4f})",
                         m_counter, max_t_change, max_t_pose, max_r_change, max_r_pose,
                         last_pose_idx, t_last.x(), t_last.y(), t_last.z(),
                         q_last.x(), q_last.y(), q_last.z(), q_last.w());
        }
    }
    
    // Update patches: patches = depth_retr(patches, dZ, kx)
    // Python: disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
    // disp_retr uses scatter_sum to add dZ to all pixels in the patch
    for (int idx = 0; idx < m; idx++) {
        int k = kx[idx];
        int frame_i = k / M;
        int patch_idx = k % M;
        
        if (frame_i < 0 || frame_i >= PatchGraph::N || patch_idx < 0 || patch_idx >= M) continue;
        
        float dZ_val = dZ[idx];
        
        // Update all pixels in the patch (Python scatter_sum adds to entire patch)
        // Clamp inverse depth to match Python exactly: [1e-3, 10.0]
        //   pd = 1e-3 means depth = 1000 (far but allowed by Python)
        //   pd = 10.0 means depth = 0.1 (very close)
        // Python: disps = disp_retr(disps, dZ, kx).clamp(min=1e-3, max=10.0)
        const float MIN_PD = 1e-3f;   // Minimum inverse depth (maximum depth = 1000) - matches Python
        const float MAX_PD = 10.0f;   // Maximum inverse depth (minimum depth = 0.1)
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                float& disp = m_pg.m_patches[frame_i][patch_idx][2][y][x];
                disp = std::max(MIN_PD, std::min(MAX_PD, disp + dZ_val));
            }
        }
    }
    
    // Post-frame-1000: Final one-line BA summary
    if (logger && log_post1000) {
        float mean_residual = valid_residuals > 0 ? residual_sum / valid_residuals : 0.0f;
        float dX_norm_final = dX.norm();
        float dZ_norm_final = dZ.norm();
        logger->info("[BA][F{}] DONE: edges={}, valid={}, n_adj={}, m={}, mean_r={:.4f}, dX={:.6f}, dZ={:.6f}, fixedp={}",
                     m_counter, num_active_filtered, valid_residuals, n_adjusted, m,
                     mean_residual, dX_norm_final, dZ_norm_final, fixedp);
    }
}

