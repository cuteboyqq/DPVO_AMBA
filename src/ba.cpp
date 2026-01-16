#include "ba.hpp"
#include "dpvo.hpp"
#include "projective_ops.hpp"
#include "eigen_common.h"
#include "eigen/Eigen/Dense"
#include "eigen/Eigen/Cholesky"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>
#include <set>

// =================================================================================================
// Bundle Adjustment Implementation
// Translated from Python BA function, adapted for C++ without PyTorch
// =================================================================================================
void DPVO::bundleAdjustment(float lmbda, float ep, bool structure_only, int fixedp)
{
    const int num_active = m_pg.m_num_edges;
    if (num_active == 0) return;

    const int M = m_cfg.PATCHES_PER_FRAME;
    const int P = m_P;
    const int b = 1; // batch size = 1

    // ---------------------------------------------------------
    // Basic setup: find number of pose variables
    // ---------------------------------------------------------
    int n = 0;
    for (int e = 0; e < num_active; e++) {
        n = std::max(n, std::max(m_pg.m_ii[e], m_pg.m_jj[e]) + 1);
    }

    // ---------------------------------------------------------
    // Forward projection (coordinates) + Jacobians
    // ---------------------------------------------------------
    std::vector<float> coords(num_active * 2 * P * P); // [num_active, 2, P, P]
    std::vector<float> Ji(num_active * 2 * P * P * 6); // [num_active, 2, P, P, 6]
    std::vector<float> Jj(num_active * 2 * P * P * 6); // [num_active, 2, P, P, 6]
    std::vector<float> Jz(num_active * 2 * P * P * 1); // [num_active, 2, P, P, 1]
    std::vector<float> valid(num_active * P * P); // [num_active, P, P]
    
    reproject(
        m_pg.m_ii, m_pg.m_jj, m_pg.m_kk, 
        num_active, 
        coords.data(),
        Ji.data(),  // Jacobian w.r.t. pose i
        Jj.data(),  // Jacobian w.r.t. pose j
        Jz.data(),  // Jacobian w.r.t. inverse depth
        valid.data() // validity mask
    );

    // ---------------------------------------------------------
    // Compute residual at patch center
    // ---------------------------------------------------------
    const int p = P;
    const int center_idx = (p / 2) * P + (p / 2); // center pixel index in patch
    std::vector<float> r(num_active * 2); // [num_active, 2]
    std::vector<float> v(num_active, 1.0f); // validity mask

    auto logger = spdlog::get("dpvo");
    float residual_sum = 0.0f;
    int valid_residuals = 0;

    int nan_residual_count = 0;
    for (int e = 0; e < num_active; e++) {
        // Extract coordinates at patch center
        float cx = coords[e * 2 * P * P + 0 * P * P + center_idx];
        float cy = coords[e * 2 * P * P + 1 * P * P + center_idx];
        
        // Validate inputs before computing residual
        float target_x = m_pg.m_target[e * 2 + 0];
        float target_y = m_pg.m_target[e * 2 + 1];
        
        if (!std::isfinite(cx) || !std::isfinite(cy) || 
            !std::isfinite(target_x) || !std::isfinite(target_y)) {
            // Invalid input - set residual to zero and invalidate edge
            r[e * 2 + 0] = 0.0f;
            r[e * 2 + 1] = 0.0f;
            v[e] = 0.0f;
            nan_residual_count++;
            if (logger && nan_residual_count <= 5) {
                logger->warn("BA: Invalid residual[{}]: target=({}, {}), coords=({}, {})", 
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
                logger->warn("BA: NaN residual[{}]: target=({}, {}), coords=({}, {}), residual=({}, {})", 
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
        logger->warn("BA: {} out of {} edges have NaN/Inf residuals", nan_residual_count, num_active);
    }
    
    if (logger) {
        logger->info("BA: Residual stats - valid={}/{}, mean_residual={:.4f}", 
                     valid_residuals, num_active, 
                     valid_residuals > 0 ? residual_sum / valid_residuals : 0.0f);
        if (valid_residuals > 0 && valid_residuals < 5) {
            // Log first few residuals for debugging
            for (int e = 0; e < std::min(3, num_active); e++) {
                float r_norm = std::sqrt(r[e * 2 + 0] * r[e * 2 + 0] + r[e * 2 + 1] * r[e * 2 + 1]);
                logger->info("BA: Residual[{}]: target=({:.2f}, {:.2f}), coords=({:.2f}, {:.2f}), "
                             "residual=({:.4f}, {:.4f}), norm={:.4f}, valid={}",
                             e, m_pg.m_target[e * 2 + 0], m_pg.m_target[e * 2 + 1],
                             coords[e * 2 * P * P + 0 * P * P + center_idx],
                             coords[e * 2 * P * P + 1 * P * P + center_idx],
                             r[e * 2 + 0], r[e * 2 + 1], r_norm, v[e]);
            }
        }
    }
    
    // Continue with remaining validity checks (this was outside the loop, moved here)
    for (int e = 0; e < num_active; e++) {
        // Extract coordinates at patch center for bounds check
        float cx = coords[e * 2 * P * P + 0 * P * P + center_idx];
        float cy = coords[e * 2 * P * P + 1 * P * P + center_idx];
        
        // Reject projections outside image bounds
        // CRITICAL: Reprojected coordinates are at 1/4 resolution (feature map resolution)
        // Use feature map dimensions, not full image dimensions
        // bounds = (xmin, ymin, xmax, ymax) = (0, 0, m_fmap1_W, m_fmap1_H)
        if (cx < 0.0f || cy < 0.0f || cx >= m_fmap1_W || cy >= m_fmap1_H) {
            v[e] = 0.0f;
        }
        
        // Also use validity from transformWithJacobians
        float valid_center = valid[e * P * P + center_idx];
        if (valid_center < 0.5f) {
            v[e] = 0.0f;
        }
    }

    // Apply validity mask to residuals and weights
    std::vector<float> weights_masked(num_active);
    for (int e = 0; e < num_active; e++) {
        r[e * 2 + 0] *= v[e];
        r[e * 2 + 1] *= v[e];
        weights_masked[e] = m_pg.m_weight[e] * v[e];
    }
    
    // Extract Jacobians at patch center: [num_active, 2, 6] for Ji, Jj, [num_active, 2, 1] for Jz
    std::vector<float> Ji_center(num_active * 2 * 6); // [num_active, 2, 6]
    std::vector<float> Jj_center(num_active * 2 * 6); // [num_active, 2, 6]
    std::vector<float> Jz_center(num_active * 2 * 1); // [num_active, 2, 1]
    
    for (int e = 0; e < num_active; e++) {
        // Extract Jacobians at patch center
        // Ji: [num_active, 2, P, P, 6] -> [num_active, 2, 6] at center
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
    // Reshape r to [num_active, 2, 1] for matrix operations
    // wJiT = (weights * Ji).transpose(2, 3) -> [num_active, 6, 2]
    // wJjT = (weights * Jj).transpose(2, 3) -> [num_active, 6, 2]
    // wJzT = (weights * Jz).transpose(2, 3) -> [num_active, 1, 2]
    
    std::vector<Eigen::Matrix<float, 6, 2>> wJiT(num_active);
    std::vector<Eigen::Matrix<float, 6, 2>> wJjT(num_active);
    std::vector<Eigen::Matrix<float, 1, 2>> wJzT(num_active);
    
    for (int e = 0; e < num_active; e++) {
        float w = weights_masked[e];
        if (w < 1e-6f) {
            wJiT[e].setZero();
            wJjT[e].setZero();
            wJzT[e].setZero();
            continue;
        }
        
        // Ji_center: [num_active, 2, 6] -> transpose to [6, 2]
        // Jj_center: [num_active, 2, 6] -> transpose to [6, 2]
        // Jz_center: [num_active, 2, 1] -> transpose to [1, 2]
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 2; j++) {
                wJiT[e](i, j) = w * Ji_center[e * 2 * 6 + j * 6 + i];
                wJjT[e](i, j) = w * Jj_center[e * 2 * 6 + j * 6 + i];
            }
        }
        for (int j = 0; j < 2; j++) {
            wJzT[e](0, j) = w * Jz_center[e * 2 * 1 + j * 1];
        }
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
    std::vector<Eigen::Matrix<float, 6, 6>> Bii(num_active);
    std::vector<Eigen::Matrix<float, 6, 6>> Bij(num_active);
    std::vector<Eigen::Matrix<float, 6, 6>> Bji(num_active);
    std::vector<Eigen::Matrix<float, 6, 6>> Bjj(num_active);
    std::vector<Eigen::Matrix<float, 6, 1>> Eik(num_active);
    std::vector<Eigen::Matrix<float, 6, 1>> Ejk(num_active);
    
    for (int e = 0; e < num_active; e++) {
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

    // ---------------------------------------------------------
    // Step 4: Compute gradients
    // ---------------------------------------------------------
    // vi = wJiT @ r [6, 1]
    // vj = wJjT @ r [6, 1]
    // w = wJzT @ r [1, 1]
    
    std::vector<Eigen::Matrix<float, 6, 1>> vi(num_active);
    std::vector<Eigen::Matrix<float, 6, 1>> vj(num_active);
    std::vector<float> w_vec(num_active);
    
    for (int e = 0; e < num_active; e++) {
        Eigen::Matrix<float, 2, 1> r_vec;
        r_vec(0, 0) = r[e * 2 + 0];
        r_vec(1, 0) = r[e * 2 + 1];
        
        vi[e] = wJiT[e] * r_vec;
        vj[e] = wJjT[e] * r_vec;
        w_vec[e] = (wJzT[e] * r_vec)(0, 0);
    }

    // ---------------------------------------------------------
    // Step 5: Fix first pose (gauge freedom)
    // ---------------------------------------------------------
    std::vector<int> ii_new(num_active);
    std::vector<int> jj_new(num_active);
    
    for (int e = 0; e < num_active; e++) {
        ii_new[e] = m_pg.m_ii[e] - fixedp;
        jj_new[e] = m_pg.m_jj[e] - fixedp;
    }
    
    int n_adjusted = n - fixedp; // number of pose variables after fixing
    
    if (logger) {
        logger->info("BA: num_active={}, n={}, fixedp={}, n_adjusted={}", 
                     num_active, n, fixedp, n_adjusted);
    }

    // ---------------------------------------------------------
    // Step 6: Reindex structure variables
    // ---------------------------------------------------------
    std::vector<int> kk_new(num_active);
    std::vector<int> kx; // unique structure indices
    std::map<int, int> kk_to_idx; // mapping from original kk to unique index
    
    // Extract unique kk values
    std::set<int> kk_set;
    for (int e = 0; e < num_active; e++) {
        kk_set.insert(m_pg.m_kk[e]);
    }
    
    kx.assign(kk_set.begin(), kk_set.end());
    std::sort(kx.begin(), kx.end());
    
    // Create mapping
    for (size_t i = 0; i < kx.size(); i++) {
        kk_to_idx[kx[i]] = static_cast<int>(i);
    }
    
    // Create new kk indices
    for (int e = 0; e < num_active; e++) {
        kk_new[e] = kk_to_idx[m_pg.m_kk[e]];
    }
    
    int m = static_cast<int>(kx.size()); // number of structure variables

    // ---------------------------------------------------------
    // Step 7: Scatter-add to assemble global Hessian B [n, n, 6, 6]
    // ---------------------------------------------------------
    // B is block-sparse: B[i, j] is a 6x6 block
    // We'll use a dense representation for now (can be optimized later)
    Eigen::MatrixXf B = Eigen::MatrixXf::Zero(6 * n_adjusted, 6 * n_adjusted);
    
    for (int e = 0; e < num_active; e++) {
        if (v[e] < 0.5f) continue;
        
        int i = ii_new[e];
        int j = jj_new[e];
        
        if (i < 0 || i >= n_adjusted || j < 0 || j >= n_adjusted) continue;
        
        // Scatter-add blocks
        B.block<6, 6>(6 * i, 6 * i) += Bii[e];
        B.block<6, 6>(6 * i, 6 * j) += Bij[e];
        B.block<6, 6>(6 * j, 6 * i) += Bji[e];
        B.block<6, 6>(6 * j, 6 * j) += Bjj[e];
    }

    // ---------------------------------------------------------
    // Step 8: Assemble pose-structure coupling E [n, m, 6, 1]
    // ---------------------------------------------------------
    // E is reshaped to [6n, m] for matrix operations
    Eigen::MatrixXf E = Eigen::MatrixXf::Zero(6 * n_adjusted, m);
    
    for (int e = 0; e < num_active; e++) {
        if (v[e] < 0.5f) continue;
        
        int i = ii_new[e];
        int j = jj_new[e];
        int k = kk_new[e];
        
        if (i < 0 || i >= n_adjusted || j < 0 || j >= n_adjusted || k < 0 || k >= m) continue;
        
        // Scatter-add Eik and Ejk
        E.block<6, 1>(6 * i, k) += Eik[e];
        E.block<6, 1>(6 * j, k) += Ejk[e];
    }

    // ---------------------------------------------------------
    // Step 9: Structure Hessian C [m] (diagonal)
    // ---------------------------------------------------------
    // C = sum over edges: wJzT @ Jz (scalar per edge)
    Eigen::VectorXf C = Eigen::VectorXf::Zero(m);
    
    for (int e = 0; e < num_active; e++) {
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

    // ---------------------------------------------------------
    // Step 10: Schur complement solve
    // ---------------------------------------------------------
    // Assemble gradient vectors
    Eigen::VectorXf v_grad = Eigen::VectorXf::Zero(6 * n_adjusted);
    Eigen::VectorXf w_grad = Eigen::VectorXf::Zero(m);
    
    for (int e = 0; e < num_active; e++) {
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
            logger->warn("BA: Gradient contains NaN/Inf values! Checking individual components...");
            // Log first few problematic gradients
            for (int e = 0; e < std::min(5, num_active); e++) {
                bool vi_has_nan = false, vj_has_nan = false;
                for (int idx = 0; idx < 6; idx++) {
                    if (!std::isfinite(vi[e](idx, 0))) vi_has_nan = true;
                    if (!std::isfinite(vj[e](idx, 0))) vj_has_nan = true;
                }
                if (vi_has_nan || vj_has_nan || !std::isfinite(w_vec[e])) {
                    logger->warn("BA: Edge[{}]: vi_has_nan={}, vj_has_nan={}, w_valid={}, "
                                "r=({:.4f}, {:.4f}), weight={:.4f}",
                                e, vi_has_nan, vj_has_nan, std::isfinite(w_vec[e]),
                                r[e * 2 + 0], r[e * 2 + 1], m_pg.m_weight[e]);
                }
            }
        }
        
        float v_grad_norm = v_grad.norm();
        float v_grad_max = v_grad.cwiseAbs().maxCoeff();
        float w_grad_norm = w_grad.norm();
        float w_grad_max = w_grad.cwiseAbs().maxCoeff();
        logger->info("BA: Gradient stats - v_grad_norm={:.6f}, v_grad_max={:.6f}, w_grad_norm={:.6f}, w_grad_max={:.6f}", 
                     v_grad_norm, v_grad_max, w_grad_norm, w_grad_max);
        
        // Check if gradients are zero
        if (v_grad_norm < 1e-6f && w_grad_norm < 1e-6f) {
            logger->warn("BA: WARNING - Both v_grad and w_grad are near zero! This means BA won't update poses.");
            logger->warn("BA: Possible causes: 1) Residuals are zero (poses optimal), 2) Weights are zero, 3) Jacobians are zero");
            // Log sample residuals and weights to diagnose
            int sample_count = std::min(5, num_active);
            for (int e = 0; e < sample_count; e++) {
                float r_norm = std::sqrt(r[e * 2 + 0] * r[e * 2 + 0] + r[e * 2 + 1] * r[e * 2 + 1]);
                logger->warn("BA: Sample edge[{}]: residual_norm={:.6f}, weight={:.6f}, valid={}", 
                            e, r_norm, m_pg.m_weight[e], v[e] > 0.5f ? 1 : 0);
            }
        }
    }
    
    // Levenberg-Marquardt damping
    Eigen::VectorXf C_lm = C.array() + lmbda;
    Eigen::VectorXf Q = 1.0f / C_lm.array(); // C^-1 (diagonal)
    
    // Schur complement: S = B - E * C^-1 * E^T
    Eigen::MatrixXf EQ = E * Q.asDiagonal(); // E * C^-1
    Eigen::MatrixXf S = B - EQ * E.transpose();
    
    // RHS: y = v - E * C^-1 * w
    Eigen::VectorXf y = v_grad - EQ * w_grad;
    
    if (logger) {
        float y_norm = y.norm();
        float EQ_w_norm = (EQ * w_grad).norm();
        logger->info("BA: RHS vector y stats - y.norm()={:.6f}, v_grad.norm()={:.6f}, (EQ*w_grad).norm()={:.6f}", 
                     y_norm, v_grad.norm(), EQ_w_norm);
        if (y_norm < 1e-6f) {
            logger->warn("BA: y vector is near zero - BA will not update poses! Check residuals and weights.");
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
        // where ep=100.0, lm=1e-4
        // This adds: (ep + lm * diag(S)) * I to S
        // For stability, we add ep * I + lm * diag(S) * I
        Eigen::VectorXf S_diag = S.diagonal();
        Eigen::MatrixXf S_damped = S;
        float lm = 1e-4f;
        for (int i = 0; i < 6 * n_adjusted; i++) {
            S_damped(i, i) += ep + lm * S_diag[i];
        }
        
        // Python uses Cholesky solver, but we'll use LDLT for stability
        // If Cholesky fails, it returns zeros
        Eigen::LDLT<Eigen::MatrixXf> solver(S_damped);
        if (solver.info() != Eigen::Success) {
            // Python: if cholesky fails, return zeros
            if (logger) {
                logger->warn("BA: LDLT solver failed with info={}", static_cast<int>(solver.info()));
            }
            dX = Eigen::VectorXf::Zero(6 * n_adjusted);
            dZ = Q.asDiagonal() * w_grad; // Still update structure even if pose solve fails
        } else {
            dX = solver.solve(y);
            if (logger) {
                float y_norm = y.norm();
                float dX_norm_before_check = dX.norm();
                logger->info("BA: Solver success - y.norm()={:.6f}, dX.norm()={:.6f}", y_norm, dX_norm_before_check);
            }
            // Back-substitute structure increments: dZ = C^-1 * (w - E^T * dX)
            // Python: dZ = Q * (w - block_matmul(E.permute(0, 2, 1, 4, 3), dX).squeeze(dim=-1))
            dZ = Q.asDiagonal() * (w_grad - E.transpose() * dX);
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
            logger->info("BA: dX stats - norm={:.6f}, max={:.6f}, size={}", 
                         dX_norm, dX_max, dX.size());
        }
        
        for (int idx = 0; idx < n_adjusted; idx++) {
            int pose_idx = fixedp + idx;
            if (pose_idx >= 0 && pose_idx < n) {
                Eigen::Matrix<float, 6, 1> dx_vec = dX.segment<6>(6 * idx);
                Eigen::Vector3f t_before = m_pg.m_poses[pose_idx].t;
                
                // Clamp pose update if it's too large (prevent divergence)
                float dx_norm = dx_vec.norm();
                if (dx_norm > 1.0f) {
                    if (logger && idx < 3) {
                        logger->warn("BA: Clamping large pose update for pose[{}]: dx_norm={:.6f} > 1.0, scaling down",
                                     pose_idx, dx_norm);
                    }
                    dx_vec = dx_vec * (1.0f / dx_norm);  // Normalize to unit length
                }
                
                m_pg.m_poses[pose_idx] = m_pg.m_poses[pose_idx].retr(dx_vec);
                Eigen::Vector3f t_after = m_pg.m_poses[pose_idx].t;
                
                // Validate updated pose: check if translation is reasonable
                float t_norm = t_after.norm();
                if (t_norm > 100.0f) {
                    if (logger) {
                        logger->warn("BA: Pose[{}] translation too large after update (norm={:.2f}), reverting update. "
                                     "t_before=({:.2f}, {:.2f}, {:.2f}), t_after=({:.2f}, {:.2f}, {:.2f})",
                                     pose_idx, t_norm,
                                     t_before.x(), t_before.y(), t_before.z(),
                                     t_after.x(), t_after.y(), t_after.z());
                    }
                    // Revert to previous pose if update caused divergence
                    m_pg.m_poses[pose_idx].t = t_before;
                }
                
                // Log pose updates for debugging
                if (logger) {
                    float t_change = (t_after - t_before).norm();
                    if (idx < 5 || dx_norm > 1e-6f || t_change > 1e-6f) {
                        logger->info("BA: Updated pose[{}]: dx_norm={:.6f}, t_change={:.6f}, "
                                     "t_before=({:.6f}, {:.6f}, {:.6f}), t_after=({:.6f}, {:.6f}, {:.6f}), "
                                     "dx_vec=({:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f})",
                                     pose_idx, dx_norm, t_change,
                                     t_before.x(), t_before.y(), t_before.z(),
                                     t_after.x(), t_after.y(), t_after.z(),
                                     dx_vec(0), dx_vec(1), dx_vec(2), dx_vec(3), dx_vec(4), dx_vec(5));
                    }
                }
            }
        }
    } else {
        if (logger) {
            logger->warn("BA: Skipping pose updates - structure_only={}, n_adjusted={}", 
                         structure_only, n_adjusted);
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
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                float& disp = m_pg.m_patches[frame_i][patch_idx][2][y][x];
                disp = std::max(1e-3f, std::min(10.0f, disp + dZ_val));
            }
        }
    }
}

