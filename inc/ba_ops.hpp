#pragma once
#include "eigen_common.h"
#include "se.hpp"
#include <vector>

namespace ba_ops {

// Scatter-add for block matrices (6x6 blocks)
void scatter_add_block_matrix(
    const Eigen::Matrix<float, 6, 6>* blocks,
    const int* indices_i,
    const int* indices_j,
    int num_blocks,
    int n,
    Eigen::MatrixXf& out
);

// Scatter-add for vectors
void scatter_add_vector(
    const Eigen::VectorXf& vec,
    const int* indices,
    int num_elements,
    int n,
    Eigen::VectorXf& out
);

// Block matrix multiplication: A * B where A is [n, m, 6, 1] and B is [m, n, 1, 6]
Eigen::MatrixXf block_matmul(
    const Eigen::MatrixXf& A,  // [n, m, 6, 1] reshaped to [6n, m]
    const Eigen::MatrixXf& B   // [m, n, 1, 6] reshaped to [m, 6n]
);

// Block solve for symmetric block-sparse system
Eigen::VectorXf block_solve(
    const Eigen::MatrixXf& S,  // Block-sparse matrix [6n, 6n]
    const Eigen::VectorXf& y,  // RHS [6n]
    float ep = 100.0f,
    float lm = 1e-4f
);

// Depth retraction: clamp and update inverse depth
void depth_retr(
    float* disps,
    const float* dZ,
    const int* kx,
    int num_patches,
    float min_disp = 1e-3f,
    float max_disp = 10.0f
);

// Pose retraction: apply SE3 update
void pose_retr(
    SE3* poses,
    const Eigen::VectorXf& dX,  // [6n] flattened
    const int* indices,
    int num_poses
);

} // namespace ba_ops

