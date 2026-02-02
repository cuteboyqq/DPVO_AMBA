#pragma once
#include "se.hpp"
#include <vector>

namespace pops {

/**
 * Transform patches from frame i -> j using SE3 poses and intrinsics.
 * Flattened array version (runtime M and P)
 *
 * @param poses        SE3 poses [num_frames]
 * @param patches_flat Flattened patches array: ((i*M + k)*3 + c)*P*P + y*P + x
 * @param intrinsics_flat Flattened intrinsics: [frame*4 + fx,fy,cx,cy]
 * @param ii           Source frame indices [num_edges]
 * @param jj           Target frame indices [num_edges]
 * @param kk           Patch indices [num_edges]
 * @param num_edges    Number of edges
 * @param M            Patches per frame
 * @param P            Patch size
 * @param coords_out   Output coordinates [num_edges][2][P][P] flattened
 */
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
    float* coords_out
);

/**
 * Transform with Jacobians for Bundle Adjustment.
 * Computes projected coordinates and Jacobians w.r.t. poses and inverse depth.
 *
 * @param poses        SE3 poses [num_frames]
 * @param patches_flat Flattened patches array: ((i*M + k)*3 + c)*P*P + y*P + x
 * @param intrinsics_flat Flattened intrinsics: [frame*4 + fx,fy,cx,cy]
 * @param ii           Source frame indices [num_edges]
 * @param jj           Target frame indices [num_edges]
 * @param kk           Patch indices [num_edges]
 * @param num_edges    Number of edges
 * @param M            Patches per frame
 * @param P            Patch size
 * @param coords_out   Output coordinates [num_edges][2][P][P] flattened
 * @param Ji_out       Jacobian w.r.t. pose i [num_edges][2][P][P][6] flattened
 * @param Jj_out       Jacobian w.r.t. pose j [num_edges][2][P][P][6] flattened
 * @param Jz_out       Jacobian w.r.t. inverse depth [num_edges][2][P][P][1] flattened
 * @param valid_out    Validity mask [num_edges][P][P] flattened
 */
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
    float* coords_out,
    float* Ji_out,
    float* Jj_out,
    float* Jz_out,
    float* valid_out,
    int frame_num = -1,  // Optional: frame number for saving intermediate values
    bool save_intermediates = false  // Optional: whether to save Ti, Tj, Gij, Jacobians
);

/**
 * Compute flow magnitude between frames for motion estimation.
 * 
 * @param poses        SE3 poses [num_frames]
 * @param patches_flat Flattened patches array
 * @param intrinsics_flat Flattened intrinsics
 * @param ii           Source frame indices [num_edges]
 * @param jj           Target frame indices [num_edges]
 * @param kk           Patch indices [num_edges]
 * @param num_edges    Number of edges
 * @param M            Patches per frame
 * @param P            Patch size
 * @param beta         Weighting factor (default 0.5)
 * @param flow_out     Output flow magnitudes [num_edges] (per edge, mean over patch)
 * @param valid_out    Output validity mask [num_edges] (optional, can be nullptr)
 */
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
    float* valid_out = nullptr
);

} // namespace pops
