#pragma once
#include "se3.h"

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

} // namespace pops
