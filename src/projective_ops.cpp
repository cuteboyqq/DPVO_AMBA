#include "projective_ops.hpp"
#include <algorithm>

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

        int i = ii[e]; // source frame
        int j = jj[e]; // target frame
        int k = kk[e]; // patch index

        const SE3& Ti = poses[i];
        const SE3& Tj = poses[j];
        SE3 Gij = Tj * Ti.inverse();

        const float* intr_i = &intrinsics_flat[i * 4]; // fx,fy,cx,cy
        const float* intr_j = &intrinsics_flat[j * 4];

        for (int c = 0; c < 3; c++) {} // placeholder if needed later

        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {

                int idx = y * P + x;

                // Compute flat index for patches: ((i*M + k)*3 + c)*P*P + y*P + x
                float px = patches_flat[((i * M + k) * 3 + 0) * P * P + idx];
                float py = patches_flat[((i * M + k) * 3 + 1) * P * P + idx];
                float pd = patches_flat[((i * M + k) * 3 + 2) * P * P + idx];

                // Inverse projection
                float X0 = (px - intr_i[2]) / intr_i[0];
                float Y0 = (py - intr_i[3]) / intr_i[1];
                float Z0 = 1.0f; // homogeneous

                // Transform point
                Eigen::Vector3f p0(X0, Y0, Z0);
                Eigen::Vector3f p1 = Gij.R() * p0 + Gij.t;

                // Project
                float z = std::max(p1.z(), 0.1f);
                float d = 1.0f / z;

                float u = intr_j[0] * (d * p1.x()) + intr_j[2];
                float v = intr_j[1] * (d * p1.y()) + intr_j[3];

                // Store coordinates
                int base = e * 2 * P * P;
                coords_out[base + 0 * P * P + idx] = u;
                coords_out[base + 1 * P * P + idx] = v;
            }
        }
    }
}


} // namespace pops
