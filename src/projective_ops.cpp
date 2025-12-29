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

                // Inverse projection: X0 = [X, Y, Z, W] in homogeneous coordinates
                float X0 = (px - intr_i[2]) / intr_i[0];
                float Y0 = (py - intr_i[3]) / intr_i[1];
                float Z0 = 1.0f;
                float W0 = pd; // inverse depth

                // Transform point: X1 = Gij * X0
                // In homogeneous coordinates: p1 = R * [X,Y,Z]^T + t * W
                Eigen::Vector3f p0(X0, Y0, Z0);
                Eigen::Vector3f p1 = Gij.R() * p0 + Gij.t * W0;

                // Project: use Z component directly (not Z/W since W is inverse depth)
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


// ------------------------------------------------------------
// transformWithJacobians(): transform with Jacobian computation
// ------------------------------------------------------------
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
    float* Ji_out,      // [num_edges, 2, P, P, 6] flattened
    float* Jj_out,      // [num_edges, 2, P, P, 6] flattened
    float* Jz_out,      // [num_edges, 2, P, P, 1] flattened
    float* valid_out)   // [num_edges, P, P] flattened
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
        
        float fx_j = intr_j[0];
        float fy_j = intr_j[1];
        float cx_j = intr_j[2];
        float cy_j = intr_j[3];

        // Get patch center coordinates
        int center_idx = (P / 2) * P + (P / 2);
        float px = patches_flat[((i * M + k) * 3 + 0) * P * P + center_idx];
        float py = patches_flat[((i * M + k) * 3 + 1) * P * P + center_idx];
        float pd = patches_flat[((i * M + k) * 3 + 2) * P * P + center_idx];

        // Inverse projection at center
        float X0 = (px - intr_i[2]) / intr_i[0];
        float Y0 = (py - intr_i[3]) / intr_i[1];
        float Z0 = 1.0f;
        float W0 = pd; // inverse depth

        // Transform point: X1 = Gij * X0
        // In homogeneous coordinates: p1 = R * [X,Y,Z]^T + t * W
        Eigen::Vector3f p0(X0, Y0, Z0);
        Eigen::Vector3f p1 = Gij.R() * p0 + Gij.t * W0;
        
        // For Jacobian computation, we need the transformed point
        float X1 = p1.x();
        float Y1 = p1.y();
        float Z1 = p1.z();
        float H1 = W0; // homogeneous coordinate (inverse depth)

        // Project
        float z = std::max(Z1, 0.1f);
        float d = 1.0f / z;

        float u = fx_j * (d * X1) + cx_j;
        float v = fy_j * (d * Y1) + cy_j;

        // Store coordinates for all patch pixels
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int idx = y * P + x;
                
                // Get patch pixel coordinates
                float px_pix = patches_flat[((i * M + k) * 3 + 0) * P * P + idx];
                float py_pix = patches_flat[((i * M + k) * 3 + 1) * P * P + idx];
                float pd_pix = patches_flat[((i * M + k) * 3 + 2) * P * P + idx];

                // Inverse projection
                float X0_pix = (px_pix - intr_i[2]) / intr_i[0];
                float Y0_pix = (py_pix - intr_i[3]) / intr_i[1];
                float Z0_pix = 1.0f;
                float W0_pix = pd_pix;

                // Transform point
                Eigen::Vector3f p0_pix(X0_pix, Y0_pix, Z0_pix);
                Eigen::Vector3f p1_pix = Gij.R() * p0_pix + Gij.t * W0_pix;

                float X1_pix = p1_pix.x();
                float Y1_pix = p1_pix.y();
                float Z1_pix = p1_pix.z();
                float H1_pix = W0_pix;

                // Project
                float z_pix = std::max(Z1_pix, 0.1f);
                float d_pix = 1.0f / z_pix;

                float u_pix = fx_j * (d_pix * X1_pix) + cx_j;
                float v_pix = fy_j * (d_pix * Y1_pix) + cy_j;

                // Store coordinates
                int base = e * 2 * P * P;
                coords_out[base + 0 * P * P + idx] = u_pix;
                coords_out[base + 1 * P * P + idx] = v_pix;
                
                // Validity (Z > 0.2)
                valid_out[e * P * P + idx] = (Z1_pix > 0.2f) ? 1.0f : 0.0f;
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

} // namespace pops
