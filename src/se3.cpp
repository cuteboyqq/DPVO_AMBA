#include "se3.h"
#include <cmath>

// -----------------------------
// Constructors
// -----------------------------
SE3::SE3() {
    t.setZero();
    q.setIdentity();
}

SE3::SE3(const Eigen::Matrix3f& R, const Eigen::Vector3f& trans) {
    q = Eigen::Quaternionf(R);
    q.normalize();  // Ensure quaternion is normalized (R might not be perfectly orthogonal due to numerical errors)
    t = trans;
}

// -----------------------------
// Rotation getter
// -----------------------------
Eigen::Matrix3f SE3::R() const {
    return q.toRotationMatrix();
}

// -----------------------------
// 4x4 homogeneous transformation
// -----------------------------
Eigen::Matrix4f SE3::matrix() const {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3,3>(0,0) = R();
    T.block<3,1>(0,3) = t;
    return T;
}

// -----------------------------
// Inverse
// -----------------------------
SE3 SE3::inverse() const {
    SE3 inv;
    inv.q = q.conjugate();
    inv.t = -(inv.R() * t);
    return inv;
}

// -----------------------------
// SE3 composition
// -----------------------------
// SE3 composition: T1 * T2 = (R1 * R2, R1 * t2 + t1)
// where T1 = (R1, t1) and T2 = (R2, t2)
SE3 SE3::operator*(const SE3& other) const {
    SE3 out;
    out.q = q * other.q;
    out.t = t + R() * other.t;  // FIXED: Use R() (rotation matrix) not q (quaternion)
    return out;
}

// -----------------------------
// Retract: retr(dx) = Exp(dx) * this
// -----------------------------
// Python: retr(self, a) = Exp(a) * X
// This implements the exponential map for SE3, matching lietorch
SE3 SE3::retr(const Eigen::Matrix<float,6,1>& dx) const {
    Eigen::Vector3f dR = dx.head<3>();  // Rotation in Lie algebra so(3)
    Eigen::Vector3f dt = dx.tail<3>();  // Translation update
    
    // Compute exponential map for rotation: Exp(dR) using Rodrigues' formula
    // This matches Python lietorch implementation (not small-angle approximation!)
    float theta = dR.norm();
    Eigen::Matrix3f dR_skew = skew(dR);
    
    // Precompute sin/cos once for both rotation and translation parts
    float sin_theta, cos_theta;
    if (theta < 1e-6f) {
        // Small angle: use Taylor expansion to avoid division by zero
        sin_theta = theta;  // sin(θ) ≈ θ for small θ
        cos_theta = 1.0f - 0.5f * theta * theta;  // cos(θ) ≈ 1 - θ²/2
    } else {
        sin_theta = std::sin(theta);
        cos_theta = std::cos(theta);
    }
    
    Eigen::Matrix3f dR_exp;  // Exp(dR) rotation matrix
    if (theta < 1e-6f) {
        // Small angle: use Taylor expansion to avoid division by zero
        // Exp(dR) ≈ I + [dR]_× + (1/2)[dR]_×²
        dR_exp = Eigen::Matrix3f::Identity() + dR_skew + 0.5f * dR_skew * dR_skew;
    } else {
        // Rodrigues' formula: Exp(dR) = I + sin(θ)/θ * [dR]_× + (1-cos(θ))/θ² * [dR]_×²
        float sin_theta_over_theta = sin_theta / theta;
        float one_minus_cos_over_theta2 = (1.0f - cos_theta) / (theta * theta);
        
        dR_exp = Eigen::Matrix3f::Identity() 
                 + sin_theta_over_theta * dR_skew
                 + one_minus_cos_over_theta2 * dR_skew * dR_skew;
    }
    
    // Compute exponential map for translation part
    // For SE3: Exp([dR, dt]) = [Exp(dR), V * dt]
    // where V = I + (1-cos(θ))/θ² * [dR]_× + (θ - sin(θ))/θ³ * [dR]_×²
    Eigen::Matrix3f V;
    if (theta < 1e-6f) {
        // Small angle: V ≈ I + (1/2)[dR]_×
        V = Eigen::Matrix3f::Identity() + 0.5f * dR_skew;
    } else {
        float one_minus_cos_over_theta2 = (1.0f - cos_theta) / (theta * theta);
        float theta_minus_sin_over_theta3 = (theta - sin_theta) / (theta * theta * theta);
        
        V = Eigen::Matrix3f::Identity()
            + one_minus_cos_over_theta2 * dR_skew
            + theta_minus_sin_over_theta3 * dR_skew * dR_skew;
    }
    
    // Compute delta SE3: delta = Exp(dx)
    Eigen::Matrix3f delta_R = dR_exp;
    Eigen::Vector3f delta_t = V * dt;
    
    // Compose: result = delta * this = Exp(dx) * this
    // This matches Python: retr(self, a) = Exp(a) * X
    SE3 delta(delta_R, delta_t);
    SE3 result = delta * (*this);
    
    // Ensure quaternion is normalized (numerical safety)
    result.q.normalize();
    return result;
}

// -----------------------------
// Skew-symmetric
// -----------------------------
Eigen::Matrix3f SE3::skew(const Eigen::Vector3f& v) {
    Eigen::Matrix3f S;
    S <<    0, -v.z(),  v.y(),
         v.z(),     0, -v.x(),
        -v.y(),  v.x(),     0;
    return S;
}

// -----------------------------
// Adjoint action: Ad_g(v) for SE3
// Ad_g = [R    0  ]
//        [t×R  R ]
// -----------------------------
Eigen::Matrix<float, 6, 1> SE3::adjoint(const Eigen::Matrix<float, 6, 1>& v) const {
    Eigen::Matrix3f R_mat = R();
    Eigen::Matrix3f t_cross_R = skew(t) * R_mat;
    
    Eigen::Vector3f v_rot = v.head<3>();
    Eigen::Vector3f v_trans = v.tail<3>();
    
    Eigen::Matrix<float, 6, 1> result;
    result.head<3>() = R_mat * v_rot;
    result.tail<3>() = t_cross_R * v_rot + R_mat * v_trans;
    
    return result;
}

// -----------------------------
// Adjoint transpose: Ad_g^T(J) for SE3
// Ad_g^T = [R^T    -R^T * [t]_×]
//          [0      R^T     ]
// -----------------------------
Eigen::Matrix<float, 2, 6> SE3::adjointT(const Eigen::Matrix<float, 2, 6>& J) const {
    Eigen::Matrix3f R_mat = R();
    Eigen::Matrix3f R_T = R_mat.transpose();
    Eigen::Matrix3f t_skew = skew(t);
    Eigen::Matrix3f neg_RT_t_skew = -R_T * t_skew;
    
    // Ad_g^T is [6, 6] matrix
    Eigen::Matrix<float, 6, 6> Ad_T;
    Ad_T.block<3,3>(0,0) = R_T;
    Ad_T.block<3,3>(0,3) = neg_RT_t_skew;
    Ad_T.block<3,3>(3,0) = Eigen::Matrix3f::Zero();
    Ad_T.block<3,3>(3,3) = R_T;
    
    // J * Ad_g^T: [2,6] * [6,6] = [2,6]
    return J * Ad_T;
}
