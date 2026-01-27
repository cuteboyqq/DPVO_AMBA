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
// CRITICAL: Input order matches lietorch convention: [tau, phi] = [translation, rotation]
SE3 SE3::retr(const Eigen::Matrix<float,6,1>& dx) const {
    Eigen::Vector3f dt = dx.head<3>();  // Translation (tau) - first 3 elements
    Eigen::Vector3f dR = dx.tail<3>();  // Rotation (phi) - last 3 elements
    
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
    // This matches lietorch's left_jacobian formula exactly
    Eigen::Matrix3f V;
    if (theta < 1e-6f) {
        // Small angle: use Taylor expansion matching lietorch exactly
        // lietorch: coef1 = 1/2 - theta²/24, coef2 = 1/6 - theta²/120
        // V = I + coef1 * Phi + coef2 * Phi²
        float theta2 = theta * theta;
        float coef1 = 0.5f - (1.0f / 24.0f) * theta2;
        float coef2 = (1.0f / 6.0f) - (1.0f / 120.0f) * theta2;
        
        V = Eigen::Matrix3f::Identity()
            + coef1 * dR_skew
            + coef2 * dR_skew * dR_skew;
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
// Logarithm map: SE3 → se3 (6-vector [tau, phi])
// Python: Log() returns [Vinv * translation, phi] where phi = so3.Log()
// -----------------------------
Eigen::Matrix<float,6,1> SE3::log() const {
    // Extract rotation part: quaternion → axis-angle (phi)
    // Python lietorch uses atan-based formula: phi = (2 * atan(n/w) / n) * q.vec()
    // where n = ||q.vec()|| and w = q.w()
    Eigen::Vector3f phi;
    float qw = q.w();
    Eigen::Vector3f qvec = q.vec();  // [qx, qy, qz]
    float squared_n = qvec.squaredNorm();
    float n = std::sqrt(squared_n);
    
    float two_atan_nbyw_by_n;
    const float EPS = 1e-6f;
    
    if (squared_n < EPS * EPS) {
        // Small rotation: q.vec() ≈ 0, so w ≈ 1
        float squared_w = qw * qw;
        two_atan_nbyw_by_n = 2.0f / qw - (2.0f / 3.0f) * squared_n / (qw * squared_w);
    } else {
        if (std::abs(qw) < EPS) {
            // w ≈ 0: rotation by π
            if (qw > 0.0f) {
                two_atan_nbyw_by_n = M_PI / n;
            } else {
                two_atan_nbyw_by_n = -M_PI / n;
            }
        } else {
            // General case: phi = (2 * atan(n/w) / n) * q.vec()
            // Python uses atan(n/w), not atan2(n, w)
            two_atan_nbyw_by_n = 2.0f * std::atan(n / qw) / n;
        }
    }
    
    phi = two_atan_nbyw_by_n * qvec;
    
    // Compute left jacobian inverse: Vinv
    // Python lietorch formula: Vinv = I - 0.5 * Phi + coef2 * Phi^2
    // where coef2 = (1 - theta * cos(half_theta) / (2 * sin(half_theta))) / (theta^2)
    // or coef2 = 1/12 for small angles
    float theta_norm = phi.norm();
    Eigen::Matrix3f phi_skew = skew(phi);
    Eigen::Matrix3f phi_skew2 = phi_skew * phi_skew;
    Eigen::Matrix3f Vinv;
    
    if (theta_norm < 1e-6f) {
        // Small angle: Vinv = I - 0.5 * Phi + (1/12) * Phi^2
        Vinv = Eigen::Matrix3f::Identity() - 0.5f * phi_skew + (1.0f / 12.0f) * phi_skew2;
    } else {
        float half_theta = 0.5f * theta_norm;
        float sin_half_theta = std::sin(half_theta);
        float cos_half_theta = std::cos(half_theta);
        
        // coef2 = (1 - theta * cos(half_theta) / (2 * sin(half_theta))) / (theta^2)
        float theta2 = theta_norm * theta_norm;
        float coef2 = (1.0f - theta_norm * cos_half_theta / (2.0f * sin_half_theta)) / theta2;
        
        Vinv = Eigen::Matrix3f::Identity() - 0.5f * phi_skew + coef2 * phi_skew2;
    }
    
    // Compute tau = Vinv * translation
    Eigen::Vector3f tau = Vinv * t;
    
    // Return [tau, phi] = [translation, rotation]
    Eigen::Matrix<float,6,1> result;
    result.head<3>() = tau;
    result.tail<3>() = phi;
    return result;
}

// -----------------------------
// Exponential map: se3 (6-vector [tau, phi]) → SE3
// Python: Exp([tau, phi]) returns SE3(SO3.Exp(phi), left_jacobian(phi) * tau)
// -----------------------------
SE3 SE3::Exp(const Eigen::Matrix<float,6,1>& tau_phi) {
    Eigen::Vector3f tau = tau_phi.head<3>();  // Translation part
    Eigen::Vector3f phi = tau_phi.tail<3>();  // Rotation part
    
    // Compute SO3 exponential: Exp(phi) → quaternion
    float theta = phi.norm();
    Eigen::Quaternionf so3_q;
    
    if (theta < 1e-6f) {
        // Small angle: use Taylor expansion
        float half_theta = theta / 2.0f;
        so3_q.w() = 1.0f - 0.5f * half_theta * half_theta;
        so3_q.vec() = 0.5f * phi;
    } else {
        float half_theta = theta / 2.0f;
        float sin_half_theta = std::sin(half_theta);
        so3_q.w() = std::cos(half_theta);
        so3_q.vec() = (sin_half_theta / theta) * phi;
    }
    so3_q.normalize();
    
    // Compute left jacobian: V
    // V = I + (1-cos(θ))/θ² * [phi]_× + (θ - sin(θ))/θ³ * [phi]_×²
    Eigen::Matrix3f phi_skew = skew(phi);
    Eigen::Matrix3f V;
    
    if (theta < 1e-6f) {
        // Small angle: use Taylor expansion matching retr()
        float theta2 = theta * theta;
        float coef1 = 0.5f - (1.0f / 24.0f) * theta2;
        float coef2 = (1.0f / 6.0f) - (1.0f / 120.0f) * theta2;
        V = Eigen::Matrix3f::Identity() + coef1 * phi_skew + coef2 * phi_skew * phi_skew;
    } else {
        float sin_theta = std::sin(theta);
        float cos_theta = std::cos(theta);
        float one_minus_cos_over_theta2 = (1.0f - cos_theta) / (theta * theta);
        float theta_minus_sin_over_theta3 = (theta - sin_theta) / (theta * theta * theta);
        
        V = Eigen::Matrix3f::Identity()
          + one_minus_cos_over_theta2 * phi_skew
          + theta_minus_sin_over_theta3 * phi_skew * phi_skew;
    }
    
    // Compute translation: t = V * tau
    Eigen::Vector3f t_exp = V * tau;
    
    // Return SE3(SO3.Exp(phi), V * tau)
    SE3 result;
    result.q = so3_q;
    result.t = t_exp;
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
// Python computes: Adj().transpose() where Adj() = [R, tx*R; 0, R]
// Adj().transpose() = [R^T, 0; (tx*R)^T, R^T] = [R^T, 0; -R^T*tx, R^T]
// 
// Previous C++ implementation built: [R^T, -R^T*tx; 0, R^T] (transpose of Python's)
// This caused Ji mismatch. Fix: compute Adj() first, then transpose to match Python.
// -----------------------------
Eigen::Matrix<float, 2, 6> SE3::adjointT(const Eigen::Matrix<float, 2, 6>& J) const {
    Eigen::Matrix3f R_mat = R();
    Eigen::Matrix3f t_skew = skew(t);
    Eigen::Matrix3f tx_R = t_skew * R_mat;  // tx*R where tx = hat(t)
    
    // Build Adj() matrix: [R, tx*R; 0, R] (matching Python lietorch)
    Eigen::Matrix<float, 6, 6> Adj;
    Adj.block<3,3>(0,0) = R_mat;
    Adj.block<3,3>(0,3) = tx_R;
    Adj.block<3,3>(3,0) = Eigen::Matrix3f::Zero();
    Adj.block<3,3>(3,3) = R_mat;
    
    // Compute Adj().transpose() to match Python: [R^T, 0; -R^T*tx, R^T]
    Eigen::Matrix<float, 6, 6> Adj_T = Adj.transpose();
    
    // Python's adjT computes: (AdjT @ J^T)^T instead of J @ AdjT
    // This applies the adjoint transpose operator to each row of J
    // (AdjT @ J^T)^T = J @ AdjT^T = J @ Adj (since AdjT = Adj^T)
    // But empirically, Python uses: (AdjT @ J^T)^T
    Eigen::Matrix<float, 6, 2> J_transpose = J.transpose();  // [6, 2]
    Eigen::Matrix<float, 6, 2> result = Adj_T * J_transpose;  // [6, 6] * [6, 2] = [6, 2]
    return result.transpose();  // [2, 6]
}
