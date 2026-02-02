#include "se.hpp"
#include <cmath>

// Constants
static constexpr float EPS = 1e-6f;

// ============================================================================
// Constructors
// ============================================================================

SE3::SE3() : t(Eigen::Vector3f::Zero()), q(Eigen::Quaternionf::Identity()) {
}

SE3::SE3(const Eigen::Matrix3f& R, const Eigen::Vector3f& trans) 
    : t(trans), q(Eigen::Quaternionf(R)) {
    q.normalize();
}

// ============================================================================
// Basic Operations
// ============================================================================

Eigen::Matrix3f SE3::R() const {
    return q.toRotationMatrix();
}

Eigen::Matrix4f SE3::matrix() const {
    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    T.block<3, 3>(0, 0) = q.toRotationMatrix();
    T.block<3, 1>(0, 3) = t;
    return T;
}

SE3 SE3::inverse() const {
    Eigen::Quaternionf q_inv = q.conjugate();
    Eigen::Vector3f t_inv = -(q_inv * t);
    SE3 result;
    result.t = t_inv;
    result.q = q_inv;
    return result;
}

SE3 SE3::operator*(const SE3& other) const {
    SE3 result;
    result.q = q * other.q;
    result.q.normalize();
    result.t = t + q * other.t;
    return result;
}

// ============================================================================
// Retraction
// ============================================================================

SE3 SE3::retr(const Eigen::Matrix<float,6,1>& dx) const {
    // Retraction: Exp(dx) * this
    SE3 delta = Exp(dx);
    return delta * (*this);
}

// ============================================================================
// Logarithm Map: SE3 → se3
// ============================================================================

Eigen::Matrix<float,6,1> SE3::log() const {
    // Extract rotation part (phi)
    Eigen::Vector3f phi = so3_log(q);
    
    // Compute left Jacobian inverse for SO3
    Eigen::Matrix3f Vinv = so3_left_jacobian_inverse(phi);
    
    // Compute translation part (tau)
    Eigen::Vector3f tau = Vinv * t;
    
    // Return [tau, phi]
    Eigen::Matrix<float,6,1> result;
    result.head<3>() = tau;
    result.tail<3>() = phi;
    return result;
}

// ============================================================================
// Exponential Map: se3 → SE3
// ============================================================================

SE3 SE3::Exp(const Eigen::Matrix<float,6,1>& tau_phi) {
    // Split into translation and rotation parts
    Eigen::Vector3f tau = tau_phi.head<3>();
    Eigen::Vector3f phi = tau_phi.tail<3>();
    
    // Compute SO3 exponential
    Eigen::Quaternionf q_rot = so3_exp(phi);
    
    // Compute left Jacobian for SO3
    Eigen::Matrix3f V = so3_left_jacobian(phi);
    
    // Compute translation
    Eigen::Vector3f t_trans = V * tau;
    
    // Construct SE3
    SE3 result;
    result.q = q_rot;
    result.t = t_trans;
    return result;
}

// ============================================================================
// Skew-Symmetric Matrix
// ============================================================================

Eigen::Matrix3f SE3::skew(const Eigen::Vector3f& v) {
    Eigen::Matrix3f S;
    S << 0.0f, -v.z(),  v.y(),
         v.z(),  0.0f, -v.x(),
        -v.y(),  v.x(),  0.0f;
    return S;
}

// ============================================================================
// Adjoint Operations
// ============================================================================

Eigen::Matrix<float, 6, 1> SE3::adjoint(const Eigen::Matrix<float, 6, 1>& v) const {
    Eigen::Matrix<float, 6, 6> Ad = adjoint_matrix();
    return Ad * v;
}

Eigen::Matrix<float, 2, 6> SE3::adjointT(const Eigen::Matrix<float, 2, 6>& J) const {
    Eigen::Matrix<float, 6, 6> Ad = adjoint_matrix();
    return J * Ad.transpose();
}

Eigen::Matrix<float, 6, 6> SE3::adjoint_matrix() const {
    Eigen::Matrix3f R = q.toRotationMatrix();
    Eigen::Matrix3f t_hat = skew(t);
    Eigen::Matrix3f Zero = Eigen::Matrix3f::Zero();
    
    Eigen::Matrix<float, 6, 6> Ad;
    Ad.block<3, 3>(0, 0) = R;
    Ad.block<3, 3>(0, 3) = t_hat * R;
    Ad.block<3, 3>(3, 0) = Zero;
    Ad.block<3, 3>(3, 3) = R;
    
    return Ad;
}

// ============================================================================
// SO3 Helper Functions
// ============================================================================

Eigen::Vector3f SE3::so3_log(const Eigen::Quaternionf& q) {
    Eigen::Vector3f v = q.vec();
    float w = q.w();
    float n = v.norm();
    
    if (n < EPS) {
        return Eigen::Vector3f::Zero();
    }
    
    float theta = 2.0f * std::atan2(n, std::abs(w));
    if (w < 0) {
        theta = -theta;
    }
    
    return (theta / n) * v;
}

Eigen::Quaternionf SE3::so3_exp(const Eigen::Vector3f& phi) {
    float theta = phi.norm();
    
    if (theta < EPS) {
        return Eigen::Quaternionf(1.0f, 0.0f, 0.0f, 0.0f);
    }
    
    float half_theta = 0.5f * theta;
    float s = std::sin(half_theta) / theta;
    return Eigen::Quaternionf(std::cos(half_theta), s * phi.x(), s * phi.y(), s * phi.z());
}

Eigen::Matrix3f SE3::so3_hat(const Eigen::Vector3f& phi) {
    return skew(phi);
}

Eigen::Matrix3f SE3::so3_left_jacobian(const Eigen::Vector3f& phi) {
    float theta = phi.norm();
    
    if (theta < EPS) {
        return Eigen::Matrix3f::Identity();
    }
    
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f Phi = so3_hat(phi);
    Eigen::Matrix3f Phi2 = Phi * Phi;
    
    float theta2 = theta * theta;
    float coef1 = (1.0f - std::cos(theta)) / theta2;
    float coef2 = (theta - std::sin(theta)) / (theta2 * theta);
    
    return I + coef1 * Phi + coef2 * Phi2;
}

Eigen::Matrix3f SE3::so3_left_jacobian_inverse(const Eigen::Vector3f& phi) {
    float theta = phi.norm();
    
    if (theta < EPS) {
        return Eigen::Matrix3f::Identity();
    }
    
    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f Phi = so3_hat(phi);
    Eigen::Matrix3f Phi2 = Phi * Phi;
    
    float theta2 = theta * theta;
    float half_theta = 0.5f * theta;
    
    float coef2 = (theta < EPS) ? 
        (1.0f / 12.0f) : 
        (1.0f - theta * std::cos(half_theta) / (2.0f * std::sin(half_theta))) / (theta * theta);
    
    return I - 0.5f * Phi + coef2 * Phi2;
}

