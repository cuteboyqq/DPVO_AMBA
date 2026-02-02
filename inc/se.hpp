#pragma once
#include "eigen_common.h"

struct SE3 {
    Eigen::Vector3f t;        // translation
    Eigen::Quaternionf q;     // rotation

    SE3();
    SE3(const Eigen::Matrix3f& R, const Eigen::Vector3f& trans);

    // Getter for rotation matrix
    Eigen::Matrix3f R() const;

    // 4x4 homogeneous transformation matrix
    Eigen::Matrix4f matrix() const;

    // Inverse transformation
    SE3 inverse() const;

    // SE3 composition
    SE3 operator*(const SE3& other) const;

    // Retract: apply small update dx ∈ R^6
    SE3 retr(const Eigen::Matrix<float,6,1>& dx) const;

    // Logarithm map: SE3 → se3 (6-vector [tau, phi])
    Eigen::Matrix<float,6,1> log() const;

    // Exponential map: se3 (6-vector [tau, phi]) → SE3
    static SE3 Exp(const Eigen::Matrix<float,6,1>& tau_phi);

    // Skew-symmetric matrix
    static Eigen::Matrix3f skew(const Eigen::Vector3f& v);
    
    // Adjoint action: Ad_g(v) where g is this SE3 and v is a 6-vector
    // Returns: Ad_g * v (6x6 matrix applied to 6-vector)
    Eigen::Matrix<float, 6, 1> adjoint(const Eigen::Matrix<float, 6, 1>& v) const;
    
    // Adjoint transpose: Ad_g^T(J) where g is this SE3 and J is a [2,6] matrix
    // Returns: J * Ad_g^T (2x6 matrix)
    Eigen::Matrix<float, 2, 6> adjointT(const Eigen::Matrix<float, 2, 6>& J) const;

private:
    // SO3 helper functions
    static Eigen::Vector3f so3_log(const Eigen::Quaternionf& q);
    static Eigen::Quaternionf so3_exp(const Eigen::Vector3f& phi);
    static Eigen::Matrix3f so3_hat(const Eigen::Vector3f& phi);
    static Eigen::Matrix3f so3_left_jacobian(const Eigen::Vector3f& phi);
    static Eigen::Matrix3f so3_left_jacobian_inverse(const Eigen::Vector3f& phi);
    
    // SE3 adjoint matrix computation
    Eigen::Matrix<float, 6, 6> adjoint_matrix() const;
};

