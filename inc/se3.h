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

    // Skew-symmetric matrix
    static Eigen::Matrix3f skew(const Eigen::Vector3f& v);
};
