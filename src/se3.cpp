#include "se3.h"

// -----------------------------
// Constructors
// -----------------------------
SE3::SE3() {
    t.setZero();
    q.setIdentity();
}

SE3::SE3(const Eigen::Matrix3f& R, const Eigen::Vector3f& trans) {
    q = Eigen::Quaternionf(R);
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
SE3 SE3::operator*(const SE3& other) const {
    SE3 out;
    out.q = q * other.q;
    out.t = t + q * other.R() * other.t;
    return out;
}

// -----------------------------
// Retract
// -----------------------------
SE3 SE3::retr(const Eigen::Matrix<float,6,1>& dx) const {
    Eigen::Vector3f dR = dx.head<3>();
    Eigen::Vector3f dt = dx.tail<3>();

    // Small-angle approx: R_new ≈ R * (I + [dR]_x)
    Eigen::Matrix3f R_new = R() * (Eigen::Matrix3f::Identity() + skew(dR));
    Eigen::Vector3f t_new = t + dt;

    return SE3(R_new, t_new);
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
