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
