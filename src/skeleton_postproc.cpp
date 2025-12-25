#include "skeleton_postproc.hpp"
#include <cmath>
#include <algorithm>

#include "skeleton_postproc.hpp"

SKELETON_POSTPROC::SKELETON_POSTPROC() {
    m_prevPoseBoundingboxList.reserve(POSE_DETECT_NUM);
}

float SKELETON_POSTPROC::distance(const Keypoint& p1, const Keypoint& p2) {
    return std::sqrt((p1.first - p2.first) * (p1.first - p2.first) +
                     (p1.second - p2.second) * (p1.second - p2.second));
}

float SKELETON_POSTPROC::angle(const Keypoint& a, const Keypoint& b, const Keypoint& c) {
    // Angle at point b
    float abx = a.first - b.first;
    float aby = a.second - b.second;
    float cbx = c.first - b.first;
    float cby = c.second - b.second;

    float dot = abx * cbx + aby * cby;
    float normA = std::sqrt(abx * abx + aby * aby);
    float normC = std::sqrt(cbx * cbx + cby * cby);

    if (normA == 0 || normC == 0) return 0.0f;
    float cos_angle = dot / (normA * normC);
    cos_angle = std::max(-1.0f, std::min(1.0f, cos_angle)); // clamp
    return std::acos(cos_angle) * 180.0f / M_PI; // degrees
}

bool SKELETON_POSTPROC::isSitting(const Keypoints& kpts) {
    if (kpts.size() < 17) return false;

    float hipY  = (kpts[LEFT_HIP].second + kpts[RIGHT_HIP].second) / 2.0f;
    float kneeY = (kpts[LEFT_KNEE].second + kpts[RIGHT_KNEE].second) / 2.0f;
    std::cout<<"hipY = "<<hipY<<" kneeY = "<<kneeY<<endl;

    return (std::abs(kneeY - hipY) <= m_sittingThreshold); // threshold
}

bool SKELETON_POSTPROC::isSitting(const BoundingBox &bbox)
{   
    Keypoints kpts              = bbox.pose_kpts;
    int       bbox_height = abs(bbox.y2 - bbox.y1 );
    int       bbox_width  = abs(bbox.x2 - bbox.x1 );
    if (bbox.pose_kpts.size() < 17) return false;

    float hipY  = (kpts[LEFT_HIP].second + kpts[RIGHT_HIP].second) / 2.0f;
    float kneeY = (kpts[LEFT_KNEE].second + kpts[RIGHT_KNEE].second) / 2.0f;

    float hipX  = (kpts[LEFT_HIP].first + kpts[RIGHT_HIP].first) / 2.0f;
    float kneeX = (kpts[LEFT_KNEE].first + kpts[RIGHT_KNEE].first) / 2.0f;

    int m_sittingThresholdY = int(bbox_height / 8.0);
    int m_sittingThresholdX = int(bbox_width * 0.40);
    int m_sittingThreshold2X = int(bbox_width / 3.0);


    float ankleY = (kpts[LEFT_ANKLE].second + kpts[RIGHT_ANKLE].second) / 2.0f;
    // float leftKneeAngle  = angle(kpts[LEFT_HIP] , kpts[LEFT_KNEE] , kpts[LEFT_ANKLE] );
    // float rightKneeAngle = angle(kpts[RIGHT_HIP], kpts[RIGHT_KNEE], kpts[RIGHT_ANKLE]);

    // float avgKneeangle = (leftKneeAngle + rightKneeAngle) / 2.0f;

    float dx_leg_l = kpts[LEFT_ANKLE].first - kpts[LEFT_HIP].first;
    float dy_leg_l = kpts[LEFT_ANKLE].second - kpts[LEFT_HIP].second;

    if (dy_leg_l == 0) dy_leg_l = 1e-5f; // avoid division by zero
    float angle_leg_l = std::atan2(dx_leg_l,dy_leg_l) * 180.0f / M_PI;

    return ((std::abs(kneeY - hipY) <= m_sittingThresholdY 
            || std::abs(kneeX - hipX) >= m_sittingThresholdX) 
            && std::abs(ankleY - kneeY)>= m_sittingThresholdY)
            || (angle_leg_l > 30.0 && std::abs(kneeY - hipY) <= m_sittingThresholdY);

}

bool SKELETON_POSTPROC::isFalling(const Keypoints& kpts) {
    if (kpts.size() < 17) return false;

    // Compute torso vector (left shoulder -> left hip)
    //-------------------Left side--------------------------------------------
    float dx_body_l = kpts[LEFT_HIP].first - kpts[LEFT_SHOULDER].first;
    float dy_body_l = kpts[LEFT_HIP].second - kpts[LEFT_SHOULDER].second;

    if (dy_body_l == 0) dy_body_l = 1e-5f; // avoid division by zero
    float angle_body_l = std::atan2(dx_body_l, dy_body_l) * 180.0f / M_PI; // degrees relative to vertical


    float dx_leg_l = kpts[LEFT_ANKLE].first - kpts[LEFT_HIP].first;
    float dy_leg_l = kpts[LEFT_ANKLE].second - kpts[LEFT_HIP].second;

    if (dy_leg_l == 0) dy_leg_l = 1e-5f; // avoid division by zero
    float angle_leg_l = std::atan2(dx_leg_l,dy_leg_l) * 180.0f / M_PI;

    bool left_falldown = (std::abs(angle_body_l) > m_fallingThreshold) && (std::abs(angle_leg_l) > m_fallingThreshold);

    //-------------------Right side--------------------------------------------
    float dx_body_r = kpts[RIGHT_HIP].first - kpts[RIGHT_SHOULDER].first;
    float dy_body_r = kpts[RIGHT_HIP].second - kpts[RIGHT_SHOULDER].second;

    if (dy_body_r == 0) dy_body_r = 1e-5f; // avoid division by zero
    float angle_body_r = std::atan2(dx_body_r, dy_body_r) * 180.0f / M_PI; // degrees relative to vertical


    float dx_leg_r = kpts[RIGHT_ANKLE].first - kpts[RIGHT_HIP].first;
    float dy_leg_r = kpts[RIGHT_ANKLE].second - kpts[RIGHT_HIP].second;

    if (dy_leg_r == 0) dy_leg_r = 1e-5f; // avoid division by zero
    float angle_leg_r = std::atan2(dx_leg_r,dy_leg_r) * 180.0f / M_PI;

    bool right_falldown = (std::abs(angle_body_r) > m_fallingThreshold) && (std::abs(angle_leg_r) > m_fallingThreshold);

    return left_falldown || right_falldown;
}

bool SKELETON_POSTPROC::isBending(const Keypoints& kpts) {
    if (kpts.size() < 17) return false;

    // Compute torso vector (left shoulder -> left hip)
    float dx = kpts[LEFT_HIP].first - kpts[LEFT_SHOULDER].first;
    float dy = kpts[LEFT_HIP].second - kpts[LEFT_SHOULDER].second;

    if (dy == 0) dy = 1e-5f; // avoid division by zero
    float angle_body = std::atan2(dx, dy) * 180.0f / M_PI; // degrees relative to vertical

    return (std::abs(angle_body) > m_BendingThreshold); // adjust threshold
}

bool SKELETON_POSTPROC::isWalking(const Keypoints& kpts) {
    if (kpts.size() < 17) return false;

    float ankleDist = distance(kpts[LEFT_ANKLE], kpts[RIGHT_ANKLE]);
    return (ankleDist > m_walkingThreshold); // threshold
}

bool SKELETON_POSTPROC::isWalking(const BoundingBox& bbox, const BoundingBox& prev_bbox) {
    if (bbox.pose_kpts.size() < 17 || prev_bbox.pose_kpts.size() < 17 ) return false;

    float ankleDist      = distance(bbox.pose_kpts[LEFT_ANKLE],      bbox.pose_kpts[RIGHT_ANKLE]);
    float ankleDist_prev = distance(prev_bbox.pose_kpts[LEFT_ANKLE], prev_bbox.pose_kpts[RIGHT_ANKLE]);

    bool bbox_move = abs(bbox.x2 - prev_bbox.x2) > 5 || abs(bbox.y2 - prev_bbox.y2) > 5;

    return ( abs(ankleDist - ankleDist_prev) > m_walkingThreshold  ) || bbox_move; // threshold
}

bool SKELETON_POSTPROC::isRunning(const Keypoints& kpts) {
    if (kpts.size() < 17) return false;

    // Left knee angle: (hip, knee, ankle)
    float leftKneeAngle  = angle(kpts[LEFT_HIP], kpts[LEFT_KNEE], kpts[LEFT_ANKLE]);
    float rightKneeAngle = angle(kpts[RIGHT_HIP], kpts[RIGHT_KNEE], kpts[RIGHT_ANKLE]);

    float avgKneeAngle = (leftKneeAngle + rightKneeAngle) / 2.0f;

    // Heuristic threshold: smaller angle -> bent knee -> running
    // Tune threshold as needed, e.g., < 140 degrees
    return avgKneeAngle < m_runningThreshold;
}

std::string SKELETON_POSTPROC::classifySkeleton(const BoundingBox& bbox,const BoundingBox& prev_bbox) {
    Keypoints kpts = bbox.pose_kpts;
    if (isFalling(kpts)) return poseToString(FALLDOWN);
    if (isBending(kpts)) return poseToString(BENDING);
    if (isSitting(bbox)) return poseToString(SITTING);
    if (isRunning(kpts)) return poseToString(RUNNING);  // check running before walking
    if (isWalking(bbox, prev_bbox)) return poseToString(WALKING);

    return poseToString(STANDING); // default
}

void SKELETON_POSTPROC::updatePrevBoundingboxList(const std::vector<BoundingBox> BoundingboxList)
{
    if(m_prevPoseBoundingboxList.size()>0){m_prevPoseBoundingboxList.clear();}
    m_prevPoseBoundingboxList = BoundingboxList;
}


std::string SKELETON_POSTPROC::poseToString(SKELETON_POSTPROC::POSE p) {
    switch (p) {
        case SKELETON_POSTPROC::FALLDOWN: return "fall_down";
        case SKELETON_POSTPROC::BENDING:  return "bending";
        case SKELETON_POSTPROC::SITTING:  return "sitting";
        case SKELETON_POSTPROC::RUNNING:  return "running";
        case SKELETON_POSTPROC::WALKING:  return "walking";
        case SKELETON_POSTPROC::STANDING: return "standing";
        default: return "unknown";
    }
}

// int main() {
//     SKELETON_POSTPROC postproc;

//     // Fake example skeleton (x,y)
//     SKELETON_POSTPROC::Keypoints kpts(17, {0,0});
//     kpts[11] = {100, 200}; // left hip
//     kpts[12] = {150, 200}; // right hip
//     kpts[13] = {100, 250}; // left knee
//     kpts[14] = {150, 250}; // right knee
//     kpts[15] = {100, 300}; // left ankle
//     kpts[16] = {180, 300}; // right ankle

//     std::string label = postproc.classifySkeleton(kpts);
//     std::cout << "Posture: " << label << std::endl;
//     return 0;
// }