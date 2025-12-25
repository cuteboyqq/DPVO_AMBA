#ifndef SKELETON_POSTPROC_HPP
#define SKELETON_POSTPROC_HPP

#include <vector>
#include <string>
#include <utility>  // for std::pair
#include <cstring>
#include <deque>
#include <iostream>
#include "bounding_box.hpp"
using namespace std;

using Keypoint = std::pair<int, int>;   // could extend later to (x,y,conf)
using Keypoints = std::vector<Keypoint>;
constexpr int POSE_DETECT_NUM = 100;

class SKELETON_POSTPROC {
public:
    enum POSE {
        FALLDOWN = 0,
        BENDING,
        SITTING,
        RUNNING,
        WALKING,
        STANDING
    };

    enum BodyPart {
        NOSE = 0,
        LEFT_EYE,
        RIGHT_EYE,
        LEFT_EAR,
        RIGHT_EAR,
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE
    };

    SKELETON_POSTPROC();

    // Static utility methods
    static float distance(const Keypoint& p1, const Keypoint& p2);
    static float angle(const Keypoint& a, const Keypoint& b, const Keypoint& c);

    bool isSitting(const Keypoints& kpts);
    bool isSitting(const BoundingBox& bbox);
    bool isBending(const Keypoints& kpts);
    bool isFalling(const Keypoints& kpts);
    bool isWalking(const Keypoints& kpts);
    bool isWalking(const BoundingBox& bbox, const BoundingBox& prev_bbox);
    bool isRunning(const Keypoints& kpts);
    // bool isWalking(const Keypoints& kpts, const Keypoints& pre_kpts);
    // bool isRunning(const Keypoints& kpts, const Keypoints& pre_kpts);

    std::string classifySkeleton(const BoundingBox& bbox,const BoundingBox& prev_bbox);

    std::vector<BoundingBox> m_prevPoseBoundingboxList;
    void updatePrevBoundingboxList(const std::vector<BoundingBox> BoundingboxList);
private:
    std::string poseToString(SKELETON_POSTPROC::POSE p);
    int   m_sittingThreshold = 15;
    float m_fallingThreshold = 45.0f;
    int   m_walkingThreshold = 1;
    float m_runningThreshold = 20;
    float m_BendingThreshold = 30.0f;

};
#endif // SKELETON_POSTPROC_HPP
