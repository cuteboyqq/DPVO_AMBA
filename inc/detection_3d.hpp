/*
 * Shared 3D detection struct for YOLOv8 back-projection (used by DPVO and viewer).
 * Kept in a separate header so dpvo.hpp does not need to include the viewer (which pulls in X11/Eigen conflicts).
 */
#pragma once
#include "patch_graph.hpp"

struct Detection3D {
    Vec3 position;   // World position (e.g. feet on ground plane)
    int classId;     // 0 = person, 1 = car/vehicle, etc.
    float confidence;
};
