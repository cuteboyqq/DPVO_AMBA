/*
  (C) 2023-2024 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/

#include "dpvo_viewer.hpp"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>
#include <sys/stat.h>

#ifdef ENABLE_PANGOLIN_VIEWER
// Only include what we need — full opencv2/opencv.hpp pulls in stitching.hpp
// which has an `enum Status` that conflicts with X11's `Status` macro
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif
/*
Visual flow
---------------------------------------------------------------------------
dpvo.cpp: enableVisualization(true)
    ↓
dpvo.cpp: m_viewer = std::make_unique<DPVOViewer>(...)
    ↓
dpvo_viewer.cpp: Constructor executes
    ↓
dpvo_viewer.cpp: Line 61 - std::thread(&DPVOViewer::run, this)
    ↓
[NEW THREAD CREATED]
    ↓
dpvo_viewer.cpp: run() starts executing in background thread
    ↓
run() loops continuously, rendering frames
--------------------------------------------------------------------------
*/
DPVOViewer::DPVOViewer(int image_width, int image_height, int max_frames, int max_points)
    : m_imageWidth(image_width)
    , m_imageHeight(image_height)
    , m_maxFrames(max_frames)
    , m_maxPoints(max_points)
{
    // Pre-allocate buffers
    m_imageBuffer.resize(image_width * image_height * 3);
    m_poses.resize(max_frames);
    m_poseMatrices.resize(max_frames * 16);  // 4x4 matrices
    m_points.resize(max_points);
    m_colors.resize(max_points * 3);
    
    m_running = true;
    /*
    -------------------------------------------------------------------------------
    Important points
        Automatic: run() starts when the viewer object is created
        Separate thread: It runs in its own thread (non-blocking)
        Continuous: It loops until m_running = false or the window is closed
        Not called directly: dpvo.cpp never directly calls run()
    -------------------------------------------------------------------------------
    */
    m_viewerThread = std::thread(&DPVOViewer::run, this);
}

DPVOViewer::~DPVOViewer()
{
    close();
    if (m_viewerThread.joinable()) {
        m_viewerThread.join();
    }
}

void DPVOViewer::updateImage(const uint8_t* image_data, int width, int height)
{
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    if (width <= 0 || height <= 0 || image_data == nullptr) {
        return;  // Invalid input
    }
    
    bool size_changed = (width != m_imageWidth || height != m_imageHeight);
    
    if (size_changed) {
        m_imageWidth = width;
        m_imageHeight = height;
        m_imageBuffer.resize(width * height * 3);
        m_textureSizeChanged = true;  // Flag to recreate texture
    }
    
    // Copy image data (assuming RGB format)
    size_t image_size = static_cast<size_t>(width) * height * 3;
    if (m_imageBuffer.size() >= image_size) {
        std::memcpy(m_imageBuffer.data(), image_data, image_size);
        m_imageUpdated = true;
    }
}

void DPVOViewer::updatePoses(const SE3* poses, int num_frames)
{
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    int original_num_frames = num_frames;
    
    // CRITICAL: Use the actual number of frames passed in (m_pg.m_n, the sliding window size)
    // Don't use force_increment - it causes issues where m_numFrames > num_frames
    // and we try to copy more poses than are available
    // 
    // The sliding window (m_pg.m_n) typically stays at 8-10 frames, so we'll only see
    // the current optimization window, not all historical frames.
    // If you want to see all frames, you need to pass m_counter (total frames) instead
    // of m_pg.m_n from DPVO::updateViewer()
    
    // Use the actual number of frames passed in
    m_numFrames = num_frames;
    
    // CRITICAL: Dynamically resize buffers if num_frames exceeds current size
    // This allows the viewer to handle more frames than initially allocated
    if (m_numFrames > static_cast<int>(m_poses.size())) {
        // Resize all pose-related buffers
        m_poses.resize(m_numFrames);
        m_poseMatrices.resize(m_numFrames * 16);  // 4x4 matrices
        
        // Also resize saved pose vectors
        m_saved_cam_positions.resize(m_numFrames);
        m_saved_forwards.resize(m_numFrames);
        m_saved_rights.resize(m_numFrames);
        m_saved_ups.resize(m_numFrames);
    }
    
    // Copy poses and normalize quaternions to prevent corruption
    int poses_to_copy = std::min(m_numFrames, original_num_frames);
    for (int i = 0; i < poses_to_copy; i++) {
        m_poses[i] = poses[i];
        // Normalize quaternion to prevent corruption (should be ~1.0)
        float q_norm = m_poses[i].q.norm();
        if (std::abs(q_norm - 1.0f) > 0.01f) {
            m_poses[i].q.normalize();
        }
    }
    
    // Convert to matrices for OpenGL
    convertPosesToMatrices();
    
    // Track frame counter for frame saving (num_frames = total historical frames processed)
    m_frameCounter = num_frames;
    m_newDataReceived = true;
}

void DPVOViewer::updatePoints(const Vec3* points, const uint8_t* colors, int num_points)
{
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    num_points = std::min(num_points, m_maxPoints);
    
    // Filter out invalid points (zero coordinates or NaN/Inf)
    // CRITICAL: Increase distance threshold to allow points from all frames to be visible
    // Previous threshold of 1000.0f was too restrictive and filtered out points from later frames
    const float MAX_POINT_DISTANCE = 100000000.0f;  // Increased from 1000.0f to allow far points
    int valid_count = 0;
    int zero_count = 0;
    int nan_inf_count = 0;
    int out_of_bounds_count = 0;
    
    for (int i = 0; i < num_points; i++) {
        const Vec3& p = points[i];
        
        // Check individual validity conditions for debugging
        bool is_zero = (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f);
        bool is_finite = std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        bool in_bounds = (std::abs(p.x) < MAX_POINT_DISTANCE && 
                         std::abs(p.y) < MAX_POINT_DISTANCE && 
                         std::abs(p.z) < MAX_POINT_DISTANCE);
        
        // Check if point is valid (not all zeros, not NaN/Inf, within bounds)
        bool is_valid = !is_zero && is_finite && in_bounds;
        
        if (is_valid) {
            m_points[valid_count] = p;
            if (colors) {
                m_colors[valid_count * 3 + 0] = colors[i * 3 + 0];
                m_colors[valid_count * 3 + 1] = colors[i * 3 + 1];
                m_colors[valid_count * 3 + 2] = colors[i * 3 + 2];
            } else {
                m_colors[valid_count * 3 + 0] = 255;
                m_colors[valid_count * 3 + 1] = 255;
                m_colors[valid_count * 3 + 2] = 255;
            }
            valid_count++;
        } else {
            // Track why points were filtered out (for debugging)
            if (is_zero) zero_count++;
            if (!is_finite) nan_inf_count++;
            if (!in_bounds) out_of_bounds_count++;
        }
    }
    
    m_numPoints = valid_count;
    
    // DIAGNOSTIC: Log filtering statistics more frequently and with frame breakdown
    static int update_count = 0;
    update_count++;
    
    // Log every update for first 5 updates, then every 10 updates
    bool should_log = (update_count <= 5 || update_count % 10 == 0);
    
    if (should_log) {
        // Count points per frame to see distribution
        const int M = 4;  // PATCHES_PER_FRAME (hardcoded for now)
        int num_frames = num_points / M;
        std::vector<int> points_per_frame(num_frames, 0);
        std::vector<int> valid_per_frame(num_frames, 0);
        
        // Count points per frame from INPUT array (before filtering)
        for (int i = 0; i < num_points; i++) {
            int frame_idx = i / M;
            if (frame_idx < num_frames) {
                points_per_frame[frame_idx]++;
            }
        }
        
        // Count valid points per frame from FILTERED OUTPUT array (after filtering)
        // Note: valid_count is the number of points that passed filtering
        // But we need to track which frames they came from
        // Since filtering removes invalid points, we can't directly map filtered indices to frame indices
        // Instead, we'll count from the input array which points passed filtering
        std::vector<int> input_valid_per_frame(num_frames, 0);
        int filtered_idx = 0;
        for (int i = 0; i < num_points; i++) {
            const Vec3& p = points[i];
            bool is_zero = (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f);
            bool is_finite = std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
            bool in_bounds = (std::abs(p.x) < MAX_POINT_DISTANCE && 
                             std::abs(p.y) < MAX_POINT_DISTANCE && 
                             std::abs(p.z) < MAX_POINT_DISTANCE);
            bool is_valid = !is_zero && is_finite && in_bounds;
            
            if (is_valid) {
                int frame_idx = i / M;
                if (frame_idx < num_frames) {
                    input_valid_per_frame[frame_idx]++;
                }
                filtered_idx++;
            }
        }
        
        // Only log if there are issues (many filtered points)
        if (valid_count < num_points * 0.5f) {
            printf("[DPVOViewer] updatePoints: WARNING - Only %d/%d points are valid (filtered: zero=%d, nan/inf=%d, out_of_bounds=%d)\n",
                   valid_count, num_points, zero_count, nan_inf_count, out_of_bounds_count);
            fflush(stdout);
        }
    }
}

void DPVOViewer::convertPosesToMatrices()
{
#ifdef ENABLE_PANGOLIN_VIEWER
    // Convert SE3 poses to 4x4 OpenGL matrices (column-major)
    // SE3 poses are world-to-camera (T_wc), we need camera-to-world (T_cw) for visualization
    
    // CRITICAL: Ensure m_poseMatrices is large enough for m_numFrames
    if (m_numFrames * 16 > static_cast<int>(m_poseMatrices.size())) {
        m_poseMatrices.resize(m_numFrames * 16);
    }
    
    // Limit loop to available matrices to prevent out-of-bounds access
    int max_frames = std::min(m_numFrames, static_cast<int>(m_poseMatrices.size()) / 16);
    
    for (int i = 0; i < max_frames; i++) {
        // Get camera-to-world transformation: T_cw = T_wc^-1
        SE3 T_wc = m_poses[i];
        
        // Validate input pose first
        Eigen::Vector3f t_wc = T_wc.t;
        Eigen::Quaternionf q_wc = T_wc.q;
        
        // Check if quaternion is normalized (should be ~1.0)
        float q_norm = q_wc.norm();
        
        // Check if translation is reasonable
        // CRITICAL: Increase bounds to allow larger trajectories (e.g., 10000 instead of 1000)
        // This was causing valid poses to be skipped if the camera moved far from origin
        bool t_valid = (std::isfinite(t_wc.x()) && std::isfinite(t_wc.y()) && std::isfinite(t_wc.z()) &&
                       std::abs(t_wc.x()) < 100000.0f && std::abs(t_wc.y()) < 100000.0f && std::abs(t_wc.z()) < 100000.0f);
        
        // Always try to normalize quaternion if it's not already normalized
        // Even if the norm is huge (like 466 or 23801), normalizing should give a valid quaternion
        bool q_valid = false;
        if (q_norm > 1e-6f && q_norm < 1e6f) {  // Reasonable range for normalization
            if (std::abs(q_norm - 1.0f) > 0.01f) {
                q_wc.normalize();
                T_wc.q = q_wc;
                m_poses[i] = T_wc;  // Update stored pose
            }
            q_valid = true;  // Quaternion is now valid after normalization
        } else {
            // Quaternion is completely corrupted (zero or infinite), use identity rotation but keep translation
            if (t_valid) {
                // Keep translation but use identity rotation
                T_wc.q = Eigen::Quaternionf::Identity();
                T_wc.t = t_wc;
                m_poses[i] = T_wc;
                q_valid = true;  // Mark as valid after fixing
            }
        }
        
        // If both are invalid, use identity as fallback
        if (!q_valid || !t_valid) {
            // Use identity as fallback
            Eigen::Matrix4f T_cw = Eigen::Matrix4f::Identity();
            float* mat = &m_poseMatrices[i * 16];
            mat[0]  = T_cw(0,0); mat[4]  = T_cw(1,0); mat[8]  = T_cw(2,0); mat[12] = T_cw(3,0);
            mat[1]  = T_cw(0,1); mat[5]  = T_cw(1,1); mat[9]  = T_cw(2,1); mat[13] = T_cw(3,1);
            mat[2]  = T_cw(0,2); mat[6]  = T_cw(1,2); mat[10] = T_cw(2,2); mat[14] = T_cw(3,2);
            mat[3]  = T_cw(0,3); mat[7]  = T_cw(1,3); mat[11] = T_cw(2,3); mat[15] = T_cw(3,3);
            continue;
        }
        
        // Normalize quaternion if needed (should be ~1.0, but allow small deviation)
        if (std::abs(q_norm - 1.0f) > 0.01f) {
            q_wc.normalize();
            T_wc.q = q_wc;
            // Update the stored pose to prevent corruption from propagating
            m_poses[i] = T_wc;
        }
        
        // Compute inverse using the same method as DPViewer's invSE3
        // T_cw = T_wc^-1
        // Formula matches DPViewer's invSE3:
        //   q_cw = conjugate(q_wc)
        //   t_cw = -R_cw * t_wc, where R_cw is rotation matrix from q_cw
        SE3 T_cw_se3 = T_wc.inverse();
        Eigen::Vector3f t_cw = T_cw_se3.t;
        Eigen::Quaternionf q_cw = T_cw_se3.q;
        
        // Build rotation matrix from quaternion components (matching DPViewer's pose_to_matrix_kernel)
        // This matches the exact formula used in viewer_cuda.cu lines 244-257
        float qx = q_cw.x();
        float qy = q_cw.y();
        float qz = q_cw.z();
        float qw = q_cw.w();
        
        Eigen::Matrix3f R_cw;
        R_cw(0,0) = 1 - 2*qy*qy - 2*qz*qz;
        R_cw(0,1) = 2*qx*qy - 2*qw*qz;
        R_cw(0,2) = 2*qx*qz + 2*qw*qy;
        R_cw(1,0) = 2*qx*qy + 2*qw*qz;
        R_cw(1,1) = 1 - 2*qx*qx - 2*qz*qz;
        R_cw(1,2) = 2*qy*qz - 2*qw*qx;
        R_cw(2,0) = 2*qx*qz - 2*qw*qy;
        R_cw(2,1) = 2*qy*qz + 2*qw*qx;
        R_cw(2,2) = 1 - 2*qx*qx - 2*qy*qy;
        
        // Build T_cw matrix (matches DPViewer's pose_to_matrix_kernel output format)
        Eigen::Matrix4f T_cw = Eigen::Matrix4f::Identity();
        T_cw.block<3,3>(0,0) = R_cw;
        T_cw.block<3,1>(0,3) = t_cw;
        
        // Validate transformation matrix and translation
        bool is_valid = true;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                if (!std::isfinite(T_cw(r, c))) {
                    is_valid = false;
                    break;
                }
            }
            if (!is_valid) break;
        }
        
        // Also validate translation - check for reasonable values
        // CRITICAL: Use same bounds as t_wc validation (100000.0f) to avoid filtering valid poses
        // t_cw is camera position in world frame, which can be far from origin for long trajectories
        if (!std::isfinite(t_cw.x()) || !std::isfinite(t_cw.y()) || !std::isfinite(t_cw.z()) ||
            std::abs(t_cw.x()) > 100000.0f || std::abs(t_cw.y()) > 100000.0f || std::abs(t_cw.z()) > 100000.0f) {
            is_valid = false;
        }
        
        if (!is_valid) {
            // Use identity matrix as fallback for invalid poses
            T_cw = Eigen::Matrix4f::Identity();
            t_cw = Eigen::Vector3f::Zero();
        }
        
        // Store in column-major order for OpenGL (glMultMatrixf expects column-major)
        // Column-major: mat[col*4 + row] = T_cw(row, col)
        float* mat = &m_poseMatrices[i * 16];
        mat[0]  = T_cw(0,0); mat[4]  = T_cw(1,0); mat[8]  = T_cw(2,0); mat[12] = T_cw(3,0);
        mat[1]  = T_cw(0,1); mat[5]  = T_cw(1,1); mat[9]  = T_cw(2,1); mat[13] = T_cw(3,1);
        mat[2]  = T_cw(0,2); mat[6]  = T_cw(1,2); mat[10] = T_cw(2,2); mat[14] = T_cw(3,2);
        mat[3]  = T_cw(0,3); mat[7]  = T_cw(1,3); mat[11] = T_cw(2,3); mat[15] = T_cw(3,3);
        
        // Diagnostic logging removed to reduce verbosity
    }
#endif
}

void DPVOViewer::drawPoints()
{
#ifdef ENABLE_PANGOLIN_VIEWER
    if (m_numPoints == 0 || !m_vboInitialized) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    // Update vertex buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_numPoints * 3 * sizeof(float), 
                 m_points.data(), GL_DYNAMIC_DRAW);
    
    // Update color buffer
    glBindBuffer(GL_ARRAY_BUFFER, m_cbo);
    glBufferData(GL_ARRAY_BUFFER, m_numPoints * 3 * sizeof(uint8_t), 
                 m_colors.data(), GL_DYNAMIC_DRAW);
    
    // Draw points
    glBindBuffer(GL_ARRAY_BUFFER, m_cbo);
    glColorPointer(3, GL_UNSIGNED_BYTE, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    
    // Increase point size and enable point smoothing for better visibility
    glEnable(GL_POINT_SMOOTH);
    glPointSize(4.0f);  // Increased from 2.0f to 4.0f for better visibility
    glDrawArrays(GL_POINTS, 0, m_numPoints);
    glDisable(GL_POINT_SMOOTH);
    
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
#endif
}

void DPVOViewer::drawPoses()
{
#ifdef ENABLE_PANGOLIN_VIEWER
    if (m_numFrames == 0) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
            // CRITICAL: Ensure buffers are resized before converting matrices
            // This is a safety check in case convertPosesToMatrices is called before updatePoses
            if (m_numFrames * 16 > static_cast<int>(m_poseMatrices.size())) {
                m_poseMatrices.resize(m_numFrames * 16);
            }
            if (m_numFrames > static_cast<int>(m_poses.size())) {
                m_poses.resize(m_numFrames);
            }
    
    // Ensure matrices are up to date (in case poses changed)
    convertPosesToMatrices();
    
    // Determine how many frames to process
    // Draw ALL frames (no limit) - each frame should have one pose
    int frames_to_process = m_numFrames;
    if (frames_to_process > static_cast<int>(m_poses.size())) {
        frames_to_process = static_cast<int>(m_poses.size());
    }
    
    // Draw all frames starting from frame 0
    int start_frame = 0;
    
    int valid_poses_drawn = 0;
    int skipped_invalid = 0;
    int skipped_out_of_bounds = 0;
    
    // Draw all frames
    
    // Draw camera poses as square pyramids (frustums) to show position and orientation
    // This matches the approach used in drawPoses_fake() but uses real poses from matrices
    glLineWidth(2.0f);
    
    // Pyramid parameters
    const float pyramid_base_size = 0.03f;  // Size of pyramid base
    const float pyramid_height = 0.05f;      // Distance from tip to base
    
    // Get pointer to transformation matrices (column-major, 16 floats per matrix)
    float* tptr = m_poseMatrices.data();
    
    // Draw all frames (from 0 to frames_to_process-1)
    for (int i = 0; i < frames_to_process; i++) {
        // Validate matrix before using it
        float* mat = &tptr[4 * 4 * i];
        bool matrix_valid = true;
        for (int j = 0; j < 16; j++) {
            if (!std::isfinite(mat[j])) {
                matrix_valid = false;
                break;
            }
        }
        
        if (!matrix_valid) {
            skipped_invalid++;
            continue;
        }
        
        // Debug: Check if matrix is identity (all poses at origin)
        // CRITICAL: In column-major format, translation is in mat[3], mat[7], mat[11] (column 3, rows 0,1,2)
        // NOT in mat[12], mat[13], mat[14] (which are the bottom row: 0, 0, 1)
        float tx = mat[3];   // T_cw(0,3) = t_cw.x()
        float ty = mat[7];   // T_cw(1,3) = t_cw.y()
        float tz = mat[11];  // T_cw(2,3) = t_cw.z()
        
        // Check if this is an identity matrix (all cameras at origin)
        // If translation is zero and rotation is identity, skip drawing to avoid clutter
        bool is_identity = (std::abs(tx) < 1e-6f && std::abs(ty) < 1e-6f && std::abs(tz) < 1e-6f &&
                           std::abs(mat[0] - 1.0f) < 1e-6f && std::abs(mat[5] - 1.0f) < 1e-6f && 
                           std::abs(mat[10] - 1.0f) < 1e-6f && std::abs(mat[15] - 1.0f) < 1e-6f);
        
        // Skip drawing identity matrices (all cameras at origin) to avoid clutter
        if (is_identity && i > 0) {
            // Skip drawing, but don't count as invalid
            continue;
        }
        
        // Extract camera position from transformation matrix
        // Translation is in mat[3], mat[7], mat[11] (column 3, rows 0,1,2)
        float cam_x = mat[3];
        float cam_y = mat[7];
        float cam_z = mat[11];
        
        // Diagnostic logging removed to reduce verbosity
        
        // Extract rotation vectors from transformation matrix
        // Matrix is stored in column-major format: mat[col*4 + row]
        // Column 0 (X axis): mat[0], mat[4], mat[8]
        // Column 1 (Y axis): mat[1], mat[5], mat[9]
        // Column 2 (Z axis): mat[2], mat[6], mat[10]
        // Forward is -Z direction in camera frame (third column, negated)
        Eigen::Vector3f forward(-mat[2], -mat[6], -mat[10]);
        // Right is X direction (first column)
        Eigen::Vector3f right(mat[0], mat[4], mat[8]);
        // Up is Y direction (second column)
        Eigen::Vector3f up(mat[1], mat[5], mat[9]);
        
        // Normalize direction vectors
        forward.normalize();
        right.normalize();
        up.normalize();
        
        // Set color: current frame in red, others in blue
        if (i + 1 == m_numFrames) {
            glColor3f(1.0f, 0.0f, 0.0f);  // Red for current frame
        } else {
            glColor3f(0.0f, 0.0f, 1.0f);  // Blue for other frames
        }
        
        // Camera position (square base at camera position - represents camera view)
        Eigen::Vector3f square_center(cam_x, cam_y, cam_z);
        
        // Pyramid tip/point is at head (forward direction from camera)
        Eigen::Vector3f tip = square_center + forward * pyramid_height;
        
        // Square corners (square at camera position, represents camera view)
        Eigen::Vector3f base_corner1 = square_center + right * pyramid_base_size + up * pyramid_base_size;
        Eigen::Vector3f base_corner2 = square_center - right * pyramid_base_size + up * pyramid_base_size;
        Eigen::Vector3f base_corner3 = square_center - right * pyramid_base_size - up * pyramid_base_size;
        Eigen::Vector3f base_corner4 = square_center + right * pyramid_base_size - up * pyramid_base_size;
        
        // Set color for edges
        if (i + 1 == m_numFrames) {
            glColor3f(1.0f, 0.0f, 0.0f);  // Red for current frame
        } else {
            glColor3f(0.0f, 0.0f, 1.0f);  // Blue for other frames
        }
        
        // Draw pyramid edges (lines from square corners to tip)
        glBegin(GL_LINES);
        
        // Lines from square corners (camera position) to tip (forward direction)
        glVertex3f(base_corner1.x(), base_corner1.y(), base_corner1.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        glVertex3f(base_corner2.x(), base_corner2.y(), base_corner2.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        glVertex3f(base_corner3.x(), base_corner3.y(), base_corner3.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        glVertex3f(base_corner4.x(), base_corner4.y(), base_corner4.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        // Base square edges (outline of square at camera position)
        glVertex3f(base_corner1.x(), base_corner1.y(), base_corner1.z());
        glVertex3f(base_corner2.x(), base_corner2.y(), base_corner2.z());
        
        glVertex3f(base_corner2.x(), base_corner2.y(), base_corner2.z());
        glVertex3f(base_corner3.x(), base_corner3.y(), base_corner3.z());
        
        glVertex3f(base_corner3.x(), base_corner3.y(), base_corner3.z());
        glVertex3f(base_corner4.x(), base_corner4.y(), base_corner4.z());
        
        glVertex3f(base_corner4.x(), base_corner4.y(), base_corner4.z());
        glVertex3f(base_corner1.x(), base_corner1.y(), base_corner1.z());
        
        glEnd();
        
        valid_poses_drawn++;
    }
#else
    // Viewer disabled - no-op
#endif
}

void DPVOViewer::drawPoses_fake()
{
#ifdef ENABLE_PANGOLIN_VIEWER
    if (m_numFrames == 0) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    // Ensure matrices are up to date (in case poses changed)
    convertPosesToMatrices();
    
    // Debug counter (reduced logging frequency)
    static int draw_count = 0;
    draw_count++;
    
    // Draw camera poses as square pyramids (frustums) to show position and orientation
    // No need for point smoothing when drawing lines
    
    // Store valid poses as we process them (for fake pose generation)
    // Use member variables to persist across draw calls - this ensures previous frames keep their saved positions
    
    // Resize if number of frames increased (new frames added)
    if (m_numFrames > static_cast<int>(m_saved_cam_positions.size())) {
        m_saved_cam_positions.resize(m_numFrames);
        m_saved_forwards.resize(m_numFrames);
        m_saved_rights.resize(m_numFrames);
        m_saved_ups.resize(m_numFrames);
    }
    
    // Use saved poses for reference - start with saved poses, then build up as we process
    std::vector<Eigen::Vector3f> valid_cam_positions;
    std::vector<Eigen::Vector3f> valid_forwards;
    std::vector<Eigen::Vector3f> valid_rights;
    std::vector<Eigen::Vector3f> valid_ups;
    
    // Initialize with saved poses (if any)
    valid_cam_positions = m_saved_cam_positions;
    valid_forwards = m_saved_forwards;
    valid_rights = m_saved_rights;
    valid_ups = m_saved_ups;
    
    int valid_poses_drawn = 0;
    int skipped_invalid_matrix = 0;
    int skipped_out_of_bounds = 0;
    
    // Reduced logging frequency
    
    // CRITICAL: Ensure we process ALL frames up to m_numFrames
    // The loop should iterate from 0 to m_numFrames-1 (inclusive)
    // m_numFrames=9 means frames 0-8 (9 frames), m_numFrames=10 means frames 0-9 (10 frames)
    // Don't limit to m_maxFrames - process all frames that were passed in
    int frames_to_process = m_numFrames;
    
    // Only limit if it would exceed the pose buffer size (safety check)
        if (frames_to_process > static_cast<int>(m_poses.size())) {
        frames_to_process = static_cast<int>(m_poses.size());
        if (draw_count % 300 == 0) {  // Reduced frequency
            printf("[DPVOViewer] drawPoses: WARNING - m_numFrames=%d > m_poses.size()=%zu, limiting to %d\n",
                   m_numFrames, m_poses.size(), frames_to_process);
            fflush(stdout);
        }
    }
    
        // Ensure saved vectors are large enough for all frames we'll process
        if (frames_to_process > static_cast<int>(m_saved_cam_positions.size())) {
            m_saved_cam_positions.resize(frames_to_process);
            m_saved_forwards.resize(frames_to_process);
            m_saved_rights.resize(frames_to_process);
            m_saved_ups.resize(frames_to_process);
        }
    
    for (int i = 0; i < frames_to_process; i++) {
        // Validate matrix before using it
        float* mat = &m_poseMatrices[i * 16];
        bool matrix_valid = true;
        int invalid_element = -1;
        for (int j = 0; j < 16; j++) {
            if (!std::isfinite(mat[j])) {
                matrix_valid = false;
                invalid_element = j;
                break;
            }
        }
        
        // Extract camera position and orientation
        float cam_x = mat[3];
        float cam_y = mat[7];
        float cam_z = mat[11];
        
        // Extract rotation vectors
        // Extract rotation vectors from transformation matrix (column-major format)
        // Column 0 (X axis): mat[0], mat[4], mat[8]
        // Column 1 (Y axis): mat[1], mat[5], mat[9]
        // Column 2 (Z axis): mat[2], mat[6], mat[10]
        Eigen::Vector3f forward = Eigen::Vector3f(-mat[2], -mat[6], -mat[10]);  // -Z direction in camera frame
        Eigen::Vector3f right = Eigen::Vector3f(mat[0], mat[4], mat[8]);         // X direction
        Eigen::Vector3f up = Eigen::Vector3f(mat[1], mat[5], mat[9]);            // Y direction
        
        // FAKE VALUES FOR TESTING: Always use values close to previous frame to create smooth trajectory
        // This ensures consecutive frames are close together, forming a line/snake pattern
        bool use_fake = false;
        
        // Check if current frame is invalid OR if we want to force fake values for testing
        // CRITICAL: Use same bounds as t_wc validation (100000.0f) to avoid filtering valid poses
        bool is_invalid = (!matrix_valid || 
            !std::isfinite(cam_x) || !std::isfinite(cam_y) || !std::isfinite(cam_z) ||
            std::abs(cam_x) > 100000.0f || std::abs(cam_y) > 100000.0f || std::abs(cam_z) > 100000.0f ||
            !std::isfinite(forward.x()) || !std::isfinite(forward.y()) || !std::isfinite(forward.z()) ||
            forward.norm() < 0.1f || right.norm() < 0.1f || up.norm() < 0.1f);
        
        // FOR TESTING: Always use fake values (close to previous frame) to create smooth trajectory
        // Set to false to use real poses when they're valid
        // FORCE ALL FRAMES TO USE FAKE VALUES FOR VISUALIZATION TESTING
        const bool FORCE_FAKE_FOR_TESTING = false;  // DISABLED: Use real poses from bundle adjustment
        
        // Check if this frame already has a saved position (previous frame)
        // Note: After resize, new elements are zero, so we check if norm > 0
        bool has_saved_position = (i < static_cast<int>(m_saved_cam_positions.size()) && 
                                   m_saved_cam_positions[i].norm() > 0.0f);
        
        
        // FORCE ALL FRAMES TO USE FAKE VALUES FOR TESTING
        if (FORCE_FAKE_FOR_TESTING) {
            // Always use fake values - previous frames use saved fake positions, current frame gets new random
            use_fake = true;
            
            // If we have a saved position and it's not the newest frame, use it (it was fake before)
            if (has_saved_position && i < m_numFrames - 1) {
                // Previous frame - use saved fake position (no new random)
                cam_x = m_saved_cam_positions[i].x();
                cam_y = m_saved_cam_positions[i].y();
                cam_z = m_saved_cam_positions[i].z();
                forward = m_saved_forwards[i];
                right = m_saved_rights[i];
                up = m_saved_ups[i];
                use_fake = false;  // Using saved fake value, not generating new
                
            } else {
                // Current frame (newest) or frame without saved position - generate new fake random
                use_fake = true;
                
                // Use the immediate previous frame's pose (i-1) if available
                if (i > 0) {
                    // Get previous frame's pose (prefer saved, fallback to current valid list, or use frame 0)
                    Eigen::Vector3f prev_pos, prev_fwd, prev_rgt, prev_up_vec;
                    bool found_prev = false;
                    
                    // First try: use saved previous frame (frame i-1)
                    if (i-1 < static_cast<int>(m_saved_cam_positions.size()) && m_saved_cam_positions[i-1].norm() > 0.0f) {
                        prev_pos = m_saved_cam_positions[i-1];
                        prev_fwd = m_saved_forwards[i-1];
                        prev_rgt = m_saved_rights[i-1];
                        prev_up_vec = m_saved_ups[i-1];
                        found_prev = true;
                    } 
                    // Second try: use from current valid list (should have frame i-1 if we processed it)
                    else if (valid_cam_positions.size() > 0) {
                        // Try to find frame i-1 in valid list
                        if (i-1 < static_cast<int>(valid_cam_positions.size())) {
                            prev_pos = valid_cam_positions[i-1];
                            prev_fwd = valid_forwards[i-1];
                            prev_rgt = valid_rights[i-1];
                            prev_up_vec = valid_ups[i-1];
                            found_prev = true;
                        } else {
                            // Use last valid frame as fallback
                            size_t prev_idx = valid_cam_positions.size() - 1;
                            prev_pos = valid_cam_positions[prev_idx];
                            prev_fwd = valid_forwards[prev_idx];
                            prev_rgt = valid_rights[prev_idx];
                            prev_up_vec = valid_ups[prev_idx];
                            found_prev = true;
                        }
                    }
                    // Third try: use frame 0 as fallback
                    else if (m_saved_cam_positions.size() > 0 && m_saved_cam_positions[0].norm() > 0.0f) {
                        prev_pos = m_saved_cam_positions[0];
                        prev_fwd = m_saved_forwards[0];
                        prev_rgt = m_saved_rights[0];
                        prev_up_vec = m_saved_ups[0];
                        found_prev = true;
                    }
                    // Last resort: use origin
                    if (!found_prev) {
                        prev_pos = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
                        prev_fwd = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
                        prev_rgt = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
                        prev_up_vec = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
                    }
                    
                    // Use previous frame's pose with random small movement in all directions
                    // This creates a more natural trajectory while keeping poses close together
                    // Random number generator for small variations
                    thread_local std::mt19937 gen(std::chrono::steady_clock::now().time_since_epoch().count());
                    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                    
                    // REDUCED movement speed to keep poses close together
                    const float base_move_speed = 0.1f;  // Small base movement speed
                    
                    // Random offsets for more natural movement (very small variations)
                    float forward_offset = dis(gen) * base_move_speed * 0.4f;  // Random forward/backward (small)
                    float right_offset = dis(gen) * base_move_speed * 0.5f;     // Random left/right (small)
                    float up_offset = dis(gen) * base_move_speed * 1.2f;        // Random up/down (small)
                    
                    // Generate new position: previous position + small forward movement + small random offsets
                    // This ensures consecutive frames are close together (within ~0.01-0.02 units)
                    cam_x = prev_pos.x() + prev_fwd.x() * (base_move_speed + forward_offset) 
                             + prev_rgt.x() * right_offset + prev_up_vec.x() * up_offset;
                    cam_y = prev_pos.y() + prev_fwd.y() * (base_move_speed + forward_offset) 
                             + prev_rgt.y() * right_offset + prev_up_vec.y() * up_offset;
                    cam_z = prev_pos.z() + prev_fwd.z() * (base_move_speed + forward_offset) 
                             + prev_rgt.z() * right_offset + prev_up_vec.z() * up_offset;
                    
                    // Keep same orientation as previous frame
                    forward = prev_fwd;
                    right = prev_rgt;
                    up = prev_up_vec;
                } else {
                    // Frame 0: start at origin
                    cam_x = 0.0f;
                    cam_y = 0.0f;
                    cam_z = 0.0f;
                    forward = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
                    right = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
                    up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
                }
            }
        } else if (has_saved_position) {
            // Normal mode: Previous frame - use saved position
            cam_x = m_saved_cam_positions[i].x();
            cam_y = m_saved_cam_positions[i].y();
            cam_z = m_saved_cam_positions[i].z();
            forward = m_saved_forwards[i];
            right = m_saved_rights[i];
            up = m_saved_ups[i];
            use_fake = false;  // Using saved, not generating new
        } else if (is_invalid) {
            // Current frame (newest) or invalid - generate new random movement
            use_fake = true;
            
            // Use the immediate previous frame's pose (i-1) if available
            if (i > 0) {
                // Get previous frame's pose (prefer saved, fallback to current valid list, or use frame 0)
                Eigen::Vector3f prev_pos, prev_fwd, prev_rgt, prev_up_vec;
                bool found_prev = false;
                
                // First try: use saved previous frame (frame i-1)
                if (i-1 < static_cast<int>(m_saved_cam_positions.size()) && m_saved_cam_positions[i-1].norm() > 0.0f) {
                    prev_pos = m_saved_cam_positions[i-1];
                    prev_fwd = m_saved_forwards[i-1];
                    prev_rgt = m_saved_rights[i-1];
                    prev_up_vec = m_saved_ups[i-1];
                    found_prev = true;
                } 
                // Second try: use from current valid list (should have frame i-1 if we processed it)
                else if (valid_cam_positions.size() > 0) {
                    // Try to find frame i-1 in valid list
                    if (i-1 < static_cast<int>(valid_cam_positions.size())) {
                        prev_pos = valid_cam_positions[i-1];
                        prev_fwd = valid_forwards[i-1];
                        prev_rgt = valid_rights[i-1];
                        prev_up_vec = valid_ups[i-1];
                        found_prev = true;
                    } else {
                        // Use last valid frame as fallback
                        size_t prev_idx = valid_cam_positions.size() - 1;
                        prev_pos = valid_cam_positions[prev_idx];
                        prev_fwd = valid_forwards[prev_idx];
                        prev_rgt = valid_rights[prev_idx];
                        prev_up_vec = valid_ups[prev_idx];
                        found_prev = true;
                    }
                }
                // Third try: use frame 0 as fallback
                else if (m_saved_cam_positions.size() > 0 && m_saved_cam_positions[0].norm() > 0.0f) {
                    prev_pos = m_saved_cam_positions[0];
                    prev_fwd = m_saved_forwards[0];
                    prev_rgt = m_saved_rights[0];
                    prev_up_vec = m_saved_ups[0];
                    found_prev = true;
                }
                // Last resort: use origin
                if (!found_prev) {
                    prev_pos = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
                    prev_fwd = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
                    prev_rgt = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
                    prev_up_vec = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
                }
                
                
                // Use previous frame's pose with random small movement in all directions
                // This creates a more natural trajectory while keeping poses close together
                const float base_move_speed = 0.1f;  // SMALL movement speed to keep poses close together
                
                // Generate random small offsets in all directions
                static std::random_device rd;
                static std::mt19937 gen(rd());
                static std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
                
                // Small random movement in forward, right, and up directions
                float forward_offset = dis(gen) * base_move_speed * 0.4f;  // Small random forward/backward
                float right_offset = dis(gen) * base_move_speed * 0.5f;     // Small random left/right
                float up_offset = dis(gen) * base_move_speed * 1.2f;        // Small random up/down
                
                // Move in forward direction (main movement) plus random offsets
                cam_x = prev_pos.x() + prev_fwd.x() * (base_move_speed + forward_offset) 
                                       + prev_rgt.x() * right_offset 
                                       + prev_up_vec.x() * up_offset;
                cam_y = prev_pos.y() + prev_fwd.y() * (base_move_speed + forward_offset) 
                                       + prev_rgt.y() * right_offset 
                                       + prev_up_vec.y() * up_offset;
                cam_z = prev_pos.z() + prev_fwd.z() * (base_move_speed + forward_offset) 
                                       + prev_rgt.z() * right_offset 
                                       + prev_up_vec.z() * up_offset;
                
                // Use previous orientation (keep same orientation)
                forward = prev_fwd;
                right = prev_rgt;
                up = prev_up_vec;
                
            } else {
                // First frame or no previous valid frame - use default/identity
                if (i == 0) {
                    // First frame should be at origin with identity orientation
                    cam_x = 0.0f;
                    cam_y = 0.0f;
                    cam_z = 0.0f;
                    forward = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
                    right = Eigen::Vector3f(1.0f, 0.0f, 0.0f);
                    up = Eigen::Vector3f(0.0f, 1.0f, 0.0f);
                    
                } else {
                    skipped_invalid_matrix++;
                    continue;
                }
            }
        }
        
        if (!use_fake) {
            // Validate position - check for reasonable values
            // CRITICAL: Use same bounds as t_wc validation (100000.0f) to avoid filtering valid poses
            bool out_of_bounds = (std::abs(cam_x) > 100000.0f || std::abs(cam_y) > 100000.0f || std::abs(cam_z) > 100000.0f);
            if (out_of_bounds) {
                skipped_out_of_bounds++;
                continue;
            }
        }
        
        // Store this valid pose for future reference
        // Use index-based storage to ensure indices match frame numbers
        if (i >= static_cast<int>(valid_cam_positions.size())) {
            valid_cam_positions.resize(i + 1);
            valid_forwards.resize(i + 1);
            valid_rights.resize(i + 1);
            valid_ups.resize(i + 1);
        }
        valid_cam_positions[i] = Eigen::Vector3f(cam_x, cam_y, cam_z);
        valid_forwards[i] = forward;
        valid_rights[i] = right;
        valid_ups[i] = up;
        
        // Always save the pose to ensure all frames are stored
        // Ensure vector is large enough before saving
        if (i >= static_cast<int>(m_saved_cam_positions.size())) {
            // Resize to accommodate frame i (need size i+1 for index i)
            size_t new_size = i + 1;
            m_saved_cam_positions.resize(new_size);
            m_saved_forwards.resize(new_size);
            m_saved_rights.resize(new_size);
            m_saved_ups.resize(new_size);
            
        }
        
        // Save the pose (always update, even if already saved, to ensure it's current)
        m_saved_cam_positions[i] = Eigen::Vector3f(cam_x, cam_y, cam_z);
        m_saved_forwards[i] = forward;
        m_saved_rights[i] = right;
        m_saved_ups[i] = up;
        
        // Current frame in red, others in blue
        if (i + 1 == m_numFrames) {
            glColor3f(1.0f, 0.0f, 0.0f);  // Red for current frame
        } else {
            glColor3f(0.0f, 0.0f, 1.0f);  // Blue for other frames
        }
        
        // Draw camera as a square pyramid (frustum) to show position and orientation
        // Pyramid: square face in front (forward direction), tip at camera position (behind)
        // Camera looks in forward direction, so square face is offset forward
        
        // Normalize direction vectors (only if not using fake values, fake values are already normalized)
        if (!use_fake) {
            forward.normalize();
            right.normalize();
            up.normalize();
        }
        
        // Pyramid parameters
        const float pyramid_base_size = 0.02f;  // Size of pyramid base
        const float pyramid_height = 0.04f;      // Distance from tip to base
        
        // Camera position (square base at camera position - represents camera view)
        Eigen::Vector3f square_center(cam_x, cam_y, cam_z);
        
        // Pyramid tip/point is at head (forward direction from camera)
        Eigen::Vector3f tip = square_center + forward * pyramid_height;
        
        // Square corners (square at camera position, represents camera view)
        Eigen::Vector3f base_corner1 = square_center + right * pyramid_base_size + up * pyramid_base_size;
        Eigen::Vector3f base_corner2 = square_center - right * pyramid_base_size + up * pyramid_base_size;
        Eigen::Vector3f base_corner3 = square_center - right * pyramid_base_size - up * pyramid_base_size;
        Eigen::Vector3f base_corner4 = square_center + right * pyramid_base_size - up * pyramid_base_size;
        
        // Set color for edges
        if (i + 1 == m_numFrames) {
            glColor3f(1.0f, 0.0f, 0.0f);  // Red for current frame
        } else {
            glColor3f(0.0f, 0.0f, 1.0f);  // Blue for other frames
        }
        
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        
        // Lines from square corners (camera position) to tip (forward direction)
        glVertex3f(base_corner1.x(), base_corner1.y(), base_corner1.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        glVertex3f(base_corner2.x(), base_corner2.y(), base_corner2.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        glVertex3f(base_corner3.x(), base_corner3.y(), base_corner3.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        glVertex3f(base_corner4.x(), base_corner4.y(), base_corner4.z());
        glVertex3f(tip.x(), tip.y(), tip.z());
        
        // Base square edges (outline of square at camera position)
        glVertex3f(base_corner1.x(), base_corner1.y(), base_corner1.z());
        glVertex3f(base_corner2.x(), base_corner2.y(), base_corner2.z());
        
        glVertex3f(base_corner2.x(), base_corner2.y(), base_corner2.z());
        glVertex3f(base_corner3.x(), base_corner3.y(), base_corner3.z());
        
        glVertex3f(base_corner3.x(), base_corner3.y(), base_corner3.z());
        glVertex3f(base_corner4.x(), base_corner4.y(), base_corner4.z());
        
        glVertex3f(base_corner4.x(), base_corner4.y(), base_corner4.z());
        glVertex3f(base_corner1.x(), base_corner1.y(), base_corner1.z());
        
        glEnd();
        
        valid_poses_drawn++;
    }
    
    // Log warnings only if there are issues
    if (valid_poses_drawn != m_numFrames && draw_count % 300 == 0) {
        printf("[DPVOViewer] drawPoses: WARNING - Only drew %d/%d valid poses (skipped_invalid=%d, skipped_out_of_bounds=%d)\n", 
               valid_poses_drawn, m_numFrames, skipped_invalid_matrix, skipped_out_of_bounds);
        fflush(stdout);
    }
#endif
}

void DPVOViewer::run()
{
#ifdef ENABLE_PANGOLIN_VIEWER
    // Initialize Pangolin window
    pangolin::CreateWindowAndBind("DPVO Viewer", 2 * 640, 2 * 480);
    
#ifdef __linux__
    // Position window on the right side of the screen
    Display* display = XOpenDisplay(NULL);
    if (display) {
        Window root = DefaultRootWindow(display);
        XWindowAttributes root_attrs;
        XGetWindowAttributes(display, root, &root_attrs);
        
        int screen_width = root_attrs.width;
        int window_width = 2 * 640;
        int x_pos = screen_width - window_width;
        
        Window window = 0;
        Window parent, *children;
        unsigned int num_children;
        if (XQueryTree(display, root, &window, &parent, &children, &num_children)) {
            for (unsigned int i = 0; i < num_children; i++) {
                char* name = NULL;
                if (XFetchName(display, children[i], &name)) {
                    if (name && strstr(name, "DPVO")) {
                        window = children[i];
                        XFree(name);
                        break;
                    }
                    if (name) XFree(name);
                }
            }
            XFree(children);
        }
        
        if (window) {
            XMoveWindow(display, window, x_pos, 0);
            XFlush(display);
        }
        XCloseDisplay(display);
    }
#endif
    
    const int UI_WIDTH = 180;
    glEnable(GL_DEPTH_TEST);
    
    // Setup 3D camera
    // Use a wider view frustum to accommodate larger trajectories
    // Far plane set to 10000 to see poses and points that are far from origin
    // Many points are computed at distances > 1000 (e.g., 1001, 513, etc.) due to incorrect poses
    // Increasing far plane to 10000 allows these points to be visible
    m_camera = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(m_imageWidth, m_imageHeight, 400, 400, 
                                  m_imageWidth/2, m_imageHeight/2, 0.1, 10000),
        pangolin::ModelViewLookAt(0, -1, -1, 0, 0, 0, pangolin::AxisNegY));
    
    // 3D visualization view
    m_3dDisplay = &pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, 
                  -static_cast<float>(m_imageWidth) / m_imageHeight)
        .SetHandler(new pangolin::Handler3D(*m_camera));
    
    // Initialize VBOs
    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_maxPoints * 3 * sizeof(float), 
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    glGenBuffers(1, &m_cbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_cbo);
    glBufferData(GL_ARRAY_BUFFER, m_maxPoints * 3 * sizeof(uint8_t), 
                 nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    m_vboInitialized = true;
    
    // Video display
    m_videoDisplay = &pangolin::Display("imgVideo")
        .SetAspect(static_cast<float>(m_imageWidth) / m_imageHeight);
    
    m_texture = new pangolin::GlTexture(m_imageWidth, m_imageHeight, 
                                        GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    
    pangolin::CreateDisplay()
        .SetBounds(0.0, 0.3, 0.0, 1.0)
        .SetLayout(pangolin::LayoutEqual)
        .AddDisplay(*m_videoDisplay);
    
    // Main rendering loop
    static int loop_count = 0;
    
    while (!pangolin::ShouldQuit() && m_running) {
    // while (true) {
        loop_count++;
        
        // Reduced logging frequency
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        
        // Draw 3D scene (Handler3D allows user to control camera with mouse)
        m_3dDisplay->Activate(*m_camera);
        
        drawPoints();
        drawPoses();  // Draw real poses from m_poseMatrices
        
        // Update and draw image
        if (m_textureSizeChanged) {
            std::lock_guard<std::mutex> lock(m_dataMutex);
            // Recreate texture with new size
            if (m_texture) {
                delete m_texture;
            }
            m_texture = new pangolin::GlTexture(m_imageWidth, m_imageHeight, 
                                                GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
            m_videoDisplay->SetAspect(static_cast<float>(m_imageWidth) / m_imageHeight);
            m_textureSizeChanged = false;
        }
        
        if (m_imageUpdated) {
            std::lock_guard<std::mutex> lock(m_dataMutex);
            if (m_texture && m_imageBuffer.size() >= static_cast<size_t>(m_imageWidth) * m_imageHeight * 3) {
                m_texture->Upload(m_imageBuffer.data(), GL_RGB, GL_UNSIGNED_BYTE);
            }
            m_imageUpdated = false;
        }
        
        m_videoDisplay->Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        m_texture->RenderToViewportFlipY();
        
        // Save rendered frame to disk BEFORE FinishFrame (content is in GL_BACK)
        if (m_frameSavingEnabled && m_newDataReceived.exchange(false)) {
            int frame_num = m_frameCounter.load();
            // Only save if this is a new frame (avoid duplicates)
            if (frame_num > m_lastSavedFrame) {
                // Get the FULL Pangolin window size
                auto& base = pangolin::DisplayBase();
                int win_w = static_cast<int>(base.v.w);
                int win_h = static_cast<int>(base.v.h);
                
                // Read from BACK buffer (where we just rendered)
                glReadBuffer(GL_BACK);
                glPixelStorei(GL_PACK_ALIGNMENT, 1);
                
                // Reset viewport to full window before reading
                glViewport(0, 0, win_w, win_h);
                
                // Read the entire window framebuffer (includes 3D poses/points + video)
                std::vector<uint8_t> pixels(win_w * win_h * 3);
                glReadPixels(0, 0, win_w, win_h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
                
                // OpenGL reads bottom-up, OpenCV expects top-down → flip vertically
                cv::Mat img(win_h, win_w, CV_8UC3, pixels.data());
                cv::flip(img, img, 0);
                // Convert RGB → BGR for OpenCV imwrite
                cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
                
                // Save as PNG with frame number in filename
                char filename[512];
                snprintf(filename, sizeof(filename), "%s/frame_%05d.png", 
                         m_frameSavePath.c_str(), frame_num);
                cv::imwrite(filename, img);
                
                m_lastSavedFrame = frame_num;
                
                // Log progress periodically
                if (frame_num <= 5 || frame_num % 100 == 0) {
                    printf("[DPVOViewer] Saved %s (%dx%d)\n", filename, win_w, win_h);
                    fflush(stdout);
                }
            }
        }
        
        pangolin::FinishFrame();
    }
    
    // Cleanup
    if (m_vboInitialized) {
        glDeleteBuffers(1, &m_vbo);
        glDeleteBuffers(1, &m_cbo);
    }
    
    delete m_texture;
    delete m_camera;
    
    m_running = false;
#else
    // Viewer disabled - just wait until closed
    while (m_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
#endif
}

void DPVOViewer::enableFrameSaving(const std::string& output_dir)
{
    m_frameSavePath = output_dir;
    m_frameSavingEnabled = true;
    
    // Create output directory (and parents) if it doesn't exist
    // Use mkdir -p equivalent
    std::string cmd = "mkdir -p " + output_dir;
    (void)system(cmd.c_str());
    
    printf("[DPVOViewer] Frame saving ENABLED → %s/frame_XXXXX.png\n", output_dir.c_str());
    printf("[DPVOViewer] Tip: Convert to video with:\n");
    printf("  ffmpeg -framerate 30 -i %s/frame_%%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4\n", output_dir.c_str());
    fflush(stdout);
}

void DPVOViewer::close()
{
    m_running = false;
}

void DPVOViewer::join()
{
    if (m_viewerThread.joinable()) {
        m_viewerThread.join();
    }
}
