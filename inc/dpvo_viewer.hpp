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

#pragma once

#include "se.hpp"
#include "patch_graph.hpp"
#include "detection_3d.hpp"
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <atomic>
#include <cstdint>

// Optional Pangolin support - define ENABLE_PANGOLIN_VIEWER to enable
#ifdef ENABLE_PANGOLIN_VIEWER
// Include Pangolin - it will handle OpenGL includes internally
// Note: Make sure GLEW or epoxy is available (Pangolin uses epoxy by default)
#include <pangolin/pangolin.h>
#ifdef __linux__
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif
#endif

#if defined(CV28) || defined(CV28_SIMULATOR)
#include "yolov8_decoder.hpp"
#endif

/**
 * @brief Native C++ viewer for DPVO visualization
 * 
 * Displays:
 * - 3D point cloud (colored points)
 * - Camera trajectory (camera poses as wireframe cameras)
 * - Current video frame
 * 
 * Runs in a separate thread for non-blocking visualization
 */
class DPVOViewer {
public:
    /**
     * @brief Constructor
     * @param image_width Image width
     * @param image_height Image height
     * @param max_frames Maximum number of frames to visualize
     * @param max_points Maximum number of points to visualize
     */
    DPVOViewer(int image_width, int image_height, int max_frames = 36, int max_points = 288);
    
    /**
     * @brief Destructor - stops viewer thread
     */
    ~DPVOViewer();
    
    /**
     * @brief Update current image frame
     * @param image_data Pointer to image data (RGB, row-major, uint8_t)
     * @param width Image width
     * @param height Image height
     */
    void updateImage(const uint8_t* image_data, int width, int height);
    
    /**
     * @brief Update poses (camera trajectory)
     * @param poses Array of SE3 poses
     * @param num_frames Number of active frames
     */
    void updatePoses(const SE3* poses, int num_frames);
    
    /**
     * @brief Update point cloud
     * @param points Array of Vec3 points
     * @param colors Array of RGB colors (uint8_t[3] per point)
     * @param num_points Number of points
     */
    void updatePoints(const Vec3* points, const uint8_t* colors, int num_points);

#if defined(CV28) || defined(CV28_SIMULATOR)
    /** Optional YOLOv8 overlay: set before updateImage so boxes are drawn on the frame */
    void setYOLOv8Boxes(const std::vector<std::vector<v8xyxy>>* boxes);
    /** YOLOv8 model input size for box scaling (default 512x288). Call when using ONNX with different input size. */
    void setYOLOv8ModelSize(int model_w, int model_h);
#endif

    /**
     * @brief 3D detections (e.g. from YOLOv8 + ground-plane back-projection) to draw in 3D view.
     * Each object is drawn as a simple shape: person = tall box, vehicle = flat box, else = small cube.
     */
    void setDetections3D(const std::vector<Detection3D>& detections);

    /**
     * @brief Enable saving rendered frames to image files
     * @param output_dir Directory to save frames (will be created if needed)
     * 
     * Saves the Pangolin window contents as PNG images after each new frame
     * is rendered. Files are named frame_XXXXX.png (e.g., frame_00015.png).
     * This is useful when AMBA model inference is slow and screen recording
     * would take too long. The saved images can be combined into a video with:
     *   ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 output.mp4
     */
    void enableFrameSaving(const std::string& output_dir);
    
    /**
     * @brief Draw 3D detections (pedestrians, vehicles) as simple wireframe shapes. Called internally.
     */
    void drawDetections3D();

    /**
     * @brief Close viewer and stop thread
     */
    void close();
    
    /**
     * @brief Wait for viewer thread to finish
     */
    void join();
    
    /**
     * @brief Check if viewer is running
     */
    bool isRunning() const { return m_running; }

private:
    void run();  // Main rendering loop (runs in separate thread)
    void drawPoints();
    void drawPoses();  // Draw real poses from m_poseMatrices
    void drawPoses_fake();
    void convertPosesToMatrices();
    
    // Thread management
    std::thread m_viewerThread;
    std::atomic<bool> m_running{false};
    std::mutex m_dataMutex;
    
    // Image data
    int m_imageWidth;
    int m_imageHeight;
    std::vector<uint8_t> m_imageBuffer;
    std::atomic<bool> m_imageUpdated{false};
    std::atomic<bool> m_textureSizeChanged{false};
    
    // Pose data
    int m_maxFrames;
    int m_numFrames{0};
    std::vector<SE3> m_poses;
    std::vector<float> m_poseMatrices;  // 4x4 matrices for OpenGL
    
    // Saved fake poses for testing (persist across draw calls)
    std::vector<Eigen::Vector3f> m_saved_cam_positions;
    std::vector<Eigen::Vector3f> m_saved_forwards;
    std::vector<Eigen::Vector3f> m_saved_rights;
    std::vector<Eigen::Vector3f> m_saved_ups;
    
    // Point cloud data
    int m_maxPoints;
    int m_numPoints{0};
    std::vector<Vec3> m_points;
    std::vector<uint8_t> m_colors;
    
#if defined(CV28) || defined(CV28_SIMULATOR)
    const std::vector<std::vector<v8xyxy>>* m_yolov8Boxes{nullptr};
    int m_yolov8ModelW = 512;
    int m_yolov8ModelH = 288;
#endif

    // 3D detections (YOLOv8 back-projected to ground plane)
    std::vector<Detection3D> m_detections3D;

    // Frame saving
    bool m_frameSavingEnabled{false};
    std::string m_frameSavePath;
    std::atomic<int> m_frameCounter{0};          // DPVO frame number (from updatePoses)
    std::atomic<bool> m_newDataReceived{false};   // Flag: new data arrived since last save
    int m_lastSavedFrame{-1};                     // Last frame number saved to disk
    
#ifdef ENABLE_PANGOLIN_VIEWER
    // OpenGL/Pangolin state
    pangolin::GlTexture* m_texture{nullptr};
    pangolin::View* m_videoDisplay{nullptr};
    pangolin::View* m_3dDisplay{nullptr};
    pangolin::OpenGlRenderState* m_camera{nullptr};
    
    // OpenGL buffers for point cloud
    GLuint m_vbo{0};  // Vertex buffer object
    GLuint m_cbo{0};  // Color buffer object
    bool m_vboInitialized{false};
#endif
};

