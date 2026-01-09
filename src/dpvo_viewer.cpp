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
    
    if (width != m_imageWidth || height != m_imageHeight) {
        m_imageWidth = width;
        m_imageHeight = height;
        m_imageBuffer.resize(width * height * 3);
    }
    
    // Copy image data (assuming RGB format)
    std::memcpy(m_imageBuffer.data(), image_data, width * height * 3);
    m_imageUpdated = true;
}

void DPVOViewer::updatePoses(const SE3* poses, int num_frames)
{
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    num_frames = std::min(num_frames, m_maxFrames);
    m_numFrames = num_frames;
    
    // Copy poses
    for (int i = 0; i < num_frames; i++) {
        m_poses[i] = poses[i];
    }
    
    // Convert to matrices for OpenGL
    convertPosesToMatrices();
}

void DPVOViewer::updatePoints(const Vec3* points, const uint8_t* colors, int num_points)
{
    std::lock_guard<std::mutex> lock(m_dataMutex);
    
    num_points = std::min(num_points, m_maxPoints);
    m_numPoints = num_points;
    
    // Copy points
    for (int i = 0; i < num_points; i++) {
        m_points[i] = points[i];
    }
    
    // Copy colors
    if (colors) {
        std::memcpy(m_colors.data(), colors, num_points * 3);
    } else {
        // Default white color if no colors provided
        std::fill(m_colors.begin(), m_colors.begin() + num_points * 3, 255);
    }
}

void DPVOViewer::convertPosesToMatrices()
{
#ifdef ENABLE_PANGOLIN_VIEWER
    // Convert SE3 poses to 4x4 OpenGL matrices (column-major)
    for (int i = 0; i < m_numFrames; i++) {
        Eigen::Matrix4f T = m_poses[i].inverse().matrix();  // Inverse for camera-to-world
        
        // Store in column-major order for OpenGL
        float* mat = &m_poseMatrices[i * 16];
        mat[0]  = T(0,0); mat[4]  = T(1,0); mat[8]  = T(2,0); mat[12] = T(3,0);
        mat[1]  = T(0,1); mat[5]  = T(1,1); mat[9]  = T(2,1); mat[13] = T(3,1);
        mat[2]  = T(0,2); mat[6]  = T(1,2); mat[10] = T(2,2); mat[14] = T(3,2);
        mat[3]  = T(0,3); mat[7]  = T(1,3); mat[11] = T(2,3); mat[15] = T(3,3);
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
    
    glPointSize(2.0f);
    glDrawArrays(GL_POINTS, 0, m_numPoints);
    
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
    
    const int NUM_POINTS = 8;
    const int NUM_LINES = 10;
    
    const float CAM_POINTS[NUM_POINTS][3] = {
        { 0,   0,   0},
        {-1,  -1, 1.5},
        { 1,  -1, 1.5},
        { 1,   1, 1.5},
        {-1,   1, 1.5},
        {-0.5, 1, 1.5},
        { 0.5, 1, 1.5},
        { 0, 1.2, 1.5}
    };
    
    const int CAM_LINES[NUM_LINES][2] = {
        {1,2}, {2,3}, {3,4}, {4,1}, {1,0}, {0,2}, {3,0}, {0,4}, {5,7}, {7,6}
    };
    
    const float SZ = 0.05f;
    
    glColor3f(0.0f, 0.5f, 1.0f);
    glLineWidth(1.5f);
    
    for (int i = 0; i < m_numFrames; i++) {
        // Current frame in red
        if (i + 1 == m_numFrames) {
            glColor3f(1.0f, 0.0f, 0.0f);
        } else {
            glColor3f(0.0f, 0.5f, 1.0f);
        }
        
        glPushMatrix();
        glMultMatrixf(&m_poseMatrices[i * 16]);
        
        glBegin(GL_LINES);
        for (int j = 0; j < NUM_LINES; j++) {
            const int u = CAM_LINES[j][0], v = CAM_LINES[j][1];
            glVertex3f(SZ * CAM_POINTS[u][0], SZ * CAM_POINTS[u][1], SZ * CAM_POINTS[u][2]);
            glVertex3f(SZ * CAM_POINTS[v][0], SZ * CAM_POINTS[v][1], SZ * CAM_POINTS[v][2]);
        }
        glEnd();
        
        glPopMatrix();
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
    m_camera = new pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(m_imageWidth, m_imageHeight, 400, 400, 
                                  m_imageWidth/2, m_imageHeight/2, 0.1, 500),
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
    while (!pangolin::ShouldQuit() && m_running) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        
        // Draw 3D scene
        m_3dDisplay->Activate(*m_camera);
        drawPoints();
        drawPoses();
        
        // Update and draw image
        if (m_imageUpdated) {
            std::lock_guard<std::mutex> lock(m_dataMutex);
            m_texture->Upload(m_imageBuffer.data(), GL_RGB, GL_UNSIGNED_BYTE);
            m_imageUpdated = false;
        }
        
        m_videoDisplay->Activate();
        glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
        m_texture->RenderToViewportFlipY();
        
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
