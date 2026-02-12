#pragma once
#include "patch_graph.hpp"
#include "patchify.hpp"  // Patchifier
#include "update.hpp"    // DPVOUpdate
#ifdef USE_ONNX_RUNTIME
#include "update_onnx.hpp"  // DPVOUpdateONNX
#endif
#include "dla_config.hpp"
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>

// Forward declaration
class DPVOViewer;

struct DPVOConfig {
    int PATCHES_PER_FRAME;
    int BUFFER_SIZE;
    int PATCH_SIZE;
    int MIXED_PRECISION;
    int LOOP_CLOSURE;
    int MAX_EDGE_AGE;
    int KEYFRAME_INDEX;
    int KEYFRAME_THRESH;
    int PATCH_LIFETIME;
    int REMOVAL_WINDOW;
    int OPTIMIZATION_WINDOW;  // Match Python's OPTIMIZATION_WINDOW=12

    DPVOConfig()
        : PATCHES_PER_FRAME(4),
          BUFFER_SIZE(4096),
          PATCH_SIZE(3),
          MIXED_PRECISION(0),
          LOOP_CLOSURE(0),
          MAX_EDGE_AGE(360),
          KEYFRAME_INDEX(4),
          KEYFRAME_THRESH(10), // 10
          PATCH_LIFETIME(6), // 6 
          REMOVAL_WINDOW(8), // 8
          OPTIMIZATION_WINDOW(5)  // Match Python default 12 , Alister test 5 on DPVO python , it works
    {}
};

class DPVO {
public:
    DPVO(const DPVOConfig& cfg, int ht, int wd);
    DPVO(const DPVOConfig& cfg, int ht, int wd, Config_S* config); // Constructor with config for update model
    ~DPVO();

    // Main processing function (called from thread)
    // intrinsics_in: [fx, fy, cx, cy] - if nullptr, uses stored m_intrinsics
    // Tensor-based version (preferred, avoids conversion)
    void run(int64_t timestamp, ea_tensor_t* imgTensor, const float* intrinsics_in = nullptr);
    
    // Helper function to continue run() logic after patchifier.forward() has been called
    // This avoids duplicate inference calls when tensor-based run() calls uint8_t* run()
    void runAfterPatchify(int64_t timestamp, const float* intrinsics_in, int H, int W,
                          int n, int n_use, int pm, int mm, int M, int P, int patch_D,
                          float* patches, uint8_t* clr, const uint8_t* image_for_viewer = nullptr);
    
    // Threading interface (similar to wnc_app)
    void startProcessingThread();
    void stopProcessingThread();
    void wakeProcessingThread();
#if defined(CV28) || defined(CV28_SIMULATOR)
    void updateInput(ea_tensor_t* imgTensor);
    void addFrame(ea_tensor_t* imgTensor);  // Convenience wrapper like wnc_app
#else
    // Fallback for non-CV28 platforms
    void updateInput(const uint8_t* image, int H, int W);
    void addFrame(const uint8_t* image, int H, int W);
#endif
    bool isProcessingComplete();
    
    // Set intrinsics (can be called to update from config)
    void setIntrinsics(const float intrinsics[4]);
    void setIntrinsicsFromConfig(Config_S* config);  // Initialize from config file
    
    void terminate();
    
    // Set update model (if not initialized in constructor)
    void setUpdateModel(Config_S* config);
    
    // Set fnet and inet models for Patchifier
    void setPatchifierModels(Config_S* fnetConfig, Config_S* inetConfig);
    
    // Initialize threads (called from constructor, similar to WNC_APP::_init)
    void _startThreads();
    
    // Inference cache: save/load FNet/INet/Update model outputs to binary files.
    // First run: models run normally and outputs are saved to cachePath.
    // Subsequent runs: outputs are loaded from cache, model inference is SKIPPED.
    // This can dramatically speed up processing when replaying the same video.
    // Call AFTER setPatchifierModels() and setUpdateModel().
    //
    // cachePath: directory for cache files (e.g. "inference_cache")
    //   └── fnet/          FNet outputs per frame
    //   └── inet/          INet outputs per frame  
    //   └── update/        Update model outputs per update call
    void enableInferenceCache(const std::string& cachePath = "inference_cache");
    
    // Visualization
    void enableVisualization(bool enable = true);
    void updateViewer();  // Update viewer with current state

private:
    // Forward declaration for InputFrame (defined later in private section)
    struct InputFrame;
    
    void update();
    void keyframe();

    void edgesForward(std::vector<int>& kk, std::vector<int>& jj);
    void edgesBackward(std::vector<int>& kk, std::vector<int>& jj);
    void appendFactors(const std::vector<int>& kk, const std::vector<int>& jj);
    void removeFactors(const bool* mask, bool store);
    void reproject(
        const int* ii, 
        const int* jj, 
        const int* kk, 
        int num_edges, 
        float* coords_out,
        float* Ji_out = nullptr,      // [num_edges, 2, P, P, 6] flattened (optional)
        float* Jj_out = nullptr,      // [num_edges, 2, P, P, 6] flattened (optional)
        float* Jz_out = nullptr,      // [num_edges, 2, P, P, 1] flattened (optional)
        float* valid_out = nullptr     // [num_edges, P, P] flattened (optional)
    ); // Alister add 2025-12-26
    float motionMagnitude(int i, int j);
    float motionProbe();  // Motion probe for initialization check (matches Python motion_probe)
    
    // Helper function to save reproject inputs for debugging/comparison
    void save_reproject_inputs(int num_active);
    
    // Helper function to save reproject outputs for debugging/comparison
    void save_reproject_outputs(int num_active, const float* coords, int P);
    
    // Helper function to save correlation outputs for debugging/comparison
    void save_correlation_outputs(int num_active, const float* coords, const float* corr,
                                  const float* corr1_8x8, const float* corr2_8x8,
                                  size_t corr_8x8_size, int P, int D);
    
    // Helper function to save update model inputs for debugging/comparison
    void save_update_model_inputs(int num_active, int frame_override = -1);
    
    // Helper function to save update model outputs for debugging/comparison
    void save_update_model_outputs(const DPVOUpdate_Prediction& pred);
    
    // Helper function to save BA outputs for debugging/comparison
    void save_ba_outputs();
    
    // Helper function to save poses after sync for debugging/comparison
    void save_poses_after_sync(int synced_count);
    
    // Helper function to save BA inputs for debugging/comparison
    void save_ba_inputs_to_bin_files(int num_active, const float* coords);
    
    // Helper functions to save keyframe inputs/outputs for debugging/comparison
    void save_keyframe_inputs(int n, int m_before, int num_edges_before);
    void save_keyframe_outputs(int n_before, int m_before, int num_edges_before,
                               int n_after, int m_after, int num_edges_after,
                               int i, int j, float m, bool should_remove);
    
    // Bundle Adjustment
    void bundleAdjustment(
        float lmbda = 1e-4f,
        float ep = 100.0f,
        bool structure_only = false,
        int fixedp = 1
    );

    // Helpers for indexing
    inline int imap_idx(int i, int j, int k) const { return i * m_cfg.PATCHES_PER_FRAME * m_DIM + j * m_DIM + k; }
    inline int gmap_idx(int i, int j, int c, int y, int x) const {
        // CRITICAL: gmap uses D_gmap = 3 (from patchify_cpu_safe with radius=1), matches P=3
        // patchify_cpu_safe: radius = m_patch_size/2 = 1, D = 2*radius + 1 = 3 (matches Python altcorr.patchify)
        const int D_gmap = 3;  // D_gmap = 2 * (m_P/2) + 1 = 3 (matches Python: .view(..., P, P) where P=3)
        return i * m_cfg.PATCHES_PER_FRAME * 128 * D_gmap * D_gmap +
               j * 128 * D_gmap * D_gmap +
               c * D_gmap * D_gmap +
               y * D_gmap +
               x;
    }
    inline int fmap1_idx(int b, int m, int c, int y, int x) const {
        return b * m_mem * 128 * m_fmap1_H * m_fmap1_W +
               m * 128 * m_fmap1_H * m_fmap1_W +
               c * m_fmap1_H * m_fmap1_W +
               y * m_fmap1_W +
               x;
    }
    inline int fmap2_idx(int b, int m, int c, int y, int x) const {
        return b * m_mem * 128 * m_fmap2_H * m_fmap2_W +
               m * 128 * m_fmap2_H * m_fmap2_W +
               c * m_fmap2_H * m_fmap2_W +
               y * m_fmap2_W +
               x;
    }

private:
    DPVOConfig m_cfg;
    PatchGraph m_pg;

    int m_ht, m_wd;
    int m_counter;
    bool m_is_initialized;

    int m_DIM;       // feature dimension
    int m_P;         // patch size
    int m_pmem, m_mem;
    int m_fmap1_H, m_fmap1_W;
    int m_fmap2_H, m_fmap2_W;
    int m_maxEdge;   // Maximum edge count for model input (default: 384, can be changed)

    float* m_imap; // (self.pmem, self.M, DIM, **kwargs)
    float* m_gmap; // (self.pmem, self.M, 128, self.P, self.P, **kwargs)
    float* m_fmap1; // (1, self.mem, 128, ht // 1, wd // 1, **kwargs)
    float* m_fmap2; // (1, self.mem, 128, ht // 4, wd // 4, **kwargs)

    float* m_cur_imap;   // pointer to latest frame in imap
    float* m_cur_gmap;   // pointer to latest frame in gmap
    float* m_cur_fmap1;  // pointer to latest frame in fmap1


    std::vector<int64_t> m_tlist;

    // ---- Patchifier for extracting patches ----
    Patchifier m_patchifier;
    
    // ---- DPVO Update Model (optional) ----
#ifdef USE_ONNX_RUNTIME
    std::unique_ptr<DPVOUpdateONNX> m_updateModel_onnx;
    bool m_useOnnxUpdateModel = false;
#endif
    std::unique_ptr<DPVOUpdate> m_updateModel;
    int m_updateFrameCounter = 0;
    
    // Pre-allocated buffers for reshapeInput (reused to avoid memory allocation overhead)
    std::vector<float> m_reshape_net_input;
    std::vector<float> m_reshape_inp_input;
    std::vector<float> m_reshape_corr_input;
    std::vector<float> m_reshape_ii_input;
    std::vector<float> m_reshape_jj_input;
    std::vector<float> m_reshape_kk_input;
    
    // ---- Threading infrastructure (similar to wnc_app) ----
    struct InputFrame {
        std::vector<uint8_t> image;  // Store converted image data [C, H, W] format (optional, for backward compatibility)
        int H, W;
#if defined(CV28) || defined(CV28_SIMULATOR)
        ea_tensor_t* tensor_img;     // Store image tensor directly (preferred, avoids conversion)
#endif
    };
    
    // ---- Intrinsics and timestamp (stored as member variables) ----
    float m_intrinsics[4];      // Camera intrinsics [fx, fy, cx, cy]
    int64_t m_currentTimestamp; // Current frame timestamp
    
    std::thread m_processingThread;
    std::atomic<bool> m_processingThreadRunning{false};
    std::mutex m_queueMutex;
    std::condition_variable m_queueCV;
    std::queue<InputFrame> m_inputFrameQueue;
    std::atomic<bool> m_bDone{true};
    
    // Helper function to check if there's work to do
    bool _hasWorkToDo();
    
    // Visualization (optional)
    std::unique_ptr<DPVOViewer> m_viewer;
    bool m_visualizationEnabled{false};
    
    // Store all historical poses for visualization (not just sliding window)
    // This allows the viewer to show the full trajectory, not just the current optimization window
    std::vector<SE3> m_allPoses;
    
    // Store all historical points and colors for visualization (not just sliding window)
    // Points are stored per frame: m_allPoints[frame_idx * M + patch_idx]
    std::vector<Vec3> m_allPoints;  // All historical 3D points
    std::vector<uint8_t> m_allColors;  // All historical colors [frame_idx * M * 3 + patch_idx * 3 + channel]
    
    // Map sliding window indices to global frame indices using timestamps
    // When keyframe() removes frames, we use timestamps to find which global frame each sliding window index corresponds to
    std::vector<int64_t> m_allTimestamps;  // Global timestamps for all frames
    
    // Helper to compute point cloud from patches and poses
    void computePointCloud();
};
