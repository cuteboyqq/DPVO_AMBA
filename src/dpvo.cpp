#include "dpvo.hpp"
#include "patchify.hpp" // Patchifier
#include "update.hpp"   // DPVOUpdate
#include "dpvo_viewer.hpp"  // DPVOViewer
#include <algorithm>
#include <cstring>
#include <cstdlib>  // For std::abort()
#include <stdexcept>
#include <chrono>
#include <random>   // For random depth initialization
#include "projective_ops.hpp"
#include "correlation_kernel.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
static_assert(sizeof(((PatchGraph*)0)->m_index[0]) ==
              sizeof(int) * PatchGraph::M,
              "PatchGraph layout mismatch");

// -------------------------------------------------------------
// Constructor
// -------------------------------------------------------------
DPVO::DPVO(const DPVOConfig& cfg, int ht, int wd)
    : DPVO(cfg, ht, wd, nullptr)
{
}

DPVO::DPVO(const DPVOConfig& cfg, int ht, int wd, Config_S* config)
    : m_cfg(cfg),
      m_ht(ht), m_wd(wd),
      m_counter(0),
      m_is_initialized(false),
      m_DIM(384),    // same as NET_DIM
      m_P(PatchGraph::P),
      m_pmem(cfg.BUFFER_SIZE),
      m_mem(cfg.BUFFER_SIZE),
      m_patchifier(3, 384),  // Initialize with patch_size=3, DIM=384 (matches INet output channels)
      m_currentTimestamp(0),
      m_pg()  // Explicitly initialize PatchGraph (calls reset() which sets m_n=0)
{
    // Ensure PatchGraph is properly initialized - call reset() explicitly
    m_pg.reset();
    // Verify initialization
    if (m_pg.m_n != 0) {
        fprintf(stderr, "[DPVO] WARNING: PatchGraph.m_n is %d after reset, forcing to 0\n", m_pg.m_n);
        fflush(stderr);
        m_pg.m_n = 0;
        m_pg.m_m = 0;
    }
    // fmap sizes - Models (FNet/INet) output at 1/4 resolution of input
    // fmap1: Model output at 1/4 of input (e.g., 132x240 for 528x960 input)
    // fmap2: Downsampled from fmap1 by 4x (1/16 of original, e.g., 33x60 for 528x960 input)
    // Python: fmap1 = F.avg_pool2d(fmap[0], 1, 1) (no downsampling, so fmap1 = fmap at 1/4 res)
    //         fmap2 = F.avg_pool2d(fmap[0], 4, 4) (downsample by 4x, so fmap2 at 1/16 res)
    m_fmap1_H = ht / 4;  // Model output height at 1/4 resolution (will be updated to exact model output when models are set)
    m_fmap1_W = wd / 4;  // Model output width at 1/4 resolution
    m_fmap2_H = ht / 16; // fmap2 height at 1/16 resolution (1/4 of fmap1)
    m_fmap2_W = wd / 16; // fmap2 width at 1/16 resolution (1/4 of fmap1)

    // Validate dimensions to prevent bad_array_new_length
    if (m_fmap1_H <= 0 || m_fmap1_W <= 0 || m_fmap2_H <= 0 || m_fmap2_W <= 0) {
        throw std::runtime_error("Invalid fmap dimensions calculated from image size");
    }
    if (m_pmem <= 0 || m_mem <= 0 || cfg.PATCHES_PER_FRAME <= 0) {
        throw std::runtime_error("Invalid buffer configuration");
    }

	const int M = cfg.PATCHES_PER_FRAME;
    
    // Calculate array sizes and validate
    // CRITICAL: gmap uses D_gmap = 3 (from patchify_cpu_safe with radius=1), matches P=3
    // patchify_cpu_safe: radius = m_patch_size/2 = 1, D = 2*radius + 1 = 3 (matches Python altcorr.patchify)
    const int patch_radius = m_P / 2;  // m_P = 3, so radius = 1
    const int D_gmap = 2 * patch_radius + 1;  // D_gmap = 3 (matches Python: .view(..., P, P) where P=3)
    
    size_t imap_size = static_cast<size_t>(m_pmem) * static_cast<size_t>(M) * static_cast<size_t>(m_DIM);
    size_t gmap_size = static_cast<size_t>(m_pmem) * static_cast<size_t>(M) * 128 * static_cast<size_t>(D_gmap) * static_cast<size_t>(D_gmap);
    size_t fmap1_size = static_cast<size_t>(m_mem) * 128 * static_cast<size_t>(m_fmap1_H) * static_cast<size_t>(m_fmap1_W);
    size_t fmap2_size = static_cast<size_t>(m_mem) * 128 * static_cast<size_t>(m_fmap2_H) * static_cast<size_t>(m_fmap2_W);
    
    if (imap_size == 0 || gmap_size == 0 || fmap1_size == 0 || fmap2_size == 0) {
        throw std::runtime_error("Calculated array size is zero");
    }
    
    // allocate float arrays
    m_imap  = new float[imap_size]();
    m_gmap  = new float[gmap_size]();
    m_fmap1 = new float[fmap1_size]();
    m_fmap2 = new float[fmap2_size]();

	// -----------------------------
    // Zero-initialize (important!)
    // -----------------------------
    std::memset(m_imap,  0, sizeof(float) * imap_size);
    std::memset(m_gmap,  0, sizeof(float) * gmap_size);
    std::memset(m_fmap1, 0, sizeof(float) * fmap1_size);
    std::memset(m_fmap2, 0, sizeof(float) * fmap2_size);

    // Initialize intrinsics from config or use defaults
    if (config != nullptr) {
        setIntrinsicsFromConfig(config);
        m_updateModel = std::make_unique<DPVOUpdate>(config, nullptr);
    } else {
        // Default intrinsics (will be updated when frame dimensions are known)
        m_intrinsics[0] = static_cast<float>(wd) * 0.5f;  // fx
        m_intrinsics[1] = static_cast<float>(ht) * 0.5f;  // fy
        m_intrinsics[2] = static_cast<float>(wd) * 0.5f;  // cx
        m_intrinsics[3] = static_cast<float>(ht) * 0.5f;  // cy
    }
    
    // Initialize max edge count for model input
    // To change this value, simply modify the number below (e.g., 256, 360, 384, 512, 768)
    // Note: You must also update your ONNX model and AMBA conversion YAML to match this value
    // This should match MAX_EDGES in patch_graph.hpp
    m_maxEdge = 360;
    
    // Pre-allocate buffers for reshapeInput to avoid memory allocation overhead
    const int CORR_DIM = 882;
    m_reshape_net_input.resize(1 * 384 * m_maxEdge * 1, 0.0f);
    m_reshape_inp_input.resize(1 * 384 * m_maxEdge * 1, 0.0f);
    m_reshape_corr_input.resize(1 * CORR_DIM * m_maxEdge * 1, 0.0f);
    m_reshape_ii_input.resize(1 * 1 * m_maxEdge * 1, 0.0f);
    m_reshape_jj_input.resize(1 * 1 * m_maxEdge * 1, 0.0f);
    m_reshape_kk_input.resize(1 * 1 * m_maxEdge * 1, 0.0f);
}

void DPVO::_startThreads()
{
#if defined(CV28) || defined(CV28_SIMULATOR)
    startProcessingThread();
    m_processingThreadRunning = true;
#endif
}

void DPVO::setUpdateModel(Config_S* config)
{
    if (config != nullptr && m_updateModel == nullptr) {
        m_updateModel = std::make_unique<DPVOUpdate>(config, nullptr);
    }
}

void DPVO::setPatchifierModels(Config_S* fnetConfig, Config_S* inetConfig)
{
    m_patchifier.setModels(fnetConfig, inetConfig);
    
    // Update fmap dimensions based on actual model output dimensions
    // Models output at 1/4 resolution of input
    int model_H = m_patchifier.getOutputHeight();
    int model_W = m_patchifier.getOutputWidth();
    
    if (model_H > 0 && model_W > 0) {
        // fmap1: Model output at 1/4 resolution (e.g., 132x240 for 528x960 input)
        m_fmap1_H = model_H;
        m_fmap1_W = model_W;
        
        // fmap2: Downsampled from fmap1 by 4x (1/16 of original, e.g., 33x60 for 528x960 input)
        m_fmap2_H = model_H / 4;
        m_fmap2_W = model_W / 4;
        
        auto logger = spdlog::get("dpvo");
        if (logger) {
            logger->info("DPVO::setPatchifierModels: Updated fmap dimensions - fmap1: {}x{}, fmap2: {}x{}",
                         m_fmap1_H, m_fmap1_W, m_fmap2_H, m_fmap2_W);
        }
    } else {
        // Fallback: use 1/4 of input dimensions if models not ready
        m_fmap1_H = m_ht / 4;
        m_fmap1_W = m_wd / 4;
        m_fmap2_H = m_fmap1_H / 4;
        m_fmap2_W = m_fmap1_W / 4;
        
        auto logger = spdlog::get("dpvo");
        if (logger) {
            logger->warn("DPVO::setPatchifierModels: Models not ready, using fallback dimensions - fmap1: {}x{}, fmap2: {}x{}",
                         m_fmap1_H, m_fmap1_W, m_fmap2_H, m_fmap2_W);
        }
    }
    
    // Start threads after models are set (similar to WNC_APP::_init -> _startThreads)
    _startThreads();
}

void DPVO::setIntrinsics(const float intrinsics[4])
{
    std::memcpy(m_intrinsics, intrinsics, sizeof(float) * 4);
    auto logger = spdlog::get("dpvo");
    if (logger) {
        logger->info("DPVO::setIntrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
                     intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]);
    }
}

void DPVO::setIntrinsicsFromConfig(Config_S* config)
{
    if (config == nullptr) return;
    
    // Python stores intrinsics as 4 values: [fx, fy, cx, cy]
    // Use intrinsic_fx/fy/cx/cy from config file if available (> 0), otherwise use frame center as fallback
    
    float intrinsic_fx = config->stCameraConfig.intrinsic_fx;
    float intrinsic_fy = config->stCameraConfig.intrinsic_fy;
    float intrinsic_cx = config->stCameraConfig.intrinsic_cx;
    float intrinsic_cy = config->stCameraConfig.intrinsic_cy;
    
    int frameWidth = config->frameWidth;
    int frameHeight = config->frameHeight;

    // Store intrinsics as 4 values: [fx, fy, cx, cy] (matching Python format)
    // fx: Use intrinsic_fx if > 0, otherwise use frameWidth/2 as fallback
    if (intrinsic_fx > 0.0f) {
        m_intrinsics[0] = intrinsic_fx;  // fx
    } else {
        // Default: use frame dimensions (rough estimate)
        m_intrinsics[0] = static_cast<float>(frameWidth) * 0.5f;   // fx
    }
    
    // fy: Use intrinsic_fy if > 0, otherwise use frameHeight/2 as fallback
    if (intrinsic_fy > 0.0f) {
        m_intrinsics[1] = intrinsic_fy;  // fy
    } else {
        // Default: use frame dimensions (rough estimate)
        m_intrinsics[1] = static_cast<float>(frameHeight) * 0.5f;  // fy
    }
    
    // cx: Use intrinsic_cx if > 0, otherwise use frameWidth/2 as fallback
    if (intrinsic_cx > 0.0f) {
        m_intrinsics[2] = intrinsic_cx;  // cx
    } else {
        m_intrinsics[2] = static_cast<float>(frameWidth) * 0.5f;   // cx (default: image center)
    }
    
    // cy: Use intrinsic_cy if > 0, otherwise use frameHeight/2 as fallback
    if (intrinsic_cy > 0.0f) {
        m_intrinsics[3] = intrinsic_cy;  // cy
    } else {
        m_intrinsics[3] = static_cast<float>(frameHeight) * 0.5f;  // cy (default: image center)
    }
    
    auto logger = spdlog::get("dpvo");
    if (logger) {
        logger->info("DPVO::setIntrinsicsFromConfig: Calculated intrinsics [fx, fy, cx, cy] = "
                     "[{:.2f}, {:.2f}, {:.2f}, {:.2f}] from config "
                     "(intrinsic_fx={:.2f}, intrinsic_fy={:.2f}, intrinsic_cx={:.2f}, intrinsic_cy={:.2f}, "
                     "frameSize={}x{})",
                     m_intrinsics[0], m_intrinsics[1], m_intrinsics[2], m_intrinsics[3],
                     intrinsic_fx, intrinsic_fy, intrinsic_cx, intrinsic_cy,
                     frameWidth, frameHeight);
    }
}

DPVO::~DPVO() {
    // Update model will be automatically destroyed by unique_ptr
    delete[] m_imap;
    delete[] m_gmap;
    delete[] m_fmap1;
    delete[] m_fmap2;
}

// -------------------------------------------------------------
// Main entry (equivalent to dpvo.py __call__)
// -------------------------------------------------------------
// Helper function to continue run() logic after patchifier.forward() has been called
void DPVO::runAfterPatchify(int64_t timestamp, const float* intrinsics_in, int H, int W,
                             int n, int n_use, int pm, int mm, int M, int P, int patch_D,
                             float* patches, uint8_t* clr, const uint8_t* image_for_viewer)
{
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    // -------------------------------------------------
    // 2. Bookkeeping
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Starting bookkeeping, n={}", n_use);
    
    // Store timestamp in both m_tlist (for compatibility) and m_pg.m_tstamps (main storage)
    m_tlist.push_back(timestamp);
    m_pg.m_tstamps[n_use] = timestamp;
    
    // Store timestamp in historical buffer for mapping sliding window to global indices
    if (static_cast<int>(m_allTimestamps.size()) <= m_counter) {
        m_allTimestamps.resize(m_counter + 1);
    }
    m_allTimestamps[m_counter] = timestamp;

    // Store camera intrinsics (Python divides by RES=4)
    // Use intrinsics_in if provided, otherwise use stored m_intrinsics
    const float* intrinsics_to_use = (intrinsics_in != nullptr) ? intrinsics_in : m_intrinsics;
    
    const float RES = 4.0f;
    float scaled_intrinsics[4];
    for (int i = 0; i < 4; i++) {
        scaled_intrinsics[i] = intrinsics_to_use[i] / RES;
    }
    std::memcpy(m_pg.m_intrinsics[n_use], scaled_intrinsics, sizeof(float) * 4);
    
    if (logger) {
        logger->info("DPVO::runAfterPatchify: Intrinsics - input: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}, "
                     "scaled (stored): fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
                     intrinsics_to_use[0], intrinsics_to_use[1], intrinsics_to_use[2], intrinsics_to_use[3],
                     scaled_intrinsics[0], scaled_intrinsics[1], scaled_intrinsics[2], scaled_intrinsics[3]);
    }
    
    if (logger) logger->info("DPVO::runAfterPatchify: Bookkeeping completed");

    // -------------------------------------------------
    // 3. Pose initialization (with motion model support)
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Starting pose initialization");
    
    // Initialize pose for this frame
    if (n_use == 0) {
        // First frame: use identity pose (origin, no rotation)
        m_pg.m_poses[n_use] = SE3();
        if (logger) logger->info("DPVO::runAfterPatchify: Initialized first frame pose to identity");
    } else if (n_use == 1) {
        // Second frame: copy first frame pose (no motion initially)
        m_pg.m_poses[n_use] = m_pg.m_poses[n_use - 1];
        if (logger) logger->info("DPVO::runAfterPatchify: Initialized second frame pose from first frame");
    } else {
        // Subsequent frames: use previous frame pose (will be updated by bundle adjustment)
        m_pg.m_poses[n_use] = m_pg.m_poses[n_use - 1];
        if (logger && n_use < 5) {
            Eigen::Vector3f t = m_pg.m_poses[n_use].t;
            logger->info("DPVO::runAfterPatchify: Initialized frame {} pose from previous frame, t=({:.3f}, {:.3f}, {:.3f})",
                         n_use, t.x(), t.y(), t.z());
        }
    }
    
    // Validate pose: check if translation is reasonable (prevent divergence)
    Eigen::Vector3f t = m_pg.m_poses[n_use].t;
    float t_norm = t.norm();
    if (t_norm > 100.0f) {
        if (logger) {
            logger->warn("DPVO::runAfterPatchify: Pose translation too large (norm={:.2f}), resetting to identity. "
                         "This indicates pose divergence. t=({:.2f}, {:.2f}, {:.2f})",
                         t_norm, t.x(), t.y(), t.z());
        }
        // Reset to identity if pose has diverged too much
        m_pg.m_poses[n_use] = SE3();
    }
    
    // Store pose in historical buffer for visualization
    if (static_cast<int>(m_allPoses.size()) <= m_counter) {
        m_allPoses.resize(m_counter + 1);
    }
    m_allPoses[m_counter] = m_pg.m_poses[n_use];
    
    if (logger) logger->info("DPVO::runAfterPatchify: Pose initialization completed");

    // -------------------------------------------------
    // 4. Patch depth initialization
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Starting patch depth initialization, m_is_initialized={}", m_is_initialized);
    
    float depth_value = 1.0f;
    
    if (m_is_initialized && n_use >= 3) 
    {
        std::vector<float> depths;
        for (int f = std::max(0, n_use - 3); f < n_use; f++) {
            for (int i = 0; i < M; i++) 
            {
                int center_y = P / 2;
                int center_x = P / 2;
                float d = m_pg.m_patches[f][i][2][center_y][center_x];
                if (d > 0.0f) 
                {
                    depths.push_back(d);
                }
            }
        }
        if (!depths.empty()) {
            std::sort(depths.begin(), depths.end());
            depth_value = depths[depths.size() / 2];
            if (logger) logger->info("DPVO::runAfterPatchify: Using median depth from last 3 frames: {}", depth_value);
        }
    } else {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.1f, 1.0f);
        depth_value = dis(gen);
        if (logger) logger->info("DPVO::runAfterPatchify: Using random depth initialization: {}", depth_value);
    }
    
    // Initialize all patches with the computed depth value
    for (int i = 0; i < M; i++) {
        int base = (i * 3 + 2) * patch_D * patch_D;
        int center_offset = (patch_D - P) / 2;
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                int patch_idx = base + (center_offset + y) * patch_D + (center_offset + x);
                patches[patch_idx] = depth_value;
            }
        }
    }
    if (logger) logger->info("DPVO::runAfterPatchify: Patch depth initialization completed");

    // -------------------------------------------------
    // 5. Store patches + colors into PatchGraph
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Starting store patches, n_use={}, M={}, P={}, patch_D={}", n_use, M, P, patch_D);
    
    const std::vector<float>& patch_coords = m_patchifier.getLastCoords();
    
    if (logger && patch_coords.size() >= M * 2) {
        logger->info("DPVO::runAfterPatchify: Patch coordinates from patchifier (full res, first 3): "
                     "patch[0]=(%.2f, %.2f), patch[1]=(%.2f, %.2f), patch[2]=(%.2f, %.2f)",
                     patch_coords[0], patch_coords[1],
                     patch_coords[2], patch_coords[3],
                     patch_coords[4], patch_coords[5]);
    }
    
    int center_offset = (patch_D - P) / 2;
    for (int i = 0; i < M; i++) {
        if (i * 2 + 1 >= static_cast<int>(patch_coords.size())) {
            if (logger) logger->error("DPVO::runAfterPatchify: Invalid coordinate index for patch {}", i);
            continue;
        }
        
        float px_center_full = patch_coords[i * 2 + 0];
        float py_center_full = patch_coords[i * 2 + 1];
        
        if (px_center_full < 0 || px_center_full >= W || py_center_full < 0 || py_center_full >= H) {
            if (logger) logger->warn("DPVO::runAfterPatchify: Patch[{}] has invalid coordinates: ({:.2f}, {:.2f}), image size=({}, {})",
                                     i, px_center_full, py_center_full, W, H);
            px_center_full = std::max(0.0f, std::min(px_center_full, static_cast<float>(W - 1)));
            py_center_full = std::max(0.0f, std::min(py_center_full, static_cast<float>(H - 1)));
        }
        
        if (logger && i < 3) {
            logger->info("DPVO::runAfterPatchify: Storing patch[{}]: center_full_res=({:.2f}, {:.2f})",
                         i, px_center_full, py_center_full);
        }
        
        // CRITICAL FIX: Store actual pixel coordinates for each pixel in the patch
        // patches array from patchify_cpu_safe contains [M, 3, patch_D, patch_D]
        // Channel 0: x coordinates, Channel 1: y coordinates, Channel 2: RGB values
        // We need to extract the center P×P region and scale coordinates by RES
        for (int y = 0; y < P; y++) {
            for (int x = 0; x < P; x++) {
                // Extract pixel coordinates from patches array (at full resolution)
                // patches layout: [M, 3, patch_D, patch_D]
                int patch_x_idx = (i * 3 + 0) * patch_D * patch_D + (center_offset + y) * patch_D + (center_offset + x);
                int patch_y_idx = (i * 3 + 1) * patch_D * patch_D + (center_offset + y) * patch_D + (center_offset + x);
                int patch_d_idx = (i * 3 + 2) * patch_D * patch_D + (center_offset + y) * patch_D + (center_offset + x);
                
                // Get pixel coordinates from patches (full resolution)
                float px_pixel_full = patches[patch_x_idx];
                float py_pixel_full = patches[patch_y_idx];
                
                // Scale to 1/4 resolution (matching intrinsics)
                float px_pixel_scaled = px_pixel_full / RES;
                float py_pixel_scaled = py_pixel_full / RES;
                
                // Store pixel coordinates and inverse depth
                m_pg.m_patches[n_use][i][0][y][x] = px_pixel_scaled;
                m_pg.m_patches[n_use][i][1][y][x] = py_pixel_scaled;
                m_pg.m_patches[n_use][i][2][y][x] = patches[patch_d_idx];  // Inverse depth (from RGB channel, will be overwritten)
                
                // Diagnostic: Log first pixel coordinates for first patch
                if (logger && i == 0 && y == 0 && x == 0) {
                    logger->info("DPVO::runAfterPatchify: Patch[0] pixel[0][0] - full=({:.2f}, {:.2f}), scaled=({:.2f}, {:.2f})",
                                 px_pixel_full, py_pixel_full, px_pixel_scaled, py_pixel_scaled);
                }
            }
        }
        
        for (int c = 0; c < 3; c++)
            m_pg.m_colors[n_use][i][c] = clr[i * 3 + c];
    }
    if (logger) logger->info("DPVO::runAfterPatchify: Store patches completed");

    // -------------------------------------------------
    // 6. Downsample fmap1 → fmap2
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Starting downsample fmap1->fmap2, fmap1_H={}, fmap1_W={}, fmap2_H={}, fmap2_W={}", 
                              m_fmap1_H, m_fmap1_W, m_fmap2_H, m_fmap2_W);
    for (int c = 0; c < 128; c++) {
        for (int y = 0; y < m_fmap2_H; y++) {
            for (int x = 0; x < m_fmap2_W; x++) {
                float sum = 0.0f;
                for (int dy = 0; dy < 4; dy++)
                    for (int dx = 0; dx < 4; dx++)
                        sum += m_cur_fmap1[c * m_fmap1_H * m_fmap1_W +
                            (y * 4 + dy) * m_fmap1_W +
                            (x * 4 + dx)];
                m_fmap2[fmap2_idx(0, mm, c, y, x)] = sum / 16.0f;
            }
        }
    }
    if (logger) logger->info("DPVO::runAfterPatchify: Downsample fmap1->fmap2 completed");

    // -------------------------------------------------
    // 7. Motion probe check
    // -------------------------------------------------
    if (n_use > 0 && !m_is_initialized) {
        if (logger) logger->info("DPVO::runAfterPatchify: Running motion probe check before initialization");
        float motion_val = motionProbe();
        if (motion_val < 2.0f) {
            if (logger) logger->info("DPVO::runAfterPatchify: Motion probe returned {} < 2.0, skipping frame", motion_val);
            return;
        }
        if (logger) logger->info("DPVO::runAfterPatchify: Motion probe returned {} >= 2.0, proceeding", motion_val);
    }

    // -------------------------------------------------
    // 8. Counters
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Updating counters");
    try {
        m_pg.m_n = n_use + 1;
        m_pg.m_m += M;
        m_counter++;
        if (logger) logger->info("DPVO::runAfterPatchify: Counters updated, m_n={} (current window size), m_m={}, m_counter={} (total frames processed)", 
                                 m_pg.m_n, m_pg.m_m, m_counter);
    } catch (...) {
        fprintf(stderr, "[DPVO] EXCEPTION updating counters, m_pg might be corrupted\n");
        fflush(stderr);
    }

    // -------------------------------------------------
    // 9. Build edges
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Starting build edges");
    std::vector<int> kk, jj;
    edgesForward(kk, jj);
    if (logger) logger->info("DPVO::runAfterPatchify: edgesForward completed, kk.size()={}, jj.size()={}", kk.size(), jj.size());
    appendFactors(kk, jj);
    if (logger) logger->info("DPVO::runAfterPatchify: appendFactors (forward) completed");
    edgesBackward(kk, jj);
    if (logger) logger->info("DPVO::runAfterPatchify: edgesBackward completed, kk.size()={}, jj.size()={}", kk.size(), jj.size());
    appendFactors(kk, jj);
    if (logger) logger->info("DPVO::runAfterPatchify: appendFactors (backward) completed");

    // -------------------------------------------------
    // 10. Optimization
    // -------------------------------------------------
    if (logger) logger->info("DPVO::runAfterPatchify: Starting optimization, m_is_initialized={}, m_n={}", m_is_initialized, m_pg.m_n);
    if (m_is_initialized) {
        if (logger) logger->info("DPVO::runAfterPatchify: Calling update()");
        update();
        if (logger) logger->info("DPVO::runAfterPatchify: update() completed");
        int m_n_before_keyframe = m_pg.m_n;
        if (logger) logger->info("DPVO::runAfterPatchify: Calling keyframe(), m_pg.m_n={} (before keyframe)", m_pg.m_n);
        keyframe();
        int m_n_after_keyframe = m_pg.m_n;
        if (logger) logger->info("DPVO::runAfterPatchify: keyframe() completed, m_pg.m_n={}→{} (before→after keyframe), m_counter={} (total frames processed)", 
                                 m_n_before_keyframe, m_n_after_keyframe, m_counter);
        
        // Update viewer after optimization
        if (m_visualizationEnabled) {
            updateViewer();
            if (m_viewer != nullptr && image_for_viewer != nullptr) {
                // Image is in [C, H, W] format (RGB), convert to [H, W, C] for viewer
                std::vector<uint8_t> image_rgb(H * W * 3);
                for (int c = 0; c < 3; c++) {
                    for (int h = 0; h < H; h++) {
                        for (int w = 0; w < W; w++) {
                            int src_idx = c * H * W + h * W + w;
                            int dst_idx = h * W * 3 + w * 3 + c;
                            if (src_idx >= 0 && src_idx < H * W * 3 && 
                                dst_idx >= 0 && dst_idx < H * W * 3) {
                                image_rgb[dst_idx] = image_for_viewer[src_idx];
                            }
                        }
                    }
                }
                m_viewer->updateImage(image_rgb.data(), W, H);
                if (logger) {
                    logger->info("Viewer: Updated image {}x{}", W, H);
                }
            }
        }
    } else if (m_pg.m_n >= 1) {
        if (logger) logger->info("DPVO::runAfterPatchify: Initializing with 12 update() calls");
        m_is_initialized = true;
        for (int i = 0; i < 1; i++) {
            if (logger) logger->info("DPVO::runAfterPatchify: Initialization update() call {}/12", i+1);
            update();
            // Update viewer after optimization
            if (m_visualizationEnabled) {
                updateViewer();
                if (m_viewer != nullptr && image_for_viewer != nullptr) {
                    // Image is in [C, H, W] format (RGB), convert to [H, W, C] for viewer
                    std::vector<uint8_t> image_rgb(H * W * 3);
                    for (int c = 0; c < 3; c++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                int src_idx = c * H * W + h * W + w;
                                int dst_idx = h * W * 3 + w * 3 + c;
                                if (src_idx >= 0 && src_idx < H * W * 3 && 
                                    dst_idx >= 0 && dst_idx < H * W * 3) {
                                    image_rgb[dst_idx] = image_for_viewer[src_idx];
                                }
                            }
                        }
                    }
                    m_viewer->updateImage(image_rgb.data(), W, H);
                    if (logger) {
                        logger->info("Viewer: Updated image {}x{}", W, H);
                    }
                }
            }
        }
        if (logger) logger->info("DPVO::runAfterPatchify: Initialization completed");
    }
    if (logger) logger->info("DPVO::runAfterPatchify: Optimization completed");
}

#if defined(CV28) || defined(CV28_SIMULATOR)
// Tensor-based overload of run() - uses tensor directly, avoids conversion
void DPVO::run(int64_t timestamp, ea_tensor_t* imgTensor, const float* intrinsics_in)
{   

    // Use both stdout and stderr to ensure output is visible
    fprintf(stderr, "[DPVO] ================ ENTERING run() ========================\n");
    fprintf(stderr, "[DPVO] DEBUG: input is ea_tensor_t* imgTensor: %p\n", (void*)this);
    printf("[DPVO] ==================== ENTERING run() =============================\n");
    fflush(stdout);
    fflush(stderr);
    if (imgTensor == nullptr) {
        throw std::runtime_error("Null tensor pointer passed to DPVO::run");
    }
    
    // Get dimensions from tensor
    const size_t* shape = ea_tensor_shape(imgTensor);
    int H = static_cast<int>(shape[EA_H]);
    int W = static_cast<int>(shape[EA_W]);
    
    auto logger = spdlog::get("dpvo");
    if (logger) {
        logger->info("DPVO::run (tensor): H={}, W={}", H, W);
    }
    
    // Use intrinsics_in if provided, otherwise use stored m_intrinsics
    const float* intrinsics = (intrinsics_in != nullptr) ? intrinsics_in : m_intrinsics;
    
    // Store timestamp
    m_currentTimestamp = timestamp;
    
    // Validate and get n (same logic as uint8_t* version)
    int n = 0;
    if (m_pg.m_n < 0 || m_pg.m_n >= PatchGraph::N || m_pg.m_n > 1000) {
        try {
            m_pg.reset();
            if (m_pg.m_n < 0 || m_pg.m_n >= PatchGraph::N || m_pg.m_n > 1000) {
                m_pg.m_n = 0;
                m_pg.m_m = 0;
            }
            n = m_pg.m_n;
        } catch (...) {
            n = 0;
        }
    } else {
        n = m_pg.m_n;
    }
    
    if (n + 1 >= PatchGraph::N)
        throw std::runtime_error("PatchGraph buffer overflow");
    
    const int pm = n % m_pmem;
    const int mm = n % m_mem;
    const int M  = m_cfg.PATCHES_PER_FRAME;
    const int P  = m_P;
    
    // Set up pointers (same as uint8_t* version)
    m_cur_imap  = &m_imap[imap_idx(pm, 0, 0)];
    m_cur_gmap  = &m_gmap[gmap_idx(pm, 0, 0, 0, 0)];
    m_cur_fmap1 = &m_fmap1[fmap1_idx(0, mm, 0, 0, 0)];
    
    // Allocate patches and color buffers (same as uint8_t* version)
    const int patch_radius = m_P / 2;
    const int patch_D = 2 * patch_radius + 1;
    const int patches_size = M * 3 * patch_D * patch_D;
    std::vector<float> patches_vec(patches_size);
    float* patches = patches_vec.data();
    uint8_t clr[M * 3];
    
    // Use tensor-based patchifier.forward() directly (avoids conversion)
    // This will use tensor-based fnet/inet.runInference() which avoids uint8_t* conversion
    if (logger) logger->info("DPVO::run (tensor): Starting patchifier.forward");
    m_patchifier.forward(
        imgTensor,      // Use tensor directly
        m_cur_fmap1,
        m_cur_imap,
        m_cur_gmap,
        patches,
        clr,
        M
    );
    if (logger) logger->info("DPVO::run (tensor): patchifier.forward completed");
    
    // Validate n_use (same as uint8_t* version)
    int n_use = n;
    if (n_use < 0 || n_use >= PatchGraph::N || n_use > 1000) {
        if (logger) logger->warn("DPVO::run (tensor): n={} is corrupted! Using n_use=0 instead.", n);
        n_use = 0;
    }
    
    // Extract image data only for viewer update (if needed)
    std::vector<uint8_t> image_data;
    const uint8_t* image_for_viewer = nullptr;
    if (m_visualizationEnabled) {
        image_data.resize(H * W * 3);
        void* tensor_data = ea_tensor_data(imgTensor);
        if (tensor_data != nullptr) {
            const uint8_t* src = static_cast<const uint8_t*>(tensor_data);
            memcpy(image_data.data(), src, H * W * 3);
            image_for_viewer = image_data.data();
        }
    }
    
    // Call helper function to continue with rest of logic after patchifier.forward()
    // This avoids calling patchifier.forward() again
    runAfterPatchify(timestamp, intrinsics, H, W, n, n_use, pm, mm, M, P, patch_D, patches, clr, image_for_viewer);
    
    if (logger) logger->info("DPVO::run (tensor): Completed");
}
#endif

// -------------------------------------------------------------
// Update (NN + BA stub)
// -------------------------------------------------------------
// void DPVO::update() {
//     if (m_pg.m_num_edges == 0) return;

//     // NN update + reprojection will go here
//     // BA_CV28(...) will go here
// }


void DPVO::update()
{
    const int num_active = m_pg.m_num_edges;
    if (num_active == 0)
        return;

    const int M = m_cfg.PATCHES_PER_FRAME;
    const int P = m_P;

    // -------------------------------------------------
    // 1. Reprojection
    // -------------------------------------------------
    std::vector<float> coords(num_active * 2 * P * P); // [num_active, 2, P, P]
    reproject(
        m_pg.m_ii, m_pg.m_jj, m_pg.m_kk,
        num_active,
        coords.data()
    );

    // -------------------------------------------------
    // 2. Correlation
    // -------------------------------------------------
    // CRITICAL: Pass full buffers (m_fmap1, m_fmap2), not single-frame pointers (m_cur_fmap1)
    // computeCorrelation needs to access multiple frames based on jj[e] indices
    // Correlation output shape: [num_active, D, D, P, P, 2] where D = 2*R + 2 = 8 (R=3)
    const int R = 3;  // Correlation radius
    const int D = 2 * R + 2;  // Correlation window diameter (D = 8)
    std::vector<float> corr(num_active * D * D * P * P * 2); // [num_active, D, D, P, P, 2] (channel last)
    
    auto logger = spdlog::get("dpvo");
    if (logger) {
        logger->info("DPVO::update: Starting correlation, num_active={}, M={}, P={}, D={}, m_mem={}, m_pmem={}", 
                     num_active, M, P, D, m_mem, m_pmem);
        logger->info("DPVO::update: fmap1 dimensions: {}x{}, fmap2 dimensions: {}x{}", 
                     m_fmap1_H, m_fmap1_W, m_fmap2_H, m_fmap2_W);
    }
    
    if (logger) {
        logger->info("DPVO::update: About to call computeCorrelation");
        logger->info("DPVO::update: Sample indices - kk[0]={}, jj[0]={}, ii[0]={} (first 5 edges)",
                     num_active > 0 ? m_pg.m_kk[0] : -1,
                     num_active > 0 ? m_pg.m_jj[0] : -1,
                     num_active > 0 ? m_pg.m_ii[0] : -1);
    }
    
    printf("[DPVO::update] About to call computeCorrelation, num_active=%d\n", num_active);
    fflush(stdout);
    
    computeCorrelation(
		m_gmap,
		m_fmap1,          // pyramid0 - full buffer [m_mem][128][fmap1_H][fmap1_W]
		m_fmap2,          // pyramid1 - full buffer [m_mem][128][fmap2_H][fmap2_W]
		coords.data(),
		m_pg.m_ii,        // ii - patch indices (within frame) - NOTE: Not used in computeCorrelation, but kept for compatibility
		m_pg.m_jj,        // jj - frame indices (for pyramid/target frame)
		m_pg.m_kk,        // kk - linear patch indices (frame * M + patch, for gmap source frame) - THIS IS USED
		num_active,
		M,
		P,
		m_mem,            // num_frames - number of frames in pyramid buffers
		m_pmem,           // num_gmap_frames - number of frames in gmap ring buffer
		m_fmap1_H, m_fmap1_W,  // Dimensions for pyramid0 (fmap1)
		m_fmap2_H, m_fmap2_W,  // Dimensions for pyramid1 (fmap2) - CRITICAL: different from fmap1!
		128,
		corr.data()
	);
    
    printf("[DPVO::update] computeCorrelation returned\n");
    fflush(stdout);
    
    if (logger) logger->info("DPVO::update: Correlation completed");


    // -------------------------------------------------
    // 3. Context slice from imap
    // -------------------------------------------------
    if (logger) logger->info("DPVO::update: Starting context slice from imap, num_active={}, m_DIM={}", num_active, m_DIM);
    
    // Check m_imap buffer statistics before slicing
    if (logger) {
        // Sample first few entries to check if m_imap is populated
        float imap_sample_min = std::numeric_limits<float>::max();
        float imap_sample_max = std::numeric_limits<float>::lowest();
        int imap_sample_zero = 0;
        int imap_sample_nonzero = 0;
        int sample_size = std::min(100, M * m_pmem * m_DIM);
        for (int i = 0; i < sample_size; i++) {
            float val = m_imap[i];
            if (val == 0.0f) imap_sample_zero++;
            else imap_sample_nonzero++;
            if (val < imap_sample_min) imap_sample_min = val;
            if (val > imap_sample_max) imap_sample_max = val;
        }
        logger->info("DPVO::update: m_imap buffer sample stats (first {} elements) - zero_count={}, nonzero_count={}, min={}, max={}",
                     sample_size, imap_sample_zero, imap_sample_nonzero, imap_sample_min, imap_sample_max);
        
        // Also check the most recent frame (should be frame 8 if n=9, so pm=8)
        int current_n = m_pg.m_n;
        int current_pm = (current_n - 1) % m_pmem;  // Most recently written frame
        int current_imap_offset = imap_idx(current_pm, 0, 0);
        float current_frame_sample = m_imap[current_imap_offset];
        logger->info("DPVO::update: Most recent frame n={}, pm={}, imap_offset={}, m_imap[{}]={}",
                     current_n, current_pm, current_imap_offset, current_imap_offset, current_frame_sample);
        
        // Check all frames in ring buffer
        int frames_with_data = 0;
        int frames_without_data = 0;
        for (int f = 0; f < m_pmem; f++) {
            int frame_offset = imap_idx(f, 0, 0);
            float frame_sample = m_imap[frame_offset];
            if (frame_sample != 0.0f) frames_with_data++;
            else frames_without_data++;
        }
        logger->info("DPVO::update: Ring buffer status - frames_with_data={}, frames_without_data={} (out of {} slots)",
                     frames_with_data, frames_without_data, m_pmem);
    }
    
    std::vector<float> ctx(num_active * m_DIM);
    for (int e = 0; e < num_active; e++) {
        // CRITICAL: kk is a linear patch index: kk = frame * M + patch
        // m_imap is a ring buffer indexed as [frame % m_pmem][patch][dim]
        // So we need to convert kk to (frame, patch) and then index correctly
        int kk_val = m_pg.m_kk[e];
        int frame = kk_val / M;  // Extract frame from linear index
        int patch = kk_val % M;  // Extract patch from linear index
        
        // Convert to ring buffer index: frame % m_pmem
        int imap_frame = frame % m_pmem;
        
        // Validate indices
        if (frame < 0 || patch < 0 || patch >= M) {
            if (logger && e < 10) logger->error("DPVO::update: Invalid kk={} -> frame={}, patch={} for edge e={}, M={}", 
                                                kk_val, frame, patch, e, M);
            // Fallback to frame 0, patch 0
            imap_frame = 0;
            patch = 0;
        }
        
        // Use imap_idx to get the correct offset: [imap_frame][patch][0]
        int imap_offset = imap_idx(imap_frame, patch, 0);
        
        // Validate source pointer before memcpy
        if (logger && e < 5) {
            float src_sample = m_imap[imap_offset];
            logger->info("DPVO::update: Edge e={}, kk={} -> frame={}, patch={}, imap_frame={}, imap_offset={}, m_imap[{}]={}", 
                         e, kk_val, frame, patch, imap_frame, imap_offset, imap_offset, src_sample);
        }
        
        std::memcpy(&ctx[e * m_DIM],&m_imap[imap_offset],sizeof(float) * m_DIM);
    }
    
    // Check ctx statistics after slicing
    if (logger) {
        float ctx_min = *std::min_element(ctx.begin(), ctx.end());
        float ctx_max = *std::max_element(ctx.begin(), ctx.end());
        int ctx_zero_count = 0;
        int ctx_nonzero_count = 0;
        for (size_t i = 0; i < ctx.size(); i++) {
            if (ctx[i] == 0.0f) ctx_zero_count++;
            else ctx_nonzero_count++;
        }
        logger->info("DPVO::update: Context (ctx) stats after slicing - zero_count={}, nonzero_count={}, min={}, max={}, size={}",
                     ctx_zero_count, ctx_nonzero_count, ctx_min, ctx_max, ctx.size());
    }
    
    if (logger) logger->info("DPVO::update: Context slice completed");

    // -------------------------------------------------
    // 4. Network update (DPVO Update Model Inference)
    // -------------------------------------------------
    if (logger) logger->info("DPVO::update: Starting network update, m_updateModel={}", (void*)m_updateModel.get());
    std::vector<float> delta(num_active * 2);
    std::vector<float> weight(num_active);
    int num_edges_to_process = 0;  // Declare outside if block for use later

    if (m_updateModel != nullptr) {
        if (logger) logger->info("DPVO::update: m_updateModel is not null, preparing model inputs");
        
        // Check m_pg.m_net state before reshapeInput
        if (logger) {
            int net_zero_count = 0;
            int net_nonzero_count = 0;
            for (int e = 0; e < std::min(num_active, 10); e++) {  // Check first 10 edges
                for (int d = 0; d < 384; d++) {
                    if (m_pg.m_net[e][d] == 0.0f) net_zero_count++;
                    else net_nonzero_count++;
                }
            }
            logger->info("DPVO::update: m_pg.m_net state (first 10 edges) - zero_count={}, nonzero_count={}",
                         net_zero_count, net_nonzero_count);
            if (net_nonzero_count == 0) {
                logger->info("DPVO::update: m_pg.m_net is all zeros - will be initialized from context in reshapeInput");
            }
        }
        
        // Reshape inputs using member function (reuses pre-allocated buffers)
        const int CORR_DIM = 882;
        num_edges_to_process = m_updateModel->reshapeInput(
            num_active,
            m_pg.m_net,  // Pointer to 2D array [MAX_EDGES][384]
            ctx.data(),  // Context data [num_active * 384]
            corr,        // Correlation data [num_active * D * D * P * P * 2]
            m_pg.m_ii,   // Indices [num_active]
            m_pg.m_jj,   // Indices [num_active]
            m_pg.m_kk,   // Indices [num_active]
            D,           // Correlation window size (typically 8)
            P,           // Patch size (typically 3)
            m_reshape_net_input,   // Pre-allocated output buffers
            m_reshape_inp_input,
            m_reshape_corr_input,
            m_reshape_ii_input,
            m_reshape_jj_input,
            m_reshape_kk_input,
            m_maxEdge,   // Use member variable instead of hardcoded constant
            CORR_DIM
        );
        
        // Check m_pg.m_net state after reshapeInput (to see if it was initialized)
        if (logger) {
            int net_zero_count = 0;
            int net_nonzero_count = 0;
            float net_min = std::numeric_limits<float>::max();
            float net_max = std::numeric_limits<float>::lowest();
            for (int e = 0; e < std::min(num_edges_to_process, 10); e++) {
                for (int d = 0; d < 384; d++) {
                    float val = m_pg.m_net[e][d];
                    if (val == 0.0f) net_zero_count++;
                    else net_nonzero_count++;
                    if (val < net_min) net_min = val;
                    if (val > net_max) net_max = val;
                }
            }
            logger->info("DPVO::update: m_pg.m_net state AFTER reshapeInput (first {} edges) - zero_count={}, nonzero_count={}, min={:.6f}, max={:.6f}",
                         std::min(num_edges_to_process, 10), net_zero_count, net_nonzero_count, net_min, net_max);
        }
        
        // Call update model inference synchronously
        DPVOUpdate_Prediction pred;
        if (logger) {
            logger->info("DPVO::update: About to call m_updateModel->runInference");
            logger->info("DPVO::update: Input data ready - net_input size={}, inp_input size={}, corr_input size={}",
                         m_reshape_net_input.size(), m_reshape_inp_input.size(), m_reshape_corr_input.size());
        }
        bool inference_success = m_updateModel->runInference(
                m_reshape_net_input.data(),
                m_reshape_inp_input.data(),
                m_reshape_corr_input.data(),
                m_reshape_ii_input.data(),
                m_reshape_jj_input.data(),
                m_reshape_kk_input.data(),
                m_updateFrameCounter++,
                pred);
        
        if (logger) {
            logger->info("DPVO::update: runInference returned: {}", inference_success);
        }
        
        if (inference_success)
        {
            if (logger) logger->info("DPVO::update: runInference returned true, extracting outputs");
            // Extract outputs: net_out [1, 384, 384, 1], d_out [1, 2, 384, 1], w_out [1, 2, 384, 1]
            // d_out contains delta: [1, 2, 384, 1] -> [num_edges, 2]
            // w_out contains weight: [1, 2, 384, 1] -> we'll use first channel
            
            if (pred.dOutBuff != nullptr && pred.wOutBuff != nullptr) {
                // Extract delta from d_out: YAML layout [N, C, H, W] = [1, 2, m_maxEdge, 1]
                int zero_weight_count = 0;
                for (int e = 0; e < num_edges_to_process; e++) {
                    // d_out layout: [N, C, H, W] = [1, 2, m_maxEdge, 1]
                    // Index: n * C * H * W + c * H * W + h * W + w
                    // Where: n=0, c=0 or 1, h=e, w=0
                    int idx0 = 0 * 2 * m_maxEdge * 1 + 0 * m_maxEdge * 1 + e * 1 + 0;
                    int idx1 = 0 * 2 * m_maxEdge * 1 + 1 * m_maxEdge * 1 + e * 1 + 0;
                    delta[e * 2 + 0] = pred.dOutBuff[idx0];
                    delta[e * 2 + 1] = pred.dOutBuff[idx1];
                    
                    // w_out layout: [1, 2, 384, 1] - try both channels to see which has data
                    // Channel 0 (same as delta x)
                    float w0 = pred.wOutBuff[idx0];
                    // Channel 1 (same as delta y)
                    float w1 = pred.wOutBuff[idx1];
                    
                    // Use channel 0 for now, but log both to debug
                    weight[e] = w0;
                    
                    if (weight[e] <= 0.0f) {
                        zero_weight_count++;
                    }
                    
                    // Debug: Log first few weights to see what we're getting
                    if (logger && e < 3) {
                        logger->debug("DPVO::update: Weight extraction for edge[{}]: idx0={}, idx1={}, w0={:.6f}, w1={:.6f}, "
                                     "wOutBuff[idx0]={:.6f}, wOutBuff[idx1]={:.6f}",
                                     e, idx0, idx1, w0, w1,
                                     pred.wOutBuff[idx0], pred.wOutBuff[idx1]);
                    }
                }
                
                // If all or most weights are zero, set random weights immediately
                if (zero_weight_count == num_edges_to_process) {
                    // All weights are zero - use random weights for testing
                    if (logger) {
                        logger->warn("DPVO::update: All {} edges have zero weight (model issue). "
                                    "Setting random weights [0.1, 0.5] immediately.",
                                    num_edges_to_process);
                    }
                    static thread_local std::random_device rd;
                    static thread_local std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> dis(0.1f, 0.5f);
                    
                    for (int e = 0; e < num_edges_to_process; e++) {
                        weight[e] = dis(gen);  // Random weight between 0.1 and 0.5
                    }
                } else if (zero_weight_count >= num_edges_to_process * 0.90) {
                    // 50% or more are zero - use random weights for the zero ones
                    if (logger) {
                        logger->warn("DPVO::update: {} out of {} edges have zero weight (>=90%). "
                                    "Setting random weights [0.1, 0.5] for zero-weight edges.",
                                    zero_weight_count, num_edges_to_process);
                    }
                    static thread_local std::random_device rd;
                    static thread_local std::mt19937 gen(rd());
                    std::uniform_real_distribution<float> dis(0.1f, 0.5f);
                    
                    for (int e = 0; e < num_edges_to_process; e++) {
                        if (weight[e] <= 0.0f) {
                            weight[e] = dis(gen);
                        }
                    }
                }
                
                // Update m_pg.m_net with net_out if available
                if (pred.netOutBuff != nullptr) {
                    if (logger) logger->info("DPVO::update: Updating m_pg.m_net from net_out");
                    // net_out: YAML layout [N, C, H, W] = [1, 384, m_maxEdge, 1]
                    float net_out_min = std::numeric_limits<float>::max();
                    float net_out_max = std::numeric_limits<float>::lowest();
                    for (int e = 0; e < num_edges_to_process; e++) {
                        for (int d = 0; d < 384; d++) {
                            // Index: n * C * H * W + c * H * W + h * W + w
                            // Where: n=0, c=d, h=e, w=0
                            int idx = 0 * 384 * m_maxEdge * 1 + d * m_maxEdge * 1 + e * 1 + 0;
                            float val = pred.netOutBuff[idx];
                            m_pg.m_net[e][d] = val;
                            if (val < net_out_min) net_out_min = val;
                            if (val > net_out_max) net_out_max = val;
                        }
                    }
                    if (logger) {
                        logger->info("DPVO::update: net_out range: [{}, {}], updated m_pg.m_net for {} edges",
                                     net_out_min, net_out_max, num_edges_to_process);
                    }
                } else {
                    if (logger) logger->warn("DPVO::update: pred.netOutBuff is null - m_pg.m_net will not be updated");
                }
            }
            
            // Free prediction buffers
            if (pred.netOutBuff) delete[] pred.netOutBuff;
            if (pred.dOutBuff) delete[] pred.dOutBuff;
            if (pred.wOutBuff) delete[] pred.wOutBuff;
        } else {
            if (logger) {
                logger->warn("DPVO::update: runInference returned false - using zero delta/weight fallback");
                logger->warn("DPVO::update: This means m_pg.m_net will remain unchanged (may stay zero)");
            }
        }
        
        // If we have more edges than processed, use zero delta/weight for remaining
        for (int e = num_edges_to_process; e < num_active; e++) {
            delta[e * 2 + 0] = 0.0f;
            delta[e * 2 + 1] = 0.0f;
            weight[e] = 0.0f;
        }
    } else {
        // Fallback: zero delta and weight if no update model
        std::fill(delta.begin(), delta.end(), 0.0f);
        std::fill(weight.begin(), weight.end(), 0.0f);
    }

    // -------------------------------------------------
    // 5. Compute target positions
    // -------------------------------------------------
    if (logger) logger->info("DPVO::update: Computing target positions, num_active={}", num_active);
    int invalid_target_count = 0;
    for (int e = 0; e < num_active; e++) {
        // Validate edge index
        if (e < 0 || e >= num_active) {
            if (logger && e < 10) logger->error("DPVO::update: Invalid edge index e={} in target positions, num_active={}", e, num_active);
            continue;
        }
        // Get center pixel coordinates (i0=1, j0=1 for P=3)
        int center_i0 = P / 2;  // 1 for P=3
        int center_j0 = P / 2;  // 1 for P=3
        // coords layout: [num_active][2][P][P] flattened
        // For edge e, channel c (0=x, 1=y), pixel (i0, j0): coords[e * 2 * P * P + c * P * P + i0 * P + j0]
        int coord_x_idx = e * 2 * P * P + 0 * P * P + center_i0 * P + center_j0;
        int coord_y_idx = e * 2 * P * P + 1 * P * P + center_i0 * P + center_j0;
        float cx = coords[coord_x_idx];
        float cy = coords[coord_y_idx];
        
        // Validate coordinates (check for NaN/Inf)
        if (!std::isfinite(cx) || !std::isfinite(cy)) {
            if (logger && invalid_target_count < 5) {
                logger->warn("DPVO::update: Invalid coords for edge[{}]: cx={}, cy={}", e, cx, cy);
            }
            invalid_target_count++;
            // Use previous target or zero as fallback
            if (e > 0) {
                m_pg.m_target[e * 2 + 0] = m_pg.m_target[(e-1) * 2 + 0];
                m_pg.m_target[e * 2 + 1] = m_pg.m_target[(e-1) * 2 + 1];
            } else {
                m_pg.m_target[e * 2 + 0] = 0.0f;
                m_pg.m_target[e * 2 + 1] = 0.0f;
            }
            m_pg.m_weight[e] = 0.0f;  // Invalidate this edge
            continue;
        }
        
        // Validate delta (check for NaN/Inf)
        float dx = delta[e * 2 + 0];
        float dy = delta[e * 2 + 1];
        if (!std::isfinite(dx) || !std::isfinite(dy)) {
            if (logger && invalid_target_count < 5) {
                logger->warn("DPVO::update: Invalid delta for edge[{}]: dx={}, dy={}", e, dx, dy);
            }
            invalid_target_count++;
            dx = 0.0f;
            dy = 0.0f;
        }

        m_pg.m_target[e * 2 + 0] = cx + dx;
        m_pg.m_target[e * 2 + 1] = cy + dy;
        m_pg.m_weight[e] = weight[e];
        
        // Final validation of target
        if (!std::isfinite(m_pg.m_target[e * 2 + 0]) || !std::isfinite(m_pg.m_target[e * 2 + 1])) {
            if (logger && invalid_target_count < 5) {
                logger->warn("DPVO::update: Invalid target for edge[{}]: target=({}, {}), cx={}, cy={}, dx={}, dy={}", 
                            e, m_pg.m_target[e * 2 + 0], m_pg.m_target[e * 2 + 1], cx, cy, dx, dy);
            }
            invalid_target_count++;
            m_pg.m_target[e * 2 + 0] = cx;  // Fallback to coords only
            m_pg.m_target[e * 2 + 1] = cy;
            m_pg.m_weight[e] = 0.0f;  // Invalidate this edge
        }
    }
    
    if (logger && invalid_target_count > 0) {
        logger->warn("DPVO::update: {} out of {} edges have invalid targets (NaN/Inf)", invalid_target_count, num_active);
    }
    
    // TEMPORARY: If all weights are zero (model issue), use random weights for testing
    // This allows bundle adjustment to proceed and helps debug the system
    int zero_weight_count = 0;
    for (int e = 0; e < num_active; e++) {
        if (m_pg.m_weight[e] <= 0.0f) {
            zero_weight_count++;
        }
    }
    
    if (zero_weight_count == num_active) {
        // All weights are zero - use random weights for testing
        if (logger) {
            logger->warn("DPVO::update: All {} edges have zero weight (model issue). "
                        "Using random weights [0.1, 0.5] for testing to allow bundle adjustment.",
                        num_active);
        }
        // Initialize random number generator (use thread_local to avoid race conditions)
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.1f, 0.5f);
        
        for (int e = 0; e < num_active; e++) {
            m_pg.m_weight[e] = dis(gen);  // Random weight between 0.1 and 0.5
        }
    } else if (zero_weight_count >= num_active / 2) {
        // 50% or more are zero - use random weights for the zero ones
        if (logger) {
            logger->warn("DPVO::update: {} out of {} edges have zero weight (>=50%). "
                        "Using random weights [0.1, 0.5] for zero-weight edges.",
                        zero_weight_count, num_active);
        }
        static thread_local std::random_device rd;
        static thread_local std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.1f, 0.5f);
        
        for (int e = 0; e < num_active; e++) {
            if (m_pg.m_weight[e] <= 0.0f) {
                m_pg.m_weight[e] = dis(gen);  // Random weight for zero-weight edges
            }
        }
    } else {
        // Set minimum weight threshold to ensure edges contribute
        const float MIN_WEIGHT = 0.01f;
        for (int e = 0; e < num_active; e++) {
            if (m_pg.m_weight[e] > 0.0f && m_pg.m_weight[e] < MIN_WEIGHT) {
                m_pg.m_weight[e] = MIN_WEIGHT;
            }
        }
    }
    
    if (logger) {
        int nonzero_weights = 0;
        float weight_sum = 0.0f;
        for (int e = 0; e < num_active; e++) {
            if (m_pg.m_weight[e] > 0.0f) {
                nonzero_weights++;
                weight_sum += m_pg.m_weight[e];
            }
        }
        logger->info("DPVO::update: Target positions computed. Weight stats: nonzero={}/{}, mean={:.6f}",
                     nonzero_weights, num_active, nonzero_weights > 0 ? weight_sum / nonzero_weights : 0.0f);
    }

    // -------------------------------------------------
    // 6. Bundle Adjustment
    // -------------------------------------------------
    if (logger) logger->info("DPVO::update: Starting bundle adjustment");
    try {
        bundleAdjustment(1e-4f, 100.0f, false, 1);
        
        // Sync updated poses from sliding window to historical buffer
        // CRITICAL: Use timestamps to map sliding window indices to global frame indices
        // After keyframe() removes frames, sliding window indices don't directly map to global indices
        // We use timestamps to find which global frame each sliding window index corresponds to
        int synced_count = 0;
        for (int sw_idx = 0; sw_idx < m_pg.m_n; sw_idx++) {
            int64_t sw_timestamp = m_pg.m_tstamps[sw_idx];
            
            // Find the global frame index that matches this timestamp
            int global_idx = -1;
            for (int g_idx = 0; g_idx < static_cast<int>(m_allTimestamps.size()); g_idx++) {
                if (m_allTimestamps[g_idx] == sw_timestamp) {
                    global_idx = g_idx;
                    break;
                }
            }
            
            // Update the corresponding global pose if we found a match
            if (global_idx >= 0 && global_idx < static_cast<int>(m_allPoses.size())) {
                m_allPoses[global_idx] = m_pg.m_poses[sw_idx];
                synced_count++;
                if (logger && sw_idx < 3) {
                    logger->debug("BA: Synced pose sw_idx={} (ts={}) -> global_idx={}", 
                                  sw_idx, sw_timestamp, global_idx);
                }
            } else if (logger && sw_idx < 3) {
                logger->warn("BA: Could not find global frame for sw_idx={} (ts={}), m_allTimestamps.size()={}", 
                            sw_idx, sw_timestamp, m_allTimestamps.size());
            }
        }
        
        if (logger && synced_count < m_pg.m_n) {
            logger->warn("BA: Only synced {}/{} poses from sliding window to historical buffer", 
                        synced_count, m_pg.m_n);
        }
        
        if (logger) logger->info("DPVO::update: Bundle adjustment completed");
    } catch (const std::exception& e) {
        if (logger) logger->error("DPVO::update: Bundle adjustment exception: {}", e.what());
    } catch (...) {
        if (logger) logger->error("DPVO::update: Bundle adjustment unknown exception");
    }
    if (logger) logger->info("DPVO::update: update() completed successfully");

    // -------------------------------------------------
    // 7. Update point cloud and viewer
    // -------------------------------------------------
    // Compute point cloud from patches and poses
    if (m_visualizationEnabled) {
        computePointCloud();
        updateViewer();
    }
}



// -------------------------------------------------------------
// Keyframe logic (minimal, safe version)
// -------------------------------------------------------------
// void DPVO::keyframe() {
//     int i = m_pg.m_n - m_cfg.KEYFRAME_INDEX - 1;
//     int j = m_pg.m_n - m_cfg.KEYFRAME_INDEX + 1;

//     if (i < 0 || j < 0) return;

//     float m = motionMagnitude(i, j) + motionMagnitude(j, i);
//     if (0.5f * m < m_cfg.KEYFRAME_THRESH) {
//         int k = m_pg.m_n - m_cfg.KEYFRAME_INDEX;

//         // shift frames left
//         for (int f = k; f < m_pg.m_n - 1; f++) {
//             m_pg.m_tstamps[f] = m_pg.m_tstamps[f + 1];
//             m_pg.m_poses[f]   = m_pg.m_poses[f + 1];
//         }

//         m_pg.m_n--;
//         m_pg.m_m -= PatchGraph::M;
//     }
// }
void DPVO::keyframe() {

    int n = m_pg.m_n;

    int i = n - m_cfg.KEYFRAME_INDEX - 1;
    int j = n - m_cfg.KEYFRAME_INDEX + 1;
    if (i < 0 || j < 0) return;

    float m = motionMagnitude(i, j) + motionMagnitude(j, i);
    
    auto logger = spdlog::get("dpvo");
    if (logger) {
        logger->info("DPVO::keyframe: n={}, i={}, j={}, motion={}, threshold={}, will_remove={}", 
                     n, i, j, 0.5f * m, m_cfg.KEYFRAME_THRESH, (0.5f * m < m_cfg.KEYFRAME_THRESH));
    }

    // =============================================================
    // Phase A: Keyframe removal decision
    // =============================================================
    if (0.5f * m < m_cfg.KEYFRAME_THRESH) {
        int k = n - m_cfg.KEYFRAME_INDEX;
        if (logger) {
            logger->info("DPVO::keyframe: Removing keyframe k={}, m_pg.m_n will decrease from {} to {}", 
                         k, n, n - 1);
        }

        // ---------------------------------------------------------
        // Phase B1: remove edges touching frame k
        // ---------------------------------------------------------
        bool remove[MAX_EDGES] = {false};

        int num_active = m_pg.m_num_edges;
        for (int e = 0; e < num_active; e++) {
            if (m_pg.m_ii[e] == k || m_pg.m_jj[e] == k)
                remove[e] = true;
        }

        removeFactors(remove, /*store=*/false);

        // ---------------------------------------------------------
        // Phase B2: reindex remaining edges
        // ---------------------------------------------------------
        num_active = m_pg.m_num_edges;
        for (int e = 0; e < num_active; e++) {

            if (m_pg.m_ii[e] > k) {
                m_pg.m_ii[e] -= 1;
                m_pg.m_kk[e] -= PatchGraph::M;
            }

            if (m_pg.m_jj[e] > k) {
                m_pg.m_jj[e] -= 1;
            }
        }

        // ---------------------------------------------------------
        // Phase B3: shift per-frame data
        // ---------------------------------------------------------
        for (int f = k; f < n - 1; f++) {

			m_pg.m_tstamps[f] = m_pg.m_tstamps[f + 1];
			m_pg.m_poses[f]   = m_pg.m_poses[f + 1];

			// ---- arrays → memcpy ----
			// void* std::memcpy(
			//     void*       dest,
			//     const void* src,
			//     std::size_t count
			// );
			// Copy all patch data of frame f+1 into frame f
			std::memcpy(
				m_pg.m_patches[f],
				m_pg.m_patches[f + 1],
				sizeof(m_pg.m_patches[0])
			);
			// Copy all patch colors from frame f+1 → frame f
			std::memcpy(
				m_pg.m_colors[f],
				m_pg.m_colors[f + 1],
				sizeof(m_pg.m_colors[0])
			);
			// Copy camera intrinsics of frame f+1 → frame f
			std::memcpy(
				m_pg.m_intrinsics[f],
				m_pg.m_intrinsics[f + 1],
				sizeof(m_pg.m_intrinsics[0])
			);

			// ---- ring buffers / lightweight objects ----
			m_imap[f % m_pmem] = m_imap[(f + 1) % m_pmem];
			m_gmap[f % m_pmem] = m_gmap[(f + 1) % m_pmem];
			m_fmap1[f % m_mem] = m_fmap1[(f + 1) % m_mem];
			m_fmap2[f % m_mem] = m_fmap2[(f + 1) % m_mem];
		}

        m_pg.m_n--;
        m_pg.m_m -= PatchGraph::M;
    }

    // =============================================================
    // Phase C: remove old edges outside optimization window
    // =============================================================
    {
        bool remove[MAX_EDGES] = {false};
        int num_active = m_pg.m_num_edges;

        for (int e = 0; e < num_active; e++) {
            if (m_pg.m_ix[m_pg.m_kk[e]] < m_pg.m_n - m_cfg.REMOVAL_WINDOW)
                remove[e] = true;
        }

        removeFactors(remove, /*store=*/true);
    }
}


// -------------------------------------------------------------
// // Edge construction (forward)
// 		- Forward edges connect:
// 		- all patches from recent frames
// 		- to the newest frame (n - 1)
// -------------------------------------------------------------
void DPVO::edgesForward(std::vector<int>& kk,
                        std::vector<int>& jj) {
    kk.clear();
    jj.clear();

    int r = m_cfg.PATCH_LIFETIME;
    int t0 = PatchGraph::M * std::max(m_pg.m_n - r, 0);
    int t1 = PatchGraph::M * std::max(m_pg.m_n - 1, 0);

    for (int k = t0; k < t1; k++) {
        kk.push_back(k);
        jj.push_back(m_pg.m_n - 1);
    }
}

// -------------------------------------------------------------
// Edge construction (backward)
// 		- Backward edges connect:
// 		- patches from the newest frame
// 		- to all frames in the lifetime window
// -------------------------------------------------------------
void DPVO::edgesBackward(std::vector<int>& kk,
                         std::vector<int>& jj) {
    kk.clear();
    jj.clear();

    int r = m_cfg.PATCH_LIFETIME;
    int t0 = PatchGraph::M * std::max(m_pg.m_n - 1, 0);
    int t1 = PatchGraph::M * m_pg.m_n;

    for (int k = t0; k < t1; k++) {
        for (int f = std::max(m_pg.m_n - r, 0); f < m_pg.m_n; f++) {
            kk.push_back(k);
            jj.push_back(f);
        }
    }
}

// -------------------------------------------------------------
// Append factors (CRITICAL FIXED VERSION)
// 1️⃣ What (ii, jj, kk) mean in DPVO
// In DPVO, each edge (factor) connects:
// patch (landmark)  ↔  pose (frame)
// The naming comes from factor-graph convention:
// Array	Meaning				Node type
// ii		patch index			Landmark node
// jj		frame index			Pose node
// kk		helper index		(frame, patch) linear ID
// -------------------------------------------------------------
void DPVO::appendFactors(const std::vector<int>& kk,
                         const std::vector<int>& jj) {
    int numNew = kk.size();
    if (m_pg.m_num_edges + numNew > MAX_EDGES) return;

    int base = m_pg.m_num_edges;

    for (int i = 0; i < numNew; i++) {
        int k = kk[i];
        int frame = k / PatchGraph::M;
        int patch = k % PatchGraph::M;

        m_pg.m_kk[base + i] = k;
        m_pg.m_jj[base + i] = jj[i];
        m_pg.m_ii[base + i] = m_pg.m_index[frame][patch];

        // zero NET_DIM manually
        for (int d = 0; d < NET_DIM; d++) {
            m_pg.m_net[base + i][d] = 0.0f;
        }
    }

    m_pg.m_num_edges += numNew;
}

// -------------------------------------------------------------
// Remove factors (already correct, kept)
// -------------------------------------------------------------
void DPVO::removeFactors(const bool* mask, bool store) {
    PatchGraph& pg = m_pg;

    const int num_active = pg.m_num_edges;
    if (num_active == 0) return;

    bool m[MAX_EDGES];

    for (int i = 0; i < num_active; i++) {
        m[i] = mask ? mask[i] : false; // if mask exists, set m[i] to mask[i], otherwise set m[i] to false
    }

    // store inactive edges if requested
    if (store) {
        int w = pg.m_num_edges_inac;
        for (int i = 0; i < num_active; i++) {
            if (!m[i]) continue;
            if (w >= MAX_EDGES) break;

            pg.m_ii_inac[w]     = pg.m_ii[i];
            pg.m_jj_inac[w]     = pg.m_jj[i];
            pg.m_kk_inac[w]     = pg.m_kk[i];
            pg.m_weight_inac[w]= pg.m_weight[i];
            pg.m_target_inac[w]= pg.m_target[i];
            w++;
        }
        pg.m_num_edges_inac = w;
    }

    // compact active edges
    int write = 0;
    for (int read = 0; read < num_active; read++) {
        if (m[read]) continue;

        if (write != read) {
            pg.m_ii[write]     = pg.m_ii[read];
            pg.m_jj[write]     = pg.m_jj[read];
            pg.m_kk[write]     = pg.m_kk[read];
            pg.m_weight[write] = pg.m_weight[read];
            pg.m_target[write] = pg.m_target[read];
            std::memcpy(pg.m_net[write],
                        pg.m_net[read],
                        sizeof(float) * NET_DIM);
        }
        write++;
    }

    pg.m_num_edges = write;
}

// -------------------------------------------------------------
// Motion magnitude (based on Python motionmag)
// -------------------------------------------------------------
float DPVO::motionMagnitude(int i, int j) {
    // Find active edges where ii == i and jj == j
    const int num_active = m_pg.m_num_edges;
    if (num_active == 0) {
        return 0.0f;
    }
    
    // Collect edges matching (i, j)
    std::vector<int> matching_ii, matching_jj, matching_kk;
    for (int e = 0; e < num_active; e++) {
        if (m_pg.m_ii[e] == i && m_pg.m_jj[e] == j) {
            matching_ii.push_back(m_pg.m_ii[e]);
            matching_jj.push_back(m_pg.m_jj[e]);
            matching_kk.push_back(m_pg.m_kk[e]);
        }
    }
    
    // If no matching edges, return 0.0
    if (matching_ii.empty()) {
        return 0.0f;
    }
    
    // Flattened pointers to patches and intrinsics
    float* patches_flat = &m_pg.m_patches[0][0][0][0][0];
    float* intrinsics_flat = &m_pg.m_intrinsics[0][0];
    
    // Allocate output for flow magnitudes
    std::vector<float> flow_out(matching_ii.size());
    
    // Call flow_mag with matching edges
    pops::flow_mag(
        m_pg.m_poses,
        patches_flat,
        intrinsics_flat,
        matching_ii.data(),
        matching_jj.data(),
        matching_kk.data(),
        static_cast<int>(matching_ii.size()),
        m_cfg.PATCHES_PER_FRAME,
        m_P,
        0.5f,  // beta = 0.5 (from Python default)
        flow_out.data(),
        nullptr  // valid_out not needed
    );
    
    // Return mean flow
    float sum = 0.0f;
    for (float f : flow_out) {
        sum += f;
    }
    return (matching_ii.size() > 0) ? (sum / static_cast<float>(matching_ii.size())) : 0.0f;
}

// -------------------------------------------------------------
// Motion probe (based on Python motion_probe)
// -------------------------------------------------------------
float DPVO::motionProbe() {
    // Python: kk = torch.arange(self.m-self.M, self.m, device="cuda")
    //         jj = self.n * torch.ones_like(kk)
    //         ii = self.ix[kk]
    //         net = torch.zeros(1, len(ii), self.DIM, **self.kwargs)
    //         coords = self.reproject(indicies=(ii, jj, kk))
    //         corr = self.corr(coords, indicies=(kk, jj))
    //         ctx = self.imap[:,kk % (self.M * self.pmem)]
    //         net, (delta, weight, _) = self.network.update(net, ctx, corr, None, ii, jj, kk)
    //         return torch.quantile(delta.norm(dim=-1).float(), 0.5)
    
    const int M = m_cfg.PATCHES_PER_FRAME;
    const int P = m_P;
    
    // Get patches from last frame: kk = [m-M, m-M+1, ..., m-1]
    int m = m_pg.m_m;
    int n = m_pg.m_n;
    
    if (m < M || n == 0) {
        return 0.0f;  // Not enough patches/frames
    }
    
    std::vector<int> kk_vec, jj_vec, ii_vec;
    for (int k = m - M; k < m; k++) {
        kk_vec.push_back(k);
        jj_vec.push_back(n);  // Target is current frame (n)
        
        // Extract frame and patch from linear index
        int frame = k / M;
        int patch = k % M;
        ii_vec.push_back(m_pg.m_index[frame][patch]);
    }
    
    int num_edges = static_cast<int>(kk_vec.size());
    if (num_edges == 0) {
        return 0.0f;
    }
    
    // Reproject
    std::vector<float> coords(num_edges * 2 * P * P);
    reproject(ii_vec.data(), jj_vec.data(), kk_vec.data(), num_edges, coords.data());
    
    // Correlation (simplified - we need correlation computation)
    // For now, we'll use a simplified version that just computes delta norm
    // In full implementation, we'd call computeCorrelation and then update model
    
    // Simplified: compute delta from reprojection error
    // Python computes delta from network update, but for motion probe we can use a simpler metric
    std::vector<float> delta_norms(num_edges);
    for (int e = 0; e < num_edges; e++) {
        // Get center pixel coordinates
        int center_i0 = P / 2;
        int center_j0 = P / 2;
        int coord_x_idx = e * 2 * P * P + 0 * P * P + center_i0 * P + center_j0;
        int coord_y_idx = e * 2 * P * P + 1 * P * P + center_i0 * P + center_j0;
        
        // For motion probe, we compute the magnitude of the reprojection
        // This is a simplified version - full version would use network update
        float dx = coords[coord_x_idx];
        float dy = coords[coord_y_idx];
        delta_norms[e] = std::sqrt(dx * dx + dy * dy);
    }
    
    // Compute median (quantile 0.5)
    std::sort(delta_norms.begin(), delta_norms.end());
    float median = delta_norms[delta_norms.size() / 2];
    
    return median;
}


// -----------------------------------------------------------------------------
// Reproject patches from source frame i to target frame j using SE3 poses
// -----------------------------------------------------------------------------
// Purpose: Projects 3D patches (with inverse depth) from frame i to frame j
//          using camera poses and intrinsics. Computes 2D coordinates for each
//          pixel in each patch, along with optional Jacobians for bundle adjustment.
//
// Input Parameters:
//   ii: [num_edges] - Source frame indices for each edge (frame containing the patch)
//   jj: [num_edges] - Target frame indices for each edge (frame to project into)
//   kk: [num_edges] - Patch indices within source frame (which patch from frame i)
//   num_edges: Number of edges (active patch-frame pairs) to reproject
//
// Output Parameters:
//   coords_out: [num_edges, 2, P, P] flattened - Reprojected 2D coordinates (u, v) for each pixel
//               Layout: [edge][channel][y][x] where channel 0=u, channel 1=v
//               Coordinates are at 1/4 resolution (scaled by intrinsics)
//
// Optional Output Parameters (for Bundle Adjustment):
//   Ji_out: [num_edges, 2, P, P, 6] flattened - Jacobian w.r.t. source pose i (SE3, 6 DOF)
//           If nullptr, temporary buffer is allocated internally
//   Jj_out: [num_edges, 2, P, P, 6] flattened - Jacobian w.r.t. target pose j (SE3, 6 DOF)
//           If nullptr, temporary buffer is allocated internally
//   Jz_out: [num_edges, 2, P, P, 1] flattened - Jacobian w.r.t. inverse depth z
//           If nullptr, temporary buffer is allocated internally
//   valid_out: [num_edges, P, P] flattened - Validity mask (1.0 if pixel projects within bounds, 0.0 otherwise)
//              If nullptr, temporary buffer is allocated internally
//
// Internal Data Used:
//   m_pg.m_poses: [N] - SE3 camera poses for all frames
//   m_pg.m_patches: [N, M, 3, P, P] - 3D patches with inverse depth (x, y, z=1/inv_depth)
//   m_pg.m_intrinsics: [N, 4] - Camera intrinsics [fx, fy, cx, cy] for each frame
//   m_P: Patch size (typically 3)
//   m_cfg.PATCHES_PER_FRAME: M (number of patches per frame, typically 4 or 8)
//
// Algorithm:
//   1. For each edge e:
//      - Get patch kk[e] from frame ii[e] (source)
//      - Get poses for frames ii[e] and jj[e]
//      - Get intrinsics for frame jj[e] (target)
//   2. For each pixel (i0, j0) in patch:
//      - Inverse project: 2D pixel → 3D point using inverse depth from patch
//      - Transform: 3D point from frame i → frame j using SE3 poses
//      - Project: 3D point → 2D pixel in frame j using target intrinsics
//      - Store (u, v) coordinates in coords_out
//      - Compute Jacobians if requested (for bundle adjustment)
//      - Mark validity if pixel projects within image bounds
//
// Note: Coordinates are at 1/4 resolution (matching feature map resolution)
//       This matches Python DPVO behavior where reprojection uses scaled intrinsics
// -----------------------------------------------------------------------------
void DPVO::reproject(
    const int* ii,          // [num_edges] - Source frame indices
    const int* jj,          // [num_edges] - Target frame indices  
    const int* kk,          // [num_edges] - Patch indices within source frame
    int num_edges,          // Number of edges to reproject
    float* coords_out,      // Output: [num_edges, 2, P, P] - Reprojected (u, v) coordinates
    float* Ji_out,          // Optional: [num_edges, 2, P, P, 6] - Jacobian w.r.t. pose i
    float* Jj_out,          // Optional: [num_edges, 2, P, P, 6] - Jacobian w.r.t. pose j
    float* Jz_out,          // Optional: [num_edges, 2, P, P, 1] - Jacobian w.r.t. inverse depth
    float* valid_out)       // Optional: [num_edges, P, P] - Validity mask
{
    if (num_edges <= 0)
        return;

    // Flattened pointers to patches and intrinsics
    // m_patches: [N, M, 3, P, P] - patches stored as (frame, patch_idx, channel, y, x)
    // m_intrinsics: [N, 4] - intrinsics stored as (frame, [fx, fy, cx, cy])
    float* patches_flat = &m_pg.m_patches[0][0][0][0][0];
    float* intrinsics_flat = &m_pg.m_intrinsics[0][0];

    const int P = m_P;
    
    // Allocate temporary buffers if Jacobians are not provided
    // This allows the function to work even when Jacobians are not needed
    std::vector<float> Ji_temp, Jj_temp, Jz_temp, valid_temp;
    float* Ji_ptr = Ji_out;
    float* Jj_ptr = Jj_out;
    float* Jz_ptr = Jz_out;
    float* valid_ptr = valid_out;
    
    if (Ji_ptr == nullptr) {
        // Allocate temporary buffer: [num_edges, 2, P, P, 6] = num_edges * 2 * P * P * 6
        Ji_temp.resize(num_edges * 2 * P * P * 6);
        Ji_ptr = Ji_temp.data();
    }
    if (Jj_ptr == nullptr) {
        // Allocate temporary buffer: [num_edges, 2, P, P, 6] = num_edges * 2 * P * P * 6
        Jj_temp.resize(num_edges * 2 * P * P * 6);
        Jj_ptr = Jj_temp.data();
    }
    if (Jz_ptr == nullptr) {
        // Allocate temporary buffer: [num_edges, 2, P, P, 1] = num_edges * 2 * P * P * 1
        Jz_temp.resize(num_edges * 2 * P * P * 1);
        Jz_ptr = Jz_temp.data();
    }
    if (valid_ptr == nullptr) {
        // Allocate temporary buffer: [num_edges, P, P] = num_edges * P * P
        valid_temp.resize(num_edges * P * P);
        valid_ptr = valid_temp.data();
    }

    // Call transformWithJacobians to perform the actual reprojection
    // This function:
    //   1. Extracts patches from m_pg.m_patches using indices (ii, jj, kk)
    //   2. Inverse projects 2D pixels to 3D using inverse depth from patches
    //   3. Transforms 3D points from frame i to frame j using SE3 poses
    //   4. Projects 3D points to 2D in target frame using intrinsics
    //   5. Computes Jacobians for bundle adjustment (if buffers provided)
    pops::transformWithJacobians(
        m_pg.m_poses,         // SE3 poses [N] - camera poses for all frames
        patches_flat,         // Flattened patches [N*M*3*P*P] - 3D patches with inverse depth
        intrinsics_flat,      // Flattened intrinsics [N*4] - [fx, fy, cx, cy] for each frame
        ii, jj, kk,           // Edge indices: source frame, target frame, patch index
        num_edges,            // Number of edges to process
        m_cfg.PATCHES_PER_FRAME,  // M - patches per frame
        m_P,                  // P - patch size (typically 3)
        coords_out,           // Output: [num_edges, 2, P, P] - Reprojected (u, v) coordinates
        Ji_ptr,               // Output: [num_edges, 2, P, P, 6] - Jacobian w.r.t. pose i (SE3)
        Jj_ptr,               // Output: [num_edges, 2, P, P, 6] - Jacobian w.r.t. pose j (SE3)
        Jz_ptr,               // Output: [num_edges, 2, P, P, 1] - Jacobian w.r.t. inverse depth
        valid_ptr             // Output: [num_edges, P, P] - Validity mask (1.0=valid, 0.0=invalid)
    );

    // Diagnostic: Check output coordinates for NaN/Inf values
    auto logger = spdlog::get("dpvo");
    if (logger) {
        int coords_total_size = num_edges * 2 * m_P * m_P;
        int nan_count = 0;
        int inf_count = 0;
        int valid_count = 0;
        float min_val = std::numeric_limits<float>::max();
        float max_val = std::numeric_limits<float>::lowest();
        
        // Check first few edges
        int edges_to_check = std::min(num_edges, 5);
        for (int e = 0; e < edges_to_check; e++) {
            for (int i0 = 0; i0 < m_P; i0++) {
                for (int j0 = 0; j0 < m_P; j0++) {
                    int coord_x_idx = e * 2 * m_P * m_P + 0 * m_P * m_P + i0 * m_P + j0;
                    int coord_y_idx = e * 2 * m_P * m_P + 1 * m_P * m_P + i0 * m_P + j0;
                    if (coord_x_idx < coords_total_size && coord_y_idx < coords_total_size) {
                        float x = coords_out[coord_x_idx];
                        float y = coords_out[coord_y_idx];
                        if (!std::isfinite(x)) {
                            if (std::isnan(x)) nan_count++;
                            else if (std::isinf(x)) inf_count++;
                        } else {
                            valid_count++;
                            min_val = std::min(min_val, x);
                            max_val = std::max(max_val, x);
                        }
                        if (!std::isfinite(y)) {
                            if (std::isnan(y)) nan_count++;
                            else if (std::isinf(y)) inf_count++;
                        } else {
                            valid_count++;
                            min_val = std::min(min_val, y);
                            max_val = std::max(max_val, y);
                        }
                    }
                }
            }
        }
        
        logger->info("DPVO::reproject: Output coords check (first {} edges) - "
                     "total_samples={}, valid={}, NaN={}, Inf={}, valid_range=[{:.2f}, {:.2f}]",
                     edges_to_check, edges_to_check * m_P * m_P * 2,
                     valid_count, nan_count, inf_count, min_val, max_val);
        
        // Check first edge's first pixel specifically
        if (num_edges > 0) {
            int first_x_idx = 0 * 2 * m_P * m_P + 0 * m_P * m_P + 0 * m_P + 0;
            int first_y_idx = 0 * 2 * m_P * m_P + 1 * m_P * m_P + 0 * m_P + 0;
            if (first_x_idx < coords_total_size && first_y_idx < coords_total_size) {
                float first_x = coords_out[first_x_idx];
                float first_y = coords_out[first_y_idx];
                logger->info("DPVO::reproject: First edge[0] pixel[0][0] coords from transformWithJacobians - "
                             "coords_out[{}]={:.6f}, coords_out[{}]={:.6f}, "
                             "is_finite_x={}, is_finite_y={}, is_nan_x={}, is_nan_y={}, "
                             "ii[0]={}, jj[0]={}, kk[0]={}",
                             first_x_idx, first_x, first_y_idx, first_y,
                             std::isfinite(first_x), std::isfinite(first_y),
                             std::isnan(first_x), std::isnan(first_y),
                             num_edges > 0 ? ii[0] : -1,
                             num_edges > 0 ? jj[0] : -1,
                             num_edges > 0 ? kk[0] : -1);
            }
        }
        
        // Check validity mask if available
        if (valid_ptr != nullptr && num_edges > 0) {
            int valid_mask_count = 0;
            int invalid_mask_count = 0;
            for (int e = 0; e < std::min(num_edges, 5); e++) {
                for (int i0 = 0; i0 < m_P; i0++) {
                    for (int j0 = 0; j0 < m_P; j0++) {
                        int valid_idx = e * m_P * m_P + i0 * m_P + j0;
                        if (valid_idx < num_edges * m_P * m_P) {
                            if (valid_ptr[valid_idx] > 0.5f) {
                                valid_mask_count++;
                            } else {
                                invalid_mask_count++;
                            }
                        }
                    }
                }
            }
            logger->info("DPVO::reproject: Validity mask (first {} edges) - valid={}, invalid={}",
                         std::min(num_edges, 5), valid_mask_count, invalid_mask_count);
        }
    }

    // Output layout matches Python coords.permute(0,1,4,2,3)
    // Each edge: [2, P, P] → 2 channels (u, v) for each pixel in the patch
    // Coordinates are at 1/4 resolution (matching feature map resolution)
    // Jacobians are stored in output buffers if provided, otherwise discarded
}




void DPVO::terminate() 
{
    stopProcessingThread();
}

// -------------------------------------------------------------
// Helper function to convert tensor to image data (used in updateInput)
// -------------------------------------------------------------
#if defined(CV28) || defined(CV28_SIMULATOR)
static bool convertTensorToImage(ea_tensor_t* imgTensor, std::vector<uint8_t>& image_out, int& H_out, int& W_out)
{
    if (imgTensor == nullptr) {
        return false;
    }
    
    // Get tensor data
    void* tensor_data_ptr = ea_tensor_data_for_read(imgTensor, EA_CPU);
    if (tensor_data_ptr == nullptr) {
        return false;
    }
    
    uint8_t* image_data = static_cast<uint8_t*>(tensor_data_ptr);
    
    // Get pitch and shape
    size_t pitch = ea_tensor_pitch(imgTensor);
    size_t tensor_size = ea_tensor_size(imgTensor);
    const size_t* shape = ea_tensor_shape(imgTensor);
    
    if (shape == nullptr) {
        return false;
    }
    
    int H = static_cast<int>(shape[EA_H]);
    int W = static_cast<int>(shape[EA_W]);
    int C = static_cast<int>(shape[EA_C]);
    
    // Validate dimensions
    if (H <= 0 || W <= 0 || C <= 0 || H > 10000 || W > 10000 || C > 10) {
        return false;
    }
    
    // Allocate output buffer [C, H, W] format
    image_out.resize(H * W * C);
    
    // Convert from [H, W, C] to [C, H, W]
    if (pitch == W * C) {
        // Contiguous memory
        for (int c = 0; c < C; c++) {
            for (int y = 0; y < H; y++) {
                for (int x = 0; x < W; x++) {
                    int src_idx = y * W * C + x * C + c;
                    int dst_idx = c * H * W + y * W + x;
                    if (src_idx >= 0 && src_idx < H * W * C && 
                        dst_idx >= 0 && dst_idx < H * W * C) {
                        image_out[dst_idx] = image_data[src_idx];
                    }
                }
            }
        }
    } else {
        // Non-contiguous or planar format
        size_t buffer_size = static_cast<size_t>(H) * W * C;
        std::vector<uint8_t> image_contiguous(buffer_size);
        
        if (pitch < static_cast<size_t>(W * C)) {
            // Planar format
            size_t channel_size = static_cast<size_t>(H) * pitch;
            if (tensor_size < channel_size * C) {
                return false;
            }
            
            for (int c = 0; c < C; c++) {
                size_t channel_offset = static_cast<size_t>(c) * channel_size;
                uint8_t* channel_src = image_data + channel_offset;
                
                for (int y = 0; y < H; y++) {
                    size_t src_row_offset = static_cast<size_t>(y) * pitch;
                    size_t dst_offset = static_cast<size_t>(c) * H * W + y * W;
                    
                    if (channel_offset + src_row_offset + pitch <= tensor_size &&
                        dst_offset + W <= buffer_size) {
                        std::memcpy(image_out.data() + dst_offset, 
                                   channel_src + src_row_offset, 
                                   pitch);
                    }
                }
            }
        } else {
            // Non-contiguous (padded)
            for (int y = 0; y < H; y++) {
                size_t src_offset = static_cast<size_t>(y) * pitch;
                size_t dst_offset = static_cast<size_t>(y) * W * C;
                size_t total_src_size = static_cast<size_t>(H) * pitch;
                
                if (src_offset + W * C <= total_src_size && 
                    dst_offset + W * C <= buffer_size) {
                    std::memcpy(image_contiguous.data() + dst_offset, 
                               image_data + src_offset, 
                               W * C);
                }
            }
            
            // Convert from [H, W, C] to [C, H, W]
            for (int c = 0; c < C; c++) {
                for (int y = 0; y < H; y++) {
                    for (int x = 0; x < W; x++) {
                        int src_idx = y * W * C + x * C + c;
                        int dst_idx = c * H * W + y * W + x;
                        if (src_idx >= 0 && src_idx < H * W * C && 
                            dst_idx >= 0 && dst_idx < H * W * C) {
                            image_out[dst_idx] = image_contiguous[src_idx];
                        }
                    }
                }
            }
        }
    }
    
    H_out = H;
    W_out = W;
    return true;
}
#endif

// -------------------------------------------------------------
// Threading interface (similar to wnc_app)
// -------------------------------------------------------------
void DPVO::startProcessingThread()
{
    // Initialize logger if it doesn't exist
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    m_processingThreadRunning = true;
    m_processingThread = std::thread([this]() {
        // Get or create logger in thread
        auto logger = spdlog::get("dpvo");
        if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
            logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
            logger = spdlog::stdout_color_mt("dpvo");
            logger->set_pattern("[%n] [%^%l%$] %v");
#endif
        }
        
        std::unique_lock<std::mutex> lock(m_queueMutex);
        while (m_processingThreadRunning)
        {
            m_queueCV.wait_for(lock, std::chrono::milliseconds(1000), [this]() {
                return !m_processingThreadRunning || !m_inputFrameQueue.empty();
            });
            if (!m_processingThreadRunning)
                break;

            if (!m_inputFrameQueue.empty())
            {
                InputFrame frame = std::move(m_inputFrameQueue.front());
                m_inputFrameQueue.pop();
                lock.unlock();
                
                m_bDone = false;
                
                try {
                    // Update timestamp (increment for each frame)
                    m_currentTimestamp++;
                    
                    // Call the main processing function (use member variables for intrinsics and timestamp)
					logger->info("Start run DPVO run function");
                    if (logger) {
                        logger->info("DPVO::startProcessingThread: Using intrinsics fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
                                     m_intrinsics[0], m_intrinsics[1], m_intrinsics[2], m_intrinsics[3]);
                    }
#if defined(CV28) || defined(CV28_SIMULATOR)
                    // CV28 builds always use tensor
                    if (frame.tensor_img != nullptr) {
                        run(m_currentTimestamp, frame.tensor_img, m_intrinsics);
                    } else {
                        if (logger) logger->error("DPVO::startProcessingThread: frame.tensor_img is nullptr in CV28 build!");
                    }
#else
                    // Non-CV28 builds use uint8_t* image
                    run(m_currentTimestamp, frame.image.data(), m_intrinsics, frame.H, frame.W);
#endif
					logger->info("DPVO run function finished");
                } catch (const std::exception& e) {
                    if (logger) logger->error("Exception in frame processing: {}", e.what());
                } catch (...) {
                    if (logger) logger->error("Unknown exception in frame processing");
                }
                
                m_bDone = true;
                lock.lock();
            }
        }
        if (logger) logger->debug("startProcessingThread is terminated");
    });
}

void DPVO::stopProcessingThread()
{
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_processingThreadRunning = false;
    }
    m_queueCV.notify_one();
    if (m_processingThread.joinable())
        m_processingThread.join();
}

void DPVO::wakeProcessingThread()
{
    m_queueCV.notify_one();
}

#if defined(CV28) || defined(CV28_SIMULATOR)
void DPVO::updateInput(ea_tensor_t* imgTensor)
{
    if (imgTensor == nullptr) {
        auto logger = spdlog::get("dpvo");
        if (logger) logger->error("updateInput: imgTensor is nullptr");
        return;
    }
    
    // Get or create logger
    auto logger = spdlog::get("dpvo");
    if (!logger) {
#ifdef SPDLOG_USE_SYSLOG
        logger = spdlog::syslog_logger_mt("dpvo", "ai-main", LOG_CONS | LOG_NDELAY, LOG_SYSLOG);
#else
        logger = spdlog::stdout_color_mt("dpvo");
        logger->set_pattern("[%n] [%^%l%$] %v");
#endif
    }
    
    if (logger) logger->debug("updateInput: Storing tensor directly (no conversion needed)");
    
    // Store tensor directly to avoid conversion overhead
    // fnet/inet can use the tensor directly via ea_cvt_color_resize
    InputFrame frame;
    frame.tensor_img = imgTensor;  // Store tensor directly
    
    // Get dimensions from tensor
    const size_t* shape = ea_tensor_shape(imgTensor);
    frame.H = static_cast<int>(shape[EA_H]);
    frame.W = static_cast<int>(shape[EA_W]);
    
    // Initialize image vector (empty, not used when tensor is available)
    frame.image.clear();
    
    if (logger) logger->debug("updateInput: Tensor stored, H={}, W={}", frame.H, frame.W);
    
    std::lock_guard<std::mutex> lock(m_queueMutex);
    
    m_inputFrameQueue.push(std::move(frame));
    
    if (logger) logger->debug("updateInput: Frame added to queue, queue_size={}", m_inputFrameQueue.size());
    
    // Limit queue size to prevent memory issues
    const size_t MAX_QUEUE_SIZE = 10;
    while (m_inputFrameQueue.size() > MAX_QUEUE_SIZE)
    {
        m_inputFrameQueue.pop();
    }
    
    wakeProcessingThread();
    
    if (logger) logger->debug("updateInput: Finished, notified processing thread");
}

void DPVO::addFrame(ea_tensor_t* imgTensor)
{
    updateInput(imgTensor);
}
#else
// Fallback implementation for non-CV28 platforms
void DPVO::updateInput(const uint8_t* image, int H, int W)
{
    if (image == nullptr)
        return;
        
    std::lock_guard<std::mutex> lock(m_queueMutex);
    
    InputFrame frame;
    frame.image.assign(image, image + H * W * 3);  // Copy image data (assuming RGB)
    frame.H = H;
    frame.W = W;
    
    m_inputFrameQueue.push(std::move(frame));
    
    // Limit queue size to prevent memory issues
    const size_t MAX_QUEUE_SIZE = 10;
    while (m_inputFrameQueue.size() > MAX_QUEUE_SIZE)
    {
        m_inputFrameQueue.pop();
    }
    
    wakeProcessingThread();
}

void DPVO::addFrame(const uint8_t* image, int H, int W)
{
    updateInput(image, H, W);
}
#endif

bool DPVO::_hasWorkToDo()
{
    return !m_inputFrameQueue.empty();
}

bool DPVO::isProcessingComplete()
{
    std::lock_guard<std::mutex> lock(m_queueMutex);
    return m_inputFrameQueue.empty() && m_bDone;
}

// -------------------------------------------------------------
// Visualization
// -------------------------------------------------------------
void DPVO::enableVisualization(bool enable)
{
#ifdef ENABLE_PANGOLIN_VIEWER
    m_visualizationEnabled = enable;
    
    if (enable && m_viewer == nullptr) {
        // Initialize viewer with current image dimensions
        try {
            m_viewer = std::make_unique<DPVOViewer>(m_wd, m_ht, PatchGraph::N, PatchGraph::N * m_cfg.PATCHES_PER_FRAME);
            auto logger = spdlog::get("dpvo");
            if (logger) logger->info("DPVO: Visualization enabled");
        } catch (const std::exception& e) {
            auto logger = spdlog::get("dpvo");
            if (logger) logger->error("DPVO: Failed to initialize viewer: {}", e.what());
            m_visualizationEnabled = false;
        }
    } else if (!enable && m_viewer != nullptr) {
        m_viewer->close();
        m_viewer->join();
        m_viewer.reset();
        auto logger = spdlog::get("dpvo");
        if (logger) logger->info("DPVO: Visualization disabled");
    }
#else
    // Viewer not compiled in - log warning if trying to enable
    if (enable) {
        auto logger = spdlog::get("dpvo");
        if (logger) logger->warn("DPVO: Visualization requested but not available (Pangolin not enabled). "
                                 "Compile with -DENABLE_PANGOLIN_VIEWER and link against Pangolin to enable.");
    }
    m_visualizationEnabled = false;
#endif
}

void DPVO::computePointCloud()
{
    // Compute 3D points from patches and poses
    // For each patch, backproject using pose and intrinsics
    const int n = m_pg.m_n;
    const int M = m_cfg.PATCHES_PER_FRAME;
    const int P = m_P;
    
    auto logger = spdlog::get("dpvo");
    
    // Expected coordinate ranges at scaled resolution
    const float max_x = static_cast<float>(m_wd) / 4.0f;  // e.g., 1920/4 = 480
    const float max_y = static_cast<float>(m_ht) / 4.0f;  // e.g., 1080/4 = 270
    
    // RGB values (from old frames) are typically in range [-0.5, 1.5] after normalization
    // Coordinates should be in range [0, max_x] and [0, max_y]
    // So if px < 2.0 or py < 2.0, it's likely RGB, not coordinates
    const float MIN_VALID_COORD = 2.0f;  // Minimum valid coordinate (RGB values are < 2.0)
    
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < M; k++) {
            int idx = i * M + k;
            
            // Get patch center coordinates and depth
            int center_y = P / 2;
            int center_x = P / 2;
            float px = m_pg.m_patches[i][k][0][center_y][center_x];
            float py = m_pg.m_patches[i][k][1][center_y][center_x];
            float pd = m_pg.m_patches[i][k][2][center_y][center_x];  // inverse depth
            
            // Skip invalid points (zero or negative inverse depth)
            if (pd <= 0.0f || pd > 10.0f) {
                m_pg.m_points[idx].x = 0.0f;
                m_pg.m_points[idx].y = 0.0f;
                m_pg.m_points[idx].z = 0.0f;
                continue;
            }
            
            // Detect if this frame has old RGB values instead of coordinates
            // RGB values are typically < 2.0, coordinates should be >= 2.0 (at scaled resolution)
            // Also check upper bound to catch corrupted values
            bool has_valid_coords = (px >= MIN_VALID_COORD && px <= max_x + 10.0f &&
                                     py >= MIN_VALID_COORD && py <= max_y + 10.0f);
            
            if (!has_valid_coords) {
                // This frame likely has old RGB values instead of coordinates
                // Skip this point - we can't compute valid 3D position without proper coordinates
                if (logger && i < 2 && k < 2) {
                    logger->debug("Point cloud: Skipping frame[{}] patch[{}] - invalid coordinates "
                                 "(px={:.2f}, py={:.2f}), likely RGB values from old frame. "
                                 "Expected coords in range: x=[{:.2f}, {:.2f}], y=[{:.2f}, {:.2f}]",
                                 i, k, px, py, MIN_VALID_COORD, max_x, MIN_VALID_COORD, max_y);
                }
                m_pg.m_points[idx].x = 0.0f;
                m_pg.m_points[idx].y = 0.0f;
                m_pg.m_points[idx].z = 0.0f;
                continue;
            }
            
            // Get intrinsics (scaled by RES=4)
            const float* intr = m_pg.m_intrinsics[i];
            float fx = intr[0];
            float fy = intr[1];
            float cx = intr[2];
            float cy = intr[3];
            
            // Debug: Check if patch coordinates are in wrong scale
            if (logger && i == 0 && k < 3) {
                logger->info("Point cloud debug [frame={}, patch={}]: px={:.4f}, py={:.4f}, "
                             "fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
                             i, k, px, py, fx, fy, cx, cy);
            }
            
            // Inverse projection: normalized camera coordinates
            // X0, Y0 are in normalized image plane (Z=1)
            float X0 = (px - cx) / fx;
            float Y0 = (py - cy) / fy;
            
            // Convert to 3D point in camera frame
            // pd is inverse depth, so depth = 1/pd
            // Point in camera frame: [X0*depth, Y0*depth, depth] = [X0/pd, Y0/pd, 1/pd]
            float depth = 1.0f / pd;
            Eigen::Vector3f p_camera(X0 * depth, Y0 * depth, depth);
            
            // Transform to world coordinates using pose
            // SE3 poses are stored as world-to-camera (T_wc), so we need inverse for camera-to-world
            // p_world = T_cw * p_camera = T_wc^-1 * p_camera
            const SE3& T_wc = m_pg.m_poses[i];
            SE3 T_cw = T_wc.inverse();
            Eigen::Vector3f p_world = T_cw.R() * p_camera + T_cw.t;
            
            // Store point
            m_pg.m_points[idx].x = p_world.x();
            m_pg.m_points[idx].y = p_world.y();
            m_pg.m_points[idx].z = p_world.z();
            
            // Debug logging for first few points
            if (logger && i == 0 && k < 3) {
                Eigen::Vector3f t_wc = T_wc.t;
                Eigen::Vector3f t_cw = T_cw.t;
                bool is_identity = (T_wc.t.norm() < 1e-6 && 
                                   (T_wc.R() - Eigen::Matrix3f::Identity()).norm() < 1e-6);
                logger->info("Point cloud [frame={}, patch={}]: px={:.2f}, py={:.2f}, pd={:.4f}, "
                             "p_camera=({:.3f}, {:.3f}, {:.3f}), p_world=({:.3f}, {:.3f}, {:.3f}), "
                             "T_wc.t=({:.3f}, {:.3f}, {:.3f}), T_cw.t=({:.3f}, {:.3f}, {:.3f}), is_identity={}",
                             i, k, px, py, pd, 
                             p_camera.x(), p_camera.y(), p_camera.z(),
                             p_world.x(), p_world.y(), p_world.z(),
                             t_wc.x(), t_wc.y(), t_wc.z(),
                             t_cw.x(), t_cw.y(), t_cw.z(),
                             is_identity);
            }
        }
    }
}

void DPVO::updateViewer()
{
    if (!m_visualizationEnabled || m_viewer == nullptr) {
        return;
    }
    
    try {
        auto logger = spdlog::get("dpvo");
        
        // Update poses
        // CRITICAL: Use m_allPoses (all historical frames) instead of m_pg.m_poses (sliding window only)
        // m_pg.m_n is the sliding window size (8-10 frames), but for visualization we want to see
        // the full trajectory (all frames processed so far, tracked by m_counter)
        if (m_counter > 0 && !m_allPoses.empty()) {
            // Pass all historical poses to viewer
            // CRITICAL: m_counter is the total number of frames processed (0-indexed, so frame N has m_counter = N+1)
            // But m_allPoses is indexed by m_counter (frame 0 stored at index 0, frame 1 at index 1, etc.)
            // So after processing N frames, m_counter = N and m_allPoses should have N entries (indices 0 to N-1)
            // However, we store at index m_counter BEFORE incrementing, so:
            //   Frame 0: store at index 0, then m_counter becomes 1
            //   Frame 1: store at index 1, then m_counter becomes 2
            //   ...
            //   Frame N: store at index N, then m_counter becomes N+1
            // So after N frames, m_counter = N+1 and m_allPoses.size() should be N+1 (indices 0 to N)
            // But we want to pass frames 0 to N (N+1 frames total), so we should pass m_counter frames
            // Actually, wait - let me check the logic again:
            //   After frame 0: m_counter=1, m_allPoses[0] exists, so we have 1 frame (index 0)
            //   After frame 1: m_counter=2, m_allPoses[0] and m_allPoses[1] exist, so we have 2 frames
            //   After frame N: m_counter=N+1, m_allPoses has indices 0 to N, so we have N+1 frames
            // So we should pass m_counter frames (which equals the number of frames stored)
            // CRITICAL: After processing N+1 frames (0 to N), m_counter = N+1 and m_allPoses has N+1 entries
            // So we pass m_counter frames, which is correct
            int num_historical_frames = m_counter;
            
            // CRITICAL: Ensure m_allPoses has enough entries
            // If m_allPoses.size() < m_counter, it means some frames weren't stored
            // This can happen if frames were processed before m_allPoses was initialized
            // In that case, we can only pass what we have
            if (num_historical_frames > static_cast<int>(m_allPoses.size())) {
                if (logger) {
                    logger->warn("Viewer update: WARNING - m_counter={} > m_allPoses.size()={}. "
                                 "Some frames were not stored in m_allPoses. Limiting to {} frames. "
                                 "This suggests m_allPoses was not initialized early enough.",
                                 m_counter, m_allPoses.size(), m_allPoses.size());
                }
                num_historical_frames = static_cast<int>(m_allPoses.size());
            }
            
            // CRITICAL: If m_allPoses.size() is only 8 (sliding window size), it means we're only storing
            // poses from the sliding window, not all historical frames. This is a bug.
            if (num_historical_frames == m_pg.m_n && m_counter > m_pg.m_n && logger) {
                logger->error("Viewer update: ERROR - Only {} poses stored (sliding window size) but {} frames processed. "
                              "m_allPoses is not being populated correctly! Check that m_allPoses[m_counter] is set in run().",
                              num_historical_frames, m_counter);
            }
            
            // Debug: Log details to diagnose why only 8 frames are shown
            static int viewer_update_count = 0;
            bool should_log_poses = (viewer_update_count++ % 5 == 0);  // Log every 5th update
            
            if (logger) {
                logger->info("Viewer update: m_pg.m_n={} (sliding window), m_counter={} (total frames processed), "
                             "m_allPoses.size()={}, passing {} frames to viewer", 
                             m_pg.m_n, m_counter, m_allPoses.size(), num_historical_frames);
                
                // Check if m_allPoses has valid data
                int valid_poses_count = 0;
                for (int i = 0; i < num_historical_frames; i++) {
                    Eigen::Vector3f t = m_allPoses[i].t;
                    if (std::isfinite(t.x()) && std::isfinite(t.y()) && std::isfinite(t.z())) {
                        valid_poses_count++;
                    }
                }
                logger->info("Viewer update: {} out of {} historical poses are valid", 
                             valid_poses_count, num_historical_frames);
                
                // Debug: Log first 5 pose translations to see if they're all the same
                if (should_log_poses && num_historical_frames > 0) {
                    logger->info("Viewer update: First 5 pose translations (T_wc):");
                    for (int i = 0; i < std::min(5, num_historical_frames); i++) {
                        Eigen::Vector3f t = m_allPoses[i].t;
                        logger->info("  Frame[{}]: t=({:.3f}, {:.3f}, {:.3f}), norm={:.3f}", 
                                     i, t.x(), t.y(), t.z(), t.norm());
                    }
                }
            }
            
            m_viewer->updatePoses(m_allPoses.data(), num_historical_frames);
            if (logger && m_pg.m_n > 0) {
                // Log first and last pose to track movement
                // Also check if poses are identity and if poses are changing
                Eigen::Vector3f t0 = m_pg.m_poses[0].t;
                Eigen::Vector3f t_last = m_pg.m_poses[m_pg.m_n - 1].t;
                bool pose0_identity = (t0.norm() < 1e-6 && 
                                      (m_pg.m_poses[0].R() - Eigen::Matrix3f::Identity()).norm() < 1e-6);
                bool pose_last_identity = (t_last.norm() < 1e-6 && 
                                          (m_pg.m_poses[m_pg.m_n - 1].R() - Eigen::Matrix3f::Identity()).norm() < 1e-6);
                
                // Check if poses are all the same (sticking together issue)
                bool all_poses_same = true;
                if (m_pg.m_n > 1) {
                    Eigen::Vector3f t_first = m_pg.m_poses[0].t;
                    for (int i = 1; i < m_pg.m_n; i++) {
                        Eigen::Vector3f t_diff = m_pg.m_poses[i].t - t_first;
                        if (t_diff.norm() > 1e-3f) {
                            all_poses_same = false;
                            break;
                        }
                    }
                }
                
                logger->info("Viewer update: n={}, pose[0].t=({:.3f}, {:.3f}, {:.3f}), pose[0].is_identity={}, "
                             "pose[{}].t=({:.3f}, {:.3f}, {:.3f}), pose[{}].is_identity={}, all_poses_same={}",
                             m_pg.m_n, 
                             t0.x(), t0.y(), t0.z(), pose0_identity,
                             m_pg.m_n - 1,
                             t_last.x(), t_last.y(), t_last.z(), pose_last_identity,
                             all_poses_same);
                
                if (all_poses_same && m_pg.m_n > 1) {
                    logger->warn("Viewer: All poses have the same translation - cameras are stuck together!");
                }
            }
        }
        
        // Compute and update point cloud
        computePointCloud();
        int num_points = m_pg.m_n * m_cfg.PATCHES_PER_FRAME;
        if (num_points > 0) {
            // Flatten colors array: m_colors[N][M][3] -> uint8_t* with num_points * 3 elements
            uint8_t* colors_flat = reinterpret_cast<uint8_t*>(m_pg.m_colors);
            m_viewer->updatePoints(m_pg.m_points, colors_flat, num_points);
            if (logger) {
                logger->info("Viewer update: num_points={}", num_points);
            }
        }
    } catch (const std::exception& e) {
        auto logger = spdlog::get("dpvo");
        if (logger) logger->error("DPVO: Error updating viewer: {}", e.what());
    }
}

