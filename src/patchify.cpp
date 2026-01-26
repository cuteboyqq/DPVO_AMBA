#include "patchify.hpp"
#include "fnet.hpp"
#include "inet.hpp"
#ifdef USE_ONNX_RUNTIME
#include "fnet_onnx.hpp"
#include "inet_onnx.hpp"
#endif
#include "correlation_kernel.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <string>
#include <spdlog/spdlog.h>

// =================================================================================================
// Patchifier Implementation
// =================================================================================================
Patchifier::Patchifier(int patch_size, int DIM)
    : m_patch_size(patch_size), m_DIM(DIM), m_fnet(nullptr), m_inet(nullptr)
#ifdef USE_ONNX_RUNTIME
    , m_fnet_onnx(nullptr), m_inet_onnx(nullptr), m_useOnnxRuntime(false)
#endif
{
}

Patchifier::Patchifier(int patch_size, int DIM, Config_S *config)
    : m_patch_size(patch_size), m_DIM(DIM)
{
    // Models will be set via setModels() if config provided
    if (config != nullptr)
    {
        // Note: You'll need separate configs for fnet and inet
        // For now, assuming same config path structure
    }
}

Patchifier::~Patchifier()
{
    // Models will be automatically destroyed by unique_ptr
}

void Patchifier::setModels(Config_S *fnetConfig, Config_S *inetConfig)
{
    // Drop existing loggers if they exist (in case models were created elsewhere)
    // This prevents "logger with name already exists" errors
#ifdef SPDLOG_USE_SYSLOG
    spdlog::drop("fnet");
    spdlog::drop("inet");
#else
    spdlog::drop("fnet");
    spdlog::drop("inet");
#endif
    
    // Check if ONNX Runtime should be used
#ifdef USE_ONNX_RUNTIME
    bool useOnnx = false;
    if (fnetConfig != nullptr && fnetConfig->useOnnxRuntime) {
        useOnnx = true;
    } else if (inetConfig != nullptr && inetConfig->useOnnxRuntime) {
        useOnnx = true;
    }
    
    m_useOnnxRuntime = useOnnx;
    
    if (useOnnx) {
        // Use ONNX Runtime models
        if (fnetConfig != nullptr) {
            m_fnet_onnx = std::make_unique<FNetInferenceONNX>(fnetConfig);
            m_fnet = nullptr;  // Clear AMBA model
        }
        if (inetConfig != nullptr) {
            m_inet_onnx = std::make_unique<INetInferenceONNX>(inetConfig);
            m_inet = nullptr;  // Clear AMBA model
        }
    } else {
        // Use AMBA EazyAI models
        if (fnetConfig != nullptr) {
            m_fnet = std::make_unique<FNetInference>(fnetConfig);
            m_fnet_onnx = nullptr;  // Clear ONNX model
        }
        if (inetConfig != nullptr) {
            m_inet = std::make_unique<INetInference>(inetConfig);
            m_inet_onnx = nullptr;  // Clear ONNX model
        }
    }
#else
    // ONNX Runtime not available, use AMBA models
    if (fnetConfig != nullptr) {
        m_fnet = std::make_unique<FNetInference>(fnetConfig);
    }
    if (inetConfig != nullptr) {
        m_inet = std::make_unique<INetInference>(inetConfig);
    }
#endif
}

// Forward pass: fill fmap, imap, gmap, patches, clr
// Note: fmap and imap are at 1/4 resolution (RES=4), but image and coords are at full resolution
// image: normalized float image [C, H, W] with values in range [-0.5, 1.5] (Python: 2 * (image / 255.0) - 0.5)
// Helper function to extract patches after inference has been run
void Patchifier::extractPatchesAfterInference(int H, int W, int fmap_H, int fmap_W, int M,
                                                float* fmap, float* imap, float* gmap,
                                                float* patches, uint8_t* clr, const uint8_t* image_for_colors,
                                                int H_image, int W_image)
{
    const int inet_output_channels = 384;
    
    printf("[Patchifier] About to create coords, M=%d\n", M);
    fflush(stdout);
    
    // ------------------------------------------------
    // Generate RANDOM coords at FEATURE MAP resolution (matching Python)
    // ------------------------------------------------
    // CRITICAL: Python generates coordinates at feature map resolution (h, w from fmap.shape)
    // Python: x = torch.randint(1, w-1, ...) where w is feature map width
    //         y = torch.randint(1, h-1, ...) where h is feature map height
    // These coordinates are used DIRECTLY for all patchify operations (no scaling)
    //
    // These random coordinates are used ONLY ONCE to initialize patches (landmarks)
    // They define WHERE patches are extracted from the current frame
    // 
    // Later, in DPVO::update(), patches are tracked across frames using REPROJECTED coordinates:
    //   - Stored patch coordinates (from this random initialization) are read from m_pg.m_patches[i][k]
    //   - Reprojected to target frames using camera poses
    //   - Correlation is computed at REPROJECTED locations (not random locations!)
    //
    // Why random? Ensures good spatial coverage of the image, avoids bias toward specific regions
    // Each frame gets a fresh set of landmarks initialized at random locations
    m_last_coords.resize(M * 2);
    // Generate coordinates at FEATURE MAP resolution (matching Python)
    for (int m = 0; m < M; m++)
    {
        m_last_coords[m * 2 + 0] = 1.0f + static_cast<float>(rand() % (fmap_W - 2));
        m_last_coords[m * 2 + 1] = 1.0f + static_cast<float>(rand() % (fmap_H - 2));
    }
    const float* coords = m_last_coords.data();  // coords are at feature map resolution
    printf("[Patchifier] Coords created at feature map resolution: fmap_H=%d, fmap_W=%d\n", fmap_H, fmap_W);
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (gmap)\n");
    fflush(stdout);
    // ------------------------------------------------
    // Patchify fmap → gmap (using coords directly at feature map resolution)
    // ------------------------------------------------
    patchify_cpu_safe(
        fmap, coords,  // Use coords directly (already at feature map resolution)
        M, 128, fmap_H, fmap_W,
        m_patch_size / 2,
        gmap);
    printf("[Patchifier] patchify_cpu_safe (gmap) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (imap)\n");
    fflush(stdout);
    // ------------------------------------------------
    // imap sampling (radius = 0) - extract patches from m_imap_buffer
    // Use coords directly (already at feature map resolution)
    // ------------------------------------------------
#ifdef USE_ONNX_RUNTIME
    bool models_available = (m_useOnnxRuntime && m_fnet_onnx != nullptr && m_inet_onnx != nullptr) ||
                             (!m_useOnnxRuntime && m_fnet != nullptr && m_inet != nullptr);
#else
    bool models_available = (m_fnet != nullptr && m_inet != nullptr);
#endif
    
    if (models_available) {
        float imap_buffer_sample_min = *std::min_element(m_imap_buffer.begin(), 
                                                          m_imap_buffer.begin() + std::min(static_cast<size_t>(100), m_imap_buffer.size()));
        float imap_buffer_sample_max = *std::max_element(m_imap_buffer.begin(), 
                                                          m_imap_buffer.begin() + std::min(static_cast<size_t>(100), m_imap_buffer.size()));
        printf("[Patchifier] Before patchify_cpu_safe (imap): m_imap_buffer sample range: [%f, %f], size=%zu\n", 
               imap_buffer_sample_min, imap_buffer_sample_max, m_imap_buffer.size());
        fflush(stdout);
        
        printf("[Patchifier] coords for imap extraction (at feature map resolution):\n");
        for (int m = 0; m < std::min(M, 8); m++) {
            printf("[Patchifier]   Patch %d: x=%.2f, y=%.2f (fmap_H=%d, fmap_W=%d)\n", 
                   m, coords[m*2+0], coords[m*2+1], fmap_H, fmap_W);
        }
        fflush(stdout);
        
        patchify_cpu_safe(
            m_imap_buffer.data(), coords,  // Use coords directly (already at feature map resolution)
            M, inet_output_channels, fmap_H, fmap_W,
            0,
            imap);
        
        float imap_min = *std::min_element(imap, imap + M * m_DIM);
        float imap_max = *std::max_element(imap, imap + M * m_DIM);
        int imap_zero_count = 0;
        int imap_nonzero_count = 0;
        for (int i = 0; i < M * m_DIM; i++) {
            if (imap[i] == 0.0f) imap_zero_count++;
            else imap_nonzero_count++;
        }
        printf("[Patchifier] After patchify_cpu_safe (imap): imap stats - zero_count=%d, nonzero_count=%d, min=%f, max=%f\n",
               imap_zero_count, imap_nonzero_count, imap_min, imap_max);
        fflush(stdout);
    } else {
        std::fill(imap, imap + M * m_DIM, 0.0f);
        printf("[Patchifier] WARNING: Models not available, zero-filling imap\n");
        fflush(stdout);
    }
    printf("[Patchifier] patchify_cpu_safe (imap) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (patches)\n");
    fflush(stdout);
    
    // ------------------------------------------------
    // Patchify grid → patches (RGB)
    // Python: patches = altcorr.patchify(grid[0], coords, P//2)
    // where grid is created from disps with shape (b, n, h, w) where h, w are feature map dimensions
    // So coords are at feature map resolution, and grid is also at feature map resolution
    // ------------------------------------------------
    // CRITICAL: Grid must be created at FEATURE MAP resolution to match Python
    // Python: grid, _ = coords_grid_with_index(disps, device=fmap.device)
    // where disps = torch.ones(b, n, h, w) and h, w are feature map dimensions
    std::vector<float> grid_fmap(3 * fmap_H * fmap_W);
    for (int y = 0; y < fmap_H; y++) {
        for (int x = 0; x < fmap_W; x++) {
            int idx = y * fmap_W + x;
            grid_fmap[0 * fmap_H * fmap_W + idx] = static_cast<float>(x);
            grid_fmap[1 * fmap_H * fmap_W + idx] = static_cast<float>(y);
            grid_fmap[2 * fmap_H * fmap_W + idx] = 1.0f;
        }
    }
    
    patchify_cpu_safe(
        grid_fmap.data(), coords,  // coords are at feature map resolution, grid is also at feature map resolution
        M, 3, fmap_H, fmap_W,
        m_patch_size / 2,
        patches);
    
    printf("[Patchifier] patchify_cpu_safe (patches) completed\n");
    fflush(stdout);
    
    // Save patchify outputs for comparison with Python (frame 0 and frame 1)
    static int patchify_frame_counter = 0;
    patchify_frame_counter++;
    
    if (patchify_frame_counter == 1 || patchify_frame_counter == 2) {
        int frame_idx = patchify_frame_counter - 1;  // frame_idx: 0 for first call, 1 for second call
        
        // Save coordinates
        std::string coords_filename = "cpp_coords_frame" + std::to_string(frame_idx) + ".bin";
        std::ofstream coords_file(coords_filename, std::ios::binary);
        if (coords_file.is_open()) {
            coords_file.write(reinterpret_cast<const char*>(coords), M * 2 * sizeof(float));
            coords_file.close();
            printf("[Patchifier] Saved coordinates to %s: M=%d, size=%zu bytes\n", 
                   coords_filename.c_str(), M, M * 2 * sizeof(float));
            fflush(stdout);
        }
        
        // Save gmap: [M, 128, P, P] = [4, 128, 3, 3]
        std::string gmap_filename = "cpp_gmap_frame" + std::to_string(frame_idx) + ".bin";
        std::ofstream gmap_file(gmap_filename, std::ios::binary);
        if (gmap_file.is_open()) {
            int gmap_size = M * 128 * m_patch_size * m_patch_size;
            gmap_file.write(reinterpret_cast<const char*>(gmap), gmap_size * sizeof(float));
            gmap_file.close();
            printf("[Patchifier] Saved gmap to %s: shape=[%d, 128, %d, %d], size=%zu bytes\n",
                   gmap_filename.c_str(), M, m_patch_size, m_patch_size, gmap_size * sizeof(float));
            fflush(stdout);
        }
        
        // Save imap: [M, 384, 1, 1] = [4, 384, 1, 1]
        std::string imap_filename = "cpp_imap_frame" + std::to_string(frame_idx) + ".bin";
        std::ofstream imap_file(imap_filename, std::ios::binary);
        if (imap_file.is_open()) {
            int imap_size = M * inet_output_channels * 1 * 1;
            imap_file.write(reinterpret_cast<const char*>(imap), imap_size * sizeof(float));
            imap_file.close();
            printf("[Patchifier] Saved imap to %s: shape=[%d, %d, 1, 1], size=%zu bytes\n",
                   imap_filename.c_str(), M, inet_output_channels, imap_size * sizeof(float));
            fflush(stdout);
        }
        
        // Save patches: [M, 3, P, P] = [4, 3, 3, 3]
        std::string patches_filename = "cpp_patches_frame" + std::to_string(frame_idx) + ".bin";
        std::ofstream patches_file(patches_filename, std::ios::binary);
        if (patches_file.is_open()) {
            int patches_size = M * 3 * m_patch_size * m_patch_size;
            patches_file.write(reinterpret_cast<const char*>(patches), patches_size * sizeof(float));
            patches_file.close();
            printf("[Patchifier] Saved patches to %s: shape=[%d, 3, %d, %d], size=%zu bytes\n",
                   patches_filename.c_str(), M, m_patch_size, m_patch_size, patches_size * sizeof(float));
            fflush(stdout);
        }
    }

    printf("[Patchifier] About to extract colors\n");
    fflush(stdout);
    // ------------------------------------------------
    // Color for visualization - full resolution
    // ------------------------------------------------
    // CRITICAL: coords are at feature map resolution (fmap_H, fmap_W)
    // but image_for_colors is at full resolution (H_image, W_image)
    // Scale coordinates from feature map resolution to full image size
    // Python: clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0)
    // where images[0] is at full resolution and coords are at feature map resolution
    // The factor 4 scales from feature map to full resolution (RES=4)
    if (image_for_colors != nullptr) {
        int H_color = (H_image > 0) ? H_image : fmap_H * 4;  // Scale from feature map to full resolution
        int W_color = (W_image > 0) ? W_image : fmap_W * 4;   // Scale from feature map to full resolution
        float scale_x = static_cast<float>(W_color) / static_cast<float>(fmap_W);
        float scale_y = static_cast<float>(H_color) / static_cast<float>(fmap_H);
        
        for (int m = 0; m < M; m++)
        {
            // Scale coordinates from feature map resolution to full image size
            // Python uses: 4*(coords + 0.5), so we match that
            float x_scaled = (coords[m * 2 + 0] + 0.5f) * scale_x;
            float y_scaled = (coords[m * 2 + 1] + 0.5f) * scale_y;
            int x = static_cast<int>(std::round(x_scaled));
            int y = static_cast<int>(std::round(y_scaled));
            x = std::max(0, std::min(x, W_color - 1));
            y = std::max(0, std::min(y, H_color - 1));
            for (int c = 0; c < 3; c++) {
                // Image is in [C, H, W] format (uint8_t) at full resolution
                clr[m * 3 + c] = image_for_colors[c * H_color * W_color + y * W_color + x];
            }
        }
    } else {
        // Fallback: zero fill if no image provided
        std::fill(clr, clr + M * 3, 0);
    }
    printf("[Patchifier] Colors extracted\n");
    fflush(stdout);
}

#if defined(CV28) || defined(CV28_SIMULATOR)
// Tensor-based forward - uses tensor directly, avoids conversion (preferred for CV28)
void Patchifier::forward(
    ea_tensor_t* imgTensor,  // Input tensor (preferred, avoids conversion)
    float* fmap, float* imap, float* gmap,
    float* patches, uint8_t* clr,
    int patches_per_image)
{
    if (imgTensor == nullptr) {
        throw std::runtime_error("Patchifier::forward: imgTensor is nullptr");
    }
    
    // Get dimensions from tensor (full image size)
    const size_t* shape = ea_tensor_shape(imgTensor);
    int H_tensor = static_cast<int>(shape[EA_H]);  // Full image height (e.g., 1080)
    int W_tensor = static_cast<int>(shape[EA_W]);  // Full image width (e.g., 1920)
    
    // Get logger early (needed for dimension logging)
    auto logger_patch = spdlog::get("fnet");
    if (!logger_patch) {
        logger_patch = spdlog::get("inet");
    }
    
    // CRITICAL: Use model INPUT dimensions for patch extraction, not tensor dimensions
    // Models resize internally, so patches should be extracted at model input size
    // This ensures coordinates are within feature map bounds after scaling
    int H = getInputHeight();  // Model input height (e.g., 528)
    int W = getInputWidth();   // Model input width (e.g., 960)
    
    if (H == 0 || W == 0) {
        // Fallback to tensor dimensions if model input not available
        H = H_tensor;
        W = W_tensor;
        if (logger_patch) {
            logger_patch->warn("[Patchifier] Model input dimensions not available, using tensor dimensions {}x{}", H, W);
        }
    } else {
        if (logger_patch) {
            logger_patch->info("[Patchifier] Using model input dimensions {}x{} for patch extraction (tensor is {}x{})", 
                              H, W, H_tensor, W_tensor);
        }
    }
    
    // Use tensor-based runInference (avoids conversion)
    const int M = patches_per_image;
    const int inet_output_channels = 384;
    int fmap_H = getOutputHeight();
    int fmap_W = getOutputWidth();
    
    if (fmap_H == 0 || fmap_W == 0) {
        throw std::runtime_error("Patchifier::forward: Model output dimensions not available");
    }
    
    // Allocate buffers
    if (m_fmap_buffer.size() != 128 * fmap_H * fmap_W) {
        m_fmap_buffer.resize(128 * fmap_H * fmap_W);
    }
    if (m_imap_buffer.size() != inet_output_channels * fmap_H * fmap_W) {
        m_imap_buffer.resize(inet_output_channels * fmap_H * fmap_W);
    }
    
    // Run inference using tensor directly
#ifdef USE_ONNX_RUNTIME
    if (m_useOnnxRuntime) {
        // Use ONNX Runtime models
        if (logger_patch) logger_patch->info("[Patchifier] About to call fnet_onnx->runInference (tensor)");
        bool fnet_success = false;
        if (m_fnet_onnx && !m_fnet_onnx->runInference(imgTensor, m_fmap_buffer.data())) {
            if (logger_patch) logger_patch->error("[Patchifier] fnet_onnx->runInference (tensor) failed");
            std::fill(m_fmap_buffer.begin(), m_fmap_buffer.end(), 0.0f);
        } else {
            fnet_success = true;
            if (logger_patch) logger_patch->info("[Patchifier] fnet_onnx->runInference (tensor) successful");
        }
        
        if (logger_patch) logger_patch->info("[Patchifier] About to call inet_onnx->runInference (tensor)");
        bool inet_success = false;
        if (m_inet_onnx && !m_inet_onnx->runInference(imgTensor, m_imap_buffer.data())) {
            if (logger_patch) logger_patch->error("[Patchifier] inet_onnx->runInference (tensor) failed");
            std::fill(m_imap_buffer.begin(), m_imap_buffer.end(), 0.0f);
        } else {
            inet_success = true;
            if (logger_patch) logger_patch->info("[Patchifier] inet_onnx->runInference (tensor) successful");
        }
        
        // Save frame 0 and frame 1 outputs to binary files for comparison with Python
        static int frame_counter = 0;
        if (fnet_success && inet_success) {
            frame_counter++;
            
            // Get output dimensions for logging
            int fnet_C = 128;  // FNet output channels
            int fnet_H = fmap_H;
            int fnet_W = fmap_W;
            int inet_C = inet_output_channels;  // INet output channels (384)
            int inet_H = fmap_H;
            int inet_W = fmap_W;
            
            // Save frame 0
            if (frame_counter == 1) {
                // Save fnet output
                std::ofstream fnet_file("fnet_frame0.bin", std::ios::binary);
                if (fnet_file.is_open()) {
                    size_t fnet_size = m_fmap_buffer.size() * sizeof(float);
                    fnet_file.write(reinterpret_cast<const char*>(m_fmap_buffer.data()), fnet_size);
                    fnet_file.close();
                    if (logger_patch) {
                        logger_patch->info("[Patchifier] Saved fnet output to fnet_frame0.bin: "
                                           "shape=[1, {}, {}, {}] (C, H, W), {} bytes, {} floats",
                                           fnet_C, fnet_H, fnet_W, fnet_size, m_fmap_buffer.size());
                    }
                    printf("[Patchifier] Saved fnet output to fnet_frame0.bin: shape=[1, %d, %d, %d] (NCHW), %zu bytes\n",
                           fnet_C, fnet_H, fnet_W, fnet_size);
                    fflush(stdout);
                } else {
                    if (logger_patch) logger_patch->error("[Patchifier] Failed to open fnet_frame0.bin for writing");
                }
                
                // Save inet output
                std::ofstream inet_file("inet_frame0.bin", std::ios::binary);
                if (inet_file.is_open()) {
                    size_t inet_size = m_imap_buffer.size() * sizeof(float);
                    inet_file.write(reinterpret_cast<const char*>(m_imap_buffer.data()), inet_size);
                    inet_file.close();
                    if (logger_patch) {
                        logger_patch->info("[Patchifier] Saved inet output to inet_frame0.bin: "
                                           "shape=[1, {}, {}, {}] (C, H, W), {} bytes, {} floats",
                                           inet_C, inet_H, inet_W, inet_size, m_imap_buffer.size());
                    }
                    printf("[Patchifier] Saved inet output to inet_frame0.bin: shape=[1, %d, %d, %d] (NCHW), %zu bytes\n",
                           inet_C, inet_H, inet_W, inet_size);
                    fflush(stdout);
                } else {
                    if (logger_patch) logger_patch->error("[Patchifier] Failed to open inet_frame0.bin for writing");
                }
            }
            // Save frame 1
            else if (frame_counter == 2) {
                // Save fnet output
                std::ofstream fnet_file("fnet_frame1.bin", std::ios::binary);
                if (fnet_file.is_open()) {
                    size_t fnet_size = m_fmap_buffer.size() * sizeof(float);
                    fnet_file.write(reinterpret_cast<const char*>(m_fmap_buffer.data()), fnet_size);
                    fnet_file.close();
                    if (logger_patch) {
                        logger_patch->info("[Patchifier] Saved fnet output to fnet_frame1.bin: "
                                           "shape=[1, {}, {}, {}] (C, H, W), {} bytes, {} floats",
                                           fnet_C, fnet_H, fnet_W, fnet_size, m_fmap_buffer.size());
                    }
                    printf("[Patchifier] Saved fnet output to fnet_frame1.bin: shape=[1, %d, %d, %d] (NCHW), %zu bytes\n",
                           fnet_C, fnet_H, fnet_W, fnet_size);
                    fflush(stdout);
                } else {
                    if (logger_patch) logger_patch->error("[Patchifier] Failed to open fnet_frame1.bin for writing");
                }
                
                // Save inet output
                std::ofstream inet_file("inet_frame1.bin", std::ios::binary);
                if (inet_file.is_open()) {
                    size_t inet_size = m_imap_buffer.size() * sizeof(float);
                    inet_file.write(reinterpret_cast<const char*>(m_imap_buffer.data()), inet_size);
                    inet_file.close();
                    if (logger_patch) {
                        logger_patch->info("[Patchifier] Saved inet output to inet_frame1.bin: "
                                           "shape=[1, {}, {}, {}] (C, H, W), {} bytes, {} floats",
                                           inet_C, inet_H, inet_W, inet_size, m_imap_buffer.size());
                    }
                    printf("[Patchifier] Saved inet output to inet_frame1.bin: shape=[1, %d, %d, %d] (NCHW), %zu bytes\n",
                           inet_C, inet_H, inet_W, inet_size);
                    fflush(stdout);
                } else {
                    if (logger_patch) logger_patch->error("[Patchifier] Failed to open inet_frame1.bin for writing");
                }
            }
        }
    } else {
        // Use AMBA EazyAI models
        if (logger_patch) logger_patch->info("[Patchifier] About to call fnet->runInference (tensor)");
        if (m_fnet && !m_fnet->runInference(imgTensor, m_fmap_buffer.data())) {
            if (logger_patch) logger_patch->error("[Patchifier] fnet->runInference (tensor) failed");
            std::fill(m_fmap_buffer.begin(), m_fmap_buffer.end(), 0.0f);
        } else {
            if (logger_patch) logger_patch->info("[Patchifier] fnet->runInference (tensor) successful");
        }
        
        if (logger_patch) logger_patch->info("[Patchifier] About to call inet->runInference (tensor)");
        if (m_inet && !m_inet->runInference(imgTensor, m_imap_buffer.data())) {
            if (logger_patch) logger_patch->error("[Patchifier] inet->runInference (tensor) failed");
            std::fill(m_imap_buffer.begin(), m_imap_buffer.end(), 0.0f);
        } else {
            if (logger_patch) logger_patch->info("[Patchifier] inet->runInference (tensor) successful");
        }
    }
#else
    // ONNX Runtime not available, use AMBA models
    if (logger_patch) logger_patch->info("[Patchifier] About to call fnet->runInference (tensor)");
    if (!m_fnet->runInference(imgTensor, m_fmap_buffer.data())) {
        if (logger_patch) logger_patch->error("[Patchifier] fnet->runInference (tensor) failed");
        std::fill(m_fmap_buffer.begin(), m_fmap_buffer.end(), 0.0f);
    } else {
        if (logger_patch) logger_patch->info("[Patchifier] fnet->runInference (tensor) successful");
    }
    
    if (logger_patch) logger_patch->info("[Patchifier] About to call inet->runInference (tensor)");
    if (!m_inet->runInference(imgTensor, m_imap_buffer.data())) {
        if (logger_patch) logger_patch->error("[Patchifier] inet->runInference (tensor) failed");
        std::fill(m_imap_buffer.begin(), m_imap_buffer.end(), 0.0f);
    } else {
        if (logger_patch) logger_patch->info("[Patchifier] inet->runInference (tensor) successful");
    }
#endif
    
    // Copy fmap buffer to output
    std::memcpy(fmap, m_fmap_buffer.data(), 128 * fmap_H * fmap_W * sizeof(float));
    
    // Extract image data for color extraction (if needed)
    // NOTE: Use tensor dimensions for color extraction (full resolution)
    // But patches are extracted at model input size (H, W) which may be smaller
    std::vector<uint8_t> image_data;
    const uint8_t* image_for_colors = nullptr;
    void* tensor_data = ea_tensor_data(imgTensor);
    if (tensor_data != nullptr) {
        image_data.resize(H_tensor * W_tensor * 3);
        const uint8_t* src = static_cast<const uint8_t*>(tensor_data);
        std::memcpy(image_data.data(), src, H_tensor * W_tensor * 3);
        image_for_colors = image_data.data();
    }
    
    // Extract patches using helper function (avoids duplicate inference)
    // Pass both model input size (H, W) for patch extraction and tensor size for color extraction
    extractPatchesAfterInference(H, W, fmap_H, fmap_W, M, fmap, imap, gmap, patches, clr, 
                                 image_for_colors, H_tensor, W_tensor);
}
#endif

int Patchifier::getOutputHeight() const
{
#ifdef USE_ONNX_RUNTIME
    if (m_useOnnxRuntime && m_fnet_onnx != nullptr) {
        return m_fnet_onnx->getOutputHeight();
    }
#endif
    if (m_fnet != nullptr) {
        return m_fnet->getOutputHeight();
    }
    return 0;
}

int Patchifier::getOutputWidth() const
{
#ifdef USE_ONNX_RUNTIME
    if (m_useOnnxRuntime && m_fnet_onnx != nullptr) {
        return m_fnet_onnx->getOutputWidth();
    }
#endif
    if (m_fnet != nullptr) {
        return m_fnet->getOutputWidth();
    }
    return 0;
}

int Patchifier::getInputHeight() const
{
#ifdef USE_ONNX_RUNTIME
    if (m_useOnnxRuntime && m_fnet_onnx != nullptr) {
        return m_fnet_onnx->getInputHeight();
    }
#endif
    if (m_fnet != nullptr) {
        return m_fnet->getInputHeight();
    }
    return 0;
}

int Patchifier::getInputWidth() const
{
#ifdef USE_ONNX_RUNTIME
    if (m_useOnnxRuntime && m_fnet_onnx != nullptr) {
        return m_fnet_onnx->getInputWidth();
    }
#endif
    if (m_fnet != nullptr) {
        return m_fnet->getInputWidth();
    }
    return 0;
}

