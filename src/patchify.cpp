#include "patchify.hpp"
#include "fnet.hpp"
#include "inet.hpp"
#include "correlation_kernel.hpp"
#include "dla_config.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>
#include <spdlog/spdlog.h>

// =================================================================================================
// Patchifier Implementation
// =================================================================================================
Patchifier::Patchifier(int patch_size, int DIM)
    : m_patch_size(patch_size), m_DIM(DIM), m_fnet(nullptr), m_inet(nullptr)
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
    
    if (fnetConfig != nullptr)
    {
        m_fnet = std::make_unique<FNetInference>(fnetConfig);
    }
    if (inetConfig != nullptr)
    {
        m_inet = std::make_unique<INetInference>(inetConfig);
    }
}

// Forward pass: fill fmap, imap, gmap, patches, clr
// Note: fmap and imap are at 1/4 resolution (RES=4), but image and coords are at full resolution
// image: normalized float image [C, H, W] with values in range [-0.5, 1.5] (Python: 2 * (image / 255.0) - 0.5)
void Patchifier::forward(
    const float *image,  // Normalized float image [C, H, W] in range [-0.5, 1.5]
    int H, int W,   // Actual image dimensions (may differ from model input)
    float *fmap,    // [128, model_H/4, model_W/4] - at 1/4 resolution of model input
    float *imap,    // [DIM, model_H/4, model_W/4] - at 1/4 resolution of model input
    float *gmap,    // [M, 128, P, P]
    float *patches, // [M, 3, P, P]
    uint8_t *clr,   // [M, 3]
    int M)
{
    const int RES = 4;  // Resolution factor (Python RES=4)
    const int inet_output_channels = 384;  // INet model output channels (not m_DIM which defaults to 64)
    
    // Get model output dimensions (models resize input internally)
    // fmap/imap buffers are sized based on model output, not actual image size
    int fmap_H = 0, fmap_W = 0;
    if (m_fnet != nullptr && m_inet != nullptr)
    {
        // Use model output dimensions (models output at 1/4 of their input size)
        fmap_H = m_fnet->getOutputHeight();  // e.g., 120 (480/4)
        fmap_W = m_fnet->getOutputWidth();   // e.g., 160 (640/4)
        
        // Validate that fnet and inet have same output dimensions
        if (fmap_H != m_inet->getOutputHeight() || fmap_W != m_inet->getOutputWidth())
        {
            throw std::runtime_error("FNet and INet output dimension mismatch");
        }
    }
    else
    {
        // Fallback: calculate from actual image dimensions (should not happen if models are set)
        fmap_H = H / RES;
        fmap_W = W / RES;
    }

    // ------------------------------------------------
    // 1. Run fnet and inet inference to get fmap and imap
    // Pass normalized float image directly to models (matching Python)
    // Models will convert to uint8 internally if needed
    // ------------------------------------------------
    if (m_fnet != nullptr && m_inet != nullptr)
    {
        // Allocate temporary buffers for model output size (1/4 resolution)
        if (m_fmap_buffer.size() != 128 * fmap_H * fmap_W)
        {
            m_fmap_buffer.resize(128 * fmap_H * fmap_W);
        }
        // INet outputs 384 channels, so use that instead of m_DIM
        // m_DIM might be 64 (default), but we need 384 for INet output
        if (m_imap_buffer.size() != inet_output_channels * fmap_H * fmap_W)
        {
            m_imap_buffer.resize(inet_output_channels * fmap_H * fmap_W);
        }
        auto logger_patch = spdlog::get("fnet");
        if (!logger_patch) {
            logger_patch = spdlog::get("inet");
        }

        
        if (logger_patch) logger_patch->info("[Patchifier] About to call fnet->runInference");
        // Run fnet inference (models will resize input internally)
        // Pass normalized float image directly (matching Python)
        if (!m_fnet->runInference(image, H, W, m_fmap_buffer.data()))
        {
            if (logger_patch) logger_patch->error("[Patchifier] fnet->runInference failed");
            // Fallback: zero fill if inference fails
            std::fill(m_fmap_buffer.begin(), m_fmap_buffer.end(), 0.0f);
        }else{
            if (logger_patch) logger_patch->info("[Patchifier] fnet->runInference successful");
        }

        if (logger_patch) logger_patch->info("[Patchifier] About to call inet->runInference");
        
        // Pass normalized float image directly (matching Python)
        if (!m_inet->runInference(image, H, W, m_imap_buffer.data()))
        {
            if (logger_patch) logger_patch->error("[Patchifier] inet->runInference failed");
            // Fallback: zero fill if inference fails
            std::fill(m_imap_buffer.begin(), m_imap_buffer.end(), 0.0f);
        } else {
            if (logger_patch) logger_patch->info("[Patchifier] inet->runInference successful");
            // Verify m_imap_buffer is populated (not all zeros)
            float imap_buffer_min = *std::min_element(m_imap_buffer.begin(), m_imap_buffer.end());
            float imap_buffer_max = *std::max_element(m_imap_buffer.begin(), m_imap_buffer.end());
            int imap_buffer_zero_count = 0;
            int imap_buffer_nonzero_count = 0;
            for (size_t i = 0; i < m_imap_buffer.size(); i++) {
                if (m_imap_buffer[i] == 0.0f) imap_buffer_zero_count++;
                else imap_buffer_nonzero_count++;
            }
            if (logger_patch) {
                logger_patch->info("[Patchifier] m_imap_buffer stats - size={}, zero_count={}, nonzero_count={}, min={}, max={}",
                                    m_imap_buffer.size(), imap_buffer_zero_count, imap_buffer_nonzero_count, 
                                    imap_buffer_min, imap_buffer_max);
            }
        }

        if (logger_patch) logger_patch->info("[Patchifier] About to memcpy fmap, size={}", 128 * fmap_H * fmap_W * sizeof(float));
        // Copy to output buffers (already at 1/4 resolution of model input)
        std::memcpy(fmap, m_fmap_buffer.data(), 128 * fmap_H * fmap_W * sizeof(float));
        if (logger_patch) logger_patch->info("[Patchifier] fmap memcpy completed");
        
        // NOTE: imap parameter is actually a buffer for patch features [M, DIM], not the full feature map
        // We need to extract patches from the full feature map [384, 120, 160] using patchify_cpu_safe
        // So we don't memcpy the full feature map to imap - instead we'll extract patches later
        if (logger_patch) logger_patch->info("[Patchifier] Skipping imap memcpy - will extract patches later");
    }
    else
    {
        // Fallback: zero fill if models not available
        std::fill(fmap, fmap + 128 * fmap_H * fmap_W, 0.0f);
        // imap will be zero-filled later in patchify_cpu_safe fallback
    }

    printf("[Patchifier] About to create grid, H=%d, W=%d\n", H, W);
    fflush(stdout);
    
    // ------------------------------------------------
    // 3. Image → float grid (for patches) - full resolution
    // Image is already normalized float [-0.5, 1.5], use directly
    // ------------------------------------------------
    std::vector<float> grid(3 * H * W);
    printf("[Patchifier] Grid created, size=%zu\n", grid.size());
    fflush(stdout);
    // Image is already normalized, so use it directly (no need to divide by 255.0)
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H * W; i++)
            grid[c * H * W + i] = image[c * H * W + i];

    printf("[Patchifier] About to create coords, M=%d\n", M);
    fflush(stdout);
    
    // ------------------------------------------------
    // 4. Generate RANDOM coords (Python RANDOM mode) - full resolution
    // ------------------------------------------------
    std::vector<float> coords(M * 2);
    for (int m = 0; m < M; m++)
    {
        coords[m * 2 + 0] = 1 + rand() % (W - 2); // Full resolution coordinates
        coords[m * 2 + 1] = 1 + rand() % (H - 2); // Full resolution coordinates
    }
    printf("[Patchifier] Coords created\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (patches)\n");
    fflush(stdout);
    
    // ------------------------------------------------
    // 5. Patchify grid → patches (RGB) - full resolution
    // ------------------------------------------------
    patchify_cpu_safe(
        grid.data(), coords.data(),
        M, 3, H, W,
        m_patch_size / 2,
        patches);
    
    printf("[Patchifier] patchify_cpu_safe (patches) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to create fmap_coords\n");
    fflush(stdout);
    
    // ------------------------------------------------
    // 6. Patchify fmap → gmap - scale coords to feature map resolution
    // ------------------------------------------------
    // fmap is at model output resolution (fmap_H x fmap_W), not necessarily 1/4 of image
    // Scale coordinates by the ratio of feature map to image dimensions
    std::vector<float> fmap_coords(M * 2);
    float scale_x = static_cast<float>(fmap_W) / static_cast<float>(W);
    float scale_y = static_cast<float>(fmap_H) / static_cast<float>(H);
    for (int m = 0; m < M; m++)
    {
        float fx = coords[m * 2 + 0] * scale_x;
        float fy = coords[m * 2 + 1] * scale_y;
        // Clamp to feature map bounds to prevent out-of-bounds access
        fmap_coords[m * 2 + 0] = std::max(0.0f, std::min(static_cast<float>(fmap_W - 1), fx));
        fmap_coords[m * 2 + 1] = std::max(0.0f, std::min(static_cast<float>(fmap_H - 1), fy));
    }
    printf("[Patchifier] fmap_coords created\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (gmap)\n");
    fflush(stdout);
    patchify_cpu_safe(
        fmap, fmap_coords.data(),
        M, 128, fmap_H, fmap_W, // Use 1/4 resolution dimensions
        m_patch_size / 2,
        gmap);
    printf("[Patchifier] patchify_cpu_safe (gmap) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (imap)\n");
    fflush(stdout);
    // ------------------------------------------------
    // 7. imap sampling (radius = 0) - scale coords to 1/4 resolution
    // Extract patches from the full feature map stored in m_imap_buffer
    // ------------------------------------------------
    // Use m_imap_buffer (full feature map [384, 120, 160]) as source
    // Extract patches and write to imap (which is [M, DIM] = [8, 384])
    if (m_fnet != nullptr && m_inet != nullptr) {
        // Verify m_imap_buffer has valid data before extracting patches
        float imap_buffer_sample_min = *std::min_element(m_imap_buffer.begin(), 
                                                          m_imap_buffer.begin() + std::min(static_cast<size_t>(100), m_imap_buffer.size()));
        float imap_buffer_sample_max = *std::max_element(m_imap_buffer.begin(), 
                                                          m_imap_buffer.begin() + std::min(static_cast<size_t>(100), m_imap_buffer.size()));
        printf("[Patchifier] Before patchify_cpu_safe (imap): m_imap_buffer sample range: [%f, %f], size=%zu\n", 
               imap_buffer_sample_min, imap_buffer_sample_max, m_imap_buffer.size());
        fflush(stdout);
        
        // Log coordinates before extraction
        printf("[Patchifier] fmap_coords for imap extraction:\n");
        for (int m = 0; m < std::min(M, 8); m++) {
            printf("[Patchifier]   Patch %d: x=%.2f, y=%.2f (fmap_H=%d, fmap_W=%d)\n", 
                   m, fmap_coords[m*2+0], fmap_coords[m*2+1], fmap_H, fmap_W);
        }
        fflush(stdout);
        
        // Extract patches from the full feature map
        patchify_cpu_safe(
            m_imap_buffer.data(), fmap_coords.data(),  // Source: full feature map, coords at 1/4 resolution
            M, inet_output_channels, fmap_H, fmap_W,   // M patches, 384 channels, 120x160 feature map
            0,                                          // radius = 0 (single pixel)
            imap                                        // Output: [M, DIM, 1, 1] = [8, 384, 1, 1]
        );
        
        // Verify patches were extracted correctly
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
        // Fallback: zero fill if models not available
        std::fill(imap, imap + M * m_DIM, 0.0f);
        printf("[Patchifier] WARNING: Models not available, zero-filling imap\n");
        fflush(stdout);
    }
    printf("[Patchifier] patchify_cpu_safe (imap) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to extract colors\n");
    fflush(stdout);
    // ------------------------------------------------
    // 8. Color for visualization - full resolution
    // Convert normalized float image back to uint8 for color extraction
    // ------------------------------------------------
    for (int m = 0; m < M; m++)
    {
        int x = static_cast<int>(coords[m * 2 + 0]);
        int y = static_cast<int>(coords[m * 2 + 1]);
        for (int c = 0; c < 3; c++) {
            // Image is normalized float [-0.5, 1.5], convert back to uint8 [0, 255]
            float normalized_val = image[c * H * W + y * W + x];
            float mapped_val = (normalized_val + 0.5f) / 2.0f * 255.0f;
            clr[m * 3 + c] = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, mapped_val)));
        }
    }
    printf("[Patchifier] Colors extracted\n");
    fflush(stdout);
    
    printf("[Patchifier] forward() about to return\n");
    fflush(stdout);
}

