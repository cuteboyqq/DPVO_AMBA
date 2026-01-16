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
// Helper function to extract patches after inference has been run
void Patchifier::extractPatchesAfterInference(int H, int W, int fmap_H, int fmap_W, int M,
                                                float* fmap, float* imap, float* gmap,
                                                float* patches, uint8_t* clr, const uint8_t* image_for_colors)
{
    const int inet_output_channels = 384;
    
    printf("[Patchifier] About to create grid, H=%d, W=%d\n", H, W);
    fflush(stdout);
    
    // ------------------------------------------------
    // Create coordinate grid (like Python's coords_grid_with_index)
    // ------------------------------------------------
    std::vector<float> grid(3 * H * W);
    printf("[Patchifier] Grid created, size=%zu\n", grid.size());
    fflush(stdout);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int idx = y * W + x;
            grid[0 * H * W + idx] = static_cast<float>(x);
            grid[1 * H * W + idx] = static_cast<float>(y);
            grid[2 * H * W + idx] = 1.0f;
        }
    }

    printf("[Patchifier] About to create coords, M=%d\n", M);
    fflush(stdout);
    
    // ------------------------------------------------
    // Generate RANDOM coords (Python RANDOM mode) - full resolution
    // ------------------------------------------------
    m_last_coords.resize(M * 2);
    for (int m = 0; m < M; m++)
    {
        m_last_coords[m * 2 + 0] = 1 + rand() % (W - 2);
        m_last_coords[m * 2 + 1] = 1 + rand() % (H - 2);
    }
    const float* coords = m_last_coords.data();
    printf("[Patchifier] Coords created\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (patches)\n");
    fflush(stdout);
    
    // ------------------------------------------------
    // Patchify grid → patches (RGB) - full resolution
    // ------------------------------------------------
    patchify_cpu_safe(
        grid.data(), coords,
        M, 3, H, W,
        m_patch_size / 2,
        patches);
    
    printf("[Patchifier] patchify_cpu_safe (patches) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to create fmap_coords\n");
    fflush(stdout);
    
    // ------------------------------------------------
    // Patchify fmap → gmap - scale coords to feature map resolution
    // ------------------------------------------------
    std::vector<float> fmap_coords(M * 2);
    float scale_x = static_cast<float>(fmap_W) / static_cast<float>(W);
    float scale_y = static_cast<float>(fmap_H) / static_cast<float>(H);
    for (int m = 0; m < M; m++)
    {
        float fx = coords[m * 2 + 0] * scale_x;
        float fy = coords[m * 2 + 1] * scale_y;
        fmap_coords[m * 2 + 0] = std::max(0.0f, std::min(static_cast<float>(fmap_W - 1), fx));
        fmap_coords[m * 2 + 1] = std::max(0.0f, std::min(static_cast<float>(fmap_H - 1), fy));
    }
    printf("[Patchifier] fmap_coords created\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (gmap)\n");
    fflush(stdout);
    patchify_cpu_safe(
        fmap, fmap_coords.data(),
        M, 128, fmap_H, fmap_W,
        m_patch_size / 2,
        gmap);
    printf("[Patchifier] patchify_cpu_safe (gmap) completed\n");
    fflush(stdout);

    printf("[Patchifier] About to call patchify_cpu_safe (imap)\n");
    fflush(stdout);
    // ------------------------------------------------
    // imap sampling (radius = 0) - extract patches from m_imap_buffer
    // ------------------------------------------------
    if (m_fnet != nullptr && m_inet != nullptr) {
        float imap_buffer_sample_min = *std::min_element(m_imap_buffer.begin(), 
                                                          m_imap_buffer.begin() + std::min(static_cast<size_t>(100), m_imap_buffer.size()));
        float imap_buffer_sample_max = *std::max_element(m_imap_buffer.begin(), 
                                                          m_imap_buffer.begin() + std::min(static_cast<size_t>(100), m_imap_buffer.size()));
        printf("[Patchifier] Before patchify_cpu_safe (imap): m_imap_buffer sample range: [%f, %f], size=%zu\n", 
               imap_buffer_sample_min, imap_buffer_sample_max, m_imap_buffer.size());
        fflush(stdout);
        
        printf("[Patchifier] fmap_coords for imap extraction:\n");
        for (int m = 0; m < std::min(M, 8); m++) {
            printf("[Patchifier]   Patch %d: x=%.2f, y=%.2f (fmap_H=%d, fmap_W=%d)\n", 
                   m, fmap_coords[m*2+0], fmap_coords[m*2+1], fmap_H, fmap_W);
        }
        fflush(stdout);
        
        patchify_cpu_safe(
            m_imap_buffer.data(), fmap_coords.data(),
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

    printf("[Patchifier] About to extract colors\n");
    fflush(stdout);
    // ------------------------------------------------
    // Color for visualization - full resolution
    // ------------------------------------------------
    if (image_for_colors != nullptr) {
        for (int m = 0; m < M; m++)
        {
            int x = static_cast<int>(coords[m * 2 + 0]);
            int y = static_cast<int>(coords[m * 2 + 1]);
            x = std::max(0, std::min(x, W - 1));
            y = std::max(0, std::min(y, H - 1));
            for (int c = 0; c < 3; c++) {
                // Image is in [C, H, W] format (uint8_t)
                clr[m * 3 + c] = image_for_colors[c * H * W + y * W + x];
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
    
    // Get dimensions from tensor
    const size_t* shape = ea_tensor_shape(imgTensor);
    int H = static_cast<int>(shape[EA_H]);
    int W = static_cast<int>(shape[EA_W]);
    
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
    
    auto logger_patch = spdlog::get("fnet");
    if (!logger_patch) {
        logger_patch = spdlog::get("inet");
    }
    
    // Run inference using tensor directly
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
    
    // Copy fmap buffer to output
    std::memcpy(fmap, m_fmap_buffer.data(), 128 * fmap_H * fmap_W * sizeof(float));
    
    // Extract image data for color extraction (if needed)
    std::vector<uint8_t> image_data;
    const uint8_t* image_for_colors = nullptr;
    void* tensor_data = ea_tensor_data(imgTensor);
    if (tensor_data != nullptr) {
        image_data.resize(H * W * 3);
        const uint8_t* src = static_cast<const uint8_t*>(tensor_data);
        std::memcpy(image_data.data(), src, H * W * 3);
        image_for_colors = image_data.data();
    }
    
    // Extract patches using helper function (avoids duplicate inference)
    extractPatchesAfterInference(H, W, fmap_H, fmap_W, M, fmap, imap, gmap, patches, clr, image_for_colors);
}
#endif

int Patchifier::getOutputHeight() const
{
    if (m_fnet != nullptr) {
        return m_fnet->getOutputHeight();
    }
    return 0;
}

int Patchifier::getOutputWidth() const
{
    if (m_fnet != nullptr) {
        return m_fnet->getOutputWidth();
    }
    return 0;
}

