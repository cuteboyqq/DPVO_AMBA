#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include "dla_config.hpp"
#include <spdlog/spdlog.h>

// Ambarella CV28
#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

// Forward declarations
class FNetInference;
class INetInference;
#ifdef USE_ONNX_RUNTIME
class FNetInferenceONNX;
class INetInferenceONNX;
#endif

// Patchifier class
class Patchifier {
public:
    Patchifier(int patch_size = 3, int DIM = 64);
    Patchifier(int patch_size, int DIM, Config_S* config); // Constructor with models
    ~Patchifier();
    
    void setModels(Config_S* fnetConfig, Config_S* inetConfig);

    // Tensor-based forward (preferred, avoids conversion)
    void forward(ea_tensor_t* imgTensor,
                 float* fmap, float* imap, float* gmap,
                 float* patches, uint8_t* clr,
                 int patches_per_image = 8);
    
    // Get the coordinates used in the last forward() call
    // Returns coordinates at full resolution [patches_per_image * 2] (x, y pairs)
    const std::vector<float>& getLastCoords() const { return m_last_coords; }
    
    // Get model output dimensions (for fmap1 sizing)
    // Returns 0,0 if models are not set
    int getOutputHeight() const;
    int getOutputWidth() const;
    
    // Get model input dimensions (for patch extraction)
    // Returns 0,0 if models are not set
    int getInputHeight() const;
    int getInputWidth() const;

    // Inference cache support: save/load FNet/INet outputs to avoid re-running inference
    // Call setCachePath() before first forward() to enable caching.
    // First run: inference runs normally and outputs are saved to cache.
    // Subsequent runs: cached outputs are loaded, inference is skipped.
    void setCachePath(const std::string& cachePath);
    bool isCacheEnabled() const { return m_cacheEnabled; }

private:
    // Helper function to extract patches after inference has been run
    // This avoids duplicate inference when tensor-based forward() calls uint8_t* forward()
    void extractPatchesAfterInference(int H, int W, int fmap_H, int fmap_W, int M,
                                      float* fmap, float* imap, float* gmap,
                                      float* patches, uint8_t* clr, const uint8_t* image_for_colors = nullptr,
                                      int H_image = 0, int W_image = 0);
    int m_patch_size;
    int m_DIM;
    
    // Model inference objects (AMBA or ONNX)
#ifdef USE_ONNX_RUNTIME
    std::unique_ptr<FNetInferenceONNX> m_fnet_onnx;
    std::unique_ptr<INetInferenceONNX> m_inet_onnx;
    bool m_useOnnxRuntime = false;
#endif
    std::unique_ptr<FNetInference> m_fnet;
    std::unique_ptr<INetInference> m_inet;
    
    // Temporary buffers for model outputs
    std::vector<float> m_fmap_buffer;
    std::vector<float> m_imap_buffer;
    
    // Store last coordinates used (for patch coordinate storage)
    std::vector<float> m_last_coords;
    
    // Inference cache: save/load FNet/INet outputs to bin files
    std::string m_cachePath;       // Directory for cache files (empty = disabled)
    bool m_cacheEnabled = false;   // Whether caching is active
    int m_cacheFrameCounter = 0;   // Frame counter for cache file naming
    
    // Helper: try to load FNet/INet from cache; returns true on success
    bool _loadFromCache(int fmap_H, int fmap_W, int inet_C);
    // Helper: save FNet/INet to cache after inference
    void _saveToCache(int fmap_H, int fmap_W, int inet_C);
    // Helper: save AMBA FNet/INet outputs + input image + metadata at TARGET_FRAME
    void _saveAmbaOutputsForComparison(ea_tensor_t* imgTensor, int fmap_H, int fmap_W,
                                       int inet_output_channels,
                                       std::shared_ptr<spdlog::logger> logger);
};

