#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include "dla_config.hpp"

// Forward declarations
class FNetInference;
class INetInference;

// Patchifier class
class Patchifier {
public:
    Patchifier(int patch_size = 3, int DIM = 64);
    Patchifier(int patch_size, int DIM, Config_S* config); // Constructor with models
    ~Patchifier();
    
    void setModels(Config_S* fnetConfig, Config_S* inetConfig);

    // forward function
    // image: normalized float image [C, H, W] with values in range [-0.5, 1.5] (Python: 2 * (image / 255.0) - 0.5)
    void forward(const float* image, int H, int W,
                 float* fmap, float* imap, float* gmap,
                 float* patches, uint8_t* clr,
                 int patches_per_image = 8);

private:
    int m_patch_size;
    int m_DIM;
    
    // Model inference objects
    std::unique_ptr<FNetInference> m_fnet;
    std::unique_ptr<INetInference> m_inet;
    
    // Temporary buffers for model outputs
    std::vector<float> m_fmap_buffer;
    std::vector<float> m_imap_buffer;
};

