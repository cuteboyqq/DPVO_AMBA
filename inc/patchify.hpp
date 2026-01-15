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
    // image: original uint8 image [C, H, W] with values in range [0, 255]
    // AMBA CV28 models (fnet/inet) will handle normalization internally
    void forward(const uint8_t* image, int H, int W,
                 float* fmap, float* imap, float* gmap,
                 float* patches, uint8_t* clr,
                 int patches_per_image = 8);
    
    // Get the coordinates used in the last forward() call
    // Returns coordinates at full resolution [patches_per_image * 2] (x, y pairs)
    const std::vector<float>& getLastCoords() const { return m_last_coords; }

private:
    int m_patch_size;
    int m_DIM;
    
    // Model inference objects
    std::unique_ptr<FNetInference> m_fnet;
    std::unique_ptr<INetInference> m_inet;
    
    // Temporary buffers for model outputs
    std::vector<float> m_fmap_buffer;
    std::vector<float> m_imap_buffer;
    
    // Store last coordinates used (for patch coordinate storage)
    std::vector<float> m_last_coords;
};

