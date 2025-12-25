#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Simple struct for patch
struct Patch {
    std::vector<std::vector<std::vector<float>>> data; // [C][D][D]
    Patch(int C, int D) : data(C, std::vector<std::vector<float>>(D, std::vector<float>(D, 0.0f))) {}
};

struct ColorPatch {
    std::array<uint8_t,3> rgb;
};


class Patchifier {
public:
    Patchifier(int patch_size = 3, int DIM = 64);

    // forward function
    void forward(const uint8_t* image, int H, int W,
                 float* fmap, float* imap, float* gmap,
                 float* patches, uint8_t* clr,
                 int patches_per_image = 8);

private:
    int m_patch_size;
    int m_DIM;

    void extractPatches(const uint8_t* image, int H, int W,
                        float* patches, uint8_t* clr,
                        int patches_per_image);
};
