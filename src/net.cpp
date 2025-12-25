#include "net.hpp"
#include "correlation_kernel.hpp"

Patchifier::Patchifier(int patch_size, int DIM)
    : m_patch_size(patch_size), m_DIM(DIM)
{}

// Forward pass: fill fmap, imap, gmap, patches, clr
void Patchifier::forward(
    const uint8_t* image,
    int H, int W,
    float* fmap,     // [128, H, W]
    float* imap,     // [DIM, H, W]
    float* gmap,     // [M, 128, P, P]
    float* patches,  // [M, 3, P, P]
    uint8_t* clr,    // [M, 3]
    int M
) {
    // ------------------------------------------------
    // 1. Image → float grid (for patches)
    // ------------------------------------------------
    std::vector<float> grid(3 * H * W);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H*W; i++)
            grid[c*H*W + i] = image[c*H*W + i] / 255.0f;

    // ------------------------------------------------
    // 2. Generate RANDOM coords (Python RANDOM mode)
    // ------------------------------------------------
    std::vector<float> coords(M * 2);
    for (int m = 0; m < M; m++) {
        coords[m*2 + 0] = 1 + rand() % (W - 2);
        coords[m*2 + 1] = 1 + rand() % (H - 2);
    }

    // ------------------------------------------------
    // 3. Patchify grid → patches (RGB)
    // ------------------------------------------------
    patchify_cpu(
        grid.data(), coords.data(),
        M, 3, H, W,
        m_patch_size / 2,
        patches
    );

    // ------------------------------------------------
    // 4. Patchify fmap → gmap
    // ------------------------------------------------
    patchify_cpu(
        fmap, coords.data(),
        M, 128, H, W,
        m_patch_size / 2,
        gmap
    );

    // ------------------------------------------------
    // 5. imap sampling (radius = 0)
    // ------------------------------------------------
    patchify_cpu(
        imap, coords.data(),
        M, m_DIM, H, W,
        0,
        imap   // reuse buffer shape [M, DIM, 1, 1]
    );

    // ------------------------------------------------
    // 6. Color for visualization
    // ------------------------------------------------
    for (int m = 0; m < M; m++) {
        int x = static_cast<int>(coords[m*2 + 0]);
        int y = static_cast<int>(coords[m*2 + 1]);
        for (int c = 0; c < 3; c++)
            clr[m*3 + c] = image[c*H*W + y*W + x];
    }
}

// void Patchifier::forward(const uint8_t* image, int H, int W,
//                          float* fmap, float* imap, float* gmap,
//                          float* patches, uint8_t* clr,
//                          int patches_per_image) {

//     // 1. Fill fmap & imap with dummy features (replace with real network if needed)
//     int fmap_size = 128 * H * W;
//     int imap_size = m_DIM * H * W;
//     for (int i = 0; i < fmap_size; i++) fmap[i] = static_cast<float>(rand())/RAND_MAX;
//     for (int i = 0; i < imap_size; i++) imap[i] = static_cast<float>(rand())/RAND_MAX;

//     // 2. Extract patches from image and imap/gmap
//     extractPatches(image, H, W, patches, clr, patches_per_image);

//     // 3. Fill gmap with dummy patch features
//     for (int i = 0; i < patches_per_image * 128 * m_patch_size * m_patch_size; i++)
//         gmap[i] = static_cast<float>(rand())/RAND_MAX;
// }

// // Simple patch extraction (random centroids)
// void Patchifier::extractPatches(const uint8_t* image, int H, int W,
//                                 float* patches, uint8_t* clr,
//                                 int patches_per_image) {
//     for (int p = 0; p < patches_per_image; p++) {
//         int cx = rand() % (W - m_patch_size);
//         int cy = rand() % (H - m_patch_size);

//         // extract patch
//         for (int c = 0; c < 3; c++) {
//             for (int y = 0; y < m_patch_size; y++) {
//                 for (int x = 0; x < m_patch_size; x++) {
//                     int idx_img = (c * H + (cy+y)) * W + (cx+x);
//                     int idx_patch = (p*3 + c)*m_patch_size*m_patch_size + y*m_patch_size + x;
//                     patches[idx_patch] = static_cast<float>(image[idx_img]) / 255.0f;
//                 }
//             }

//             // color for patch
//             clr[p*3 + c] = image[(c*H + cy) * W + cx];
//         }
//     }
// }
