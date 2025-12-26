#include "correlation_kernel.hpp"
#include <cmath>
#include <cstring>
#include <vector>

inline bool within_bounds(int h, int w, int H, int W)
{
    return (h >= 0 && h < H && w >= 0 && w < W);
}

void patchify_cpu(
    const float* fmap,    // [C][H][W]
    const float* coords,  // [M][2]
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap           // [M][C][D][D]
)
{
    const int D = 2 * radius + 2;

    // zero output (matches CUDA behavior)
    std::memset(gmap, 0, sizeof(float) * M * C * D * D);

    for (int m = 0; m < M; m++) {

        const float x = coords[m*2 + 0];
        const float y = coords[m*2 + 1];

        const int cx = static_cast<int>(std::floor(x));
        const int cy = static_cast<int>(std::floor(y));

        for (int ii = 0; ii < D; ii++) {
            for (int jj = 0; jj < D; jj++) {

                const int i = cy + (ii - radius);
                const int j = cx + (jj - radius);

                if (!within_bounds(i, j, H, W))
                    continue;

                for (int c = 0; c < C; c++) {

                    // fmap[c][i][j]
                    const int fmap_idx = 
                            (c * (H * W)) + (i * W) + j;
                    // (c * H + i) * W + j;

                    // gmap[m][c][ii][jj]
                    const int gmap_idx =
                            m * C * D * D +
                            c * D * D +
                            ii * D +
                            jj;

                            // ((m * C + c) * D + ii) * D + jj;

                    gmap[gmap_idx] = fmap[fmap_idx];
                }
            }
        }
    }
}


void patchify_cpu_safe(
    const float* fmap,    // [C][H][W]
    const float* coords,  // [M][2]
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap           // [M][C][D][D]
)
{
    const int D = 2 * radius + 2;
    const int fmap_size = C * H * W;
    const int gmap_size = M * C * D * D;

    // --------------------------------------------------
    // 1. Detect aliasing (in-place call)
    // --------------------------------------------------
    const bool inplace = (fmap == gmap);

    // --------------------------------------------------
    // 2. Prepare safe source pointer
    // --------------------------------------------------
    std::vector<float> fmap_copy;
    const float* src = fmap;

    if (inplace) {
        // Copy entire fmap (DPVO semantics: read original fmap)
        fmap_copy.resize(fmap_size);
        std::memcpy(fmap_copy.data(), fmap,
                    sizeof(float) * fmap_size);
        src = fmap_copy.data();
    }

    // --------------------------------------------------
    // 3. Zero output (DPVO behavior)
    // --------------------------------------------------
    std::memset(gmap, 0, sizeof(float) * gmap_size);

    // --------------------------------------------------
    // 4. Patch extraction
    // --------------------------------------------------
    for (int m = 0; m < M; m++) {

        const int cx = static_cast<int>(std::floor(coords[m*2 + 0]));
        const int cy = static_cast<int>(std::floor(coords[m*2 + 1]));

        const int gmap_m_offset = m * C * D * D;

        for (int c = 0; c < C; c++) {

            const int fmap_c_offset = c * H * W;
            const int gmap_c_offset = gmap_m_offset + c * D * D;

            for (int ii = 0; ii < D; ii++) {
                const int y = cy + ii - radius;
                if ((unsigned)y >= (unsigned)H) continue;

                for (int jj = 0; jj < D; jj++) {
                    const int x = cx + jj - radius;
                    if ((unsigned)x >= (unsigned)W) continue;

                    gmap[gmap_c_offset + ii * D + jj] =
                        src[fmap_c_offset + y * W + x];
                }
            }
        }
    }
}
