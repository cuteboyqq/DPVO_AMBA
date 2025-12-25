#pragma once

// fmap   : [C][H][W]
// coords : [M][2]  (x, y)
// gmap   : [M][C][D][D], D = 2*radius + 2

void patchify_cpu(
    const float* fmap,
    const float* coords,
    int M,
    int C,
    int H,
    int W,
    int radius,
    float* gmap
);
