#pragma once
#include "patch_graph.hpp"
#include "net.hpp"
#include <cstdint>
#include <vector>

struct DPVOConfig {
    int PATCHES_PER_FRAME;
    int BUFFER_SIZE;
    int PATCH_SIZE;
    int MIXED_PRECISION;
    int LOOP_CLOSURE;
    int MAX_EDGE_AGE;
    int KEYFRAME_INDEX;
    int KEYFRAME_THRESH;
    int PATCH_LIFETIME;
    int REMOVAL_WINDOW;
};

class DPVO {
public:
    DPVO(const DPVOConfig& cfg, int ht, int wd);
    ~DPVO();

    void run(int64_t timestamp, const uint8_t* image, const float intrinsics[4], int H, int W);
    void terminate();

private:
    void update();
    void keyframe();

    void edgesForward(std::vector<int>& kk, std::vector<int>& jj);
    void edgesBackward(std::vector<int>& kk, std::vector<int>& jj);
    void appendFactors(const std::vector<int>& kk, const std::vector<int>& jj);
    void removeFactors(const bool* mask, bool store);

    float motionMagnitude(int i, int j);

    // Helpers for indexing
    inline int imap_idx(int i, int j, int k) const { return i * m_cfg.PATCHES_PER_FRAME * m_DIM + j * m_DIM + k; }
    inline int gmap_idx(int i, int j, int c, int y, int x) const {
        return i * m_cfg.PATCHES_PER_FRAME * 128 * m_P * m_P +
               j * 128 * m_P * m_P +
               c * m_P * m_P +
               y * m_P +
               x;
    }
    inline int fmap1_idx(int b, int m, int c, int y, int x) const {
        return b * m_mem * 128 * m_fmap1_H * m_fmap1_W +
               m * 128 * m_fmap1_H * m_fmap1_W +
               c * m_fmap1_H * m_fmap1_W +
               y * m_fmap1_W +
               x;
    }
    inline int fmap2_idx(int b, int m, int c, int y, int x) const {
        return b * m_mem * 128 * m_fmap2_H * m_fmap2_W +
               m * 128 * m_fmap2_H * m_fmap2_W +
               c * m_fmap2_H * m_fmap2_W +
               y * m_fmap2_W +
               x;
    }

private:
    DPVOConfig m_cfg;
    PatchGraph m_pg;

    int m_ht, m_wd;
    int m_counter;
    bool m_is_initialized;

    int m_DIM;       // feature dimension
    int m_P;         // patch size
    int m_pmem, m_mem;
    int m_fmap1_H, m_fmap1_W;
    int m_fmap2_H, m_fmap2_W;

    float* m_imap;
    float* m_gmap;
    float* m_fmap1;
    float* m_fmap2;

    std::vector<int64_t> m_tlist;

    // ---- Patchifier for extracting patches ----
    Patchifier m_patchifier;
};
