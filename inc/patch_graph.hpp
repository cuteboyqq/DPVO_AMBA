#pragma once
#include "se3.h"
#include <cstdint>

constexpr int BUFFER_SIZE = 36;
constexpr int PATCHES_PER_FRAME = 8;
constexpr int PATCH_SIZE = 3;
constexpr int MAX_EDGES = 360;
constexpr int NET_DIM = 384;

struct Vec2 { float x, y; };
struct Vec3 { float x, y, z; };

class PatchGraph {
public:
    static constexpr int N = BUFFER_SIZE;
    static constexpr int M = PATCHES_PER_FRAME;
    static constexpr int P = PATCH_SIZE;

    // ---- counters ----
    int m_n;              // number of frames
    int m_m;              // number of patches
    int m_num_edges;
    int m_num_edges_inac;

    // ---- frame data ----
    int64_t m_tstamps[N];
    SE3 m_poses[N];

    // patches: (N, M, 3, P, P)
    float m_patches[N][M][3][P][P];

    int m_ix[N * M];   // patch → frame index Alsiter 2025-12-25 added

    // intrinsics: fx fy cx cy
    float m_intrinsics[N][4];

    // ---- map ----
    Vec3 m_points[N * M];
    uint8_t m_colors[N][M][3];

    // ---- index mapping ----
    // int m_index[N + 1];
    int m_index[N][M];
    int m_index_map[N + 1];

    // ---- active edges ----
    float m_net[MAX_EDGES][NET_DIM];
    int m_ii[MAX_EDGES];
    int m_jj[MAX_EDGES];
    int m_kk[MAX_EDGES];
    float m_weight[MAX_EDGES];
    float  m_target[MAX_EDGES];

    // ---- inactive edges ----
    int m_ii_inac[MAX_EDGES];
    int m_jj_inac[MAX_EDGES];
    int m_kk_inac[MAX_EDGES];
    float m_weight_inac[MAX_EDGES];
    float  m_target_inac[MAX_EDGES];

public:
    PatchGraph();

    void reset();
    void normalize();

private:
    void _normalizeDepth(float scale);
};
