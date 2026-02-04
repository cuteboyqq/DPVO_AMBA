// patch_graph.cpp
#include "patch_graph.hpp"
#include <cstring>
#include <cmath>
#include <random>

// ------------------------------------------------------------
// Constructor
// ------------------------------------------------------------
PatchGraph::PatchGraph() {
    reset();
}

// ------------------------------------------------------------
// Reset all graph state
// ------------------------------------------------------------
void PatchGraph::reset() {
    m_n = 0;
    m_m = 0;
    m_num_edges = 0;
    m_num_edges_inac = 0;

    // timestamps
    for (int i = 0; i < N; i++) {
        m_tstamps[i] = 0;
    }

    // poses
    for (int i = 0; i < N; i++) {
        m_poses[i] = SE3(); // default constructor
    }

    // patches
    std::memset(m_patches, 0, sizeof(m_patches));

    // intrinsics
    std::memset(m_intrinsics, 0, sizeof(m_intrinsics));

    // map points
    std::memset(m_points, 0, sizeof(m_points));

    // colors
    std::memset(m_colors, 0, sizeof(m_colors));

    // index mapping
    // CRITICAL: m_index[frame][patch] stores SOURCE FRAME INDEX (where patch was created)
    // Python: index_[frame][patch] stores source frame index (initially frame)
    // Initially, patches are created in their own frame, so source frame = frame index
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            m_index[i][j] = i;   // Source frame index is initially the frame itself
        }
        m_index_map[i] = i * M;
    }

    // active edges
    // Initialize m_net with small random values (not zeros) to match Python behavior
    // Python: self.pg.net is typically initialized with small random values or from previous state
    // Using small random initialization helps the model start with non-zero gradients
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 0.01f);  // Small random values around 0
    for (int e = 0; e < MAX_EDGES; e++) {
        for (int d = 0; d < NET_DIM; d++) {
            m_net[e][d] = dis(gen);
        }
    }
    std::memset(m_ii, 0, sizeof(m_ii));
    std::memset(m_jj, 0, sizeof(m_jj));
    std::memset(m_kk, 0, sizeof(m_kk));
    std::memset(m_weight, 0, sizeof(m_weight));  // Initialize array [MAX_EDGES]
    std::memset(m_target, 0, sizeof(m_target));

    // inactive edges
    std::memset(m_ii_inac, 0, sizeof(m_ii_inac));
    std::memset(m_jj_inac, 0, sizeof(m_jj_inac));
    std::memset(m_kk_inac, 0, sizeof(m_kk_inac));
    std::memset(m_weight_inac, 0, sizeof(m_weight_inac));  // Initialize array [MAX_EDGES]
    std::memset(m_target_inac, 0, sizeof(m_target_inac));
}

// ------------------------------------------------------------
// Normalize patch depths (used before BA)
// ------------------------------------------------------------
void PatchGraph::normalize() {
    if (m_m == 0) return;

    float sum = 0.0f;
    int count = 0;

    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < M; j++) {
            float d = m_patches[i][j][2][1][1];
            if (d > 0.0f) {
                sum += d;
                count++;
            }
        }
    }

    if (count == 0) return;

    float mean_depth = sum / count;
    _normalizeDepth(mean_depth);
}

// ------------------------------------------------------------
// Scale all depths by a factor
// ------------------------------------------------------------
void PatchGraph::_normalizeDepth(float scale) {
    if (scale <= 0.0f) return;

    float inv = 1.0f / scale;

    for (int i = 0; i < m_n; i++) {
        for (int j = 0; j < M; j++) {
            for (int y = 0; y < P; y++) {
                for (int x = 0; x < P; x++) {
                    m_patches[i][j][2][y][x] *= inv;
                }
            }
        }
    }

    // Also scale poses translation (DPVO convention)
    for (int i = 0; i < m_n; i++) {
        m_poses[i].t[0] *= inv;
        m_poses[i].t[1] *= inv;
        m_poses[i].t[2] *= inv;
    }
}
