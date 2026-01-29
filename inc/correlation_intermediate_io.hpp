#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include "logger.hpp"

/**
 * Utility functions for saving correlation intermediate results for step-by-step comparison
 * Similar to BA intermediate results saving
 */
namespace correlation_intermediate_io {

/**
 * Save normalized coordinates for correlation sampling
 * @param frame_num Frame number (for filename)
 * @param coords_norm Normalized coordinates [num_active, D, D, P, P, 2] (x_norm, y_norm)
 * @param num_active Number of active edges
 * @param D Correlation window diameter
 * @param P Patch size
 * @param level Pyramid level (0 or 1) for filename suffix
 * @param logger Optional logger
 */
inline void save_normalized_coords(
    int frame_num,
    const float* coords_norm,
    int num_active,
    int D,
    int P,
    int level,
    std::shared_ptr<spdlog::logger> logger = nullptr)
{
    std::string level_suffix = (level == 0) ? "_level0" : "_level1";
    std::string filename = "bin_file/corr_intermediate_frame" + std::to_string(frame_num) + level_suffix + "_normalized_coords.bin";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        size_t num_elements = static_cast<size_t>(num_active) * D * D * P * P * 2;
        file.write(reinterpret_cast<const char*>(coords_norm), sizeof(float) * num_elements);
        file.close();
        if (logger) logger->info("Saved normalized coordinates to {} ({} elements)", filename, num_elements);
    } else {
        if (logger) logger->error("Failed to open {} for writing", filename);
    }
}

/**
 * Save pixel coordinates for correlation sampling
 * @param frame_num Frame number
 * @param coords_pixel Pixel coordinates [num_active, D, D, P, P, 2] (x_pixel, y_pixel)
 * @param num_active Number of active edges
 * @param D Correlation window diameter
 * @param P Patch size
 * @param level Pyramid level (0 or 1) for filename suffix
 * @param logger Optional logger
 */
inline void save_pixel_coords(
    int frame_num,
    const float* coords_pixel,
    int num_active,
    int D,
    int P,
    int level,
    std::shared_ptr<spdlog::logger> logger = nullptr)
{
    std::string level_suffix = (level == 0) ? "_level0" : "_level1";
    std::string filename = "bin_file/corr_intermediate_frame" + std::to_string(frame_num) + level_suffix + "_pixel_coords.bin";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        size_t num_elements = static_cast<size_t>(num_active) * D * D * P * P * 2;
        file.write(reinterpret_cast<const char*>(coords_pixel), sizeof(float) * num_elements);
        file.close();
        if (logger) logger->info("Saved pixel coordinates to {} ({} elements)", filename, num_elements);
    } else {
        if (logger) logger->error("Failed to open {} for writing", filename);
    }
}

/**
 * Save out-of-bounds flags for correlation sampling
 * @param frame_num Frame number
 * @param oob_flags Out-of-bounds flags [num_active, D, D, P, P] (1 if out-of-bounds, 0 if in-bounds)
 * @param num_active Number of active edges
 * @param D Correlation window diameter
 * @param P Patch size
 * @param level Pyramid level (0 or 1) for filename suffix
 * @param logger Optional logger
 */
inline void save_oob_flags(
    int frame_num,
    const int32_t* oob_flags,
    int num_active,
    int D,
    int P,
    int level,
    std::shared_ptr<spdlog::logger> logger = nullptr)
{
    std::string level_suffix = (level == 0) ? "_level0" : "_level1";
    std::string filename = "bin_file/corr_intermediate_frame" + std::to_string(frame_num) + level_suffix + "_oob_flags.bin";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        size_t num_elements = static_cast<size_t>(num_active) * D * D * P * P;
        file.write(reinterpret_cast<const char*>(oob_flags), sizeof(int32_t) * num_elements);
        file.close();
        if (logger) logger->info("Saved OOB flags to {} ({} elements)", filename, num_elements);
    } else {
        if (logger) logger->error("Failed to open {} for writing", filename);
    }
}

/**
 * Save sampled fmap values for a subset of edges and positions
 * @param frame_num Frame number
 * @param sampled_fmap Sampled fmap values [num_sample_edges, num_sample_positions, feature_dim]
 * @param num_sample_edges Number of edges to sample
 * @param num_sample_positions Number of positions to sample per edge
 * @param feature_dim Feature dimension
 * @param logger Optional logger
 */
inline void save_sampled_fmap(
    int frame_num,
    const float* sampled_fmap,
    int num_sample_edges,
    int num_sample_positions,
    int feature_dim,
    std::shared_ptr<spdlog::logger> logger = nullptr)
{
    std::string filename = "bin_file/corr_intermediate_frame" + std::to_string(frame_num) + "_sampled_fmap.bin";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        size_t num_elements = static_cast<size_t>(num_sample_edges) * num_sample_positions * feature_dim;
        file.write(reinterpret_cast<const char*>(sampled_fmap), sizeof(float) * num_elements);
        file.close();
        if (logger) logger->info("Saved sampled fmap to {} ({} elements)", filename, num_elements);
    } else {
        if (logger) logger->error("Failed to open {} for writing", filename);
    }
}

/**
 * Save gmap patch values for a subset of edges and positions
 * @param frame_num Frame number
 * @param gmap_patches gmap patch values [num_sample_edges, num_sample_positions, feature_dim]
 * @param num_sample_edges Number of edges to sample
 * @param num_sample_positions Number of positions to sample per edge
 * @param feature_dim Feature dimension
 * @param logger Optional logger
 */
inline void save_gmap_patches(
    int frame_num,
    const float* gmap_patches,
    int num_sample_edges,
    int num_sample_positions,
    int feature_dim,
    std::shared_ptr<spdlog::logger> logger = nullptr)
{
    std::string filename = "bin_file/corr_intermediate_frame" + std::to_string(frame_num) + "_gmap_patches.bin";
    std::ofstream file(filename, std::ios::binary);
    if (file.is_open()) {
        size_t num_elements = static_cast<size_t>(num_sample_edges) * num_sample_positions * feature_dim;
        file.write(reinterpret_cast<const char*>(gmap_patches), sizeof(float) * num_elements);
        file.close();
        if (logger) logger->info("Saved gmap patches to {} ({} elements)", filename, num_elements);
    } else {
        if (logger) logger->error("Failed to open {} for writing", filename);
    }
}

} // namespace correlation_intermediate_io

