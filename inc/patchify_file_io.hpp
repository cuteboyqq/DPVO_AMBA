#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include "logger.hpp"

/**
 * Utility functions for saving Patchifier output parameters to binary files
 * Used for comparison with Python DPVO implementation
 */
namespace patchify_file_io {

/**
 * Save a float array to a binary file
 * @param filename Output filename
 * @param data Pointer to data array
 * @param num_elements Number of elements to write
 * @param logger Optional logger for status messages
 * @param description Optional description for logging
 * @return true if successful, false otherwise
 */
inline bool save_float_array(const std::string& filename, const float* data, size_t num_elements,
                             std::shared_ptr<spdlog::logger> logger = nullptr,
                             const std::string& description = "") {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) {
            logger->error("[Patchifier] Failed to open {} for writing", filename);
        }
        printf("[Patchifier] Failed to open %s for writing\n", filename.c_str());
        fflush(stdout);
        return false;
    }
    
    size_t bytes_written = num_elements * sizeof(float);
    file.write(reinterpret_cast<const char*>(data), bytes_written);
    file.close();
    
    if (logger) {
        if (!description.empty()) {
            logger->info("[Patchifier] Saved {} to {}: {} elements, {} bytes", 
                         description, filename, num_elements, bytes_written);
        } else {
            logger->info("[Patchifier] Saved {} floats to {}: {} bytes", 
                         num_elements, filename, bytes_written);
        }
    }
    printf("[Patchifier] Saved %s to %s: %zu elements, %zu bytes\n",
           description.empty() ? "data" : description.c_str(), filename.c_str(), num_elements, bytes_written);
    fflush(stdout);
    
    return true;
}

/**
 * Save model output (fmap/imap) to binary file
 * @param filename Output filename (e.g., "fnet_frame0.bin")
 * @param data Pointer to model output data [C, H, W]
 * @param C Number of channels
 * @param H Height
 * @param W Width
 * @param logger Optional logger for status messages
 * @param model_name Model name for logging (e.g., "fnet", "inet")
 * @return true if successful, false otherwise
 */
inline bool save_model_output(const std::string& filename, const float* data,
                              int C, int H, int W,
                              std::shared_ptr<spdlog::logger> logger = nullptr,
                              const std::string& model_name = "") {
    size_t num_elements = static_cast<size_t>(C) * H * W;
    std::string description = model_name.empty() ? "model output" : model_name + " output";
    description += " [C=" + std::to_string(C) + ", H=" + std::to_string(H) + ", W=" + std::to_string(W) + "]";
    
    return save_float_array(filename, data, num_elements, logger, description);
}

/**
 * Save coordinates to binary file [M, 2]
 * @param filename Output filename (e.g., "cpp_coords_frame0.bin")
 * @param coords Pointer to coordinates array [M * 2]
 * @param M Number of patches/coordinates
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_coordinates(const std::string& filename, const float* coords, int M,
                             std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::string description = "coordinates [M=" + std::to_string(M) + ", 2]";
    return save_float_array(filename, coords, M * 2, logger, description);
}

/**
 * Save patch data to binary file [M, C, P, P]
 * @param filename Output filename (e.g., "cpp_gmap_frame0.bin")
 * @param patches Pointer to patch data array [M * C * P * P]
 * @param M Number of patches
 * @param C Number of channels
 * @param P Patch size
 * @param logger Optional logger for status messages
 * @param patch_type Type of patch for logging (e.g., "gmap", "imap", "patches")
 * @return true if successful, false otherwise
 */
inline bool save_patch_data(const std::string& filename, const float* patches,
                            int M, int C, int P,
                            std::shared_ptr<spdlog::logger> logger = nullptr,
                            const std::string& patch_type = "patches") {
    size_t num_elements = static_cast<size_t>(M) * C * P * P;
    std::string description = patch_type + " [M=" + std::to_string(M) + ", C=" + std::to_string(C) + 
                              ", P=" + std::to_string(P) + ", P=" + std::to_string(P) + "]";
    
    return save_float_array(filename, patches, num_elements, logger, description);
}

/**
 * Save patch data with different height/width dimensions [M, C, H, W]
 * @param filename Output filename (e.g., "cpp_imap_frame0.bin")
 * @param patches Pointer to patch data array [M * C * H * W]
 * @param M Number of patches
 * @param C Number of channels
 * @param H Height dimension
 * @param W Width dimension
 * @param logger Optional logger for status messages
 * @param patch_type Type of patch for logging (e.g., "imap")
 * @return true if successful, false otherwise
 */
inline bool save_patch_data_hw(const std::string& filename, const float* patches,
                               int M, int C, int H, int W,
                               std::shared_ptr<spdlog::logger> logger = nullptr,
                               const std::string& patch_type = "patches") {
    size_t num_elements = static_cast<size_t>(M) * C * H * W;
    std::string description = patch_type + " [M=" + std::to_string(M) + ", C=" + std::to_string(C) + 
                              ", H=" + std::to_string(H) + ", W=" + std::to_string(W) + "]";
    
    return save_float_array(filename, patches, num_elements, logger, description);
}

} // namespace patchify_file_io

