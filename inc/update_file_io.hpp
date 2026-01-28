#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "logger.hpp"

/**
 * Utility functions for saving Update model input/output parameters to binary files
 * Used for comparison with Python ONNX update model inference
 */
namespace update_file_io {

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
            logger->error("[Update] Failed to open {} for writing", filename);
        }
        return false;
    }
    
    size_t bytes_written = num_elements * sizeof(float);
    file.write(reinterpret_cast<const char*>(data), bytes_written);
    file.close();
    
    if (logger) {
        if (!description.empty()) {
            logger->info("[Update] Saved {} to {}: {} elements, {} bytes", 
                         description, filename, num_elements, bytes_written);
        } else {
            logger->info("[Update] Saved {} floats to {}: {} bytes", 
                         num_elements, filename, bytes_written);
        }
    }
    
    return true;
}

/**
 * Save an int32 array to a binary file
 * @param filename Output filename
 * @param data Pointer to data array
 * @param num_elements Number of elements to write
 * @param logger Optional logger for status messages
 * @param description Optional description for logging
 * @return true if successful, false otherwise
 */
inline bool save_int32_array(const std::string& filename, const int32_t* data, size_t num_elements,
                             std::shared_ptr<spdlog::logger> logger = nullptr,
                             const std::string& description = "") {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        if (logger) {
            logger->error("[Update] Failed to open {} for writing", filename);
        }
        return false;
    }
    
    size_t bytes_written = num_elements * sizeof(int32_t);
    file.write(reinterpret_cast<const char*>(data), bytes_written);
    file.close();
    
    if (logger) {
        if (!description.empty()) {
            logger->info("[Update] Saved {} to {}: {} elements, {} bytes", 
                         description, filename, num_elements, bytes_written);
        } else {
            logger->info("[Update] Saved {} int32s to {}: {} bytes", 
                         num_elements, filename, bytes_written);
        }
    }
    
    return true;
}

/**
 * Save update model input: net_input [1, DIM, MAX_EDGE, 1]
 * @param filename Output filename (e.g., "update_net_input_frame101.bin")
 * @param data Pointer to net_input data
 * @param DIM Feature dimension (384)
 * @param MAX_EDGE Maximum number of edges (360)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_net_input(const std::string& filename, const float* data, int DIM, int MAX_EDGE,
                           std::shared_ptr<spdlog::logger> logger = nullptr) {
    size_t num_elements = static_cast<size_t>(1) * DIM * MAX_EDGE * 1;
    std::string description = "net_input [1, " + std::to_string(DIM) + ", " + 
                              std::to_string(MAX_EDGE) + ", 1]";
    return save_float_array(filename, data, num_elements, logger, description);
}

/**
 * Save update model input: inp_input [1, DIM, MAX_EDGE, 1]
 * @param filename Output filename (e.g., "update_inp_input_frame101.bin")
 * @param data Pointer to inp_input data
 * @param DIM Feature dimension (384)
 * @param MAX_EDGE Maximum number of edges (360)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_inp_input(const std::string& filename, const float* data, int DIM, int MAX_EDGE,
                           std::shared_ptr<spdlog::logger> logger = nullptr) {
    size_t num_elements = static_cast<size_t>(1) * DIM * MAX_EDGE * 1;
    std::string description = "inp_input [1, " + std::to_string(DIM) + ", " + 
                              std::to_string(MAX_EDGE) + ", 1]";
    return save_float_array(filename, data, num_elements, logger, description);
}

/**
 * Save update model input: corr_input [1, CORR_DIM, MAX_EDGE, 1]
 * @param filename Output filename (e.g., "update_corr_input_frame101.bin")
 * @param data Pointer to corr_input data
 * @param CORR_DIM Correlation dimension (882)
 * @param MAX_EDGE Maximum number of edges (360)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_corr_input(const std::string& filename, const float* data, int CORR_DIM, int MAX_EDGE,
                            std::shared_ptr<spdlog::logger> logger = nullptr) {
    size_t num_elements = static_cast<size_t>(1) * CORR_DIM * MAX_EDGE * 1;
    std::string description = "corr_input [1, " + std::to_string(CORR_DIM) + ", " + 
                              std::to_string(MAX_EDGE) + ", 1]";
    return save_float_array(filename, data, num_elements, logger, description);
}

/**
 * Save update model input: index inputs (ii, jj, kk) [1, 1, MAX_EDGE, 1]
 * Converts from float vector to int32 array
 * @param filename Output filename (e.g., "update_ii_input_frame101.bin")
 * @param data Pointer to index input data (stored as float)
 * @param MAX_EDGE Maximum number of edges (360)
 * @param logger Optional logger for status messages
 * @param index_name Name of index for logging (e.g., "ii", "jj", "kk")
 * @return true if successful, false otherwise
 */
inline bool save_index_input(const std::string& filename, const float* data, int MAX_EDGE,
                             std::shared_ptr<spdlog::logger> logger = nullptr,
                             const std::string& index_name = "index") {
    size_t num_elements = static_cast<size_t>(1) * 1 * MAX_EDGE * 1;
    
    // Convert float to int32
    std::vector<int32_t> int_data(num_elements);
    for (size_t i = 0; i < num_elements; i++) {
        int_data[i] = static_cast<int32_t>(data[i]);
    }
    
    std::string description = index_name + "_input [1, 1, " + std::to_string(MAX_EDGE) + ", 1]";
    return save_int32_array(filename, int_data.data(), num_elements, logger, description);
}

/**
 * Save update model output: net_out [1, DIM, MAX_EDGE, 1]
 * @param filename Output filename (e.g., "update_net_out_cpp_frame101.bin")
 * @param data Pointer to net_out data
 * @param DIM Feature dimension (384)
 * @param MAX_EDGE Maximum number of edges (360)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_net_output(const std::string& filename, const float* data, int DIM, int MAX_EDGE,
                            std::shared_ptr<spdlog::logger> logger = nullptr) {
    size_t num_elements = static_cast<size_t>(1) * DIM * MAX_EDGE * 1;
    std::string description = "net_out [1, " + std::to_string(DIM) + ", " + 
                              std::to_string(MAX_EDGE) + ", 1]";
    return save_float_array(filename, data, num_elements, logger, description);
}

/**
 * Save update model output: d_out [1, 2, MAX_EDGE, 1]
 * @param filename Output filename (e.g., "update_d_out_cpp_frame101.bin")
 * @param data Pointer to d_out data
 * @param MAX_EDGE Maximum number of edges (360)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_d_output(const std::string& filename, const float* data, int MAX_EDGE,
                          std::shared_ptr<spdlog::logger> logger = nullptr) {
    size_t num_elements = static_cast<size_t>(1) * 2 * MAX_EDGE * 1;
    std::string description = "d_out [1, 2, " + std::to_string(MAX_EDGE) + ", 1]";
    return save_float_array(filename, data, num_elements, logger, description);
}

/**
 * Save update model output: w_out [1, 2, MAX_EDGE, 1]
 * @param filename Output filename (e.g., "update_w_out_cpp_frame101.bin")
 * @param data Pointer to w_out data
 * @param MAX_EDGE Maximum number of edges (360)
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_w_output(const std::string& filename, const float* data, int MAX_EDGE,
                          std::shared_ptr<spdlog::logger> logger = nullptr) {
    size_t num_elements = static_cast<size_t>(1) * 2 * MAX_EDGE * 1;
    std::string description = "w_out [1, 2, " + std::to_string(MAX_EDGE) + ", 1]";
    return save_float_array(filename, data, num_elements, logger, description);
}

/**
 * Save update model metadata to text file
 * @param filename Output filename (e.g., "update_metadata_frame101.txt")
 * @param frame Frame number
 * @param num_active Number of active edges
 * @param MAX_EDGE Maximum number of edges
 * @param DIM Feature dimension
 * @param CORR_DIM Correlation dimension
 * @param logger Optional logger for status messages
 * @return true if successful, false otherwise
 */
inline bool save_metadata(const std::string& filename, int frame, int num_active, int MAX_EDGE,
                          int DIM, int CORR_DIM,
                          std::shared_ptr<spdlog::logger> logger = nullptr) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        if (logger) {
            logger->error("[Update] Failed to open {} for writing", filename);
        }
        return false;
    }
    
    file << "frame=" << frame << "\n";
    file << "num_active=" << num_active << "\n";
    file << "MAX_EDGE=" << MAX_EDGE << "\n";
    file << "DIM=" << DIM << "\n";
    file << "CORR_DIM=" << CORR_DIM << "\n";
    file.close();
    
    if (logger) {
        logger->info("[Update] Saved metadata to {}", filename);
    }
    
    return true;
}

} // namespace update_file_io

