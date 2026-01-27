#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>
#include "app/inc/update_onnx.hpp"
#include "app/inc/dla_config.hpp"

// Simple test program to generate random inputs and run ONNX inference
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <update_model.onnx>" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    
    // Create minimal config with model path
    Config_S config = {};
    config.updateModelPath = model_path;
    
    // Initialize ONNX model (model is loaded automatically in constructor via _initModel())
    // The constructor will check if the model file exists and load it
    std::cout << "Loading ONNX model from: " << model_path << std::endl;
    DPVOUpdateONNX update_model(&config);
    
    // Note: Model loading happens in constructor, so if it fails, we'll know when we try to run inference
    
    // Test parameters
    const int num_active = 10;  // Small number for testing
    const int MAX_EDGE = 360;   // Model's max edge count
    const int DIM = 384;
    const int CORR_DIM = 882;
    
    std::cout << "Generating random inputs..." << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int_dis(0, 5);
    
    // Allocate input buffers
    float m_net[MAX_EDGE][DIM];
    std::vector<float> ctx(num_active * DIM);
    std::vector<float> corr(num_active * CORR_DIM);
    int m_ii[num_active];
    int m_jj[num_active];
    int m_kk[num_active];
    
    // Generate random data
    for (int e = 0; e < num_active; e++) {
        for (int d = 0; d < DIM; d++) {
            m_net[e][d] = dis(gen);
        }
        for (int d = 0; d < DIM; d++) {
            ctx[e * DIM + d] = dis(gen);
        }
        for (int c = 0; c < CORR_DIM; c++) {
            corr[e * CORR_DIM + c] = dis(gen);
        }
        m_ii[e] = int_dis(gen);
        m_jj[e] = int_dis(gen);
        m_kk[e] = int_dis(gen);
    }
    
    // Zero out unused edges
    for (int e = num_active; e < MAX_EDGE; e++) {
        for (int d = 0; d < DIM; d++) {
            m_net[e][d] = 0.0f;
        }
    }
    
    std::cout << "Reshaping inputs..." << std::endl;
    
    // Reshape inputs (same as in reshapeInput)
    std::vector<float> net_input(1 * DIM * MAX_EDGE * 1);
    std::vector<float> inp_input(1 * DIM * MAX_EDGE * 1);
    std::vector<float> corr_input(1 * CORR_DIM * MAX_EDGE * 1);
    std::vector<float> ii_input(1 * 1 * MAX_EDGE * 1);
    std::vector<float> jj_input(1 * 1 * MAX_EDGE * 1);
    std::vector<float> kk_input(1 * 1 * MAX_EDGE * 1);
    
    // Zero out buffers
    std::fill(net_input.begin(), net_input.end(), 0.0f);
    std::fill(inp_input.begin(), inp_input.end(), 0.0f);
    std::fill(corr_input.begin(), corr_input.end(), 0.0f);
    std::fill(ii_input.begin(), ii_input.end(), 0.0f);
    std::fill(jj_input.begin(), jj_input.end(), 0.0f);
    std::fill(kk_input.begin(), kk_input.end(), 0.0f);
    
    // Reshape from [H, DIM] to [1, DIM, H, 1]
    for (int e = 0; e < num_active && e < MAX_EDGE; e++) {
        // Reshape net: [H, DIM] → [1, DIM, H, 1]
        for (int d = 0; d < DIM; d++) {
            int idx = 0 * (DIM * MAX_EDGE * 1) + 
                      d * (MAX_EDGE * 1) + 
                      e * 1 + 
                      0;
            net_input[idx] = m_net[e][d];
            inp_input[idx] = ctx[e * DIM + d];
        }
        
        // Reshape correlation: [H, CORR_DIM] → [1, CORR_DIM, H, 1]
        for (int c = 0; c < CORR_DIM; c++) {
            int idx = 0 * (CORR_DIM * MAX_EDGE * 1) + 
                      c * (MAX_EDGE * 1) + 
                      e * 1 + 
                      0;
            corr_input[idx] = corr[e * CORR_DIM + c];
        }
        
        // Reshape indices: [H] → [1, 1, H, 1]
        int idx_ii = 0 * (1 * MAX_EDGE * 1) + 
                     0 * (MAX_EDGE * 1) + 
                     e * 1 + 
                     0;
        ii_input[idx_ii] = static_cast<float>(m_ii[e]);
        jj_input[idx_ii] = static_cast<float>(m_jj[e]);
        kk_input[idx_ii] = static_cast<float>(m_kk[e]);
    }
    
    std::cout << "Saving inputs to files..." << std::endl;
    
    // Save inputs to binary files for Python comparison
    std::ofstream net_file("test_net_input.bin", std::ios::binary);
    net_file.write(reinterpret_cast<const char*>(net_input.data()), net_input.size() * sizeof(float));
    net_file.close();
    
    std::ofstream inp_file("test_inp_input.bin", std::ios::binary);
    inp_file.write(reinterpret_cast<const char*>(inp_input.data()), inp_input.size() * sizeof(float));
    inp_file.close();
    
    std::ofstream corr_file("test_corr_input.bin", std::ios::binary);
    corr_file.write(reinterpret_cast<const char*>(corr_input.data()), corr_input.size() * sizeof(float));
    corr_file.close();
    
    std::ofstream ii_file("test_ii_input.bin", std::ios::binary);
    ii_file.write(reinterpret_cast<const char*>(ii_input.data()), ii_input.size() * sizeof(float));
    ii_file.close();
    
    std::ofstream jj_file("test_jj_input.bin", std::ios::binary);
    jj_file.write(reinterpret_cast<const char*>(jj_input.data()), jj_input.size() * sizeof(float));
    jj_file.close();
    
    std::ofstream kk_file("test_kk_input.bin", std::ios::binary);
    kk_file.write(reinterpret_cast<const char*>(kk_input.data()), kk_input.size() * sizeof(float));
    kk_file.close();
    
    // Save metadata
    std::ofstream meta_file("test_metadata.txt");
    meta_file << "num_active=" << num_active << std::endl;
    meta_file << "MAX_EDGE=" << MAX_EDGE << std::endl;
    meta_file << "DIM=" << DIM << std::endl;
    meta_file << "CORR_DIM=" << CORR_DIM << std::endl;
    meta_file.close();
    
    std::cout << "Running C++ ONNX inference..." << std::endl;
    
    // Run inference
    DPVOUpdate_Prediction pred;
    bool success = update_model.runInference(
        net_input.data(),
        inp_input.data(),
        corr_input.data(),
        ii_input.data(),
        jj_input.data(),
        kk_input.data(),
        0,  // frameIdx
        pred
    );
    
    if (!success) {
        std::cerr << "C++ ONNX inference failed!" << std::endl;
        return 1;
    }
    
    std::cout << "Saving C++ outputs..." << std::endl;
    
    // Save outputs
    std::ofstream net_out_file("test_net_out_cpp.bin", std::ios::binary);
    net_out_file.write(reinterpret_cast<const char*>(pred.netOutBuff), 
                      1 * DIM * MAX_EDGE * 1 * sizeof(float));
    net_out_file.close();
    
    std::ofstream d_out_file("test_d_out_cpp.bin", std::ios::binary);
    d_out_file.write(reinterpret_cast<const char*>(pred.dOutBuff), 
                    1 * 2 * MAX_EDGE * 1 * sizeof(float));
    d_out_file.close();
    
    std::ofstream w_out_file("test_w_out_cpp.bin", std::ios::binary);
    w_out_file.write(reinterpret_cast<const char*>(pred.wOutBuff), 
                    1 * 2 * MAX_EDGE * 1 * sizeof(float));
    w_out_file.close();
    
    std::cout << "C++ inference completed successfully!" << std::endl;
    std::cout << "Output files saved:" << std::endl;
    std::cout << "  - test_net_out_cpp.bin" << std::endl;
    std::cout << "  - test_d_out_cpp.bin" << std::endl;
    std::cout << "  - test_w_out_cpp.bin" << std::endl;
    std::cout << "\nNow run: python3 compare_update_onnx_outputs.py <update_model.onnx>" << std::endl;
    
    // Cleanup
    if (pred.netOutBuff) delete[] pred.netOutBuff;
    if (pred.dOutBuff) delete[] pred.dOutBuff;
    if (pred.wOutBuff) delete[] pred.wOutBuff;
    
    return 0;
}

