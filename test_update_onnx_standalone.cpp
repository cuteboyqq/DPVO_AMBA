#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cstring>

#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <update_model.onnx>" << std::endl;
        return 1;
    }
    
    const char* model_path = argv[1];
    
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, model_path, session_options);
    
    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    
    std::cout << "Model has " << num_input_nodes << " inputs and " << num_output_nodes << " outputs" << std::endl;
    
    // Store allocated strings to keep them alive
    std::vector<Ort::AllocatedStringPtr> input_name_allocated;
    std::vector<std::string> input_names;
    std::vector<std::vector<int64_t>> input_shapes;
    
    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        input_name_allocated.push_back(std::move(input_name));
        input_names.push_back(input_name_allocated[i].get());
        
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        input_shapes.push_back(shape);
        
        std::cout << "Input " << i << " name: " << input_name_allocated[i].get() << ", shape: [";
        for (size_t j = 0; j < shape.size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << shape[j];
        }
        std::cout << "]" << std::endl;
    }
    
    // Get output names and shapes (keep allocated strings alive)
    std::vector<Ort::AllocatedStringPtr> output_name_allocated;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> output_shapes;
    
    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session.GetOutputNameAllocated(i, allocator);
        output_name_allocated.push_back(std::move(output_name));
        output_names.push_back(output_name_allocated[i].get());
        
        auto type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        output_shapes.push_back(shape);
        
        std::cout << "Output " << i << " name: " << output_name_allocated[i].get() << ", shape: [";
        for (size_t j = 0; j < shape.size(); j++) {
            if (j > 0) std::cout << ", ";
            std::cout << shape[j];
        }
        std::cout << "]" << std::endl;
    }
    
    // Test parameters
    const int num_active = 10;
    const int MAX_EDGE = 360;
    const int DIM = 384;
    const int CORR_DIM = 882;
    
    std::cout << "\nGenerating random inputs..." << std::endl;
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int_dis(0, 5);
    
    // Generate random data: [H, DIM] layout
    float m_net[MAX_EDGE][DIM];
    std::vector<float> ctx(num_active * DIM);
    std::vector<float> corr(num_active * CORR_DIM);
    int m_ii[num_active];
    int m_jj[num_active];
    int m_kk[num_active];
    
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
    
    std::cout << "Reshaping inputs from [H, DIM] to [1, DIM, H, 1]..." << std::endl;
    
    // Reshape inputs to [1, DIM, H, 1] layout
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
    // Index formula: idx = n * (DIM * H * 1) + c * (H * 1) + h * 1 + w
    // For [1, DIM, H, 1]: idx = 0 + c * H + h + 0 = c * H + h
    for (int e = 0; e < num_active && e < MAX_EDGE; e++) {
        // Reshape net and inp: [H, DIM] → [1, DIM, H, 1]
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
    
    // Save inputs
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
    
    // Prepare input tensors
    std::vector<const char*> input_name_ptrs;
    for (const auto& name : input_names) {
        input_name_ptrs.push_back(name.c_str());
    }
    
    std::vector<Ort::Value> input_tensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Create input tensors
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, net_input.data(), net_input.size(),
        input_shapes[0].data(), input_shapes[0].size()));
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, inp_input.data(), inp_input.size(),
        input_shapes[1].data(), input_shapes[1].size()));
    
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, corr_input.data(), corr_input.size(),
        input_shapes[2].data(), input_shapes[2].size()));
    
    // Convert indices to int32
    std::vector<int32_t> ii_int32(MAX_EDGE);
    std::vector<int32_t> jj_int32(MAX_EDGE);
    std::vector<int32_t> kk_int32(MAX_EDGE);
    for (int i = 0; i < MAX_EDGE; i++) {
        ii_int32[i] = static_cast<int32_t>(ii_input[i]);
        jj_int32[i] = static_cast<int32_t>(jj_input[i]);
        kk_int32[i] = static_cast<int32_t>(kk_input[i]);
    }
    
    input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
        memory_info, ii_int32.data(), MAX_EDGE,
        input_shapes[3].data(), input_shapes[3].size()));
    
    input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
        memory_info, jj_int32.data(), MAX_EDGE,
        input_shapes[4].data(), input_shapes[4].size()));
    
    input_tensors.push_back(Ort::Value::CreateTensor<int32_t>(
        memory_info, kk_int32.data(), MAX_EDGE,
        input_shapes[5].data(), input_shapes[5].size()));
    
    // Prepare output name pointers (using already stored output_names)
    std::vector<const char*> output_name_ptrs;
    for (const auto& name : output_names) {
        output_name_ptrs.push_back(name.c_str());
    }
    
    if (output_name_ptrs.empty()) {
        std::cerr << "Error: No output names found!" << std::endl;
        return 1;
    }
    
    std::cout << "Running inference..." << std::endl;
    
    // Run inference
    auto outputs = session.Run(Ort::RunOptions{nullptr},
                               input_name_ptrs.data(), input_tensors.data(), input_tensors.size(),
                               output_name_ptrs.data(), output_name_ptrs.size());
    
    std::cout << "Saving C++ outputs..." << std::endl;
    
    // Extract and save outputs
    float* net_out_data = outputs[0].GetTensorMutableData<float>();
    size_t net_out_size = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    float* d_out_data = outputs[1].GetTensorMutableData<float>();
    size_t d_out_size = outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
    
    float* w_out_data = outputs[2].GetTensorMutableData<float>();
    size_t w_out_size = outputs[2].GetTensorTypeAndShapeInfo().GetElementCount();
    
    std::ofstream net_out_file("test_net_out_cpp.bin", std::ios::binary);
    net_out_file.write(reinterpret_cast<const char*>(net_out_data), net_out_size * sizeof(float));
    net_out_file.close();
    
    std::ofstream d_out_file("test_d_out_cpp.bin", std::ios::binary);
    d_out_file.write(reinterpret_cast<const char*>(d_out_data), d_out_size * sizeof(float));
    d_out_file.close();
    
    std::ofstream w_out_file("test_w_out_cpp.bin", std::ios::binary);
    w_out_file.write(reinterpret_cast<const char*>(w_out_data), w_out_size * sizeof(float));
    w_out_file.close();
    
    std::cout << "C++ inference completed successfully!" << std::endl;
    std::cout << "Output files saved:" << std::endl;
    std::cout << "  - test_net_out_cpp.bin (" << net_out_size << " elements)" << std::endl;
    std::cout << "  - test_d_out_cpp.bin (" << d_out_size << " elements)" << std::endl;
    std::cout << "  - test_w_out_cpp.bin (" << w_out_size << " elements)" << std::endl;
    std::cout << "\nNow run: python3 compare_update_onnx_outputs.py " << model_path << std::endl;
    
    return 0;
}

#else
int main() {
    std::cerr << "ONNX Runtime not enabled. Compile with -DUSE_ONNX_RUNTIME" << std::endl;
    return 1;
}
#endif

