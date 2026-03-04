/*
 * YOLOv8 ONNX Runtime inference for DPVO.
 * Use when UseOnnxRuntime = 1 and Yolov8ModelPath points to a .onnx file.
 */
#include "yolov8_onnx.hpp"
#include "onnx_env.hpp"
#include "logger.hpp"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>

#ifdef USE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#if defined(CV28) || defined(CV28_SIMULATOR)
#include <eazyai.h>
#endif

YOLOv8ONNX::YOLOv8ONNX(Config_S* config)
{
    auto logger = spdlog::get("dpvo");
    if (!logger) logger = spdlog::get("yolov8");
    if (!logger) logger = spdlog::stdout_color_mt("yolov8_onnx");
    logger->set_pattern("[%n] [%^%l%$] %v");

#ifdef USE_ONNX_RUNTIME
    if (!config || config->yolov8ModelPath.empty()) {
        logger->info("YOLOv8ONNX: no yolov8ModelPath set, disabled");
        return;
    }
    if (!config->useOnnxRuntime) {
        logger->info("YOLOv8ONNX: UseOnnxRuntime not set, use AMBA YOLOv8 instead");
        return;
    }
    m_modelPath = config->yolov8ModelPath;
    std::ifstream file(m_modelPath);
    if (!file.good()) {
        logger->error("YOLOv8ONNX: model file does not exist: {}", m_modelPath);
        return;
    }
    _initModel();
#else
    if (config && !config->yolov8ModelPath.empty()) {
        logger->error("YOLOv8ONNX: ONNX Runtime not enabled. Build with -DUSE_ONNX_RUNTIME.");
    }
#endif
}

YOLOv8ONNX::~YOLOv8ONNX()
{
#ifdef USE_ONNX_RUNTIME
    if (m_session) {
        Ort::Session* session = static_cast<Ort::Session*>(m_session);
        delete session;
        m_session = nullptr;
    }
#endif
}

void YOLOv8ONNX::_initModel()
{
    auto logger = spdlog::get("yolov8_onnx");
    if (!logger) logger = spdlog::get("dpvo");

#ifdef USE_ONNX_RUNTIME
    try {
        Ort::Env& env = OnnxEnvSingleton::getInstance();
        Ort::SessionOptions session_options;
        Ort::Session* session = new Ort::Session(env, m_modelPath.c_str(), session_options);
        m_session = session;

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_inputs = session->GetInputCount();
        if (num_inputs != 1) {
            logger->error("YOLOv8ONNX: expected 1 input, got {}", num_inputs);
            return;
        }
        auto input_name = session->GetInputNameAllocated(0, allocator);
        m_inputNames.push_back(input_name.get());
        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        m_inputShape = input_tensor_info.GetShape();
        if (m_inputShape.size() == 4) {
            m_inputChannel = m_inputShape[1] > 0 ? static_cast<int>(m_inputShape[1]) : 3;
            m_inputHeight  = m_inputShape[2] > 0 ? static_cast<int>(m_inputShape[2]) : 288;
            m_inputWidth   = m_inputShape[3] > 0 ? static_cast<int>(m_inputShape[3]) : 512;
        }

        size_t num_outputs = session->GetOutputCount();
        m_outputShapes.clear();
        for (size_t i = 0; i < num_outputs; i++) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            m_outputNames.push_back(output_name.get());
            Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = output_tensor_info.GetShape();
            m_outputShapes.push_back(shape);
        }

        if (num_outputs == 1) {
            const auto& s = m_outputShapes[0];
            if (s.size() == 3) {
                int64_t ch = s[1];
                int64_t n  = s[2];
                m_numAnchorBox = static_cast<int>(n);
                m_boxBufferSize  = 4 * m_numAnchorBox;
                m_confBufferSize = m_numAnchorBox;
                m_classBufferSize = m_numAnchorBox;
                logger->info("YOLOv8ONNX: single output (1, {}, {}), numAnchorBox={}", ch, n, m_numAnchorBox);
            } else {
                logger->error("YOLOv8ONNX: unexpected single output shape size {}", s.size());
                return;
            }
        } else if (num_outputs >= 3) {
            const auto& s0 = m_outputShapes[0];
            if (s0.size() == 3) {
                int64_t n = s0[2];
                m_numAnchorBox = static_cast<int>(n);
            } else {
                logger->error("YOLOv8ONNX: expected output0 shape [1,4,N], got {} dims", s0.size());
                return;
            }
            m_boxBufferSize  = 4 * m_numAnchorBox;
            m_confBufferSize = m_numAnchorBox;
            m_classBufferSize = m_numAnchorBox;
            logger->info("YOLOv8ONNX: 3 outputs, numAnchorBox={}", m_numAnchorBox);
        } else {
            logger->error("YOLOv8ONNX: expected 1 or 3 outputs, got {}", num_outputs);
            return;
        }

        logger->info("YOLOv8ONNX: model loaded. Input: {}x{}x{}", m_inputChannel, m_inputHeight, m_inputWidth);
    } catch (const std::exception& e) {
        logger->error("YOLOv8ONNX: init failed: {}", e.what());
        m_session = nullptr;
    }
#endif
}

#if defined(CV28) || defined(CV28_SIMULATOR)
bool YOLOv8ONNX::_loadInput(ea_tensor_t* imgTensor, std::vector<float>& input_data)
#else
bool YOLOv8ONNX::_loadInput(void* imgTensor, std::vector<float>& input_data)
#endif
{
#ifdef USE_ONNX_RUNTIME
#if defined(CV28) || defined(CV28_SIMULATOR)
    const size_t* shape = ea_tensor_shape(imgTensor);
    int H = static_cast<int>(shape[EA_H]);
    int W = static_cast<int>(shape[EA_W]);
    void* tensor_data = ea_tensor_data(imgTensor);
    if (!tensor_data) return false;
    const uint8_t* src = static_cast<const uint8_t*>(tensor_data);
#else
    (void)imgTensor;
    int H = m_inputHeight, W = m_inputWidth;
    const uint8_t* src = nullptr;
    return false;
#endif
    input_data.resize(static_cast<size_t>(m_inputChannel) * m_inputHeight * m_inputWidth);

    cv::Mat img_bgr(H, W, CV_8UC3);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int src_idx = c * H * W + y * W + x;
                img_bgr.at<cv::Vec3b>(y, x)[c] = src[src_idx];
            }
        }
    }
    cv::Mat img_resized;
    if (H == m_inputHeight && W == m_inputWidth) {
        img_resized = img_bgr.clone();
    } else {
        cv::resize(img_bgr, img_resized, cv::Size(m_inputWidth, m_inputHeight), 0, 0, cv::INTER_LINEAR);
    }
    cv::Mat img_rgb;
    cv::cvtColor(img_resized, img_rgb, cv::COLOR_BGR2RGB);
    img_rgb.convertTo(img_rgb, CV_32F, 1.0f / 255.0f, 0.0f);
    for (int c = 0; c < m_inputChannel; c++) {
        for (int y = 0; y < m_inputHeight; y++) {
            for (int x = 0; x < m_inputWidth; x++) {
                size_t dst_idx = static_cast<size_t>(c * m_inputHeight * m_inputWidth + y * m_inputWidth + x);
                input_data[dst_idx] = img_rgb.at<cv::Vec3f>(y, x)[c];
            }
        }
    }
    return true;
#else
    (void)input_data;
    return false;
#endif
}

bool YOLOv8ONNX::runSync(
#if defined(CV28) || defined(CV28_SIMULATOR)
    ea_tensor_t* imgTensor,
#else
    void* imgTensor,
#endif
    int /*frameIdx*/,
    YOLOv8_Prediction& out_pred)
{
    auto logger = spdlog::get("yolov8_onnx");
    if (!logger) logger = spdlog::get("dpvo");

#ifdef USE_ONNX_RUNTIME
    if (!m_session || m_numAnchorBox <= 0) {
        if (logger) logger->error("YOLOv8ONNX: session not ready");
        return false;
    }

    if (out_pred.objBoxBuff)  { delete[] out_pred.objBoxBuff;  out_pred.objBoxBuff  = nullptr; }
    if (out_pred.objConfBuff) { delete[] out_pred.objConfBuff; out_pred.objConfBuff = nullptr; }
    if (out_pred.objClsBuff)  { delete[] out_pred.objClsBuff;  out_pred.objClsBuff  = nullptr; }

    std::vector<float> input_data;
    if (!_loadInput(imgTensor, input_data)) {
        if (logger) logger->error("YOLOv8ONNX: load input failed");
        return false;
    }

    try {
        Ort::Session* session = static_cast<Ort::Session*>(m_session);
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_shape = {1, m_inputChannel, m_inputHeight, m_inputWidth};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(),
            input_shape.data(), input_shape.size());

        std::vector<const char*> input_names_c = {m_inputNames[0].c_str()};
        std::vector<const char*> output_names_c;
        for (const auto& n : m_outputNames)
            output_names_c.push_back(n.c_str());

        auto output_tensors = session->Run(Ort::RunOptions{nullptr},
                                           input_names_c.data(), &input_tensor, 1,
                                           output_names_c.data(), static_cast<size_t>(output_names_c.size()));

        out_pred.objBoxBuff  = new float[static_cast<size_t>(m_boxBufferSize)];
        out_pred.objConfBuff = new float[static_cast<size_t>(m_confBufferSize)];
        out_pred.objClsBuff  = new float[static_cast<size_t>(m_classBufferSize)];

        if (output_tensors.size() == 1) {
            float* raw = output_tensors[0].GetTensorMutableData<float>();
            const auto& s = m_outputShapes[0];
            int64_t ch = s[1];
            int64_t n  = s[2];
            int num_classes = static_cast<int>(ch) - 4;
            for (int64_t i = 0; i < n; i++) {
                float cx = raw[0 * n + i];
                float cy = raw[1 * n + i];
                float w  = raw[2 * n + i];
                float h  = raw[3 * n + i];
                float x1 = cx - w * 0.5f;
                float y1 = cy - h * 0.5f;
                float x2 = cx + w * 0.5f;
                float y2 = cy + h * 0.5f;
                out_pred.objBoxBuff[0 * m_numAnchorBox + static_cast<int>(i)] = x1;
                out_pred.objBoxBuff[1 * m_numAnchorBox + static_cast<int>(i)] = y1;
                out_pred.objBoxBuff[2 * m_numAnchorBox + static_cast<int>(i)] = x2;
                out_pred.objBoxBuff[3 * m_numAnchorBox + static_cast<int>(i)] = y2;
                float best_conf = 0.f;
                int best_cls = 0;
                for (int k = 0; k < num_classes; k++) {
                    float score = raw[(4 + k) * n + i];
                    if (score > best_conf) {
                        best_conf = score;
                        best_cls = k;
                    }
                }
                out_pred.objConfBuff[static_cast<int>(i)] = best_conf;
                out_pred.objClsBuff[static_cast<int>(i)] = static_cast<float>(best_cls);
            }
        } else {
            float* dbox = output_tensors[0].GetTensorMutableData<float>();
            float* conf = output_tensors[1].GetTensorMutableData<float>();
            float* cls  = output_tensors[2].GetTensorMutableData<float>();
            const auto& s0 = m_outputShapes[0];
            int64_t n = s0[2];
            for (int64_t i = 0; i < n; i++) {
                int ii = static_cast<int>(i);
                out_pred.objBoxBuff[0 * m_numAnchorBox + ii] = dbox[0 * n + i];
                out_pred.objBoxBuff[1 * m_numAnchorBox + ii] = dbox[1 * n + i];
                out_pred.objBoxBuff[2 * m_numAnchorBox + ii] = dbox[2 * n + i];
                out_pred.objBoxBuff[3 * m_numAnchorBox + ii] = dbox[3 * n + i];
                out_pred.objConfBuff[ii] = conf[i];
                out_pred.objClsBuff[ii]  = cls[i];
            }
        }
        out_pred.isProcessed = true;
        return true;
    } catch (const std::exception& e) {
        if (logger) logger->error("YOLOv8ONNX: inference failed: {}", e.what());
        if (out_pred.objBoxBuff)  { delete[] out_pred.objBoxBuff;  out_pred.objBoxBuff  = nullptr; }
        if (out_pred.objConfBuff) { delete[] out_pred.objConfBuff; out_pred.objConfBuff = nullptr; }
        if (out_pred.objClsBuff)  { delete[] out_pred.objClsBuff;  out_pred.objClsBuff  = nullptr; }
        return false;
    }
#else
    (void)imgTensor;
    (void)out_pred;
    if (logger) logger->error("YOLOv8ONNX: ONNX Runtime not enabled");
    return false;
#endif
}
