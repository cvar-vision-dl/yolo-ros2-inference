// src/inference_backend.cpp
#include "yolo_inference_cpp/inference_backend.hpp"

#ifdef HAVE_TENSORRT
#include "yolo_inference_cpp/tensorrt_backend.hpp"
#endif

#ifdef HAVE_ONNXRUNTIME
#include "yolo_inference_cpp/onnx_backend.hpp"
#endif

#include <filesystem>
#include <iostream>

namespace yolo_inference {

ModelFormat detectModelFormat(const std::string& model_path) {
    std::filesystem::path path(model_path);
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if (extension == ".onnx") {
        return ModelFormat::ONNX;
    } else if (extension == ".engine" || extension == ".trt") {
        return ModelFormat::TENSORRT;
    } else {
        return ModelFormat::UNKNOWN;
    }
}

TaskType stringToTaskType(const std::string& task_str) {
    std::string lower_task = task_str;
    std::transform(lower_task.begin(), lower_task.end(), lower_task.begin(), ::tolower);

    if (lower_task == "pose") {
        return TaskType::POSE;
    } else if (lower_task == "detect") {
        return TaskType::DETECT;
    } else if (lower_task == "segment") {
        return TaskType::SEGMENT;
    } else {
        std::cerr << "Warning: Unknown task type '" << task_str << "', defaulting to POSE" << std::endl;
        return TaskType::POSE;
    }
}

std::unique_ptr<InferenceBackend> createInferenceBackend(const std::string& model_path) {
    ModelFormat format = detectModelFormat(model_path);

    // Create memory pool (shared across backends)
    static auto memory_pool = std::make_unique<MemoryPool>();

    switch (format) {
        case ModelFormat::TENSORRT:
#ifdef HAVE_TENSORRT
            std::cout << "Creating TensorRT backend for: " << model_path << std::endl;
            return std::make_unique<TensorRTBackend>(*memory_pool);
#else
            std::cerr << "TensorRT support not compiled. Please rebuild with TensorRT support." << std::endl;
            return nullptr;
#endif

        case ModelFormat::ONNX:
#ifdef HAVE_ONNXRUNTIME
            std::cout << "Creating ONNX Runtime backend for: " << model_path << std::endl;
            return std::make_unique<ONNXBackend>(*memory_pool);
#else
            std::cerr << "ONNX Runtime support not compiled. Please rebuild with ONNX Runtime support." << std::endl;
            return nullptr;
#endif

        default:
            std::cerr << "Unsupported model format for: " << model_path << std::endl;
            return nullptr;
    }
}

} // namespace yolo_inference