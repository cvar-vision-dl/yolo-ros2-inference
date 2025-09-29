// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#include "yolo_inference_cpp/inference_backend.hpp"
#ifdef HAVE_TENSORRT
#include "yolo_inference_cpp/tensorrt_backend.hpp"
#endif
#ifdef HAVE_ONNXRUNTIME
#include "yolo_inference_cpp/onnx_backend.hpp"
#endif
#include <filesystem>
#include <iostream>

namespace yolo_inference
{

ModelFormat detectModelFormat(const std::string & model_path)
{
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

TaskType stringToTaskType(const std::string & task_str)
{
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

std::unique_ptr<InferenceBackend> createInferenceBackend(const std::string & model_path)
{
  ModelFormat format = detectModelFormat(model_path);

  switch (format) {
    case ModelFormat::TENSORRT:
#ifdef HAVE_TENSORRT
      std::cout << "Creating TensorRT backend for: " << model_path << std::endl;
      return std::make_unique<TensorRTBackend>();
#else
      std::cerr << "TensorRT support not compiled. Please rebuild with TensorRT support." <<
        std::endl;
      return nullptr;
#endif
    case ModelFormat::ONNX:
#ifdef HAVE_ONNXRUNTIME
      std::cout << "Creating ONNX Runtime backend for: " << model_path << std::endl;
      return std::make_unique<ONNXBackend>();
#else
      std::cerr << "ONNX Runtime support not compiled. Please rebuild with ONNX Runtime support." <<
        std::endl;
      return nullptr;
#endif
    default:
      std::cerr << "Unsupported model format for: " << model_path << std::endl;
      return nullptr;
  }
}

}  // namespace yolo_inference
