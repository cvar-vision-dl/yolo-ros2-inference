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

#pragma once
#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "inference_backend.hpp"

namespace yolo_inference
{

class TensorRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

class TensorRTBackend : public InferenceBackend
{
public:
  TensorRTBackend();
  ~TensorRTBackend() override;

  bool initialize(
    const std::string & model_path,
    TaskType task,
    int input_size = 640) override;

  InferenceResult infer(
    const cv::Mat & image,
    float conf_threshold = 0.5f,
    float nms_threshold = 0.4f,
    float keypoint_threshold = 0.3f) override;

  std::vector<std::string> getClassNames() const override;
  std::vector<std::string> getKeypointNames() const override;
  bool isInitialized() const override {return initialized_;}
  ModelFormat getFormat() const override {return ModelFormat::TENSORRT;}
  TaskType getTask() const override {return task_type_;}

private:
  bool loadEngine(const std::string & engine_path);
  bool setupBindings();
  void allocateBuffers();
  void deallocateBuffers();

  std::vector<Detection> postProcessDetection(
    float * output,
    cv::Size input_size,
    cv::Size original_size,
    cv::Size2f scale_factors,
    cv::Point2f padding,
    float conf_threshold,
    float nms_threshold);

  std::vector<Detection> postProcessPose(
    float * output,
    cv::Size input_size,
    cv::Size original_size,
    cv::Size2f scale_factors,
    cv::Point2f padding,
    float conf_threshold,
    float nms_threshold,
    float keypoint_threshold);

  TaskType task_type_;
  int input_size_;
  bool initialized_;

  // TensorRT objects
  std::unique_ptr<TensorRTLogger> logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  // CUDA objects
  cudaStream_t stream_;

  // Modern TensorRT API uses individual buffers instead of arrays
  void * input_device_buffer_;
  void * output_device_buffer_;
  void * input_host_buffer_;
  void * output_host_buffer_;

  // Model info - using tensor names instead of binding indices
  std::string input_tensor_name_;
  std::string output_tensor_name_;
  size_t input_size_bytes_;
  size_t output_size_bytes_;
  nvinfer1::Dims input_dims_;
  nvinfer1::Dims output_dims_;

  // Class and keypoint names
  std::vector<std::string> class_names_;
  std::vector<std::string> keypoint_names_;
};

}  // namespace yolo_inference

#endif  // HAVE_TENSORRT
