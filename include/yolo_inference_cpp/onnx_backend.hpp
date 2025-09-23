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
#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

#include <memory>
#include <string>
#include <vector>

#include "inference_backend.hpp"

namespace yolo_inference
{

class ONNXBackend : public InferenceBackend
{
public:
  ONNXBackend();
  ~ONNXBackend() override;

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
  ModelFormat getFormat() const override {return ModelFormat::ONNX;}
  TaskType getTask() const override {return task_type_;}

private:
  std::vector<Detection> postProcessPose(
    const float * output,
    const std::vector<int64_t> & output_shape,
    cv::Size input_size,
    cv::Size original_size,
    cv::Size2f scale_factors,
    cv::Point2f padding,
    float conf_threshold,
    float nms_threshold,
    float keypoint_threshold);

  std::vector<Detection> postProcessDetection(
    const float * output,
    const std::vector<int64_t> & output_shape,
    cv::Size input_size,
    cv::Size original_size,
    cv::Size2f scale_factors,
    cv::Point2f padding,
    float conf_threshold,
    float nms_threshold);

  TaskType task_type_;
  int input_size_;
  bool initialized_;

  // ONNX Runtime objects
  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::SessionOptions> session_options_;
  std::unique_ptr<Ort::MemoryInfo> memory_info_;

  // Model info
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;

  // Class and keypoint names
  std::vector<std::string> class_names_;
  std::vector<std::string> keypoint_names_;
};

}  // namespace yolo_inference

#endif  // HAVE_ONNXRUNTIME
