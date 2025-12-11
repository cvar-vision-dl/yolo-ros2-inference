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
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace yolo_inference
{

enum class TaskType
{
  DETECT,
  POSE,
  SEGMENT,
  GATENET
};

enum class ModelFormat
{
  TENSORRT,
  ONNX,
  UNKNOWN
};

struct Detection
{
  cv::Rect2f bbox;    // Using float precision for better coordinate accuracy
  float confidence;
  int class_id;
  std::vector<cv::Point3f> keypoints;   // x, y, confidence
  cv::Mat mask;   // For segmentation
};

struct InferenceResult
{
  bool success;
  std::vector<Detection> detections;
  cv::Size input_size;
  cv::Size original_size;
  double inference_time_ms;
};

class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;

  virtual bool initialize(
    const std::string & model_path,
    TaskType task,
    int input_size = 640,
    int input_width = -1,
    int input_height = -1) = 0;

  virtual InferenceResult infer(
    const cv::Mat & image,
    float conf_threshold = 0.5f,
    float nms_threshold = 0.4f,
    float keypoint_threshold = 0.3f) = 0;

  virtual std::vector<std::string> getClassNames() const = 0;
  virtual std::vector<std::string> getKeypointNames() const = 0;
  virtual bool isInitialized() const = 0;
  virtual ModelFormat getFormat() const = 0;
  virtual TaskType getTask() const = 0;
};

std::unique_ptr<InferenceBackend> createInferenceBackend(const std::string & model_path);
ModelFormat detectModelFormat(const std::string & model_path);
TaskType stringToTaskType(const std::string & task_str);

}  // namespace yolo_inference
