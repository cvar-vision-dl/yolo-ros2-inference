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


#include "yolo_inference_cpp/preprocessing.hpp"
#include "yolo_inference_cpp/inference_backend.hpp"
#include <opencv2/opencv.hpp>

namespace yolo_inference
{

Preprocessor::Preprocessor()
: scale_factors_(1.0f, 1.0f)
  , padding_(0.0f, 0.0f)
{
}

cv::Mat Preprocessor::preprocess(
  const cv::Mat & input,
  int target_size,
  TaskType task,
  bool normalize,
  cv::Scalar mean,
  cv::Scalar std)
{
  // Task-aware preprocessing
  cv::Mat padded;

  // YOLO tasks: aspect-ratio preserving resize with letterbox padding
  if (task == TaskType::POSE || task == TaskType::DETECT || task == TaskType::SEGMENT) {
    // Calculate scale factor to maintain aspect ratio
    float scale = std::min(
      static_cast<float>(target_size) / input.cols,
      static_cast<float>(target_size) / input.rows);

    scale_factors_ = cv::Size2f(scale, scale);

    // Calculate new size after scaling
    int new_width = static_cast<int>(input.cols * scale);
    int new_height = static_cast<int>(input.rows * scale);

    // Resize the image maintaining aspect ratio
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // Calculate padding to reach target size
    int pad_x = (target_size - new_width) / 2;
    int pad_y = (target_size - new_height) / 2;
    padding_ = cv::Point2f(static_cast<float>(pad_x), static_cast<float>(pad_y));

    // Create output image with padding
    cv::copyMakeBorder(
      resized, padded,
      pad_y, target_size - new_height - pad_y,
      pad_x, target_size - new_width - pad_x,
      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  } else {
    // GateNet: Direct resize (stretch to fit), no aspect ratio preservation
    scale_factors_ = cv::Size2f(
      static_cast<float>(target_size) / input.cols,
      static_cast<float>(target_size) / input.rows);
    padding_ = cv::Point2f(0.0f, 0.0f);

    cv::resize(input, padded, cv::Size(target_size, target_size), 0, 0, cv::INTER_LINEAR);
  }

  // Convert to float and normalize if requested
  cv::Mat result;
  padded.convertTo(result, CV_32F);

  if (normalize) {
    result /= 255.0f;

    // Apply mean and std normalization if provided
    if (mean != cv::Scalar(0, 0, 0) || std != cv::Scalar(1, 1, 1)) {
      std::vector<cv::Mat> channels;
      cv::split(result, channels);

      for (int i = 0; i < channels.size(); ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
      }

      cv::merge(channels, result);
    }
  }

  // Convert from HWC to CHW format (required by most inference engines)
  cv::Mat chw_result;
  if (result.channels() == 3) {
    // Create CHW tensor: [3, height, width]
    chw_result = cv::Mat(1 * 3 * target_size * target_size, 1, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(result, channels);

    // Copy channels in CHW order (RGB)
    for (int c = 0; c < 3; ++c) {
      cv::Mat channel = channels[2 - c];       // BGR to RGB conversion
      std::memcpy(
        chw_result.ptr<float>() + c * target_size * target_size,
        channel.ptr<float>(), target_size * target_size * sizeof(float));
    }

    // Reshape to [1, 3, height, width]
    chw_result = chw_result.reshape(1, {1, 3, target_size, target_size});
  } else if (result.channels() == 1) {
    // Grayscale image - reshape to [1, 1, height, width]
    chw_result = result.reshape(1, {1, 1, target_size, target_size});
  } else {
    throw std::runtime_error(
            "Unsupported number of channels: " +
            std::to_string(result.channels()));
  }

  return chw_result;
}

cv::Mat Preprocessor::preprocess(
  const cv::Mat & input,
  int target_width,
  int target_height,
  TaskType task,
  bool normalize,
  cv::Scalar mean,
  cv::Scalar std)
{
  // Task-aware preprocessing
  cv::Mat padded;

  if (task == TaskType::GATENET) {
    // GateNet: Direct resize (stretch to fit target dimensions)
    // No aspect ratio preservation, no padding
    scale_factors_ = cv::Size2f(
      static_cast<float>(target_width) / input.cols,
      static_cast<float>(target_height) / input.rows);
    padding_ = cv::Point2f(0.0f, 0.0f);

    // Direct resize to target dimensions
    cv::resize(input, padded, cv::Size(target_width, target_height), 0, 0, cv::INTER_LINEAR);
  } else {
    // YOLO tasks: Aspect-ratio preserving resize with letterbox padding
    float scale_x = static_cast<float>(target_width) / input.cols;
    float scale_y = static_cast<float>(target_height) / input.rows;
    float scale = std::min(scale_x, scale_y);

    scale_factors_ = cv::Size2f(scale, scale);

    // Calculate new size after scaling
    int new_width = static_cast<int>(input.cols * scale);
    int new_height = static_cast<int>(input.rows * scale);

    // Resize maintaining aspect ratio
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // Calculate padding
    int pad_x = (target_width - new_width) / 2;
    int pad_y = (target_height - new_height) / 2;
    padding_ = cv::Point2f(static_cast<float>(pad_x), static_cast<float>(pad_y));

    // Create padded output
    cv::copyMakeBorder(
      resized, padded,
      pad_y, target_height - new_height - pad_y,
      pad_x, target_width - new_width - pad_x,
      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  }

  // Convert to float and normalize if requested
  cv::Mat result;
  padded.convertTo(result, CV_32F);

  if (normalize) {
    result /= 255.0f;

    // Apply mean and std normalization if provided
    if (mean != cv::Scalar(0, 0, 0) || std != cv::Scalar(1, 1, 1)) {
      std::vector<cv::Mat> channels;
      cv::split(result, channels);

      for (size_t i = 0; i < channels.size(); ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
      }

      cv::merge(channels, result);
    }
  }

  // Convert from HWC to CHW format (required by most inference engines)
  cv::Mat chw_result;
  if (result.channels() == 3) {
    // Create CHW tensor: [3, height, width]
    chw_result = cv::Mat(1 * 3 * target_height * target_width, 1, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(result, channels);

    // Copy channels in CHW order
    for (int c = 0; c < 3; ++c) {
        cv::Mat channel;
        if (task == TaskType::GATENET) {
            channel = channels[c];      // Keep BGR order for GateNet
        } else {
            channel = channels[2 - c];  // BGR to RGB for YOLO
        }
        std::memcpy(
            chw_result.ptr<float>() + c * target_height * target_width,
            channel.ptr<float>(), target_height * target_width * sizeof(float));
    }

    // Reshape to [1, 3, height, width]
    chw_result = chw_result.reshape(1, {1, 3, target_height, target_width});
  } else if (result.channels() == 1) {
    // Grayscale image - reshape to [1, 1, height, width]
    chw_result = result.reshape(1, {1, 1, target_height, target_width});
  } else {
    throw std::runtime_error(
            "Unsupported number of channels: " +
            std::to_string(result.channels()));
  }

  return chw_result;
}

}  // namespace yolo_inference
