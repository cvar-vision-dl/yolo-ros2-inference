// Copyright 2025 UNIVERSIDAD POLITÉCNICA DE MADRID
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
//    * Neither the name of the UNIVERSIDAD POLITÉCNICA DE MADRID nor the names of its
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

/**
 * @file yolo_interface.hpp
 *
 * YOLOInterface class definition
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#ifndef YOLO_INFERENCE_CPP__YOLO_INTERFACE_HPP_
#define YOLO_INFERENCE_CPP__YOLO_INTERFACE_HPP_

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "yolo_inference_cpp/inference_backend.hpp"
#include "yolo_inference_cpp/profiler.hpp"
#include "yolo_inference_cpp/msg/keypoint_detection_array.hpp"
#include "yolo_inference_cpp/msg/performance_info.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

namespace yolo_inference
{

class YOLOInterface
{
public:
  explicit YOLOInterface(rclcpp::Node * node_ptr);
  virtual ~YOLOInterface() = default;

  // Get logger from the node
  rclcpp::Logger get_logger() const {return node_ptr_->get_logger();}

  // Public method to process images (moved from private callback)
  void processImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg);

  // Public methods to create messages without publishing
  yolo_inference_cpp::msg::KeypointDetectionArray createDetectionsMessage(
    const InferenceResult & result, const std_msgs::msg::Header & header);
  std_msgs::msg::String createRawOutputMessage(
    const InferenceResult & result, const std_msgs::msg::Header & header);
  sensor_msgs::msg::Image::SharedPtr createVisualizationMessage(
    const cv::Mat & image, const InferenceResult & result,
    const std_msgs::msg::Header & header);
  yolo_inference_cpp::msg::PerformanceInfo createPerformanceInfoMessage(
    const std_msgs::msg::Header & header);

private:
  void publishDetections(const InferenceResult & result, const std_msgs::msg::Header & header);
  void publishRawOutput(const InferenceResult & result, const std_msgs::msg::Header & header);
  void publishVisualization(
    const cv::Mat & image, const InferenceResult & result,
    const std_msgs::msg::Header & header);
  void publishPerformanceInfo(const std_msgs::msg::Header & header);
  void updatePerformanceMetrics();

  // Node pointer
  rclcpp::Node * node_ptr_;

  // Parameters
  std::string model_path_;
  std::string task_str_;
  TaskType task_type_;
  int input_size_;
  double confidence_threshold_;
  double nms_threshold_;
  double keypoint_threshold_;
  int max_detections_;
  bool publish_visualization_;
  bool enable_profiling_;

  std::string input_topic_;
  std::string output_topic_;
  std::string output_image_topic_;
  std::string performance_topic_;

  // Core components
  std::unique_ptr<InferenceBackend> backend_;
  std::unique_ptr<Profiler> profiler_;

  // ROS components
  rclcpp::Publisher<yolo_inference_cpp::msg::KeypointDetectionArray>::SharedPtr detections_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr raw_output_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::Publisher<yolo_inference_cpp::msg::PerformanceInfo>::SharedPtr performance_pub_;

  // Performance tracking
  int frame_count_;
  double total_time_;
  std::chrono::steady_clock::time_point last_log_time_;
};  // class YOLOInterface

}  // namespace yolo_inference

#endif  // YOLO_INFERENCE_CPP__YOLO_INTERFACE_HPP_
