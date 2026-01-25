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
 * @file yolo_interface.cpp
 *
 * YOLOInterface class implementation
 *
 * @author Rafael Perez-Segui <r.psegui@upm.es>
 */

#include "yolo_inference_cpp/yolo_interface.hpp"

namespace yolo_inference
{

YOLOInterface::YOLOInterface(rclcpp::Node * node_ptr)
: node_ptr_(node_ptr)
{
  // Declare parameters
  node_ptr_->declare_parameter<std::string>("model_path", "yolo11n-pose.onnx");
  node_ptr_->declare_parameter<std::string>("task", "pose");
  node_ptr_->declare_parameter<int>("input_size", 640);
  node_ptr_->declare_parameter<double>("confidence_threshold", 0.5);
  node_ptr_->declare_parameter<double>("nms_threshold", 0.4);
  node_ptr_->declare_parameter<double>("keypoint_threshold", 0.3);
  node_ptr_->declare_parameter<int>("max_detections", 20);
  node_ptr_->declare_parameter<bool>("publish_visualization", false);
  node_ptr_->declare_parameter<bool>("enable_profiling", true);
  node_ptr_->declare_parameter<std::string>("output_topic", "/yolo/detections");
  node_ptr_->declare_parameter<std::string>("output_image_topic", "/yolo/result_image");
  node_ptr_->declare_parameter<std::string>("performance_topic", "/yolo/performance");

  // Get parameters
  model_path_ = node_ptr_->get_parameter("model_path").as_string();
  task_str_ = node_ptr_->get_parameter("task").as_string();
  input_size_ = node_ptr_->get_parameter("input_size").as_int();
  confidence_threshold_ = node_ptr_->get_parameter("confidence_threshold").as_double();
  nms_threshold_ = node_ptr_->get_parameter("nms_threshold").as_double();
  keypoint_threshold_ = node_ptr_->get_parameter("keypoint_threshold").as_double();
  max_detections_ = node_ptr_->get_parameter("max_detections").as_int();
  publish_visualization_ = node_ptr_->get_parameter("publish_visualization").as_bool();
  enable_profiling_ = node_ptr_->get_parameter("enable_profiling").as_bool();

  output_topic_ = node_ptr_->get_parameter("output_topic").as_string();
  output_image_topic_ = node_ptr_->get_parameter("output_image_topic").as_string();
  performance_topic_ = node_ptr_->get_parameter("performance_topic").as_string();

  // Parse task
  task_type_ = stringToTaskType(task_str_);

  // Initialize profiler
  if (enable_profiling_) {
    profiler_ = std::make_unique<Profiler>();
  }

  // Initialize inference backend
  backend_ = createInferenceBackend(model_path_);
  if (!backend_ || !backend_->initialize(model_path_, task_type_, input_size_)) {
    RCLCPP_ERROR(get_logger(), "Failed to initialize inference backend");
    return;
  }

  RCLCPP_INFO(
    get_logger(), "Initialized %s backend for %s task",
    backend_->getFormat() == ModelFormat::TENSORRT ? "TensorRT" : "ONNX",
    task_str_.c_str());

  // Initialize publishers
  detections_pub_ = node_ptr_->create_publisher<yolo_inference_cpp::msg::KeypointDetectionArray>(
    output_topic_, 10);

  raw_output_pub_ = node_ptr_->create_publisher<std_msgs::msg::String>(
    output_topic_ + "/raw", 10);

  if (publish_visualization_) {
    image_pub_ = node_ptr_->create_publisher<sensor_msgs::msg::Image>(
      output_image_topic_, 10);
  }

  if (enable_profiling_) {
    performance_pub_ = node_ptr_->create_publisher<yolo_inference_cpp::msg::PerformanceInfo>(
      performance_topic_, 10);
  }

  // Performance tracking
  frame_count_ = 0;
  total_time_ = 0.0;
  last_log_time_ = std::chrono::steady_clock::now();

  RCLCPP_INFO(get_logger(), "YOLO Inference Node initialized");
  RCLCPP_INFO(get_logger(), "  Model: %s", model_path_.c_str());
  RCLCPP_INFO(get_logger(), "  Task: %s", task_str_.c_str());
  RCLCPP_INFO(get_logger(), "  Input size: %dx%d", input_size_, input_size_);
  RCLCPP_INFO(get_logger(), "  Subscribing to: %s", input_topic_.c_str());
  RCLCPP_INFO(get_logger(), "  Publishing to: %s", output_topic_.c_str());
  if (publish_visualization_) {
    RCLCPP_INFO(get_logger(), "  Visualization: %s", output_image_topic_.c_str());
  }
}

InferenceResult YOLOInterface::processImage(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
{
  RCLCPP_INFO(
    get_logger(), "=== Received image: %zu bytes ===",
    msg->data.size());
  // auto total_timer = enable_profiling_ ?
  //   profiler_->scopedTimer("total_processing") :
  //   Profiler::ScopedTimer(*profiler_, "dummy");
  InferenceResult result;
  try {
    // Convert compressed image to OpenCV Mat
    cv::Mat image;

    // auto timer = enable_profiling_ ? profiler_->scopedTimer("image_conversion") :
    //   Profiler::ScopedTimer(*profiler_, "dummy");

    std::vector<uint8_t> buffer(msg->data.begin(), msg->data.end());
    image = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (image.empty()) {
      RCLCPP_ERROR(get_logger(), "Failed to decode compressed image");
      result.success = false;
      return result;
    }
    RCLCPP_INFO(get_logger(), "Image decoded successfully: %dx%d", image.cols, image.rows);


    // Run inference

    // auto timer = enable_profiling_ ? profiler_->scopedTimer("inference") :
    //   Profiler::ScopedTimer(*profiler_, "dummy");

    result = backend_->infer(
      image,
      confidence_threshold_,
      nms_threshold_,
      keypoint_threshold_);
    RCLCPP_INFO(
      get_logger(), "Inference completed: %zu detections, %.2fms",
      result.detections.size(), result.inference_time_ms);


    // Publish results

    // auto timer = enable_profiling_ ? profiler_->scopedTimer("message_creation") :
    //   Profiler::ScopedTimer(*profiler_, "dummy");

    publishDetections(result, msg->header);
    publishRawOutput(result, msg->header);

    if (publish_visualization_) {
      publishVisualization(image, result, msg->header);
    }

    if (enable_profiling_) {
      publishPerformanceInfo(msg->header);
    }

    // Update performance metrics
    // RCLCPP_INFO(get_logger(), "Updating performance metricsssss");
    // updatePerformanceMetrics();

  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Error processing image: %s", e.what());
  }
  return result;
}

yolo_inference_cpp::msg::KeypointDetectionArray YOLOInterface::createDetectionsMessage(
  const InferenceResult & result,
  const std_msgs::msg::Header & header)
{
  auto msg = yolo_inference_cpp::msg::KeypointDetectionArray();
  msg.header = header;
  msg.model_type = backend_->getFormat() == ModelFormat::TENSORRT ? "TensorRT" : "ONNX";
  msg.task = task_str_;

  auto class_names = backend_->getClassNames();
  auto keypoint_names = backend_->getKeypointNames();

  for (const auto & detection : result.detections) {
    if (msg.detections.size() >= static_cast<size_t>(max_detections_)) {break;}

    yolo_inference_cpp::msg::KeypointDetection det_msg;
    det_msg.header = header;

    // Basic detection info
    det_msg.class_id = detection.class_id;
    det_msg.label = static_cast<size_t>(detection.class_id) < class_names.size() ?
      class_names[detection.class_id] : "unknown";
    det_msg.confidence = detection.confidence;

    // Bounding box - convert from cv::Rect2f to message format
    det_msg.bounding_box.x1 = detection.bbox.x;
    det_msg.bounding_box.y1 = detection.bbox.y;
    det_msg.bounding_box.x2 = detection.bbox.x + detection.bbox.width;
    det_msg.bounding_box.y2 = detection.bbox.y + detection.bbox.height;
    det_msg.bounding_box.confidence = detection.confidence;

    // Keypoints
    for (size_t i = 0; i < detection.keypoints.size() && i < keypoint_names.size(); ++i) {
      const auto & kpt = detection.keypoints[i];

      yolo_inference_cpp::msg::Keypoint kpt_msg;
      kpt_msg.name = keypoint_names[i];
      kpt_msg.x = kpt.x;
      kpt_msg.y = kpt.y;
      kpt_msg.confidence = kpt.z;
      kpt_msg.visible = kpt.z > keypoint_threshold_;

      det_msg.keypoints.push_back(kpt_msg);
    }

    msg.detections.push_back(det_msg);
  }

  return msg;
}

void YOLOInterface::publishDetections(
  const InferenceResult & result,
  const std_msgs::msg::Header & header)
{
  auto msg = createDetectionsMessage(result, header);
  detections_pub_->publish(msg);
}

std_msgs::msg::String YOLOInterface::createRawOutputMessage(
  const InferenceResult & result,
  const std_msgs::msg::Header & /*header*/)
{
  // Create raw array format for high-performance applications
  std::vector<float> raw_data;

  for (const auto & detection : result.detections) {
    if (raw_data.size() / 6 >= static_cast<size_t>(max_detections_)) {break;}

    // Basic format: [x1, y1, x2, y2, confidence, class_id, num_keypoints, kpt_data...]
    raw_data.push_back(detection.bbox.x);
    raw_data.push_back(detection.bbox.y);
    raw_data.push_back(detection.bbox.x + detection.bbox.width);
    raw_data.push_back(detection.bbox.y + detection.bbox.height);
    raw_data.push_back(detection.confidence);
    raw_data.push_back(static_cast<float>(detection.class_id));

    // Add keypoints
    raw_data.push_back(static_cast<float>(detection.keypoints.size()));
    for (const auto & kpt : detection.keypoints) {
      raw_data.push_back(kpt.x);
      raw_data.push_back(kpt.y);
      raw_data.push_back(kpt.z);         // confidence
    }
  }

  // Create string message for compatibility
  std::string raw_str;
  for (size_t i = 0; i < raw_data.size(); ++i) {
    raw_str += std::to_string(raw_data[i]);
    if (i < raw_data.size() - 1) {raw_str += ",";}
  }

  auto str_msg = std_msgs::msg::String();
  str_msg.data = raw_str;

  return str_msg;
}

void YOLOInterface::publishRawOutput(
  const InferenceResult & result,
  const std_msgs::msg::Header & header)
{
  auto str_msg = createRawOutputMessage(result, header);
  raw_output_pub_->publish(str_msg);
}

sensor_msgs::msg::Image::SharedPtr YOLOInterface::createVisualizationMessage(
  const cv::Mat & image, const InferenceResult & result,
  const std_msgs::msg::Header & header)
{
  cv::Mat vis_image = image.clone();

  auto keypoint_names = backend_->getKeypointNames();
  auto class_names = backend_->getClassNames();

  // Define colors for visualization
  std::vector<cv::Scalar> colors = {
    cv::Scalar(0, 255, 0),          // Green
    cv::Scalar(255, 0, 0),          // Blue
    cv::Scalar(0, 0, 255),          // Red
    cv::Scalar(255, 255, 0),        // Cyan
    cv::Scalar(255, 0, 255),        // Magenta
    cv::Scalar(0, 255, 255),        // Yellow
  };

  for (size_t i = 0; i < result.detections.size(); ++i) {
    const auto & detection = result.detections[i];
    cv::Scalar color = colors[i % colors.size()];

    // Draw bounding box - convert cv::Rect2f to cv::Rect for drawing
    cv::Point2f tl(detection.bbox.x, detection.bbox.y);
    cv::Point2f br(detection.bbox.x + detection.bbox.width,
      detection.bbox.y + detection.bbox.height);

    // Convert to integer coordinates for drawing
    cv::Point tl_int(static_cast<int>(std::round(tl.x)), static_cast<int>(std::round(tl.y)));
    cv::Point br_int(static_cast<int>(std::round(br.x)), static_cast<int>(std::round(br.y)));

    cv::rectangle(vis_image, tl_int, br_int, color, 2);

    // Draw label
    std::string label = static_cast<size_t>(detection.class_id) < class_names.size() ?
      class_names[detection.class_id] : "unknown";
    label += " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

    int baseline = 0;
    cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
    cv::rectangle(
      vis_image,
      cv::Point(tl_int.x, tl_int.y - text_size.height - baseline),
      cv::Point(tl_int.x + text_size.width, tl_int.y),
      color, -1);
    cv::putText(
      vis_image, label, cv::Point(tl_int.x, tl_int.y - baseline),
      cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

    // Draw keypoints
    for (size_t j = 0; j < detection.keypoints.size(); ++j) {
      const auto & kpt = detection.keypoints[j];
      if (kpt.z > keypoint_threshold_) {          // Only draw visible keypoints
        cv::Point2f point(kpt.x, kpt.y);
        cv::Point point_int(static_cast<int>(std::round(point.x)),
          static_cast<int>(std::round(point.y)));
        cv::circle(vis_image, point_int, 4, color, -1);

        // Draw keypoint name
        if (j < keypoint_names.size()) {
          cv::putText(
            vis_image, keypoint_names[j],
            cv::Point(point_int.x + 5, point_int.y - 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
        }
      }
    }
  }

  // Add performance info on image
  if (enable_profiling_) {
    double fps = frame_count_ > 0 ? frame_count_ / total_time_ : 0.0;
    std::string perf_text = "FPS: " + std::to_string(static_cast<int>(fps)) +
      " | Detections: " + std::to_string(result.detections.size());
    cv::putText(
      vis_image, perf_text, cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }

  // Create and return image message
  return cv_bridge::CvImage(header, "bgr8", vis_image).toImageMsg();
}

void YOLOInterface::publishVisualization(
  const cv::Mat & image, const InferenceResult & result,
  const std_msgs::msg::Header & header)
{
  auto img_msg = createVisualizationMessage(image, result, header);
  image_pub_->publish(*img_msg);
}

yolo_inference_cpp::msg::PerformanceInfo YOLOInterface::createPerformanceInfoMessage(
  const std_msgs::msg::Header & /*header*/)
{
  auto msg = yolo_inference_cpp::msg::PerformanceInfo();
  if (enable_profiling_) {
    msg.total_time_ms = profiler_->getLastTime("total_processing");
    msg.image_conversion_ms = profiler_->getLastTime("image_conversion");
    msg.preprocessing_ms = 0.0;
    msg.inference_ms = profiler_->getLastTime("inference");
    msg.postprocessing_ms = 0.0;
    msg.message_creation_ms = profiler_->getLastTime("message_creation");
    msg.detections_count = 0;
    msg.fps = frame_count_ > 0 ? frame_count_ / total_time_ : 0.0;
  }
  return msg;
}

void YOLOInterface::publishPerformanceInfo(const std_msgs::msg::Header & header)
{
  if (!enable_profiling_) {return;}

  auto msg = createPerformanceInfoMessage(header);
  performance_pub_->publish(msg);
}

void YOLOInterface::updatePerformanceMetrics()
{
  frame_count_++;
  auto now = std::chrono::steady_clock::now();
  auto elapsed = std::chrono::duration<double>(now - last_log_time_).count();
  total_time_ = elapsed;

  // Log performance every 5 seconds
  if (elapsed > 5.0) {
    double fps = frame_count_ / elapsed;
    RCLCPP_INFO(
      get_logger(), "Performance: %.1f FPS, %d frames processed",
      fps, frame_count_);

    if (enable_profiling_) {
      profiler_->logStats();
    }

    // Reset counters
    frame_count_ = 0;
    total_time_ = 0.0;
    last_log_time_ = now;

    if (enable_profiling_) {
      profiler_->reset();
    }
  }
}

}  // namespace yolo_inference
