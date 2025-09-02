// src/yolo_inference_node.cpp
#include "yolo_inference_cpp/inference_backend.hpp"
#include "yolo_inference_cpp/profiler.hpp"
#include "yolo_inference_cpp/memory_pool.hpp"
#include "yolo_inference_cpp/KeypointDetectionArray.hpp"
#include "yolo_inference_cpp/PerformanceInfo.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <memory>
#include <string>

namespace yolo_inference {

class YOLOInferenceNode : public rclcpp::Node {
public:
    YOLOInferenceNode() : Node("yolo_inference_node") {
        // Declare parameters
        declare_parameter<std::string>("model_path", "yolo11n-pose.onnx");
        declare_parameter<std::string>("task", "pose");
        declare_parameter<int>("input_size", 640);
        declare_parameter<double>("confidence_threshold", 0.5);
        declare_parameter<double>("nms_threshold", 0.4);
        declare_parameter<double>("keypoint_threshold", 0.3);
        declare_parameter<int>("max_detections", 20);
        declare_parameter<bool>("publish_visualization", false);
        declare_parameter<bool>("enable_profiling", true);
        declare_parameter<std::string>("input_topic", "/camera/image_raw/compressed");
        declare_parameter<std::string>("output_topic", "/yolo/detections");
        declare_parameter<std::string>("output_image_topic", "/yolo/result_image");
        declare_parameter<std::string>("performance_topic", "/yolo/performance");

        // Get parameters
        model_path_ = get_parameter("model_path").as_string();
        task_str_ = get_parameter("task").as_string();
        input_size_ = get_parameter("input_size").as_int();
        confidence_threshold_ = get_parameter("confidence_threshold").as_double();
        nms_threshold_ = get_parameter("nms_threshold").as_double();
        keypoint_threshold_ = get_parameter("keypoint_threshold").as_double();
        max_detections_ = get_parameter("max_detections").as_int();
        publish_visualization_ = get_parameter("publish_visualization").as_bool();
        enable_profiling_ = get_parameter("enable_profiling").as_bool();

        input_topic_ = get_parameter("input_topic").as_string();
        output_topic_ = get_parameter("output_topic").as_string();
        output_image_topic_ = get_parameter("output_image_topic").as_string();
        performance_topic_ = get_parameter("performance_topic").as_string();

        // Parse task
        task_type_ = stringToTaskType(task_str_);

        // Initialize memory pool (4MB default)
        memory_pool_ = std::make_unique<MemoryPool>(4 * 1024 * 1024);

        // Initialize profiler
        if (enable_profiling_) {
            profiler_ = std::make_unique<Profiler>();
        }

        // Initialize inference backend
        backend_ = createInferenceBackend(model_path_);
        if (!backend_ || !backend_->initialize(model_path_, task_type_, input_size_)) {
            RCLCPP_ERROR(get_logger(), "Failed to initialize inference backend");
            rclcpp::shutdown();
            return;
        }

        RCLCPP_INFO(get_logger(), "Initialized %s backend for %s task",
                   backend_->getFormat() == ModelFormat::TENSORRT ? "TensorRT" : "ONNX",
                   task_str_.c_str());

        // Initialize subscribers
        image_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
            input_topic_, 10,
            std::bind(&YOLOInferenceNode::imageCallback, this, std::placeholders::_1));

        // Initialize publishers
        detections_pub_ = create_publisher<yolo_inference::msg::KeypointDetectionArray>(
            output_topic_, 10);

        raw_output_pub_ = create_publisher<std_msgs::msg::String>(
            output_topic_ + "/raw", 10);

        if (publish_visualization_) {
            image_pub_ = create_publisher<sensor_msgs::msg::Image>(
                output_image_topic_, 10);
        }

        if (enable_profiling_) {
            performance_pub_ = create_publisher<yolo_inference::msg::PerformanceInfo>(
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

private:
    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        auto total_timer = enable_profiling_ ?
            profiler_->scopedTimer("total_processing") :
            Profiler::ScopedTimer(*profiler_, "dummy"); // Dummy timer if profiling disabled

        try {
            // Reset memory pool for each frame
            memory_pool_->reset();

            // Convert compressed image to OpenCV Mat
            cv::Mat image;
            {
                auto timer = enable_profiling_ ? profiler_->scopedTimer("image_conversion") :
                    Profiler::ScopedTimer(*profiler_, "dummy");

                std::vector<uint8_t> buffer(msg->data.begin(), msg->data.end());
                image = cv::imdecode(buffer, cv::IMREAD_COLOR);

                if (image.empty()) {
                    RCLCPP_ERROR(get_logger(), "Failed to decode compressed image");
                    return;
                }
            }

            // Run inference
            InferenceResult result;
            {
                auto timer = enable_profiling_ ? profiler_->scopedTimer("inference") :
                    Profiler::ScopedTimer(*profiler_, "dummy");

                result = backend_->infer(image,
                                       confidence_threshold_,
                                       nms_threshold_,
                                       keypoint_threshold_);
            }

            // Publish results
            {
                auto timer = enable_profiling_ ? profiler_->scopedTimer("message_creation") :
                    Profiler::ScopedTimer(*profiler_, "dummy");

                publishDetections(result, msg->header);
                publishRawOutput(result, msg->header);

                if (publish_visualization_) {
                    publishVisualization(image, result, msg->header);
                }

                if (enable_profiling_) {
                    publishPerformanceInfo(msg->header);
                }
            }

            // Update performance metrics
            updatePerformanceMetrics();

        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error processing image: %s", e.what());
        }
    }

    void publishDetections(const InferenceResult& result, const std_msgs::msg::Header& header) {
        auto msg = yolo_inference::msg::KeypointDetectionArray();
        msg.header = header;
        msg.model_type = backend_->getFormat() == ModelFormat::TENSORRT ? "TensorRT" : "ONNX";
        msg.task = task_str_;

        auto class_names = backend_->getClassNames();
        auto keypoint_names = backend_->getKeypointNames();

        for (const auto& detection : result.detections) {
            if (msg.detections.size() >= static_cast<size_t>(max_detections_)) break;

            yolo_inference::msg::KeypointDetection det_msg;
            det_msg.header = header;

            // Basic detection info
            det_msg.class_id = detection.class_id;
            det_msg.label = detection.class_id < class_names.size() ?
                          class_names[detection.class_id] : "unknown";
            det_msg.confidence = detection.confidence;

            // Bounding box
            det_msg.bounding_box.x1 = detection.bbox.x;
            det_msg.bounding_box.y1 = detection.bbox.y;
            det_msg.bounding_box.x2 = detection.bbox.x + detection.bbox.width;
            det_msg.bounding_box.y2 = detection.bbox.y + detection.bbox.height;
            det_msg.bounding_box.confidence = detection.confidence;

            // Keypoints
            for (size_t i = 0; i < detection.keypoints.size() && i < keypoint_names.size(); ++i) {
                const auto& kpt = detection.keypoints[i];

                yolo_inference::msg::Keypoint kpt_msg;
                kpt_msg.name = keypoint_names[i];
                kpt_msg.x = kpt.x;
                kpt_msg.y = kpt.y;
                kpt_msg.confidence = kpt.z;
                kpt_msg.visible = kpt.z > keypoint_threshold_;

                det_msg.keypoints.push_back(kpt_msg);
            }

            msg.detections.push_back(det_msg);
        }

        detections_pub_->publish(msg);
    }

    void publishRawOutput(const InferenceResult& result, const std_msgs::msg::Header& header) {
        // Create raw array format for high-performance applications
        std::vector<float> raw_data;

        for (const auto& detection : result.detections) {
            if (raw_data.size() / 6 >= static_cast<size_t>(max_detections_)) break;

            // Basic format: [x1, y1, x2, y2, confidence, class_id, num_keypoints, kpt_data...]
            raw_data.push_back(detection.bbox.x);
            raw_data.push_back(detection.bbox.y);
            raw_data.push_back(detection.bbox.x + detection.bbox.width);
            raw_data.push_back(detection.bbox.y + detection.bbox.height);
            raw_data.push_back(detection.confidence);
            raw_data.push_back(static_cast<float>(detection.class_id));

            // Add keypoints
            raw_data.push_back(static_cast<float>(detection.keypoints.size()));
            for (const auto& kpt : detection.keypoints) {
                raw_data.push_back(kpt.x);
                raw_data.push_back(kpt.y);
                raw_data.push_back(kpt.z); // confidence
            }
        }

        auto msg = yolo_inference::msg::KeypointDetectionArray();
        msg.header = header;
        msg.raw_output = raw_data;

        // Also publish as string for compatibility
        std::string raw_str;
        for (size_t i = 0; i < raw_data.size(); ++i) {
            raw_str += std::to_string(raw_data[i]);
            if (i < raw_data.size() - 1) raw_str += ",";
        }

        auto str_msg = std_msgs::msg::String();
        str_msg.data = raw_str;
        raw_output_pub_->publish(str_msg);
    }

    void publishVisualization(const cv::Mat& image, const InferenceResult& result,
                             const std_msgs::msg::Header& header) {
        cv::Mat vis_image = image.clone();

        auto keypoint_names = backend_->getKeypointNames();
        auto class_names = backend_->getClassNames();

        // Define colors for visualization
        std::vector<cv::Scalar> colors = {
            cv::Scalar(0, 255, 0),    // Green
            cv::Scalar(255, 0, 0),    // Blue
            cv::Scalar(0, 0, 255),    // Red
            cv::Scalar(255, 255, 0),  // Cyan
            cv::Scalar(255, 0, 255),  // Magenta
            cv::Scalar(0, 255, 255),  // Yellow
        };

        for (size_t i = 0; i < result.detections.size(); ++i) {
            const auto& detection = result.detections[i];
            cv::Scalar color = colors[i % colors.size()];

            // Draw bounding box
            cv::Point2f tl(detection.bbox.x, detection.bbox.y);
            cv::Point2f br(detection.bbox.x + detection.bbox.width,
                          detection.bbox.y + detection.bbox.height);
            cv::rectangle(vis_image, tl, br, color, 2);

            // Draw label
            std::string label = detection.class_id < class_names.size() ?
                              class_names[detection.class_id] : "unknown";
            label += " " + std::to_string(static_cast<int>(detection.confidence * 100)) + "%";

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            cv::rectangle(vis_image,
                         cv::Point(tl.x, tl.y - text_size.height - baseline),
                         cv::Point(tl.x + text_size.width, tl.y),
                         color, -1);
            cv::putText(vis_image, label, cv::Point(tl.x, tl.y - baseline),
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

            // Draw keypoints
            for (size_t j = 0; j < detection.keypoints.size(); ++j) {
                const auto& kpt = detection.keypoints[j];
                if (kpt.z > keypoint_threshold_) {  // Only draw visible keypoints
                    cv::Point2f point(kpt.x, kpt.y);
                    cv::circle(vis_image, point, 4, color, -1);

                    // Draw keypoint name
                    if (j < keypoint_names.size()) {
                        cv::putText(vis_image, keypoint_names[j],
                                  cv::Point(point.x + 5, point.y - 5),
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
            cv::putText(vis_image, perf_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        }

        // Publish visualization
        auto img_msg = cv_bridge::CvImage(header, "bgr8", vis_image).toImageMsg();
        image_pub_->publish(*img_msg);
    }

    void publishPerformanceInfo(const std_msgs::msg::Header& header) {
        if (!enable_profiling_) return;

        auto msg = yolo_inference::msg::PerformanceInfo();
        msg.total_time_ms = profiler_->getLastTime("total_processing");
        msg.image_conversion_ms = profiler_->getLastTime("image_conversion");
        msg.preprocessing_ms = 0.0; // Included in inference time
        msg.inference_ms = profiler_->getLastTime("inference");
        msg.postprocessing_ms = 0.0; // Included in inference time
        msg.message_creation_ms = profiler_->getLastTime("message_creation");
        msg.detections_count = 0; // Will be set by callback
        msg.fps = frame_count_ > 0 ? frame_count_ / total_time_ : 0.0;

        performance_pub_->publish(msg);
    }

    void updatePerformanceMetrics() {
        frame_count_++;
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - last_log_time_).count();
        total_time_ = elapsed;

        // Log performance every 5 seconds
        if (elapsed > 5.0) {
            double fps = frame_count_ / elapsed;
            RCLCPP_INFO(get_logger(), "Performance: %.1f FPS, %d frames processed",
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
    std::unique_ptr<MemoryPool> memory_pool_;
    std::unique_ptr<Profiler> profiler_;

    // ROS components
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    rclcpp::Publisher<yolo_inference::msg::KeypointDetectionArray>::SharedPtr detections_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr raw_output_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    rclcpp::Publisher<yolo_inference::msg::PerformanceInfo>::SharedPtr performance_pub_;

    // Performance tracking
    int frame_count_;
    double total_time_;
    std::chrono::steady_clock::time_point last_log_time_;
};

} // namespace yolo_inference

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    try {
        auto node = std::make_shared<yolo_inference::YOLOInferenceNode>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("yolo_inference"), "Exception: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}