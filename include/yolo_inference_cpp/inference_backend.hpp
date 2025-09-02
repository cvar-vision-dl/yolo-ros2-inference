#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

namespace yolo_inference {

enum class TaskType {
    DETECT,
    POSE,
    SEGMENT
};

enum class ModelFormat {
    TENSORRT,
    ONNX,
    UNKNOWN
};

struct Detection {
    cv::Rect2f bbox;
    float confidence;
    int class_id;
    std::vector<cv::Point3f> keypoints; // x, y, confidence
    cv::Mat mask; // For segmentation
};

struct InferenceResult {
    std::vector<Detection> detections;
    cv::Size input_size;
    cv::Size original_size;
    double inference_time_ms;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;

    virtual bool initialize(const std::string& model_path,
                          TaskType task,
                          int input_size = 640) = 0;

    virtual InferenceResult infer(const cv::Mat& image,
                                float conf_threshold = 0.5f,
                                float nms_threshold = 0.4f,
                                float keypoint_threshold = 0.3f) = 0;

    virtual std::vector<std::string> getClassNames() const = 0;
    virtual std::vector<std::string> getKeypointNames() const = 0;
    virtual bool isInitialized() const = 0;
    virtual ModelFormat getFormat() const = 0;
    virtual TaskType getTask() const = 0;
};

std::unique_ptr<InferenceBackend> createInferenceBackend(const std::string& model_path);
ModelFormat detectModelFormat(const std::string& model_path);
TaskType stringToTaskType(const std::string& task_str);

} // namespace yolo_inference