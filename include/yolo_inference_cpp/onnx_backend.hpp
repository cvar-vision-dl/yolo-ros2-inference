#pragma once
#ifdef HAVE_ONNXRUNTIME
#include "inference_backend.hpp"
#include <onnxruntime_cxx_api.h>
#include <memory>

namespace yolo_inference {

class ONNXBackend : public InferenceBackend {
public:
    ONNXBackend();
    ~ONNXBackend() override;

    bool initialize(const std::string& model_path,
                   TaskType task,
                   int input_size = 640) override;

    InferenceResult infer(const cv::Mat& image,
                         float conf_threshold = 0.5f,
                         float nms_threshold = 0.4f,
                         float keypoint_threshold = 0.3f) override;

    std::vector<std::string> getClassNames() const override;
    std::vector<std::string> getKeypointNames() const override;
    bool isInitialized() const override { return initialized_; }
    ModelFormat getFormat() const override { return ModelFormat::ONNX; }
    TaskType getTask() const override { return task_type_; }

private:
    std::vector<Detection> postProcessPose(const float* output,
                                         const std::vector<int64_t>& output_shape,
                                         cv::Size input_size,
                                         cv::Size original_size,
                                         cv::Size2f scale_factors,
                                         cv::Point2f padding,
                                         float conf_threshold,
                                         float nms_threshold,
                                         float keypoint_threshold);

    std::vector<Detection> postProcessDetection(const float* output,
                                              const std::vector<int64_t>& output_shape,
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

} // namespace yolo_inference

#endif // HAVE_ONNXRUNTIME