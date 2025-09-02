#include "yolo_inference_cpp/preprocessing.hpp"

namespace yolo_inference {

Preprocessor::Preprocessor(MemoryPool& memory_pool) : memory_pool_(memory_pool) {}

cv::Mat Preprocessor::preprocess(const cv::Mat& input, int target_size, bool normalize,
                                cv::Scalar mean, cv::Scalar std) {
    // Calculate scale and padding for letterbox
    float scale = std::min(static_cast<float>(target_size) / input.cols,
                          static_cast<float>(target_size) / input.rows);

    int new_width = static_cast<int>(input.cols * scale);
    int new_height = static_cast<int>(input.rows * scale);

    scale_factors_ = cv::Size2f(scale, scale);
    padding_ = cv::Point2f((target_size - new_width) / 2.0f, (target_size - new_height) / 2.0f);

    // Resize image
    cv::Mat resized;
    cv::resize(input, resized, cv::Size(new_width, new_height), 0, 0, cv::INTER_LINEAR);

    // Create padded image
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                      padding_.y, target_size - new_height - padding_.y,
                      padding_.x, target_size - new_width - padding_.x,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    // Convert to float and normalize
    cv::Mat float_img;
    padded.convertTo(float_img, CV_32F, 1.0 / 255.0);

    if (normalize) {
        cv::subtract(float_img, mean, float_img);
        cv::divide(float_img, std, float_img);
    }

    // Convert BGR to RGB and change layout to CHW
    cv::Mat rgb;
    cv::cvtColor(float_img, rgb, cv::COLOR_BGR2RGB);

    // Convert HWC to CHW format
    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    // Get memory from pool for output
    cv::Mat output = memory_pool_.getFloatMat(1, target_size * target_size * 3, CV_32F);

    // Copy channel data
    float* output_ptr = output.ptr<float>();
    size_t channel_size = target_size * target_size;

    for (int c = 0; c < 3; ++c) {
        memcpy(output_ptr + c * channel_size, channels[c].ptr<float>(),
               channel_size * sizeof(float));
    }

    return output.reshape(1, {1, 3, target_size, target_size});
}
