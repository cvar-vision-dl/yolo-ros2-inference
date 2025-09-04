#include "yolo_inference_cpp/preprocessing.hpp"
#include <opencv2/opencv.hpp>

namespace yolo_inference {

Preprocessor::Preprocessor()
    : scale_factors_(1.0f, 1.0f)
    , padding_(0.0f, 0.0f) {
}

cv::Mat Preprocessor::preprocess(const cv::Mat& input,
                                int target_size,
                                bool normalize,
                                cv::Scalar mean,
                                cv::Scalar std) {
    // Calculate scale factor to maintain aspect ratio
    float scale = std::min(static_cast<float>(target_size) / input.cols,
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
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                      pad_y, target_size - new_height - pad_y,
                      pad_x, target_size - new_width - pad_x,
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

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
            cv::Mat channel = channels[2 - c]; // BGR to RGB conversion
            std::memcpy(chw_result.ptr<float>() + c * target_size * target_size,
                       channel.ptr<float>(), target_size * target_size * sizeof(float));
        }

        // Reshape to [1, 3, height, width]
        chw_result = chw_result.reshape(1, {1, 3, target_size, target_size});
    } else if (result.channels() == 1) {
        // Grayscale image - reshape to [1, 1, height, width]
        chw_result = result.reshape(1, {1, 1, target_size, target_size});
    } else {
        throw std::runtime_error("Unsupported number of channels: " + std::to_string(result.channels()));
    }

    return chw_result;
}

} // namespace yolo_inference