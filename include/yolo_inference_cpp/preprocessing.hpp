#pragma once

#include <opencv2/opencv.hpp>

namespace yolo_inference {

class Preprocessor {
public:
    Preprocessor();

    cv::Mat preprocess(const cv::Mat& input,
                      int target_size,
                      bool normalize = true,
                      cv::Scalar mean = cv::Scalar(0, 0, 0),
                      cv::Scalar std = cv::Scalar(1, 1, 1));

    // Get scaling factors for coordinate transformation
    cv::Size2f getScaleFactors() const { return scale_factors_; }
    cv::Point2f getPadding() const { return padding_; }

private:
    cv::Size2f scale_factors_;
    cv::Point2f padding_;
};

} // namespace yolo_inference