#pragma once

#include <memory>
#include <vector>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace yolo_inference {

class MemoryPool {
public:
    MemoryPool(size_t buffer_size = 1920 * 1080 * 3 * 4); // Default for FHD RGB float
    ~MemoryPool();

    void* allocate(size_t size, size_t alignment = 32);
    void reset();

    // OpenCV Mat allocators
    cv::Mat getFloatMat(int rows, int cols, int type = CV_32FC3);
    cv::Mat getUCharMat(int rows, int cols, int type = CV_8UC3);

private:
    std::unique_ptr<uint8_t[]> buffer_;
    size_t buffer_size_;
    size_t current_offset_;
    std::mutex mutex_;

    std::vector<cv::Mat> float_mats_;
    std::vector<cv::Mat> uchar_mats_;
    size_t float_mat_index_;
    size_t uchar_mat_index_;
};

} // namespace yolo_inference