#include "yolo_inference_cpp/memory_pool.hpp"
#include <cstring>
#include <stdexcept>

namespace yolo_inference {

MemoryPool::MemoryPool(size_t buffer_size)
    : buffer_size_(buffer_size)
    , current_offset_(0)
    , float_mat_index_(0)
    , uchar_mat_index_(0) {

    buffer_ = std::make_unique<uint8_t[]>(buffer_size_);

    // Pre-allocate some common Mat sizes
    float_mats_.reserve(10);
    uchar_mats_.reserve(10);
}

MemoryPool::~MemoryPool() = default;

void* MemoryPool::allocate(size_t size, size_t alignment) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Align the current offset
    size_t aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);

    if (aligned_offset + size > buffer_size_) {
        throw std::runtime_error("MemoryPool: Out of memory");
    }

    void* ptr = buffer_.get() + aligned_offset;
    current_offset_ = aligned_offset + size;

    return ptr;
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_offset_ = 0;
    float_mat_index_ = 0;
    uchar_mat_index_ = 0;
}

cv::Mat MemoryPool::getFloatMat(int rows, int cols, int type) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (float_mat_index_ < float_mats_.size()) {
        cv::Mat& mat = float_mats_[float_mat_index_++];
        if (mat.rows == rows && mat.cols == cols && mat.type() == type) {
            return mat;
        }
    }

    // Allocate new Mat
    size_t size_needed = rows * cols * CV_ELEM_SIZE(type);
    void* data = allocate(size_needed, 32); // 32-byte alignment for SIMD

    cv::Mat mat(rows, cols, type, data);

    if (float_mats_.size() <= float_mat_index_) {
        float_mats_.push_back(mat.clone());
    } else {
        float_mats_[float_mat_index_] = mat.clone();
    }

    float_mat_index_++;
    return mat;
}

cv::Mat MemoryPool::getUCharMat(int rows, int cols, int type) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (uchar_mat_index_ < uchar_mats_.size()) {
        cv::Mat& mat = uchar_mats_[uchar_mat_index_++];
        if (mat.rows == rows && mat.cols == cols && mat.type() == type) {
            return mat;
        }
    }

    // Allocate new Mat
    size_t size_needed = rows * cols * CV_ELEM_SIZE(type);
    void* data = allocate(size_needed, 32);

    cv::Mat mat(rows, cols, type, data);

    if (uchar_mats_.size() <= uchar_mat_index_) {
        uchar_mats_.push_back(mat.clone());
    } else {
        uchar_mats_[uchar_mat_index_] = mat.clone();
    }

    uchar_mat_index_++;
    return mat;
}