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
    std::cout << "MEMPOOL DEBUG: allocate() called - size: " << size
              << ", alignment: " << alignment << std::endl;

    // NOTE: No mutex lock here because caller (getFloatMat) already holds it

    std::cout << "MEMPOOL DEBUG: current_offset_: " << current_offset_
              << ", buffer_size_: " << buffer_size_ << std::endl;

    // Align the current offset
    std::cout << "MEMPOOL DEBUG: Calculating aligned offset..." << std::endl;
    size_t aligned_offset = (current_offset_ + alignment - 1) & ~(alignment - 1);
    std::cout << "MEMPOOL DEBUG: aligned_offset: " << aligned_offset << std::endl;

    std::cout << "MEMPOOL DEBUG: Checking bounds..." << std::endl;
    if (aligned_offset + size > buffer_size_) {
        std::cout << "MEMPOOL DEBUG: Out of memory! Requested: " << size
                  << ", Available: " << (buffer_size_ - aligned_offset)
                  << ", Total pool: " << buffer_size_ << std::endl;
        throw std::runtime_error("MemoryPool: Out of memory");
    }
    std::cout << "MEMPOOL DEBUG: Bounds check passed" << std::endl;

    std::cout << "MEMPOOL DEBUG: Getting pointer..." << std::endl;
    void* ptr = buffer_.get() + aligned_offset;
    std::cout << "MEMPOOL DEBUG: Pointer obtained: " << ptr << std::endl;

    std::cout << "MEMPOOL DEBUG: Updating current_offset_..." << std::endl;
    current_offset_ = aligned_offset + size;
    std::cout << "MEMPOOL DEBUG: current_offset_ updated to: " << current_offset_ << std::endl;

    std::cout << "MEMPOOL DEBUG: Returning pointer" << std::endl;
    return ptr;
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    current_offset_ = 0;
    float_mat_index_ = 0;
    uchar_mat_index_ = 0;
}

cv::Mat MemoryPool::getFloatMat(int rows, int cols, int type) {
    std::cout << "MEMPOOL DEBUG: getFloatMat called - rows: " << rows
              << ", cols: " << cols << ", type: " << type << std::endl;

    std::cout << "MEMPOOL DEBUG: About to acquire lock..." << std::endl;
    std::lock_guard<std::mutex> lock(mutex_);
    std::cout << "MEMPOOL DEBUG: Lock acquired" << std::endl;

    std::cout << "MEMPOOL DEBUG: float_mat_index_: " << float_mat_index_
              << ", float_mats_.size(): " << float_mats_.size() << std::endl;

    if (float_mat_index_ < float_mats_.size()) {
        std::cout << "MEMPOOL DEBUG: Checking existing mat..." << std::endl;
        cv::Mat& mat = float_mats_[float_mat_index_++];
        if (mat.rows == rows && mat.cols == cols && mat.type() == type) {
            std::cout << "MEMPOOL DEBUG: Reusing existing mat" << std::endl;
            return mat;
        }
        std::cout << "MEMPOOL DEBUG: Existing mat doesn't match, creating new" << std::endl;
    }

    // Allocate new Mat
    std::cout << "MEMPOOL DEBUG: Calculating size needed..." << std::endl;
    size_t size_needed = rows * cols * CV_ELEM_SIZE(type);
    std::cout << "MEMPOOL DEBUG: Size needed: " << size_needed << " bytes" << std::endl;

    std::cout << "MEMPOOL DEBUG: About to allocate memory..." << std::endl;
    void* data = allocate(size_needed, 32); // 32-byte alignment for SIMD
    std::cout << "MEMPOOL DEBUG: Memory allocated at: " << data << std::endl;

    std::cout << "MEMPOOL DEBUG: Creating cv::Mat..." << std::endl;
    cv::Mat mat(rows, cols, type, data);
    std::cout << "MEMPOOL DEBUG: cv::Mat created" << std::endl;

    std::cout << "MEMPOOL DEBUG: About to store in vector..." << std::endl;
    if (float_mats_.size() <= float_mat_index_) {
        std::cout << "MEMPOOL DEBUG: Expanding vector and pushing back..." << std::endl;
        float_mats_.push_back(mat.clone());
        std::cout << "MEMPOOL DEBUG: Vector expanded" << std::endl;
    } else {
        std::cout << "MEMPOOL DEBUG: Updating existing vector entry..." << std::endl;
        float_mats_[float_mat_index_] = mat.clone();
        std::cout << "MEMPOOL DEBUG: Vector entry updated" << std::endl;
    }

    float_mat_index_++;
    std::cout << "MEMPOOL DEBUG: Incremented index, returning mat" << std::endl;
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

} // namespace yolo_inference