#ifdef HAVE_TENSORRT

#include "yolo_inference_cpp/tensorrt_backend.hpp"
#include "yolo_inference_cpp/preprocessing.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>

namespace yolo_inference {

void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TensorRT] " << msg << std::endl;
    }
}

TensorRTBackend::TensorRTBackend(MemoryPool& memory_pool)
    : memory_pool_(memory_pool)
    , task_type_(TaskType::POSE)
    , input_size_(640)
    , initialized_(false)
    , stream_(nullptr)
    , device_buffers_(nullptr)
    , host_buffers_(nullptr)
    , input_binding_(-1)
    , output_binding_(-1)
    , input_size_bytes_(0)
    , output_size_bytes_(0) {

    logger_ = std::make_unique<TensorRTLogger>();
}

TensorRTBackend::~TensorRTBackend() {
    deallocateBuffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool TensorRTBackend::initialize(const std::string& model_path,
                                TaskType task,
                                int input_size) {
    task_type_ = task;
    input_size_ = input_size;

    // Initialize CUDA
    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "Failed to set CUDA device" << std::endl;
        return false;
    }

    // Create CUDA stream
    if (cudaStreamCreate(&stream_) != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream" << std::endl;
        return false;
    }

    // Load TensorRT engine
    if (!loadEngine(model_path)) {
        std::cerr << "Failed to load TensorRT engine" << std::endl;
        return false;
    }

    // Setup input/output bindings
    if (!setupBindings()) {
        std::cerr << "Failed to setup bindings" << std::endl;
        return false;
    }

    // Allocate GPU memory
    allocateBuffers();

    // Initialize class names (COCO classes for now)
    class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };

    // Initialize keypoint names (COCO pose format)
    keypoint_names_ = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    };

    initialized_ = true;
    std::cout << "TensorRT backend initialized successfully" << std::endl;

    return true;
}

bool TensorRTBackend::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open engine file: " << engine_path << std::endl;
        return false;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read engine data
    std::vector<char> engine_data(size);
    file.read(engine_data.data(), size);
    file.close();

    // Create runtime and deserialize engine
    runtime_.reset(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), size));
    if (!engine_) {
        std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
        return false;
    }

    // Create execution context
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    return true;
}

bool TensorRTBackend::setupBindings() {
    int num_bindings = engine_->getNbBindings();

    for (int i = 0; i < num_bindings; ++i) {
        if (engine_->bindingIsInput(i)) {
            input_binding_ = i;
            input_dims_ = engine_->getBindingDimensions(i);
            input_size_bytes_ = 1;
            for (int j = 0; j < input_dims_.nbDims; ++j) {
                input_size_bytes_ *= input_dims_.d[j];
            }
            input_size_bytes_ *= sizeof(float);
        } else {
            output_binding_ = i;
            output_dims_ = engine_->getBindingDimensions(i);
            output_size_bytes_ = 1;
            for (int j = 0; j < output_dims_.nbDims; ++j) {
                output_size_bytes_ *= output_dims_.d[j];
            }
            output_size_bytes_ *= sizeof(float);
        }
    }

    return input_binding_ != -1 && output_binding_ != -1;
}

void TensorRTBackend::allocateBuffers() {
    device_buffers_ = new void*[2];
    host_buffers_ = new void*[2];

    // Allocate device memory
    cudaMalloc(&device_buffers_[input_binding_], input_size_bytes_);
    cudaMalloc(&device_buffers_[output_binding_], output_size_bytes_);

    // Allocate host memory
    cudaMallocHost(&host_buffers_[input_binding_], input_size_bytes_);
    cudaMallocHost(&host_buffers_[output_binding_], output_size_bytes_);
}

void TensorRTBackend::deallocateBuffers() {
    if (device_buffers_) {
        cudaFree(device_buffers_[input_binding_]);
        cudaFree(device_buffers_[output_binding_]);
        delete[] device_buffers_;
        device_buffers_ = nullptr;
    }

    if (host_buffers_) {
        cudaFreeHost(host_buffers_[input_binding_]);
        cudaFreeHost(host_buffers_[output_binding_]);
        delete[] host_buffers_;
        host_buffers_ = nullptr;
    }
}

InferenceResult TensorRTBackend::infer(const cv::Mat& image,
                                     float conf_threshold,
                                     float nms_threshold,
                                     float keypoint_threshold) {
    InferenceResult result;
    result.original_size = image.size();
    result.input_size = cv::Size(input_size_, input_size_);

    auto start_time = std::chrono::high_resolution_clock::now();

    // Preprocess image
    Preprocessor preprocessor(memory_pool_);
    cv::Mat processed = preprocessor.preprocess(image, input_size_);

    // Copy data to host buffer
    float* input_host = static_cast<float*>(host_buffers_[input_binding_]);
    memcpy(input_host, processed.ptr<float>(), input_size_bytes_);

    // Copy from host to device
    cudaMemcpyAsync(device_buffers_[input_binding_],
                   host_buffers_[input_binding_],
                   input_size_bytes_,
                   cudaMemcpyHostToDevice,
                   stream_);

    // Run inference
    context_->enqueueV2(device_buffers_, stream_, nullptr);

    // Copy output from device to host
    cudaMemcpyAsync(host_buffers_[output_binding_],
                   device_buffers_[output_binding_],
                   output_size_bytes_,
                   cudaMemcpyDeviceToHost,
                   stream_);

    // Synchronize
    cudaStreamSynchronize(stream_);

    auto end_time = std::chrono::high_resolution_clock::now();
    result.inference_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time).count();

    // Post-process results
    float* output_host = static_cast<float*>(host_buffers_[output_binding_]);

    if (task_type_ == TaskType::POSE) {
        result.detections = postProcessPose(output_host,
                                          result.input_size,
                                          result.original_size,
                                          conf_threshold,
                                          nms_threshold,
                                          keypoint_threshold);
    } else {
        result.detections = postProcessDetection(output_host,
                                               result.input_size,
                                               result.original_size,
                                               conf_threshold,
                                               nms_threshold);
    }

    return result;
}

std::vector<Detection> TensorRTBackend::postProcessPose(float* output,
                                                      cv::Size input_size,
                                                      cv::Size original_size,
                                                      float conf_threshold,
                                                      float nms_threshold,
                                                      float keypoint_threshold) {
    std::vector<Detection> detections;

    // YOLO pose output format: [batch, (x,y,w,h,conf,class_conf) + keypoints*3, num_anchors]
    // Assuming output_dims_: [1, 56, 8400] for pose (4+1+1+17*3 = 56)

    int num_classes = 1; // Person class for pose
    int num_keypoints = (output_dims_.d[1] - 6) / 3; // (total - bbox - conf - class) / 3
    int num_anchors = output_dims_.d[2];

    float scale_x = static_cast<float>(original_size.width) / input_size.width;
    float scale_y = static_cast<float>(original_size.height) / input_size.height;

    std::vector<cv::Rect2f> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point3f>> keypoints_list;

    for (int i = 0; i < num_anchors; ++i) {
        float conf = output[4 * num_anchors + i]; // Object confidence

        if (conf >= conf_threshold) {
            // Extract bounding box
            float cx = output[0 * num_anchors + i];
            float cy = output[1 * num_anchors + i];
            float w = output[2 * num_anchors + i];
            float h = output[3 * num_anchors + i];

            // Convert to corner format and scale
            float x1 = (cx - w/2) * scale_x;
            float y1 = (cy - h/2) * scale_y;
            float x2 = (cx + w/2) * scale_x;
            float y2 = (cy + h/2) * scale_y;

            boxes.push_back(cv::Rect2f(x1, y1, x2-x1, y2-y1));
            confidences.push_back(conf);

            // Extract keypoints
            std::vector<cv::Point3f> kpts;
            for (int k = 0; k < num_keypoints; ++k) {
                float kx = output[(6 + k*3) * num_anchors + i] * scale_x;
                float ky = output[(6 + k*3 + 1) * num_anchors + i] * scale_y;
                float kconf = output[(6 + k*3 + 2) * num_anchors + i];

                kpts.push_back(cv::Point3f(kx, ky, kconf));
            }
            keypoints_list.push_back(kpts);
        }
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];
        det.confidence = confidences[idx];
        det.class_id = 0; // Person class
        det.keypoints = keypoints_list[idx];

        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> TensorRTBackend::postProcessDetection(float* output,
                                                           cv::Size input_size,
                                                           cv::Size original_size,
                                                           float conf_threshold,
                                                           float nms_threshold) {
    // Standard YOLO detection post-processing
    // Implementation similar to pose but without keypoints
    std::vector<Detection> detections;

    // This would be implemented based on the specific YOLO detection model format
    // For now, return empty for detection models

    return detections;
}

std::vector<std::string> TensorRTBackend::getClassNames() const {
    return class_names_;
}

std::vector<std::string> TensorRTBackend::getKeypointNames() const {
    return keypoint_names_;
}

} // namespace yolo_inference

#endif // HAVE_TENSORRT