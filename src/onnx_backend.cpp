#ifdef HAVE_ONNXRUNTIME

#include "yolo_inference_cpp/onnx_backend.hpp"
#include "yolo_inference_cpp/preprocessing.hpp"
#include <iostream>
#include <algorithm>

namespace yolo_inference {

ONNXBackend::ONNXBackend(MemoryPool& memory_pool)
    : memory_pool_(memory_pool)
    , task_type_(TaskType::POSE)
    , input_size_(640)
    , initialized_(false) {
}

ONNXBackend::~ONNXBackend() {
    // Cleanup handled by unique_ptr destructors
}

bool ONNXBackend::initialize(const std::string& model_path,
                            TaskType task,
                            int input_size) {
    task_type_ = task;
    input_size_ = input_size;

    try {
        // Create environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOInference");

        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Enable CUDA provider if available
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        session_options_->AppendExecutionProvider_CUDA(cuda_options);

        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);

        // Create memory info
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        // Get input/output names and shapes
        Ort::AllocatorWithDefaultOptions allocator;

        // Input info
        size_t num_input_nodes = session_->GetInputCount();
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(input_name.get());

            Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_shape = input_tensor_info.GetShape();
            input_shapes_.push_back(input_shape);
        }

        // Output info
        size_t num_output_nodes = session_->GetOutputCount();
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(output_name.get());

            Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_shape = output_tensor_info.GetShape();
            output_shapes_.push_back(output_shape);
        }

        // Initialize class names (COCO classes)
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
        std::cout << "ONNX Runtime backend initialized successfully" << std::endl;

        // Print model info
        std::cout << "Input shape: [";
        for (size_t i = 0; i < input_shapes_[0].size(); ++i) {
            std::cout << input_shapes_[0][i];
            if (i < input_shapes_[0].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        std::cout << "Output shape: [";
        for (size_t i = 0; i < output_shapes_[0].size(); ++i) {
            std::cout << output_shapes_[0][i];
            if (i < output_shapes_[0].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

InferenceResult ONNXBackend::infer(const cv::Mat& image,
                                  float conf_threshold,
                                  float nms_threshold,
                                  float keypoint_threshold) {
    std::cout << "ONNX DEBUG: Starting inference..." << std::endl;
    InferenceResult result;
    result.original_size = image.size();
    result.input_size = cv::Size(input_size_, input_size_);

    auto start_time = std::chrono::high_resolution_clock::now();

    try {
        std::cout << "ONNX DEBUG: Starting preprocessing..." << std::endl;
        // Preprocess image
        Preprocessor preprocessor(memory_pool_);
        cv::Mat processed = preprocessor.preprocess(image, input_size_);

        std::cout << "ONNX DEBUG: Preprocessing completed, processed size: "
                  << processed.rows << "x" << processed.cols << std::endl;

        // Create input tensor
        std::cout << "ONNX DEBUG: Creating input tensor..." << std::endl;
        std::vector<int64_t> input_shape = input_shapes_[0];
        input_shape[0] = 1; // batch size
        input_shape[2] = input_size_; // height
        input_shape[3] = input_size_; // width
        std::cout << "ONNX DEBUG: Input shape: [" << input_shape[0] << ","
          << input_shape[1] << "," << input_shape[2] << "," << input_shape[3] << "]" << std::endl;


        size_t input_tensor_size = 1;
        for (auto& dim : input_shape) {
            input_tensor_size *= dim;
        }
        std::cout << "ONNX DEBUG: Input tensor size: " << input_tensor_size << std::endl;
        std::vector<float> input_tensor_values(processed.ptr<float>(),
                                             processed.ptr<float>() + input_tensor_size);
        std::cout << "ONNX DEBUG: Input tensor values copied" << std::endl;

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            *memory_info_, input_tensor_values.data(), input_tensor_size,
            input_shape.data(), input_shape.size()));

        // Run inference
        std::cout << "ONNX DEBUG: About to run session..." << std::endl;
        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                          input_names_.data(),
                                          input_tensors.data(),
                                          input_names_.size(),
                                          output_names_.data(),
                                          output_names_.size());
        std::cout << "ONNX DEBUG: Session run completed!" << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        result.inference_time_ms = std::chrono::duration<double, std::milli>(
            end_time - start_time).count();

        // Process output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        if (task_type_ == TaskType::POSE) {
            result.detections = postProcessPose(output_data,
                                              output_shape,
                                              result.input_size,
                                              result.original_size,
                                              conf_threshold,
                                              nms_threshold,
                                              keypoint_threshold);
        } else {
            result.detections = postProcessDetection(output_data,
                                                   output_shape,
                                                   result.input_size,
                                                   result.original_size,
                                                   conf_threshold,
                                                   nms_threshold);
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
    }

    return result;
}

std::vector<Detection> ONNXBackend::postProcessPose(const float* output,
                                                   const std::vector<int64_t>& output_shape,
                                                   cv::Size input_size,
                                                   cv::Size original_size,
                                                   float conf_threshold,
                                                   float nms_threshold,
                                                   float keypoint_threshold) {
    std::vector<Detection> detections;

    // YOLO pose output format: [batch, anchors, (x,y,w,h,conf,class_conf) + keypoints*3]
    // For pose models: [1, 8400, 56] where 56 = 4+1+1+17*3

    if (output_shape.size() != 3) {
        std::cerr << "Unexpected output shape dimensions: " << output_shape.size() << std::endl;
        return detections;
    }

    int64_t batch_size = output_shape[0];
    int64_t num_anchors = output_shape[1];
    int64_t data_size = output_shape[2];

    int num_keypoints = (data_size - 6) / 3; // (total - bbox - conf - class) / 3

    float scale_x = static_cast<float>(original_size.width) / input_size.width;
    float scale_y = static_cast<float>(original_size.height) / input_size.height;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point3f>> keypoints_list;

    for (int64_t i = 0; i < num_anchors; ++i) {
        const float* anchor_data = output + i * data_size;

        float conf = anchor_data[4]; // Object confidence

        if (conf >= conf_threshold) {
            // Extract bounding box (center format)
            float cx = anchor_data[0];
            float cy = anchor_data[1];
            float w = anchor_data[2];
            float h = anchor_data[3];

            // Convert to corner format and scale
            float x1 = (cx - w/2) * scale_x;
            float y1 = (cy - h/2) * scale_y;
            float x2 = (cx + w/2) * scale_x;
            float y2 = (cy + h/2) * scale_y;

            boxes.push_back(cv::Rect(static_cast<int>(x1), static_cast<int>(y1),
                        static_cast<int>(x2-x1), static_cast<int>(y2-y1)));
            confidences.push_back(conf);

            // Extract keypoints
            std::vector<cv::Point3f> kpts;
            for (int k = 0; k < num_keypoints; ++k) {
                float kx = anchor_data[6 + k*3] * scale_x;
                float ky = anchor_data[6 + k*3 + 1] * scale_y;
                float kconf = anchor_data[6 + k*3 + 2];

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

std::vector<Detection> ONNXBackend::postProcessDetection(const float* output,
                                                        const std::vector<int64_t>& output_shape,
                                                        cv::Size input_size,
                                                        cv::Size original_size,
                                                        float conf_threshold,
                                                        float nms_threshold) {
    // Standard YOLO detection post-processing
    std::vector<Detection> detections;

    // Implementation would be similar to pose processing but without keypoints
    // For now, return empty for detection models

    return detections;
}

std::vector<std::string> ONNXBackend::getClassNames() const {
    return class_names_;
}

std::vector<std::string> ONNXBackend::getKeypointNames() const {
    return keypoint_names_;
}

} // namespace yolo_inference

#endif // HAVE_ONNXRUNTIME