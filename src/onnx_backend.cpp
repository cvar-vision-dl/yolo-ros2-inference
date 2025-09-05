#ifdef HAVE_ONNXRUNTIME

#include "yolo_inference_cpp/onnx_backend.hpp"
#include "yolo_inference_cpp/preprocessing.hpp"
#include <iostream>
#include <algorithm>

namespace yolo_inference {

ONNXBackend::ONNXBackend()
    : task_type_(TaskType::POSE)
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

        class_names_ = {
            "gate"
        };

        // Initialize keypoint names (COCO pose format)
        keypoint_names_ = {
            "bottom_right_outer",
            "bottom_left_outer",
            "top_right_outer",
            "top_left_outer",
            "bottom_right_inner",
            "bottom_left_inner",
            "top_right_inner",
            "top_left_inner"
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
        Preprocessor preprocessor;
        cv::Mat processed = preprocessor.preprocess(image, input_size_);

        // Store preprocessing info for coordinate transformation
        cv::Size2f scale_factors = preprocessor.getScaleFactors();
        cv::Point2f padding = preprocessor.getPadding();

        std::cout << "ONNX DEBUG: Preprocessing completed, processed size: "
                  << processed.rows << "x" << processed.cols << std::endl;
        std::cout << "ONNX DEBUG: Scale factors: " << scale_factors.width << ", " << scale_factors.height << std::endl;
        std::cout << "ONNX DEBUG: Padding: " << padding.x << ", " << padding.y << std::endl;

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

        // Convert string vectors to const char* arrays for ONNX Runtime
        std::vector<const char*> input_name_ptrs;
        std::vector<const char*> output_name_ptrs;

        for (const auto& name : input_names_) {
            input_name_ptrs.push_back(name.c_str());
        }
        for (const auto& name : output_names_) {
            output_name_ptrs.push_back(name.c_str());
        }

        auto output_tensors = session_->Run(Ort::RunOptions{nullptr},
                                          input_name_ptrs.data(),
                                          input_tensors.data(),
                                          input_name_ptrs.size(),
                                          output_name_ptrs.data(),
                                          output_name_ptrs.size());
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
                                              scale_factors,
                                              padding,
                                              conf_threshold,
                                              nms_threshold,
                                              keypoint_threshold);
        } else {
            result.detections = postProcessDetection(output_data,
                                                   output_shape,
                                                   result.input_size,
                                                   result.original_size,
                                                   scale_factors,
                                                   padding,
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
                                                   cv::Size2f scale_factors,
                                                   cv::Point2f padding,
                                                   float conf_threshold,
                                                   float nms_threshold,
                                                   float keypoint_threshold) {
    std::vector<Detection> detections;

    if (output_shape.size() != 3) {
        std::cerr << "Unexpected output shape dimensions: " << output_shape.size() << std::endl;
        return detections;
    }

    std::cout << "ONNX DEBUG: Output shape: [" << output_shape[0] << ", "
              << output_shape[1] << ", " << output_shape[2] << "]" << std::endl;
    std::cout << "ONNX DEBUG: Confidence threshold: " << conf_threshold << std::endl;

    int64_t batch_size = output_shape[0];
    int64_t dim1 = output_shape[1];
    int64_t dim2 = output_shape[2];

    // Standard YOLO output is typically [1, num_anchors, features] or [1, features, num_anchors]
    // For 8 keypoints: features = 4(bbox) + 1(conf) + 8*3(keypoints) = 29
    // So we expect either [1, 8400, 29] or [1, 29, 8400]

    int64_t num_anchors, features;
    bool transposed = false;

    if (dim1 == 29 && dim2 > 1000) {
        // Format: [1, 29, 8400]
        features = dim1;
        num_anchors = dim2;
        transposed = true;
        std::cout << "ONNX DEBUG: Detected transposed format [1, 29, " << num_anchors << "]" << std::endl;
    } else if (dim2 == 29 && dim1 > 1000) {
        // Format: [1, 8400, 29]
        num_anchors = dim1;
        features = dim2;
        transposed = false;
        std::cout << "ONNX DEBUG: Detected standard format [1, " << num_anchors << ", 29]" << std::endl;
    } else {
        std::cerr << "ONNX DEBUG: Unexpected dimensions - dim1: " << dim1 << ", dim2: " << dim2 << std::endl;
        // Try to guess based on which dimension is larger
        if (dim1 > dim2) {
            num_anchors = dim1;
            features = dim2;
            transposed = false;
        } else {
            features = dim1;
            num_anchors = dim2;
            transposed = true;
        }
    }

    int num_keypoints = (features - 5) / 3;
    if (num_keypoints != 8) {
        std::cout << "ONNX DEBUG: Warning - expected 8 keypoints, got " << num_keypoints << std::endl;
    }

    // Calculate proper coordinate transformation parameters
    // The preprocessing used uniform scaling and padding to maintain aspect ratio
    float scale = scale_factors.width; // Both width and height should be the same
    float pad_x = padding.x;
    float pad_y = padding.y;

    std::cout << "ONNX DEBUG: Using scale factor: " << scale << std::endl;
    std::cout << "ONNX DEBUG: Using padding: (" << pad_x << ", " << pad_y << ")" << std::endl;

    std::vector<cv::Rect2f> boxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point3f>> keypoints_list;

    int valid_detections = 0;
    int high_conf_detections = 0;

    for (int64_t i = 0; i < num_anchors; ++i) {
        float cx, cy, w, h, conf;

        // Extract basic detection data based on tensor layout
        if (transposed) {
            // [features, anchors] layout
            cx = output[0 * num_anchors + i];
            cy = output[1 * num_anchors + i];
            w = output[2 * num_anchors + i];
            h = output[3 * num_anchors + i];
            conf = output[4 * num_anchors + i];
        } else {
            // [anchors, features] layout
            const float* anchor_data = output + i * features;
            cx = anchor_data[0];
            cy = anchor_data[1];
            w = anchor_data[2];
            h = anchor_data[3];
            conf = anchor_data[4];
        }

        // Debug first few detections
        if (i < 5) {
            std::cout << "ONNX DEBUG: Anchor " << i << " - conf: " << conf
                      << ", bbox: [" << cx << ", " << cy << ", " << w << ", " << h << "]" << std::endl;
        }

        if (conf > 0.01) high_conf_detections++;  // Count any reasonable confidence

        if (conf >= conf_threshold) {
            valid_detections++;

            // Convert from preprocessed image coordinates to original image coordinates
            // 1. Remove padding
            float cx_no_pad = cx - pad_x;
            float cy_no_pad = cy - pad_y;
            float w_no_pad = w;  // Width/height scaling is the same as coordinate scaling
            float h_no_pad = h;

            // 2. Apply inverse scale
            float cx_orig = cx_no_pad / scale;
            float cy_orig = cy_no_pad / scale;
            float w_orig = w_no_pad / scale;
            float h_orig = h_no_pad / scale;

            // Convert to corner format
            float x1 = cx_orig - w_orig/2;
            float y1 = cy_orig - h_orig/2;
            float x2 = cx_orig + w_orig/2;
            float y2 = cy_orig + h_orig/2;

            // Sanity check bounds
            if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 &&
                x1 < original_size.width && y1 < original_size.height) {

                boxes.push_back(cv::Rect2f(x1, y1, x2-x1, y2-y1));
                confidences.push_back(conf);

                // Extract keypoints (8 keypoints for gate)
                std::vector<cv::Point3f> kpts;
                for (int k = 0; k < 8 && k < num_keypoints; ++k) {
                    float kx, ky, kconf;

                    if (transposed) {
                        kx = output[(5 + k*3) * num_anchors + i];
                        ky = output[(5 + k*3 + 1) * num_anchors + i];
                        kconf = output[(5 + k*3 + 2) * num_anchors + i];
                    } else {
                        const float* anchor_data = output + i * features;
                        kx = anchor_data[5 + k*3];
                        ky = anchor_data[5 + k*3 + 1];
                        kconf = anchor_data[5 + k*3 + 2];
                    }

                    // Apply the same coordinate transformation to keypoints
                    float kx_no_pad = kx - pad_x;
                    float ky_no_pad = ky - pad_y;
                    float kx_orig = kx_no_pad / scale;
                    float ky_orig = ky_no_pad / scale;

                    kpts.push_back(cv::Point3f(kx_orig, ky_orig, kconf));
                }
                keypoints_list.push_back(kpts);
            }
        }
    }

    std::cout << "ONNX DEBUG: Processed " << num_anchors << " anchors" << std::endl;
    std::cout << "ONNX DEBUG: Found " << high_conf_detections << " detections with conf > 0.01" << std::endl;
    std::cout << "ONNX DEBUG: Found " << valid_detections << " detections above threshold " << conf_threshold << std::endl;
    std::cout << "ONNX DEBUG: Valid boxes after bounds check: " << boxes.size() << std::endl;

    if (boxes.empty()) {
        std::cout << "ONNX DEBUG: No valid detections found" << std::endl;
        return detections;
    }

    // Convert cv::Rect2f to cv::Rect for NMS
    std::vector<cv::Rect> int_boxes;
    for (const auto& box : boxes) {
        int_boxes.push_back(cv::Rect(static_cast<int>(box.x), static_cast<int>(box.y),
                                    static_cast<int>(box.width), static_cast<int>(box.height)));
    }

    // Apply NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(int_boxes, confidences, conf_threshold, nms_threshold, indices);

    std::cout << "ONNX DEBUG: " << indices.size() << " detections after NMS" << std::endl;

    for (int idx : indices) {
        Detection det;
        det.bbox = boxes[idx];  // Use the original float bbox
        det.confidence = confidences[idx];
        det.class_id = 0; // Gate class
        det.keypoints = keypoints_list[idx];

        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> ONNXBackend::postProcessDetection(const float* output,
                                                        const std::vector<int64_t>& output_shape,
                                                        cv::Size input_size,
                                                        cv::Size original_size,
                                                        cv::Size2f scale_factors,
                                                        cv::Point2f padding,
                                                        float conf_threshold,
                                                        float nms_threshold) {
    // Standard YOLO detection post-processing with corrected coordinate transformation
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