// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#ifdef HAVE_TENSORRT

#include <fstream>
#include <iostream>
#include <algorithm>

#include "yolo_inference_cpp/tensorrt_backend.hpp"
#include "yolo_inference_cpp/preprocessing.hpp"

namespace yolo_inference
{

void TensorRTLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    std::cout << "[TensorRT] " << msg << std::endl;
  }
}

TensorRTBackend::TensorRTBackend()
: task_type_(TaskType::POSE)
  , input_size_(640)
  , initialized_(false)
  , stream_(nullptr)
  , input_device_buffer_(nullptr)
  , output_device_buffer_(nullptr)
  , input_host_buffer_(nullptr)
  , output_host_buffer_(nullptr)
  , input_size_bytes_(0)
  , output_size_bytes_(0)
{
  logger_ = std::make_unique<TensorRTLogger>();
}

TensorRTBackend::~TensorRTBackend()
{
  deallocateBuffers();
  if (stream_) {
    cudaStreamDestroy(stream_);
  }
}

bool TensorRTBackend::initialize(
  const std::string & model_path,
  TaskType task,
  int input_size)
{
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

  // Initialize class names (same as ONNX backend)
  class_names_ = {
    "gate"
  };

  // Initialize keypoint names (same as ONNX backend)
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
  std::cout << "TensorRT backend initialized successfully" << std::endl;

  // Print model info
  std::cout << "Input tensor name: " << input_tensor_name_ << std::endl;
  std::cout << "Output tensor name: " << output_tensor_name_ << std::endl;
  std::cout << "Input shape: [";
  for (int i = 0; i < input_dims_.nbDims; ++i) {
    std::cout << input_dims_.d[i];
    if (i < input_dims_.nbDims - 1) {std::cout << ", ";}
  }
  std::cout << "]" << std::endl;
  std::cout << "Output shape: [";
  for (int i = 0; i < output_dims_.nbDims; ++i) {
    std::cout << output_dims_.d[i];
    if (i < output_dims_.nbDims - 1) {std::cout << ", ";}
  }
  std::cout << "]" << std::endl;

  return true;
}

bool TensorRTBackend::loadEngine(const std::string & engine_path)
{
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

bool TensorRTBackend::setupBindings()
{
  // Use modern TensorRT API (TRT 10.x)
  int32_t num_io_tensors = engine_->getNbIOTensors();
  std::cout << "Number of I/O tensors: " << num_io_tensors << std::endl;

  for (int32_t i = 0; i < num_io_tensors; ++i) {
    const char * tensor_name = engine_->getIOTensorName(i);
    nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);

    if (io_mode == nvinfer1::TensorIOMode::kINPUT) {
      input_tensor_name_ = std::string(tensor_name);
      input_dims_ = engine_->getTensorShape(tensor_name);
      input_size_bytes_ = 1;
      for (int j = 0; j < input_dims_.nbDims; ++j) {
        input_size_bytes_ *= input_dims_.d[j];
      }
      input_size_bytes_ *= sizeof(float);
      std::cout << "Found input tensor: " << input_tensor_name_ << std::endl;
    } else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_tensor_name_ = std::string(tensor_name);
      output_dims_ = engine_->getTensorShape(tensor_name);
      output_size_bytes_ = 1;
      for (int j = 0; j < output_dims_.nbDims; ++j) {
        output_size_bytes_ *= output_dims_.d[j];
      }
      output_size_bytes_ *= sizeof(float);
      std::cout << "Found output tensor: " << output_tensor_name_ << std::endl;
    }
  }

  return !input_tensor_name_.empty() && !output_tensor_name_.empty();
}

void TensorRTBackend::allocateBuffers()
{
  // Allocate device memory
  cudaMalloc(&input_device_buffer_, input_size_bytes_);
  cudaMalloc(&output_device_buffer_, output_size_bytes_);

  // Allocate host memory
  cudaMallocHost(&input_host_buffer_, input_size_bytes_);
  cudaMallocHost(&output_host_buffer_, output_size_bytes_);
}

void TensorRTBackend::deallocateBuffers()
{
  if (input_device_buffer_) {
    cudaFree(input_device_buffer_);
    input_device_buffer_ = nullptr;
  }
  if (output_device_buffer_) {
    cudaFree(output_device_buffer_);
    output_device_buffer_ = nullptr;
  }
  if (input_host_buffer_) {
    cudaFreeHost(input_host_buffer_);
    input_host_buffer_ = nullptr;
  }
  if (output_host_buffer_) {
    cudaFreeHost(output_host_buffer_);
    output_host_buffer_ = nullptr;
  }
}

InferenceResult TensorRTBackend::infer(
  const cv::Mat & image,
  float conf_threshold,
  float nms_threshold,
  float keypoint_threshold)
{
  // std::cout << "TensorRT DEBUG: Starting inference..." << std::endl;
  InferenceResult result;
  result.original_size = image.size();
  result.input_size = cv::Size(input_size_, input_size_);

  auto start_time = std::chrono::high_resolution_clock::now();

  // std::cout << "TensorRT DEBUG: Starting preprocessing..." << std::endl;
  // Preprocess image
  Preprocessor preprocessor;
  cv::Mat processed = preprocessor.preprocess(image, input_size_);

  // Store preprocessing info for coordinate transformation
  cv::Size2f scale_factors = preprocessor.getScaleFactors();
  cv::Point2f padding = preprocessor.getPadding();

//    std::cout << "TensorRT DEBUG: Preprocessing completed, processed size: "
//              << processed.rows << "x" << processed.cols << std::endl;
//    std::cout << "TensorRT DEBUG: Scale factors: " << scale_factors.width << ", " << scale_factors.height << std::endl;
//    std::cout << "TensorRT DEBUG: Padding: " << padding.x << ", " << padding.y << std::endl;

  // Copy data to host buffer
  float * input_host = static_cast<float *>(input_host_buffer_);
  memcpy(input_host, processed.ptr<float>(), input_size_bytes_);

  // Copy from host to device
  cudaMemcpyAsync(
    input_device_buffer_,
    input_host_buffer_,
    input_size_bytes_,
    cudaMemcpyHostToDevice,
    stream_);

  // Set tensor addresses using modern API
  context_->setTensorAddress(input_tensor_name_.c_str(), input_device_buffer_);
  context_->setTensorAddress(output_tensor_name_.c_str(), output_device_buffer_);

  // Create bindings array for executeV2 (still needed even with modern API)
  std::vector<void *> bindings = {input_device_buffer_, output_device_buffer_};

  // Run inference using modern API
//    std::cout << "TensorRT DEBUG: About to execute inference..." << std::endl;
  bool status = context_->executeV2(bindings.data());
  if (!status) {
    std::cerr << "TensorRT inference execution failed" << std::endl;
    return result;
  }

  // Copy output from device to host
  cudaMemcpyAsync(
    output_host_buffer_,
    output_device_buffer_,
    output_size_bytes_,
    cudaMemcpyDeviceToHost,
    stream_);

  // Synchronize
  cudaStreamSynchronize(stream_);

  auto end_time = std::chrono::high_resolution_clock::now();
  result.inference_time_ms = std::chrono::duration<double, std::milli>(
    end_time - start_time).count();

//    std::cout << "TensorRT DEBUG: Inference completed!" << std::endl;

  // Post-process results
  float * output_host = static_cast<float *>(output_host_buffer_);

  if (task_type_ == TaskType::POSE) {
    result.detections = postProcessPose(
      output_host,
      result.input_size,
      result.original_size,
      scale_factors,
      padding,
      conf_threshold,
      nms_threshold,
      keypoint_threshold);
  } else {
    result.detections = postProcessDetection(
      output_host,
      result.input_size,
      result.original_size,
      scale_factors,
      padding,
      conf_threshold,
      nms_threshold);
  }

  return result;
}

std::vector<Detection> TensorRTBackend::postProcessPose(
  float * output,
  cv::Size input_size,
  cv::Size original_size,
  cv::Size2f scale_factors,
  cv::Point2f padding,
  float conf_threshold,
  float nms_threshold,
  float keypoint_threshold)
{
  std::vector<Detection> detections;

//    std::cout << "TensorRT DEBUG: Output dims: " << output_dims_.nbDims << " [";
  for (int i = 0; i < output_dims_.nbDims; ++i) {
    std::cout << output_dims_.d[i];
    if (i < output_dims_.nbDims - 1) {std::cout << ", ";}
  }
  std::cout << "]" << std::endl;
//    std::cout << "TensorRT DEBUG: Confidence threshold: " << conf_threshold << std::endl;

  if (output_dims_.nbDims != 3) {
    std::cerr << "Unexpected output shape dimensions: " << output_dims_.nbDims << std::endl;
    return detections;
  }

  int64_t batch_size = output_dims_.d[0];
  int64_t dim1 = output_dims_.d[1];
  int64_t dim2 = output_dims_.d[2];

  // Determine tensor layout (same logic as ONNX backend)
  int64_t num_anchors, features;
  bool transposed = false;

  if (dim1 == 29 && dim2 > 1000) {
    // Format: [1, 29, 8400]
    features = dim1;
    num_anchors = dim2;
    transposed = true;
//        std::cout << "TensorRT DEBUG: Detected transposed format [1, 29, " << num_anchors << "]" << std::endl;
  } else if (dim2 == 29 && dim1 > 1000) {
    // Format: [1, 8400, 29]
    num_anchors = dim1;
    features = dim2;
    transposed = false;
//        std::cout << "TensorRT DEBUG: Detected standard format [1, " << num_anchors << ", 29]" << std::endl;
  } else {
//        std::cerr << "TensorRT DEBUG: Unexpected dimensions - dim1: " << dim1 << ", dim2: " << dim2 << std::endl;
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
//        std::cout << "TensorRT DEBUG: Warning - expected 8 keypoints, got " << num_keypoints << std::endl;
  }

  // Calculate proper coordinate transformation parameters
  float scale = scale_factors.width;   // Both width and height should be the same
  float pad_x = padding.x;
  float pad_y = padding.y;

//    std::cout << "TensorRT DEBUG: Using scale factor: " << scale << std::endl;
//    std::cout << "TensorRT DEBUG: Using padding: (" << pad_x << ", " << pad_y << ")" << std::endl;

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
      const float * anchor_data = output + i * features;
      cx = anchor_data[0];
      cy = anchor_data[1];
      w = anchor_data[2];
      h = anchor_data[3];
      conf = anchor_data[4];
    }

    // Debug first few detections
//        if (i < 5) {
//            std::cout << "TensorRT DEBUG: Anchor " << i << " - conf: " << conf
//                      << ", bbox: [" << cx << ", " << cy << ", " << w << ", " << h << "]" << std::endl;
//        }

    if (conf > 0.01) {high_conf_detections++;}

    if (conf >= conf_threshold) {
      valid_detections++;

      // Convert from preprocessed image coordinates to original image coordinates
      // Same logic as ONNX backend
      float cx_no_pad = cx - pad_x;
      float cy_no_pad = cy - pad_y;
      float w_no_pad = w;
      float h_no_pad = h;

      float cx_orig = cx_no_pad / scale;
      float cy_orig = cy_no_pad / scale;
      float w_orig = w_no_pad / scale;
      float h_orig = h_no_pad / scale;

      // Convert to corner format
      float x1 = cx_orig - w_orig / 2;
      float y1 = cy_orig - h_orig / 2;
      float x2 = cx_orig + w_orig / 2;
      float y2 = cy_orig + h_orig / 2;

      // Sanity check bounds
      if (x1 >= 0 && y1 >= 0 && x2 > x1 && y2 > y1 &&
        x1 < original_size.width && y1 < original_size.height)
      {

        boxes.push_back(cv::Rect2f(x1, y1, x2 - x1, y2 - y1));
        confidences.push_back(conf);

        // Extract keypoints (8 keypoints for gate)
        std::vector<cv::Point3f> kpts;
        for (int k = 0; k < 8 && k < num_keypoints; ++k) {
          float kx, ky, kconf;

          if (transposed) {
            kx = output[(5 + k * 3) * num_anchors + i];
            ky = output[(5 + k * 3 + 1) * num_anchors + i];
            kconf = output[(5 + k * 3 + 2) * num_anchors + i];
          } else {
            const float * anchor_data = output + i * features;
            kx = anchor_data[5 + k * 3];
            ky = anchor_data[5 + k * 3 + 1];
            kconf = anchor_data[5 + k * 3 + 2];
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

//    std::cout << "TensorRT DEBUG: Processed " << num_anchors << " anchors" << std::endl;
//    std::cout << "TensorRT DEBUG: Found " << high_conf_detections << " detections with conf > 0.01" << std::endl;
//    std::cout << "TensorRT DEBUG: Found " << valid_detections << " detections above threshold " << conf_threshold << std::endl;
//    std::cout << "TensorRT DEBUG: Valid boxes after bounds check: " << boxes.size() << std::endl;

  if (boxes.empty()) {
//        std::cout << "TensorRT DEBUG: No valid detections found" << std::endl;
    return detections;
  }

  // Convert cv::Rect2f to cv::Rect for NMS
  std::vector<cv::Rect> int_boxes;
  for (const auto & box : boxes) {
    int_boxes.push_back(
      cv::Rect(
        static_cast<int>(box.x), static_cast<int>(box.y),
        static_cast<int>(box.width), static_cast<int>(box.height)));
  }

  // Apply NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(int_boxes, confidences, conf_threshold, nms_threshold, indices);

//    std::cout << "TensorRT DEBUG: " << indices.size() << " detections after NMS" << std::endl;

  for (int idx : indices) {
    Detection det;
    det.bbox = boxes[idx];
    det.confidence = confidences[idx];
    det.class_id = 0;     // Gate class
    det.keypoints = keypoints_list[idx];

    detections.push_back(det);
  }

  return detections;
}

std::vector<Detection> TensorRTBackend::postProcessDetection(
  float * output,
  cv::Size input_size,
  cv::Size original_size,
  cv::Size2f scale_factors,
  cv::Point2f padding,
  float conf_threshold,
  float nms_threshold)
{
  // Standard YOLO detection post-processing with corrected coordinate transformation
  std::vector<Detection> detections;

  // Implementation would be similar to pose processing but without keypoints
  // For now, return empty for detection models

  return detections;
}

std::vector<std::string> TensorRTBackend::getClassNames() const
{
  return class_names_;
}

std::vector<std::string> TensorRTBackend::getKeypointNames() const
{
  return keypoint_names_;
}

} // namespace yolo_inference

#endif // HAVE_TENSORRT
