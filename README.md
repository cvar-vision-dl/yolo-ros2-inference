# YOLO Inference C++ for ROS2

High-performance YOLO inference implementation in C++ for ROS2, optimized for drone applications and real-time processing. Supports both TensorRT and ONNX Runtime backends with comprehensive profiling and memory management.

## Features

- **🚀 High Performance**: Optimized for real-time inference on drones and edge devices
- **🔧 Dual Backend Support**: TensorRT (best performance) and ONNX Runtime (cross-platform)
- **📊 Comprehensive Profiling**: Detailed timing analysis for all processing stages
- **🎯 Multiple Tasks**: Pose detection, object detection, and segmentation support
- **🔌 ROS2 Integration**: Native ROS2 Humble support with custom messages
- **🐳 Container Ready**: Docker support for both Jetson and x86_64 platforms [not yet validated]
- **⚡ Jetson Optimized**: Special optimizations for NVIDIA Jetson platforms

## Supported Platforms

- **NVIDIA Jetson** (Xavier NX, Orin, AGX): Primary target for drone applications
- **x86_64 with NVIDIA GPU**: Development and testing
- **ARM64**: General ARM64 support (limited testing)

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd yolo_inference_cpp

# Install dependencies (choose your platform)
chmod +x scripts/install_dependencies.sh
./scripts/install_dependencies.sh

# Build the package
chmod +x scripts/build_package.sh
./scripts/build_package.sh
```

### 2. Model Preparation

Convert your YOLO model to the appropriate format:

```bash
# Convert PyTorch model to ONNX and TensorRT
python3 scripts/convert_model.py yolo11n-pose.pt --format both --precision fp16

# For Jetson-specific optimization
python3 scripts/convert_model.py yolo11n-pose.pt --format tensorrt --precision fp16 --workspace-size 2
```

### 3. Launch the Node

```bash
# Source the workspace
source ~/ros2_ws/install/setup.bash

# ONNX Runtime (cross-platform)
ros2 launch yolo_inference_cpp yolo_pose.launch.py model_path:=yolo11n-pose.onnx

# TensorRT (high performance)
ros2 launch yolo_inference_cpp yolo_tensorrt.launch.py model_path:=yolo11n-pose-fp16.engine
```

### 4. Test with Sample Data

```bash
# Run the test script
python3 tests/test_inference.py

# Benchmark performance
./install/yolo_inference_cpp/lib/yolo_inference_cpp/benchmark yolo11n-pose.onnx pose 640 100
```

## Usage Examples

### Basic Pose Detection

```bash
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=models/yolo11n-pose.onnx \
    confidence_threshold:=0.5 \
    keypoint_threshold:=0.3 \
    publish_visualization:=true \
    input_topic:=/camera/image_raw/compressed
```

### High-Performance Drone Configuration

```bash
ros2 launch yolo_inference_cpp yolo_tensorrt.launch.py \
    model_path:=models/yolo11m-pose-jetson-fp16.engine \
    confidence_threshold:=0.7 \
    max_detections:=5 \
    publish_visualization:=false \
    enable_profiling:=true
```

### Multi-Model Detection

```bash
# Object detection
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=yolo11n.onnx \
    task:=detect \
    confidence_threshold:=0.6

# Segmentation
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=yolo11n-seg.onnx \
    task:=segment \
    confidence_threshold:=0.5
```

### Batch Export to ONNX and TensorRT
```
python scripts/yolo_batch_exporter_validator.py --model-folders
<folder_1> <folder_2> ...
--dataset-yaml
<path_to_validation_dataset_yolo>/dataset.yaml
--output-dir
<output_dir>
--task
pose
--trtexec-path
/usr/src/tensorrt/bin/trtexec
--tensorrt-precision
all
--use-tensorrt
```

### Benchmark all models FPS vs Accuracy Plot
```
python scripts/yolo_benchmarking.py --models-folder
<path_to_all_models_to_compare>
--dataset-yaml
<path_to_validation_dataset_yolo>/dataset.yaml
--output-dir
<output_dir>
--task
pose
--timing-runs
100
--warmup-runs
10
```

## Configuration

### Launch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `yolo11n-pose.onnx` | Path to model file |
| `task` | string | `pose` | Task type: pose, detect, segment |
| `input_size` | int | `640` | Input image size (square) |
| `confidence_threshold` | float | `0.5` | Detection confidence threshold |
| `nms_threshold` | float | `0.4` | Non-maximum suppression threshold |
| `keypoint_threshold` | float | `0.3` | Keypoint visibility threshold |
| `max_detections` | int | `20` | Maximum detections per frame |
| `publish_visualization` | bool | `false` | Enable visualization output |
| `enable_profiling` | bool | `true` | Enable detailed profiling |

### Configuration Files

Use YAML configuration files for complex setups:

```bash
# Development configuration
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    config:=config/development.yaml

# Jetson optimized configuration
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    config:=config/jetson_optimized.yaml
```

## Message Types

### KeypointDetection

```yaml
std_msgs/Header header
string label
int32 class_id
float32 confidence
BoundingBox bounding_box
Keypoint[] keypoints
```

### KeypointDetectionArray

```yaml
std_msgs/Header header
string model_type
string task
KeypointDetection[] detections
float32[] raw_output      # High-performance array format
PerformanceInfo performance
```

### PerformanceInfo

```yaml
float64 total_time_ms
float64 inference_ms
float64 image_conversion_ms
float64 message_creation_ms
int32 detections_count
float64 fps
```

## Docker Deployment

### Jetson Deployment

```bash
# Build Jetson image
docker build -f docker/Dockerfile.jetson -t yolo_inference:jetson .

# Run with models volume
docker run --runtime nvidia --rm -it \
    --network host \
    -v $(pwd)/models:/models:ro \
    yolo_inference:jetson \
    ros2 launch yolo_inference_cpp yolo_tensorrt.launch.py \
    model_path:=/models/yolo11n-pose-jetson-fp16.engine
```

### Desktop Development

```bash
# Build desktop image
docker build -f docker/Dockerfile.x86_64 -t yolo_inference:desktop .

# Run with visualization
docker run --runtime nvidia --rm -it \
    --network host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/models:/models:ro \
    yolo_inference:desktop \
    ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=/models/yolo11n-pose.onnx \
    publish_visualization:=true
```

## Performance Optimization

### TensorRT Optimization

For maximum performance on Jetson:

1. **Use FP16 precision**: Reduces memory usage and improves speed
2. **Optimize workspace size**: Balance between speed and memory
3. **Build platform-specific engines**: Jetson engines won't work on desktop

```bash
# Build Jetson-optimized engine
python3 scripts/convert_model.py yolo11n-pose.pt \
    --format tensorrt \
    --precision fp16 \
    --workspace-size 2  # GB, adjust based on available memory
```

### Profiling and Monitoring

Enable profiling to monitor performance:

```bash
# View performance in real-time
ros2 topic echo /yolo/performance

# Monitor resource usage
nvidia-smi -l 1  # GPU usage
htop             # CPU usage
```

## Troubleshooting

### Common Issues

**1. TensorRT Engine Compatibility**
```bash
# Error: Engine built on different platform
# Solution: Rebuild engine on target platform
python3 scripts/convert_model.py model.pt --format tensorrt --precision fp16
```

**2. CUDA Out of Memory**
```bash
# Error: CUDA out of memory during inference
# Solutions:
# - Reduce workspace size: --workspace-size 1
# - Use smaller input size: --input-size 416
# - Reduce max detections: max_detections:=5
```

**3. Low FPS Performance**
```bash
# Check GPU utilization
nvidia-smi

# Disable visualization for maximum speed
publish_visualization:=false

# Use TensorRT instead of ONNX
# Convert model: python3 scripts/convert_model.py model.pt --format tensorrt
```

**4. Missing Dependencies**
```bash
# Reinstall dependencies
./scripts/install_dependencies.sh

# Check ONNX Runtime
ls -la /usr/local/onnxruntime/lib

# Check TensorRT
ldconfig -p | grep tensorrt
```

### Debug Mode

Enable debug logging:

```bash
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    --log-level debug
```

### Performance Benchmarking

Compare different configurations:

```bash
# Benchmark ONNX vs TensorRT
./benchmark models/yolo11n-pose.onnx pose 640 100
./benchmark models/yolo11n-pose-fp16.engine pose 640 100

# Test different input sizes
for size in 416 512 640 800; do
    ./benchmark model.onnx pose $size 50
done
```

## Development

### Building from Source

```bash
# Clone repository
git clone <repo-url> ~/ros2_ws/src/yolo_inference_cpp

# Install dependencies
cd ~/ros2_ws/src/yolo_inference_cpp
./scripts/install_dependencies.sh

# Build with specific backends
cd ~/ros2_ws
colcon build --packages-select yolo_inference_cpp \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
                 -DHAVE_TENSORRT=ON \
                 -DHAVE_ONNXRUNTIME=ON
```

### Testing

```bash
# Unit tests
colcon test --packages-select yolo_inference_cpp

# Integration tests
python3 tests/test_inference.py

# Performance tests
./tests/benchmark models/test_model.onnx
```

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Discussions**: Ask questions and share experiences in GitHub Discussions
- **Documentation**: Additional documentation available in the `docs/` directory

## Authors
Alejandro Rodríguez-Ramos [alejandro.dosr@gmail.com]
