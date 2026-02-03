# YOLO Inference C++ for ROS2

High-performance **YOLO inference implementation in C++** for ROS2, optimized for robotics applications and real-time computer vision. This package provides a production-ready solution for deploying [Ultralytics](https://ultralytics.com/) YOLO models on edge devices and robotic systems.

## Compatibility

✅ **Fully compatible with [Ultralytics YOLO](https://docs.ultralytics.com/) models**, including:
- **YOLOv8** (all variants: n/s/m/l/x for detect, pose, and segment tasks)
- **YOLOv11** (all variants: n/s/m/l/x for detect, pose, and segment tasks)
- Any future Ultralytics releases following the same export format

Models trained with the Ultralytics framework can be directly exported to ONNX or TensorRT and deployed using this package with no modifications required.

## Features

- **🚀 High Performance**: Optimized for real-time inference on edge devices
- **🔧 Dual Backend Support**: TensorRT (best performance) and ONNX Runtime (cross-platform)
- **📊 Comprehensive Profiling**: Detailed timing analysis for all processing stages
- **🎯 Multiple Tasks**: Pose detection, object detection, and segmentation support
- **🔌 ROS2 Integration**: Native ROS2 Humble support with custom messages
- **⚡ Jetson Optimized**: Special optimizations for NVIDIA Jetson platforms

## Supported Platforms

- **NVIDIA Jetson** (Xavier NX, Orin, AGX): Primary target for robotics applications
- **x86_64 with NVIDIA GPU**: Development and testing
- **ARM64**: General ARM64 support (Not yet validated)

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

### 2. Usage Examples

> **Note**: This package is designed to work seamlessly with models trained using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework. If you have already a trained model, check the preparation section below to export it to ONNX and/or TensorRT.

**Basic Pose Detection**

```bash
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=models/yolo11n-pose.onnx \
    confidence_threshold:=0.1 \
    keypoint_threshold:=0.3 \
    publish_visualization:=true \
    input_topic:=/camera/image_raw/compressed \
    input_size:=640 \
    task:=pose
```

**High-Performance Edge Configuration**

```bash
ros2 launch yolo_inference_cpp yolo_tensorrt.launch.py \
    model_path:=models/yolo11m-pose-jetson-fp16.engine \
    confidence_threshold:=0.1 \
    keypoint_threshold:=0.3 \
    max_detections:=7 \
    publish_visualization:=false \
    enable_profiling:=true \
    input_size:=640 \
    task:=pose
```

**Multi-Model Detection**

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

# Pose
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=yolo11n-pose.onnx \
    task:=pose \
    confidence_threshold:=0.5
```

### 3. [optional] Model Training & Preparation

> **Note**: This package is designed to work seamlessly with models trained using the [Ultralytics](https://github.com/ultralytics/ultralytics) framework. Visit their [documentation](https://docs.ultralytics.com/) for detailed training guides and best practices.

**Model Training**

Train your Yolo model. The following parameters have been found the optimal. However, feel free to experiment. You can choose your yolo variant with `--model` e.g. `yolo11x-pose.pt, yolo11n-pose.pt, yolov8m-pose.pt, etc.`.

```
python scripts/yolo_training.py
--data
<path_to_train_val_dataset_yolo>/dataset.yaml
--project
<path_to_experiments_folder>
--model
yolo11x-pose.pt
--pretrained
--batch-size
4
--imgsz
640
--multiscale
--val-scales
320
640
832
--val-every
3
```

Use tensorboard to visualize training performance. To log common metrics, execute this command:
```
yolo settings tensorboard=True
```

#### Training Script Parameters

| Parameter | Type | Default | Possible Values | Description |
|-----------|------|---------|-----------------|-------------|
| `--data` | string | *required* | Path to YAML file | Path to dataset YAML file |
| `--model` | string | `yolo11n.pt` | `yolo11n.pt`, `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`, `yolo11x.pt` | Model to use for training |
| `--epochs` | int | `100` | Any positive integer | Number of epochs to train |
| `--batch-size` | int | `16` | Any positive integer | Batch size for training |
| `--imgsz` | int | `320` | Any positive integer | Image size for training |
| `--lr0` | float | `0.01` | Any positive float | Initial learning rate |
| `--lrf` | float | `0.01` | Any positive float | Final learning rate (as fraction of lr0) |
| `--patience` | int | `100` | Any positive integer | Epochs to wait for no improvement (early stopping) |
| `--optimizer` | string | `auto` | `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, `auto` | Optimizer to use |
| `--seed` | int | `0` | Any integer | Random seed for reproducibility |
| `--momentum` | float | `0.937` | 0.0 - 1.0 | Optimizer momentum |
| `--weight-decay` | float | `0.0005` | Any positive float | Weight decay (L2 regularization) |
| `--project` | string | `runs/train` | Any path | Project directory for outputs |
| `--name` | string | `exp` | Any string | Experiment name |
| `--save-period` | int | `1` | Any positive integer | Save checkpoint every N epochs |
| `--val-scales` | int[] | `None` | List of integers | Multiple validation sizes (e.g., `320 640 832`) |
| `--val-every` | int | `1` | Any positive integer | Run multi-scale validation every N epochs |
| `--val-split` | string | `val` | `val`, `test`, `train` | Dataset split for multi-scale validation |
| `--val-metric` | string | `map50` | `map`, `map50` | Metric to track for best models at each scale |
| `--override-best` | flag | `False` | `True`/`False` | Overwrite Ultralytics best.pt when alt metric improves |
| `--tb-samples` | int | `3` | Any positive integer | Number of sample images for TensorBoard |
| `--tb-sample-images` | string[] | `None` | List of image paths | Specific images for TensorBoard visualization |
| `--multiscale` | flag | `False` | `True`/`False` | Enable multi-scale training (0.5x-1.5x range) |
| `--multiscale-range` | float | `0.5` | Any float | **DEPRECATED**: YOLO uses fixed 0.5x-1.5x range |
| `--multiscale-min` | int | `None` | Any integer | **DEPRECATED**: YOLO uses fixed 0.5x-1.5x range |
| `--multiscale-max` | int | `None` | Any integer | **DEPRECATED**: YOLO uses fixed 0.5x-1.5x range |
| `--device` | string | `""` (auto) | `cuda:0`, `cpu`, etc. | Device to use (empty for auto-detection) |
| `--workers` | int | `8` | Any positive integer | Number of dataloader workers |
| `--hsv-h` | float | `0.015` | 0.0 - 1.0 | HSV-Hue augmentation (fraction) |
| `--hsv-s` | float | `0.7` | 0.0 - 1.0 | HSV-Saturation augmentation (fraction) |
| `--hsv-v` | float | `0.4` | 0.0 - 1.0 | HSV-Value augmentation (fraction) |
| `--degrees` | float | `40` | 0.0 - 360.0 | Rotation augmentation (degrees) |
| `--translate` | float | `0.1` | 0.0 - 1.0 | Translation augmentation (fraction) |
| `--scale` | float | `0.9` | 0.0 - 1.0+ | Scale augmentation (fraction) |
| `--shear` | float | `15` | Any float | Shear augmentation (degrees) |
| `--perspective` | float | `0.0004` | 0.0 - 1.0 | Perspective augmentation (fraction) |
| `--flipud` | float | `0.5` | 0.0 - 1.0 | Vertical flip augmentation (probability) |
| `--fliplr` | float | `0.5` | 0.0 - 1.0 | Horizontal flip augmentation (probability) |
| `--mosaic` | float | `0.9` | 0.0 - 1.0 | Mosaic augmentation (probability) |
| `--mixup` | float | `0.5` | 0.0 - 1.0 | MixUp augmentation (probability) |
| `--copy-paste` | float | `0.0` | 0.0 - 1.0 | Copy-paste augmentation (probability) |
| `--resume` | string | `""` | Path to checkpoint | Resume training from checkpoint |
| `--pretrained` | flag | `False` | `True`/`False` | Use pretrained weights |
| `--freeze` | int | `0` | Any non-negative integer | Number of layers to freeze (0 = no freezing) |
| `--cache` | flag | `False` | `True`/`False` | Cache images for faster training |
| `--amp` | flag | `False` | `True`/`False` | Use Automatic Mixed Precision training |
| `--plots` | flag | `True` | `True`/`False` | Generate training plots |

**Model Distillation**

A common and useful technique is to distill a big model (teacher) into a smaller and faster model (student). Before, preapre your environment:

```
pip uninstall ultralytics
git clone https://github.com/alejodosr/yolo-distiller
```


Here a possible usage of the script:

```
PYTHONPATH=$PYTHONPATH:<path_to_yolo-distiller> python scripts/yolo_distillation.py
--teacher
<path_to_trained_teacher>/yolo11_x_img640.pt
--student
yolo11n-pose.pt
--data
<path_to_train_val_dataset_yolo>/dataset.yaml
--project
<path_to_experiments_folder>
--batch-size
4
--imgsz
640
--multiscale
--val-scales
320
640
832
--val-every
3
```

Use tensorboard to visualize training performance.

**Batch Export to ONNX and TensorRT**

For maximum performance on Jetson:

1. **Use FP16 precision**: Reduces memory usage and improves speed
2. **Optimize workspace size**: Balance between speed and memory
3. **Build platform-specific engines**: Jetson engines won't work on desktop

Convert your YOLO model to the appropriate format:

```
python scripts/yolo_batch_exporter_validator.py --model-folders
<folder_1> <folder_2> ...
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

**Benchmark all models FPS vs Accuracy Plot [optional]**
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

After execution, feel free to open the generated `.html` files to visualize interactively every model performance.

### 4. Test with Sample Data

```bash
# Run the test script
python3 tests/test_inference.py

# Benchmark performance
./install/yolo_inference_cpp/lib/yolo_inference_cpp/benchmark yolo11n-pose.onnx pose 640 100
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

**1. CUDA Out of Memory**
```bash
# Error: CUDA out of memory during inference
# Solutions:
# - Reduce workspace size: --workspace-size 1
# - Use smaller input size: --input-size 416
# - Reduce max detections: max_detections:=5
```

**2. Low FPS Performance**
```bash
# Check GPU utilization
nvidia-smi

# Disable visualization for maximum speed
publish_visualization:=false

# Use TensorRT instead of ONNX
```

**3. Missing Dependencies**
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

## Keywords

`YOLO` `YOLOv8` `YOLOv11` `Ultralytics` `ROS2` `C++` `TensorRT` `ONNX` `pose estimation` `object detection` `instance segmentation` `edge AI` `Jetson` `drone` `embedded` `UAV` `autonomous systems` `real-time inference` `computer vision` `robotics`

## Authors
Alejandro Rodríguez-Ramos [alejandro.dosr@gmail.com]
