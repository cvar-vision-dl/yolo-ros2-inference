# YOLO Inference C++ for ROS2

High-performance **YOLO inference implementation in C++** for ROS2, optimized for robotics applications and real-time computer vision. This package provides a production-ready solution for deploying [Ultralytics](https://ultralytics.com/) YOLO models on edge devices and robotic systems.

![yolo_cpp_workspace_gif](https://github.com/user-attachments/assets/750ed7c3-a30e-49a4-b2ec-7f87ff9d15fd)

## Compatibility

Compatible with [Ultralytics YOLO](https://docs.ultralytics.com/) models, including:
- **YOLOv8** and **YOLOv11** (all variants: n/s/m/l/x — detect, pose, segment tasks)
- Any future Ultralytics releases following the same export format

Models trained with Ultralytics can be directly exported to ONNX or TensorRT and deployed with no modifications required.

## Features

- **High Performance**: Optimized for real-time inference on edge devices
- **Dual Backend**: TensorRT (best performance) and ONNX Runtime (cross-platform)
- **Multiple Tasks**: Object detection, pose estimation, instance segmentation, GateNet gate detection
- **Comprehensive Profiling**: Detailed timing for all processing stages
- **ROS2 Integration**: Native ROS2 Humble support with custom messages
- **Jetson Optimized**: Special optimizations for NVIDIA Jetson platforms

## Supported Platforms

- **NVIDIA Jetson** (Xavier NX, Orin, AGX): Primary target for robotics applications
- **x86_64 with NVIDIA GPU**: Development and testing
- **ARM64**: General ARM64 support (not yet validated)

---

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd yolo-ros2-inference

chmod +x scripts/install_dependencies.sh
./scripts/install_dependencies.sh

chmod +x scripts/build_package.sh
./scripts/build_package.sh
```

### 2. Run Segmentation

```bash
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=<your_model>.onnx \
    task:=segment \
    input_topic:=/your/camera/compressed \
    input_width:=832 \
    input_height:=832 \
    confidence_threshold:=0.25 \
    max_detections:=40 \
    publish_visualization:=true \
    draw_bboxes:=false \
    class_names:="['class_a', 'class_b', 'class_c']"
```

### 3. Run Pose Estimation

```bash
ros2 launch yolo_inference_cpp gatenet.launch.py \
    model_path:=<your_model>.onnx \
    task:=pose \
    input_topic:=/your/camera/compressed \
    input_width:=640 \
    input_height:=480 \
    confidence_threshold:=0.4 \
    publish_visualization:=true
```

> **Tip**: Use `input_width` + `input_height` for non-square inputs. For square inputs, use `input_size` instead (e.g. `input_size:=640`).

---

## More Examples

### Object Detection

```bash
ros2 launch yolo_inference_cpp yolo_pose.launch.py \
    model_path:=yolo11n.onnx \
    task:=detect \
    input_topic:=/your/camera/compressed \
    input_size:=640 \
    confidence_threshold:=0.5
```

### High-Performance with TensorRT (Jetson)

```bash
ros2 launch yolo_inference_cpp yolo_tensorrt.launch.py \
    model_path:=yolo11m-pose-fp16.engine \
    task:=pose \
    input_topic:=/your/camera/compressed \
    input_size:=640 \
    confidence_threshold:=0.3 \
    max_detections:=10 \
    publish_visualization:=false \
    enable_profiling:=true
```

### GateNet Gate Detection

```bash
ros2 launch yolo_inference_cpp gatenet.launch.py \
    model_path:=gatenet.onnx \
    task:=gatenet \
    input_topic:=/your/camera/compressed \
    input_width:=480 \
    input_height:=368
```

---

## Launch Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | string | `yolo11n-pose.onnx` | Path to ONNX or TensorRT engine file |
| `task` | string | `pose` | Task type: `pose`, `detect`, `segment`, `gatenet` |
| `input_topic` | string | `/camera/image_raw/compressed` | Input compressed image topic |
| `input_size` | int | `640` | Input image size (square) |
| `input_width` | int | `-1` | Input width for non-square models |
| `input_height` | int | `-1` | Input height for non-square models |
| `confidence_threshold` | float | `0.5` | Detection confidence threshold |
| `nms_threshold` | float | `0.4` | Non-maximum suppression threshold |
| `keypoint_threshold` | float | `0.3` | Keypoint visibility threshold |
| `max_detections` | int | `20` | Maximum detections per frame |
| `class_names` | string[] | `[]` | Override class names (auto-detected from model if empty) |
| `publish_visualization` | bool | `false` | Enable visualization output |
| `draw_bboxes` | bool | `true` | Draw bounding boxes in visualization |
| `enable_profiling` | bool | `true` | Enable detailed profiling |
| `paf_threshold` | float | `0.3` | PAF affinity threshold *(GateNet only)* |
| `corner_threshold` | float | `0.5` | Corner confidence threshold *(GateNet only)* |
| `min_corners` | int | `3` | Minimum corners to form a valid gate *(GateNet only)* |

### Launch Files

| File | Backend | Typical Use |
|------|---------|-------------|
| `yolo_pose.launch.py` | ONNX Runtime | Cross-platform, any task |
| `yolo_tensorrt.launch.py` | TensorRT | Maximum performance on GPU/Jetson |
| `gatenet.launch.py` | ONNX Runtime | Gate detection with non-square inputs |

---

## Monitoring

```bash
# View real-time performance metrics
ros2 topic echo /yolo/performance

# GPU and CPU usage
nvidia-smi -l 1
htop
```

---

## Message Types

### KeypointDetectionArray

Main output topic published on `/yolo/detections`:

```yaml
std_msgs/Header header
string model_type
string task
KeypointDetection[] detections
float32[] raw_output
PerformanceInfo performance
```

### KeypointDetection

```yaml
std_msgs/Header header
string label
int32 class_id
float32 confidence
BoundingBox bounding_box
Keypoint[] keypoints
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

---

## Model Training & Preparation

> Full documentation including training parameters, distillation, export, and benchmarking is available in **[docs/training.md](docs/training.md)**.

Quick reference:

**Train**
```bash
python scripts/yolo_training.py \
    --data dataset.yaml \
    --model yolo11x-pose.pt \
    --pretrained --batch-size 4 --imgsz 640 --multiscale
```

**Export**
```bash
python scripts/yolo_batch_exporter_validator.py \
    --model-folders <folder> --output-dir <output_dir> \
    --task pose --use-tensorrt
```

**Benchmark**
```bash
python scripts/yolo_benchmarking.py \
    --models-folder <folder> --dataset-yaml dataset.yaml \
    --output-dir <output_dir> --task pose
```

---

## Troubleshooting

**CUDA Out of Memory**
- Reduce `input_size` (e.g. 416 instead of 640)
- Lower `max_detections`
- Use FP16 TensorRT engine

**Low FPS**
- Set `publish_visualization:=false`
- Switch to TensorRT backend
- Use a smaller model variant (n/s instead of m/l/x)

**Missing Dependencies**
```bash
./scripts/install_dependencies.sh
ls -la /usr/local/onnxruntime/lib   # check ONNX Runtime
ldconfig -p | grep tensorrt         # check TensorRT
```

**Debug Logging**
```bash
ros2 launch yolo_inference_cpp yolo_pose.launch.py --log-level debug
```

---

## Support

- **Issues**: Report bugs via GitHub Issues

## Authors

Alejandro Rodríguez-Ramos [https://alejandrorodriguezramos.me]

## Keywords

`YOLO` `YOLOv8` `YOLOv11` `Ultralytics` `ROS2` `C++` `TensorRT` `ONNX` `pose estimation` `object detection` `instance segmentation` `edge AI` `Jetson` `drone` `UAV` `real-time inference` `computer vision` `robotics`
