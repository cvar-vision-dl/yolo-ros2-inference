# Model Training & Preparation

This document covers training, distillation, export, and benchmarking of YOLO models for use with `yolo_inference_cpp`.

> This package works with models trained using [Ultralytics](https://github.com/ultralytics/ultralytics). See their [documentation](https://docs.ultralytics.com/) for additional training guides and best practices.

---

## Training

Train your YOLO model using the provided script. The following parameters have been found to work well in practice. Choose your YOLO variant with `--model` (e.g. `yolo11x-pose.pt`, `yolo11n-pose.pt`, `yolov8m-pose.pt`).

```bash
python scripts/yolo_training.py \
    --data dataset.yaml \
    --project experiments/ \
    --model yolo11x-pose.pt \
    --pretrained \
    --batch-size 4 \
    --imgsz 640 \
    --multiscale \
    --val-scales 320 640 832 \
    --val-every 3
```

To visualize training with TensorBoard:
```bash
yolo settings tensorboard=True
```

### Training Script Parameters

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
| `--multiscale` | flag | `False` | `True`/`False` | Enable multi-scale training (0.5x–1.5x range) |
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

---

## Distillation (Optional)

Distill a large teacher model into a smaller, faster student model for deployment on edge devices.

```bash
pip uninstall ultralytics
git clone https://github.com/alejodosr/yolo-distiller
```

```bash
PYTHONPATH=$PYTHONPATH:<path_to_yolo-distiller> python scripts/yolo_distillation.py \
    --teacher teacher_model.pt \
    --student yolo11n-pose.pt \
    --data dataset.yaml \
    --project experiments/ \
    --batch-size 4 \
    --imgsz 640 \
    --multiscale \
    --val-scales 320 640 832 \
    --val-every 3
```

Use TensorBoard to visualize distillation performance.

---

## Export to ONNX / TensorRT

```bash
python scripts/yolo_batch_exporter_validator.py \
    --model-folders models/experiment_1 models/experiment_2 \
    --output-dir exported_models/ \
    --task pose \
    --trtexec-path /usr/src/tensorrt/bin/trtexec \
    --tensorrt-precision all \
    --use-tensorrt
```

> **Jetson note**: TensorRT engines are platform-specific and must be built on the target device. ONNX models are cross-platform. Use FP16 precision on Jetson for optimal performance.

---

## Benchmarking

Compare models across FPS and accuracy:

```bash
python scripts/yolo_benchmarking.py \
    --models-folder exported_models/ \
    --dataset-yaml dataset.yaml \
    --output-dir benchmark_results/ \
    --task pose \
    --timing-runs 100 \
    --warmup-runs 10
```

Open the generated `.html` files for interactive FPS vs. accuracy plots.
