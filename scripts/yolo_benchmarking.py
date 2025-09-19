#!/usr/bin/env python3
"""
Model Performance and Speed Benchmark Analyzer
Loads all models in a folder, computes metrics and inference speed,
generates performance reports with time vs accuracy visualizations.
"""

import argparse
import os
import sys
import json
import time
import random
import re
import gc
from pathlib import Path
from typing import List, Dict, Any, Tuple
import warnings

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from PIL import Image
    import torch
except ImportError as e:
    print(f"Error: {e}")
    print("Install with: pip install ultralytics matplotlib seaborn pandas pillow torch")
    sys.exit(1)


class ModelBenchmarkAnalyzer:
    """Analyzes model performance metrics and inference speed"""

    def __init__(self,
                 models_folder: str,
                 dataset_yaml: str,
                 output_dir: str,
                 task: str = 'pose',
                 device: int = 0,
                 timing_runs: int = 100,
                 warmup_runs: int = 10,
                 recursive: bool = False):

        self.models_folder = models_folder
        self.dataset_yaml = dataset_yaml
        self.output_dir = output_dir
        self.task = task.lower()
        self.device = device
        self.timing_runs = timing_runs
        self.warmup_runs = warmup_runs
        self.recursive = recursive

        self.results = []
        self.timing_images = []

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Validate inputs
        self._validate_inputs()

        # Set up plotting style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'use') else 'default')
        sns.set_palette("husl")

    def _validate_inputs(self):
        """Validate all input paths exist"""
        if not os.path.exists(self.models_folder):
            raise FileNotFoundError(f"Models folder not found: {self.models_folder}")

        if not os.path.exists(self.dataset_yaml):
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")

        print("✓ All input paths validated")
        print(f"✓ Output directory: {self.output_dir}")
        print(f"✓ Task type: {self.task}")
        print(f"✓ Timing runs: {self.timing_runs} (warmup: {self.warmup_runs})")

    def _extract_resolution_from_name(self, model_name: str) -> int:
        """Extract resolution from model name (e.g., 'img832' -> 832)"""
        match = re.search(r'img(\d+)', model_name.lower())
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: Could not extract resolution from '{model_name}', using default 640")
            return 640

    def find_model_files(self) -> List[Dict[str, Any]]:
        """Find all model files in the specified folder"""
        model_extensions = ['.pt', '.onnx', '.engine']
        model_files = []

        search_path = Path(self.models_folder)

        if self.recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in search_path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in model_extensions:
                resolution = self._extract_resolution_from_name(file_path.stem)
                model_info = {
                    'name': file_path.stem,
                    'path': str(file_path),
                    'extension': file_path.suffix.lower(),
                    'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                    'resolution': resolution
                }
                model_files.append(model_info)
                print(f"✓ Found model: {file_path.name} ({model_info['size_mb']} MB, {resolution}px)")

        print(f"\nFound {len(model_files)} model files to analyze")
        return sorted(model_files, key=lambda x: x['name'])

    def prepare_timing_images(self, num_images: int = 50) -> List[str]:
        """Prepare a set of images for timing measurements"""
        print(f"\nPreparing {num_images} images for timing measurements...")

        try:
            # Load dataset info to get image paths
            import yaml
            with open(self.dataset_yaml, 'r') as f:
                dataset_info = yaml.safe_load(f)

            # Get validation images path
            val_images_path = None
            if 'val' in dataset_info:
                val_images_path = dataset_info['val']
            elif 'path' in dataset_info:
                # Try common validation folder names
                base_path = Path(dataset_info['path'])
                for val_folder in ['val', 'valid', 'validation', 'test']:
                    potential_path = base_path / val_folder
                    if potential_path.exists():
                        val_images_path = str(potential_path)
                        break

            if not val_images_path or not os.path.exists(val_images_path):
                raise FileNotFoundError(f"Validation images path not found")

            # Find image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []

            for ext in image_extensions:
                image_files.extend(Path(val_images_path).glob(f"**/*{ext}"))
                image_files.extend(Path(val_images_path).glob(f"**/*{ext.upper()}"))

            if len(image_files) < num_images:
                print(f"Warning: Only found {len(image_files)} images, using all of them")
                num_images = len(image_files)

            # Randomly select images for timing
            selected_images = random.sample(image_files, num_images)
            self.timing_images = [str(img) for img in selected_images]

            print(f"✓ Prepared {len(self.timing_images)} timing images")
            return self.timing_images

        except Exception as e:
            print(f"Warning: Could not prepare timing images: {e}")
            print("Will use synthetic data for timing")
            # Create synthetic timing data - this won't be accurate but allows testing
            self.timing_images = [f"synthetic_{i}.jpg" for i in range(num_images)]
            return self.timing_images

    def measure_inference_time(self, model, image_size: int = 640) -> float:
        """Measure average inference time for a model (excludes model loading time)"""
        print(f"  Measuring inference time...")

        times = []

        # Warmup runs
        print(f"    Warmup runs: {self.warmup_runs}")
        for _ in range(self.warmup_runs):
            if self.timing_images and os.path.exists(self.timing_images[0]):
                # Use real image
                img_path = random.choice(self.timing_images)
                try:
                    _ = model.predict(img_path, imgsz=image_size, device=self.device, verbose=False, batch=1)
                except:
                    # Fallback to synthetic data
                    dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                    _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)
            else:
                # Use synthetic data
                dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)

        # Timing runs
        print(f"    Timing runs: {self.timing_runs}")
        for i in range(self.timing_runs):
            if i % 20 == 0:
                print(f"    Progress: {i}/{self.timing_runs}")

            start_time = time.perf_counter()

            if self.timing_images and len(self.timing_images) > 0:
                # Use real image
                img_path = random.choice(self.timing_images)
                try:
                    if os.path.exists(img_path):
                        _ = model.predict(img_path, imgsz=image_size, device=self.device, verbose=False, batch=1)
                    else:
                        # Fallback
                        dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                        _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)
                except:
                    # Fallback to synthetic data
                    dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                    _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False, batch=1)
            else:
                # Use synthetic data
                dummy_img = np.random.randint(0, 255, (image_size, image_size, 3), dtype=np.uint8)
                _ = model.predict(dummy_img, imgsz=image_size, device=self.device, verbose=False)

            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"    Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
        return avg_time, std_time, times

    def validate_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model and collect metrics + timing"""
        print(f"\n=== Analyzing {model_info['name']} ===")
        print(f"Path: {model_info['path']}")
        print(f"Size: {model_info['size_mb']} MB")
        print(f"Format: {model_info['extension']}")

        result = {
            'model_name': model_info['name'],
            'model_path': model_info['path'],
            'file_extension': model_info['extension'],
            'file_size_mb': model_info['size_mb'],
            'resolution': model_info['resolution'],
            'validation_success': False,
            'metrics': {},
            'timing': {}
        }

        try:
            # Load the model
            print("  Loading model...")
            model_load_start = time.time()
            model = YOLO(model_info['path'], task=self.task)
            model_load_time = time.time() - model_load_start
            print(f"  Model loaded in {model_load_time:.2f}s")

            # Run validation using the extracted resolution
            print("  Running validation...")
            image_size = model_info['resolution']
            print(f"  Using resolution for validation: {image_size}px")

            validation_start = time.time()
            validation_results = model.val(
                data=self.dataset_yaml,
                imgsz=image_size,  # Use extracted resolution
                batch=1,  # Use batch size 1 for streaming video use case
                device=self.device,
                verbose=False
            )
            validation_time = time.time() - validation_start

            # Extract metrics based on task type
            if validation_results is not None:
                metrics = {
                    'validation_time': validation_time,
                    'model_load_time': model_load_time,
                }

                # Extract metrics based on task type
                if self.task == 'pose' and hasattr(validation_results, 'pose') and validation_results.pose is not None:
                    metrics.update({
                        'map50_95': float(validation_results.pose.map),
                        'map50': float(validation_results.pose.map50),
                        'precision': float(validation_results.pose.mp),
                        'recall': float(validation_results.pose.mr),
                    })
                    print(f"  Using POSE metrics")
                elif hasattr(validation_results, 'box') and validation_results.box is not None:
                    metrics.update({
                        'map50_95': float(validation_results.box.map),
                        'map50': float(validation_results.box.map50),
                        'precision': float(validation_results.box.mp),
                        'recall': float(validation_results.box.mr),
                    })
                    print(f"  Using BBOX metrics")
                else:
                    # Fallback
                    print(f"  Warning: Could not extract metrics for task '{self.task}'")
                    metrics.update({
                        'map50_95': 0.0,
                        'map50': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                    })

                # Measure inference time using the resolution from model name
                image_size = model_info['resolution']
                print(f"  Using resolution: {image_size}px (extracted from model name)")

                avg_inference_time, std_inference_time, all_times = self.measure_inference_time(model, image_size)

                # Add timing information
                timing_info = {
                    'avg_inference_time_ms': avg_inference_time,
                    'std_inference_time_ms': std_inference_time,
                    'min_inference_time_ms': float(np.min(all_times)),
                    'max_inference_time_ms': float(np.max(all_times)),
                    'fps': 1000.0 / avg_inference_time,  # Convert ms to FPS
                    'image_size': image_size,
                    'timing_runs': self.timing_runs
                }

                result['metrics'] = metrics
                result['timing'] = timing_info
                result['validation_success'] = True

                print(f"  ✓ Validation completed in {validation_time:.2f}s")
                print(f"  ✓ mAP@0.5:0.95: {metrics['map50_95']:.4f}")
                print(f"  ✓ mAP@0.5:     {metrics['map50']:.4f}")
                print(f"  ✓ Avg inference: {avg_inference_time:.2f} ms ({timing_info['fps']:.1f} FPS)")

            else:
                print("  ❌ Validation failed - no results returned")

        except Exception as e:
            print(f"  ❌ Analysis failed: {e}")
            result['error'] = str(e)

        finally:
            # Detailed cleanup for TensorRT models
            try:
                # Try to access TensorRT-specific components and delete in specified order
                if hasattr(model, 'predictor') and model.predictor is not None:
                    predictor = model.predictor

                    # 1. Context
                    if hasattr(predictor, 'context') and predictor.context is not None:
                        try:
                            del predictor.context
                            print(f"    Deleted TensorRT Context")
                        except:
                            pass

                    # 2. Engine
                    if hasattr(predictor, 'engine') and predictor.engine is not None:
                        try:
                            del predictor.engine
                            print(f"    Deleted TensorRT Engine")
                        except:
                            pass

                    # 3. Runtime
                    if hasattr(predictor, 'runtime') and predictor.runtime is not None:
                        try:
                            del predictor.runtime
                            print(f"    Deleted TensorRT Runtime")
                        except:
                            pass

                    # 4. Logger
                    if hasattr(predictor, 'logger') and predictor.logger is not None:
                        try:
                            del predictor.logger
                            print(f"    Deleted TensorRT Logger")
                        except:
                            pass

                    # Delete predictor
                    try:
                        del model.predictor
                        print(f"    Deleted Predictor")
                    except:
                        pass

                # Delete the model itself
                try:
                    del model
                    print(f"    Deleted Model")
                except:
                    pass

            except Exception as cleanup_error:
                print(f"    Warning during TensorRT cleanup: {cleanup_error}")

            # Force garbage collection
            gc.collect()

            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for all operations to complete
                print(f"  🧹 GPU memory cleaned and synchronized")

        return result

    def analyze_all_models(self) -> List[Dict[str, Any]]:
        """Analyze all models in the folder"""
        model_files = self.find_model_files()

        if not model_files:
            print("No model files found to analyze!")
            return []

        # Prepare timing images
        self.prepare_timing_images()

        print(f"\n{'=' * 60}")
        print(f"ANALYZING {len(model_files)} MODELS")
        print(f"{'=' * 60}")

        for i, model_info in enumerate(model_files, 1):
            print(f"\n[{i}/{len(model_files)}] Processing {model_info['name']}")

            result = self.validate_model(model_info)
            self.results.append(result)

        return self.results

    def generate_visualizations(self):
        """Generate performance vs speed visualizations with 100+ unique color/shape combinations"""
        successful_results = [r for r in self.results if r['validation_success']]

        if not successful_results:
            print("No successful results to visualize!")
            return

        # Create DataFrame for easier plotting
        plot_data = []
        for result in successful_results:
            metrics = result['metrics']
            timing = result['timing']

            plot_data.append({
                'model_name': result['model_name'],
                'file_extension': result['file_extension'],
                'file_size_mb': result['file_size_mb'],
                'resolution': result['resolution'],
                'inference_time_ms': timing['avg_inference_time_ms'],
                'fps': timing['fps'],
                'map50_95': metrics['map50_95'],
                'map50': metrics['map50'],
                'precision': metrics['precision'],
                'recall': metrics['recall']
            })

        df = pd.DataFrame(plot_data)

        # Define metrics to plot based on task
        if self.task == 'pose':
            metrics_to_plot = {
                'map50_95': 'mAP@0.5:0.95 (Keypoints)',
                'map50': 'mAP@0.5 (Keypoints)',
                'precision': 'Precision (Keypoints)',
                'recall': 'Recall (Keypoints)'
            }
        else:
            metrics_to_plot = {
                'map50_95': 'mAP@0.5:0.95 (Boxes)',
                'map50': 'mAP@0.5 (Boxes)',
                'precision': 'Precision (Boxes)',
                'recall': 'Recall (Boxes)'
            }

        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        # Extended marker list with 40+ unique shapes
        base_markers = [
            # Basic shapes
            'o', 's', '^', 'v', '<', '>', 'D', 'p', '*', 'h', 'H', '+', 'x', '8',
            'P', 'X', '1', '2', '3', '4', 'd', '|', '_', '.', ',',
            # Additional numbered markers
            '0', '5', '6', '7', '9', '10', '11',
            # More geometric shapes (if supported by matplotlib version)
            '$\\bigodot$', '$\\bigoplus$', '$\\bigotimes$', '$\\bigcirc$',
            '$\\blacksquare$', '$\\blacktriangle$', '$\\blacktriangledown$',
            '$\\diamond$', '$\\star$', '$\\heartsuit$', '$\\spadesuit$',
            '$\\clubsuit$', '$\\diamondsuit$'
        ]

        # Define a diverse color palette with 20+ distinct colors
        color_palette = [
            '#1f77b4',  # blue
            '#ff7f0e',  # orange
            '#2ca02c',  # green
            '#d62728',  # red
            '#9467bd',  # purple
            '#8c564b',  # brown
            '#e377c2',  # pink
            '#7f7f7f',  # gray
            '#bcbd22',  # olive
            '#17becf',  # cyan
            '#aec7e8',  # light blue
            '#ffbb78',  # light orange
            '#98df8a',  # light green
            '#ff9896',  # light red
            '#c5b0d5',  # light purple
            '#c49c94',  # light brown
            '#f7b6d3',  # light pink
            '#c7c7c7',  # light gray
            '#dbdb8d',  # light olive
            '#9edae5',  # light cyan
            '#393b79',  # dark blue
            '#637939',  # dark green
            '#8c6d31',  # dark brown
            '#843c39',  # dark red
            '#7b4173',  # dark purple
            '#5254a3',  # indigo
            '#8ca252',  # sage
            '#bd9e39',  # gold
            '#ad494a',  # crimson
            '#a55194'  # magenta
        ]

        # Create color-shape combinations (100+ unique combinations)
        color_shape_combinations = []
        for i in range(max(200, len(df))):  # Ensure we have enough combinations
            color_idx = i % len(color_palette)
            marker_idx = (i // len(color_palette)) % len(base_markers)
            color_shape_combinations.append((color_palette[color_idx], base_markers[marker_idx]))

        # Create legend labels with model info
        legend_labels = [f"{row['model_name']} ({row['resolution']}px, {row['file_extension']})"
                         for _, row in df.iterrows()]

        for idx, (metric, title) in enumerate(metrics_to_plot.items()):
            ax = axes[idx]

            # Scatter plot with distinct color-shape combinations
            for i, (_, row) in enumerate(df.iterrows()):
                color, marker = color_shape_combinations[i]
                ax.scatter(row['fps'], row[metric],
                           c=color, marker=marker, s=120, alpha=0.8,
                           edgecolors='black', linewidth=1)

            # Set axis labels and title
            ax.set_xlabel('Frames Per Second (FPS)', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} vs FPS', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Adjust y-axis to show full range with some padding
            y_min, y_max = df[metric].min(), df[metric].max()
            y_padding = (y_max - y_min) * 0.05  # 5% padding
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

            # Add trend line (using FPS now)
            if len(df) > 1:  # Only add trend line if we have multiple points
                z = np.polyfit(df['fps'], df[metric], 1)
                p = np.poly1d(z)
                x_trend = np.linspace(df['fps'].min(), df['fps'].max(), 100)
                ax.plot(x_trend, p(x_trend), "k--", alpha=0.6, linewidth=2, label='Trend')

            # Add x-axis padding for better visibility
            x_min, x_max = df['fps'].min(), df['fps'].max()
            x_padding = (x_max - x_min) * 0.05
            ax.set_xlim(x_min - x_padding, x_max + x_padding)

        # Create custom legend with color-shape combinations
        legend_elements = []
        for i, label in enumerate(legend_labels):
            color, marker = color_shape_combinations[i]
            from matplotlib.lines import Line2D
            legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                          markerfacecolor=color, markersize=10,
                                          markeredgecolor='black', markeredgewidth=1,
                                          label=label))

        # Add a single legend for all subplots
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                   fontsize=10, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_vs_fps.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create enhanced FPS vs mAP plot with legend (using same color-shape combinations)
        plt.figure(figsize=(14, 10))

        # Plot each model with distinct color-shape combinations
        for i, (_, row) in enumerate(df.iterrows()):
            color, marker = color_shape_combinations[i]
            plt.scatter(row['fps'], row['map50_95'],
                        c=color, marker=marker, s=row['file_size_mb'] * 15,  # Size based on file size
                        alpha=0.8, edgecolors='black', linewidth=1)

        plt.xlabel('Frames Per Second (FPS)', fontsize=12)
        plt.ylabel('mAP@0.5:0.95', fontsize=12)
        plt.title(f'Model Performance: Speed vs Accuracy\n(Bubble size = File size)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        # Adjust y-axis for better visibility
        y_min, y_max = df['map50_95'].min(), df['map50_95'].max()
        y_padding = (y_max - y_min) * 0.05
        plt.ylim(y_min - y_padding, y_max + y_padding)

        # Adjust x-axis for better visibility
        x_min, x_max = df['fps'].min(), df['fps'].max()
        x_padding = (x_max - x_min) * 0.05
        plt.xlim(x_min - x_padding, x_max + x_padding)

        # Create custom legend with model details and color-shape combinations
        legend_elements_detailed = []
        legend_labels_detailed = [
            f"{row['model_name']}\n({row['resolution']}px, {row['file_size_mb']:.1f}MB, {row['file_extension']})"
            for _, row in df.iterrows()]

        for i, label in enumerate(legend_labels_detailed):
            color, marker = color_shape_combinations[i]
            legend_elements_detailed.append(Line2D([0], [0], marker=marker, color='w',
                                                   markerfacecolor=color, markersize=10,
                                                   markeredgecolor='black', markeredgewidth=1,
                                                   label=label))

        plt.legend(handles=legend_elements_detailed, loc='center left', bbox_to_anchor=(1.02, 0.5),
                   fontsize=9, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'fps_vs_map_enhanced.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Create a color-shape reference guide
        plt.figure(figsize=(16, 12))

        # Create a grid layout for the reference
        n_models = len(df)
        cols = min(8, max(6, n_models // 4))  # Adaptive columns based on number of models
        rows = (n_models + cols - 1) // cols  # Calculate needed rows

        for i, (_, row) in enumerate(df.iterrows()):
            color, marker = color_shape_combinations[i]
            col = i % cols
            row_idx = i // cols

            x = col
            y = rows - row_idx - 1  # Flip y to start from top

            plt.scatter(x, y, c=color, marker=marker, s=300,
                        alpha=0.8, edgecolors='black', linewidth=1.5)

            # Add model name below each point
            plt.text(x, y - 0.15, row['model_name'], ha='center', va='top',
                     fontsize=8, rotation=0, weight='bold')
            plt.text(x, y - 0.25, f"{row['resolution']}px", ha='center', va='top',
                     fontsize=7, style='italic')
            # Add color/marker info
            plt.text(x, y - 0.35, f"C:{color_palette.index(color) + 1} M:{base_markers.index(marker) + 1}",
                     ha='center', va='top', fontsize=6, alpha=0.7)

        plt.xlim(-0.5, cols - 0.5)
        plt.ylim(-0.5, rows - 0.5)
        plt.title('Model Color-Shape Reference Guide\n(C = Color Index, M = Marker Index)',
                  fontsize=16, fontweight='bold')
        plt.axis('off')  # Hide axes

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_reference_guide.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # Calculate total possible combinations
        total_combinations = len(color_palette) * len(base_markers)
        unique_combinations_used = min(len(df), total_combinations)

        print(f"✓ Visualizations saved to {self.output_dir}")
        print(f"  - performance_vs_fps.png (4-plot analysis)")
        print(f"  - fps_vs_map_enhanced.png (main performance plot)")
        print(f"  - model_reference_guide.png (color-shape reference guide)")
        print(
            f"  - Total possible combinations: {total_combinations} ({len(color_palette)} colors × {len(base_markers)} shapes)")
        print(f"  - Unique combinations used: {unique_combinations_used} for {len(df)} models")
        print(f"  - Can handle up to {total_combinations} different models with unique color-shape combinations")

    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.results:
            print("No results to generate report from!")
            return {}

        successful_results = [r for r in self.results if r['validation_success']]
        failed_results = [r for r in self.results if not r['validation_success']]

        # Sort by mAP for ranking
        successful_results.sort(key=lambda x: x['metrics'].get('map50_95', 0), reverse=True)

        # Calculate efficiency metrics
        for result in successful_results:
            if result['validation_success']:
                metrics = result['metrics']
                timing = result['timing']

                # Efficiency metrics
                map_per_mb = metrics['map50_95'] / max(result['file_size_mb'], 0.1)
                map_per_ms = metrics['map50_95'] / max(timing['avg_inference_time_ms'], 0.1)

                result['efficiency'] = {
                    'map_per_mb': map_per_mb,
                    'map_per_ms': map_per_ms,
                    'accuracy_speed_score': metrics['map50_95'] * timing['fps'] / 100  # Normalized score
                }

        # Create comprehensive report
        report = {
            'analysis_info': {
                'models_folder': self.models_folder,
                'dataset_yaml': self.dataset_yaml,
                'task_type': self.task,
                'timing_runs': self.timing_runs,
                'warmup_runs': self.warmup_runs,
                'device': self.device,
                'total_models': len(self.results),
                'successful_analyses': len(successful_results),
                'failed_analyses': len(failed_results)
            },
            'successful_models': successful_results,
            'failed_models': failed_results,
            'rankings': {
                'best_accuracy': successful_results[:5] if successful_results else [],
                'fastest_models': sorted(successful_results,
                                         key=lambda x: x['timing']['avg_inference_time_ms'])[:5],
                'most_efficient': sorted(successful_results,
                                         key=lambda x: x['efficiency']['accuracy_speed_score'],
                                         reverse=True)[:5] if successful_results else []
            }
        }

        # Save report
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'benchmark_report.json')

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Report saved to: {output_file}")

        # Save CSV summary
        self.save_csv_summary(successful_results)

        return report

    def save_csv_summary(self, successful_results: List[Dict]):
        """Save CSV summary of all results"""
        csv_file = os.path.join(self.output_dir, 'benchmark_summary.csv')

        try:
            import csv

            with open(csv_file, 'w', newline='') as f:
                fieldnames = [
                    'model_name', 'file_extension', 'file_size_mb', 'resolution',
                    'map50_95', 'map50', 'precision', 'recall',
                    'avg_inference_time_ms', 'fps', 'accuracy_speed_score'
                ]

                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for result in successful_results:
                    row = {
                        'model_name': result['model_name'],
                        'file_extension': result['file_extension'],
                        'file_size_mb': result['file_size_mb'],
                        'resolution': result['resolution'],
                        'map50_95': result['metrics']['map50_95'],
                        'map50': result['metrics']['map50'],
                        'precision': result['metrics']['precision'],
                        'recall': result['metrics']['recall'],
                        'avg_inference_time_ms': result['timing']['avg_inference_time_ms'],
                        'fps': result['timing']['fps'],
                        'accuracy_speed_score': result['efficiency']['accuracy_speed_score']
                    }
                    writer.writerow(row)

            print(f"📊 CSV summary saved to: {csv_file}")

        except Exception as e:
            print(f"Warning: Could not save CSV summary: {e}")

    def print_summary(self, report: Dict[str, Any]):
        """Print summary to console"""
        print(f"\n{'=' * 80}")
        print("MODEL BENCHMARK ANALYSIS SUMMARY")
        print(f"{'=' * 80}")

        info = report['analysis_info']
        print(f"Task type: {info['task_type']}")
        print(f"Models analyzed: {info['total_models']}")
        print(f"Successful: {info['successful_analyses']}")
        print(f"Failed: {info['failed_analyses']}")
        print(f"Timing runs per model: {info['timing_runs']}")

        if report['rankings']['best_accuracy']:
            print(f"\n🏆 TOP 5 BY ACCURACY (mAP@0.5:0.95):")
            print("-" * 115)
            print(
                f"{'Rank':<4} {'Model':<25} {'Resolution':<10} {'Format':<8} {'mAP@0.5:0.95':<12} {'Time(ms)':<10} {'FPS':<8} {'Size(MB)':<10}")
            print("-" * 115)

            for i, model in enumerate(report['rankings']['best_accuracy'], 1):
                print(f"{i:<4} {model['model_name']:<25} {model['resolution']:<10} {model['file_extension']:<8} "
                      f"{model['metrics']['map50_95']:.4f}      "
                      f"{model['timing']['avg_inference_time_ms']:.1f}      "
                      f"{model['timing']['fps']:.1f}   "
                      f"{model['file_size_mb']:.1f}")

        if report['rankings']['fastest_models']:
            print(f"\n⚡ TOP 5 FASTEST MODELS:")
            print("-" * 115)
            print(
                f"{'Rank':<4} {'Model':<25} {'Resolution':<10} {'Format':<8} {'Time(ms)':<10} {'FPS':<8} {'mAP@0.5:0.95':<12} {'Size(MB)':<10}")
            print("-" * 115)

            for i, model in enumerate(report['rankings']['fastest_models'], 1):
                print(f"{i:<4} {model['model_name']:<25} {model['resolution']:<10} {model['file_extension']:<8} "
                      f"{model['timing']['avg_inference_time_ms']:.1f}      "
                      f"{model['timing']['fps']:.1f}   "
                      f"{model['metrics']['map50_95']:.4f}      "
                      f"{model['file_size_mb']:.1f}")

        if report['rankings']['most_efficient']:
            print(f"\n💡 TOP 5 MOST EFFICIENT (Accuracy×Speed Score):")
            print("-" * 125)
            print(
                f"{'Rank':<4} {'Model':<25} {'Resolution':<10} {'Format':<8} {'Score':<8} {'mAP@0.5:0.95':<12} {'Time(ms)':<10} {'FPS':<8}")
            print("-" * 125)

            for i, model in enumerate(report['rankings']['most_efficient'], 1):
                print(f"{i:<4} {model['model_name']:<25} {model['resolution']:<10} {model['file_extension']:<8} "
                      f"{model['efficiency']['accuracy_speed_score']:.2f}   "
                      f"{model['metrics']['map50_95']:.4f}      "
                      f"{model['timing']['avg_inference_time_ms']:.1f}      "
                      f"{model['timing']['fps']:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Model Performance and Speed Benchmark Analyzer')

    parser.add_argument('--models-folder', required=True,
                        help='Folder containing model files to analyze')
    parser.add_argument('--dataset-yaml', required=True,
                        help='Path to validation dataset YAML file')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for reports and visualizations')
    parser.add_argument('--task', choices=['pose', 'detect', 'segment'], default='pose',
                        help='Model task type')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--timing-runs', type=int, default=100,
                        help='Number of timing runs per model')
    parser.add_argument('--warmup-runs', type=int, default=10,
                        help='Number of warmup runs before timing')
    parser.add_argument('--recursive', action='store_true',
                        help='Search for models recursively in subfolders')
    parser.add_argument('--output-json',
                        help='Custom output path for JSON report')

    args = parser.parse_args()

    print("🔍 Starting Model Benchmark Analysis")
    print(f"Models folder: {args.models_folder}")
    print(f"Dataset: {args.dataset_yaml}")
    print(f"Task: {args.task}")
    print(f"Output directory: {args.output_dir}")
    print(f"Timing runs: {args.timing_runs}")

    try:
        # Create analyzer
        analyzer = ModelBenchmarkAnalyzer(
            models_folder=args.models_folder,
            dataset_yaml=args.dataset_yaml,
            output_dir=args.output_dir,
            task=args.task,
            device=args.device,
            timing_runs=args.timing_runs,
            warmup_runs=args.warmup_runs,
            recursive=args.recursive
        )

        # Analyze all models
        results = analyzer.analyze_all_models()

        if not results:
            print("No models were analyzed!")
            sys.exit(1)

        # Generate visualizations
        analyzer.generate_visualizations()

        # Generate and save report
        report = analyzer.generate_report(args.output_json)

        # Print summary
        analyzer.print_summary(report)

        print(f"\n✅ Analysis completed!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Visualizations: performance_vs_speed.png, fps_vs_map.png")
        print(f"📋 Reports: benchmark_report.json, benchmark_summary.csv")

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

# Usage examples:
"""
# Basic usage for pose estimation models
python model_benchmark_analyzer.py \
    --models-folder ./trained_models \
    --dataset-yaml ./validation_dataset.yaml \
    --output-dir ./benchmark_results \
    --task pose

# For detection models with custom timing
python model_benchmark_analyzer.py \
    --models-folder ./detection_models \
    --dataset-yaml ./val_data.yaml \
    --output-dir ./detection_benchmark \
    --task detect \
    --timing-runs 200 \
    --warmup-runs 20 \
    --device 0

# Recursive search with custom output
python model_benchmark_analyzer.py \
    --models-folder ./experiments \
    --dataset-yaml ./dataset.yaml \
    --output-dir ./analysis_results \
    --task segment \
    --recursive \
    --output-json ./custom_report.json
"""