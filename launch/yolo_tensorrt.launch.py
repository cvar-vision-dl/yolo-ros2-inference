#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch configuration for TensorRT optimized inference"""

    # Declare launch arguments optimized for drone applications
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='yolo11n-pose-fp16.engine',
        description='Path to TensorRT engine file'
    )

    task_arg = DeclareLaunchArgument(
        'task',
        default_value='pose',
        description='Task type: pose, detect, or segment'
    )

    input_size_arg = DeclareLaunchArgument(
        'input_size',
        default_value='640',
        description='Input image size (must match engine)'
    )

    # Drone-optimized thresholds
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.6',
        description='Higher confidence for drone applications'
    )

    keypoint_threshold_arg = DeclareLaunchArgument(
        'keypoint_threshold',
        default_value='0.4',
        description='Higher keypoint threshold for reliability'
    )

    max_detections_arg = DeclareLaunchArgument(
        'max_detections',
        default_value='10',
        description='Maximum detections for performance'
    )

    # Performance settings
    publish_visualization_arg = DeclareLaunchArgument(
        'publish_visualization',
        default_value='false',
        description='Disable visualization for max performance'
    )

    enable_profiling_arg = DeclareLaunchArgument(
        'enable_profiling',
        default_value='true',
        description='Enable profiling to monitor performance'
    )

    # YOLO inference node with performance settings
    yolo_node = Node(
        package='yolo_inference_cpp',
        executable='yolo_inference_node',
        name='yolo_tensorrt_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'task': LaunchConfiguration('task'),
            'input_size': LaunchConfiguration('input_size'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'keypoint_threshold': LaunchConfiguration('keypoint_threshold'),
            'max_detections': LaunchConfiguration('max_detections'),
            'publish_visualization': LaunchConfiguration('publish_visualization'),
            'enable_profiling': LaunchConfiguration('enable_profiling'),
            'input_topic': '/camera/image_raw/compressed',
            'output_topic': '/drone/pose_detections',
            'output_image_topic': '/drone/pose_visualization',
            'performance_topic': '/drone/inference_performance'
        }],
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        model_path_arg,
        task_arg,
        input_size_arg,
        confidence_threshold_arg,
        keypoint_threshold_arg,
        max_detections_arg,
        publish_visualization_arg,
        enable_profiling_arg,
        yolo_node
    ])