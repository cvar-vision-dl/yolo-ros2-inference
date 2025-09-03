#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('yolo_inference_cpp')

    # Declare launch arguments
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='yolo11n-pose.onnx',
        description='Path to YOLO model file (.onnx or .engine)'
    )

    task_arg = DeclareLaunchArgument(
        'task',
        default_value='pose',
        description='Task type: pose, detect, or segment'
    )

    input_size_arg = DeclareLaunchArgument(
        'input_size',
        default_value='640',
        description='Input image size (square)'
    )

    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.5',
        description='Detection confidence threshold'
    )

    keypoint_threshold_arg = DeclareLaunchArgument(
        'keypoint_threshold',
        default_value='0.3',
        description='Keypoint visibility threshold'
    )

    publish_visualization_arg = DeclareLaunchArgument(
        'publish_visualization',
        default_value='true',
        description='Whether to publish visualization images'
    )

    enable_profiling_arg = DeclareLaunchArgument(
        'enable_profiling',
        default_value='true',
        description='Enable detailed profiling'
    )

    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/image_raw/compressed',
        description='Input compressed image topic'
    )

    # YOLO inference node
    yolo_node = Node(
        package='yolo_inference_cpp',
        executable='yolo_inference_node',
        name='yolo_inference_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'task': LaunchConfiguration('task'),
            'input_size': LaunchConfiguration('input_size'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'keypoint_threshold': LaunchConfiguration('keypoint_threshold'),
            'publish_visualization': LaunchConfiguration('publish_visualization'),
            'enable_profiling': LaunchConfiguration('enable_profiling'),
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': '/yolo/detections',
            'output_image_topic': '/yolo/result_image',
            'performance_topic': '/yolo/performance'
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
        publish_visualization_arg,
        enable_profiling_arg,
        input_topic_arg,
        yolo_node
    ])
