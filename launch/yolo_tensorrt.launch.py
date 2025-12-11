#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# import os
# from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Launch configuration for TensorRT optimized inference."""

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

    input_topic_arg = DeclareLaunchArgument(
        'input_topic',
        default_value='/camera/image_raw/compressed',
        description='Input compressed image topic'
    )

    # YOLO inference node with performance settings
    yolo_node = Node(
        package='yolo_inference_cpp',
        executable='yolo_inference_cpp_node',
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
            'input_topic': LaunchConfiguration('input_topic'),
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
        input_topic_arg,
        yolo_node
    ])