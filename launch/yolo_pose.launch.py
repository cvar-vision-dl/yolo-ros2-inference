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


# from ament_index_python.packages import get_package_share_directory
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):
    class_names_str = LaunchConfiguration('class_names').perform(context)
    try:
        class_names = yaml.safe_load(class_names_str)
        if not isinstance(class_names, list):
            class_names = []
    except Exception:
        class_names = []

    yolo_node = Node(
        package='yolo_inference_cpp',
        executable='yolo_inference_cpp_node',
        name='yolo_inference_node',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'task': LaunchConfiguration('task'),
            'input_size': LaunchConfiguration('input_size'),
            'input_width': LaunchConfiguration('input_width'),
            'input_height': LaunchConfiguration('input_height'),
            'confidence_threshold': LaunchConfiguration('confidence_threshold'),
            'keypoint_threshold': LaunchConfiguration('keypoint_threshold'),
            'publish_visualization': LaunchConfiguration('publish_visualization'),
            'draw_bboxes': LaunchConfiguration('draw_bboxes'),
            'enable_profiling': LaunchConfiguration('enable_profiling'),
            'input_topic': LaunchConfiguration('input_topic'),
            'output_topic': '/yolo/detections',
            'output_image_topic': '/yolo/result_image',
            'performance_topic': '/yolo/performance',
            'class_names': class_names,
        }],
        output='screen',
        emulate_tty=True
    )
    return [yolo_node]


def generate_launch_description():
    # Get package directory
    # pkg_dir = get_package_share_directory('yolo_inference_cpp')

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

    input_width_arg = DeclareLaunchArgument(
        'input_width',
        default_value='-1',
        description='Input image width (overrides input_size if > 0)'
    )

    input_height_arg = DeclareLaunchArgument(
        'input_height',
        default_value='-1',
        description='Input image height (overrides input_size if > 0)'
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

    draw_bboxes_arg = DeclareLaunchArgument(
        'draw_bboxes',
        default_value='true',
        description='Draw bounding boxes in visualization'
    )

    class_names_arg = DeclareLaunchArgument(
        'class_names',
        default_value='[]',
        description='List of class names, e.g. "[\'person\', \'car\']"'
    )

    return LaunchDescription([
        model_path_arg,
        task_arg,
        input_size_arg,
        input_width_arg,
        input_height_arg,
        confidence_threshold_arg,
        keypoint_threshold_arg,
        publish_visualization_arg,
        enable_profiling_arg,
        input_topic_arg,
        draw_bboxes_arg,
        class_names_arg,
        OpaqueFunction(function=launch_setup),
    ])
