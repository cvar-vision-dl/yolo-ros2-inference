#!/bin/bash
# scripts/build_package.sh - Build the ROS2 package

set -e

echo "Building YOLO Inference C++ package..."

# Source ROS2
source /opt/ros/humble/setup.bash

# Create workspace if needed
#WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/ros2_ws}"
#if [ ! -d "$WORKSPACE_DIR" ]; then
#    echo "Creating workspace at $WORKSPACE_DIR"
#    mkdir -p "$WORKSPACE_DIR/src"
#fi
#
#cd "$WORKSPACE_DIR"

# Install rosdep dependencies
echo "Installing ROS dependencies..."
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build with specific configurations
echo "Building package..."

# Check what backends are available
TRT_AVAILABLE=false
ONNX_AVAILABLE=false

if [ -d "/usr/local/onnxruntime" ] || [ -n "$ONNXRUNTIME_ROOT" ]; then
    ONNX_AVAILABLE=true
    echo "ONNX Runtime detected"
fi

if command -v nvcc &> /dev/null && ([ -f "/usr/include/NvInfer.h" ] || [ -f "/usr/local/TensorRT-*/include/NvInfer.h" ]); then
    TRT_AVAILABLE=true
    echo "TensorRT detected"
fi

# Set build configuration based on available backends
if [ "$TRT_AVAILABLE" = true ] && [ "$ONNX_AVAILABLE" = true ]; then
    echo "Building with both TensorRT and ONNX Runtime support"
    BUILD_CONFIG="Release"
elif [ "$TRT_AVAILABLE" = true ]; then
    echo "Building with TensorRT support only"
    BUILD_CONFIG="Release"
elif [ "$ONNX_AVAILABLE" = true ]; then
    echo "Building with ONNX Runtime support only"
    BUILD_CONFIG="Release"
else
    echo "Warning: No inference backends detected. Building basic version."
    BUILD_CONFIG="Release"
fi

# Build the package
colcon build \
    --packages-select yolo_inference_cpp \
    --cmake-args \
        -DCMAKE_BUILD_TYPE="$BUILD_CONFIG" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    --parallel-workers $(nproc) \
    --event-handlers console_direct+

echo "Build completed successfully!"
#echo ""
#echo "To use the package:"
#echo "  source $WORKSPACE_DIR/install/setup.bash"
#echo ""