#!/bin/bash
# scripts/install_dependencies.sh - Install dependencies for Jetson/x86_64

set -e

echo "Installing YOLO Inference C++ dependencies..."

# Detect platform
if [ -f /etc/nv_tegra_release ]; then
    PLATFORM="jetson"
    echo "Detected Jetson platform"
else
    PLATFORM="x86_64"
    echo "Detected x86_64 platform"
fi

# Install system dependencies
#echo "Installing system dependencies..."
#sudo apt update
#sudo apt install -y \
#    build-essential \
#    cmake \
#    git \
#    wget \
#    curl \
#    unzip \
#    python3-pip \
#    libopencv-dev \
#    libopencv-contrib-dev \
#    pkg-config

## Install ROS2 Humble (if not installed)
#if ! command -v ros2 &> /dev/null; then
#    echo "Installing ROS2 Humble..."
#    sudo apt install -y software-properties-common
#    sudo add-apt-repository universe
#    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
#    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
#    sudo apt update
#    sudo apt install -y ros-humble-desktop-full
#    sudo apt install -y python3-rosdep2
#    sudo rosdep init || true
#    rosdep update
#fi

# Install ROS2 dependencies
echo "Installing ROS2 dependencies..."
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-vision-msgs

# Install CUDA (platform specific)
#if [ "$PLATFORM" = "jetson" ]; then
#    echo "CUDA should already be installed on Jetson. Checking..."
#    if ! command -v nvcc &> /dev/null; then
#        echo "CUDA not found. Please install JetPack SDK."
#        exit 1
#    fi
#    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
#    echo "Found CUDA version: $CUDA_VERSION"
#else
#    # x86_64 CUDA installation
#    echo "Installing CUDA for x86_64..."
#    if ! command -v nvcc &> /dev/null; then
#        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#        sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#        sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
#        sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
#        sudo apt update
#        sudo apt install -y cuda-toolkit-11-8
#        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
#        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
#    fi
#fi

# Install TensorRT (platform specific)
install_tensorrt() {
    if [ "$PLATFORM" = "jetson" ]; then
        echo "Installing TensorRT for Jetson..."
        sudo apt install -y tensorrt
        sudo apt install -y python3-libnvinfer-dev
        sudo apt install -y libnvinfer-plugin-dev
        sudo apt install -y libnvparsers-dev
        sudo apt install -y libnvonnxparsers-dev
    else
        echo "Installing TensorRT for x86_64..."
        # Download and install TensorRT
        TRT_VERSION="8.6.1.6"
        TRT_FILE="TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-11.8.cudnn8.6.tar.gz"

        if [ ! -f "/tmp/$TRT_FILE" ]; then
            echo "Please download TensorRT from NVIDIA Developer website:"
            echo "https://developer.nvidia.com/nvidia-tensorrt-download"
            echo "Place the file at /tmp/$TRT_FILE"
            echo "Or provide the path:"
            read -p "Enter TensorRT tar.gz path (or press Enter to skip): " TRT_PATH

            if [ -n "$TRT_PATH" ] && [ -f "$TRT_PATH" ]; then
                sudo tar -xzf "$TRT_PATH" -C /usr/local/
                echo 'export LD_LIBRARY_PATH=/usr/local/TensorRT-*/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
            else
                echo "Skipping TensorRT installation"
                return
            fi
        fi
    fi

    echo "TensorRT installation completed"
}

# Install ONNX Runtime
install_onnxruntime() {
    echo "Installing ONNX Runtime..."

    if [ "$PLATFORM" = "jetson" ]; then
        # Install ONNX Runtime for Jetson
        ORT_VERSION="1.16.3"
        ORT_FILE="onnxruntime-linux-aarch64-gpu-${ORT_VERSION}.tgz"
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"

        cd /tmp
        wget "$ORT_URL"
        sudo tar -xzf "$ORT_FILE" -C /usr/local/
        sudo mv /usr/local/onnxruntime-linux-aarch64-gpu-${ORT_VERSION} /usr/local/onnxruntime

        echo 'export LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export ONNXRUNTIME_ROOT=/usr/local/onnxruntime' >> ~/.bashrc
    else
        # Install ONNX Runtime for x86_64
        ORT_VERSION="1.16.3"
        ORT_FILE="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
        ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILE}"

        cd /tmp
        wget "$ORT_URL"
        sudo tar -xzf "$ORT_FILE" -C /usr/local/
        sudo mv /usr/local/onnxruntime-linux-x64-gpu-${ORT_VERSION} /usr/local/onnxruntime

        echo 'export LD_LIBRARY_PATH=/usr/local/onnxruntime/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
        echo 'export ONNXRUNTIME_ROOT=/usr/local/onnxruntime' >> ~/.bashrc
    fi

    echo "ONNX Runtime installation completed"
}

# Ask user which backends to install
echo "Which inference backends would you like to install?"
echo "1) TensorRT only (recommended for Jetson)"
echo "2) ONNX Runtime only (cross-platform)"
echo "3) Both TensorRT and ONNX Runtime (maximum compatibility)"
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        install_tensorrt
        ;;
    2)
        install_onnxruntime
        ;;
    3)
        install_tensorrt
        install_onnxruntime
        ;;
    *)
        echo "Invalid choice. Installing ONNX Runtime only."
        install_onnxruntime
        ;;
esac

# Create workspace if it doesn't exist
#WORKSPACE_DIR="$HOME/ros2_ws"
#if [ ! -d "$WORKSPACE_DIR" ]; then
#    echo "Creating ROS2 workspace..."
#    mkdir -p "$WORKSPACE_DIR/src"
#    cd "$WORKSPACE_DIR"
#    source /opt/ros/humble/setup.bash
#    rosdep install --from-paths src --ignore-src -r -y || true
#fi

echo "Dependencies installation completed!"
#echo ""
#echo "Next steps:"
#echo "1. Source your bashrc: source ~/.bashrc"
#echo "2. Clone this repository to $WORKSPACE_DIR/src/"
#echo "3. Build the package: cd $WORKSPACE_DIR && colcon build"
#echo ""