cmake_minimum_required(VERSION 3.16)
project(yolo_inference_cpp)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# CUDA support
find_package(CUDA REQUIRED)
enable_language(CUDA)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES include)

find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES lib lib64)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/KeypointDetection.msg"
  "msg/KeypointDetectionArray.msg"
  "msg/BoundingBox.msg"
  "msg/Keypoint.msg"
  "msg/PerformanceInfo.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

# Include directories
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# Conditional compilation flags
set(COMPILE_DEFINITIONS "")
set(LINK_LIBRARIES "")

# TensorRT support
if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
  message(STATUS "TensorRT found: ${TENSORRT_LIBRARY}")
  list(APPEND COMPILE_DEFINITIONS "HAVE_TENSORRT")
  include_directories(${TENSORRT_INCLUDE_DIR})
  list(APPEND LINK_LIBRARIES ${TENSORRT_LIBRARY} ${TENSORRT_PLUGIN_LIBRARY})
else()
  message(WARNING "TensorRT not found. TensorRT support will be disabled.")
endif()

# ONNX Runtime support
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
  list(APPEND COMPILE_DEFINITIONS "HAVE_ONNXRUNTIME")
  include_directories(${ONNXRUNTIME_INCLUDE_DIR})
  list(APPEND LINK_LIBRARIES ${ONNXRUNTIME_LIBRARY})
else()
  message(WARNING "ONNX Runtime not found. ONNX support will be disabled.")
endif()

# Create the main executable
add_executable(yolo_inference_node
  src/yolo_inference_node.cpp
  src/inference_backend.cpp
  src/tensorrt_backend.cpp
  src/onnx_backend.cpp
  src/memory_pool.cpp
  src/profiler.cpp
  src/preprocessing.cpp
  src/postprocessing.cpp
)

# Add compile definitions
target_compile_definitions(yolo_inference_node PRIVATE ${COMPILE_DEFINITIONS})

# Link libraries
ament_target_dependencies(yolo_inference_node
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
  OpenCV
)

target_link_libraries(yolo_inference_node
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${LINK_LIBRARIES}
)

# Link against the generated messages
rosidl_target_interfaces(yolo_inference_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

# Install
install(TARGETS yolo_inference_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/)

install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config/)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()