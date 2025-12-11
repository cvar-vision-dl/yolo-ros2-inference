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
set(PROJECT_DEPENDENCIES
  ament_cmake
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  image_transport
  OpenCV
  rosidl_default_generators
)

foreach(DEPENDENCY ${PROJECT_DEPENDENCIES})
  find_package(${DEPENDENCY} REQUIRED)
endforeach()

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
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR} /usr/include /usr/local/include
  PATH_SUFFIXES include cuda/include)

find_library(TENSORRT_LIBRARY nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib /usr/local/lib
  PATH_SUFFIXES lib lib64 lib/x64 cuda/lib64)

find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR} /usr/lib /usr/local/lib
  PATH_SUFFIXES lib lib64 lib/x64 cuda/lib64)

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT} /usr/local/onnxruntime $ENV{ONNXRUNTIME_ROOT}
  PATH_SUFFIXES include)

find_library(ONNXRUNTIME_LIBRARY onnxruntime
  HINTS ${ONNXRUNTIME_ROOT} /usr/local/onnxruntime $ENV{ONNXRUNTIME_ROOT}
  PATH_SUFFIXES lib lib64)

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

# Create the YOLO inference library
add_library(${PROJECT_NAME}_lib
  src/yolo_inference.cpp
  src/inference_backend.cpp
  src/tensorrt_backend.cpp
  src/onnx_backend.cpp
  src/profiler.cpp
  src/preprocessing.cpp
  src/gatenet_postprocessing.cpp
)

# Set include directories for the library
target_include_directories(${PROJECT_NAME}_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

# Add compile definitions to library
target_compile_definitions(${PROJECT_NAME}_lib PRIVATE ${COMPILE_DEFINITIONS})

# Link libraries to the library
ament_target_dependencies(${PROJECT_NAME}_lib
  ${PROJECT_DEPENDENCIES}
)

target_link_libraries(${PROJECT_NAME}_lib
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
  ${LINK_LIBRARIES}
)

# Link against the generated messages for the library
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_lib ${cpp_typesupport_target})

# Create the main executable (only the node)
add_executable(${PROJECT_NAME}_node
  src/yolo_inference_node.cpp
)

# Link the executable with the library
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME}_lib)

# Export libraries and targets
install(
  TARGETS
    ${PROJECT_NAME}_lib
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install the headers
install(
  DIRECTORY include/
  DESTINATION include
)

# Install the launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Install the config directory
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  file(GLOB_RECURSE EXCLUDE_FILES
    build/*
    install/*
  )
  set(AMENT_LINT_AUTO_FILE_EXCLUDE ${EXCLUDE_FILES})
  ament_lint_auto_find_test_dependencies()
endif()

# Export libraries and targets for being used by other packages
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${PROJECT_DEPENDENCIES})

ament_package()