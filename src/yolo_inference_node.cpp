// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <memory>

#include "yolo_inference_cpp/yolo_interface.hpp"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

namespace yolo_inference
{

class YOLOInferenceNode : public rclcpp::Node
{
public:
  YOLOInferenceNode()
  : Node("yolo_inference_node")
  {
    // Create the YOLO interface with this node
    yolo_interface_ = std::make_unique<YOLOInterface>(this);

    // Get input topic parameter for subscriber
    this->declare_parameter<std::string>("input_topic", "/camera/image_raw/compressed");
    std::string input_topic = this->get_parameter("input_topic").as_string();

    // Create subscriber
    image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
      input_topic, 10,
      std::bind(&YOLOInferenceNode::imageCallback, this, std::placeholders::_1));
  }

private:
  void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    // Forward the message to YOLOInterface for processing
    yolo_interface_->processImage(msg);
  }

  std::unique_ptr<YOLOInterface> yolo_interface_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
};

}  // namespace yolo_inference

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  try {
    auto node = std::make_shared<yolo_inference::YOLOInferenceNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(rclcpp::get_logger("yolo_inference"), "Exception: %s", e.what());
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}
