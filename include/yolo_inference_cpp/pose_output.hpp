// Copyright 2026 Universidad Politécnica de Madrid
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

/**
 * @file pose_output.hpp
 *
 * Where each value of a YOLO pose output sits, for any number of classes and keypoints
 */

#ifndef YOLO_INFERENCE_CPP__POSE_OUTPUT_HPP_
#define YOLO_INFERENCE_CPP__POSE_OUTPUT_HPP_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace yolo_inference
{

// A pose output is [1, features, anchors] (the Ultralytics export) or [1, anchors, features].
// Each anchor's features are 4 box values (cx, cy, w, h), one score per class, then x, y and
// confidence per keypoint: 29 = 4 + 1 + 8 * 3 for the one-class gate model.
struct PoseOutputLayout
{
  int64_t num_anchors = 0;
  int64_t features = 0;
  bool transposed = false;  // [1, features, anchors]
  int num_classes = 1;
  int num_keypoints = 0;

  float at(const float * output, int64_t anchor, int64_t feature) const
  {
    return transposed ? output[feature * num_anchors + anchor] : output[anchor * features + feature];
  }

  // The anchor's best class and its score, which is the detection's confidence.
  std::pair<int, float> bestClass(const float * output, int64_t anchor) const
  {
    int best = 0;
    float score = at(output, anchor, 4);
    for (int c = 1; c < num_classes; ++c) {
      const float s = at(output, anchor, 4 + c);
      if (s > score) {
        best = c;
        score = s;
      }
    }
    return {best, score};
  }

  // Keypoint k's x, y and confidence, in model input pixels.
  float keypoint(const float * output, int64_t anchor, int k, int value) const
  {
    return at(output, anchor, 4 + num_classes + k * 3 + value);
  }
};

/**
 * @brief The layout of a pose output with `num_classes` classes.
 *
 * The class count cannot be read from the shape: 68 features are 4 + 4 + 20 * 3 and also
 * 4 + 1 + 21 * 3. So it comes from the class names the backend holds, one for the gate model.
 *
 * @param shape Output tensor shape, [batch, dim1, dim2].
 * @param num_classes Classes the model was trained with; below 1 counts as 1.
 * @param error Set to why, when there is no layout.
 * @return The layout, or nullopt when the features do not split into
 *         4 + num_classes + 3 per keypoint.
 */
inline std::optional<PoseOutputLayout> poseOutputLayout(
  const std::vector<int64_t> & shape, int num_classes, std::string & error)
{
  if (shape.size() != 3) {
    error = "expected 3 dimensions, got " + std::to_string(shape.size());
    return std::nullopt;
  }
  PoseOutputLayout layout;
  // Anchors outnumber features in every YOLO head (8400 at 640 px), so the larger one is anchors.
  layout.transposed = shape[1] < shape[2];
  layout.features = std::min(shape[1], shape[2]);
  layout.num_anchors = std::max(shape[1], shape[2]);
  layout.num_classes = std::max(num_classes, 1);
  const int64_t keypoint_values = layout.features - 4 - layout.num_classes;
  if (keypoint_values <= 0 || keypoint_values % 3 != 0) {
    error = std::to_string(layout.features) + " features are not 4 box + " +
      std::to_string(layout.num_classes) + " class scores + 3 per keypoint; are the class names "
      "the model's?";
    return std::nullopt;
  }
  layout.num_keypoints = static_cast<int>(keypoint_values / 3);
  return layout;
}

}  // namespace yolo_inference

#endif  // YOLO_INFERENCE_CPP__POSE_OUTPUT_HPP_
