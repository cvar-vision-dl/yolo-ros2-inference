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


#include "yolo_inference_cpp/gatenet_postprocessing.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <sstream>

namespace yolo_inference
{

std::vector<cv::Point3f> detectPeaks(
    const cv::Mat& heatmap,
    float threshold,
    int min_distance)
{
    std::vector<cv::Point3f> peaks;

    if (heatmap.empty() || heatmap.channels() != 1) {
        return peaks;
    }

    // Normalize
    cv::Mat normalized;
    double min_val, max_val;
    cv::minMaxLoc(heatmap, &min_val, &max_val);
    if (max_val <= 0) return peaks;
    normalized = heatmap / max_val;

    // Find all candidate peaks above threshold
    std::vector<std::tuple<float, int, int>> candidates;  // (value, x, y)

    int kernel_size = min_distance * 2 + 1;
    cv::Mat dilated;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT,
                                                cv::Size(kernel_size, kernel_size));
    cv::dilate(normalized, dilated, kernel);

    for (int y = 0; y < normalized.rows; ++y) {
        for (int x = 0; x < normalized.cols; ++x) {
            float val = normalized.at<float>(y, x);
            float dil_val = dilated.at<float>(y, x);

            if (val > threshold && std::abs(val - dil_val) < 1e-6f) {
                candidates.emplace_back(val, x, y);
            }
        }
    }

    // Sort by confidence (descending)
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    // Greedy NMS - only keep peaks that are min_distance apart
    std::vector<bool> suppressed(candidates.size(), false);

    for (size_t i = 0; i < candidates.size(); ++i) {
        if (suppressed[i]) continue;

        auto [val, x, y] = candidates[i];
        peaks.emplace_back(static_cast<float>(x), static_cast<float>(y), val);

        // Suppress nearby candidates
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (suppressed[j]) continue;

            auto [_, x2, y2] = candidates[j];
            float dist = std::sqrt(static_cast<float>((x - x2) * (x - x2) +
                                                       (y - y2) * (y - y2)));
            if (dist < min_distance) {
                suppressed[j] = true;
            }
        }
    }

    return peaks;
}

// ============================================================================
// FIXED: calculatePAFAffinity
// Two bugs were fixed:
// 1. Must count ALL points in denominator (not just non-zero PAF points)
// 2. Ensure correct dot product: dx*paf_x + dy*paf_y
// ============================================================================
float calculatePAFAffinity(
  const cv::Point2f & corner1,
  const cv::Point2f & corner2,
  const cv::Mat & vx_map,
  const cv::Mat & vy_map)
{
  // Calculate vector between corners
  cv::Point2f vector = corner2 - corner1;
  float norm = std::sqrt(vector.x * vector.x + vector.y * vector.y);

  if (norm < 1e-6f) {
    return 0.0f;
  }

  // Unit vector (x = horizontal/col direction, y = vertical/row direction)
  cv::Point2f unit_vector(vector.x / norm, vector.y / norm);

  // Sample points along the line
  std::vector<cv::Point> line_points;
  cv::LineIterator it(
    vx_map,
    cv::Point(static_cast<int>(corner1.x), static_cast<int>(corner1.y)),
    cv::Point(static_cast<int>(corner2.x), static_cast<int>(corner2.y)),
    8);  // 8-connectivity

  for (int i = 0; i < it.count; ++i, ++it) {
    line_points.push_back(it.pos());
  }

  if (line_points.empty()) {
    return 0.0f;
  }

  // Calculate mean cosine similarity
  // CRITICAL FIX: Count ALL points, not just those with non-zero PAF
  float total_similarity = 0.0f;

  for (const auto & point : line_points) {
    if (point.x >= 0 && point.x < vx_map.cols &&
        point.y >= 0 && point.y < vx_map.rows)
    {
      float vx = vx_map.at<float>(point.y, point.x);
      float vy = vy_map.at<float>(point.y, point.x);

      float paf_norm = std::sqrt(vx * vx + vy * vy);

      // FIX: Default to 0 for zero-magnitude PAF (matches Python behavior)
      float cosine = 0.0f;
      if (paf_norm > 1e-6f) {
        // Cosine similarity: dot product of unit vectors
        // unit_vector.x = dx (horizontal), unit_vector.y = dy (vertical)
        // vx = x-component of PAF, vy = y-component of PAF
        cosine = (unit_vector.x * vx + unit_vector.y * vy) / paf_norm;
      }
      // FIX: ALWAYS add to total, even if cosine is 0
      total_similarity += cosine;
    }
    // Points outside bounds implicitly contribute 0
  }

  // FIX: Divide by TOTAL number of line points, not just valid PAF points
  // This matches Python's behavior where zero-PAF regions dilute the score
  return total_similarity / static_cast<float>(line_points.size());
}

// ============================================================================
// FIXED: detectSides
// Bug fix: vx_maps and vy_maps were passed in wrong order
// ============================================================================
std::vector<std::vector<Side>> detectSides(
  const std::vector<std::vector<cv::Point3f>> & corners,
  const std::vector<cv::Mat> & vx_maps,
  const std::vector<cv::Mat> & vy_maps,
  float paf_threshold)
{
  std::vector<std::vector<Side>> all_detected_sides(GATENET_NUM_EDGES);

  for (int edge_idx = 0; edge_idx < GATENET_NUM_EDGES; ++edge_idx) {
    int corner_type1 = GATENET_EDGE_CONNECTIONS[edge_idx].first;
    int corner_type2 = GATENET_EDGE_CONNECTIONS[edge_idx].second;

    const auto & corners1 = corners[corner_type1];
    const auto & corners2 = corners[corner_type2];

    if (corners1.empty() || corners2.empty()) {
      continue;
    }

    // Get candidate sides (all possible pairs)
    std::vector<Side> candidate_sides;
    for (const auto & c1 : corners1) {
      for (const auto & c2 : corners2) {
        // FIX: Pass vx_maps first, then vy_maps (was swapped before!)
        float score = calculatePAFAffinity(
          cv::Point2f(c1.x, c1.y),
          cv::Point2f(c2.x, c2.y),
          vy_maps[edge_idx],
          vx_maps[edge_idx]);

        if (score > paf_threshold) {
          Side side;
          side.corner1 = cv::Point2f(c1.x, c1.y);
          side.corner2 = cv::Point2f(c2.x, c2.y);
          side.corner_type1 = corner_type1;
          side.corner_type2 = corner_type2;
          side.edge_idx = edge_idx;
          side.score = score;
          side.used = false;
          candidate_sides.push_back(side);
        }
      }
    }

    // Greedy matching: select best sides without sharing corners
    std::vector<Side> selected_sides;
    while (!candidate_sides.empty()) {
      // Find highest scoring candidate
      auto max_it = std::max_element(
        candidate_sides.begin(),
        candidate_sides.end(),
        [](const Side & a, const Side & b) {return a.score < b.score;});

      if (max_it != candidate_sides.end()) {
        selected_sides.push_back(*max_it);

        cv::Point2f selected_c1 = max_it->corner1;
        cv::Point2f selected_c2 = max_it->corner2;

        // Remove all candidates sharing either corner
        candidate_sides.erase(
          std::remove_if(
            candidate_sides.begin(),
            candidate_sides.end(),
            [&](const Side & s) {
              float dist1_1 = cv::norm(s.corner1 - selected_c1);
              float dist1_2 = cv::norm(s.corner2 - selected_c2);
              float dist2_1 = cv::norm(s.corner1 - selected_c2);
              float dist2_2 = cv::norm(s.corner2 - selected_c1);
              return dist1_1 < 1e-3 || dist1_2 < 1e-3 || dist2_1 < 1e-3 || dist2_2 < 1e-3;
            }),
          candidate_sides.end());
      }
    }

    all_detected_sides[edge_idx] = selected_sides;
  }

  return all_detected_sides;
}

std::vector<Gate> assembleGates(
  std::vector<std::vector<Side>> & all_sides,
  int min_corners)
{
  std::vector<Gate> gates;

  // Build graph of corner connections
  std::unordered_map<std::string, std::vector<std::pair<int, Side *>>> corner_graph;

  auto point_to_key = [](const cv::Point2f & p) -> std::string {
      std::ostringstream oss;
      oss << std::round(p.x * 10.0f) / 10.0f << ","
          << std::round(p.y * 10.0f) / 10.0f;
      return oss.str();
    };

  for (auto & sides_for_edge : all_sides) {
    for (auto & side : sides_for_edge) {
      if (!side.used) {
        std::string key1 = point_to_key(side.corner1);
        std::string key2 = point_to_key(side.corner2);

        corner_graph[key1].emplace_back(side.corner_type1, &side);
        corner_graph[key2].emplace_back(side.corner_type2, &side);
      }
    }
  }

  // Find gates by exploring connected components
  for (auto & sides_for_edge : all_sides) {
    for (auto & start_side : sides_for_edge) {
      if (start_side.used) {
        continue;
      }

      // Try to build a gate starting from this side
      Gate gate;
      gate.corners.resize(GATENET_NUM_CORNERS, cv::Point3f(-1, -1, 0));
      gate.num_corners = 0;
      gate.avg_score = 0.0f;

      std::vector<Side *> sides_in_gate;
      sides_in_gate.push_back(&start_side);

      // Add starting side corners
      gate.corners[start_side.corner_type1] = cv::Point3f(start_side.corner1.x,
          start_side.corner1.y, 1.0f);
      gate.corners[start_side.corner_type2] = cv::Point3f(start_side.corner2.x,
          start_side.corner2.y, 1.0f);
      gate.num_corners = 2;
      gate.avg_score = start_side.score;

      // Try to extend the gate
      bool changed = true;
      while (changed) {
        changed = false;

        for (auto & sides_for_edge_inner : all_sides) {
          for (auto & side : sides_for_edge_inner) {
            if (side.used ||
              std::find(sides_in_gate.begin(), sides_in_gate.end(),
              &side) != sides_in_gate.end())
            {
              continue;
            }

            // Check if this side connects to our gate
            bool connects = false;
            cv::Point2f side_c1 = side.corner1;
            cv::Point2f side_c2 = side.corner2;
            int type1 = side.corner_type1;
            int type2 = side.corner_type2;

            for (int i = 0; i < GATENET_NUM_CORNERS; ++i) {
                if (gate.corners[i].z > 0) {   // Corner exists in gate
                    cv::Point2f existing(gate.corners[i].x, gate.corners[i].y);

                    // Check if side's corner1 matches existing corner position and type
                    if (type1 == i && cv::norm(side_c1 - existing) < 3.0f) {
                        if (gate.corners[type2].z <= 0) {
                            gate.corners[type2] = cv::Point3f(side_c2.x, side_c2.y, 1.0f);
                            gate.num_corners++;
                            connects = true;
                        }
                    }

                    // Check reverse: side's corner2 matches existing
                    if (type2 == i && cv::norm(side_c2 - existing) < 3.0f) {
                        if (gate.corners[type1].z <= 0) {
                            gate.corners[type1] = cv::Point3f(side_c1.x, side_c1.y, 1.0f);
                            gate.num_corners++;
                            connects = true;
                        }
                    }
                }
            }

            if (connects) {
              sides_in_gate.push_back(&side);
              gate.avg_score += side.score;
              changed = true;
              break;
            }
          }
          if (changed) {
            break;
          }
        }
      }

      // Check if we have enough corners for a valid gate
      if (gate.num_corners >= min_corners) {
        // Mark sides as used
        for (auto * side_ptr : sides_in_gate) {
          side_ptr->used = true;
        }

        gate.avg_score /= sides_in_gate.size();
        gates.push_back(gate);
      }
    }
  }

  return gates;
}

InferenceResult gateNetPostProcess(
  const std::vector<cv::Mat> & output,
  const cv::Size & original_size,
  const cv::Size & input_size,
  const cv::Size2f & scale_factors,
  const cv::Point2f & padding,
  float paf_threshold,
  float corner_threshold,
  int min_corners)
{
  InferenceResult result;
  result.success = false;
  result.input_size = input_size;
  result.original_size = original_size;

  // Validate output shape
  if (output.size() != GATENET_OUTPUT_CHANNELS) {
    std::cerr << "Invalid GateNet output: expected " << GATENET_OUTPUT_CHANNELS
              << " channels, got " << output.size() << std::endl;
    return result;
  }

  // Extract heatmaps and PAF maps
  std::vector<cv::Mat> heatmaps(GATENET_NUM_CORNERS);
  std::vector<cv::Mat> vx_maps(GATENET_NUM_EDGES);
  std::vector<cv::Mat> vy_maps(GATENET_NUM_EDGES);

  for (int i = 0; i < GATENET_NUM_CORNERS; ++i) {
    heatmaps[i] = output[i];
  }

  for (int i = 0; i < GATENET_NUM_EDGES; ++i) {
    vx_maps[i] = output[GATENET_NUM_CORNERS + i];
    vy_maps[i] = output[GATENET_NUM_CORNERS + GATENET_NUM_EDGES + i];
  }

  // Step 1: Detect corners from heatmaps
  std::vector<std::vector<cv::Point3f>> detected_corners(GATENET_NUM_CORNERS);
  for (int i = 0; i < GATENET_NUM_CORNERS; ++i) {
    detected_corners[i] = detectPeaks(heatmaps[i], corner_threshold, 8);
  }

  // Step 2: Detect sides using PAF
  auto detected_sides = detectSides(detected_corners, vx_maps, vy_maps, paf_threshold);

  // Step 3: Assemble gates
  auto gates = assembleGates(detected_sides, min_corners);

  // Step 4: Convert gates to Detection format
  for (const auto & gate : gates) {
    Detection det;

    det.class_id = 0;   // "gate"
    det.confidence = gate.avg_score;

    // Convert corners to original image coordinates
    det.keypoints.clear();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = 0.0f;
    float max_y = 0.0f;

    for (int i = 0; i < GATENET_NUM_CORNERS; ++i) {
      cv::Point3f corner = gate.corners[i];

      if (corner.z > 0) {   // Valid corner
        int hm_width = heatmaps[0].cols;
        int hm_height = heatmaps[0].rows;

        float hm_to_input_x = static_cast<float>(input_size.width) / hm_width;
        float hm_to_input_y = static_cast<float>(input_size.height) / hm_height;

        // Scale from heatmap to inference coords
        float x_input = corner.x * hm_to_input_x;
        float y_input = corner.y * hm_to_input_y;

        // Then scale from inference to original
        float x_orig = x_input * (static_cast<float>(original_size.width) / input_size.width);
        float y_orig = y_input * (static_cast<float>(original_size.height) / input_size.height);

        det.keypoints.emplace_back(x_orig, y_orig, corner.z);

        min_x = std::min(min_x, x_orig);
        min_y = std::min(min_y, y_orig);
        max_x = std::max(max_x, x_orig);
        max_y = std::max(max_y, y_orig);
      } else {
        det.keypoints.emplace_back(-1.0f, -1.0f, 0.0f);
      }
    }

    if (det.keypoints.size() > 0 && max_x > min_x && max_y > min_y) {
      det.bbox = cv::Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
    } else {
      det.bbox = cv::Rect2f(0, 0, 0, 0);
    }

    result.detections.push_back(det);
  }

  result.success = true;
  return result;
}

}  // namespace yolo_inference
