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


#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "yolo_inference_cpp/inference_backend.hpp"

namespace yolo_inference
{

// GateNet constants (8-corner gates)
constexpr int GATENET_NUM_CORNERS = 8;
constexpr int GATENET_NUM_EDGES = 8;
constexpr int GATENET_OUTPUT_CHANNELS = GATENET_NUM_CORNERS + GATENET_NUM_EDGES * 2;  // 24

// Edge connections for 8-corner gate topology
// Forms path: 0 -> 1 -> 2 -> 3 -> 7 -> 6 -> 5 -> 4 -> 0
constexpr std::pair<int, int> GATENET_EDGE_CONNECTIONS[GATENET_NUM_EDGES] = {
  {0, 1},  // Edge 0: outer top-left to outer top-right
  {1, 2},  // Edge 1: outer top-right to outer bottom-right
  {2, 3},  // Edge 2: outer bottom-right to outer bottom-left
  {3, 7},  // Edge 3: outer bottom-left to inner bottom-left
  {7, 6},  // Edge 4: inner bottom-left to inner bottom-right
  {6, 5},  // Edge 5: inner bottom-right to inner top-right
  {5, 4},  // Edge 6: inner top-right to inner top-left
  {4, 0}   // Edge 7: inner top-left to outer top-left
};

// Structure to represent a detected side (edge between two corners)
struct Side
{
  cv::Point2f corner1;
  cv::Point2f corner2;
  int corner_type1;
  int corner_type2;
  int edge_idx;
  float score;
  bool used;
};

// Structure to represent a gate
struct Gate
{
  std::vector<cv::Point3f> corners;  // x, y, confidence for each corner (8 corners)
  float avg_score;
  int num_corners;
};

/**
 * @brief Detect peaks (local maxima) in a heatmap
 *
 * @param heatmap Single-channel heatmap
 * @param threshold Minimum confidence threshold
 * @param min_distance Minimum distance between peaks
 * @return Vector of detected peak positions with confidences (x, y, conf)
 */
std::vector<cv::Point3f> detectPeaks(
  const cv::Mat & heatmap,
  float threshold = 0.5f,
  int min_distance = 8);

/**
 * @brief Calculate Part Affinity Field (PAF) affinity between two corners
 *
 * @param corner1 First corner position
 * @param corner2 Second corner position
 * @param vx_map PAF X-component map
 * @param vy_map PAF Y-component map
 * @return Affinity score (mean cosine similarity along the line)
 */
float calculatePAFAffinity(
  const cv::Point2f & corner1,
  const cv::Point2f & corner2,
  const cv::Mat & vx_map,
  const cv::Mat & vy_map);

/**
 * @brief Detect sides (connected corner pairs) using PAF
 *
 * @param corners Detected corners for each corner type (8 types)
 * @param vx_maps PAF X-component maps for each edge (8 edges)
 * @param vy_maps PAF Y-component maps for each edge (8 edges)
 * @param paf_threshold Minimum PAF affinity threshold
 * @return Vector of detected sides for each edge type
 */
std::vector<std::vector<Side>> detectSides(
  const std::vector<std::vector<cv::Point3f>> & corners,
  const std::vector<cv::Mat> & vx_maps,
  const std::vector<cv::Mat> & vy_maps,
  float paf_threshold = 0.3f);

/**
 * @brief Assemble gates from detected sides using graph-based approach
 *
 * @param all_sides Detected sides from all edge types
 * @param min_corners Minimum number of corners required to form a valid gate
 * @return Vector of assembled gates
 */
std::vector<Gate> assembleGates(
  std::vector<std::vector<Side>> & all_sides,
  int min_corners = 3);

/**
 * @brief Main GateNet post-processing function
 *
 * @param output Model output tensor (C x H x W) with 24 channels
 * @param original_size Original image size before preprocessing
 * @param input_size Model input size
 * @param scale_factors Scaling factors from preprocessing
 * @param padding Padding from preprocessing
 * @param paf_threshold PAF affinity threshold
 * @param corner_threshold Corner detection threshold
 * @param min_corners Minimum corners to form a gate
 * @return InferenceResult with detected gates
 */
InferenceResult gateNetPostProcess(
  const std::vector<cv::Mat> & output,
  const cv::Size & original_size,
  const cv::Size & input_size,
  const cv::Size2f & scale_factors,
  const cv::Point2f & padding,
  float paf_threshold = 0.3f,
  float corner_threshold = 0.5f,
  int min_corners = 3);

}  // namespace yolo_inference
