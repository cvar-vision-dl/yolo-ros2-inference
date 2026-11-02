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
 * @file frame_edge.hpp
 *
 * Boxes that end a hair past the frame's left or top edge
 */

#ifndef YOLO_INFERENCE_CPP__FRAME_EDGE_HPP_
#define YOLO_INFERENCE_CPP__FRAME_EDGE_HPP_

namespace yolo_inference
{

// An object cut by the frame's edge is labelled with its box clipped to the frame, so the model
// puts that side of the box on the frame's edge, give or take a pixel. The backends drop every
// box with x1 < 0 or y1 < 0 (the right and bottom edges have no such check), so whether a gate
// cut by the left or top edge was seen came down to that pixel: of the ~230k boxes in the drones'
// recorded /gate_detections (25 flights up to 2026-10-27) none starts left of or above the frame,
// ~2500 end past its right or bottom. A box at most kEdgeTolerancePx (model input pixels) past
// the left or top edge is now kept, clipped to the frame. Further out it is still dropped: those
// keypoints are poor (S3 gates 1.5-4 m away, synthetic val, 2026-10-31: median keypoint error
// 11 px within 5 px of the edge, 16 px at 5-20 px, 31 px beyond).
constexpr float kEdgeTolerancePx = 5.0f;

// Puts a box's x1 or y1 that lies at most `tolerance` before the frame's edge (0) on the edge.
inline void snapToFrameEdge(float & lo, float tolerance)
{
  if (lo < 0.0f && lo >= -tolerance) {
    lo = 0.0f;
  }
}

}  // namespace yolo_inference

#endif  // YOLO_INFERENCE_CPP__FRAME_EDGE_HPP_
