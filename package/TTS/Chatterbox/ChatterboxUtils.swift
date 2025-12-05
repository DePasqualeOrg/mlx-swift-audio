//
//  ChatterboxUtils.swift
//  MLXAudio
//
//  Shared utility functions for Chatterbox TTS model
//

import Foundation
import MLX

/// Reverse array along a specific axis
///
/// - Parameters:
///   - x: Input array
///   - axis: Axis to reverse along (negative values count from end)
/// - Returns: Array with elements reversed along the specified axis
func reverseAlongAxis(_ x: MLXArray, axis: Int) -> MLXArray {
  let actualAxis = axis < 0 ? x.ndim + axis : axis
  let size = x.shape[actualAxis]
  // Create reversed indices using MLX operations (avoid Swift array allocation)
  // indices = [size-1, size-2, ..., 1, 0]
  let indices = MLXArray(Int32(size - 1)) - MLXArray(0 ..< size).asType(.int32)
  return MLX.take(x, indices, axis: actualAxis)
}
