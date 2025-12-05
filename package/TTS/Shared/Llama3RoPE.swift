//
//  Llama3RoPE.swift
//  MLXAudio
//
//  Unified Llama3-style Rotary Position Embedding implementation.
//  Uses MLXFast.RoPE for optimized computation.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon

/// Llama3-style RoPE with frequency scaling for extended context.
///
/// This implementation pre-computes scaled frequencies and uses the optimized
/// MLXFast.RoPE kernel for the actual rotation computation.
///
/// The Llama3 scaling applies smooth interpolation between frequency bands:
/// - High frequency band (short wavelengths): kept unchanged
/// - Low frequency band (long wavelengths): scaled by factor
/// - Medium frequency band: smoothly interpolated
///
/// Note: This is intentionally NOT a Module subclass because RoPE has no
/// learnable parameters. It's a pure computation helper that pre-computes
/// frequency scaling factors and applies rotary embeddings.
final class Llama3RoPE {
  private let dims: Int
  private let traditional: Bool
  private let scale: Float
  private let freqs: MLXArray

  /// Initialize with explicit parameters
  init(
    dims: Int,
    traditional: Bool = false,
    base: Float = 500_000.0,
    scaleFactor: Float = 8.0,
    lowFreqFactor: Float = 1.0,
    highFreqFactor: Float = 4.0,
    oldContextLen: Float = 8192.0,
  ) {
    precondition(dims % 2 == 0, "RoPE dims must be even")
    self.dims = dims
    self.traditional = traditional
    scale = 1.0

    // Compute Llama3-style scaled frequencies
    let lowFreqWavelen = oldContextLen / lowFreqFactor
    let highFreqWavelen = oldContextLen / highFreqFactor

    // Base frequencies: base^(indices/dims)
    let indices = MLXArray(stride(from: 0, to: dims, by: 2))
    var frequencies = MLX.pow(MLXArray(base), indices / Float(dims))
    let wavelens = 2 * Float.pi * frequencies

    // High frequency band: wavelens > lowFreqWavelen -> scale by factor
    frequencies = MLX.where(
      wavelens .> MLXArray(lowFreqWavelen),
      frequencies * scaleFactor,
      frequencies,
    )

    // Medium frequency band: interpolate between scaled and unscaled
    let isMediumFreq = MLX.logicalAnd(
      wavelens .> MLXArray(highFreqWavelen),
      wavelens .< MLXArray(lowFreqWavelen),
    )
    let smoothFactors = (oldContextLen / wavelens - lowFreqFactor) / (highFreqFactor - lowFreqFactor)
    let smoothFreqs = frequencies / ((1 - smoothFactors) / scaleFactor + smoothFactors)

    freqs = MLX.where(isMediumFreq, smoothFreqs, frequencies)
  }

  /// Initialize from a rope_scaling dictionary (common config format)
  convenience init(
    dims: Int,
    traditional: Bool = false,
    base: Float = 500_000.0,
    ropeScaling: [String: StringOrNumber]?,
  ) {
    let scaleFactor: Float
    let lowFreqFactor: Float
    let highFreqFactor: Float
    let oldContextLen: Float

    if let scaling = ropeScaling {
      scaleFactor = scaling.floatValue(for: "factor") ?? 8.0
      lowFreqFactor = scaling.floatValue(for: "low_freq_factor") ?? 1.0
      highFreqFactor = scaling.floatValue(for: "high_freq_factor") ?? 4.0
      oldContextLen = scaling.floatValue(for: "original_max_position_embeddings") ?? 8192.0
    } else {
      scaleFactor = 8.0
      lowFreqFactor = 1.0
      highFreqFactor = 4.0
      oldContextLen = 8192.0
    }

    self.init(
      dims: dims,
      traditional: traditional,
      base: base,
      scaleFactor: scaleFactor,
      lowFreqFactor: lowFreqFactor,
      highFreqFactor: highFreqFactor,
      oldContextLen: oldContextLen,
    )
  }

  /// Apply RoPE using the optimized MLXFast kernel
  func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
    MLXFast.RoPE(
      x,
      dimensions: dims,
      traditional: traditional,
      base: nil, // Using pre-computed freqs instead
      scale: scale,
      offset: offset,
      freqs: freqs,
    )
  }
}

// MARK: - Helper for parsing config dictionaries

extension [String: StringOrNumber] {
  func floatValue(for key: String) -> Float? {
    guard let value = self[key] else { return nil }
    switch value {
      case let .string(s): return Float(s)
      case let .float(f): return f
      case let .int(i): return Float(i)
      case .bool, .ints, .floats: return nil
    }
  }
}
