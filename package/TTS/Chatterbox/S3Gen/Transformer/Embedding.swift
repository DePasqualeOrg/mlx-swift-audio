//
//  Embedding.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/transformer/embedding.py
//  Positional encoding modules for Conformer encoder
//

import Foundation
import MLX
import MLXNN

// MARK: - PositionalEncoding

/// Sinusoidal positional encoding
/// PE(pos, 2i)   = sin(pos/(10000^(2i/d_model)))
/// PE(pos, 2i+1) = cos(pos/(10000^(2i/d_model)))
class PositionalEncoding: Module {
  let dModel: Int
  let xscale: Float
  let dropoutRate: Float
  let maxLen: Int
  /// Positional encoding buffer - underscore prefix excludes from parameter validation
  var _pe: MLXArray

  init(dModel: Int, dropoutRate: Float, maxLen: Int = 5000) {
    self.dModel = dModel
    xscale = sqrt(Float(dModel))
    self.dropoutRate = dropoutRate
    self.maxLen = maxLen
    _pe = Self.createPE(maxLen: maxLen, dModel: dModel)
  }

  static func createPE(maxLen: Int, dModel: Int) -> MLXArray {
    let position = MLXArray(0 ..< maxLen).asType(.float32).expandedDimensions(axis: 1)
    let divTerm = MLX.exp(
      MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32) *
        Float(-log(10000.0) / Float(dModel)),
    )

    let peSin = MLX.sin(position * divTerm) // (maxLen, dModel/2)
    let peCos = MLX.cos(position * divTerm) // (maxLen, dModel/2)

    // Interleave sin and cos using vectorized reshape and transpose
    // Stack sin and cos: (2, maxLen, dModel/2)
    let stacked = MLX.stacked([peSin, peCos], axis: 0)
    // Transpose to (maxLen, 2, dModel/2)
    let transposed = stacked.transposed(1, 0, 2)
    // Reshape to interleave: (maxLen, dModel)
    let peArray = transposed.reshaped([maxLen, dModel])

    return peArray.expandedDimensions(axis: 0) // (1, max_len, d_model)
  }

  func positionEncoding(offset: Int, size: Int) -> MLXArray {
    precondition(offset + size <= maxLen, "Position encoding out of range")
    return _pe[0..., offset ..< offset + size, 0...]
  }

  func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
    let posEmb = positionEncoding(offset: offset, size: x.shape[1])
    var xOut = x * xscale + posEmb

    // Note: Dropout during training would be applied here
    // For inference, we skip dropout
    return (xOut, posEmb)
  }
}

// MARK: - RelPositionalEncoding

/// Relative positional encoding module
/// Unlike PositionalEncoding, this does NOT add pos_emb to input.
/// The pos_emb is returned separately for use in relative attention.
class RelPositionalEncoding: PositionalEncoding {
  override init(dModel: Int, dropoutRate: Float, maxLen: Int = 5000) {
    super.init(dModel: dModel, dropoutRate: dropoutRate, maxLen: maxLen)
  }

  override func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
    let xScaled = x * xscale
    let posEmb = positionEncoding(offset: offset, size: x.shape[1])
    return (xScaled, posEmb)
  }
}

// MARK: - EspnetRelPositionalEncoding

/// Relative positional encoding module (ESPnet implementation)
/// This version computes both positive and negative position encodings
/// for bidirectional relative attention.
class EspnetRelPositionalEncoding: Module {
  let dModel: Int
  let xscale: Float
  let dropoutRate: Float
  var maxLen: Int
  /// Positional encoding buffer - underscore prefix excludes from parameter validation
  var _pe: MLXArray?

  init(dModel: Int, dropoutRate: Float, maxLen: Int = 5000) {
    self.dModel = dModel
    xscale = sqrt(Float(dModel))
    self.dropoutRate = dropoutRate
    self.maxLen = maxLen
    super.init()
    extendPE(size: maxLen)
  }

  private func extendPE(size: Int) {
    if let existingPE = _pe, existingPE.shape[1] >= size * 2 - 1 {
      return
    }

    let position = MLXArray(0 ..< size).asType(.float32).expandedDimensions(axis: 1)
    let divTerm = MLX.exp(
      MLXArray(stride(from: 0, to: dModel, by: 2)).asType(.float32) *
        Float(-log(10000.0) / Float(dModel)),
    )

    // Positive positions - vectorized interleaving
    let pePositiveSin = MLX.sin(position * divTerm)
    let pePositiveCos = MLX.cos(position * divTerm)
    let pePositiveStacked = MLX.stacked([pePositiveSin, pePositiveCos], axis: 0)
    let pePositiveTransposed = pePositiveStacked.transposed(1, 0, 2)
    let pePositive = pePositiveTransposed.reshaped([size, dModel])

    // Negative positions - vectorized interleaving
    let peNegativeSin = MLX.sin(-1 * position * divTerm)
    let peNegativeCos = MLX.cos(-1 * position * divTerm)
    let peNegativeStacked = MLX.stacked([peNegativeSin, peNegativeCos], axis: 0)
    let peNegativeTransposed = peNegativeStacked.transposed(1, 0, 2)
    let peNegative = peNegativeTransposed.reshaped([size, dModel])

    // Reverse positive and concatenate
    let pePositiveFlipped = reverseAlongAxis(pePositive, axis: 0).expandedDimensions(axis: 0)
    // Skip first element of negative (which is 0)
    let peNegativeSliced = peNegative[1...].expandedDimensions(axis: 0)

    // Concatenate: [..., 2, 1, 0, 1, 2, ...]
    _pe = MLX.concatenated([pePositiveFlipped, peNegativeSliced], axis: 1)
    maxLen = size
  }

  func positionEncoding(size: Int, offset _: Int = 0) -> MLXArray {
    guard let pe = _pe else {
      fatalError("PE not initialized")
    }
    let center = pe.shape[1] / 2
    let start = center - size + 1
    let end = center + size
    return pe[0..., start ..< end, 0...]
  }

  func callAsFunction(_ x: MLXArray, offset: Int = 0) -> (MLXArray, MLXArray) {
    extendPE(size: x.shape[1])
    let xScaled = x * xscale
    let posEmb = positionEncoding(size: x.shape[1], offset: offset)
    return (xScaled, posEmb)
  }
}

// MARK: - NoPositionalEncoding

/// No positional encoding - returns zeros
class NoPositionalEncoding: Module {
  let dModel: Int
  let dropoutRate: Float

  init(dModel: Int, dropoutRate: Float) {
    self.dModel = dModel
    self.dropoutRate = dropoutRate
  }

  func positionEncoding(offset _: Int, size: Int) -> MLXArray {
    MLXArray.zeros([1, size, dModel])
  }

  func callAsFunction(_ x: MLXArray, offset _: Int = 0) -> (MLXArray, MLXArray) {
    let posEmb = MLXArray.zeros([1, x.shape[1], dModel])
    return (x, posEmb)
  }
}
