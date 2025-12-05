//
//  Subsampling.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/transformer/subsampling.py
//  Subsampling modules for Conformer encoder
//

import Foundation
import MLX
import MLXNN

// MARK: - BaseSubsampling

/// Base class for subsampling modules
class BaseSubsampling: Module {
  var rightContext: Int = 0
  var subsamplingRate: Int = 1
  var posEnc: Module?

  func positionEncoding(offset: Int, size: Int) -> MLXArray {
    if let enc = posEnc as? PositionalEncoding {
      return enc.positionEncoding(offset: offset, size: size)
    } else if let enc = posEnc as? RelPositionalEncoding {
      return enc.positionEncoding(offset: offset, size: size)
    }
    return MLXArray.zeros([1, size, 1])
  }
}

// MARK: - LinearNoSubsampling

/// Linear transform the input without subsampling
/// Used in UpsampleConformerEncoder
class LinearNoSubsampling: BaseSubsampling {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm
  @ModuleInfo(key: "pos_enc") var posEncModule: Module?
  let dropoutRate: Float

  init(
    idim: Int,
    odim: Int,
    dropoutRate: Float,
    posEncClass: Module,
  ) {
    _linear.wrappedValue = Linear(idim, odim)
    _norm.wrappedValue = LayerNorm(dimensions: odim, eps: 1e-5)
    _posEncModule.wrappedValue = posEncClass
    self.dropoutRate = dropoutRate
    super.init()
    posEnc = posEncClass
    rightContext = 0
    subsamplingRate = 1
  }

  /// Apply linear transformation without subsampling
  func callAsFunction(
    _ x: MLXArray,
    xMask: MLXArray,
    offset: Int = 0,
  ) -> (MLXArray, MLXArray, MLXArray) {
    var xOut = linear(x)
    xOut = norm(xOut)
    // Note: Dropout would be applied here during training

    var posEmb: MLXArray
    if let enc = posEnc as? PositionalEncoding {
      (xOut, posEmb) = enc(xOut, offset: offset)
    } else if let enc = posEnc as? RelPositionalEncoding {
      (xOut, posEmb) = enc(xOut, offset: offset)
    } else if let enc = posEnc as? EspnetRelPositionalEncoding {
      (xOut, posEmb) = enc(xOut, offset: offset)
    } else {
      posEmb = MLXArray.zeros([1, xOut.shape[1], 1])
    }

    return (xOut, posEmb, xMask)
  }
}
