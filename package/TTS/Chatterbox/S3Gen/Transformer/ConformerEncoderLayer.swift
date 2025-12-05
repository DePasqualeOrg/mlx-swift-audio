//
//  ConformerEncoderLayer.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/transformer/encoder_layer.py
//  Conformer encoder layer module
//

import Foundation
import MLX
import MLXNN

// MARK: - ConformerEncoderLayer

/// Conformer encoder layer module
/// Combines self-attention, convolution, and feed-forward modules
/// with residual connections and layer normalization.
class ConformerEncoderLayer: Module {
  let size: Int
  let dropoutRate: Float
  let normalizeBefore: Bool
  let ffScale: Float

  @ModuleInfo(key: "self_attn") var selfAttn: Module
  @ModuleInfo(key: "feed_forward") var feedForward: PositionwiseFeedForward?
  @ModuleInfo(key: "feed_forward_macaron") var feedForwardMacaron: PositionwiseFeedForward?
  @ModuleInfo(key: "conv_module") var convModule: ConvolutionModule?

  @ModuleInfo(key: "norm_ff") var normFF: LayerNorm
  @ModuleInfo(key: "norm_mha") var normMHA: LayerNorm
  @ModuleInfo(key: "norm_ff_macaron") var normFFMacaron: LayerNorm?
  @ModuleInfo(key: "norm_conv") var normConv: LayerNorm?
  @ModuleInfo(key: "norm_final") var normFinal: LayerNorm?

  init(
    size: Int,
    selfAttn: Module,
    feedForward: PositionwiseFeedForward?,
    feedForwardMacaron: PositionwiseFeedForward?,
    convModule: ConvolutionModule?,
    dropoutRate: Float = 0.1,
    normalizeBefore: Bool = true,
  ) {
    self.size = size
    self.dropoutRate = dropoutRate
    self.normalizeBefore = normalizeBefore

    _selfAttn.wrappedValue = selfAttn
    _feedForward.wrappedValue = feedForward
    _feedForwardMacaron.wrappedValue = feedForwardMacaron
    _convModule.wrappedValue = convModule

    _normFF.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
    _normMHA.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)

    if feedForwardMacaron != nil {
      _normFFMacaron.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
      ffScale = 0.5
    } else {
      ffScale = 1.0
    }

    if convModule != nil {
      _normConv.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
      _normFinal.wrappedValue = LayerNorm(dimensions: size, eps: 1e-12)
    }
  }

  /// Compute encoded features
  func callAsFunction(
    _ x: MLXArray,
    mask: MLXArray,
    posEmb: MLXArray,
    maskPad: MLXArray? = nil,
    attCache: MLXArray? = nil,
    cnnCache: MLXArray? = nil,
  ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
    var xOut = x

    // Macaron-style feed-forward (optional, half-step)
    if let ffMacaron = feedForwardMacaron, let normFFM = normFFMacaron {
      let residual = xOut
      if normalizeBefore {
        xOut = normFFM(xOut)
      }
      let ffOut = ffMacaron(xOut)
      xOut = residual + ffScale * ffOut
      if !normalizeBefore {
        xOut = normFFM(xOut)
      }
    }

    // Multi-headed self-attention
    var residual = xOut
    if normalizeBefore {
      xOut = normMHA(xOut)
    }

    var xAtt: MLXArray
    var newAttCache: MLXArray

    if let relAttn = selfAttn as? RelPositionMultiHeadedAttention {
      (xAtt, newAttCache) = relAttn(
        query: xOut,
        key: xOut,
        value: xOut,
        mask: mask,
        posEmb: posEmb,
        cache: attCache,
      )
    } else if let mha = selfAttn as? MultiHeadedAttention {
      (xAtt, newAttCache) = mha(
        query: xOut,
        key: xOut,
        value: xOut,
        mask: mask,
        posEmb: posEmb,
        cache: attCache,
      )
    } else {
      fatalError("Unsupported attention type")
    }

    xOut = residual + xAtt
    if !normalizeBefore {
      xOut = normMHA(xOut)
    }

    // Convolution module (optional)
    var newCnnCache = MLXArray.zeros([0, 0, 0])
    if let conv = convModule, let normC = normConv {
      residual = xOut
      if normalizeBefore {
        xOut = normC(xOut)
      }

      (xOut, newCnnCache) = conv(xOut, maskPad: maskPad, cache: cnnCache)
      xOut = residual + xOut

      if !normalizeBefore {
        xOut = normC(xOut)
      }
    }

    // Feed-forward module
    if let ff = feedForward {
      residual = xOut
      if normalizeBefore {
        xOut = normFF(xOut)
      }

      let ffOut = ff(xOut)
      xOut = residual + ffScale * ffOut

      if !normalizeBefore {
        xOut = normFF(xOut)
      }
    }

    // Final normalization (if using conv module)
    if let normF = normFinal, convModule != nil {
      xOut = normF(xOut)
    }

    return (xOut, mask, newAttCache, newCnnCache)
  }
}
