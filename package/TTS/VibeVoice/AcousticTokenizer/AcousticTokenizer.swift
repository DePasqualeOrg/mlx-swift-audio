// Copyright © Microsoft (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/microsoft/VibeVoice
// License: licenses/vibevoice.txt

import Foundation
import MLX
import MLXNN

// MARK: - ConvRMSNorm

/// RMSNorm for convolutional features (B, C, T) format
class ConvRMSNorm: Module {
  let dim: Int
  let eps: Float
  let elementwiseAffine: Bool

  @ParameterInfo(key: "weight") var weight: MLXArray?

  init(dim: Int, eps: Float = 1e-5, elementwiseAffine: Bool = true) {
    self.dim = dim
    self.eps = eps
    self.elementwiseAffine = elementwiseAffine

    if elementwiseAffine {
      _weight.wrappedValue = MLXArray.ones([dim])
    } else {
      _weight.wrappedValue = nil
    }
  }

  private func norm(_ x: MLXArray) -> MLXArray {
    x * MLX.rsqrt(MLX.mean(x * x, axis: -1, keepDims: true) + eps)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T) -> transpose to (B, T, C) for normalization
    var out = x.transposed(0, 2, 1)
    out = norm(out.asType(.float32)).asType(x.dtype)
    if let w = weight {
      out = out * w
    }
    // Transpose back to (B, C, T)
    return out.transposed(0, 2, 1)
  }
}

// MARK: - CausalConv1d

/// Causal 1D convolution with padding on the left
/// Input/output format: (B, C, T) - batch, channels, time (PyTorch convention)
/// MLX Conv1d expects: (B, T, C) - batch, time, channels
class CausalConv1d: Module {
  let inChannels: Int
  let outChannels: Int
  let kernelSize: Int
  let stride: Int
  let dilation: Int
  let groups: Int
  let padding: Int

  @ModuleInfo(key: "conv") var conv: Conv1d

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    dilation: Int = 1,
    groups: Int = 1,
    bias: Bool = true
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize
    self.stride = stride
    self.dilation = dilation
    self.groups = groups

    // Calculate padding for causal convolution
    padding = (kernelSize - 1) * dilation

    _conv.wrappedValue = Conv1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: 0, // We handle padding manually
      dilation: dilation,
      groups: groups,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T) - input in PyTorch format
    // Transpose to MLX format: (B, C, T) -> (B, T, C)
    var out = x.transposed(0, 2, 1)

    // Add causal padding on the time dimension (now axis 1)
    if padding > 0 {
      out = MLX.padded(out, widths: [IntOrPair(0), IntOrPair((padding, 0)), IntOrPair(0)])
    }

    // Apply conv - MLX expects (B, T, C)
    out = conv(out)

    // Transpose back to PyTorch format: (B, T, C) -> (B, C, T)
    return out.transposed(0, 2, 1)
  }
}

// MARK: - CausalConvTranspose1d

/// Causal transposed 1D convolution for upsampling
/// Input/output format: (B, C, T) - batch, channels, time (PyTorch convention)
class CausalConvTranspose1d: Module {
  let inChannels: Int
  let outChannels: Int
  let kernelSize: Int
  let stride: Int
  let trimRightRatio: Float
  let paddingTotal: Int

  @ModuleInfo(key: "convtr") var convtr: ConvTranspose1d

  init(
    inChannels: Int,
    outChannels: Int,
    kernelSize: Int,
    stride: Int = 1,
    bias: Bool = true,
    trimRightRatio: Float = 1.0
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.kernelSize = kernelSize
    self.stride = stride
    self.trimRightRatio = trimRightRatio

    // Calculate padding
    paddingTotal = kernelSize - stride

    _convtr.wrappedValue = ConvTranspose1d(
      inputChannels: inChannels,
      outputChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      padding: 0,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T) - input in PyTorch format
    // Transpose to MLX format: (B, C, T) -> (B, T, C)
    var out = x.transposed(0, 2, 1)

    // Apply transposed conv
    out = convtr(out)

    // Transpose back to PyTorch format: (B, T, C) -> (B, C, T)
    out = out.transposed(0, 2, 1)

    // Trim padding for causal (on time dimension, now axis 2)
    let paddingRight = Int(ceil(Float(paddingTotal) * trimRightRatio))
    let paddingLeft = paddingTotal - paddingRight

    if paddingLeft > 0 {
      out = out[0..., 0..., paddingLeft...]
    }
    if paddingRight > 0 {
      out = out[0..., 0..., 0 ..< (out.shape[2] - paddingRight)]
    }

    return out
  }
}

// MARK: - DepthwiseConv

/// Depthwise separable convolution wrapped in a conv module
class DepthwiseConv: Module {
  let dim: Int
  let kernelSize: Int
  let causal: Bool

  @ModuleInfo(key: "conv") var conv: CausalConv1d

  init(dim: Int, kernelSize: Int = 7, causal: Bool = true, bias: Bool = true) {
    self.dim = dim
    self.kernelSize = kernelSize
    self.causal = causal

    _conv.wrappedValue = CausalConv1d(
      inChannels: dim,
      outChannels: dim,
      kernelSize: kernelSize,
      groups: dim,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv(x)
  }
}

// MARK: - Mixer

/// Mixer module wrapping depthwise conv
class Mixer: Module {
  @ModuleInfo(key: "conv") var conv: DepthwiseConv

  init(dim: Int, kernelSize: Int = 7, causal: Bool = true, bias: Bool = true) {
    _conv.wrappedValue = DepthwiseConv(dim: dim, kernelSize: kernelSize, causal: causal, bias: bias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv(x)
  }
}

// MARK: - FeedForward

/// Feed-forward network with GELU activation
/// Note: Uses linear1/linear2 naming to match HuggingFace weights
class AcousticFeedForward: Module {
  @ModuleInfo(key: "linear1") var linear1: Linear
  @ModuleInfo(key: "linear2") var linear2: Linear

  init(dim: Int, mult: Float = 4.0, bias: Bool = true) {
    let hiddenDim = Int(Float(dim) * mult)
    _linear1.wrappedValue = Linear(dim, hiddenDim, bias: bias)
    _linear2.wrappedValue = Linear(hiddenDim, dim, bias: bias)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var out = linear1(x)
    out = gelu(out)
    out = linear2(out)
    return out
  }
}

// MARK: - Block1D

/// 1D convolutional block with depthwise conv and FFN
class Block1D: Module {
  let dim: Int

  @ModuleInfo(key: "norm") var norm: ConvRMSNorm
  @ModuleInfo(key: "ffn_norm") var ffnNorm: ConvRMSNorm
  @ModuleInfo(key: "mixer") var mixer: Mixer
  @ModuleInfo(key: "ffn") var ffn: AcousticFeedForward

  @ParameterInfo(key: "gamma") var gamma: MLXArray?
  @ParameterInfo(key: "ffn_gamma") var ffnGamma: MLXArray?

  init(
    dim: Int,
    eps: Float = 1e-6,
    causal: Bool = true,
    bias: Bool = true,
    layerScaleInitValue: Float = 1e-6
  ) {
    self.dim = dim

    // Normalization
    _norm.wrappedValue = ConvRMSNorm(dim: dim, eps: eps)
    _ffnNorm.wrappedValue = ConvRMSNorm(dim: dim, eps: eps)

    // Mixer (depthwise conv)
    _mixer.wrappedValue = Mixer(dim: dim, kernelSize: 7, causal: causal, bias: bias)

    // FFN
    _ffn.wrappedValue = AcousticFeedForward(dim: dim, mult: 4.0, bias: bias)

    // Layer scale
    if layerScaleInitValue > 0 {
      _gamma.wrappedValue = MLXArray.ones([dim]) * layerScaleInitValue
      _ffnGamma.wrappedValue = MLXArray.ones([dim]) * layerScaleInitValue
    } else {
      _gamma.wrappedValue = nil
      _ffnGamma.wrappedValue = nil
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x: (B, C, T)

    // Mixer path
    var out = x
    var residual = out
    out = norm(out)
    out = mixer(out)
    if let g = gamma {
      out = out * g.expandedDimensions(axes: [0, 2])
    }
    out = residual + out

    // FFN path
    residual = out
    out = ffnNorm(out)
    // Transpose for FFN: (B, C, T) -> (B, T, C)
    out = out.transposed(0, 2, 1)
    out = ffn(out)
    // Transpose back: (B, T, C) -> (B, C, T)
    out = out.transposed(0, 2, 1)
    if let g = ffnGamma {
      out = out * g.expandedDimensions(axes: [0, 2])
    }
    out = residual + out

    return out
  }
}

// MARK: - StemConv

/// Stem convolution layer wrapped in Sequential structure to match HF
class StemConv: Module {
  @ModuleInfo(key: "conv") var conv: CausalConv1d

  init(inChannels: Int, outChannels: Int, kernelSize: Int = 7, bias: Bool = true) {
    _conv.wrappedValue = CausalConv1d(
      inChannels: inChannels,
      outChannels: outChannels,
      kernelSize: kernelSize,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv(x)
  }
}

// MARK: - UpsampleLayer

/// Upsample layer with transposed convolution
class UpsampleLayer: Module {
  @ModuleInfo(key: "convtr") var convtr: CausalConvTranspose1d

  init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, bias: Bool = true) {
    _convtr.wrappedValue = CausalConvTranspose1d(
      inChannels: inChannels,
      outChannels: outChannels,
      kernelSize: kernelSize,
      stride: stride,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    convtr(x)
  }
}

// MARK: - HeadConv

/// Output head convolution
class HeadConv: Module {
  @ModuleInfo(key: "conv") var conv: CausalConv1d

  init(inChannels: Int, outChannels: Int, kernelSize: Int = 7, bias: Bool = true) {
    _conv.wrappedValue = CausalConv1d(
      inChannels: inChannels,
      outChannels: outChannels,
      kernelSize: kernelSize,
      bias: bias
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    conv(x)
  }
}

// MARK: - TokenizerDecoder

/// Decoder that converts latent representations back to audio
/// Architecture matches HuggingFace VibeVoice structure:
/// - upsample_layers[0] is stem conv
/// - upsample_layers[1-6] are transposed convolutions
/// - stages[0-6] are transformer blocks
/// - head is output convolution
class TokenizerDecoder: Module {
  let dimension: Int
  let channels: Int
  let nFilters: Int
  let ratios: [Int]
  let depths: [Int]
  let causal: Bool
  let nStages: Int

  @ModuleInfo(key: "upsample_layers") var upsampleLayers: [[Module]]
  @ModuleInfo(key: "stages") var stages: [[Block1D]]
  @ModuleInfo(key: "head") var head: HeadConv

  init(config: AcousticTokenizerConfig) {
    dimension = config.vaeDim
    channels = config.channels
    nFilters = config.decoderNFilters ?? config.encoderNFilters
    causal = config.causal

    // Use decoder ratios or fallback to encoder ratios
    ratios = config.decoderRatios ?? config.encoderRatios

    // Parse depths - should be reversed encoder depths for decoder
    if let decoderDepthsStr = config.decoderDepths {
      depths = decoderDepthsStr.split(separator: "-").compactMap { Int($0) }
    } else {
      let encoderDepths = config.encoderDepths.split(separator: "-").compactMap { Int($0) }
      depths = encoderDepths.reversed()
    }

    nStages = depths.count

    // Build upsample layers
    var upLayers: [[Module]] = []

    // First upsample layer is stem conv
    let stemOutCh = nFilters * Int(pow(2.0, Double(nStages - 1)))
    upLayers.append([StemConv(inChannels: dimension, outChannels: stemOutCh, kernelSize: 7, bias: config.convBias)])

    // Remaining upsample layers are transposed convolutions
    for i in 0 ..< ratios.count {
      let inCh = nFilters * Int(pow(2.0, Double(nStages - 1 - i)))
      let outCh: Int
      if i < ratios.count - 1 {
        outCh = nFilters * Int(pow(2.0, Double(nStages - 2 - i)))
      } else {
        outCh = nFilters
      }

      upLayers.append([UpsampleLayer(
        inChannels: inCh,
        outChannels: outCh,
        kernelSize: ratios[i] * 2,
        stride: ratios[i],
        bias: config.convBias
      )])
    }

    _upsampleLayers.wrappedValue = upLayers

    // Build stages
    var stagesList: [[Block1D]] = []
    for i in 0 ..< nStages {
      let inCh = nFilters * Int(pow(2.0, Double(nStages - 1 - i)))
      var stageBlocks: [Block1D] = []
      for _ in 0 ..< depths[i] {
        stageBlocks.append(Block1D(
          dim: inCh,
          eps: config.layernormEps,
          causal: config.causal,
          bias: config.convBias,
          layerScaleInitValue: config.layerScaleInitValue
        ))
      }
      stagesList.append(stageBlocks)
    }

    _stages.wrappedValue = stagesList

    // Output head
    _head.wrappedValue = HeadConv(
      inChannels: nFilters,
      outChannels: channels,
      kernelSize: 7,
      bias: config.convBias
    )
  }

  /// Decode latent representations to audio
  /// - Parameter x: Latent tensor of shape (B, T, D) or (B, D, T)
  /// - Returns: Audio tensor of shape (B, 1, T')
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // Ensure x is in (B, D, T) format
    var out: MLXArray
    if x.shape[1] != dimension {
      out = x.transposed(0, 2, 1)
    } else {
      out = x
    }

    // Apply stem (first upsample layer)
    if let stemLayer = upsampleLayers[0].first as? StemConv {
      out = stemLayer(out)
    }

    // Process through stages and upsampling
    for i in 0 ..< nStages {
      // Apply stage blocks
      for block in stages[i] {
        out = block(out)
      }

      // Apply upsampling (skip first upsample which was stem)
      if i + 1 < upsampleLayers.count {
        if let upsampleLayer = upsampleLayers[i + 1].first as? UpsampleLayer {
          out = upsampleLayer(out)
        }
      }
    }

    // Output head
    out = head(out)

    return out
  }
}

// MARK: - AcousticTokenizer

/// VibeVoice acoustic tokenizer (decoder only for inference)
class AcousticTokenizer: Module {
  let config: AcousticTokenizerConfig
  let fixStd: Float
  let stdDistType: String

  @ModuleInfo(key: "decoder") var decoder: TokenizerDecoder

  init(config: AcousticTokenizerConfig) {
    self.config = config
    fixStd = config.fixStd
    stdDistType = config.stdDistType

    _decoder.wrappedValue = TokenizerDecoder(config: config)
  }

  /// Convert latent representations to audio
  /// - Parameter latents: Latent tensor of shape (B, T, D) where D = vaeDim
  /// - Returns: Audio tensor of shape (B, 1, T')
  func decode(_ latents: MLXArray) -> MLXArray {
    decoder(latents)
  }

  func callAsFunction(_ latents: MLXArray) -> MLXArray {
    decode(latents)
  }
}
