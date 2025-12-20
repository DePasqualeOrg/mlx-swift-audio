// Copyright © 2023 Shivam Mehta (original model implementation)
// Copyright © Anthony DePasquale (MLX port)
// Ported to MLX from https://github.com/shivammehta25/Matcha-TTS
// License: licenses/matcha.txt

import Foundation
import MLX
import MLXNN

// MARK: - Sinusoidal Position Embeddings

/// Sinusoidal position embeddings for timestep encoding
class SinusoidalPosEmb: Module {
  let dim: Int

  init(dim: Int) {
    precondition(dim % 2 == 0, "SinusoidalPosEmb requires dim to be even")
    self.dim = dim
  }

  func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
    var input = x
    if input.ndim < 1 {
      input = input.expandedDimensions(axis: 0)
    }

    let halfDim = dim / 2
    let embScale = log(Float(10000)) / Float(halfDim - 1)
    let emb = MLX.exp(MLXArray(0 ..< halfDim).asType(.float32) * -embScale)

    // scale * x[:, None] * emb[None, :]
    let scaledX = scale * input.expandedDimensions(axis: 1)
    let embExpanded = emb.expandedDimensions(axis: 0)
    let result = scaledX * embExpanded

    return MLX.concatenated([MLX.sin(result), MLX.cos(result)], axis: -1)
  }
}

// MARK: - Timestep Embedding

/// MLP for timestep embedding
class TimestepEmbedding: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear
  let actFn: String

  init(inChannels: Int, timeEmbedDim: Int, actFn: String = "silu") {
    _linear1.wrappedValue = Linear(inChannels, timeEmbedDim)
    _linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim)
    self.actFn = actFn
  }

  func callAsFunction(_ sample: MLXArray) -> MLXArray {
    var h = linear1(sample)
    h = actFn == "silu" ? silu(h) : gelu(h)
    h = linear2(h)
    return h
  }
}

// MARK: - Block1D

/// 1D convolutional block with group norm
class Block1D: Module {
  @ModuleInfo(key: "conv") var conv: Conv1d
  @ModuleInfo(key: "norm") var norm: GroupNorm

  init(dim: Int, dimOut: Int, groups: Int = 8) {
    _conv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 3, padding: 1)
    _norm.wrappedValue = GroupNorm(groupCount: groups, dimensions: dimOut)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
    // x is (B, C, T) but MLX Conv1d expects (B, T, C)
    var output = (x * mask).swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    output = conv(output)
    output = output.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)
    output = norm(output)
    output = mish(output)
    return output * mask
  }
}

// MARK: - ResnetBlock1D

/// 1D ResNet block with time embedding
class ResnetBlock1D: Module {
  @ModuleInfo(key: "mlp_linear") var mlpLinear: Linear
  @ModuleInfo(key: "block1") var block1: Block1D
  @ModuleInfo(key: "block2") var block2: Block1D
  @ModuleInfo(key: "res_conv") var resConv: Conv1d

  init(dim: Int, dimOut: Int, timeEmbDim: Int, groups: Int = 8) {
    _mlpLinear.wrappedValue = Linear(timeEmbDim, dimOut)
    _block1.wrappedValue = Block1D(dim: dim, dimOut: dimOut, groups: groups)
    _block2.wrappedValue = Block1D(dim: dimOut, dimOut: dimOut, groups: groups)
    _resConv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dimOut, kernelSize: 1)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXArray, timeEmb: MLXArray) -> MLXArray {
    var h = block1(x, mask: mask)
    // Original: h += self.mlp(time_emb) where mlp = Sequential(Mish(), Linear())
    h = h + mlpLinear(mish(timeEmb)).expandedDimensions(axis: -1)
    h = block2(h, mask: mask)

    // res_conv
    var xRes = (x * mask).swappedAxes(1, 2) // (B, C, T) -> (B, T, C)
    xRes = resConv(xRes)
    xRes = xRes.swappedAxes(1, 2) // (B, T, C) -> (B, C, T)

    return h + xRes
  }
}

// MARK: - Downsample1D

/// 1D downsampling with stride-2 convolution
class Downsample1D: Module {
  @ModuleInfo(key: "conv") var conv: Conv1d

  init(dim: Int) {
    _conv.wrappedValue = Conv1d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2, padding: 1)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // x is (B, C, T) but MLX Conv1d expects (B, T, C)
    var output = x.swappedAxes(1, 2)
    output = conv(output)
    output = output.swappedAxes(1, 2)
    return output
  }
}

// MARK: - Upsample1D

/// 1D upsampling with transposed convolution
class MatchaUpsample1D: Module {
  let channels: Int
  let useConvTranspose: Bool

  @ModuleInfo(key: "conv") var conv: ConvTransposed1d

  init(channels: Int, useConvTranspose: Bool = true) {
    self.channels = channels
    self.useConvTranspose = useConvTranspose

    if useConvTranspose {
      _conv.wrappedValue = ConvTransposed1d(
        inputChannels: channels,
        outputChannels: channels,
        kernelSize: 4,
        stride: 2,
        padding: 1,
      )
    } else {
      // Placeholder - will use nearest neighbor upsampling
      _conv.wrappedValue = ConvTransposed1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    if useConvTranspose {
      // x is (B, C, T) but MLX ConvTranspose1d expects (B, T, C)
      var output = x.swappedAxes(1, 2)
      output = conv(output)
      output = output.swappedAxes(1, 2)
      return output
    } else {
      // Nearest neighbor upsampling
      return MLX.repeated(x, count: 2, axis: 2)
    }
  }
}

// MARK: - Mish Activation

/// Mish activation function: x * tanh(softplus(x))
func mish(_ x: MLXArray) -> MLXArray {
  x * tanh(softplus(x))
}

/// Softplus activation function: log(1 + exp(x))
func softplus(_ x: MLXArray) -> MLXArray {
  MLX.log(1 + MLX.exp(x))
}
