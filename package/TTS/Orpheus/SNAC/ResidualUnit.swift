//
//  ResidualUnit.swift
//  MLXAudio
//
//  ResidualUnit for SNAC decoder - residual block with snake activations
//
import Foundation
import MLX
import MLXNN

/// Snake activation wrapper for SNAC decoder
/// Contains the learnable alpha parameter
class SNACSnake: Module {
  @ModuleInfo var alpha: MLXArray

  init(channels: Int) {
    _alpha.wrappedValue = MLX.ones([1, channels, 1])
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    SNACDecoder.snake(x, alpha: alpha)
  }
}

/// ResidualUnit for SNAC decoder - residual block with snake activations
/// Clean structure with semantic property names
class SNACResidualUnit: Module {
  @ModuleInfo var snake1: SNACSnake
  @ModuleInfo var conv1: WNConv1d
  @ModuleInfo var snake2: SNACSnake
  @ModuleInfo var conv2: WNConv1d

  init(dim: Int, dilation: Int, kernelSize: Int, groups: Int) {
    let pad1 = ((kernelSize - 1) * dilation) / 2

    _snake1.wrappedValue = SNACSnake(channels: dim)
    _conv1.wrappedValue = WNConv1d(
      inChannels: dim,
      outChannels: dim,
      kernelSize: kernelSize,
      padding: pad1,
      dilation: dilation,
      groups: groups,
      bias: true,
    )
    _snake2.wrappedValue = SNACSnake(channels: dim)
    _conv2.wrappedValue = WNConv1d(
      inChannels: dim,
      outChannels: dim,
      kernelSize: 1,
      padding: 0,
      dilation: 1,
      groups: 1,
      bias: true,
    )
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var residual = x
    var y = x

    // Ensure input is [B, C, T] format
    if y.shape[1] != snake1.alpha.shape[1], y.shape[2] == snake1.alpha.shape[1] {
      y = y.transposed(axes: [0, 2, 1])
    }
    if residual.shape[1] != snake1.alpha.shape[1], residual.shape[2] == snake1.alpha.shape[1] {
      residual = residual.transposed(axes: [0, 2, 1])
    }

    // Apply the sequence: Snake -> Conv1 -> Snake -> Conv2
    // snake outputs [B, T, C], need to transpose back to [B, C, T] for conv
    y = snake1(y)
    y = y.transposed(axes: [0, 2, 1])
    y = conv1(y)

    y = snake2(y)
    y = y.transposed(axes: [0, 2, 1])
    y = conv2(y)

    // Crop residual if needed to match y's time dim
    let tRes = residual.shape[2]
    let tY = y.shape[2]

    if tRes != tY {
      let diff = tRes - tY
      if diff > 0 {
        let pad = diff / 2
        let end = tRes - (diff - pad)
        let b = residual.shape[0]
        let c = residual.shape[1]
        residual = residual[0 ..< b, 0 ..< c, pad ..< end]
      }
    }
    return residual + y
  }
}
