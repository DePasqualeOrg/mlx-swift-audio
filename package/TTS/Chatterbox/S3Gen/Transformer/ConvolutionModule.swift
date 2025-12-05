//
//  ConvolutionModule.swift
//  MLXAudio
//
//  Ported from mlx_audio/tts/models/chatterbox/s3gen/transformer/convolution.py
//  Convolution module for Conformer encoder
//

import Foundation
import MLX
import MLXNN

// MARK: - ConvolutionModule

/// Convolution module for Conformer
class ConvolutionModule: Module {
  let channels: Int
  let kernelSize: Int
  let lorder: Int
  let useLayerNorm: Bool

  @ModuleInfo(key: "pointwise_conv1") var pointwiseConv1: Conv1d
  @ModuleInfo(key: "depthwise_conv") var depthwiseConv: Conv1d
  @ModuleInfo(key: "norm") var norm: Module
  @ModuleInfo(key: "pointwise_conv2") var pointwiseConv2: Conv1d

  let activation: UnaryLayer

  init(
    channels: Int,
    kernelSize: Int = 15,
    activation: UnaryLayer? = nil,
    norm: String = "batch_norm",
    causal: Bool = false,
    bias: Bool = true,
  ) {
    self.channels = channels
    self.kernelSize = kernelSize

    // Pointwise expansion
    _pointwiseConv1.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: 2 * channels,
      kernelSize: 1,
      stride: 1,
      padding: 0,
      bias: bias,
    )

    // Causal vs symmetric padding
    let padding: Int
    if causal {
      padding = 0
      lorder = kernelSize - 1
    } else {
      precondition((kernelSize - 1) % 2 == 0, "kernel_size must be odd for symmetric conv")
      padding = (kernelSize - 1) / 2
      lorder = 0
    }

    // Depthwise convolution (groups=channels for depthwise)
    _depthwiseConv.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: kernelSize,
      stride: 1,
      padding: padding,
      bias: bias,
    )

    // Normalization
    precondition(norm == "batch_norm" || norm == "layer_norm", "norm must be batch_norm or layer_norm")
    if norm == "batch_norm" {
      useLayerNorm = false
      _norm.wrappedValue = BatchNorm(featureCount: channels)
    } else {
      useLayerNorm = true
      _norm.wrappedValue = LayerNorm(dimensions: channels, eps: 1e-5)
    }

    // Pointwise compression
    _pointwiseConv2.wrappedValue = Conv1d(
      inputChannels: channels,
      outputChannels: channels,
      kernelSize: 1,
      stride: 1,
      padding: 0,
      bias: bias,
    )

    self.activation = activation ?? SiLU()
  }

  func callAsFunction(
    _ x: MLXArray,
    maskPad: MLXArray? = nil,
    cache: MLXArray? = nil,
  ) -> (MLXArray, MLXArray) {
    // Transpose to (B, C, T)
    var xConv = x.swappedAxes(1, 2)

    // Apply mask
    if let mask = maskPad, mask.shape[2] > 0 {
      xConv = MLX.where(mask, xConv, MLXArray(0.0))
    }

    // Handle causal convolution caching
    var newCache: MLXArray
    if lorder > 0 {
      if cache == nil || cache!.shape[2] == 0 {
        // Pad on left for causal conv
        xConv = MLX.padded(xConv, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((lorder, 0))])
      } else {
        // Use cache
        xConv = MLX.concatenated([cache!, xConv], axis: 2)
      }
      newCache = xConv[0..., 0..., (-lorder)...]
    } else {
      newCache = MLXArray.zeros([0, 0, 0])
    }

    // GLU mechanism: pointwise expansion + gated linear unit
    xConv = pointwiseConv1(xConv) // (B, 2C, T)
    xConv = glu(xConv, axis: 1) // (B, C, T)

    // Depthwise convolution
    xConv = depthwiseConv(xConv)

    // Normalization
    if useLayerNorm {
      xConv = xConv.swappedAxes(1, 2) // (B, T, C)
      xConv = (norm as! LayerNorm)(xConv)
      xConv = activation(xConv)
      xConv = xConv.swappedAxes(1, 2) // (B, C, T)
    } else {
      xConv = (norm as! BatchNorm)(xConv)
      xConv = activation(xConv)
    }

    // Pointwise compression
    xConv = pointwiseConv2(xConv)

    // Apply mask again
    if let mask = maskPad, mask.shape[2] > 0 {
      xConv = MLX.where(mask, xConv, MLXArray(0.0))
    }

    // Transpose back to (B, T, C)
    return (xConv.swappedAxes(1, 2), newCache)
  }
}

// MARK: - GLU Helper

/// Gated Linear Unit activation
func glu(_ x: MLXArray, axis: Int) -> MLXArray {
  let half = x.shape[axis] / 2
  let parts = MLX.split(x, indices: [half], axis: axis)
  return parts[0] * sigmoid(parts[1])
}
